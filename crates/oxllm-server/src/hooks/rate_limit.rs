//! `RateLimitHook` — enforces per-key (and per-team) RPM and TPM limits
//! using a **truly atomic reservation** primitive (`Cache::incr_if_below`).
//!
//! Pre-call atomically checks-and-increments the counter; if the increment
//! would exceed the limit the counter is left untouched and the request is
//! rejected. Post-success adjusts the TPM reservation to the actual reported
//! usage (delta between estimate and actual). RPM reservations are released
//! on upstream failure via `post_failure` so a burst of failures cannot
//! temporarily exhaust RPM (they would otherwise linger until the 60s TTL).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::{
    Cache, CallContext, Error, Hook, HookAction, IncrResult, RateLimitInfo, ResponseSummary,
};

use crate::hooks::spend_keys::{org_spend_key, project_spend_key, spend_key, team_spend_key};

const MINUTE: Duration = Duration::from_secs(60);

fn rpm_key(api_key_hash: &str) -> String {
    format!("rpm:{api_key_hash}")
}

fn tpm_key(api_key_hash: &str) -> String {
    format!("tpm:{api_key_hash}")
}

fn team_rpm_key(team_id: &str) -> String {
    format!("rpm:team:{team_id}")
}

fn team_tpm_key(team_id: &str) -> String {
    format!("tpm:team:{team_id}")
}

fn org_rpm_key(org_id: &str) -> String {
    format!("rpm:org:{org_id}")
}

fn org_tpm_key(org_id: &str) -> String {
    format!("tpm:org:{org_id}")
}

fn project_rpm_key(project_id: &str) -> String {
    format!("rpm:project:{project_id}")
}

fn project_tpm_key(project_id: &str) -> String {
    format!("tpm:project:{project_id}")
}

/// Concurrent-request gauge key (no TTL). Released on post_success/failure.
fn parallel_key(api_key_hash: &str) -> String {
    format!("parallel:{api_key_hash}")
}

/// A hook that enforces per-key requests-per-minute (RPM) and tokens-per-minute
/// (TPM) limits. Uses global defaults from `oxllm_settings` when per-key limits are unset.
///
/// Concurrent-request gauge: when `CallContext::max_parallel_requests` is set
/// (from org records today), this hook reserves one slot via
/// [`Cache::incr_if_below`] with **no TTL** (true gauge) in `pre_call` and
/// releases it in `post_success` / `post_failure`. That matches the reference
/// intent without the sliding-window bug (U5 / commit 561b6796bc).
pub struct RateLimitHook {
    /// Backing cache (in-memory or Redis).
    pub cache: Arc<dyn Cache>,
    /// Maximum requests per minute per key.
    pub rpm: Option<u32>,
    /// Maximum tokens per minute per key.
    pub tpm: Option<u32>,
    /// Optional budget-throttle percentage in `[0.0, 1.0]`. When set, a
    /// scope (key/team/org/project) whose spend is at or over its
    /// `max_budget` has its effective RPM/TPM limit multiplied by this
    /// fraction instead of being allowed the full rate. The throttle
    /// tightens the rate limit; it does not replace the budget hard-block
    /// (enforced separately by `BudgetHook`). `None` disables throttling.
    pub budget_throttle_pct: Option<f64>,
}

impl RateLimitHook {
    /// Create with explicit RPM/TPM (None = no limit on that dimension).
    pub fn new(cache: Arc<dyn Cache>, rpm: Option<u32>, tpm: Option<u32>) -> Self {
        Self {
            cache,
            rpm,
            tpm,
            budget_throttle_pct: None,
        }
    }

    /// Builder: set the budget-throttle percentage. When set (e.g. `0.5`),
    /// a scope over its `max_budget` gets its RPM/TPM limit halved.
    pub fn with_budget_throttle_pct(mut self, pct: Option<f64>) -> Self {
        self.budget_throttle_pct = pct;
        self
    }

    /// Compute the effective RPM/TPM limit after applying budget
    /// throttling. When `budget_throttle_pct` is set AND the scope's spend
    /// counter (`spend_key`) is at or over its `max_budget`, the limit is
    /// multiplied by `budget_throttle_pct` (floored at 1 so a throttled
    /// scope can always make at least one request per window). Returns
    /// `(effective_limit, was_throttled)`.
    ///
    /// Best-effort, cache-dependent: the spend reading goes through
    /// `counter_value(...).unwrap_or(0.0)`, so during a cache outage the
    /// reading is `0.0` and the throttle does NOT engage (the full RPM/TPM
    /// limit applies). This is acceptable because the throttle is a soft
    /// mechanism — the hard budget block lives in `BudgetHook::pre_call`,
    /// which routes the same spend key through `incr_if_below` and rejects
    /// fail-closed on the same outage. The narrow gap (a Redis hiccup that
    /// resolves between this read and the budget reservation) is bounded
    /// and low-severity; making the throttle fail-closed would duplicate
    /// the hard block and risk double-rejecting.
    async fn apply_budget_throttle(
        &self,
        spend_key: &str,
        max_budget: Option<f64>,
        limit: u32,
    ) -> (u32, bool) {
        let Some(pct) = self.budget_throttle_pct else {
            return (limit, false);
        };
        let Some(max_budget) = max_budget else {
            return (limit, false);
        };
        if max_budget <= 0.0 || pct <= 0.0 {
            return (limit, false);
        }
        let current = self.cache.counter_value(spend_key).await.unwrap_or(0.0);
        if current < max_budget {
            return (limit, false);
        }
        // Over budget: throttle the effective limit.
        let throttled = ((limit as f64) * pct).round() as u32;
        let throttled = throttled.max(1);
        (throttled, true)
    }

    /// Atomically reserve `delta` against `limit` on `key`. Returns the
    /// new counter value on success, or `Err(Error::RateLimited)` when the
    /// reservation would exceed the ceiling (the counter is left unchanged
    /// in that case — no rollback required).
    ///
    /// For `delta > 0` this delegates to [`Cache::incr_if_below`], which
    /// performs the read-check-write atomically (write-lock for in-memory,
    /// Lua for Redis), so concurrent callers can never collectively exceed
    /// the limit. For `delta == 0` (e.g. an empty-body TPM estimate) there
    /// is nothing to reserve, so a read-only "already at limit" check is
    /// used instead — matching the prior semantics that a zero-delta caller
    /// is still rejected once the counter has reached the ceiling.
    async fn reserve(&self, key: &str, delta: f64, limit: u32, reason: &str) -> Result<f64, Error> {
        let ttl = Some(MINUTE);
        if delta > 0.0 {
            match self
                .cache
                .incr_if_below(key, delta, limit as f64, ttl)
                .await?
            {
                IncrResult::Allowed(v) => Ok(v),
                IncrResult::Rejected(_) => Err(Error::RateLimited(format!(
                    "{reason} ({limit}) exceeded for this key"
                ))),
            }
        } else {
            // Zero/negative delta (e.g. an empty-body TPM estimate): no
            // counter change, but still gate on the ceiling through
            // `incr_if_below` so a backing-store outage (e.g. Redis down)
            // propagates as `Err` via `?` (fail-closed) instead of being
            // silently swallowed by `counter_value(...).unwrap_or(0.0)`. A
            // zero delta never exceeds the ceiling on its own, so we
            // additionally reject when the counter has already reached the
            // ceiling, preserving the prior "reject once at the limit"
            // semantics.
            match self
                .cache
                .incr_if_below(key, 0.0, limit as f64, ttl)
                .await?
            {
                IncrResult::Allowed(current) if current >= limit as f64 => Err(Error::RateLimited(
                    format!("{reason} ({limit}) exceeded for this key"),
                )),
                IncrResult::Allowed(current) => Ok(current),
                IncrResult::Rejected(_) => Err(Error::RateLimited(format!(
                    "{reason} ({limit}) exceeded for this key"
                ))),
            }
        }
    }

    /// Roll back the RPM reservations tracked in `reserved_rpm` by
    /// decrementing each counter by 1. Iterates in reverse so the most
    /// recently reserved key is released first (cosmetic — order doesn't
    /// matter for correctness since each key is independent). Best-effort:
    /// errors are swallowed because rollback must never mask the original
    /// rejection.
    ///
    /// Uses [`Cache::decr_if_exists`] instead of `incr(-1)`: a stream (or any
    /// slow request) whose rollback lands after the 60s window expired would
    /// otherwise mint a fresh **negative, never-expiring** counter that
    /// permanently grants the scope one extra request per window. Skipping
    /// the decrement on an expired counter accepts at most one over-counted
    /// request per window (same trade-off as the pipeline's
    /// `release_deployment_rpm_slot` and `BudgetHook`'s release).
    async fn release_rpm(&self, reserved_rpm: &[String]) {
        for k in reserved_rpm.iter().rev() {
            let _ = self.cache.decr_if_exists(k, 1.0).await;
        }
    }

    /// Roll back both RPM (unit increments) and TPM (`delta` per key)
    /// reservations. Used when a TPM reservation fails after one or more
    /// RPM/TPM reservations have already succeeded.
    async fn release_all(&self, reserved_rpm: &[String], reserved_tpm: &[(String, f64)]) {
        for (k, delta) in reserved_tpm.iter().rev() {
            let _ = self.cache.decr_if_exists(k, *delta).await;
        }
        self.release_rpm(reserved_rpm).await;
    }

    /// Reserve one concurrent-request gauge slot when `max_parallel_requests`
    /// is set. Fail-closed on cache errors.
    async fn reserve_parallel(&self, ctx: &mut CallContext) -> oxllm_core::Result<HookAction> {
        let Some(limit) = ctx.max_parallel_requests else {
            return Ok(HookAction::Continue);
        };
        let pkey = parallel_key(&ctx.api_key_hash);
        match self
            .cache
            .incr_if_below(&pkey, 1.0, limit as f64, None)
            .await
        {
            Ok(IncrResult::Allowed(_)) => {
                ctx.parallel_reserved = true;
                Ok(HookAction::Continue)
            }
            Ok(IncrResult::Rejected(_)) => Ok(HookAction::Reject(Error::RateLimited(format!(
                "max_parallel_requests ({limit}) exceeded for this key"
            )))),
            Err(e) => Ok(HookAction::Reject(e)),
        }
    }

    /// Release a concurrent-request gauge slot when one was reserved.
    async fn release_parallel_if_reserved(&self, ctx: &CallContext) {
        if !ctx.parallel_reserved {
            return;
        }
        if let Err(error) = self
            .cache
            .decr_if_exists(&parallel_key(&ctx.api_key_hash), 1.0)
            .await
        {
            tracing::warn!(%error, "parallel request slot release failed");
        }
    }

    /// Release the TPM prompt-estimate reservation across every configured
    /// TPM scope. Used on the cache-hit path of `post_success`, where the
    /// request was served without an upstream call so no tokens were
    /// consumed: the prompt-estimate reserved in `pre_call` would otherwise
    /// leak for the rest of the 60s window. RPM is intentionally NOT
    /// released here (cache hits still count as a request per the
    /// architecture contract). Best-effort, mirroring `release_rpm` —
    /// including its [`Cache::decr_if_exists`] guard against minting a
    /// negative never-expiring counter when the 60s window already expired
    /// (e.g. a stream failing after more than a minute).
    async fn release_tpm(&self, ctx: &CallContext, estimate: f64) {
        if estimate <= 0.0 {
            return;
        }
        if most_restrictive_u32(ctx.tpm, self.tpm).is_some() {
            let _ = self
                .cache
                .decr_if_exists(&tpm_key(&ctx.api_key_hash), estimate)
                .await;
        }
        if ctx.team_tpm.is_some() {
            if let Some(team_id) = ctx.team_id.as_deref() {
                let _ = self
                    .cache
                    .decr_if_exists(&team_tpm_key(team_id), estimate)
                    .await;
            }
        }
        if ctx.org_tpm.is_some() {
            if let Some(org_id) = ctx.org_id.as_deref() {
                let _ = self
                    .cache
                    .decr_if_exists(&org_tpm_key(org_id), estimate)
                    .await;
            }
        }
        if ctx.project_tpm.is_some() {
            if let Some(project_id) = ctx.project_id.as_deref() {
                let _ = self
                    .cache
                    .decr_if_exists(&project_tpm_key(project_id), estimate)
                    .await;
            }
        }
    }

    /// Settle the TPM reservation against actual usage.
    ///
    /// `pre_call` sizes the counter with a 60s (`MINUTE`) TTL. If the
    /// upstream call outlives that window (slow model, network stall), the
    /// counter expires before `post_success` runs; the `incr(delta,
    /// Some(MINUTE))` below then recreates it with a **fresh** 60s window
    /// starting from the post_success timestamp. The effective window for
    /// that one request's tokens is therefore up to ~120s (pre_call TTL +
    /// post_success TTL). This is an inherent property of the
    /// reserve-then-adjust pattern with TTL-based eviction — the reference
    /// instead anchors on calendar windows via `budget_reservation`
    /// start-time tracking. The impact is minor: only a slow request's own
    /// delta lingers slightly longer than the nominal 60s window, and only
    /// for that request. Keeping the code as-is is the documented
    /// trade-off; stricter windows would require a `window_start`
    /// timestamp on the counter.
    async fn settle_tpm(&self, ctx: &CallContext, resp: &ResponseSummary) {
        if ctx.is_master {
            return;
        }
        let key_tpm = most_restrictive_u32(ctx.tpm, self.tpm);
        let team_tpm = ctx.team_tpm;
        let org_tpm = ctx.org_tpm;
        let project_tpm = ctx.project_tpm;
        if key_tpm.is_none() && team_tpm.is_none() && org_tpm.is_none() && project_tpm.is_none() {
            return;
        }

        // Cache hit: the request was served without an upstream call, so no
        // tokens were consumed. RPM is intentionally kept (cache hits count
        // as a request per the architecture contract), but the TPM
        // prompt-estimate reserved in `pre_call` must be released —
        // otherwise a cache-heavy workload with large prompts would
        // prematurely exhaust a key's TPM limit despite zero upstream token
        // usage. Only release when `pre_call` actually reserved (i.e. the
        // token estimate is stashed); when it is absent nothing was
        // reserved and there is nothing to release.
        if ctx.cache_hit {
            if let Some(&reserved) = ctx.token_estimate.get() {
                self.release_tpm(ctx, reserved as f64).await;
            }
            return;
        }

        let actual = resp
            .usage
            .as_ref()
            .map(|u| u.total_tokens as f64)
            .unwrap_or(0.0);
        // post_success reuses the pre_call estimate when available; otherwise
        // it recomputes. Borrowing the Arc-backed body avoids the deep clone
        // of the JSON tree on this path; the estimate is usually already
        // cached from pre_call in `token_estimate`.
        let reserved = match ctx.token_estimate.get() {
            Some(cached) => *cached as f64,
            None => {
                crate::tokens::estimate_request_tokens_async(ctx.request_body.as_ref()).await as f64
            }
        };
        let delta = actual - reserved;
        // Exact-zero short-circuit (NOT a tolerance): see BudgetHook::adjust.
        if delta == 0.0 {
            return;
        }
        if key_tpm.is_some() {
            if let Err(e) = self
                .cache
                .incr(&tpm_key(&ctx.api_key_hash), delta, Some(MINUTE))
                .await
            {
                tracing::warn!(error = %e, "tpm post_success adjust failed");
            }
        }
        if team_tpm.is_some() {
            if let Some(team_id) = ctx.team_id.as_deref() {
                if let Err(e) = self
                    .cache
                    .incr(&team_tpm_key(team_id), delta, Some(MINUTE))
                    .await
                {
                    tracing::warn!(error = %e, "team tpm post_success adjust failed");
                }
            }
        }
        if org_tpm.is_some() {
            if let Some(org_id) = ctx.org_id.as_deref() {
                if let Err(e) = self
                    .cache
                    .incr(&org_tpm_key(org_id), delta, Some(MINUTE))
                    .await
                {
                    tracing::warn!(error = %e, "org tpm post_success adjust failed");
                }
            }
        }
        if project_tpm.is_some() {
            if let Some(project_id) = ctx.project_id.as_deref() {
                if let Err(e) = self
                    .cache
                    .incr(&project_tpm_key(project_id), delta, Some(MINUTE))
                    .await
                {
                    tracing::warn!(error = %e, "project tpm post_success adjust failed");
                }
            }
        }
    }
}

#[async_trait]
impl Hook for RateLimitHook {
    async fn pre_call(&self, ctx: &mut CallContext) -> oxllm_core::Result<HookAction> {
        if ctx.is_master {
            return Ok(HookAction::Continue);
        }

        let key_rpm = most_restrictive_u32(ctx.rpm, self.rpm);
        let key_tpm = most_restrictive_u32(ctx.tpm, self.tpm);
        let team_rpm = ctx.team_rpm;
        let team_tpm = ctx.team_tpm;
        let org_rpm = ctx.org_rpm;
        let org_tpm = ctx.org_tpm;
        let project_rpm = ctx.project_rpm;
        let project_tpm = ctx.project_tpm;

        if key_rpm.is_none()
            && key_tpm.is_none()
            && team_rpm.is_none()
            && team_tpm.is_none()
            && org_rpm.is_none()
            && org_tpm.is_none()
            && project_rpm.is_none()
            && project_tpm.is_none()
        {
            // No RPM/TPM — still enforce the concurrent-request gauge when set.
            return self.reserve_parallel(ctx).await;
        }

        // H9: budget throttle — when a scope is over its `max_budget` and
        // `budget_throttle_pct` is configured, the scope's effective
        // RPM/TPM limit is multiplied by the fraction. The throttle is
        // evaluated per-scope by reading the spend counter before each
        // reservation. The local `was_throttled` flag aggregates the
        // signal across scopes so it can be logged once at the end.
        let mut was_throttled = false;

        // Track the most-restrictive remaining counters across key and team
        // so we can stamp `x-ratelimit-*` headers on the response. The binding
        // scope's limit is tracked alongside so the `limit-*` header can be
        // stamped from a consistent pre_call snapshot (instead of re-deriving
        // `remaining + live_counter`, which races under concurrency).
        let mut remaining_rpm: Option<u32> = None;
        let mut remaining_tpm: Option<u32> = None;
        let mut limit_rpm: Option<u32> = None;
        let mut limit_tpm: Option<u32> = None;

        // Track each successfully reserved key so that on a later failure we
        // can roll back in reverse order without per-branch release code
        // (matching the `BudgetHook` pattern). RPM and TPM are tracked
        // separately because the TPM delta varies per-key estimate while RPM
        // is always a unit increment.
        let mut reserved_rpm: Vec<String> = Vec::new();
        let mut reserved_tpm: Vec<(String, f64)> = Vec::new();

        // Key-level RPM.
        if let Some(limit) = key_rpm {
            let (limit, throttled) = self
                .apply_budget_throttle(&spend_key(&ctx.api_key_hash), ctx.max_budget, limit)
                .await;
            if throttled {
                was_throttled = true;
            }
            let key = rpm_key(&ctx.api_key_hash);
            match self.reserve(&key, 1.0, limit, "rpm limit").await {
                Ok(new_value) => {
                    let remaining = saturated_remaining(limit, new_value);
                    remaining_rpm = Some(remaining);
                    limit_rpm = Some(limit);
                    reserved_rpm.push(key);
                }
                Err(err) => return Ok(HookAction::Reject(err)),
            }
        }
        // Team-level RPM — only when the team has its own ceiling.
        if let (Some(limit), Some(team_id)) = (team_rpm, ctx.team_id.as_deref()) {
            let (limit, throttled) = self
                .apply_budget_throttle(&team_spend_key(team_id), ctx.team_max_budget, limit)
                .await;
            if throttled {
                was_throttled = true;
            }
            let tkey = team_rpm_key(team_id);
            match self.reserve(&tkey, 1.0, limit, "team rpm limit").await {
                Ok(new_value) => {
                    let team_remaining = saturated_remaining(limit, new_value);
                    if remaining_rpm.is_none_or(|prev| team_remaining < prev) {
                        remaining_rpm = Some(team_remaining);
                        limit_rpm = Some(limit);
                    }
                    reserved_rpm.push(tkey);
                }
                Err(err) => {
                    self.release_rpm(&reserved_rpm).await;
                    return Ok(HookAction::Reject(err));
                }
            }
        }
        // Org-level RPM.
        if let (Some(limit), Some(org_id)) = (org_rpm, ctx.org_id.as_deref()) {
            let (limit, throttled) = self
                .apply_budget_throttle(&org_spend_key(org_id), ctx.org_max_budget, limit)
                .await;
            if throttled {
                was_throttled = true;
            }
            let okey = org_rpm_key(org_id);
            match self.reserve(&okey, 1.0, limit, "org rpm limit").await {
                Ok(new_value) => {
                    let org_remaining = saturated_remaining(limit, new_value);
                    if remaining_rpm.is_none_or(|prev| org_remaining < prev) {
                        remaining_rpm = Some(org_remaining);
                        limit_rpm = Some(limit);
                    }
                    reserved_rpm.push(okey);
                }
                Err(err) => {
                    self.release_rpm(&reserved_rpm).await;
                    return Ok(HookAction::Reject(err));
                }
            }
        }
        // Project-level RPM.
        if let (Some(limit), Some(project_id)) = (project_rpm, ctx.project_id.as_deref()) {
            let (limit, throttled) = self
                .apply_budget_throttle(
                    &project_spend_key(project_id),
                    ctx.project_max_budget,
                    limit,
                )
                .await;
            if throttled {
                was_throttled = true;
            }
            let pkey = project_rpm_key(project_id);
            match self.reserve(&pkey, 1.0, limit, "project rpm limit").await {
                Ok(new_value) => {
                    let project_remaining = saturated_remaining(limit, new_value);
                    if remaining_rpm.is_none_or(|prev| project_remaining < prev) {
                        remaining_rpm = Some(project_remaining);
                        limit_rpm = Some(limit);
                    }
                    reserved_rpm.push(pkey);
                }
                Err(err) => {
                    self.release_rpm(&reserved_rpm).await;
                    return Ok(HookAction::Reject(err));
                }
            }
        }

        let estimate = if key_tpm.is_some()
            || team_tpm.is_some()
            || org_tpm.is_some()
            || project_tpm.is_some()
        {
            // H17: cache the prompt-token estimate so BudgetHook::pre_call
            // (which runs after this hook in the chain) can reuse it instead
            // of re-running tokenization on the same body.
            let est = match ctx.token_estimate.get() {
                Some(cached) => *cached,
                None => {
                    let est =
                        crate::tokens::estimate_request_tokens_async(ctx.request_body.as_ref())
                            .await;
                    let _ = ctx.token_estimate.set(est);
                    est
                }
            };
            est as f64
        } else {
            0.0
        };

        if let Some(limit) = key_tpm {
            let (limit, throttled) = self
                .apply_budget_throttle(&spend_key(&ctx.api_key_hash), ctx.max_budget, limit)
                .await;
            if throttled {
                was_throttled = true;
            }
            let key = tpm_key(&ctx.api_key_hash);
            match self.reserve(&key, estimate, limit, "tpm limit").await {
                Ok(new_value) => {
                    let remaining = saturated_remaining_f64(limit as f64, new_value);
                    remaining_tpm = Some(remaining);
                    limit_tpm = Some(limit);
                    reserved_tpm.push((key, estimate));
                }
                Err(err) => {
                    self.release_all(&reserved_rpm, &reserved_tpm).await;
                    return Ok(HookAction::Reject(err));
                }
            }
        }
        if let (Some(limit), Some(team_id)) = (team_tpm, ctx.team_id.as_deref()) {
            let (limit, throttled) = self
                .apply_budget_throttle(&team_spend_key(team_id), ctx.team_max_budget, limit)
                .await;
            if throttled {
                was_throttled = true;
            }
            let tkey = team_tpm_key(team_id);
            match self.reserve(&tkey, estimate, limit, "team tpm limit").await {
                Ok(new_value) => {
                    let team_remaining = saturated_remaining_f64(limit as f64, new_value);
                    if remaining_tpm.is_none_or(|prev| team_remaining < prev) {
                        remaining_tpm = Some(team_remaining);
                        limit_tpm = Some(limit);
                    }
                    reserved_tpm.push((tkey, estimate));
                }
                Err(err) => {
                    self.release_all(&reserved_rpm, &reserved_tpm).await;
                    return Ok(HookAction::Reject(err));
                }
            }
        }
        if let (Some(limit), Some(org_id)) = (org_tpm, ctx.org_id.as_deref()) {
            let (limit, throttled) = self
                .apply_budget_throttle(&org_spend_key(org_id), ctx.org_max_budget, limit)
                .await;
            if throttled {
                was_throttled = true;
            }
            let okey = org_tpm_key(org_id);
            match self.reserve(&okey, estimate, limit, "org tpm limit").await {
                Ok(new_value) => {
                    let org_remaining = saturated_remaining_f64(limit as f64, new_value);
                    if remaining_tpm.is_none_or(|prev| org_remaining < prev) {
                        remaining_tpm = Some(org_remaining);
                        limit_tpm = Some(limit);
                    }
                    reserved_tpm.push((okey, estimate));
                }
                Err(err) => {
                    self.release_all(&reserved_rpm, &reserved_tpm).await;
                    return Ok(HookAction::Reject(err));
                }
            }
        }
        if let (Some(limit), Some(project_id)) = (project_tpm, ctx.project_id.as_deref()) {
            let (limit, throttled) = self
                .apply_budget_throttle(
                    &project_spend_key(project_id),
                    ctx.project_max_budget,
                    limit,
                )
                .await;
            if throttled {
                was_throttled = true;
            }
            let pkey = project_tpm_key(project_id);
            match self
                .reserve(&pkey, estimate, limit, "project tpm limit")
                .await
            {
                Ok(new_value) => {
                    let project_remaining = saturated_remaining_f64(limit as f64, new_value);
                    if remaining_tpm.is_none_or(|prev| project_remaining < prev) {
                        remaining_tpm = Some(project_remaining);
                        limit_tpm = Some(limit);
                    }
                    reserved_tpm.push((pkey, estimate));
                }
                Err(err) => {
                    self.release_all(&reserved_rpm, &reserved_tpm).await;
                    return Ok(HookAction::Reject(err));
                }
            }
        }

        // The counter window is 60 seconds (MINUTE), so all `reset_*` values
        // are 60. We only set a field when its corresponding limit applied.
        ctx.ratelimit_info = RateLimitInfo {
            remaining_rpm,
            remaining_tpm,
            limit_rpm,
            limit_tpm,
            reset_rpm_secs: if remaining_rpm.is_some() {
                Some(60)
            } else {
                None
            },
            reset_tpm_secs: if remaining_tpm.is_some() {
                Some(60)
            } else {
                None
            },
        };

        // H9: log the throttle event once for observability when any scope
        // was throttled. (The throttle signal could be stashed on
        // `CallContext` if a response header is ever needed — do not
        // re-introduce a thread-local: tasks migrate between worker threads
        // at `.await` points, so a thread-local would read stale/cross-
        // request values.)
        if was_throttled {
            tracing::warn!(
                api_key_hash = %ctx.api_key_hash,
                "budget throttle applied: scope over max_budget, RPM/TPM limit reduced"
            );
        }

        // Reserve the concurrent-request gauge last so RPM/TPM failures
        // do not need to roll it back.
        match self.reserve_parallel(ctx).await {
            Ok(HookAction::Continue) => Ok(HookAction::Continue),
            Ok(other) => {
                self.release_all(&reserved_rpm, &reserved_tpm).await;
                Ok(other)
            }
            Err(e) => {
                self.release_all(&reserved_rpm, &reserved_tpm).await;
                Err(e)
            }
        }
    }

    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        self.release_parallel_if_reserved(ctx).await;
        self.settle_tpm(ctx, resp).await;
    }

    async fn release_concurrency(&self, ctx: &CallContext) {
        self.release_parallel_if_reserved(ctx).await;
    }

    async fn post_failure(&self, ctx: &CallContext, err: &Error) {
        // Always release the concurrency gauge; it is not an RPM window.
        self.release_parallel_if_reserved(ctx).await;
        // `is_master` bypasses rate limits. The `cache_hit` guard that used
        // to live here is intentionally removed: if `post_failure` is
        // reached at all (even on a cache-hit request whose response later
        // failed downstream), the RPM reservation must be released rather
        // than leaking until the 60s TTL. The happy cache-hit path is
        // handled by `post_success` (which keeps RPM); `post_failure`
        // handles the unhappy path.
        if ctx.is_master {
            return;
        }
        // Only release the RPM reservation when the request did NOT reach
        // the upstream provider. For `Error::Upstream` (the provider
        // returned a response — 4xx/5xx, including 429s which map to
        // `Upstream { status: 429 }`) the provider processed the request
        // and it counted against the provider's rate limit; releasing the
        // local RPM slot would allow an immediate retry that could exceed
        // the provider's RPM ceiling. Pre-flight failures (`Network`,
        // local hook rejections during chain rollback, serialization, ...)
        // never reached the provider, so their slots are released as before
        // so a burst of failures cannot temporarily exhaust local RPM.
        if ctx.reached_upstream || matches!(err, Error::Upstream { .. }) {
            return;
        }
        // Release the RPM slots reserved in `pre_call`. The upstream call
        // failed before reaching the provider, so no RPM was actually
        // consumed against it; left alone, the reservation would linger
        // until the 60s TTL and a burst of failures could temporarily
        // exhaust RPM.
        let key_rpm = most_restrictive_u32(ctx.rpm, self.rpm);
        let team_rpm = ctx.team_rpm;
        let org_rpm = ctx.org_rpm;
        let project_rpm = ctx.project_rpm;
        if key_rpm.is_none() && team_rpm.is_none() && org_rpm.is_none() && project_rpm.is_none() {
            return;
        }
        // Build the same key list `pre_call` reserves (in the same order) and
        // release via the shared helper so rollback semantics stay in sync.
        let mut reserved_rpm: Vec<String> = Vec::new();
        if key_rpm.is_some() {
            reserved_rpm.push(rpm_key(&ctx.api_key_hash));
        }
        if team_rpm.is_some() {
            if let Some(team_id) = ctx.team_id.as_deref() {
                reserved_rpm.push(team_rpm_key(team_id));
            }
        }
        if org_rpm.is_some() {
            if let Some(org_id) = ctx.org_id.as_deref() {
                reserved_rpm.push(org_rpm_key(org_id));
            }
        }
        if project_rpm.is_some() {
            if let Some(project_id) = ctx.project_id.as_deref() {
                reserved_rpm.push(project_rpm_key(project_id));
            }
        }
        self.release_rpm(&reserved_rpm).await;
        // Release the TPM prompt-estimate reservation too, mirroring the
        // cache-hit settlement in `post_success`. A pre-flight failure
        // (network/serialization/local rejection) consumed no upstream tokens,
        // so leaving the estimate to decay via TTL would let a burst of
        // large-prompt retries drain TPM and block legitimate traffic for up
        // to 60s. The asymmetry with the cache-hit path (which releases TPM)
        // is removed: both "no upstream tokens consumed" arms now release.
        // Provider-side failures (`Error::Upstream`) returned above and keep
        // both RPM and TPM.
        if let Some(&estimate) = ctx.token_estimate.get() {
            self.release_tpm(ctx, estimate as f64).await;
        }
    }
}

/// Pick the most restrictive (lowest) of two optional u32 limits.
fn most_restrictive_u32(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    }
}

/// Compute `limit - current` as a u32, clamped at 0. `current` is the counter
/// value AFTER incrementing, so this is the number of slots remaining in the
/// window. Defensive clamp protects against inconsistent cache state where
/// `current > limit` (e.g. concurrent in-flight increments that briefly
/// overshoot before a rollback).
fn saturated_remaining(limit: u32, current: f64) -> u32 {
    if current >= limit as f64 {
        return 0;
    }
    (limit as f64 - current).round() as u32
}

/// Same as [`saturated_remaining`] but accepts the counter value as `f64`
/// (TPM is estimated as `f64` and may overshoot between estimates).
fn saturated_remaining_f64(limit: f64, current: f64) -> u32 {
    if current >= limit {
        return 0;
    }
    (limit - current).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use oxllm_core::UsageSummary;
    use serde_json::json;

    fn make_ctx() -> CallContext {
        CallContext::for_test("gpt-4o")
    }

    fn make_response(tokens: u64) -> ResponseSummary {
        ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: tokens / 2,
                completion_tokens: tokens / 2,
                total_tokens: tokens,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    async fn no_limits_always_continues() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache, None, None);
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("continue");
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn under_rpm_continues() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 5.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("continue");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(cache.counter_value("rpm:abc").await, Some(6.0));
    }

    #[tokio::test]
    async fn at_rpm_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 10.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(10.0),
            "rejected reservation must be rolled back"
        );
    }

    #[tokio::test]
    async fn at_tpm_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("tpm:abc", 1000.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), None, Some(1000));
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert_eq!(
            cache.counter_value("tpm:abc").await,
            Some(1000.0),
            "rejected TPM reservation must be rolled back"
        );
    }

    #[tokio::test]
    async fn max_parallel_requests_gauge_reserves_and_releases() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), None, None);
        let mut ctx = make_ctx();
        ctx.max_parallel_requests = Some(2);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert!(ctx.parallel_reserved);
        assert_eq!(cache.counter_value("parallel:abc").await, Some(1.0));

        let resp = make_response(1);
        hook.post_success(&ctx, &resp).await;
        assert_eq!(
            cache.counter_value("parallel:abc").await,
            Some(0.0),
            "post_success must release the gauge slot"
        );
    }

    #[tokio::test]
    async fn release_concurrency_frees_gauge_without_touching_rpm() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        ctx.max_parallel_requests = Some(2);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(cache.counter_value("parallel:abc").await, Some(1.0));
        assert_eq!(cache.counter_value("rpm:abc").await, Some(1.0));

        hook.release_concurrency(&ctx).await;
        assert_eq!(
            cache.counter_value("parallel:abc").await,
            Some(0.0),
            "disconnect path must free the no-TTL gauge"
        );
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(1.0),
            "disconnect must keep the RPM debit"
        );
    }

    #[tokio::test]
    async fn max_parallel_requests_rejects_when_full() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("parallel:abc", 2.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), None, None);
        let mut ctx = make_ctx();
        ctx.max_parallel_requests = Some(2);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert!(!ctx.parallel_reserved);
        assert_eq!(cache.counter_value("parallel:abc").await, Some(2.0));
    }

    #[tokio::test]
    async fn zero_parallel_limit_blocks_without_reserving_a_slot() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), None, None);
        let mut ctx = make_ctx();
        ctx.max_parallel_requests = Some(0);
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Reject(_)
        ));
        assert!(!ctx.parallel_reserved);
        assert_eq!(
            cache.counter_value("parallel:abc").await.unwrap_or(0.0),
            0.0
        );
    }

    #[tokio::test]
    async fn post_success_adjusts_tpm_to_actual() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache.counter_value("tpm:abc").await.unwrap_or(0.0);

        let resp = make_response(50);
        hook.post_success(&ctx, &resp).await;
        let final_tpm = cache.counter_value("tpm:abc").await.unwrap_or(0.0);
        assert!(
            (final_tpm - 50.0).abs() < 1.0,
            "final TPM ({final_tpm}) should be ~50 (actual), reserved was {reserved}"
        );
    }

    #[tokio::test]
    async fn per_key_rpm_enforced_without_global() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 5.0, None).await;
        let hook = RateLimitHook::new(cache, None, None);
        let mut ctx = make_ctx();
        ctx.rpm = Some(5);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    #[tokio::test]
    async fn per_key_rpm_overrides_global() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 3.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(100), None);
        let mut ctx = make_ctx();
        ctx.rpm = Some(3);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(3.0),
            "rejected reservation must be rolled back"
        );
    }

    #[tokio::test]
    async fn per_key_tpm_enforced_without_global() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("tpm:abc", 1000.0, None).await;
        let hook = RateLimitHook::new(cache, None, None);
        let mut ctx = make_ctx();
        ctx.tpm = Some(1000);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    #[tokio::test]
    async fn per_key_limits_none_fall_back_to_global() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache, Some(10), None);
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn master_key_bypasses_pre_call_rate_limit() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 1_000_000.0, None).await;
        let _ = cache.incr("tpm:abc", 1_000_000.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(1), Some(1));
        let mut ctx = make_ctx();
        ctx.rpm = Some(1);
        ctx.tpm = Some(1);
        ctx.is_master = true;
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "master key should bypass rate limits"
        );
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(1_000_000.0),
            "master key should not reserve RPM"
        );
    }

    #[tokio::test]
    async fn master_key_skips_post_success_rate_adjust() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.is_master = true;
        let resp = make_response(50);
        hook.post_success(&ctx, &resp).await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            None,
            "master key should not touch RPM"
        );
        assert_eq!(
            cache.counter_value("tpm:abc").await,
            None,
            "master key should not touch TPM"
        );
    }

    #[tokio::test]
    async fn cache_hit_skips_post_success_rate_adjust() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        let resp = make_response(50);
        hook.post_success(&ctx, &resp).await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            None,
            "cache hits should not touch RPM"
        );
        assert_eq!(
            cache.counter_value("tpm:abc").await,
            None,
            "cache hits should not touch TPM"
        );
    }

    /// Full lifecycle for a cache hit: `pre_call` reserves both RPM (+1)
    /// and TPM (+prompt estimate) BEFORE the cache hook short-circuits.
    /// `post_success` with `cache_hit = true` must release the TPM estimate
    /// (no upstream tokens were consumed) while KEEPING the RPM reservation
    /// (cache hits count as a request per the architecture contract).
    /// Without the release, the TPM counter leaks the full prompt estimate
    /// on every cache hit.
    #[tokio::test]
    async fn cache_hit_releases_tpm_reservation_but_keeps_rpm() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there friend"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved_tpm = cache.counter_value("tpm:abc").await.unwrap_or(0.0);
        assert!(reserved_tpm > 0.0, "pre_call must reserve a TPM estimate");
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(1.0),
            "pre_call must reserve RPM"
        );

        // Simulate the cache-hook short-circuit: the orchestrator stamps
        // `cache_hit` and then runs `post_success` on every hook.
        ctx.cache_hit = true;
        hook.post_success(&ctx, &make_response(50)).await;

        let final_tpm = cache.counter_value("tpm:abc").await.unwrap_or(0.0);
        assert!(
            final_tpm.abs() < 1e-9,
            "cache hit must release the TPM reservation (reserved {reserved_tpm}), got {final_tpm}"
        );
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(1.0),
            "RPM reservation must be KEPT on a cache hit"
        );
    }

    #[test]
    fn most_restrictive_u32_picks_lower() {
        assert_eq!(most_restrictive_u32(Some(10), Some(20)), Some(10));
        assert_eq!(most_restrictive_u32(Some(20), Some(10)), Some(10));
        assert_eq!(most_restrictive_u32(Some(10), None), Some(10));
        assert_eq!(most_restrictive_u32(None, Some(10)), Some(10));
        assert_eq!(most_restrictive_u32(None, None), None);
    }

    #[tokio::test]
    async fn post_failure_releases_rpm_reservation() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        hook.pre_call(&mut ctx).await.expect("continue");
        assert_eq!(cache.counter_value("rpm:abc").await, Some(1.0));
        hook.post_failure(&ctx, &Error::Network("boom".to_string()))
            .await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(0.0),
            "post_failure must release the RPM reservation"
        );
    }

    /// A pre-flight failure (network error before the request reached the
    /// provider) must release the RPM slot so a burst of failures cannot
    /// temporarily exhaust local RPM. (Baseline for the upstream test
    /// below; kept explicit to pin the two policies side by side.)
    #[tokio::test]
    async fn post_failure_releases_rpm_on_network_error() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        hook.pre_call(&mut ctx).await.expect("continue");
        assert_eq!(cache.counter_value("rpm:abc").await, Some(1.0));
        hook.post_failure(&ctx, &Error::Network("connection reset".to_string()))
            .await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(0.0),
            "network (pre-flight) failures must release RPM"
        );
    }

    /// A provider-side failure (`Error::Upstream`, including 429s which map
    /// to `Upstream { status: 429 }`) means the provider processed the
    /// request and it counted against the provider's rate limit. The local
    /// RPM slot must be KEPT so an immediate retry cannot exceed the
    /// provider's RPM ceiling.
    #[tokio::test]
    async fn post_failure_keeps_rpm_on_upstream_error() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        hook.pre_call(&mut ctx).await.expect("continue");
        assert_eq!(cache.counter_value("rpm:abc").await, Some(1.0));

        hook.post_failure(
            &ctx,
            &Error::Upstream {
                status: 429,
                message: "rate limited".into(),
            },
        )
        .await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(1.0),
            "provider-side failures (429) must NOT release RPM"
        );

        // Same policy for a provider 5xx.
        hook.post_failure(
            &ctx,
            &Error::Upstream {
                status: 500,
                message: "boom".into(),
            },
        )
        .await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(1.0),
            "provider-side failures (5xx) must NOT release RPM"
        );
    }

    #[tokio::test]
    async fn post_failure_releases_team_rpm_reservation() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(100), None);
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.team_rpm = Some(10);
        hook.pre_call(&mut ctx).await.expect("continue");
        assert_eq!(cache.counter_value("rpm:team:t1").await, Some(1.0));
        hook.post_failure(&ctx, &Error::Network("boom".to_string()))
            .await;
        assert_eq!(cache.counter_value("rpm:abc").await, Some(0.0));
        assert_eq!(
            cache.counter_value("rpm:team:t1").await,
            Some(0.0),
            "post_failure must release the team RPM reservation"
        );
    }

    /// LOW (C31): a pre-flight failure (network error before the request
    /// reached the provider) must release the TPM prompt-estimate reservation
    /// too, mirroring the cache-hit settlement. Without this, a burst of
    /// large-prompt retries on a flaky upstream could drain TPM and block
    /// legitimate traffic for up to 60s even though no upstream tokens were
    /// consumed.
    #[tokio::test]
    async fn post_failure_releases_tpm_on_network_error() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there friend"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved_tpm = cache.counter_value("tpm:abc").await.unwrap_or(0.0);
        assert!(
            reserved_tpm > 0.0,
            "pre_call must reserve a TPM estimate, got {reserved_tpm}"
        );

        hook.post_failure(&ctx, &Error::Network("connection reset".to_string()))
            .await;
        let final_tpm = cache.counter_value("tpm:abc").await.unwrap_or(0.0);
        assert!(
            final_tpm.abs() < 1e-9,
            "network (pre-flight) failures must release the TPM estimate (reserved {reserved_tpm}), got {final_tpm}"
        );
    }

    /// LOW (C31): a provider-side failure (`Error::Upstream`, including 429)
    /// must still KEEP the TPM reservation — the provider processed the request
    /// so its TPM may have been consumed. This pins the asymmetry: only
    /// pre-flight failures release TPM.
    #[tokio::test]
    async fn post_failure_keeps_tpm_on_upstream_error() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there friend"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved_tpm = cache.counter_value("tpm:abc").await.unwrap_or(0.0);
        assert!(reserved_tpm > 0.0);

        hook.post_failure(
            &ctx,
            &Error::Upstream {
                status: 429,
                message: "rate limited".into(),
            },
        )
        .await;
        assert_eq!(
            cache.counter_value("tpm:abc").await,
            Some(reserved_tpm),
            "provider-side failures must NOT release TPM"
        );
    }

    #[tokio::test]
    async fn post_failure_master_key_skips_release() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        ctx.is_master = true;
        hook.post_failure(&ctx, &Error::Network("boom".to_string()))
            .await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            None,
            "master key must not touch RPM on failure"
        );
    }

    #[tokio::test]
    async fn global_more_restrictive_than_per_key_wins() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 5.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(5), None);
        let mut ctx = make_ctx();
        ctx.rpm = Some(100);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "global=5 must override per-key=100 when more restrictive"
        );
        assert_eq!(cache.counter_value("rpm:abc").await, Some(5.0));
    }

    /// With the optimistic-reservation pattern, concurrent callers can never
    /// exceed the limit: every successful `pre_call` atomically consumes a
    /// slot, and rejected reservations are rolled back.
    #[tokio::test]
    async fn concurrent_pre_call_never_exceeds_rpm_limit() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = Arc::new(RateLimitHook::new(cache.clone(), None, None));
        let limit = 3u32;
        let mut handles = Vec::new();
        for _ in 0..8 {
            let hook = hook.clone();
            handles.push(tokio::spawn(async move {
                let mut ctx = make_ctx();
                ctx.rpm = Some(limit);
                hook.pre_call(&mut ctx).await
            }));
        }
        let mut allowed = 0u32;
        for h in handles {
            if matches!(h.await.expect("join").expect("ok"), HookAction::Continue) {
                allowed += 1;
            }
        }
        assert_eq!(
            allowed, limit,
            "exactly `limit` callers should be allowed; got {allowed}"
        );
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(limit as f64),
            "counter should hold exactly `limit` after concurrent run"
        );
    }

    #[tokio::test]
    async fn team_rpm_counter_check_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:team:t1", 10.0, None).await;
        // Global key RPM is high; only team_rpm is tight.
        let hook = RateLimitHook::new(cache.clone(), Some(1000), None);
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.team_rpm = Some(10);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(0.0),
            "key-level reservation must be released when the request is rejected"
        );
        assert_eq!(
            cache.counter_value("rpm:team:t1").await,
            Some(10.0),
            "team reservation must be rolled back when over limit"
        );
    }

    #[tokio::test]
    async fn team_rpm_not_checked_without_team_limit() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:team:t1", 1_000.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        // team_rpm is None — high team counter must not block.
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    /// Key with RPM=10, after 3 prior requests, pre_call should stash
    /// `remaining_rpm = 6` (one more is consumed by this call).
    #[tokio::test]
    async fn pre_call_stashes_remaining_rpm_after_in_flight_reservation() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 3.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(ctx.ratelimit_info.remaining_rpm, Some(6));
        assert_eq!(
            ctx.ratelimit_info.limit_rpm,
            Some(10),
            "binding key-scope limit must be stashed for a race-free limit header"
        );
        assert_eq!(ctx.ratelimit_info.reset_rpm_secs, Some(60));
        assert!(ctx.ratelimit_info.remaining_tpm.is_none());
        assert!(ctx.ratelimit_info.reset_tpm_secs.is_none());
    }

    #[tokio::test]
    async fn pre_call_stashes_remaining_tpm_when_tpm_limit_set() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("tpm:abc", 100.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), None, Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert!(ctx.ratelimit_info.remaining_tpm.is_some());
        assert!(ctx.ratelimit_info.remaining_tpm.unwrap() <= 900);
        assert_eq!(ctx.ratelimit_info.reset_tpm_secs, Some(60));
        assert!(ctx.ratelimit_info.remaining_rpm.is_none());
    }

    /// When both RPM and TPM limits apply, both dimensions should be stashed.
    #[tokio::test]
    async fn pre_call_stashes_both_rpm_and_tpm() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(ctx.ratelimit_info.remaining_rpm, Some(9));
        assert_eq!(ctx.ratelimit_info.limit_rpm, Some(10));
        assert!(ctx.ratelimit_info.remaining_tpm.is_some());
        assert_eq!(ctx.ratelimit_info.limit_tpm, Some(1000));
        assert_eq!(ctx.ratelimit_info.reset_rpm_secs, Some(60));
        assert_eq!(ctx.ratelimit_info.reset_tpm_secs, Some(60));
    }

    /// Master key bypasses rate limits → no `ratelimit_info` stashed.
    /// The response pipeline uses this signal to skip the headers.
    #[tokio::test]
    async fn pre_call_master_key_leaves_ratelimit_info_empty() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.is_master = true;
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(ctx.ratelimit_info.remaining_rpm, None);
        assert_eq!(ctx.ratelimit_info.remaining_tpm, None);
        assert_eq!(ctx.ratelimit_info.reset_rpm_secs, None);
        assert_eq!(ctx.ratelimit_info.reset_tpm_secs, None);
    }

    #[tokio::test]
    async fn pre_call_no_limits_leaves_ratelimit_info_empty() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), None, None);
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(ctx.ratelimit_info.remaining_rpm, None);
        assert_eq!(ctx.ratelimit_info.remaining_tpm, None);
    }

    /// `remaining_rpm` reflects the most restrictive of key vs team RPM.
    #[tokio::test]
    async fn pre_call_picks_lower_remaining_across_key_and_team() {
        let cache = Arc::new(InMemoryCache::default());
        // Key counter at 2 (limit 100 → 97 remaining after this call).
        let _ = cache.incr("rpm:abc", 2.0, None).await;
        // Team counter at 5 (limit 50 → 44 remaining after this call).
        let _ = cache.incr("rpm:team:t1", 5.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(100), None);
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.team_rpm = Some(50);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        // 44 (team) is lower than 97 (key) — should win.
        assert_eq!(ctx.ratelimit_info.remaining_rpm, Some(44));
    }

    /// LOW (C12): the stashed `limit_rpm` must follow the binding (most
    /// restrictive) scope, so the `limit-*` header reflects the ceiling that
    /// actually applied. With key=100 (97 left) and team=50 (44 left), team
    /// binds → limit_rpm = 50.
    #[tokio::test]
    async fn pre_call_stashes_binding_scope_limit_rpm() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:abc", 2.0, None).await;
        let _ = cache.incr("rpm:team:t1", 5.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(100), None);
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.team_rpm = Some(50);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(
            ctx.ratelimit_info.limit_rpm,
            Some(50),
            "limit_rpm must be the binding team scope (50), not the looser key scope (100)"
        );
        assert_eq!(ctx.ratelimit_info.remaining_rpm, Some(44));
    }

    #[test]
    fn saturated_remaining_clamps_at_zero() {
        assert_eq!(saturated_remaining(10, 5.0), 5);
        assert_eq!(saturated_remaining(10, 10.0), 0);
        assert_eq!(saturated_remaining(10, 99.0), 0);
    }

    #[test]
    fn saturated_remaining_f64_rounds_correctly() {
        assert_eq!(saturated_remaining_f64(100.0, 50.5), 50);
        assert_eq!(saturated_remaining_f64(100.0, 50.4), 50);
        assert_eq!(saturated_remaining_f64(100.0, 100.0), 0);
        assert_eq!(saturated_remaining_f64(100.0, 1_000.0), 0);
    }

    #[tokio::test]
    async fn org_rpm_counter_check_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:org:o1", 10.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(1000), None);
        let mut ctx = make_ctx();
        ctx.org_id = Some("o1".to_string());
        ctx.org_rpm = Some(10);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert_eq!(
            cache.counter_value("rpm:org:o1").await,
            Some(10.0),
            "org reservation must be rolled back when over limit"
        );
    }

    #[tokio::test]
    async fn org_rpm_rolls_back_key_reservation_on_reject() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:org:o1", 10.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(1000), None);
        let mut ctx = make_ctx();
        ctx.org_id = Some("o1".to_string());
        ctx.org_rpm = Some(10);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(0.0),
            "key reservation must be released when org RPM rejects"
        );
    }

    #[tokio::test]
    async fn project_rpm_counter_check_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("rpm:project:p1", 10.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(1000), None);
        let mut ctx = make_ctx();
        ctx.project_id = Some("p1".to_string());
        ctx.project_rpm = Some(10);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        assert_eq!(
            cache.counter_value("rpm:project:p1").await,
            Some(10.0),
            "project reservation must be rolled back when over limit"
        );
    }

    #[tokio::test]
    async fn org_rpm_passes_when_under_limit() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(1000), None);
        let mut ctx = make_ctx();
        ctx.org_id = Some("o1".to_string());
        ctx.org_rpm = Some(100);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(
            cache.counter_value("rpm:org:o1").await,
            Some(1.0),
            "org RPM must be reserved on success"
        );
    }

    #[tokio::test]
    async fn project_tpm_post_success_adjusts_to_actual() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), None, Some(1000));
        let mut ctx = make_ctx();
        ctx.project_id = Some("p1".to_string());
        ctx.project_tpm = Some(1000);
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let resp = make_response(50);
        hook.post_success(&ctx, &resp).await;
        let final_tpm = cache.counter_value("tpm:project:p1").await.unwrap_or(0.0);
        assert!(
            (final_tpm - 50.0).abs() < 5.0,
            "project TPM should reflect actual usage, got {final_tpm}"
        );
    }

    // ---------------------------------------------------------------
    // H9: budget throttle
    // ---------------------------------------------------------------

    /// Over-budget + throttle_pct set → effective RPM limit halved, so a
    /// counter that was under the original limit is now over the throttled
    /// limit and the request is rejected.
    #[tokio::test]
    async fn budget_throttle_reduces_limit_when_over_budget() {
        let cache = Arc::new(InMemoryCache::default());
        // Spend at 100 (max_budget=100 → over budget).
        let _ = cache.incr("spend:abc", 100.0, None).await;
        // RPM counter at 5 (throttled limit is also 5, so +1 = 6 > 5 rejects).
        // Under the original limit 10 this would have been allowed.
        let _ = cache.incr("rpm:abc", 5.0, None).await;
        let hook =
            RateLimitHook::new(cache.clone(), Some(10), None).with_budget_throttle_pct(Some(0.5));
        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "throttled RPM limit (5) must reject when current+1 exceeds it"
        );
    }

    /// Over-budget + throttle_pct set but counter still under throttled
    /// limit → allowed (throttled, not blocked).
    #[tokio::test]
    async fn budget_throttle_allows_when_under_throttled_limit() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 100.0, None).await;
        let hook =
            RateLimitHook::new(cache.clone(), Some(10), None).with_budget_throttle_pct(Some(0.5));
        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        // Throttled limit = 5, current rpm = 0, +1 = 1 < 5 → allow.
        assert!(
            matches!(action, HookAction::Continue),
            "request under throttled limit must be allowed"
        );
    }

    /// Over-budget but throttle_pct NOT set → no throttle, original limit
    /// applies (existing behavior).
    #[tokio::test]
    async fn budget_throttle_disabled_when_pct_unset() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 100.0, None).await;
        let _ = cache.incr("rpm:abc", 9.0, None).await;
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "no throttle when pct is unset, even if over budget"
        );
    }

    /// Under-budget + throttle_pct set → no throttle (spend < max_budget).
    #[tokio::test]
    async fn budget_throttle_inactive_when_under_budget() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 50.0, None).await;
        let hook =
            RateLimitHook::new(cache.clone(), Some(10), None).with_budget_throttle_pct(Some(0.5));
        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    /// Throttle applies to TPM limits too: over-budget halves the TPM ceiling.
    #[tokio::test]
    async fn budget_throttle_applies_to_tpm() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 100.0, None).await;
        // TPM counter above the throttled ceiling (50). With a non-trivial
        // token estimate, 50 + est > 50 → reject.
        let _ = cache.incr("tpm:abc", 50.0, None).await;
        let hook =
            RateLimitHook::new(cache.clone(), None, Some(100)).with_budget_throttle_pct(Some(0.5));
        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        // Throttled TPM = 100 * 0.5 = 50. Current = 50, +estimate > 50 → reject.
        assert!(
            matches!(action, HookAction::Reject(_)),
            "throttled TPM limit must reject when counter is at ceiling + estimate"
        );
    }

    /// No max_budget configured → throttle cannot engage (nothing to be
    /// "over"), even if pct is set.
    #[tokio::test]
    async fn budget_throttle_no_max_budget_no_throttle() {
        let cache = Arc::new(InMemoryCache::default());
        let hook =
            RateLimitHook::new(cache.clone(), Some(10), None).with_budget_throttle_pct(Some(0.5));
        let mut ctx = make_ctx();
        // ctx.max_budget is None
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    // ---------------------------------------------------------------
    // C31: post_failure cache-hit edge case, TPM window-stretch, throttle
    // best-effort behavior.
    // ---------------------------------------------------------------

    /// LOW (C31): `post_failure` must release the RPM reservation even when
    /// `ctx.cache_hit = true`. The guard that used to skip release on a
    /// cache hit is removed: if `post_failure` is reached at all (e.g. the
    /// cache-hit response later failed downstream), the RPM slot should be
    /// freed rather than leaking until the 60s TTL. The happy cache-hit
    /// path is handled by `post_success` (which keeps RPM).
    #[tokio::test]
    async fn post_failure_releases_rpm_even_when_cache_hit_set() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), None);
        let mut ctx = make_ctx();
        hook.pre_call(&mut ctx).await.expect("continue");
        assert_eq!(cache.counter_value("rpm:abc").await, Some(1.0));
        // Simulate the cache-hit response later failing downstream.
        ctx.cache_hit = true;
        hook.post_failure(
            &ctx,
            &Error::Network("downstream serialize failed".to_string()),
        )
        .await;
        assert_eq!(
            cache.counter_value("rpm:abc").await,
            Some(0.0),
            "post_failure must release RPM even when cache_hit is set"
        );
    }

    /// LOW (C31): documents the TPM window-stretch trade-off. `pre_call`
    /// creates the TPM counter with a 60s TTL; if the upstream call outlives
    /// that window the counter expires, and `post_success` then recreates it
    /// with a FRESH 60s window holding only the delta. Here the counter is
    /// removed (the post-expiry state also reached via `sweep_now`/TTL) to
    /// deterministically simulate expiry, then `post_success` is run.
    #[tokio::test]
    async fn tpm_post_success_after_ttl_expiry_recreates_counter() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), None, Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = match ctx.token_estimate.get() {
            Some(&v) => v as f64,
            None => panic!("pre_call must stash the token estimate"),
        };
        assert!(
            cache.counter_value("tpm:abc").await.is_some(),
            "pre_call must create the TPM counter"
        );

        // Simulate the 60s window expiring before post_success runs.
        cache.delete("tpm:abc").await;
        assert!(cache.counter_value("tpm:abc").await.is_none());

        // post_success recreates the counter with the delta (actual - reserved)
        // on a fresh 60s window. Use an actual clearly above the estimate.
        let actual = reserved + 40.0;
        let resp = make_response(actual as u64);
        hook.post_success(&ctx, &resp).await;

        let final_tpm = cache.counter_value("tpm:abc").await.unwrap_or(0.0);
        assert!(
            (final_tpm - 40.0).abs() < 1e-9,
            "expired TPM counter must be recreated holding only the delta (40.0), got {final_tpm}"
        );
    }

    /// A rollback that lands AFTER the 60s window expired (e.g. a stream
    /// failing more than a minute after `pre_call`) must not mint a
    /// negative, never-expiring counter: `decr_if_exists` skips the
    /// decrement on an absent/expired key, so a scope loses at most one
    /// slot for a single window instead of permanently gaining one extra
    /// request (or `estimate` tokens) per window. `incr(key, -n, None)`
    /// would recreate the key at `-n` with no TTL.
    #[tokio::test]
    async fn post_failure_after_window_expiry_does_not_mint_negative_counters() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = RateLimitHook::new(cache.clone(), Some(10), Some(1000));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there friend"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        assert_eq!(cache.counter_value("rpm:abc").await, Some(1.0));
        assert!(cache.counter_value("tpm:abc").await.is_some());

        // Simulate the 60s window expiring before the failure accounting
        // runs (a long stream whose upstream died late).
        cache.delete("rpm:abc").await;
        cache.delete("tpm:abc").await;
        assert!(cache.counter_value("rpm:abc").await.is_none());
        assert!(cache.counter_value("tpm:abc").await.is_none());

        hook.post_failure(
            &ctx,
            &Error::Network("stream failed after the window expired".into()),
        )
        .await;
        assert!(
            cache.counter_value("rpm:abc").await.is_none(),
            "expired-window rollback must not mint a negative RPM counter"
        );
        assert!(
            cache.counter_value("tpm:abc").await.is_none(),
            "expired-window rollback must not mint a negative TPM counter"
        );
    }

    /// A cache whose spend read fails (simulating a Redis outage). Used to
    /// verify the budget throttle is best-effort and does not engage when
    /// the cache is unavailable.
    struct FailingCounterCache;

    #[async_trait]
    impl Cache for FailingCounterCache {
        async fn get(&self, _key: &str) -> Option<String> {
            None
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
            Err(Error::Network("cache backend unavailable".into()))
        }
        async fn try_counter_value(&self, _key: &str) -> Result<Option<f64>, Error> {
            Err(Error::Network("simulated spend-counter outage".into()))
        }
    }

    /// LOW (C31): the budget throttle is best-effort and cache-dependent.
    /// When the spend-counter read fails (Redis outage), the reading falls
    /// back to `0.0` and the throttle does NOT engage — the full limit
    /// applies. This is acceptable because the hard budget block lives in
    /// `BudgetHook::pre_call` (fail-closed on the same outage). Here we
    /// pin the soft-throttle behavior directly: outage → unthrottled.
    #[tokio::test]
    async fn apply_budget_throttle_is_noop_when_cache_unavailable() {
        let cache: Arc<dyn Cache> = Arc::new(FailingCounterCache);
        let hook = RateLimitHook::new(cache, Some(10), None).with_budget_throttle_pct(Some(0.5));
        // max_budget set and scope would be "over" if the read worked, but
        // the outage makes the reading 0.0 → not throttled.
        let (limit, throttled) = hook
            .apply_budget_throttle(&spend_key("abc"), Some(100.0), 10)
            .await;
        assert!(
            !throttled,
            "throttle must not engage when the cache is unavailable (best-effort)"
        );
        assert_eq!(
            limit, 10,
            "full limit must apply when the throttle cannot read the spend counter"
        );
    }
}
