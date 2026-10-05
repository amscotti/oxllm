//! Budget threshold alerting hook.
//!
//! A [`Hook`] that fires webhook notifications when a key/team/user spend
//! crosses a configurable percentage of its `max_budget`. Unlike
//! [`BudgetHook`](crate::hooks::BudgetHook) (which hard-rejects at the limit),
//! this hook is "soft": it only emits out-of-band alerts so operators get
//! warning *before* requests start failing.
//!
//! Alerts are deduplicated per `{scope, threshold}` pair via [`AlertState`]:
//! once the 80% threshold fires for a key, it won't re-fire until the budget
//! window resets (detected when spend drops back below the threshold). The
//! [`BudgetResetScheduler`](crate::scheduler::BudgetResetScheduler) clears the
//! state for a scope when its spend counter is zeroed.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::{Cache, CallContext, Hook, ResponseSummary};

use crate::hooks::budget::most_restrictive;
use crate::hooks::callback::CallbackConfig;
use crate::hooks::spend_keys::{
    customer_spend_key, model_spend_key, org_spend_key, project_spend_key, spend_key,
    tag_spend_key, team_spend_key, user_spend_key,
};

fn now_iso() -> String {
    jiff::Timestamp::now().to_string()
}

/// Render a URL safe for structured logging by stripping the path, query, and
/// fragment. Operator webhook URLs (notably Slack incoming webhooks) embed a
/// secret token in the URL path; emitting the full URL on a POST failure or
/// SSRF-validation rejection would leak that token into logs. Returns the
/// scheme + host followed by `/***`, or `***` when the URL cannot be parsed.
pub(crate) fn redact_url(url: &str) -> String {
    match reqwest::Url::parse(url) {
        Ok(parsed) => match parsed.host_str() {
            Some(host) => format!("{}://{host}/***", parsed.scheme()),
            None => format!("{}://***", parsed.scheme()),
        },
        Err(_) => "***".to_string(),
    }
}

/// Sentinel threshold value used as the dedup key for soft-budget alerts.
/// The soft-budget check compares spend against an absolute USD ceiling
/// (not a ratio), so it carries no meaningful fraction; this sentinel
/// (negative, never a valid ratio) keeps it distinct from the configured
/// ratio thresholds in `AlertState::fired`. Cleared alongside ratio
/// markers on budget reset.
pub const SOFT_BUDGET_SENTINEL: f64 = -1.0;

/// Tracks which budget thresholds have already been alerted for each scope.
///
/// Alert keys are formatted as `{scope}:{threshold}` (e.g.
/// `spend:abc123:0.8`). Entries are cleared when the budget window resets —
/// either by spend dropping back below the lowest threshold (detected in
/// [`AlertingHook::post_success`]) or by the budget reset scheduler clearing
/// the scope directly.
pub struct AlertState {
    /// In-memory dedup set for single-replica mode. Bounded by
    /// `(distinct scopes) × (configured thresholds)`; entries are dropped by
    /// [`Self::clear_scope`] when a scope's spend resets to ~0. Deployments
    /// that mint many short-lived keys whose spend never returns to ~0 can
    /// grow this set over the process lifetime (a slow leak, not a DoS);
    /// Redis mode is unaffected because fired markers TTL out with the budget
    /// window there.
    fired: std::sync::Mutex<HashSet<String>>,
    cache: Option<Arc<dyn Cache>>,
    distributed: bool,
    thresholds: Vec<f64>,
    budget_window: Duration,
    /// Hook-type discriminator embedded in every dedup key. Without it, two
    /// cache-backed hooks (e.g. Slack + PagerDuty) sharing a Redis backend
    /// would collide on `alertfired:{scope}:{threshold}` and only the first
    /// hook in the chain would fire — `mark_fired_if_new` claims the key and
    /// every later hook sees it already set. Each builder passes a distinct
    /// name (`"webhook"`, `"slack"`, `"pagerduty"`, `"email"`) so the keys
    /// namespace per channel. In-memory mode is unaffected (separate
    /// `HashSet`s per hook) but the discriminator is harmless there.
    name: &'static str,
}

impl AlertState {
    pub fn new() -> Self {
        Self {
            fired: std::sync::Mutex::new(HashSet::new()),
            cache: None,
            distributed: false,
            thresholds: Vec::new(),
            budget_window: Duration::from_secs(86400),
            name: "",
        }
    }

    pub fn with_cache(mut self, cache: Option<Arc<dyn Cache>>) -> Self {
        self.distributed = cache.is_some();
        self.cache = cache;
        self
    }

    pub fn with_thresholds(mut self, thresholds: Vec<f64>) -> Self {
        self.thresholds = thresholds;
        self
    }

    pub fn with_budget_window(mut self, window: Duration) -> Self {
        self.budget_window = window;
        self
    }

    /// Builder: set the hook-type discriminator embedded in every dedup key
    /// (e.g. `"slack"`, `"pagerduty"`). Required for distributed (Redis) mode
    /// so multiple alerting hooks sharing one backend do not suppress each
    /// other's first fire. See `Self::key`.
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    pub fn distributed(&self) -> bool {
        self.distributed
    }

    fn key(&self, scope: &str, threshold: f64) -> String {
        // The hook-type discriminator namespaces the key per channel so two
        // cache-backed hooks with the same {scope, threshold} (e.g. Slack +
        // PagerDuty both watching key:abc at 0.8) do not collide.
        //
        // The soft-budget marker uses a string-distinct key segment ("soft")
        // so it can never collide with a numeric ratio threshold formatted via
        // `f64` Display (which would otherwise also render as "-1"). See
        // `SOFT_BUDGET_SENTINEL`.
        if threshold == SOFT_BUDGET_SENTINEL {
            format!("alertfired:{}:{scope}:soft", self.name)
        } else {
            format!("alertfired:{}:{scope}:{threshold}", self.name)
        }
    }

    pub async fn was_fired(&self, scope: &str, threshold: f64) -> bool {
        let key = self.key(scope, threshold);
        if let Some(cache) = &self.cache {
            return cache.get(&key).await.is_some();
        }
        self.fired
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&key)
    }

    pub async fn mark_fired_if_new(&self, scope: &str, threshold: f64) -> bool {
        let key = self.key(scope, threshold);
        if let Some(cache) = &self.cache {
            // Fast path: a threshold that already fired in this budget
            // window is the overwhelmingly-common case in steady state
            // (every post-success eval re-asks the same already-fired
            // marker). A cheap read short-circuits before the write, so a
            // busy single-replica Redis-backed deployment no longer pays an
            // `INCRBYFLOAT` per threshold per scope per request whose
            // result is always "lose". The first-crossing guarantee is
            // preserved because `incr` below remains the atomic claim of
            // record: at worst, two truly-concurrent first-crossers both
            // observe a `get` miss and one wins the `incr` race, which is
            // exactly the duplicate-fire bound the atomic path defends.
            if cache.get(&key).await.is_some() {
                return false;
            }
            match cache.incr(&key, 1.0, Some(self.budget_window)).await {
                Ok(v) => (v - 1.0).abs() < 0.5,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        key = %key,
                        "alert dedup claim failed; skipping fire to avoid duplicates"
                    );
                    false
                }
            }
        } else {
            self.fired
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key)
        }
    }

    pub async fn mark_fired(&self, scope: &str, threshold: f64) {
        let key = self.key(scope, threshold);
        if let Some(cache) = &self.cache {
            cache
                .set(&key, "1".to_string(), Some(self.budget_window))
                .await;
            return;
        }
        self.fired
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key);
    }

    /// Clear every fired marker for `scope`.
    ///
    /// **Distributed (cache-backed) branch:** deletes only the keys for
    /// thresholds in `self.thresholds` plus the soft-budget sentinel. A
    /// marker for a threshold that fired under a previous config (e.g.
    /// `0.85` after a hot-reload narrowed the configured set to `[0.8]`)
    /// is left behind in Redis until its own TTL expires. The in-memory
    /// branch below uses prefix matching and cleans up every key for the
    /// scope. The asymmetry is academic in practice because hot-reload
    /// constructs a fresh `AlertState` per hook and the orphaned Redis
    /// keys TTL out with the budget window, but it is real and documented
    /// here for future maintainers. A `SCAN alertfired:{scope}:*` would
    /// close the gap but at non-trivial cost on every reset; not worth it
    /// for a marker whose TTL is already bounded.
    pub async fn clear_scope(&self, scope: &str) {
        if let Some(cache) = &self.cache {
            for threshold in &self.thresholds {
                let key = self.key(scope, *threshold);
                cache.delete(&key).await;
            }
            // Also clear the soft-budget sentinel marker (not in
            // self.thresholds) so a budget reset re-arms the soft alert.
            cache.delete(&self.key(scope, SOFT_BUDGET_SENTINEL)).await;
            return;
        }
        let prefix = format!("alertfired:{}:{scope}:", self.name);
        let mut set = self.fired.lock().unwrap_or_else(|e| e.into_inner());
        set.retain(|k| !k.starts_with(&prefix));
    }

    /// Evaluate one spend scope against a threshold set, returning every
    /// threshold that newly crossed.
    ///
    /// Reads the current spend from `cache.counter_value(key)` once, performs
    /// budget-reset detection (clearing this scope's fired markers when spend
    /// drops back to ~zero), and returns every threshold in `thresholds` that
    /// is newly crossed — i.e. `current / limit >= threshold` and the
    /// threshold was not already marked fired for `scope`.
    ///
    /// Each returned [`ThresholdHit`] carries the single spend snapshot used
    /// both for the comparison and for downstream payload building, so callers
    /// never need to re-read the counter (which would be racy on Redis).
    ///
    /// This is the shared threshold-evaluation core used by every alerting
    /// hook (generic webhook, Slack, PagerDuty, email). Each hook maps the
    /// returned hits into its own payload format.
    pub async fn eval_thresholds(
        &self,
        cache: &Arc<dyn Cache>,
        scope: &str,
        key: &str,
        limit: f64,
        thresholds: &[f64],
    ) -> Vec<ThresholdHit> {
        // Delegate with no soft budget — preserves the original contract for
        // every existing caller (slack, pagerduty, email hooks).
        self.eval_thresholds_with_soft(cache, scope, key, limit, thresholds, None)
            .await
    }

    /// Extended threshold evaluation that also checks an absolute
    /// `soft_budget` ceiling (H10). When `soft_budget` is `Some(positive)`
    /// and `current >= soft_budget`, a [`ThresholdHit`] with
    /// [`SOFT_BUDGET_SENTINEL`] as its threshold is appended; callers map
    /// that sentinel to a `"soft_budget_exceeded"` event. The soft-budget
    /// alert uses the same per-scope dedup as ratio thresholds
    /// ([`Self::mark_fired_if_new`]) and is cleared alongside them on
    /// budget reset. Non-blocking: this method only records hits; it never
    /// rejects.
    pub async fn eval_thresholds_with_soft(
        &self,
        cache: &Arc<dyn Cache>,
        scope: &str,
        key: &str,
        limit: f64,
        thresholds: &[f64],
        soft_budget: Option<f64>,
    ) -> Vec<ThresholdHit> {
        let mut hits = Vec::new();
        let has_ratio = limit > 0.0 && !thresholds.is_empty();
        let has_soft = soft_budget.map(|s| s > 0.0).unwrap_or(false);
        if !has_ratio && !has_soft {
            return hits;
        }
        // Distinguish a genuine counter absence from a backend error (e.g. a
        // Redis blip). Treating an outage as `current = 0.0` would satisfy the
        // reset guard below, wipe this scope's fired markers, and cause every
        // previously-fired threshold to re-fire when the backend recovers with
        // the real (high) spend. On a read error, skip evaluation entirely:
        // do not clear, do not fire.
        let current = match cache.try_counter_value(key).await {
            Ok(opt) => opt.unwrap_or(0.0),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    scope = %scope,
                    key = %key,
                    "spend counter read failed; skipping alert evaluation (no clear, no fire)"
                );
                return hits;
            }
        };

        // Budget reset detection: only clear the scope's fired alerts when
        // spend is effectively zeroed (an exact budget-window reset zeroes the
        // counter). The previous heuristic also cleared when spend dropped
        // below 10% of the limit, which false-triggered on large spend
        // corrections (refunds, manual adjustments) — wiping every fired
        // marker mid-window and causing duplicate alerts once spend rose
        // again. Budget resets zero the counter exactly, so the absolute
        // floor (`< 0.01`) is sufficient and avoids the false-positive risk.
        if current < 0.01 {
            self.clear_scope(scope).await;
        }
        if current <= 0.0 {
            return hits;
        }

        // Ratio thresholds.
        if has_ratio {
            let ratio = current / limit;
            for &threshold in thresholds {
                if ratio >= threshold && self.mark_fired_if_new(scope, threshold).await {
                    hits.push(ThresholdHit {
                        threshold,
                        current_spend: current,
                    });
                }
            }
        }

        // Soft-budget check (absolute USD): fire a dedicated, deduped alert
        // when spend crosses the soft ceiling. Independent of the ratio
        // thresholds — a scope with no ratio thresholds but a configured
        // soft_budget still fires.
        if has_soft
            && current >= soft_budget.unwrap_or(f64::INFINITY)
            && self.mark_fired_if_new(scope, SOFT_BUDGET_SENTINEL).await
        {
            hits.push(ThresholdHit {
                threshold: SOFT_BUDGET_SENTINEL,
                current_spend: current,
            });
        }

        hits
    }
}

/// A budget threshold that newly crossed during an [`AlertState::eval_thresholds`]
/// call, paired with the spend snapshot at the moment of evaluation.
///
/// `current_spend` / `limit` (known to the caller) gives the ratio that
/// triggered `threshold`. Each alerting hook formats these fields into its own
/// notification payload.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdHit {
    /// The threshold fraction that was crossed (e.g. `0.8` for 80%).
    pub threshold: f64,
    /// The spend counter value at the moment of evaluation.
    pub current_spend: f64,
}

impl Default for AlertState {
    fn default() -> Self {
        Self::new()
    }
}

/// Payload POSTed to the configured webhook endpoint when a budget threshold
/// is crossed.
#[derive(Debug, serde::Serialize)]
pub struct BudgetAlert {
    pub event: String,
    pub scope: String,
    pub scope_id: String,
    /// The threshold fraction that was crossed (e.g. `0.8` for 80%). `None`
    /// for `"soft_budget_exceeded"` events, where a ratio is meaningless (the
    /// soft ceiling is an absolute USD amount, not a fraction of the limit).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold_pct: Option<f64>,
    pub current_spend: f64,
    pub max_budget: f64,
    pub timestamp: String,
    pub request_id: Option<String>,
}

/// A `Hook` that fires webhook alerts when budget thresholds are crossed.
pub struct AlertingHook {
    cache: Arc<dyn Cache>,
    config: CallbackConfig,
    thresholds: Vec<f64>,
    state: Arc<AlertState>,
    client: reqwest::Client,
    global_max_budget: Option<f64>,
    /// Per-model global spend caps, mirrored from `oxllm_settings.model_max_budget`.
    model_max_budget: Arc<HashMap<String, f64>>,
    /// Optional soft-budget percentage of each scope's `max_budget`. When
    /// set (e.g. `0.8`), a scope's soft ceiling is `max_budget * pct`;
    /// crossing it fires a non-blocking `"soft_budget_exceeded"` alert
    /// (H10), deduped per scope via [`AlertState`]. `None` disables
    /// soft-budget alerts.
    soft_budget_pct: Option<f64>,
}

impl std::fmt::Debug for AlertingHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlertingHook")
            .field("url", &redact_url(&self.config.url))
            .field("thresholds", &self.thresholds)
            .finish()
    }
}

impl AlertingHook {
    /// Build an alerting hook from a webhook config, threshold list, and
    /// shared alert state. The HTTP client is configured with a 5s timeout
    /// and redirects disabled (matching the generic callback).
    pub fn new(
        cache: Arc<dyn Cache>,
        config: CallbackConfig,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
    ) -> anyhow::Result<Self> {
        Self::with_global_budget(cache, config, thresholds, state, None)
    }

    /// Like [`Self::new`] but also consults the global `max_budget` when
    /// computing the effective per-key limit (mirroring `BudgetHook`).
    /// Uses the SSRF-aware HTTP client with private-range blocking enabled
    /// (i.e. `allow_private_api_base = false`); production callers that need
    /// to honour an operator opt-in should use [`Self::with_model_budget`].
    pub fn with_global_budget(
        cache: Arc<dyn Cache>,
        config: CallbackConfig,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
        global_max_budget: Option<f64>,
    ) -> anyhow::Result<Self> {
        Self::with_model_budget(
            cache,
            config,
            thresholds,
            state,
            global_max_budget,
            HashMap::new(),
            false,
        )
    }

    /// Like [`Self::with_global_budget`] but also carries the per-model
    /// global spend caps so model-budget thresholds can fire alerts.
    ///
    /// The HTTP client carries the connect-time
    /// [`crate::security::SsrfResolver`] unless `allow_private` is set, so a
    /// DNS-rebinding attack on the operator-configured webhook after config
    /// load cannot steer the POST (which carries `request_id`, spend, and
    /// scope ids) to a private address. This mirrors the runtime-SSRF
    /// defence already present on [`SlackAlertingHook`](crate::hooks::SlackAlertingHook).
    pub fn with_model_budget(
        cache: Arc<dyn Cache>,
        config: CallbackConfig,
        thresholds: Vec<f64>,
        state: Arc<AlertState>,
        global_max_budget: Option<f64>,
        model_max_budget: HashMap<String, f64>,
        allow_private: bool,
    ) -> anyhow::Result<Self> {
        let client =
            crate::security::build_ssrf_http_client(allow_private, Duration::from_secs(5))?;
        Ok(Self {
            cache,
            config,
            thresholds,
            state,
            client,
            global_max_budget,
            model_max_budget: Arc::new(model_max_budget),
            soft_budget_pct: None,
        })
    }

    /// Builder: set the soft-budget percentage (e.g. `Some(0.8)` → fire a
    /// non-blocking `"soft_budget_exceeded"` alert when a scope's spend
    /// crosses 80% of its `max_budget`). `None` (the default) disables
    /// soft-budget alerts.
    pub fn with_soft_budget_pct(mut self, pct: Option<f64>) -> Self {
        self.soft_budget_pct = pct;
        self
    }

    /// Evaluate a single budget scope, firing any newly-crossed thresholds.
    async fn eval_scope(
        &self,
        scope: &str,
        spend_cache_key: &str,
        scope_id: &str,
        limit: Option<f64>,
        ctx: &CallContext,
    ) {
        let Some(limit) = limit else { return };
        // H10: derive the absolute soft-budget ceiling from the scope's
        // max_budget and the configured percentage. Crossing it fires a
        // "soft_budget_exceeded" alert alongside (or independent of) the
        // ratio thresholds.
        let soft_budget = self
            .soft_budget_pct
            .filter(|pct| pct > &0.0)
            .map(|pct| limit * pct);
        for hit in self
            .state
            .eval_thresholds_with_soft(
                &self.cache,
                scope,
                spend_cache_key,
                limit,
                &self.thresholds,
                soft_budget,
            )
            .await
        {
            // Sentinel threshold → soft-budget event; otherwise the standard
            // ratio-threshold event.
            let event = if hit.threshold == SOFT_BUDGET_SENTINEL {
                "soft_budget_exceeded"
            } else {
                "budget_threshold_crossed"
            };
            // `threshold_pct` is a 0..=1 ratio and is meaningless for the
            // soft-budget event (whose ceiling is an absolute USD amount).
            // Omit it rather than ship the sentinel `-1.0`.
            let threshold_pct = if hit.threshold == SOFT_BUDGET_SENTINEL {
                None
            } else {
                Some(hit.threshold)
            };
            let alert = BudgetAlert {
                event: event.to_string(),
                scope: scope.to_string(),
                scope_id: scope_id.to_string(),
                threshold_pct,
                current_spend: hit.current_spend,
                max_budget: limit,
                timestamp: now_iso(),
                request_id: Some(ctx.request_id.clone()),
            };
            self.dispatch(alert);
        }
    }

    fn dispatch(&self, alert: BudgetAlert) {
        let config = self.config.clone();
        let client = self.client.clone();
        tokio::spawn(async move {
            if let Err(e) = send_alert(&client, &config, &alert).await {
                tracing::warn!(
                    error = %e,
                    url = %redact_url(&config.url),
                    "budget alert webhook POST failed"
                );
            }
        });
    }
}

async fn send_alert(
    client: &reqwest::Client,
    config: &CallbackConfig,
    alert: &BudgetAlert,
) -> Result<(), reqwest::Error> {
    let mut req = client.post(&config.url).json(alert);
    for (k, v) in &config.headers {
        if let Ok(value) = reqwest::header::HeaderValue::from_str(v) {
            if let Ok(name) = reqwest::header::HeaderName::from_bytes(k.as_bytes()) {
                req = req.header(name, value);
            }
        }
    }
    req.send().await?.error_for_status()?;
    Ok(())
}

#[async_trait]
impl Hook for AlertingHook {
    async fn post_success(&self, ctx: &CallContext, _resp: &ResponseSummary) {
        if ctx.is_master || ctx.cache_hit {
            return;
        }
        for (scope, spend_cache_key, scope_id, limit) in
            budget_scopes(ctx, self.global_max_budget, &self.model_max_budget)
        {
            self.eval_scope(&scope, &spend_cache_key, &scope_id, Some(limit), ctx)
                .await;
        }
    }
}

/// Convert a spend counter key back into the matching alert scope, so the
/// budget reset scheduler can clear stale alerts when a counter is zeroed.
///
/// - `spend:abc`          → `key:abc`
/// - `spend:team:t1`      → `team:t1`
/// - `spend:user:u1`      → `user:u1`
/// - `spend:org:o1`       → `org:o1`
/// - `spend:project:p1`   → `project:p1`
/// - `spend:model:gpt-4o` → `model:gpt-4o`
///
/// Returns `None` for non-spend keys, and for `spend:` sub-scopes that have
/// no alerting counterpart — notably router-scoped counters (`spend:dep:*`,
/// `spend:provider:*`) whose TTLs are owned by the router and are never
/// written under an alert scope.
pub fn scope_for_spend_key(spend_key: &str) -> Option<String> {
    let rest = spend_key.strip_prefix("spend:")?;
    // Router-scoped spend counters carry their own TTLs and are never
    // associated with an alert scope; returning None lets the scheduler's
    // `if let Some(scope)` guard skip them (avoiding a stream of no-op
    // Redis deletes for keys that cannot exist).
    if rest.starts_with("dep:") || rest.starts_with("provider:") {
        return None;
    }
    if let Some(team) = rest.strip_prefix("team:") {
        Some(format!("team:{team}"))
    } else if let Some(user) = rest.strip_prefix("user:") {
        Some(format!("user:{user}"))
    } else if let Some(org) = rest.strip_prefix("org:") {
        Some(format!("org:{org}"))
    } else if let Some(project) = rest.strip_prefix("project:") {
        Some(format!("project:{project}"))
    } else if let Some(customer) = rest.strip_prefix("customer:") {
        Some(format!("customer:{customer}"))
    } else if let Some(tag) = rest.strip_prefix("tag:") {
        Some(format!("tag:{tag}"))
    } else if let Some(model) = rest.strip_prefix("model:") {
        Some(format!("model:{model}"))
    } else {
        Some(format!("key:{rest}"))
    }
}

/// Yield every `(scope, spend_cache_key, scope_id, limit)` quad for which
/// `ctx` carries a budget ceiling, in the canonical scope order
/// (key → team → org → project → user → model). The global `max_budget` is
/// honoured via [`most_restrictive`] on the key scope (mirroring
/// [`BudgetHook`](crate::hooks::BudgetHook)); the per-model cap is looked
/// up in `model_max_budget`.
///
/// This is the shared six-scope iteration used by every alerting hook's
/// `post_success` (generic webhook, Slack, PagerDuty, email), so a new
/// scope or a new eval-style flag lands in one place rather than four.
/// Each hook maps every yielded quad through its own `eval_scope`.
pub(crate) fn budget_scopes(
    ctx: &CallContext,
    global_max_budget: Option<f64>,
    model_max_budget: &HashMap<String, f64>,
) -> Vec<(String, String, String, f64)> {
    let mut out = Vec::new();
    let key_limit = most_restrictive(ctx.max_budget, global_max_budget);
    if let Some(limit) = key_limit {
        let scope = format!("key:{}", ctx.api_key_hash);
        let scope_id = ctx
            .key_id
            .clone()
            .unwrap_or_else(|| ctx.api_key_hash.clone());
        out.push((scope, spend_key(&ctx.api_key_hash), scope_id, limit));
    }
    if let Some(limit) = ctx.team_max_budget {
        if let Some(team_id) = ctx.team_id.as_deref() {
            out.push((
                format!("team:{team_id}"),
                team_spend_key(team_id),
                team_id.to_string(),
                limit,
            ));
        }
    }
    if let Some(limit) = ctx.org_max_budget {
        if let Some(org_id) = ctx.org_id.as_deref() {
            out.push((
                format!("org:{org_id}"),
                org_spend_key(org_id),
                org_id.to_string(),
                limit,
            ));
        }
    }
    if let Some(limit) = ctx.project_max_budget {
        if let Some(project_id) = ctx.project_id.as_deref() {
            out.push((
                format!("project:{project_id}"),
                project_spend_key(project_id),
                project_id.to_string(),
                limit,
            ));
        }
    }
    if let Some(limit) = ctx.user_max_budget {
        if let Some(user_id) = ctx.user_id.as_deref() {
            out.push((
                format!("user:{user_id}"),
                user_spend_key(user_id),
                user_id.to_string(),
                limit,
            ));
        }
    }
    if let Some(limit) = ctx.customer_max_budget {
        if let Some(end_user_id) = ctx.end_user_id.as_deref() {
            out.push((
                format!("customer:{end_user_id}"),
                customer_spend_key(end_user_id),
                end_user_id.to_string(),
                limit,
            ));
        }
    }
    if !ctx.tag_budgets.is_empty() {
        for (tag, limit) in &ctx.tag_budgets {
            out.push((
                format!("tag:{tag}"),
                tag_spend_key(tag),
                tag.clone(),
                *limit,
            ));
        }
    } else if let Some(limit) = ctx.tag_max_budget {
        if let Some(tag) = ctx.request_tag.as_deref() {
            out.push((
                format!("tag:{tag}"),
                tag_spend_key(tag),
                tag.to_string(),
                limit,
            ));
        }
    }
    if let Some(limit) = model_max_budget.get(&ctx.model).copied() {
        out.push((
            format!("model:{}", ctx.model),
            model_spend_key(&ctx.model),
            ctx.model.clone(),
            limit,
        ));
    }
    out
}

/// Build the shared [`AlertState`] when alerting is configured, else `None`.
/// The same `Arc` is handed to both the [`AlertingHook`] (added by the
/// state hook-chain builder) and the budget reset scheduler.
pub(crate) fn build_alert_state(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
) -> Option<Arc<AlertState>> {
    let alerting = config.oxllm_settings.alerting.as_ref()?;
    if !alerting.iter().any(|s| s == "webhook") {
        return None;
    }
    let webhook = config
        .oxllm_settings
        .alerting_config
        .as_ref()?
        .webhook
        .as_ref()?;

    let thresholds = webhook.thresholds.clone();
    let budget_window = crate::scheduler::parse_budget_duration(
        config
            .oxllm_settings
            .budget_duration
            .as_deref()
            .unwrap_or("30d"),
    );

    let is_redis = crate::state::redis_configured(config);
    let alert_cache = if is_redis { Some(cache) } else { None };

    let state = AlertState::new()
        .with_name("webhook")
        .with_thresholds(thresholds)
        .with_budget_window(budget_window)
        .with_cache(alert_cache);

    Some(Arc::new(state))
}

/// Construct an [`AlertingHook`] from `oxllm_settings.alerting_config.webhook`
/// when `alerting` lists `"webhook"`. Returns `None` when alerting is disabled
/// or misconfigured (misconfiguration is logged at warn level).
pub(crate) fn build_alerting_hook(
    config: &crate::config::ProxyConfig,
    cache: Arc<dyn Cache>,
    state: Arc<AlertState>,
) -> Option<AlertingHook> {
    let webhook = config
        .oxllm_settings
        .alerting_config
        .as_ref()?
        .webhook
        .as_ref()?;
    if webhook.url.is_empty() {
        tracing::warn!("alerting webhook URL is empty; skipping alerting hook");
        return None;
    }
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if let Err(e) = crate::config::validate_api_base(&webhook.url, allow_private) {
        tracing::warn!(
            error = %e,
            "alerting webhook URL '{}' failed SSRF validation; skipping",
            redact_url(&webhook.url)
        );
        return None;
    }
    let callback_cfg = CallbackConfig {
        url: webhook.url.clone(),
        headers: webhook.headers.clone().unwrap_or_default(),
    };
    let global_max_budget = config.oxllm_settings.max_budget;
    let model_max_budget = config
        .oxllm_settings
        .model_max_budget
        .clone()
        .unwrap_or_default();
    match AlertingHook::with_model_budget(
        cache,
        callback_cfg,
        webhook.thresholds.clone(),
        state,
        global_max_budget,
        model_max_budget,
        allow_private,
    ) {
        Ok(hook) => Some(hook.with_soft_budget_pct(webhook.soft_budget_pct)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build alerting hook; skipping");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use crate::hooks::callback::CallbackConfig;
    use oxllm_core::UsageSummary;
    use std::collections::HashMap;

    fn make_ctx() -> CallContext {
        CallContext {
            key_id: Some("k-1".to_string()),
            ..CallContext::for_test("gpt-4o")
        }
    }

    fn empty_config() -> CallbackConfig {
        CallbackConfig {
            url: "http://example.invalid/hook".to_string(),
            headers: HashMap::new(),
        }
    }

    fn make_response() -> ResponseSummary {
        ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: Some(0.0025),
            usage: Some(UsageSummary {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
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
    async fn multi_tag_alerts_use_each_tags_own_limit() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("hook");
        cache.incr("spend:tag:app", 80.0, None).await.unwrap();
        cache.incr("spend:tag:prod", 80.0, None).await.unwrap();
        let mut ctx = make_ctx();
        ctx.request_tag = Some("app".into());
        ctx.tag_max_budget = Some(100.0);
        ctx.tag_budgets = vec![("app".into(), 200.0), ("prod".into(), 100.0)];
        hook.post_success(&ctx, &make_response()).await;
        assert!(!state.was_fired("tag:app", 0.8).await, "app is only at 40%");
        assert!(
            state.was_fired("tag:prod", 0.8).await,
            "secondary tag reached 80%"
        );
    }

    #[tokio::test]
    async fn alert_state_fire_once_does_not_refire() {
        let state = AlertState::new();
        assert!(!state.was_fired("key:abc", 0.8).await);
        state.mark_fired("key:abc", 0.8).await;
        assert!(state.was_fired("key:abc", 0.8).await);
        state.mark_fired("key:abc", 0.8).await;
        assert!(state.was_fired("key:abc", 0.8).await);
    }

    /// MEDIUM: Slack (and generic webhook) URLs embed the secret token in the
    /// path. `redact_url` must strip the path/query/fragment so failure and
    /// SSRF-validation logs never contain the token.
    #[test]
    fn redact_url_strips_secret_bearing_path() {
        let token = "SUPERSECRET123";
        let slack_url = format!("https://hooks.slack.com/services/T/B/{token}");
        let redacted = redact_url(&slack_url);
        assert!(
            !redacted.contains(token),
            "redacted URL must not contain the token, got: {redacted}"
        );
        assert_eq!(redacted, "https://hooks.slack.com/***");

        // Generic webhook with credentials in the path/query also redacted.
        let op_url = "https://ops.example.invalid/hook?secret=abc";
        let redacted = redact_url(op_url);
        assert!(!redacted.contains("secret=abc"));
        assert_eq!(redacted, "https://ops.example.invalid/***");

        // Unparseable input is masked entirely.
        assert_eq!(redact_url("not a url"), "***");
    }

    /// The in-memory dedup set must not grow without bound when scopes fire
    /// and then reset: `clear_scope` drops every marker for a reset scope, so
    /// after N distinct scopes fire and reset the set returns to its baseline.
    #[tokio::test]
    async fn in_memory_fired_set_returns_to_baseline_after_resets() {
        let state = AlertState::new();
        let baseline = state.fired.lock().unwrap().len();
        for i in 0..50 {
            let scope = format!("key:k{i}");
            state.mark_fired(&scope, 0.8).await;
            state.mark_fired(&scope, 1.0).await;
        }
        assert_eq!(
            state.fired.lock().unwrap().len(),
            baseline + 100,
            "50 scopes x 2 thresholds must be tracked"
        );
        for i in 0..50 {
            state.clear_scope(&format!("key:k{i}")).await;
        }
        assert_eq!(
            state.fired.lock().unwrap().len(),
            baseline,
            "in-memory set must return to baseline after every scope resets"
        );
    }

    #[tokio::test]
    async fn alert_state_scopes_are_independent() {
        let state = AlertState::new();
        state.mark_fired("key:abc", 0.8).await;
        assert!(state.was_fired("key:abc", 0.8).await);
        assert!(!state.was_fired("key:def", 0.8).await);
        assert!(!state.was_fired("key:abc", 0.9).await);
    }

    #[tokio::test]
    async fn alert_state_clear_scope_removes_only_that_scope() {
        let state = AlertState::new();
        state.mark_fired("key:abc", 0.8).await;
        state.mark_fired("key:abc", 0.9).await;
        state.mark_fired("key:def", 0.8).await;
        state.clear_scope("key:abc").await;
        assert!(!state.was_fired("key:abc", 0.8).await);
        assert!(!state.was_fired("key:abc", 0.9).await);
        assert!(state.was_fired("key:def", 0.8).await);
    }

    #[tokio::test]
    async fn threshold_80pct_of_100_budget_fires_at_80_spend() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = AlertingHook::new(
            cache.clone(),
            empty_config(),
            vec![0.5, 0.8, 0.9, 1.0],
            Arc::new(AlertState::new()),
        )
        .expect("build hook");

        let _ = cache.incr("spend:abc", 79.0, None).await;
        let ctx = make_ctx();
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(!hook.state.was_fired("key:abc", 0.8).await);

        let _ = cache.incr("spend:abc", 1.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(
            hook.state.was_fired("key:abc", 0.8).await,
            "spend == 80 should fire the 0.8 threshold"
        );
    }

    #[tokio::test]
    async fn multiple_thresholds_fire_in_order() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(
            cache.clone(),
            empty_config(),
            vec![0.5, 0.8, 1.0],
            state.clone(),
        )
        .expect("build hook");

        let ctx = make_ctx();

        let _ = cache.incr("spend:abc", 50.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(state.was_fired("key:abc", 0.5).await);
        assert!(!state.was_fired("key:abc", 0.8).await);
        assert!(!state.was_fired("key:abc", 1.0).await);

        let _ = cache.incr("spend:abc", 30.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(state.was_fired("key:abc", 0.8).await);
        assert!(!state.was_fired("key:abc", 1.0).await);

        let _ = cache.incr("spend:abc", 20.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(state.was_fired("key:abc", 1.0).await);
    }

    #[tokio::test]
    async fn budget_reset_clears_fired_alerts() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8, 1.0], state.clone())
            .expect("build hook");

        let ctx = make_ctx();

        let _ = cache.incr("spend:abc", 90.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(state.was_fired("key:abc", 0.8).await);

        // Simulate a budget window reset: spend drops below the lowest threshold.
        let _ = cache.incr("spend:abc", -90.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(
            !state.was_fired("key:abc", 0.8).await,
            "spend dropping below threshold must clear fired state"
        );

        // Re-crossing should fire again.
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(state.was_fired("key:abc", 0.8).await);
    }

    /// A large spend correction that drops spend well below 10% of the limit
    /// (but NOT to zero) must NOT clear fired markers — the previous 10%
    /// heuristic would wipe them mid-window, causing duplicate alerts once
    /// spend rose again. Only an exact budget-window reset (spend → ~0)
    /// re-arms alerts.
    #[tokio::test]
    async fn spend_correction_below_10pct_does_not_refire() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("build hook");

        let ctx = make_ctx();

        // Spend $85 → 0.8 threshold fires.
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(state.was_fired("key:abc", 0.8).await);

        // A refund/correction drops spend to $8 (below 10% of $100 but NOT
        // zero). Markers must survive.
        let _ = cache.incr("spend:abc", -77.0, None).await;
        hook.eval_scope("key:abc", "spend:abc", "abc", Some(100.0), &ctx)
            .await;
        assert!(
            state.was_fired("key:abc", 0.8).await,
            "a non-zero spend correction must not clear fired markers"
        );

        // Spend rises back to $85 — the alert must NOT re-fire (same window).
        let _ = cache.incr("spend:abc", 77.0, None).await;
        let hits = state
            .eval_thresholds(&cache, "key:abc", "spend:abc", 100.0, &[0.8])
            .await;
        assert!(
            hits.is_empty(),
            "alert must not re-fire after a non-reset spend dip"
        );
    }

    #[tokio::test]
    async fn post_success_skips_cache_hit() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("build hook");

        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 90.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            !state.was_fired("key:abc", 0.8).await,
            "cache hits must not fire alerts"
        );
    }

    #[tokio::test]
    async fn post_success_skips_master_key() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("build hook");

        let mut ctx = make_ctx();
        ctx.is_master = true;
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 90.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            !state.was_fired("key:abc", 0.8).await,
            "master key must not fire alerts"
        );
    }

    #[tokio::test]
    async fn post_success_fires_on_threshold_cross() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("build hook");

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(state.was_fired("key:abc", 0.8).await);
    }

    #[tokio::test]
    async fn post_success_fires_on_model_budget_threshold_cross() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let mut model_budgets = HashMap::new();
        model_budgets.insert("gpt-4o".to_string(), 100.0);
        let hook = AlertingHook::with_model_budget(
            cache.clone(),
            empty_config(),
            vec![0.8],
            state.clone(),
            None,
            model_budgets,
            false,
        )
        .expect("build hook");

        let ctx = make_ctx(); // model is "gpt-4o"
        let _ = cache.incr("spend:model:gpt-4o", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            state.was_fired("model:gpt-4o", 0.8).await,
            "per-model budget threshold must fire alerts with the model scope"
        );
    }

    #[test]
    fn webhook_payload_serializes_to_expected_json() {
        let alert = BudgetAlert {
            event: "budget_threshold_crossed".to_string(),
            scope: "key".to_string(),
            scope_id: "abc".to_string(),
            threshold_pct: Some(0.8),
            current_spend: 80.0,
            max_budget: 100.0,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            request_id: Some("req-1".to_string()),
        };
        let json = serde_json::to_value(&alert).expect("serialize");
        assert_eq!(json["event"], "budget_threshold_crossed");
        assert_eq!(json["scope"], "key");
        assert_eq!(json["scope_id"], "abc");
        assert_eq!(json["threshold_pct"], 0.8);
        assert_eq!(json["current_spend"], 80.0);
        assert_eq!(json["max_budget"], 100.0);
        assert_eq!(json["timestamp"], "2024-01-01T00:00:00Z");
        assert_eq!(json["request_id"], "req-1");
    }

    /// A `"soft_budget_exceeded"` payload must not carry the `-1.0` sentinel
    /// in `threshold_pct`; the field is omitted (the ratio is meaningless for
    /// an absolute-USD soft ceiling).
    #[test]
    fn soft_budget_payload_omits_threshold_pct() {
        let alert = BudgetAlert {
            event: "soft_budget_exceeded".to_string(),
            scope: "key".to_string(),
            scope_id: "abc".to_string(),
            threshold_pct: None,
            current_spend: 85.0,
            max_budget: 100.0,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            request_id: Some("req-1".to_string()),
        };
        let json = serde_json::to_string(&alert).expect("serialize");
        assert!(
            !json.contains("threshold_pct"),
            "soft_budget payload must omit threshold_pct, got: {json}"
        );
        assert!(json.contains("soft_budget_exceeded"));
    }

    #[test]
    fn scope_for_spend_key_maps_each_scope() {
        assert_eq!(scope_for_spend_key("spend:abc").as_deref(), Some("key:abc"));
        assert_eq!(
            scope_for_spend_key("spend:team:t1").as_deref(),
            Some("team:t1")
        );
        assert_eq!(
            scope_for_spend_key("spend:user:u1").as_deref(),
            Some("user:u1")
        );
        assert_eq!(
            scope_for_spend_key("spend:org:o1").as_deref(),
            Some("org:o1")
        );
        assert_eq!(
            scope_for_spend_key("spend:project:p1").as_deref(),
            Some("project:p1")
        );
        assert_eq!(
            scope_for_spend_key("spend:customer:eu-1").as_deref(),
            Some("customer:eu-1")
        );
        assert_eq!(
            scope_for_spend_key("spend:tag:canary").as_deref(),
            Some("tag:canary")
        );
        assert_eq!(
            scope_for_spend_key("spend:model:gpt-4o").as_deref(),
            Some("model:gpt-4o")
        );
        assert_eq!(scope_for_spend_key("rpm:abc"), None);
    }

    /// Router-scoped spend counters map to `None`: they carry their own TTLs
    /// and are never written under an alert scope, so the scheduler must skip
    /// them (avoiding no-op Redis deletes for bogus `key:dep:*` scopes).
    #[test]
    fn scope_for_spend_key_returns_none_for_router_scopes() {
        assert_eq!(scope_for_spend_key("spend:dep:x:y"), None);
        assert_eq!(scope_for_spend_key("spend:dep:gpt-4o:openai/gpt-4o"), None);
        assert_eq!(scope_for_spend_key("spend:provider:openai"), None);
    }

    #[tokio::test]
    async fn post_success_fires_on_org_budget_threshold_cross() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("build hook");

        let mut ctx = make_ctx();
        ctx.org_id = Some("o1".to_string());
        ctx.org_max_budget = Some(100.0);
        let _ = cache.incr("spend:org:o1", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            state.was_fired("org:o1", 0.8).await,
            "org budget threshold must fire alerts with the org scope"
        );
    }

    #[tokio::test]
    async fn post_success_fires_on_project_budget_threshold_cross() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("build hook");

        let mut ctx = make_ctx();
        ctx.project_id = Some("p1".to_string());
        ctx.project_max_budget = Some(100.0);
        let _ = cache.incr("spend:project:p1", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        assert!(
            state.was_fired("project:p1", 0.8).await,
            "project budget threshold must fire alerts with the project scope"
        );
    }

    #[tokio::test]
    async fn webhook_fires_when_threshold_crossed_httpmock() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/alert";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/alert");
            then.status(200);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(
            cache.clone(),
            CallbackConfig {
                url,
                headers: HashMap::new(),
            },
            vec![0.8],
            state.clone(),
        )
        .expect("build hook");

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "webhook should have been called when threshold crossed"
        );
    }

    /// The generic-webhook `AlertingHook` must install the runtime SSRF
    /// DNS resolver on its HTTP client unless `allow_private` is set, so a
    /// DNS-rebinding attack on the operator's webhook domain after config
    /// load cannot steer the alert POST to a private address. The test
    /// points the hook at the httpmock listener via the `localhost`
    /// hostname (which resolves to the loopback IP `127.0.0.1`); the
    /// SSRF resolver must reject that resolution at connect time. (A
    /// literal-IP URL bypasses reqwest's `dns_resolver`, so the hostname
    /// form is required to exercise the runtime path.)
    #[tokio::test]
    async fn webhook_ssrf_resolver_blocks_private_loopback() {
        use httpmock::MockServer;

        let server = MockServer::start();
        // Build the URL with `localhost` so reqwest routes the host through
        // the configured `dns_resolver` rather than short-circuiting the
        // literal IP.
        let url = format!("http://localhost:{}/alert", server.port());

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/alert");
            then.status(200);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::with_model_budget(
            cache.clone(),
            CallbackConfig {
                url,
                headers: HashMap::new(),
            },
            vec![0.8],
            state.clone(),
            None,
            HashMap::new(),
            false,
        )
        .expect("build hook");

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        // Give the spawned POST plenty of time to either land or fail; the
        // SSRF resolver rejects the connection at connect time, so even a
        // slow failure path is much faster than this bound.
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            post_mock.hits(),
            0,
            "SSRF resolver must block the loopback webhook target when allow_private is unset"
        );
    }

    #[tokio::test]
    async fn webhook_does_not_refire_for_same_threshold() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/alert";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/alert");
            then.status(200);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(
            cache.clone(),
            CallbackConfig {
                url,
                headers: HashMap::new(),
            },
            vec![0.8],
            state.clone(),
        )
        .expect("build hook");

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        // Second call: spend went up but we're still in the same threshold
        // band (0.8 already fired). Must NOT re-fire.
        let _ = cache.incr("spend:abc", 5.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            post_mock.hits(),
            1,
            "webhook must fire exactly once per threshold until budget resets"
        );
    }

    #[tokio::test]
    async fn distributed_alert_state_uses_cache() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = AlertState::new()
            .with_thresholds(vec![0.8, 1.0])
            .with_cache(Some(cache.clone()));

        assert!(state.distributed());
        assert!(!state.was_fired("key:abc", 0.8).await);

        state.mark_fired("key:abc", 0.8).await;
        assert!(state.was_fired("key:abc", 0.8).await);
        let fired_key = state.key("key:abc", 0.8);
        assert!(
            cache.get(&fired_key).await.is_some(),
            "fired marker must exist in cache"
        );

        assert!(
            !state.was_fired("key:abc", 1.0).await,
            "different threshold must not be fired"
        );
    }

    /// HIGH: two cache-backed alerting hooks sharing one Redis-shaped backend
    /// must BOTH fire on the first threshold crossing for the same
    /// `{scope, threshold}` pair. Without a hook-type discriminator in the
    /// dedup key, the first hook's `mark_fired_if_new` claim would suppress
    /// every later hook (e.g. Slack claims the key, PagerDuty sees it set and
    /// silently drops its alert). This exercises the distributed (cache) path
    /// — the in-memory path was always correct because each hook owns a
    /// separate `HashSet`.
    #[tokio::test]
    async fn distributed_two_named_hooks_both_fire_on_shared_cache() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let slack = AlertState::new()
            .with_name("slack")
            .with_thresholds(vec![0.8])
            .with_cache(Some(cache.clone()));
        let pagerduty = AlertState::new()
            .with_name("pagerduty")
            .with_thresholds(vec![0.8])
            .with_cache(Some(cache.clone()));

        // Slack fires first (chain order), claiming its namespaced key.
        assert!(
            slack.mark_fired_if_new("key:abc", 0.8).await,
            "slack hook must fire on first crossing"
        );
        // PagerDuty must still fire: its dedup key is namespaced separately,
        // so the shared cache must not report it already fired.
        assert!(
            pagerduty.mark_fired_if_new("key:abc", 0.8).await,
            "pagerduty hook must fire independently when sharing a cache backend"
        );

        // Both markers now live in the shared cache under distinct keys.
        assert!(cache.get(&slack.key("key:abc", 0.8)).await.is_some());
        assert!(cache.get(&pagerduty.key("key:abc", 0.8)).await.is_some());
        // Re-asking either must be deduped (idempotent claim).
        assert!(!slack.mark_fired_if_new("key:abc", 0.8).await);
        assert!(!pagerduty.mark_fired_if_new("key:abc", 0.8).await);
    }

    #[tokio::test]
    async fn distributed_alert_state_mark_fired_if_new() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = AlertState::new()
            .with_thresholds(vec![0.8])
            .with_cache(Some(cache));

        assert!(
            state.mark_fired_if_new("key:abc", 0.8).await,
            "first mark must return true"
        );
        assert!(
            !state.mark_fired_if_new("key:abc", 0.8).await,
            "second mark must return false (already fired)"
        );
    }

    /// Concurrent `mark_fired_if_new` callers against a cache-shaped backend
    /// must produce exactly one winner. This is a regression test for the
    /// TOCTOU in the old get-then-set path: this stub cache reports every
    /// `get` as a miss (so the old code would have let every concurrent caller
    /// "win") while `incr` is truly atomic — mirroring the Redis
    /// `INCRBYFLOAT` / in-memory write-lock contract.
    #[tokio::test]
    async fn distributed_mark_fired_if_new_atomic_under_concurrency() {
        use std::sync::atomic::{AtomicU64, Ordering};

        struct ClaimCache {
            val: AtomicU64,
        }

        #[async_trait::async_trait]
        impl Cache for ClaimCache {
            async fn get(&self, _key: &str) -> Option<String> {
                None
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {
                self.val.store(1, Ordering::Relaxed);
            }
            async fn delete(&self, _key: &str) {
                self.val.store(0, Ordering::Relaxed);
            }
            async fn incr(
                &self,
                _key: &str,
                by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, oxllm_core::Error> {
                let prev = self.val.fetch_add(1, Ordering::Relaxed) as f64;
                Ok(prev + by)
            }
        }

        let cache: Arc<dyn Cache> = Arc::new(ClaimCache {
            val: AtomicU64::new(0),
        });
        let state = Arc::new(
            AlertState::new()
                .with_thresholds(vec![0.8])
                .with_cache(Some(cache)),
        );

        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = state.clone();
            handles.push(tokio::spawn(async move {
                s.mark_fired_if_new("key:abc", 0.8).await
            }));
        }
        let mut winners = 0;
        for h in handles {
            if h.await.unwrap() {
                winners += 1;
            }
        }
        assert_eq!(
            winners, 1,
            "exactly one concurrent claim must win under the atomic contract"
        );
    }

    /// Steady-state write avoidance: once a threshold has fired, repeated
    /// `mark_fired_if_new` calls for the same (scope, threshold) must not
    /// keep writing to the cache. The first crossing performs exactly one
    /// `incr`; every subsequent call in the same window must short-circuit
    /// on the cheap `get` and never reach `incr`. A regression that drops
    /// the fast-path would make `incr_count` grow without bound here.
    #[tokio::test]
    async fn distributed_mark_fired_if_new_skips_incr_after_first_fire() {
        use std::sync::atomic::{AtomicU64, Ordering};

        #[derive(Default)]
        struct CountingCache {
            incr_count: AtomicU64,
            value: std::sync::Mutex<Option<f64>>,
        }

        #[async_trait::async_trait]
        impl Cache for CountingCache {
            async fn get(&self, _key: &str) -> Option<String> {
                self.value.lock().unwrap().map(|v| v.to_string())
            }
            async fn set(&self, _key: &str, value: String, _ttl: Option<Duration>) {
                *self.value.lock().unwrap() = value.parse::<f64>().ok();
            }
            async fn delete(&self, _key: &str) {
                *self.value.lock().unwrap() = None;
            }
            async fn incr(
                &self,
                _key: &str,
                by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, oxllm_core::Error> {
                let n = self.incr_count.fetch_add(1, Ordering::Relaxed);
                let new = n as f64 + by;
                *self.value.lock().unwrap() = Some(new);
                Ok(new)
            }
        }

        let cache: Arc<CountingCache> = Arc::new(CountingCache::default());
        let state = Arc::new(
            AlertState::new()
                .with_thresholds(vec![0.8])
                .with_cache(Some(cache.clone() as Arc<dyn Cache>)),
        );

        assert!(
            state.mark_fired_if_new("key:abc", 0.8).await,
            "first call must claim"
        );
        // N subsequent calls must all short-circuit on `get` and not `incr`.
        for _ in 0..10 {
            assert!(
                !state.mark_fired_if_new("key:abc", 0.8).await,
                "subsequent calls must not claim"
            );
        }
        assert_eq!(
            cache.incr_count.load(Ordering::Relaxed),
            1,
            "exactly one incr per (scope, threshold) across repeated calls"
        );
    }

    /// The soft-budget sentinel must use a string-distinct dedup key so it
    /// cannot collide with any numeric ratio threshold (notably `-1.0`, which
    /// would otherwise format identically).
    #[test]
    fn soft_budget_key_does_not_collide_with_ratio_threshold() {
        let state = AlertState::new();
        let soft = state.key("key:abc", SOFT_BUDGET_SENTINEL);
        let ratio = state.key("key:abc", 0.8);
        assert_ne!(soft, ratio, "soft-budget key must differ from a ratio key");
        assert!(
            soft.ends_with(":soft"),
            "soft-budget key must be string-distinct, got: {soft}"
        );
        assert!(ratio.ends_with(":0.8"));
    }

    /// A transient backend read failure during evaluation must NOT clear the
    /// scope's fired markers (which would re-arm every threshold and cause an
    /// alert storm when the backend recovers). `try_counter_value` surfaces the
    /// outage as `Err`, and the evaluator skips the scope entirely.
    #[tokio::test]
    async fn eval_skips_and_preserves_fired_on_backend_read_error() {
        struct FailingReadCache;
        #[async_trait::async_trait]
        impl Cache for FailingReadCache {
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
                Ok(0.0)
            }
            async fn try_counter_value(
                &self,
                _key: &str,
            ) -> Result<Option<f64>, oxllm_core::Error> {
                Err(oxllm_core::Error::Network(
                    "simulated backend outage".to_string(),
                ))
            }
        }

        let cache: Arc<dyn Cache> = Arc::new(FailingReadCache);
        let state = AlertState::new();
        state.mark_fired("key:abc", 0.8).await;
        assert!(state.was_fired("key:abc", 0.8).await);

        let hits = state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[0.8], None)
            .await;
        assert!(hits.is_empty(), "no alerts during a backend outage");
        assert!(
            state.was_fired("key:abc", 0.8).await,
            "fired marker must survive a backend read error (no clear on outage)"
        );
    }

    #[tokio::test]
    async fn distributed_alert_state_clear_scope_deletes_keys() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = AlertState::new()
            .with_thresholds(vec![0.8, 0.9, 1.0])
            .with_cache(Some(cache.clone()));

        state.mark_fired("key:abc", 0.8).await;
        state.mark_fired("key:abc", 0.9).await;
        state.mark_fired("key:abc", 1.0).await;

        state.clear_scope("key:abc").await;

        assert!(!state.was_fired("key:abc", 0.8).await);
        assert!(!state.was_fired("key:abc", 0.9).await);
        assert!(!state.was_fired("key:abc", 1.0).await);
    }

    /// Documents the documented asymmetry between the two `clear_scope`
    /// branches: the distributed branch deletes only currently-configured
    /// threshold keys (plus the soft sentinel), while the in-memory branch
    /// uses prefix matching and cleans up every key for the scope. A
    /// marker for a threshold not in `self.thresholds` (e.g. one left
    /// behind by a previous config) survives in distributed mode but is
    /// cleared in in-memory mode. In practice hot-reload builds a fresh
    /// `AlertState` and the orphaned Redis key TTLs out, so this is a
    /// documented edge case, not a defect.
    #[tokio::test]
    async fn clear_scope_asymmetry_orphan_threshold_survives_distributed() {
        // Distributed branch: orphan marker survives.
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let distributed = AlertState::new()
            .with_thresholds(vec![0.8])
            .with_cache(Some(cache.clone()));
        distributed.mark_fired("key:abc", 0.85).await;
        assert!(distributed.was_fired("key:abc", 0.85).await);
        distributed.clear_scope("key:abc").await;
        assert!(
            distributed.was_fired("key:abc", 0.85).await,
            "distributed clear_scope leaves orphan threshold markers (documented asymmetry)"
        );

        // In-memory branch: orphan marker is cleaned up by prefix match.
        let in_memory = AlertState::new().with_thresholds(vec![0.8]);
        in_memory.mark_fired("key:abc", 0.85).await;
        assert!(in_memory.was_fired("key:abc", 0.85).await);
        in_memory.clear_scope("key:abc").await;
        assert!(
            !in_memory.was_fired("key:abc", 0.85).await,
            "in-memory clear_scope uses prefix match and removes orphan markers"
        );
    }

    #[tokio::test]
    async fn distributed_alert_state_ttl_expiry() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = AlertState::new()
            .with_thresholds(vec![0.8])
            .with_budget_window(Duration::from_millis(50))
            .with_cache(Some(cache));

        state.mark_fired("key:abc", 0.8).await;
        assert!(state.was_fired("key:abc", 0.8).await);

        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(
            !state.was_fired("key:abc", 0.8).await,
            "fired marker must expire after the budget window TTL"
        );
    }

    // ---------------------------------------------------------------
    // H10: soft-budget alerting
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn soft_budget_fires_when_spend_crosses_ceiling() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        // soft_budget = 80 (80% of limit 100). Spend at 79 → no fire.
        let _ = cache.incr("spend:abc", 79.0, None).await;
        let hits = state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[0.8], Some(80.0))
            .await;
        assert!(
            !hits.iter().any(|h| h.threshold == SOFT_BUDGET_SENTINEL),
            "soft_budget must not fire below the ceiling"
        );
        // Spend at 80 → fire.
        let _ = cache.incr("spend:abc", 1.0, None).await;
        let hits = state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[0.8], Some(80.0))
            .await;
        assert!(
            hits.iter().any(|h| h.threshold == SOFT_BUDGET_SENTINEL),
            "soft_budget must fire when spend crosses the ceiling"
        );
        assert!(
            state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "soft_budget marker must be set after firing"
        );
    }

    #[tokio::test]
    async fn soft_budget_does_not_refire_after_dedup() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let _ = cache.incr("spend:abc", 85.0, None).await;
        // First eval → fire.
        let hits = state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[], Some(80.0))
            .await;
        assert_eq!(hits.len(), 1);
        // Second eval → deduped, no new hit.
        let hits = state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[], Some(80.0))
            .await;
        assert!(
            hits.is_empty(),
            "soft_budget must not refire within the same budget window"
        );
    }

    #[tokio::test]
    async fn soft_budget_cleared_on_budget_reset() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let _ = cache.incr("spend:abc", 85.0, None).await;
        state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[], Some(80.0))
            .await;
        assert!(state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await);
        // Simulate window reset: spend drops to ~0.
        let _ = cache.incr("spend:abc", -85.0, None).await;
        state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[], Some(80.0))
            .await;
        assert!(
            !state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "soft_budget marker must be cleared on budget reset"
        );
    }

    #[tokio::test]
    async fn soft_budget_none_does_not_fire() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let _ = cache.incr("spend:abc", 200.0, None).await;
        let hits = state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[], None)
            .await;
        assert!(
            hits.is_empty(),
            "no soft_budget alert when soft_budget is None"
        );
        assert!(
            !state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "no soft_budget marker when soft_budget is None"
        );
    }

    /// The distributed (cache-backed) AlertState must also clear the
    /// soft-budget sentinel on budget reset.
    #[tokio::test]
    async fn soft_budget_cleared_on_reset_distributed() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(
            AlertState::new()
                .with_thresholds(vec![0.8])
                .with_cache(Some(cache.clone())),
        );
        let _ = cache.incr("spend:abc", 85.0, None).await;
        state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[0.8], Some(80.0))
            .await;
        assert!(state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await);
        assert!(
            cache
                .get(&state.key("key:abc", SOFT_BUDGET_SENTINEL))
                .await
                .is_some(),
            "sentinel marker must exist in cache"
        );
        // Budget reset.
        let _ = cache.incr("spend:abc", -85.0, None).await;
        state
            .eval_thresholds_with_soft(&cache, "key:abc", "spend:abc", 100.0, &[0.8], Some(80.0))
            .await;
        assert!(
            cache
                .get(&state.key("key:abc", SOFT_BUDGET_SENTINEL))
                .await
                .is_none(),
            "sentinel marker must be deleted from cache on reset"
        );
    }

    /// AlertingHook with soft_budget_pct fires a "soft_budget_exceeded"
    /// webhook event when spend crosses the derived ceiling.
    #[tokio::test]
    async fn hook_fires_soft_budget_exceeded_event() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let url = server.base_url() + "/alert";

        let post_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/alert")
                .json_body_partial(r#"{"event":"soft_budget_exceeded"}"#);
            then.status(200);
        });

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(
            cache.clone(),
            CallbackConfig {
                url,
                headers: HashMap::new(),
            },
            vec![],
            state.clone(),
        )
        .expect("build hook")
        .with_soft_budget_pct(Some(0.8));

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;

        for _ in 0..40 {
            if post_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            post_mock.hits() >= 1,
            "soft_budget_exceeded webhook must fire when spend crosses 80% of max_budget"
        );
        assert!(
            state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "sentinel marker must be set"
        );
    }

    /// AlertingHook without soft_budget_pct does NOT fire soft events even
    /// when spend is high (only ratio thresholds fire).
    #[tokio::test]
    async fn hook_without_soft_budget_pct_does_not_fire_soft_event() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let hook = AlertingHook::new(cache.clone(), empty_config(), vec![0.8], state.clone())
            .expect("build hook");
        // No with_soft_budget_pct → soft_budget_pct is None.

        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        let _ = cache.incr("spend:abc", 85.0, None).await;
        hook.post_success(&ctx, &make_response()).await;
        // Ratio threshold fires; soft does not.
        assert!(state.was_fired("key:abc", 0.8).await);
        assert!(
            !state.was_fired("key:abc", SOFT_BUDGET_SENTINEL).await,
            "soft_budget must not fire when pct is unset"
        );
    }

    /// Soft-budget alert fires independently of ratio thresholds: a scope
    /// with no configured ratio thresholds still fires the soft alert.
    #[tokio::test]
    async fn soft_budget_fires_without_ratio_thresholds() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let state = Arc::new(AlertState::new());
        let _ = cache.incr("spend:abc", 90.0, None).await;
        let hits = state
            .eval_thresholds_with_soft(
                &cache,
                "key:abc",
                "spend:abc",
                100.0,
                &[], // no ratio thresholds
                Some(80.0),
            )
            .await;
        assert_eq!(
            hits.len(),
            1,
            "soft_budget must fire even with no ratio thresholds"
        );
        assert_eq!(hits[0].threshold, SOFT_BUDGET_SENTINEL);
    }
}
