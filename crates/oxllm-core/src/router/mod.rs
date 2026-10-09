//! The router: deployment selection across multiple deployments per model.
//!
//! The router operates on [`crate::Deployment`] values and the
//! [`CooldownStore`] only. It never imports providers or server code.
//! The router is pure: no network, filesystem, or env access.

pub mod adaptive;
pub mod complexity;
pub mod cooldown;
pub mod health;
pub mod latency_based;
pub mod least_busy;
pub mod lowest_cost;
pub mod lowest_latency;
pub mod pre_call_check;
pub mod quality;
pub mod round_robin;
pub mod session_affinity;
pub mod simple_shuffle;
pub mod stores;
pub mod strategy;
pub mod tag_based;
pub mod types;
pub mod usage_based;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::deployment::Deployment;

pub use adaptive::{AdaptiveRouter, BanditStore, BetaCell, RequestType};
pub use complexity::{ComplexityRouter, Tier};
pub use cooldown::CooldownStore;
pub use health::{HealthEntry, HealthState};
pub use latency_based::LatencyBased;
pub use least_busy::LeastBusy;
pub use lowest_cost::LowestCost;
pub use lowest_latency::LowestLatency;
pub use pre_call_check::{HealthCheck, PreCallCheck, normalize_check_name};
pub use quality::QualityRouter;
pub use session_affinity::SessionAffinity;
pub use simple_shuffle::SimpleShuffle;
pub use stores::{LatencyStore, UsageStore, deployment_key, stable_key};
pub use strategy::{PreCallData, RequestContext, RoutingStrategy};
pub use tag_based::TagBased;
pub use types::{
    AdaptiveRouterConfig, ComplexityRouterConfig, FallbackRule, ProviderBudgetConfig, RetryPolicy,
    RouterConfig, RouterGroupOverride, RoutingStrategyKind,
};
pub use usage_based::UsageBased;

/// Decision returned by [`Router::record_failure_and_check`] after recording
/// a failed attempt under the allowed-fails policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowedFailDecision {
    /// No allowed-fails policy covers this status. The failure was **not**
    /// counted; existing cooldown behavior (cool after retries are
    /// exhausted) applies unchanged.
    NoPolicy,
    /// The failure was counted and is below the configured limit — keep
    /// retrying the deployment (up to `max_retries`).
    Tolerated {
        /// Failures recorded for this status within the current window.
        count: u32,
        /// The effective limit (per-status policy or global `allowed_fails`).
        limit: u32,
    },
    /// The failure count exceeded the configured limit (exclusive semantics:
    /// `count > limit`, so `allowed_fails: N` tolerates N failures and
    /// cools on the (N+1)th) — cool the deployment and move on to the next
    /// one.
    LimitReached {
        /// Failures recorded for this status within the current window.
        count: u32,
        /// The effective limit (per-status policy or global `allowed_fails`).
        limit: u32,
    },
}

/// The router selects deployments for requests, managing cooldowns and
/// fallback rules.
///
/// It holds the routing configuration, a cooldown store, latency/usage stores,
/// a boxed default routing strategy, pre-built per-model routing-group
/// strategies, the built-in health pre-call check (when health-check routing
/// is enabled), and host-injected pre-call checks. All methods are
/// synchronous (pure core).
pub struct Router {
    config: RouterConfig,
    cooldown: Arc<CooldownStore>,
    latency_store: Arc<LatencyStore>,
    usage_store: Arc<UsageStore>,
    health: Arc<HealthState>,
    /// Shared bandit cell table for the `adaptive_router` strategy. Built
    /// once per router (like [`HealthState`]) and shared with the server's
    /// learning hook via [`Router::bandit_store_arc`]; preserved across
    /// config hot-reloads via [`Router::rebuild_from_stores`].
    bandit_store: Arc<BanditStore>,
    /// Shared per-model-group monotonic counters for the `round_robin`
    /// strategy. Each model group (keyed by the normalized group key —
    /// see [`PreCallData::model_group_key`]; the raw requested model only
    /// when it resolves exactly) gets its own counter, so a
    /// global `round_robin` config with two or more groups strictly alternates
    /// within each group regardless of cross-group interleaving — sharing a
    /// single counter would let inter-group picks advance one group's index
    /// in lockstep, pinning it to a single deployment (e.g. an A,B,A,B,…
    /// interleave over groups A=`[a1,a2]`, B=`[b1,b2,b3]` leaves A stuck on
    /// `a1`). Built once per router (like [`HealthState`] and
    /// [`BanditStore`]) and preserved across config hot-reloads via
    /// [`Router::rebuild_from_stores`], so a canary add or credentials
    /// rotation does not reset the strict-rotation position back to
    /// `deployments[0]`. Other stateless strategies ignore this field.
    round_robin_counters: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    strategy: Box<dyn RoutingStrategy>,
    routing_group_strategies: HashMap<String, Box<dyn RoutingStrategy>>,
    /// Host-injected pre-call checks (e.g. prompt-caching, budget limiter),
    /// named by [`RouterConfig::optional_pre_call_checks`]. The built-in
    /// health check is held separately in [`Router::health_check`].
    pre_call_checks: Vec<Arc<dyn PreCallCheck>>,
    /// [`RouterConfig::optional_pre_call_checks`] normalized
    /// (kebab→snake) and deduplicated, preserving first-occurrence order.
    /// Listing the same check twice must execute it once (the pipeline
    /// below iterates this list, not the raw config).
    check_order: Vec<String>,
    /// The built-in health pre-call check, constructed when
    /// `enable_health_check_routing` is set. Runs at its listed position in
    /// `optional_pre_call_checks`, or appended after all other checks when
    /// unlisted (preserving the pre-framework behavior where health routing
    /// needed no opt-in list entry).
    health_check: Option<HealthCheck>,
}

impl Router {
    /// Create a new router with the given configuration.
    ///
    /// ```rust
    /// use oxllm_core::router::Router;
    /// use oxllm_core::router::types::RouterConfig;
    ///
    /// let router = Router::new(RouterConfig::default());
    /// assert_eq!(router.config().num_retries, 2);
    /// ```
    pub fn new(config: RouterConfig) -> Self {
        let latency_store = Arc::new(LatencyStore::default());
        let usage_store = Arc::new(UsageStore::new());
        let cooldown = Arc::new(CooldownStore::new());
        let health = Arc::new(HealthState::new());
        let bandit_store = Arc::new(BanditStore::new());
        let round_robin_counters = Arc::new(Mutex::new(HashMap::new()));
        Self::build_inner(
            config,
            cooldown,
            latency_store,
            usage_store,
            health,
            bandit_store,
            round_robin_counters,
        )
    }

    fn build_inner(
        config: RouterConfig,
        cooldown: Arc<CooldownStore>,
        latency_store: Arc<LatencyStore>,
        usage_store: Arc<UsageStore>,
        health: Arc<HealthState>,
        bandit_store: Arc<BanditStore>,
        round_robin_counters: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    ) -> Self {
        let deps = strategy::StrategyDeps {
            quality_scores: &config.quality_scores,
            latency_store: &latency_store,
            usage_store: &usage_store,
            complexity_config: &config.complexity_router_config,
            adaptive_config: &config.adaptive_router_config,
            bandit_store: &bandit_store,
            round_robin_counters: &round_robin_counters,
        };
        let strategy = strategy::strategy_factory(config.routing_strategy)(&deps);
        let mut routing_group_strategies: HashMap<String, Box<dyn RoutingStrategy>> =
            HashMap::new();
        for (model, group) in &config.routing_groups {
            let Some(kind) = group.routing_strategy.as_ref() else {
                continue;
            };
            routing_group_strategies
                .insert(model.clone(), strategy::strategy_factory(*kind)(&deps));
        }
        let health_check = config.enable_health_check_routing.then(|| {
            HealthCheck::new(
                health.clone(),
                Duration::from_secs(config.health_check_staleness_threshold_secs),
            )
        });
        let mut check_order: Vec<String> =
            Vec::with_capacity(config.optional_pre_call_checks.len());
        for name in &config.optional_pre_call_checks {
            let normalized = normalize_check_name(name);
            if !check_order.contains(&normalized) {
                check_order.push(normalized);
            }
        }
        if health_check.is_none() {
            let health_listed = config
                .optional_pre_call_checks
                .iter()
                .any(|n| normalize_check_name(n) == "health_check");
            if health_listed {
                tracing::warn!(
                    event = "router_config",
                    "`optional_pre_call_checks` lists `health_check` but \
                     `enable_health_check_routing` is false; the health check \
                     will be a silent no-op. Set `enable_health_check_routing: \
                     true` to enable health-check routing."
                );
            }
        }
        if config.failure_threshold_min_requests == Some(0)
            && config.failure_threshold_percent.is_some()
        {
            // `failure_threshold_min_requests: 0` makes the percentage check
            // trip on the first failure (the `win_total >= 1` guard is
            // already true after a failure). This is almost always a config
            // typo (unset → defaults to 1); surface it loudly instead of
            // silently turning percentage cooldown into one-strike.
            tracing::warn!(
                event = "router_config",
                "`failure_threshold_min_requests` is 0: the percentage-based \
                 cooldown will trip on the first failure. Set it to at least 1 \
                 (or leave it unset) unless one-strike cooldown is intended."
            );
        }
        if let Some(threshold) = config.failure_threshold_percent {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                // An out-of-range value (e.g. `80` from an operator meaning
                // 80%) silently disables the percentage cooldown — the check
                // guards every comparison against the [0,1] range. Sibling
                // config mistakes warn loudly at load; this one must too.
                tracing::warn!(
                    event = "router_config",
                    value = threshold,
                    "`failure_threshold_percent` is outside [0,1]; the \
                     percentage-based cooldown is disabled. Express the \
                     threshold as a fraction (e.g. 0.8 for 80%)."
                );
            }
        }
        if config.cooldown_time_secs == 0 {
            Self::warn_zero_cooldown("<global>");
        }
        for (model, group) in &config.routing_groups {
            if group.cooldown_time_secs == Some(0) {
                Self::warn_zero_cooldown(model);
            }
        }
        Self {
            config,
            cooldown,
            latency_store,
            usage_store,
            health,
            bandit_store,
            round_robin_counters,
            strategy,
            routing_group_strategies,
            pre_call_checks: Vec::new(),
            check_order,
            health_check,
        }
    }

    /// Load-time warning for a `cooldown_time` of 0 (global when `scope` is
    /// `"<global>"`, else the routing-group name): a zero cooldown means a
    /// failed deployment is never excluded from rotation, so every request
    /// re-attempts it up to the retry-loop bounds. Almost always a typo
    /// (unset → 60s default); the orchestration loops bound their pick
    /// counts, but the config deserves a loud signal anyway.
    fn warn_zero_cooldown(scope: &str) {
        tracing::warn!(
            event = "router_config",
            scope = %scope,
            "`cooldown_time` is 0: failed deployments are never excluded \
             from rotation (the mark expires instantly), so a persistently \
             failing deployment is re-attempted up to the retry bounds on \
             every request. Set a positive cooldown unless disabled \
             cooldowns are intended."
        );
    }

    /// Rebuild the router with a new config while preserving the runtime
    /// stores (cooldowns, latency samples, usage tracking, in-flight counters,
    /// health state, bandit cells).
    ///
    /// This is the correct way to apply a config hot-reload: deployments that
    /// are cooling down stay in cooldown, accumulated latency data is kept,
    /// usage-based routing continues from known baselines, and the adaptive
    /// bandit keeps its learned posteriors.
    pub fn rebuild_from_stores(
        config: RouterConfig,
        cooldown: Arc<CooldownStore>,
        latency_store: Arc<LatencyStore>,
        usage_store: Arc<UsageStore>,
        health: Arc<HealthState>,
        bandit_store: Arc<BanditStore>,
        round_robin_counters: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    ) -> Self {
        Self::build_inner(
            config,
            cooldown,
            latency_store,
            usage_store,
            health,
            bandit_store,
            round_robin_counters,
        )
    }

    /// Sweep every runtime store, dropping entries for deployments no longer
    /// in the active set. The intended call site is a config hot-reload, right
    /// after [`Router::rebuild_from_stores`]: the stores are deliberately
    /// preserved across reloads (cooldowns, latency samples, in-flight
    /// counters, health entries, learned bandit cells), so without this sweep
    /// a deployment removed from the model list leaks its state forever —
    /// under rolling config updates, canary churn, or wildcard re-indexing the
    /// maps grow monotonically.
    ///
    /// `deployments` is the full current deployment list (the same slice a
    /// reload hands to [`Router::pick_deployment`]). Two key spaces are
    /// derived from it:
    /// - latency/usage/in-flight: the stable per-deployment key
    ///   ([`stores::stable_key`], `"{model_name}:<config-hash>"`);
    /// - cooldown/health/bandit: the same stable key
    ///   ([`Router::cooldown_key`]).
    ///
    /// The cooldown store additionally tracks last-observed model-group sizes
    /// by normalized group name; those and round-robin counters retain live
    /// `model_name`s (including `"*"`) and resolvable configured aliases.
    pub fn prune_stale_store_keys(&self, deployments: &[Deployment]) {
        let mut known_latency: HashSet<String> = HashSet::with_capacity(deployments.len());
        let mut known_models: HashSet<String> = HashSet::with_capacity(deployments.len());
        // Live per-model deployment counts, used below to refresh
        // `cooldown.group_sizes`. Without this refresh, a hot-reload that
        // shrinks a group from N to 1 leaves the stale `group_size == N`
        // until the next `pick_deployment` (which is the only other writer).
        // In that one-pick window, a failing request could cool the now-solo
        // deployment via the percentage path — the exact outcome the
        // `group_size != 1` exemption exists to prevent. Recomputing here
        // (the reload call site, which already iterates the live list) closes
        // the window without changing `record_failure_and_check`'s signature.
        let mut live_sizes: HashMap<String, usize> = HashMap::new();
        for d in deployments {
            known_latency.insert(self.cooldown_key(d));
            known_models.insert(d.model_name.clone());
            *live_sizes.entry(d.model_name.clone()).or_insert(0) += 1;
        }
        // Picks resolving through aliases are keyed by the requested alias,
        // not its members' model_names. Use the same resolution path here,
        // including exact/wildcard fallback for an empty alias group.
        for alias in self.config.model_group_alias.keys() {
            let candidates = self.deployments_for_model(deployments, alias);
            if !candidates.is_empty() {
                known_models.insert(alias.clone());
                live_sizes.insert(alias.clone(), candidates.len());
            }
        }
        // An empty deployment set leaves the sets empty, so `retain_keys`
        // drops everything — the stored state is unreachable anyway.
        self.latency_store.retain_keys(&known_latency);
        self.usage_store.retain_keys(&known_latency);
        self.cooldown.retain_keys(&known_latency);
        self.health.retain_keys(&known_latency);
        self.bandit_store.retain_keys(&known_latency);
        self.cooldown.retain_group_sizes(&known_models);
        // The per-model-group round-robin counters share the normalized group
        // key space of `cooldown.group_sizes`, so they are swept against the
        // same `known_models` set — otherwise a retired/renamed model group
        // leaks its counter forever (the exact class of growth this sweep
        // exists to prevent under canary churn / wildcard re-indexing).
        self.round_robin_counters
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|m, _| known_models.contains(m));
        for (model, size) in live_sizes {
            self.cooldown.set_group_size(&model, size);
        }
    }

    /// Get a clone of the cooldown store handle (for preserving state across reloads).
    pub fn cooldown_arc(&self) -> Arc<CooldownStore> {
        self.cooldown.clone()
    }

    /// Get a clone of the latency store handle (for preserving state across reloads).
    pub fn latency_store_arc(&self) -> Arc<LatencyStore> {
        self.latency_store.clone()
    }

    /// Get a clone of the usage store handle (for preserving state across reloads).
    pub fn usage_store_arc(&self) -> Arc<UsageStore> {
        self.usage_store.clone()
    }

    /// Get a clone of the health state handle (for preserving state across reloads).
    pub fn health_store_arc(&self) -> Arc<HealthState> {
        self.health.clone()
    }

    /// Get a clone of the bandit store handle. The server's learning hook
    /// clones this `Arc` so post-call outcomes land in the same cells the
    /// `adaptive_router` strategies sample from; also used to preserve the
    /// learned posteriors across config hot-reloads.
    pub fn bandit_store_arc(&self) -> Arc<BanditStore> {
        self.bandit_store.clone()
    }

    /// Get a clone of the per-model-group round-robin counters handle. The
    /// strategy factory shares this map so a config hot-reload (which
    /// rebuilds the strategy via [`Router::rebuild_from_stores`]) preserves
    /// each group's strict-rotation position rather than resetting it to
    /// `deployments[0]`. Each model group gets its own counter so a global
    /// `round_robin` config with multiple groups strictly alternates within
    /// each group regardless of cross-group interleaving.
    pub fn round_robin_counters_arc(&self) -> Arc<Mutex<HashMap<String, Arc<AtomicU64>>>> {
        self.round_robin_counters.clone()
    }

    /// Access the router configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Access the cooldown store (for testing, direct manipulation, or
    /// state inspection). The returned `Arc` handle can be cloned to share
    /// the store with an async inspection endpoint; store methods resolve
    /// through the `Arc` via auto-deref.
    pub fn cooldown_store(&self) -> &Arc<CooldownStore> {
        &self.cooldown
    }

    /// Access the latency store.
    pub fn latency_store(&self) -> &LatencyStore {
        &self.latency_store
    }

    /// Access the usage store.
    pub fn usage_store(&self) -> &UsageStore {
        &self.usage_store
    }

    /// Access the health state (for testing or direct manipulation).
    pub fn health_store(&self) -> &HealthState {
        &self.health
    }

    /// Access the shared health-state handle. The returned `Arc` can be
    /// cloned for an async state-inspection endpoint; pair
    /// [`HealthState::is_healthy`] (effective fail-open state) with
    /// [`HealthState::reason`] (last recorded unhealthy reason).
    pub fn health_state(&self) -> &Arc<HealthState> {
        &self.health
    }

    /// Access the bandit store (for testing, direct manipulation, or
    /// state inspection — see [`BanditStore::all_cells`]). The returned
    /// `Arc` handle can be cloned to share the store; store methods
    /// resolve through the `Arc` via auto-deref.
    pub fn bandit_store(&self) -> &Arc<BanditStore> {
        &self.bandit_store
    }

    /// Names of the pre-call checks available to the pick pipeline: the
    /// host-injected checks (registration order) plus the built-in
    /// `health_check` when health-check routing is enabled. Note this is
    /// the set of *registered* checks for state inspection — the checks
    /// that actually *run* on a pick, and their order, are governed by
    /// [`RouterConfig::optional_pre_call_checks`] (unlisted injected
    /// checks never run).
    pub fn pre_call_check_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .pre_call_checks
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        if let Some(health_check) = &self.health_check {
            names.push(health_check.name().to_string());
        }
        names
    }

    /// Inject the host-built pre-call checks (prompt-caching, budget
    /// limiter, ...). Called once at construction by the host, after
    /// [`Router::new`] / [`Router::rebuild_from_stores`]. Checks run in the
    /// order named by [`RouterConfig::optional_pre_call_checks`]; checks
    /// whose names are not listed never run.
    ///
    /// Any name listed in `optional_pre_call_checks` (other than the
    /// router-managed `health_check`) that no injected check provides is
    /// warned about here — **once per config load**, matching the load-time
    /// `router_config` warnings in `Router::build_inner` — so a config
    /// typo does not emit a warn on every pick (the pick pipeline only
    /// logs the mismatch at `debug`).
    pub fn set_pre_call_checks(&mut self, checks: Vec<Arc<dyn PreCallCheck>>) {
        for name in &self.check_order {
            if name == "health_check" {
                // Router-managed: satisfied by `self.health_check`, not by
                // an injected check (its absence is already warned about at
                // load time in `build_inner`).
                continue;
            }
            if !checks.iter().any(|c| c.name() == name.as_str()) {
                tracing::warn!(
                    event = "pre_call_check",
                    check = %name,
                    "`optional_pre_call_checks` lists an unregistered pre-call check; \
                     it will be a silent no-op"
                );
            }
        }
        self.pre_call_checks = checks;
    }

    /// The effective routing-strategy kind for a model: the routing-group
    /// override when set, otherwise the global strategy.
    pub fn strategy_kind_for(&self, model: &str) -> RoutingStrategyKind {
        self.config
            .routing_groups
            .get(model)
            .and_then(|g| g.routing_strategy)
            .unwrap_or(self.config.routing_strategy)
    }

    /// Whether the `adaptive_router` strategy is configured anywhere —
    /// globally or for any routing group. The server uses this to register
    /// the learning hook and to decide whether the once-per-request
    /// request-type classification is worth computing: an adaptive leg can
    /// be reached via fallback chains even when the requested model's own
    /// group is not adaptive.
    pub fn adaptive_active(&self) -> bool {
        self.config.routing_strategy == RoutingStrategyKind::AdaptiveRouter
            || self
                .config
                .routing_groups
                .values()
                .any(|g| g.routing_strategy == Some(RoutingStrategyKind::AdaptiveRouter))
    }

    /// Record a connection-level failure (network error, DNS failure, TLS
    /// error, client-side timeout) for a deployment key under health-check
    /// routing. Returns `true` when this call **newly** marked the
    /// deployment unhealthy (consecutive failures crossed
    /// `health_check_failure_threshold` on this call) — exactly once per
    /// unhealthy episode, so the caller can fire transition side effects
    /// (alerting, unhealthy-state metrics) without deduplicating. For the
    /// effective current state, query
    /// [`HealthState::is_healthy`](crate::router::HealthState::is_healthy).
    /// No-op returning `false` when `enable_health_check_routing` is false.
    pub fn record_connection_failure(&self, key: &str, reason: &str) -> bool {
        if !self.config.enable_health_check_routing {
            return false;
        }
        let newly_unhealthy =
            self.health
                .record_failure(key, reason, self.config.health_check_failure_threshold);
        tracing::debug!(
            event = "health_check",
            deployment = %key,
            newly_unhealthy,
            reason = %reason,
            "connection failure recorded"
        );
        newly_unhealthy
    }

    /// Record a successful connection for a deployment key, clearing its
    /// health entry (auto-recovery). Any HTTP-level response — even an error
    /// status — proves connectivity; only transport failures degrade health.
    /// No-op when `enable_health_check_routing` is false.
    pub fn record_connection_success(&self, key: &str) {
        if !self.config.enable_health_check_routing {
            return;
        }
        self.health.mark_healthy(key);
    }

    /// Record a latency sample (in milliseconds) for a deployment key.
    pub fn record_latency(&self, key: &str, duration_ms: u64) {
        self.latency_store.record(key, duration_ms);
    }

    /// Record a usage score for a deployment key.
    pub fn record_usage(&self, key: &str, usage: f64) {
        self.usage_store.record(key, usage);
    }

    /// Adjust the in-flight request count for a deployment key by `delta`
    /// (typically `+1` before an upstream call and `-1` after). Used by the
    /// `LeastBusy` strategy.
    pub fn record_in_flight(&self, key: &str, delta: i64) {
        self.usage_store.incr_in_flight(key, delta);
    }

    /// Get all deployments matching a model name, including wildcard matches
    /// and model-group aliases.
    ///
    /// Resolution order:
    /// 1. **Alias expansion.** If `model` matches a key in
    ///    [`RouterConfig::model_group_alias`], return deployments for every
    ///    model name in the alias group (in declaration order).
    /// 2. **Exact match.** If any deployment has `model_name == model`,
    ///    return those.
    /// 3. **Wildcard.** Otherwise, fall back to deployments with
    ///    `model_name == "*"`. A wildcard deployment matches only requested
    ///    models whose provider prefix is compatible with the deployment's
    ///    `oxllm_params.model` pattern: a bare `*` configured model matches
    ///    anything (a bare non-`*` name is an exact model match), `provider/*`
    ///    matches only requested models whose own provider prefix equals
    ///    `provider`, and a partial pattern like `provider/prefix-*` (e.g.
    ///    `groq/llama-*`) matches same-provider requests whose model name
    ///    matches the glob (`groq/llama-3.1-70b` but not `groq/mistral-7b`).
    ///    A requested model with no `/` defaults to the `openai` provider.
    ///    This prevents cross-provider misrouting (e.g. an `openai/*`
    ///    wildcard serving `anthropic/...`).
    pub fn deployments_for_model<'a>(
        &self,
        deployments: &'a [Deployment],
        model: &str,
    ) -> Vec<&'a Deployment> {
        if let Some(alias_group) = self.config.model_group_alias.get(model) {
            let mut result: Vec<&Deployment> = Vec::new();
            for alias_model in alias_group {
                for d in deployments {
                    if &d.model_name == alias_model
                        && !result.iter().any(|seen| std::ptr::eq(*seen, d))
                    {
                        result.push(d);
                    }
                }
            }
            if !result.is_empty() {
                return result;
            }
        }

        let exact: Vec<&'a Deployment> = deployments
            .iter()
            .filter(|d| d.model_name == model)
            .collect();
        if !exact.is_empty() {
            return exact;
        }
        deployments
            .iter()
            .filter(|d| d.model_name == "*" && wildcard_model_matches(&d.oxllm_params.model, model))
            .collect()
    }

    /// Pick a deployment using the configured strategy, excluding cooled-down
    /// deployments, restricting to the lowest available priority-order tier
    /// (`oxllm_params.order`, when any candidate sets it), and applying the
    /// pre-call check pipeline (health routing, plus any host-injected
    /// checks) before the strategy pick.
    ///
    /// When the effective strategy for the model is
    /// [`RoutingStrategyKind::ComplexityRouter`], the candidate list is first
    /// expanded from the requested model's group to the classified tier's
    /// configured model pool (with `default_model`, then the requested
    /// model's group as fallbacks — each level
    /// cooldown-filtered, first non-empty level wins).
    ///
    /// Returns `None` when no matching deployment is available (either there
    /// are zero candidates, every candidate is cooling down, or required /
    /// excluded tags reject the available candidates). Use
    /// [`Self::try_pick_deployment`] for an explicit selector-rejection error.
    ///
    /// ```rust
    /// use oxllm_core::router::Router;
    /// use oxllm_core::router::types::RouterConfig;
    /// use oxllm_core::router::strategy::RequestContext;
    ///
    /// let router = Router::new(RouterConfig::default());
    /// let deployments: Vec<oxllm_core::Deployment> = Vec::new();
    /// let ctx = RequestContext::new("gpt-4o");
    /// // No deployments configured → nothing to pick.
    /// assert!(router.pick_deployment(&deployments, &ctx).is_none());
    /// ```
    pub fn pick_deployment<'a>(
        &self,
        deployments: &'a [Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        match self.try_pick_deployment(deployments, ctx) {
            Ok(picked) => picked,
            Err(error) => {
                tracing::debug!(
                    event = "routing_rejected",
                    error = %error,
                    "deployment selection rejected by request constraints"
                );
                None
            }
        }
    }

    /// Checked deployment selection. Unlike [`Self::pick_deployment`], a
    /// non-empty candidate pool rejected by hard routing tags returns an
    /// explicit error. Hosts should propagate that error instead of trying
    /// an unconstrained fallback model group. Ordinary unavailability still
    /// returns `Ok(None)`.
    pub fn try_pick_deployment<'a>(
        &self,
        deployments: &'a [Deployment],
        ctx: &RequestContext,
    ) -> crate::Result<Option<&'a Deployment>> {
        let strategy_kind = self.strategy_kind_for(&ctx.model);
        // H4: apply default_tag for the tag_based strategy when the request
        // carries no tag. The tag is injected into a cloned context so the
        // original is untouched.
        let ctx_with_default = if strategy_kind == RoutingStrategyKind::TagBased
            && ctx.tags.is_empty()
            && (ctx.tag.as_deref().map(|t| t.is_empty()).unwrap_or(true))
        {
            self.config.default_tag.as_ref().map(|default_tag| {
                let mut c = ctx.clone();
                c.tag = Some(default_tag.clone());
                c
            })
        } else {
            None
        };
        let effective_ctx: &RequestContext = ctx_with_default.as_ref().unwrap_or(ctx);
        let (candidates, classification, mut keys) =
            if strategy_kind == RoutingStrategyKind::ComplexityRouter {
                let mut keys = CooldownKeyCache::new(self, 0);
                let (candidates, classification) =
                    self.complexity_candidates(deployments, effective_ctx, &mut keys);
                (candidates, classification, keys)
            } else {
                let candidates = self.deployments_for_model(deployments, &effective_ctx.model);
                let keys = CooldownKeyCache::new(self, candidates.len());
                (candidates, None, keys)
            };
        // H8: record the candidate count as the model group size so
        // percentage-based cooldown can apply single-deployment safety.
        // The key is the normalized group key (see [`Self::group_size_key`]),
        // never the raw client-requested string. On the complexity path the
        // recorded size is the requested model's OWN resolved group (not the
        // tier pool, which may span several model groups): the
        // single-deployment exemption compares against the model group a
        // failing deployment belongs to, and an inflated cross-group pool
        // size would disable the exemption for a genuinely solo deployment
        // — the exact outcome it exists to prevent. On the complexity path
        // `candidates` is also already cooldown-filtered (see
        // [`Self::complexity_candidates`]), so the configured group size is
        // the conservative direction there too: a smaller available pool
        // engages the safety sooner, never later.
        let group_key_and_size = if strategy_kind == RoutingStrategyKind::ComplexityRouter {
            let own: Vec<&Deployment> =
                self.deployments_for_model(deployments, &effective_ctx.model);
            (self.group_size_key(&own, &effective_ctx.model), own.len())
        } else {
            (
                self.group_size_key(&candidates, &effective_ctx.model),
                candidates.len(),
            )
        };
        let group_key = group_key_and_size.0;
        if let Some(key) = &group_key {
            self.cooldown.set_group_size(key, group_key_and_size.1);
        }
        // Stateful strategies that key per-group state by the request's
        // model (round_robin) must use the same normalized group key, so a
        // wildcard deployment cannot grow per-group state unboundedly off
        // client-controlled model names. See
        // [`PreCallData::model_group_key`].
        let keyed_ctx_holder;
        let effective_ctx: &RequestContext = if strategy_kind == RoutingStrategyKind::RoundRobin
            && group_key
                .as_deref()
                .is_some_and(|k| k != effective_ctx.model)
        {
            let mut c = effective_ctx.clone();
            c.pre_call.model_group_key = group_key.clone();
            keyed_ctx_holder = c;
            &keyed_ctx_holder
        } else {
            effective_ctx
        };
        // This cooldown filter is intentionally redundant with the
        // per-pool-level filter inside `complexity_candidates`: a
        // deployment may START cooling between that earlier filter and
        // this final pick, and re-filtering here only ever removes
        // newly-cooled deployments (safe). Keeping expired-cooldown
        // deployments out of a pool level is merely conservative — the
        // next fallback level covers them. Keep both filters as defense
        // against that race.
        // Full candidate set before cooldown (session affinity prefers the
        // sticky target from this set when it is still available).
        let full_group: Vec<&Deployment> = candidates.clone();
        let available: Vec<&Deployment> = candidates
            .into_iter()
            .filter(|d| {
                let key = keys.key_for(d);
                !self.cooldown.is_cooling(&key)
                    && !effective_ctx
                        .pre_call
                        .excluded_deployment_keys
                        .contains(&key)
            })
            .collect();
        // Hard tag constraints define eligibility, before priority and
        // fail-open pre-call checks. A lower-priority eligible deployment
        // must remain reachable when a higher-priority one is forbidden.
        let available = if strategy_kind == RoutingStrategyKind::TagBased {
            tag_based::constrain_deployments(available, effective_ctx)?
        } else {
            available
        };
        // Priority-order tiers run AFTER cooldown filtering (cooled
        // deployments are already gone, so the next tier becomes eligible
        // automatically) and BEFORE pre-call checks and the strategy pick
        // (weighted shuffle still applies WITHIN the surviving tier).
        let available = order_tier_filter(available);
        let available = self.run_pre_call_checks(available, effective_ctx, &mut keys);
        if available.is_empty() {
            return Ok(None);
        }
        // Session affinity: when the sticky deployment from the full group is
        // still available, prefer it instead of remapping over the cooled-
        // filtered set (which would break prompt-cache stickiness exactly
        // when the preferred host is unhealthy and siblings remain).
        if strategy_kind == RoutingStrategyKind::SessionAffinity {
            if let Some(sticky) =
                session_affinity::preferred_from_full_group(&full_group, effective_ctx)
            {
                let sticky_key = keys.key_for(sticky);
                if let Some(still) = available
                    .iter()
                    .copied()
                    .find(|d| keys.key_for(d) == sticky_key)
                {
                    return Ok(Some(still));
                }
            }
        }
        // Complexity candidates are already expanded through alias/wildcard
        // resolution. Do not re-filter them by literal tier model names:
        // that would discard alias members whenever a direct name also
        // matches. Share the resolved-pool picker with the standalone strategy.
        let pick = if let Some(classification) = classification {
            Ok(complexity::pick_resolved_pool(
                &available,
                effective_ctx,
                classification,
            ))
        } else if strategy_kind == RoutingStrategyKind::AdaptiveRouter {
            // The adaptive strategy needs the once-per-request request-type
            // classification and the cooldown keys of the final candidate list
            // (bandit cells are keyed by them). The host normally pre-computes
            // the type when it builds the request context (it stamps the cell
            // for the learning hook); classify on demand only when it did not.
            let mut adaptive_ctx = effective_ctx.clone();
            if adaptive_ctx.pre_call.adaptive_request_type.is_none() {
                adaptive_ctx.pre_call.adaptive_request_type =
                    Some(adaptive::classify_request_type(
                        effective_ctx.request_text.as_deref().unwrap_or(""),
                    ));
            }
            adaptive_ctx.pre_call.candidate_keys =
                available.iter().map(|d| keys.key_for(d)).collect();
            adaptive_ctx.pre_call.candidate_ids = available
                .iter()
                .map(|d| stores::deployment_key(d))
                .collect();
            self.strategy_for_model(&effective_ctx.model)
                .try_pick(&available, &adaptive_ctx)
        } else {
            self.strategy_for_model(&effective_ctx.model)
                .try_pick(&available, effective_ctx)
        }?;
        // H6: weighted intra-group failover — when the strategy pick returns
        // None (edge case) or the group is degraded, re-pick from the
        // remaining same-group candidates using SimpleShuffle weighting
        // before escalating to cross-group fallbacks.
        if pick.is_none() && self.config.enable_weighted_failover {
            return Ok(SimpleShuffle.pick(&available, effective_ctx));
        }
        Ok(pick)
    }

    /// Expand candidates for the complexity router: classify the request
    /// and resolve the tier's configured model pool, then fall back —
    /// each level cooldown-filtered, first non-empty level wins — to the
    /// `default_model` pool, then the requested model's group. Widening the
    /// pool beats failing the request when the tier pool is entirely cooling
    /// down ("degraded beats none", like the health-check safety net), but
    /// the widening is confined to the requested model's own group: an
    /// unrelated model is never served, matching the reference complexity router
    /// (which falls back to the requested model group, not the whole model
    /// list). A request whose model group resolves nothing (including an
    /// unknown model) yields an empty chain → `None` → 404. Also returns the
    /// classification so the strategy pick can reuse it instead of
    /// re-classifying.
    fn complexity_candidates<'a>(
        &self,
        deployments: &'a [Deployment],
        ctx: &RequestContext,
        keys: &mut CooldownKeyCache<'_>,
    ) -> (Vec<&'a Deployment>, Option<(Tier, f64)>) {
        let Some(config) = &self.config.complexity_router_config else {
            return (self.deployments_for_model(deployments, &ctx.model), None);
        };
        let (tier, score) = complexity::classify(ctx, config);
        let mut chain: Vec<Vec<&Deployment>> = Vec::with_capacity(3);
        if let Some(models) = config.tiers.get(tier.as_str()) {
            let mut pool: Vec<&Deployment> = Vec::new();
            for model in models {
                for d in self.deployments_for_model(deployments, model) {
                    if !pool.iter().any(|seen| std::ptr::eq(*seen, d)) {
                        pool.push(d);
                    }
                }
            }
            chain.push(pool);
        }
        if let Some(default_model) = &config.default_model {
            chain.push(self.deployments_for_model(deployments, default_model));
        }
        chain.push(self.deployments_for_model(deployments, &ctx.model));
        for pool in chain {
            let available: Vec<&Deployment> = pool
                .into_iter()
                .filter(|d| !self.cooldown.is_cooling(&keys.key_for(d)))
                .collect();
            if !available.is_empty() {
                return (available, Some((tier, score)));
            }
        }
        (Vec::new(), Some((tier, score)))
    }

    /// The model-group key under which [`Self::pick_deployment`] records its
    /// candidate count ([`CooldownStore::set_group_size`]) and keys the
    /// round-robin group counter: the requested model when it resolves by
    /// exact `model_name` match or configured alias (both operator-bounded
    /// key spaces), otherwise the wildcard group name `"*"`.
    ///
    /// A wildcard deployment (`model_name: "*"` with a matching configured
    /// model pattern) resolves **every** distinct client-requested model
    /// string, so keying group state by the raw requested string would let a
    /// client spraying `model: "x-<random>"` grow `group_sizes` /
    /// `round_robin_counters` without bound between config reloads — the
    /// only sweep of those maps ([`Self::prune_stale_store_keys`]) runs at
    /// reload. Aggregating wildcard-served requests under `"*"` bounds both
    /// maps by the configured model list. [`Self::record_failure_and_check`]
    /// mirrors this on the read side (an unknown model falls back to the
    /// `"*"` group size) so the single-deployment percentage safety still
    /// engages for a solo wildcard deployment.
    ///
    /// Returns `None` when there are no candidates: a zero-size entry reads
    /// identically to an absent one (`group_size_for` returns 0 either way),
    /// so the write is skipped — otherwise an unknown-model request would
    /// itself allocate a fresh map entry per distinct requested string.
    fn group_size_key(&self, candidates: &[&Deployment], requested: &str) -> Option<String> {
        if candidates.is_empty() {
            return None;
        }
        if candidates.iter().any(|d| d.model_name == requested)
            || self.config.model_group_alias.contains_key(requested)
        {
            return Some(requested.to_string());
        }
        Some("*".to_string())
    }

    /// Run the pre-call check pipeline over `available` candidates.
    ///
    /// Order: the checks named by `optional_pre_call_checks` in list order
    /// (duplicates in the list run once, at the first listed position; the
    /// built-in health check runs at its listed position), then the
    /// health check appended last when health routing is enabled but
    /// unlisted. Each check's input is the previous check's output. A check
    /// that errors is skipped (fail-open); a check that filters out every
    /// candidate is treated as a no-op (safety net). No-op fast path when
    /// no checks are active — the request context is not cloned and no keys
    /// are computed.
    fn run_pre_call_checks<'a>(
        &self,
        mut available: Vec<&'a Deployment>,
        ctx: &RequestContext,
        keys: &mut CooldownKeyCache<'_>,
    ) -> Vec<&'a Deployment> {
        let run_health = self.health_check.is_some();
        if !run_health && self.check_order.is_empty() {
            return available;
        }

        let mut check_ctx = ctx.clone();
        for name in &self.check_order {
            if name == "health_check" {
                if let Some(health_check) = &self.health_check {
                    self.refresh_candidate_keys(&mut check_ctx, &available, keys);
                    available = Self::apply_check(health_check, available, &check_ctx);
                }
                continue;
            }
            let Some(check) = self
                .pre_call_checks
                .iter()
                .find(|c| c.name() == name.as_str())
            else {
                // The name is listed in `optional_pre_call_checks` but no
                // check with that name was registered (a host-injected check
                // that was dropped, or a typo). Fail-open: keep routing.
                // debug, not warn: this branch is evaluated per pick, and a
                // persistent config typo must not log a warn per request at
                // high RPS — the mismatch is already surfaced once at
                // registration time by [`Router::set_pre_call_checks`]
                // (mirroring the load-time `router_config` warnings); this
                // covers hosts that never inject checks at all.
                tracing::debug!(
                    event = "pre_call_check",
                    check = %name,
                    "`optional_pre_call_checks` lists an unregistered pre-call check; \
                     it will be a silent no-op"
                );
                continue;
            };
            self.refresh_candidate_keys(&mut check_ctx, &available, keys);
            available = Self::apply_check(check.as_ref(), available, &check_ctx);
        }
        if run_health && !self.check_order.iter().any(|n| n == "health_check") {
            if let Some(health_check) = &self.health_check {
                self.refresh_candidate_keys(&mut check_ctx, &available, keys);
                available = Self::apply_check(health_check, available, &check_ctx);
            }
        }
        available
    }

    /// Recompute the candidate keys a check will see (aligned by position
    /// with `available`). Keys must be refreshed before every check because
    /// each check may shrink (or reorder) the candidate list. Sourced from
    /// the per-pick key cache so a pick with N candidates and K active
    /// checks performs one serialization per deployment, not (K+1)×N.
    fn refresh_candidate_keys(
        &self,
        check_ctx: &mut RequestContext,
        available: &[&Deployment],
        keys: &mut CooldownKeyCache<'_>,
    ) {
        check_ctx.pre_call.candidate_keys = available.iter().map(|d| keys.key_for(d)).collect();
        check_ctx.pre_call.candidate_ids = available
            .iter()
            .map(|d| stores::deployment_key(d))
            .collect();
    }

    /// Apply one check to the candidate list with fail-open error handling
    /// and the all-filtered safety net.
    fn apply_check<'a>(
        check: &dyn PreCallCheck,
        input: Vec<&'a Deployment>,
        ctx: &RequestContext,
    ) -> Vec<&'a Deployment> {
        let before = input.len();
        match check.filter_deployments(&input, ctx) {
            Ok(filtered) => {
                if filtered.is_empty() && before > 0 {
                    // Safety net: a check must never turn "deployments
                    // exist" into "no deployment available".
                    tracing::warn!(
                        event = "pre_call_check",
                        check = check.name(),
                        filtered_before = before,
                        restored = true,
                        "check filtered out every candidate; restoring its input (safety net)"
                    );
                    input
                } else {
                    tracing::debug!(
                        event = "pre_call_check",
                        check = check.name(),
                        filtered_before = before,
                        filtered_after = filtered.len(),
                        "pre-call check applied"
                    );
                    filtered
                }
            }
            Err(e) => {
                // Fail-open: a broken check must not break routing.
                tracing::warn!(
                    event = "pre_call_check",
                    check = check.name(),
                    error = %e,
                    "pre-call check failed; keeping candidates (fail-open)"
                );
                input
            }
        }
    }

    /// Resolve the effective routing strategy for a specific model.
    /// Checks routing_groups first, then falls back to the default strategy.
    fn strategy_for_model(&self, model: &str) -> &dyn RoutingStrategy {
        if let Some(strategy) = self.routing_group_strategies.get(model) {
            return strategy.as_ref();
        }
        &*self.strategy
    }

    /// Pick a deployment index (same as [`pick_deployment`](Self::pick_deployment)
    /// but returns the index into the original slice).
    pub fn pick_deployment_index(
        &self,
        deployments: &[Deployment],
        ctx: &RequestContext,
    ) -> Option<usize> {
        let picked = self.pick_deployment(deployments, ctx)?;
        deployments.iter().position(|d| std::ptr::eq(d, picked))
    }

    /// Checked counterpart of [`Self::pick_deployment_index`]. Selector
    /// errors must be propagated by the host, rather than treated as an
    /// empty pool eligible for unconstrained cross-group failover.
    pub fn try_pick_deployment_index(
        &self,
        deployments: &[Deployment],
        ctx: &RequestContext,
    ) -> crate::Result<Option<usize>> {
        Ok(self
            .try_pick_deployment(deployments, ctx)?
            .and_then(|picked| deployments.iter().position(|d| std::ptr::eq(d, picked))))
    }

    /// Resolve the effective cooldown duration (seconds) for a model,
    /// preferring a routing-group override over the global default.
    fn cooldown_time_secs_for(&self, model: &str) -> u64 {
        self.config
            .routing_groups
            .get(model)
            .and_then(|g| g.cooldown_time_secs)
            .unwrap_or(self.config.cooldown_time_secs)
    }

    /// Resolve the effective cooldown duration (seconds) for a model — exposed
    /// for tests that need to verify per-group override resolution without
    /// waiting on real time.
    #[cfg(test)]
    pub fn cooldown_time_secs_for_model(&self, model: &str) -> u64 {
        self.cooldown_time_secs_for(model)
    }

    /// Mark a deployment as cooling down for the effective cooldown duration
    /// (model-group override if set, otherwise the global default).
    pub fn mark_cooldown_for(&self, key: &str, model: &str) {
        self.cooldown
            .mark(key, Duration::from_secs(self.cooldown_time_secs_for(model)));
    }

    /// Mark a deployment as cooling down for `cooldown_time_secs`.
    pub fn mark_cooldown(&self, key: &str) {
        self.cooldown
            .mark(key, Duration::from_secs(self.config.cooldown_time_secs));
    }

    /// Resolve the allowed-fails limit for an HTTP status. Precedence:
    /// 1. [`RetryPolicy`] (exception-type policy) — only for statuses it
    ///    covers (429/408/5xx/400).
    /// 2. `allowed_fails_policy` — per-status map.
    /// 3. `allowed_fails` — global.
    ///
    /// Returns `None` when no policy covers the status. When an exception
    /// retry policy is configured, [`Self::record_failure_and_check`] falls
    /// back to `num_retries` as the final tier — for cooldown-worthy
    /// statuses only (server errors, 429, 408, 401, 404); other client-error
    /// 4xx return `NoPolicy` and are not counted.
    pub fn allowed_fails_limit_for(&self, status: u16) -> Option<u32> {
        if let Some(policy) = &self.config.exception_retry_policy {
            if let Some(limit) = policy.limit_for_status(status) {
                return Some(limit);
            }
        }
        if let Some(policy) = &self.config.allowed_fails_policy {
            if let Some(limit) = policy.get(&status.to_string()) {
                return Some(*limit);
            }
        }
        self.config.allowed_fails
    }

    /// Record a failed attempt for a deployment key and evaluate the
    /// allowed-fails policy. The failure count lives for the effective
    /// cooldown window of `model` (failures expire after the window).
    ///
    /// A `cooldown_time_secs` of 0 would expire failure counts instantly and
    /// silently disable the allowed-fails policy, so the counting window
    /// falls back to the 60-second default in that case. (The zero cooldown
    /// itself still applies to `mark_cooldown*`, where it means "cool for
    /// zero seconds".)
    ///
    /// When no policy covers `status`, the failure is **not** counted and
    /// [`AllowedFailDecision::NoPolicy`] is returned so callers keep the
    /// existing retry/cooldown behavior — unless an exception retry policy
    /// ([`RouterConfig::exception_retry_policy`]) is configured, in which
    /// case `num_retries` acts as the final fallback tier for
    /// cooldown-worthy statuses (server errors, 429, 408, 401, 404 —
    /// the reference `_is_cooldown_required` set). Other client-error 4xx are
    /// never counted implicitly: an abusive client cannot cool a healthy
    /// deployment for generic 4xx (mirroring the client-error exclusion of
    /// the percentage path below). Explicit per-status opt-ins
    /// (`allowed_fails_policy` entries, `RetryPolicy` categories naming the
    /// status) override the exclusion.
    ///
    /// After the per-status check, the percentage-based cooldown
    /// ([`RouterConfig::failure_threshold_percent`]) is also evaluated: when
    /// the sliding-window failure ratio exceeds the threshold (and the
    /// sample size is sufficient), the decision is upgraded to
    /// [`AllowedFailDecision::LimitReached`]. Single-deployment model groups
    /// are exempt (never cool on percentage).
    ///
    /// Only failures that reflect on the deployment's own health move the
    /// sliding-window ratio: retryable/server statuses (5xx, 429, 408),
    /// timeouts, and network errors (status 0). Client-error statuses (other
    /// 4xx) are excluded — an abusive client spraying 401s at a healthy
    /// deployment must not cool it via the percentage path (which would then
    /// cascade the bad client across every deployment in the group). The
    /// policy-gated per-status counting is unaffected.
    pub fn record_failure_and_check(
        &self,
        key: &str,
        status: u16,
        model: &str,
    ) -> AllowedFailDecision {
        // H8: record the failure outcome in the sliding window only when the
        // status reflects on the deployment's own health. Client errors leave
        // the ratio untouched (the window still counts, so existing outcomes
        // stay meaningful).
        let (win_failures, win_total) = if is_server_reflecting_status(status) {
            self.cooldown.record_failure_outcome(key)
        } else {
            self.cooldown.outcome_counts(key)
        };

        // H5: resolve the limit. When an exception retry policy is configured,
        // fall back to num_retries — but only for statuses the reference considers
        // cooldown-worthy (`_is_cooldown_required`, cooldown_handlers.py:
        // server errors, 429, 408, 401, 404). Other client-error 4xx are the
        // client's fault, not the deployment's: counting them (cool dep A →
        // the abusive client hits dep B → cool B → …) contradicts the
        // client-error exclusion the percentage path applies one screen up.
        // Explicit opt-ins (a `RetryPolicy` category naming the status, or an
        // `allowed_fails_policy` entry for it) still count, mirroring
        // the reference explicit allowed-fails policy check.
        let mut limit = self.allowed_fails_limit_for(status);
        if limit.is_none()
            && self.config.exception_retry_policy.is_some()
            && cooldown_worthy_status(status)
        {
            limit = Some(
                self.config
                    .routing_groups
                    .get(model)
                    .and_then(|g| g.num_retries)
                    .unwrap_or(self.config.num_retries),
            );
        }

        let decision = match limit {
            None => AllowedFailDecision::NoPolicy,
            Some(limit) => {
                let window_secs = match self.cooldown_time_secs_for(model) {
                    0 => 60,
                    secs => secs,
                };
                let count =
                    self.cooldown
                        .record_failure(key, status, Duration::from_secs(window_secs));
                // Failure-policy parity (`should_cooldown_based_on_allowed_fails_policy`,
                // `router_utils/cooldown_handlers.py:542`, with the
                // `updated_fails > allowed_fails` comparison at `:586`): the limit is
                // the number of failures *tolerated* within the window; the
                // deployment cools only on the (N+1)th failure (`count > limit`),
                // NOT the Nth (`count >= limit`). With `allowed_fails: 2` the
                // deployment tolerates 2 failures and cools on the 3rd, matching
                // upstream semantics for an operator porting an existing config.
                if count > limit {
                    AllowedFailDecision::LimitReached { count, limit }
                } else {
                    AllowedFailDecision::Tolerated { count, limit }
                }
            }
        };

        // H8: percentage-based cooldown — additive check after per-status.
        if let Some(threshold) = self.config.failure_threshold_percent {
            if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
                let min_req = self.config.failure_threshold_min_requests.unwrap_or(1);
                // `0` means "unknown" (no entry). Wildcard-served models
                // record their group size under the wildcard group name
                // (see [`Self::group_size_key`]), so fall back to it — this
                // keeps the single-deployment safety engaged for a solo
                // wildcard deployment.
                let group_size = match self.cooldown.group_size_for(model) {
                    0 => self.cooldown.group_size_for("*"),
                    n => n,
                };
                // Single-deployment safety: never cool on percentage when the
                // model group has only one deployment.
                if win_total >= min_req && group_size != 1 {
                    let ratio = win_failures as f64 / win_total as f64;
                    if ratio > threshold {
                        // Report a same-domain (count, limit) pair so telemetry
                        // can compare them directly: `count` is sliding-window
                        // failures (any status), so `limit` is the threshold-
                        // derived sliding-window limit too — NOT the per-status
                        // policy limit (which counts a different domain and
                        // would make `count >= limit` misleading). The limit
                        // uses `ceil` (not `floor`) so it is conservative at
                        // float boundaries: `ratio > threshold` firing always
                        // implies `count >= limit` here. (`floor` would round
                        // the threshold down and could report `count == limit`
                        // in a `LimitReached` event even when the decision path
                        // and the rounded limit straddle the exact boundary
                        // differently under IEEE-754.)
                        let threshold_limit = ((threshold * win_total as f64).ceil() as u32).max(1);
                        return AllowedFailDecision::LimitReached {
                            count: win_failures,
                            limit: threshold_limit,
                        };
                    }
                }
            }
        }

        decision
    }

    /// Clear all failure counts for a deployment key (called on success). Also
    /// records a success outcome in the sliding window used by percentage-based
    /// cooldown so healthy traffic dilutes the failure ratio.
    pub fn clear_failure_counts(&self, key: &str) {
        self.cooldown.record_success_outcome(key);
        self.cooldown.clear_failure_counts(key);
    }

    /// The stable identity key for a deployment, used by the cooldown,
    /// health, and bandit stores. It is a pure function of the deployment
    /// (see [`stable_key`]), so:
    /// - a config hot-reload that reorders or inserts deployments keeps each
    ///   deployment's state attached to the deployment, never to a list slot
    ///   (the old positional `"{model_name}:{index}"` scheme misattributed
    ///   cooldowns when the list reordered), and
    /// - deployments that differ in endpoint/credentials (`api_base`,
    ///   `api_key`, ...) get distinct keys even when they share a
    ///   `model_name` and provider model, so a failing deployment cools only
    ///   itself, never its sibling backend.
    pub fn cooldown_key(&self, deployment: &Deployment) -> String {
        stable_key(deployment)
    }

    /// Get the fallback model names for a given primary model.
    ///
    /// A routing-group override only wins when one of its rules has
    /// `primary == model`; a group that lists fallback rules for *other*
    /// primaries falls through to the global chain rather than silently
    /// returning an empty list (fail-open on a config typo, matching the
    /// fallthrough behaviour of [`Self::retries_for_status_model`] and
    /// `Self::cooldown_time_secs_for`). Rules with an empty `fallbacks`
    /// list are treated as absent everywhere: a half-written entry
    /// (`{special: []}`) must not shadow the broader chain that would
    /// otherwise serve the model.
    pub fn fallbacks_for(&self, model: &str) -> Vec<&str> {
        if let Some(group) = self.config.routing_groups.get(model) {
            if let Some(rule) = group
                .fallbacks
                .iter()
                .find(|r| r.primary == model && !r.fallbacks.is_empty())
            {
                return rule.fallbacks.iter().map(String::as_str).collect();
            }
        }
        self.config
            .fallbacks
            .iter()
            .find(|r| r.primary == model && !r.fallbacks.is_empty())
            .map(|r| r.fallbacks.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Get the default fallback chain for an UNKNOWN model group (no
    /// deployments resolve for `model`). An exact-primary rule (dict form)
    /// wins over the wildcard `"*"` rule (list form); a rule with an empty
    /// `fallbacks` list is treated as absent (it falls through to the
    /// wildcard chain instead of suppressing it). Callers must NOT use
    /// this for a known model whose deployments all cooled — that case
    /// belongs to [`Router::fallbacks_for`].
    pub fn default_fallbacks_for(&self, model: &str) -> Vec<&str> {
        self.config
            .default_fallbacks
            .iter()
            .find(|r| r.primary == model && !r.fallbacks.is_empty())
            .or_else(|| {
                self.config
                    .default_fallbacks
                    .iter()
                    .find(|r| r.primary == "*" && !r.fallbacks.is_empty())
            })
            .map(|r| r.fallbacks.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Get context-window fallback model names for a given primary model.
    /// Empty-`fallbacks` rules are treated as absent (see
    /// [`Self::fallbacks_for`]).
    pub fn context_window_fallbacks_for(&self, model: &str) -> Vec<&str> {
        if let Some(group) = self.config.routing_groups.get(model) {
            if let Some(rule) = group
                .context_window_fallbacks
                .iter()
                .find(|r| r.primary == model && !r.fallbacks.is_empty())
            {
                return rule.fallbacks.iter().map(String::as_str).collect();
            }
        }
        self.config
            .context_window_fallbacks
            .iter()
            .find(|r| r.primary == model && !r.fallbacks.is_empty())
            .map(|r| r.fallbacks.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Get content-policy fallback model names for a given primary model.
    /// Empty-`fallbacks` rules are treated as absent (see
    /// [`Self::fallbacks_for`]).
    pub fn content_policy_fallbacks_for(&self, model: &str) -> Vec<&str> {
        if let Some(group) = self.config.routing_groups.get(model) {
            if let Some(rule) = group
                .content_policy_fallbacks
                .iter()
                .find(|r| r.primary == model && !r.fallbacks.is_empty())
            {
                return rule.fallbacks.iter().map(String::as_str).collect();
            }
        }
        self.config
            .content_policy_fallbacks
            .iter()
            .find(|r| r.primary == model && !r.fallbacks.is_empty())
            .map(|r| r.fallbacks.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Resolve the per-status retry count from the effective retry policy.
    /// A routing-group override retry_policy takes precedence; missing
    /// statuses fall back to the global retry_policy, then the structured
    /// exception policy, then effective `num_retries` (group first, then global).
    pub fn retries_for_status(&self, status: u16) -> u32 {
        self.retries_for_status_model(status, "")
    }

    /// Resolve the per-status retry count for a specific model.
    /// Same precedence as [`retries_for_status`](Self::retries_for_status)
    /// but consults the model's routing-group override first.
    pub fn retries_for_status_model(&self, status: u16, model: &str) -> u32 {
        self.retry_policy_for_status_model(status, model)
            .unwrap_or_else(|| {
                self.config
                    .routing_groups
                    .get(model)
                    .and_then(|group| group.num_retries)
                    .unwrap_or(self.config.num_retries)
            })
    }

    /// Explicit per-status policy, before request/deployment/default retry budgets.
    pub fn retry_policy_for_status_model(&self, status: u16, model: &str) -> Option<u32> {
        let group = self.config.routing_groups.get(model);
        let key = status.to_string();
        if let Some(g) = group {
            if let Some(n) = g.retry_policy.get(&key) {
                return Some(*n);
            }
        }
        if let Some(n) = self.config.retry_policy.get(&key) {
            return Some(*n);
        }
        if let Some(count) = self
            .config
            .exception_retry_policy
            .as_ref()
            .and_then(|policy| policy.limit_for_status(status))
        {
            return Some(count);
        }
        None
    }
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("config", &self.config)
            .field("cooldown", &self.cooldown)
            .finish()
    }
}

/// Per-pick memo for [`stable_key`] computations.
///
/// [`stable_key`] serializes the deployment's full `oxllm_params` to JSON
/// and hashes it — a pure function of an immutable [`Deployment`], yet a
/// single [`Router::pick_deployment`] call needs each candidate's key up to
/// `2 + K` times (cooldown filter, each of K active pre-call checks,
/// session-affinity sticky match, adaptive candidate keys), recomputing the
/// serialization every time. This cache computes each distinct candidate's
/// key once per pick and serves the rest as `String` clones (the key is
/// `"{model_name}:{16 hex chars}"` — far cheaper to clone than the params
/// are to serialize).
///
/// Keys are memoized by deployment address. This is sound without cache
/// invalidation because the cache lives only for the duration of one
/// `pick_deployment` call, whose candidate references all point into the
/// caller-owned `deployments` slice (filters only ever remove entries); two
/// live deployments can never share an address.
struct CooldownKeyCache<'r> {
    router: &'r Router,
    map: HashMap<usize, String>,
}

impl<'r> CooldownKeyCache<'r> {
    fn new(router: &'r Router, capacity: usize) -> Self {
        Self {
            router,
            map: HashMap::with_capacity(capacity),
        }
    }

    fn key_for(&mut self, deployment: &Deployment) -> String {
        let addr = std::ptr::from_ref(deployment) as usize;
        if let Some(key) = self.map.get(&addr) {
            return key.clone();
        }
        let key = self.router.cooldown_key(deployment);
        self.map.insert(addr, key.clone());
        key
    }
}

/// Whether an HTTP status reflects the deployment's own health rather than a
/// client-side mistake, for the percentage-based cooldown ratio. Client
/// errors (4xx, except 429 rate-limit and 408 timeout, which do indicate a
/// server-side condition) are excluded — an abusive client spraying 4xx at a
/// healthy deployment must not cool it. Everything else — status 0 (the
/// network-error bucket), 1xx/2xx/3xx oddities, and 5xx — reflects the
/// deployment, matching the reference `_is_cooldown_required` catch-all
/// ("should cool down for all other errors", cooldown_handlers.py:245-248).
fn is_server_reflecting_status(status: u16) -> bool {
    status == 0 || !(400..500).contains(&status) || status == 429 || status == 408
}

/// Whether an HTTP status may cool a deployment through the implicit
/// `num_retries` fallback tier of [`Router::record_failure_and_check`]:
/// server-reflecting statuses plus the two 4xx the reference
/// `_is_cooldown_required` (cooldown_handlers.py:233-247) singles out as
/// deployment-fault (401 bad credentials, 404 model gone). All other 4xx
/// are client-caused and must not count toward cooldown.
fn cooldown_worthy_status(status: u16) -> bool {
    is_server_reflecting_status(status) || matches!(status, 401 | 404)
}

/// Priority-order tiers (`oxllm_params.order`): lower
/// `order` = higher priority. When at least one candidate sets `order`,
/// only candidates in the lowest-order tier survive; deployments without
/// `order` count as the lowest-priority tier (`u32::MAX`), matching
/// the reference, where unset-order deployments become eligible only once every
/// ordered deployment has cooled down.
///
/// Composes with cooldowns: the filter runs after the cooldown filter in
/// [`Router::pick_deployment`], so once every deployment of the top tier
/// is cooling, the next pick sees the next tier — each tier effectively
/// gets the retry budget before escalation, with no retry-loop changes.
/// No-op (returns the input unchanged) when no candidate sets `order`,
/// preserving existing behavior and the hot path.
fn order_tier_filter(available: Vec<&Deployment>) -> Vec<&Deployment> {
    if !available.iter().any(|d| d.oxllm_params.order.is_some()) {
        return available;
    }
    // The `any` guard above implies a non-empty list, so `min` always
    // yields `Some` here; fall back to `u32::MAX` instead of panicking —
    // an empty list filters down to nothing either way.
    let min_order = available
        .iter()
        .map(|d| d.oxllm_params.order.unwrap_or(u32::MAX))
        .min()
        .unwrap_or(u32::MAX);
    available
        .into_iter()
        .filter(|d| d.oxllm_params.order.unwrap_or(u32::MAX) == min_order)
        .collect()
}

/// Whether a wildcard deployment's configured `oxllm_params.model` pattern
/// is provider-compatible with the requested model.
///
/// - A bare configured model that is exactly `*` matches any requested
///   model. Any other bare configured model (no `/`) — including the empty
///   string — is an exact model match: it serves only a requested model
///   equal to it. A config typo dropping the provider prefix
///   (`model: "gpt-4o"`, or a broken `model: ""`) must not silently turn a
///   scoped wildcard into a global catch-all that serves (and bills)
///   cross-provider requests.
/// - A `provider/*` configured model matches only requested models whose own
///   provider prefix equals `provider`. A requested model without a `/`
///   defaults to the `openai` provider, so `openai/*`
///   matches bare names like `gpt-4o` while `anthropic/*` does not.
/// - A partial pattern whose model part carries a `*` (e.g. `groq/llama-*`,
///   `openai/gpt-4*`) matches same-provider requests whose model name matches
///   the glob: `groq/llama-*` matches `groq/llama-3.1-70b` but not
///   `groq/mistral-7b`. The provider prefix must still match, so an
///   `openai/gpt-4*` wildcard never serves an `anthropic/gpt-4...` request.
/// - A fully-qualified configured model with no `*` anywhere
///   (e.g. `openai/gpt-4o`) is an exact provider-model match only — it never
///   becomes a provider-wide catch-all, matching the bare-name rule. A config
///   typo putting `model: "openai/gpt-4o"` on a `model_name: "*"` deployment
///   must not silently serve (and bill) every `openai/*` request.
fn wildcard_model_matches(configured_model: &str, requested_model: &str) -> bool {
    let Some((cfg_provider, cfg_pattern)) = configured_model.split_once('/') else {
        // Bare configured model (no `/`): only a literal `*` matches
        // anything; any other bare name (including the empty string, a
        // near-certain config breakage) is an exact match.
        return configured_model == "*" || configured_model == requested_model;
    };
    // A requested model without a `/` defaults to the `openai` provider;
    // the whole string is then the model name.
    let (req_provider, req_model) = match requested_model.split_once('/') {
        Some((p, m)) => (p, m),
        None => ("openai", requested_model),
    };
    if cfg_provider != req_provider {
        return false;
    }
    // `provider/*` — any model under this provider.
    if cfg_pattern == "*" {
        return true;
    }
    // Partial pattern (the model part carries a `*`): glob the model name.
    if cfg_pattern.contains('*') {
        return glob_model_match(cfg_pattern, req_model);
    }
    // No wildcard in the model part: an exact provider-model match only.
    cfg_pattern == req_model
}

/// Minimal `*` glob for model-name matching. A `*` matches any (possibly
/// empty) sequence; multiple `*` are supported (e.g. `gpt-*-mini`). Mirrors
/// the server's `glob_match` but lives in core so the router has no server
/// dependency. An exact match (no `*`) requires equality.
/// Two-pointer glob with star backtracking (the classic O(p·t) worst-case
/// algorithm): when a `*` is found, remember the star position and the text
/// position after the last partial match, and on mismatch retry the middle
/// segment one character later in the text. This correctly handles
/// multi-`*` patterns where a greedy leftmost placement of the middle parts
/// fails (e.g. `*ab*ba` matching `aba`).
fn glob_model_match(pattern: &str, text: &str) -> bool {
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    let (pb, tb) = (pattern.as_bytes(), text.as_bytes());
    while t < tb.len() {
        if p < pb.len() && pb[p] == tb[t] {
            p += 1;
            t += 1;
        } else if p < pb.len() && pb[p] == b'*' {
            star = Some(p);
            mark = t;
            p += 1;
        } else if let Some(s) = star {
            p = s + 1;
            mark += 1;
            t = mark;
        } else {
            return false;
        }
    }
    while p < pb.len() && pb[p] == b'*' {
        p += 1;
    }
    p == pb.len()
}

#[cfg(test)]
mod tests;
