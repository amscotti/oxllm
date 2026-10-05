# Routing Features Implementation Plan

> **Historical implementation plan.** Current routing is documented in
> [architecture](../architecture.md#routing) and implemented in core/server
> routing modules and tests. Planned file names, estimates and completion
> markers below are retained for context, not a current backlog or validation
> result. In particular, pre-call checks are synchronous filters.

Based on LiteLLM analysis. Ordered by effort, smallest first.

---

## Batch 1: Prompt Caching Deployment Check + Allowed Fails Policy ✅ Complete (2026-07-23)

### 1a. Prompt Caching Deployment Check

**What it does:** When a request's prompt is large enough for provider-side caching,
pin the request to the deployment that already has the cache warm.

**LiteLLM behavior:**
- Pre-call: hash prompt prefix → look up model_id in cache
- Post-call: write prefix → model_id mapping
- Uses DualCache (in-memory + Redis)

**Our fit:**
- We have `DualCache` for cross-replica state
- We have `SessionAffinity` using FNV-1a hash (similar pattern)
- We can add a `PromptCachingCheck` in the pipeline

**Implementation:**
- New type: `PromptCachingCheck` in `oxllm-server/src/pre_call_checks/`
- Pre-call: `fnv1a64(system_prompt_or_first_message)` → `cache.get("pcache:{hash}")` → if found, prefer that deployment
- Post-call: write `cache.set("pcache:{hash}", model_id, ttl=3600)`
- Config: `prompt_caching_min_tokens: Option<u32>` per deployment (min tokens to trigger)
- Log: `tracing::debug!(event = "prompt_cache_pinned", model_id = ..., hash = ...)`

**Files:**
- `crates/oxllm-server/src/pre_call_checks/mod.rs` (new dir)
- `crates/oxllm-server/src/pre_call_checks/prompt_caching.rs`
- Wire into `orchestration/chat.rs` retry loop
- Config in `config/types.rs`

**Tests:** min threshold, pinning, non-pinning when below threshold, empty candidates

### 1b. Allowed Fails Policy

**What it does:** Custom thresholds per error type before cooling a deployment.
E.g., "allow 3 failures on 500s, but cool immediately on 401s"

**LiteLLM behavior:**
- `allowed_fails: u32` — global threshold
- `allowed_fails_policy: HashMap<String, u32>` — per-status thresholds
- Counts failures per deployment in a TTL window

**Our fit:**
- We have `CooldownStore` with global `cooldown_time_secs`
- We have `retries_for_status` per-status retry counts
- Allowed fails adds per-status fail thresholds

**Implementation:**
- Add to `RouterConfig`: `allowed_fails: Option<u32>`, `allowed_fails_policy: Option<HashMap<String, u32>>`
- Add `FailureCounter` to `CooldownStore`: count failures per deployment per status
- In the retry loop (`orchestration/chat.rs`), after each failed attempt:
  - If status in `allowed_fails_policy` and count < policy limit → retry again
  - If status in `allowed_fails_policy` and count >= policy limit → mark cooldown
  - If not in policy → use existing cooldown logic
- Log: `tracing::debug!(event = "allowed_fail", deployment = ..., status = ..., count = ..., limit = ...)`

**Files:**
- `crates/oxllm-core/src/router/cooldown.rs` — extend with failure counting
- `crates/oxllm-core/src/router/types.rs` — add config fields
- `crates/oxllm-core/src/router/mod.rs` — integrate into retry loop

**Tests:** threshold per status, global threshold, no threshold (existing behavior), failure counting

---

## Batch 2: Budget Limiter + Health-Check Routing ✅ Complete (2026-07-23)

### 2a. Budget Limiter (per-deployment + per-provider)

**What it does:** Filter deployments whose provider or deployment spend exceeds a budget cap.

**LiteLLM behavior:**
- Provider budgets: `{openai: {budget_limit: 100, time_period: "1d"}}`
- Deployment budgets: `max_budget`/`budget_duration` on `oxllm_params`
- Tag budgets: per-request tag budgets (Enterprise)

**Our fit:**
- We have `model_max_budget` (global per-model spend cap)
- We have per-key/team/org/project budgets
- Budget limiter adds per-deployment and per-provider budgets

**Implementation:**
- Add to `OxllmParams`: `max_budget: Option<f64>`, `budget_duration: Option<String>` (per-deployment)
- Add to `RouterSettings`: `provider_budget_config: Option<HashMap<String, ProviderBudget>>`
- In the retry loop, before picking: check deployment budget (`spend:dep:{id}`) and provider budget (`spend:provider:{slug}`)
- Post-call: increment spend counters
- Log: `tracing::info!(event = "budget_exceeded", scope = "deployment", id = ..., spend = ..., limit = ...)`

**Files:**
- `crates/oxllm-core/src/deployment.rs` — add fields
- `crates/oxllm-server/src/config/types.rs` — add provider budget config
- `crates/oxllm-server/src/orchestration/chat.rs` — integrate into retry loop
- `crates/oxllm-server/src/hooks/budget.rs` — extend for deployment/provider budgets

**Tests:** deployment budget enforcement, provider budget enforcement, budget window reset, passthrough when no budgets set

### 2b. Health-Check Routing

**What it does:** Track per-deployment health and skip unhealthy ones.

**LiteLLM behavior:**
- Background health checks mark deployments healthy/unhealthy
- `enable_health_check_routing: bool`
- Staleness threshold for health state
- Safety net: if all unhealthy, use all deployments

**Our fit:**
- We already have `/health/readiness` endpoint
- We can add per-deployment health tracking (separate from cooldown)
- Cooldown is for error rates; health is for connectivity

**Implementation:**
- New type: `HealthState` in `oxllm-core/src/router/health.rs`
- Track: `{deployment_id: {is_healthy: bool, timestamp, reason}}`
- Mark unhealthy on connection errors, unhealthy after `health_check_failure_threshold` consecutive failures
- Auto-recover after `health_check_recovery_time`
- In `pick_deployment`, filter unhealthy deployments
- Safety net: if all unhealthy, return all deployments
- Config: `enable_health_check_routing: bool`, `health_check_staleness_threshold: u64`
- Log: `tracing::debug!(event = "health_check", deployment = ..., is_healthy = ...)`

**Files:**
- `crates/oxllm-core/src/router/health.rs` (new)
- `crates/oxllm-core/src/router/mod.rs` — integrate
- `crates/oxllm-server/src/config/types.rs` — add config

**Tests:** unhealthy filtering, all-unhealthy safety net, staleness threshold, auto-recovery

---

## Batch 3: Pre-call Checks Framework + Complexity Router ✅ Complete (2026-07-23)

### 3a. Pre-call Checks Framework

**What it does:** Plugin architecture for routing pre-checks.

**LiteLLM behavior:**
- `optional_pre_call_checks: ["prompt_caching", "router_budget_limiting", ...]`
- Each check filters the candidate deployment list
- Checks run in order before the strategy pick

**Our fit:**
- We have the `Hook` trait for post-call checks
- We can add a `PreCallCheck` trait for routing pre-checks
- Prompt caching and budget limiter become `PreCallCheck` implementations

**Implementation:**
- New trait: `PreCallCheck` in `oxllm-core/src/router/pre_call_check.rs`
- `fn filter_deployments(&self, candidates: &[&Deployment], ctx: &RequestContext) -> Result<Vec<&Deployment>, oxllm_core::Error>`
- `Router` holds `Vec<Arc<dyn PreCallCheck>>`
- In `pick_deployment`, run all checks in order before strategy pick
- Config: `optional_pre_call_checks: Vec<String>` in `RouterSettings`
- Log: `tracing::debug!(event = "pre_call_check", check = ..., filtered = ..., remaining = ...)`

**Files:**
- `crates/oxllm-core/src/router/pre_call_check.rs` (new)
- `crates/oxllm-core/src/router/mod.rs` — integrate
- `crates/oxllm-server/src/config/types.rs` — add config

**Tests:** check ordering, filtering, multiple checks, no checks

### 3b. Complexity Router

**What it does:** Classify request complexity (SIMPLE/MEDIUM/COMPLEX/REASONING) and route to appropriate model pool.

**LiteLLM behavior:**
- 7-dimension weighted keyword scoring
- Optional LLM classifier for complex cases
- Tier boundaries map to model pools
- Adaptive bandit for pool selection

**Our fit:**
- New `RoutingStrategyKind::ComplexityRouter`
- `ComplexityRouter` implementing `RoutingStrategy`
- Keyword-based scoring (no LLM call needed for MVP)

**Implementation:**
- New strategy: `ComplexityRouter` in `oxllm-core/src/router/complexity.rs`
- Dimensions: token count, code presence, reasoning markers, technical terms, simple indicators
- Tier boundaries from config
- Config: `complexity_router_config: Option<ComplexityRouterConfig>` with tier boundaries and per-tier model pools
- Log: `tracing::info!(event = "complexity_route", tier = ..., score = ..., model = ...)`

**Files:**
- `crates/oxllm-core/src/router/complexity.rs` (new)
- `crates/oxllm-core/src/router/types.rs` — add config + variant
- `crates/oxllm-core/src/router/mod.rs` — integrate

**Tests:** tier classification, boundary values, model pool selection, simple/complex/reasoning

---

## Batch 3.5 (done in fix cycle)

Follow-ups landed alongside the Batch 1-3 review cycle, outside the original
batch plan:

- **Weighted simple-shuffle**: the default `simple_shuffle` strategy is now
  weighted — selection probability is proportional to
  `oxllm_params.weight`, falling back to `rpm` when no explicit
  weight is set, then to 1 (uniform) — matching LiteLLM's default behavior.
- **`order` field deserialization**: `oxllm_params.order` now parses from
  config as a typed field; priority/`order`-based routing is planned on top
  of it (implemented in Batch 3.6).
- **Strategy factory registry**: strategies are constructed via
  `strategy_factory(kind)` + `StrategyDeps` (`router/strategy.rs`) instead
  of a central match arm in the picker, so new strategies (e.g. the Batch 4
  adaptive router) plug in without editing `Router::build_inner`.

## Batch 3.6: Priority-Order Tiers + Default Fallbacks ✅ Complete (2026-07-24)

- **Priority-order routing (`oxllm_params.order`)**: lower `order` =
  higher priority. `pick_deployment` now applies an order-tier filter after
  cooldown filtering and before pre-call checks / the strategy pick: when
  any candidate sets `order`, only the lowest-order tier of healthy
  deployments is eligible (deployments without `order` count as the
  lowest-priority tier, matching LiteLLM). Cooled top-tier deployments stay
  excluded, so the next pick escalates to the next tier automatically —
  each tier effectively gets the retry budget before escalation, and the
  weighted shuffle still weights WITHIN the surviving tier. The filter is a
  no-op when no candidate sets `order`.
- **`default_fallbacks`** (`router_settings.default_fallbacks`): when the
  REQUESTED model group is unknown (no deployments resolve), the request is
  rerouted to the first chain entry that passes the caller's model ACL and
  has deployments — instead of erroring with `model_not_found`. Distinct
  from `fallbacks`, which handles failures of a KNOWN model; a known model
  whose deployments all cooled never triggers `default_fallbacks`. Accepts
  the list form (`["model-a", "model-b"]`, any unknown model) and the dict
  form (`[{"original-model": ["a"]}]`, per-model chains win over the list).
  A successful reroute stamps the `x-oxllm-fallback-used` header.

## Post-review fixes

The architecture review cycle over Batches 1-3 found and fixed:

- **Cooldown key collision**: deployments sharing a model name collided on
  the cooldown key; keys are now `{model_name}:{index}` (also used by
  prompt-cache pins and health entries).
- **Cache-hit budget drift**: a cache short-circuit no longer reserves or
  settles spend; metering hooks skip `post_success` work on cache hits.
- **Atomic router budgets**: router budget enforcement documented as
  approximate soft caps (read-only check, post-response spend recording —
  overshoot scales with concurrency); the reservation/settle path in
  `BudgetHook` remains atomic via `Cache::incr_if_below`.
- **Weighted simple-shuffle**: see Batch 3.5 above.
- **Health staleness**: health entries carry a timestamp and go stale
  (`health_check_staleness_threshold_secs`, default 60s); stale entries
  read healthy again (fail-open) so recovered deployments re-enter
  rotation without an explicit probe, and locally-synthesized timeouts (no
  response ever received) degrade health like other connection failures.

---

## Batch 4: Adaptive Router ✅ Complete (2026-07-24)

Landed per [adaptive-router-design.md](adaptive-router-design.md), with
these deviations from the design doc (all reviewed-consistent):

- **No DualCache delta-sync**: no existing router store (cooldown, latency,
  usage, health) syncs cross-replica, so the bandit store follows the same
  in-memory `Arc`-shared pattern. `BanditStore::merge_delta` is the
  documented extension point for a future delta-sync (deltas commute, so
  merge order is irrelevant); the Redis-atomic path was not built.
- **Cost normalization source**: per-token rates come from the deployment's
  `pricing_override()` or the embedded price table (`lookup_pricing`) —
  the same rates the cost hooks charge. Unpriced candidates normalize to
  1.0 (pessimistic) when any candidate is priced; the cost term is 0 when
  none are. Default `cost_weight: 0.0` (quality-only) — cost blending is
  opt-in.
- **RNG**: production randomness is a `ChaChaRng` seeded from OS entropy
  (`rand::rng()` is `!Send` and cannot live in the shared strategy); the
  test-only constructor takes any boxed `RngCore` (seeded `ChaChaRng` in
  tests).
- **Classification site**: the host classifies once per request when
  building the `RequestContext` (it needs the type to stamp the bandit
  cell); `Router::pick_deployment` reuses it via
  `PreCallData::adaptive_request_type` (the `complexity_classification`
  precedent) and classifies on demand only for standalone picks.
- **Learning granularity**: hook-level, once per request — `post_success`
  credits the serving (last-picked) deployment, `post_failure` debits the
  last-tried one. Intermediate per-attempt failures inside the retry loop
  are invisible to the bandit (cooldowns / allowed-fails handle them).
- **Logging**: per-pick and per-update logs at `tracing::debug!`, not the
  `info!` sketched above — per-request `info!` logging floods (the same
  lesson as the complexity-router log fix).

### 4a. Adaptive Router (Thompson Sampling Bandit)

**What it does:** Multi-armed bandit that learns which model is best for each request type.

**LiteLLM behavior:**
- Request-type classification (7 types via regex)
- Thompson sampling with Beta distributions
- Post-call learning from signals (satisfaction, failure, stagnation, etc.)
- Quality + cost weighted scoring

**Our fit:**
- New `RoutingStrategyKind::AdaptiveRouter`
- `AdaptiveRouter` implementing `RoutingStrategy`
- In-memory bandit cells (no Postgres — DualCache for cross-replica)

**Implementation:**
- New strategy: `AdaptiveRouter` in `oxllm-core/src/router/adaptive.rs`
- Request type classification (regex-based, 5-7 types)
- Bandit cells: `HashMap<(RequestType, String), BetaCell>`
- Thompson sampling per eligible model
- Quality + cost weighted scoring
- Post-call learning via Hook (update bandit cells from `post_success`/`post_failure`)
- Config: `adaptive_router_config` with quality/cost weights
- Log: `tracing::info!(event = "adaptive_router", request_type = ..., model = ..., alpha = ..., beta = ...)`

**Files:**
- `crates/oxllm-core/src/router/adaptive.rs` (new)
- `crates/oxllm-core/src/router/types.rs` — add variant
- `crates/oxllm-core/src/router/mod.rs` — integrate
- `crates/oxllm-server/src/hooks/adaptive_hook.rs` — post-call learning

**Tests:** Thompson sampling, request classification, learning from outcomes, cold start, convergence

**Design considerations (from the architecture review):**

- **(a) Atomic cross-replica bandit state**: Beta-cell alpha/beta updates
  must not lose counts when replicas race. Either use Redis atomic ops
  (`INCRBY`-style counter pairs per cell) or explicitly accept approximate
  delta updates via DualCache — bandits tolerate noisy feedback, so
  approximate is acceptable if documented.
- **(b) Cold-start strategy**: with zero observations per cell, Thompson
  sampling is a uniform coin flip. Use epsilon-greedy (explore with
  probability ε) or optimistic initial values, falling back to least-busy
  until N observations per cell.
- **(c) Composability with the complexity router**: adaptive and complexity
  routing are *alternative* strategies for the same pick, not layers.
  Document precedence: when `complexity_router` classifies a request into
  a tier pool, the adaptive router may pick *within* that pool, but a
  request never undergoes two top-level strategy picks.
- **(d) Classification cost on the hot path**: request-type classification
  runs per request. Share the classification with the complexity router
  (one classification per request, carried on `RequestContext`/`PreCallData`)
  instead of classifying twice.
- **(e) Strategy registration**: register the adaptive strategy through
  `strategy_factory` (`router/strategy.rs`, added in the fix cycle, see
  Batch 3.5) rather than adding dispatch logic in the picker.

See [adaptive-router-design.md](adaptive-router-design.md) for the full
design.

---

## Logging/Observability Standards

For ALL routing features, produce:

1. **Structured tracing events:**
   - `tracing::debug!(event = "routing_decision", strategy = ..., check = ..., model = ...)`
   - `tracing::info!(event = "budget_exceeded" | "health_check_failed" | "prompt_cache_pinned" | "complexity_classified" | "adaptive_learned", ...)`

2. **Response headers for observability:** ✅ shipped
   (`apply_router_headers` in `orchestration/mod.rs`; on `Ok` responses
   including streams, absent on hard-error responses):
   - `x-oxllm-router-strategy: <strategy_name>`
   - `x-oxllm-router-decision: picked | fallback | default_fallback | prompt_cache_pin`
     (as-built values; the earlier example tags — "prompt_cache_pinned",
     "complexity:simple", "adaptive:thompson" — were illustrative)

3. **Metrics:** ✅ shipped, with these as-built names/labels:
   - `oxllm_router_decisions_total{strategy, model}` (counter, `MetricsHook`)
   - `oxllm_budget_exceeded_total{scope}` (counter, `hooks/budget.rs`)
   - `oxllm_health_check_unhealthy_total{deployment}` (counter, chat orchestration)
   - `oxllm_prompt_cache_hits_total{model}` (counter, `CacheHook`)
   - `oxllm_complexity_tier_total{tier}` (counter, chat orchestration)
   - `oxllm_adaptive_alpha{request_type, model}`, `oxllm_adaptive_beta{request_type, model}` (gauges, `AdaptiveHook`)

## Agent Assignments

| Batch | Agent | Scope | Est. Tests |
|-------|-------|-------|-----------|
| 1 | Agent A | Prompt caching check + Allowed fails policy | ~20 |
| 2 | Agent B | Budget limiter (deployment + provider) + Health-check routing | ~25 |
| 3 | Agent C | Pre-call checks framework + Complexity Router | ~30 |
| 4 | Agent D | Adaptive Router (bandit + classification + learning) | ~40 |
