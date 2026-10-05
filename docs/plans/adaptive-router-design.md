# Adaptive Router Design (Batch 4)

> **Historical design proposal.** The adaptive router and learning hook now
> exist. The implemented `BanditStore` is per-process; the distributed delta
> sync / Redis-atomic tiers proposed below are not a current feature guarantee.
> See [architecture](../architecture.md#routing), `router/adaptive.rs` and
> `hooks/adaptive_hook.rs` for current behavior and tests.

A Thompson-sampling multi-armed bandit that learns which deployment is best
for each request type, from post-call outcomes. This doc is the design
contract for Batch 4 of [routing-features-plan.md](routing-features-plan.md);
it reflects the architecture-review decisions made after Batches 1-3.

Planned locations:

- `crates/oxllm-core/src/router/adaptive.rs` — the `AdaptiveRouter` strategy
  (pure: no I/O, no async)
- `crates/oxllm-server/src/hooks/adaptive_hook.rs` — post-call learning
- `RouterConfig::adaptive_router_config` in
  `crates/oxllm-core/src/router/types.rs` — quality/cost weights, cold-start
  knobs, epsilon

## Bandit model

One **Beta cell per (request_type, deployment key)**:

```text
HashMap<(RequestType, String), BetaCell>   // BetaCell { alpha: f64, beta: f64 }
```

A pick samples `Beta(alpha, beta)` for every eligible candidate cell and
takes the argmax, optionally blended with cost:
`score = sample * quality_weight − normalized_cost * cost_weight`. Successes
increment `alpha`, failures increment `beta`.

## Where the state lives

Two tiers, chosen by deployment shape:

- **In-memory + DualCache sync (default)**: cells live in a `Mutex<HashMap>`
  inside the strategy, matching the other router stores (cooldown, health).
  Post-call updates are applied locally and published as **deltas**
  (`(cell, alpha_delta, beta_delta)`) through `DualCache` so sibling
  replicas merge them. Deltas commute, so merge order doesn't matter.
- **Redis-atomic (when Redis is configured)**: each cell is a counter pair
  (`bandit:{type}:{key}:alpha` / `:beta`) updated with atomic increments, so
  no update is lost under cross-replica races. Reads sample from the
  merged counters.

Decision: implement the DualCache-delta path first — bandits are robust to
noisy/late feedback, so approximate updates are acceptable and the code
stays identical for single-replica deployments. The Redis-atomic path is an
optimization, not a correctness requirement (see *Concurrency semantics*).

## Post-call learning via Hook

Learning happens in `adaptive_hook.rs`, never in the strategy (the strategy
stays pure):

- `post_success` → `alpha += reward` where `reward` starts at 1.0 and can
  later incorporate latency/quality signals.
- `post_failure` → `beta += 1`.
- Cache hits are skipped (no upstream observation), matching the metering
  hooks.

The hook needs the cell identity: the strategy stamps the chosen
`(request_type, deployment_key)` onto the `CallContext` at pick time, the
same way `deployment_pricing` is stashed for the cost hooks today.

## Request-type classification (shared, once per request)

Classification is the most expensive part of an adaptive pick, and the
complexity router already pays a similar cost. **One classification per
request**, following the existing `PreCallData` pattern: the host (or the
complexity candidate expansion) classifies once and carries the result on
`RequestContext::pre_call` (see `PreCallData::complexity_classification`
for the exact precedent — the complexity strategy reuses it instead of
classifying a second time). The adaptive router reads the shared
classification; it never re-tokenizes the request body.

Request types are regex/keyword-based (5-7 types: code, reasoning,
creative, qa, summarization, translation, other) — no LLM call on the hot
path.

## Precedence vs. the complexity router

Adaptive and complexity routing are **alternative strategies**, not layers.
`routing_strategy` selects one per model group (or globally). When
`complexity_router` resolves a tier pool, the pool is the candidate set; an
adaptive pick may run *within* that set only if explicitly configured —
never two top-level picks for one request.

## Cold start

With zero observations every cell is `Beta(1, 1)` — uniform. To avoid
random spraying of production traffic:

- **Epsilon-greedy exploration**: with probability ε (config, e.g. 0.1)
  pick uniformly at random; otherwise sample Thompson scores.
- **Fallback to least-busy**: until a cell has N observations (config,
  e.g. 20), defer to the least-busy pick among candidates. This reuses an
  existing, well-tested signal and gives sane behavior from the first
  request.

## Concurrency semantics

Approximate updates are acceptable. Beta-cell updates are small, frequent,
and commutative; a lost or reordered delta only slightly perturbs the
posterior, and Thompson sampling is explicitly robust to noisy feedback.
Guarantees we do keep:

- Local updates are atomic (single `Mutex` acquisition per cell batch).
- Deltas (not absolute values) cross replicas, so concurrent increments
  never overwrite each other.
- Sampling takes a consistent snapshot of the cells it reads.

## Testing strategy

- **Seeded RNG**: the strategy takes its randomness from an injectable,
  seedable source so unit tests are deterministic (the existing strategies
  use `rand::rng()`; adaptive needs a seedable `ChaChaRng` or equivalent
  behind a test-only constructor).
- **Convergence bounds**: simulation tests — a rigged environment (cell A
  succeeds 90%, cell B 50%) must converge to picking A with frequency
  above a loose bound (e.g. >80% over the last 1000 of 5000 picks), with a
  fixed seed.
- **Regression tests**: cold start (least-busy fallback below N
  observations), epsilon exploration rate, cache-hit skip, delta merge
  commutativity, classification sharing (one classification per request),
  and precedence vs. `complexity_router`.
- **e2e**: mirror `tests/router_e2e.rs` patterns — httpmock upstreams with
  per-deployment failure rates, config via `router_settings`, asserting
  traffic shifts toward the healthy/high-reward deployment.

## Known limitations

- **Chat-only learning.** Only chat traffic teaches the bandit: the chat
  orchestration is the only pick site that classifies the request type and
  stamps the bandit cell (`CallContext::adaptive_cell`) for the learning
  hook. Embeddings, passthrough, and realtime picks go through the same
  `Router::pick_deployment` but with a bare request context — they classify
  as `RequestType::Other` on demand and their outcomes never reach the
  store. When the adaptive strategy is configured globally, those endpoints
  therefore ride the least-busy cold start indefinitely (or sample the
  uniform `Beta(1, 1)` prior when `cold_start_min_observations` is 0). This
  is a deliberate safe degradation: they neither pollute chat's learned
  posteriors nor starve themselves. (Mirrors the `adaptive.rs` module
  docs.)
