# Adding a feature to oxllm

Decision guide for where a change belongs and the exact steps for each kind
of feature. Read [architecture.md](architecture.md) first — it explains the
crate layering and request path this guide builds on.

The golden rule: **pure logic goes left (core/providers), I/O goes right
(server/db).** If a feature needs both, split it.

---

## 1. Adding a new provider

**When:** a new upstream LLM API (chat, embeddings, or rerank).

Providers are pure transforms. A normal chat or embeddings provider needs no
handler or route changes; a new price shard and optional passthrough support
are data-only edits in core/server. Full checklist:
[adding-a-provider.md](adding-a-provider.md).

Short version:

1. Create `crates/oxllm-providers/src/<provider>/mod.rs`; implement
   `Provider` + the capability traits you support (`ChatProvider`,
   `EmbeddingProvider`, `RerankProvider`).
2. Add `"<provider>" = []` to `[features]` in
   `crates/oxllm-providers/Cargo.toml` (and to `default` if always built).
3. Wire `pub mod <provider>;` + a `registry.register(...)` arm in
   `crates/oxllm-providers/src/lib.rs`. OpenAI-compatible providers declare
   `"<provider>" = ["openai_compat"]`; there is no `cfg(any(...))` list to
   edit.
4. Unit tests in `src/<provider>/tests.rs` (+ `goldens/` when shapes are
   stable). Price rows in `crates/oxllm-core/data/prices/<provider>.toml` if
   new; register a new shard in `PRICE_SHARDS`.

If the provider needs non-static credentials (cloud signing), that's a host
concern: add a `CredentialResolver` in `oxllm-server/src/credentials/` keyed
by provider slug — never do I/O inside the provider module.

---

## 2. Adding a new hook

**When:** per-request metering, enforcement, or observability that must run
before/after the upstream call (budgets, rate limits, caching, tracing).

1. Create `crates/oxllm-server/src/hooks/<name>.rs` and implement
   `oxllm_core::Hook`:

   ```rust
   #[async_trait]
   impl Hook for MyHook {
       async fn pre_call(&self, ctx: &mut CallContext) -> Result<HookAction> { ... }
       async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) { ... }
       async fn post_failure(&self, ctx: &CallContext, err: &oxllm_core::Error) { ... }
   }
   ```

   - `pre_call` returns `HookAction::Continue`, `ShortCircuit(value)`
     (skip upstream), or `Reject(err)`.
   - Honor `ctx.is_master` (master key bypasses enforcement) and
     `ctx.cache_hit` (metering hooks skip `post_success` work on cache hits).

2. Export it from `hooks/mod.rs`.
3. Wire it into the chain in `state/hook_builders.rs::build_hooks`.
   **Order matters** — document why your hook sits where it does. Current
   order: RateLimit → Cache → Budget → CostTracking → Metrics, then the
   optional post-call hooks (Adaptive → GenericHttpCallback → Langfuse
   → Mlflow → Langsmith → Datadog → Helicone → Wandb → S3Log → Slack
   → PagerDuty → Email → Alerting) — full chain and config keys in
   [architecture.md](architecture.md#hook-chain).
   - Enforcement that must run even on cache hits goes before `CacheHook`.
   - Anything that reserves spend goes after `CacheHook`.
4. If the hook reserves resources in `pre_call`, implement the
   reservation/rollback pattern: reserve conservatively, settle in
   `post_success`, release in `post_failure` (see `BudgetHook`).
5. If the hook has config, add it to `config/types/` and read it in
   `build_hooks`. If it owns long-lived state (flush tasks, shared tables),
   store the `Arc` on `AppState` and preserve it across reloads with a
   `*_config_unchanged` helper in `state/reload.rs` (see `LangfuseHook` /
   `AlertState`).

---

## 3. Adding a new endpoint

**When:** a new client-facing API surface (e.g. a new `/v1/...` route).

1. **Handler**: create `crates/oxllm-server/src/handlers/<name>.rs`. Keep it
   thin: parse the request, build the `EndpointPipeline`, delegate.
2. **Route**: register in the right group under `routes/`:
   - `routes/llm.rs` — core LLM APIs (chat, embeddings, …)
   - `routes/platform.rs` — platform APIs (batches, files, assistants; feature
     `platform-apis`)
   - `routes/passthrough.rs` — raw forwarding endpoints
   - `routes/observability.rs` — health/metrics
   Auth middleware is applied to the merged group in `routes/mod.rs`, not per
   route.
3. **Pipeline integration**: if the endpoint shares the LLM request lifecycle
   (auth metering, hooks, deployment limits, `x-oxllm-*` headers), funnel
   through `pipeline::EndpointPipeline` — construct a `CallContext`, run
   `run_pre_call_hooks`, and on completion call `post_success`/`post_failure`.
   If it needs retries/fallbacks/streaming, add an orchestration module under
   `orchestration/` (chat is the reference).
4. **Provider support**: if the endpoint is a new *endpoint kind* (like
   rerank), that's a core change: add the capability trait to
   `oxllm-core/src/provider/mod.rs`, an `Endpoint` variant in
   `oxllm-core/src/endpoint.rs`, and typed accessors.
   Endpoint kinds change rarely and deliberately touch core; providers then
   opt in per-module.
5. E2E test in `tests/` (see §10).

---

## 4. Adding a new routing strategy

**When:** a new way to pick among the deployments in a model group (like
`complexity_router` or `adaptive_router`).

Strategies live in `oxllm-core` and are **pure**: a synchronous `pick`, no
I/O, no async, no env reads. Anything the pick needs that only the host can
provide arrives pre-resolved on `RequestContext` / `PreCallData`, and any
learning from outcomes happens in a server-side hook — never in the
strategy.

1. **Kind**: add a `RoutingStrategyKind` variant in
   `crates/oxllm-core/src/router/types.rs`, plus its snake_case spelling in
   `RoutingStrategyKind::as_str()` (the label spelling used for response
   headers and metric labels).
2. **Strategy**: implement `RoutingStrategy` in
   `crates/oxllm-core/src/router/<name>.rs`:

   ```rust
   impl RoutingStrategy for MyRouter {
       fn pick<'a>(
           &self,
           deployments: &[&'a Deployment],
           ctx: &RequestContext,
       ) -> Option<&'a Deployment> { ... }
   }
   ```

   Shared stores arrive as `Arc`s at construction; per-request inputs arrive
   on `RequestContext`. Export the module and the type from `router/mod.rs`.
3. **Factory**: add one arm to `strategy_factory` in
   `crates/oxllm-core/src/router/strategy.rs`, reading what you need from
   `StrategyDeps`. Router construction (`Router::build_inner`) is unchanged.
   If the strategy needs a shared store or config slice `StrategyDeps` does
   not carry, add a field to `StrategyDeps` and populate it at the single
   construction site in `Router::build_inner`.
4. **Config type** (only if the strategy has knobs): add a
   `<name>_router_config` struct in `types.rs` with a `Default` impl and a
   `validate()` that rejects bad values loudly at load — the
   `ComplexityRouterConfig::validate` / `AdaptiveRouterConfig::validate`
   precedent — plus an `Option` field on `RouterConfig`.
5. **Parsing**: wire `crates/oxllm-server/src/state/router_config.rs`:
   - a `parse_strategy_kind` arm accepting both `snake_case` and
     `kebab-case` (unknown names warn and fall back to `simple_shuffle`);
   - for a config struct: an `Option` field on `RouterSettings`
     (`config/types/`), validated and copied into `RouterConfig` in
     `build_router_config` (the `complexity_router_config` /
     `adaptive_router_config` precedent, including the `RoutingKnobs`
     plumbing for `oxllm_settings` / `router_settings` precedence that the
     strategy name itself rides on).
   - Generic, strategy-independent routing knobs go on `RoutingKnobs`
     instead — see §5.
6. **Optional host integration**:
   - *Pre-resolved data*: if a pick needs host-computed input (request-body
     classification, async cache lookups), add a `PreCallData` field in
     `strategy.rs` and populate it in the chat orchestration
     (`orchestration/mod.rs` — the `adaptive_request_type` classification is
     the precedent). The strategy reads the shared value and computes on
     demand only for standalone picks.
   - *Post-call learning*: keep the strategy pure; add a
     `crates/oxllm-server/src/hooks/<name>.rs` hook holding a clone of the
     shared store `Arc`, keyed by a pick identity stashed on `CallContext`
     (the `AdaptiveHook` + `adaptive_cell` precedent). Register it in
     `state/hook_builders.rs::build_hooks` only when the strategy is
     configured.

Conventions:

- **Observability**: label spellings come from
  `RoutingStrategyKind::as_str()` — they feed the `x-oxllm-router-strategy`
  / `x-oxllm-router-decision` response headers (`apply_router_headers` in
  `crates/oxllm-server/src/orchestration/mod.rs`) and the
  `oxllm_router_decisions_total{strategy, model}` counter
  (`crates/oxllm-server/src/hooks/metrics.rs`); the contracts are documented in
  [architecture.md](architecture.md#routing-observability). A new variant with
  a correct `as_str()` gets the right labels for free.
- **Logging**: per-pick events at `tracing::debug!`, never `info!` —
  per-request `info!` floods logs.
- **Testability**: stochastic strategies take randomness from an injectable
  source — the `AdaptiveRouter::with_rng` seeded-RNG precedent (production
  seeds a `ChaChaRng` from OS entropy; tests inject a seeded RNG) so picks
  are deterministic under test.
- Tests: unit tests next to the strategy (+ `router/tests.rs` for router
  integration), parse tests in `router_config.rs`, e2e following
  `tests/router_e2e.rs`.

---

## 5. Adding a new config option

**When:** a new operator-facing YAML knob.

1. Add the field to the right struct in
   `crates/oxllm-server/src/config/types/`:
   - `OxllmSettings` — behavior defaults (budgets, limits, cache, callbacks)
   - `GeneralSettings` — server/security settings (incl. `GuardrailsConfig`)
   - `RouterSettings` — router-only settings
   - A routing knob (strategy/retries/cooldown/fallbacks) goes on
     **`RoutingKnobs`**, which is `#[serde(flatten)]`ed into all three
     locations. Do not duplicate routing fields onto the containers.
    - Match neighboring defaults and compatibility behavior. Optional fields
      often use `Option<T>`; required fields and invalid known values should
      fail validation. Some nested configuration structs deliberately use
      `deny_unknown_fields`.
2. Wire it where it's consumed:
   - Hook behavior → read in `state/hook_builders.rs::build_hooks`.
   - Anything else built into state → add a field to `AppState`
     (`state/mod.rs`) and construct it in `state/build.rs::assemble`. One
     field + one construction site; don't add nested sub-structs or new
     `AppState::with_*` wrappers — extend `StateBuilder` only if the value is
     an *injected dependency* (cache/db/registry style).
3. If the option is expensive to rebuild (compiled regexes, spawned tasks,
   shared tables), add a `*_config_unchanged` helper in `state/reload.rs` so
   hot-reload preserves the existing `Arc` when the config didn't change.
4. Routing knobs get precedence for free via `RoutingKnobs::merge`
   (`router_settings` over `oxllm_settings`; `routing_groups` per-model at
   request time).
5. Add a sample to `examples/` if it's user-facing, and a parse/compat test
   (`tests/config_compat.rs` or a unit test in `config/`).

---

## 6. Adding a new budget scope

**When:** a new enforcement level in the budget/rate-limit hierarchy. The
hook currently checks key → team → org → project → user → customer → tag,
then the global per-model cap.

This is a cross-crate change — budget scopes touch identity, storage, and
enforcement:

1. **Core types**: extend `ScopeLimits` usage in
   `crates/oxllm-core/src/limits.rs`. `AllScopeLimits` holds the five identity
   scopes (key, team, user, org, project); customer and tag limits come from
   their own records. Add the flat fields and grouped accessors on
   `CallContext` in `crates/oxllm-core/src/hook.rs` (`<scope>_limits()`,
   `scope_limits()`, `set_scope_limits()`).
2. **Spend key format**: add `<scope>_spend_key` to
   `oxllm-server/src/hooks/spend_keys.rs`. The byte format is a
   cross-component contract (budget hook, alerting hook, reset scheduler,
   spend analytics) — never inline `format!("spend:...")`.
3. **Enforcement**: wire the new scope into `hooks/budget.rs` (`pre_call`
   reservation, `post_success` settle, `post_failure` release) and
   `hooks/rate_limit.rs` if the scope carries RPM/TPM.
4. **Identity loading**: populate the limits when the caller is resolved —
   `auth/` (`AuthUser`) and, with `db`, the virtual-key load path
   (`oxllm-db` repo + `rows.rs` mapping). Scopes without a DB column keep
   `None` fields.
5. **Alerting + reset**: the alerting hook and `BudgetResetScheduler` scan
   `spend:` prefixes — make sure the new scope's TTL/window semantics are
   handled.
6. Tests at every layer: core roundtrip (`scope_limits` ↔ flat fields), hook
   unit tests, and a metering e2e.

---

## 7. Adding a new guardrail

**When:** pre-call content screening or rewriting (PII, injection, toxicity,
semantic similarity, MCP-backed policy).

There are **ten** guardrails today: PII, Presidio, Injection, AIM, Lakera,
Bedrock, Model Armor, Custom (regex/keyword/json_path/http), Semantic
(MinHash), and MCP. Execution order is PII → Injection → local
Custom/Semantic → Presidio → AIM → Lakera → Bedrock → Model Armor → remote
Custom/MCP.

Guardrails are **not** hooks today — they run in the request pipeline via
`apply_guardrails` before the orchestrator. Raw provider passthrough and
embeddings have their own entry points; wire a new guardrail explicitly where
it should apply. Follow that pattern:

1. Create `crates/oxllm-server/src/guardrails/<name>.rs`. Masking guardrails
   rewrite the body in place; blocking guardrails return
   `Error::BadRequest` with a client-safe (non-detailed) reason. Remote
   guardrails must be **fail-open** unless configured otherwise.
2. Add config to `GuardrailsConfig` in `config/types/`. Enablement is
   guardrail-specific: some require an explicit `enabled: true`, while others
   activate when their config block is present. Follow the existing type for
   the guardrail you are adding. Validate remote URLs with
   `config::validate_api_base` (SSRF) at build time.
3. Add a field to `AppState` (`state/mod.rs`), build it in
   `guardrails::build_guardrails` (called from `state/build.rs::assemble`),
   and pass it through `apply_guardrails` in the right order — fast/local
   first, slow/remote last.
4. Call it from the chat handler's `apply_request_guardrails`
   (`handlers/chat.rs`). If the guardrail should apply to other endpoints
   (messages, responses), wire it there too — explicitly, per endpoint.
5. Unit tests for detection/masking + `tests/guardrails_e2e.rs` coverage.

---

## 8. Adding a new management endpoint

**When:** a new admin/ops REST endpoint (key/team/user/spend/model
management). Requires the `db` feature.

1. **Handler**: add (or extend) a module in
   `crates/oxllm-server/src/handlers/management/` (`keys.rs`, `teams.rs`,
   `spend.rs`, …). Handlers take `State<SharedState>`, load the current state
   with `shared.load_full()`, and use the `db` handle — return typed
   `ApiError`s, never `unwrap`.
2. **Persistence**: add the query to the matching `oxllm-db/src/repos/<entity>.rs`
   (`impl Database` block), row mapping in `rows.rs`, and a new migration
    batch in `oxllm-db/src/migrate.rs` if the schema changes (add a
    `v{N}_statements_for` function and a `run_migration_batch` +
    `set_schema_version` arm in `run_all_migrations`, passing its existing
    connection so DDL and version writes stay under the migration lock and
    transaction). Keep earlier batches idempotent. Also extend the record types in
   `oxllm-db/src/types.rs` (`KeyRecord`, `TeamRecord`, `SpendLog`, …) if new
   columns appear.
3. **Route**: register in `routes/management.rs::attach`. The group already
   sits behind master-key-only auth + auth-brake + optional CORS — don't add
   per-route auth.
4. **Invalidation**: if the endpoint mutates cached state (keys, models),
   publish on the `oxllm:invalidate` channel via the `Coordinator` so other
   replicas drop their local caches. Use the
   `auth::invalidate_cache_key(&state, cache_key)` helper (`auth/mod.rs`) —
   it clears both the local moka entry and the negative-cache entry, and
   publishes the cross-replica invalidation in one call (see existing
   mutation handlers like key delete / team add).
5. **Admin UI** (optional): add a page/handler in `oxllm-admin/src/pages/`
   and call the endpoint through `proxy_client.rs`.
6. E2E in `tests/management_e2e.rs` (or a new topic file).

---

## 9. Adding a new pre-call check

**When:** you want to filter, pin, or reject deployments **at routing time** —
before any upstream call — based on the request (prompt-caching pin, budget
soft-cap, staleness, custom policy). Pre-call checks are a distinct extension
point from hooks (§2) and guardrails (§7): they run *inside* the router over
the candidate deployment list, not over the request body.

The contract lives in `oxllm-core/src/router/pre_call_check.rs`:

```rust
pub trait PreCallCheck: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn filter_deployments<'a>(
        &self,
        candidates: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Result<Vec<&'a Deployment>, oxllm_core::Error>;
}
```

A check synchronously returns eligible deployments. Async inputs are resolved
by the host into `ctx.pre_call` before selection. There is no `PreCallOutcome`
enum or async `check` method. The router skips a failed check and restores the
input if a check empties the candidate list, so these are availability-oriented
filters, not authorization gates. Enforce hard rejection in the appropriate
policy/pipeline path. Prompt-cache routing narrows candidates using the
pre-resolved pin and stamps its decision in orchestration.

Existing precedents to copy:

- **`HealthCheck`** (`oxllm-core/src/router/pre_call_check.rs`) — enabled by
  `enable_health_check_routing`; filters recently unhealthy deployments and
  permits stale/unknown health entries.
- **`prompt_caching`** (`oxllm-server/src/pre_call_checks/prompt_caching.rs`)
  — opt-in via `optional_pre_call_checks: ["prompt_caching"]`; pins a
  deployment whose prompt cache has the request's prefix when the token
  estimate clears `prompt_caching_min_tokens`.
- **`budget_limiter`** (`oxllm-server/src/pre_call_checks/budget_limiter.rs`)
  — opt-in via `optional_pre_call_checks: ["budget_limiter"]`; prunes
  deployments over their per-deployment budget soft cap.

Steps:

1. **Implement the check.** Pure-core checks (no I/O) go in
   `oxllm-core/src/router/pre_call_check.rs` next to `HealthCheck`. Checks
   that need host data (token estimates, live spend counters) live in
   `oxllm-server/src/pre_call_checks/<name>.rs` and read from `PreCallData`,
   which the orchestration populates (`orchestration/mod.rs`).
2. **Pick a config name** — snake_case, e.g. `"my_check"`. Accept both
   snake_case and kebab-case (`"my-check"`) via `normalize_check_name`
   (the `prompt_caching` / `budget_limiter` precedent).
3. **Register** host checks in `build_pre_call_checks`
   (`oxllm-server/src/pre_call_checks/mod.rs`); state construction passes its
   `Vec<Arc<dyn PreCallCheck>>` to `Router::set_pre_call_checks`. `HealthCheck`
   is constructed by the router when health-check routing is enabled.
   Host checks run when their name appears in
   `router_settings.optional_pre_call_checks`, wired in
   `state/router_config.rs`.
4. **Wire any host-side data** the check needs onto `PreCallData`
   (`strategy.rs`), populated once in the chat orchestration before the
   pick (the `adaptive_request_type` / prompt-token-estimate precedents).
5. Tests: unit tests on the check (`pre_call_check.rs` /
   `pre_call_checks/<name>.rs`), parse/registration tests in
   `router_config.rs`, and an e2e in `tests/router_e2e.rs`.

---

## 10. Testing requirements

Every feature lands with tests at the layers it touches:

| Layer | Where | What |
|-------|-------|------|
| **Unit** | `#[cfg(test)]` next to the code | Pure logic: transforms, parsing, cost math, limit grouping, config precedence. No I/O. |
| **Integration** | `tests/<topic>_e2e.rs` | Boot oxllm in-process via the shared harness (`tests/common/mod.rs`: `start_oxllm`, `proxy_config`, `OxllmHandle`) against httpmock upstreams. Name files by topic, not milestone; update `tests/README.md`. |
| **E2E (browser)** | `tests/admin-ui/` | Only for admin-UI changes (Playwright). |
| **Bench** | `crates/*/benches/` (criterion) | Only for hot-path changes (router, cost, tokens, transforms). |

Rules of thumb:

- New provider → unit tests (+ goldens); e2e only if it adds new host
  behavior (credentials, streaming quirks).
- New hook → unit tests for pre/post semantics + a metering/router e2e if it
  changes request outcomes.
- New config option → parse test + a test proving precedence/merge behavior.
- New budget scope → core roundtrip test, hook tests, metering e2e.
- New endpoint → e2e happy path + auth failure + upstream failure.
- Layering changes → `tests/workspace_layering.rs` must still pass.

Run before committing:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

---

## 11. Architectural rules — what goes where

| Crate | Put here | Never put here |
|-------|----------|----------------|
| `oxllm-core` | Shared types, traits (`Provider`, `Hook`, `Cache`), router strategies, cost math, token estimation, `ScopeLimits` | Concrete network/fs/env I/O, concrete DB/Redis types; async contracts are limited to abstractions such as `Hook` and `Cache` |
| `oxllm-providers` | Pure request/response/stream transforms, `endpoint_url`, `apply_auth` | HTTP calls, credential *acquisition*, SDK signing with I/O, anything importing the server |
| `oxllm-server` | Handlers, routes, hooks, orchestration, pipeline, state, auth, config, credentials, guardrails, coordination | Client-facing wire types that belong in core, price data |
| `oxllm-db` | sqlx queries, migrations, row mapping | Business logic, hook/enforcement decisions |
| `oxllm-cache-redis` | `Cache` trait impl for Redis | Anything else |
| `oxllm-admin` | Admin UI pages, proxy client | Proxy business logic (call the management API) |

Enforced by `tests/workspace_layering.rs`; CI fails on forbidden edges.

---

## 12. Common pitfalls — what NOT to do

- **Don't add fields to `CallContext` directly for grouped concepts.**
  Per-scope limits go through `ScopeLimits` / `AllScopeLimits` and the
  `*_limits()` / `set_scope_limits()` accessors. Flat fields remain the
  storage format, but new scoped state must extend the grouped types so every
  constructor/conversion site gets it for free. (A truly singular per-request
  field — like `cache_hit` — is fine as a flat field.)
- **Don't bypass the established integration point for guardrails.**
  Guardrails today are *not* `Hook`s: they're built into `AppState` and run
  in the chat handler via `apply_guardrails`. If you need guardrails on
  another endpoint, wire them explicitly there — don't silently turn a
  guardrail into a hook or vice versa without migrating all ten.
- **Don't add global state.** No `static mut`, no `lazy_static`/`OnceLock`
  singletons for runtime state. Shared state lives on `AppState` and is
  swapped atomically via `SharedState` (`ArcSwap`) on hot-reload. If it's
  expensive to rebuild, preserve it with a `*_config_unchanged` helper in
  `state/reload.rs`.
- **Don't do I/O in `oxllm-core` or `oxllm-providers`.** Transforms are
  synchronous and pure; the host resolves credentials (`ResolvedCredentials`)
  and performs all HTTP. The layering test will fail, and rightly so.
- **Don't create `reqwest::Client` per request.** Reuse
  `state.http_client`.
- **Preserve each config block's compatibility contract.** Broad LiteLLM
  compatibility blocks tolerate unknown keys; strict nested blocks reject them.
  Validate malformed known values rather than silently disabling behavior.
- **Don't resolve `os.environ/...` on the hot path.** It is resolved once at
  config load.
- **Don't log secrets or raw API keys.** Log key *hashes*; `tracing` fields
  must redact.
- **Don't `unwrap`/`expect`/`panic` on user or provider input.** Return
  typed errors (`oxllm_core::Error` / `ApiError`).
- **Don't inline spend-counter key formats** (`format!("spend:...")`) — use
  `hooks/spend_keys.rs`. The budget hook, alerting hook, reset scheduler, and
  spend analytics must agree byte-for-byte.
- **Don't add routing fields to `OxllmSettings` / `RouterSettings`
  directly** — extend `RoutingKnobs` so all three YAML locations and the
  merge precedence keep working.
- **Don't add per-route auth middleware.** Auth stacks are applied to merged
  route groups in `routes/mod.rs` (proxy auth) and `routes/management.rs`
  (master-key only).

---

## Related

- [Architecture](architecture.md) — crates, request path, layering rules
- [Adding a provider](adding-a-provider.md) — provider-specific checklist
- [tests/README.md](../tests/README.md) — integration test inventory
