# Architecture

How oxllm is put together: crates, dependency rules, and the path of a request.

This doc is for humans and agents working on the codebase. For usage and
deploy, see the root [README](../README.md). For step-by-step feature work,
see [adding-a-feature.md](adding-a-feature.md) and
[adding-a-provider.md](adding-a-provider.md).

---

## What oxllm is

oxllm is an **LLM proxy gateway**. Clients speak OpenAI (and some sibling
shapes like Anthropic Messages). oxllm:

1. Authenticates and meters the caller
2. Picks a **deployment** (provider + model + credentials) via the router
3. **Transforms** the request to the provider's native format (pure code)
4. Performs **HTTP/WS I/O** to the upstream
5. Transforms the response back and runs post-call hooks (spend, cache, metrics)

The design goal: **providers are pure transforms; the host owns all I/O.**

---

## Workspace layout

```
oxllm/                          # root package: CLI binaries
├── src/main.rs                 # `oxllm` binary (serve, keygen, …)
├── src/bin/oxllm-admin.rs      # optional admin UI binary
├── crates/
│   ├── oxllm-core/             # pure types, traits, router, cost, tokens, limits
│   ├── oxllm-providers/        # 34 provider transforms (feature-gated)
│   ├── oxllm-server/           # Axum host: routes, hooks, config, HTTP, state
│   ├── oxllm-cache-redis/      # Redis Cache impl (optional)
│   ├── oxllm-cache-s3/         # S3-compatible object-storage cache backend (optional)
│   ├── oxllm-cache-gcs/        # Google Cloud Storage cache backend (optional)
│   ├── oxllm-cache-azure/      # Azure Blob Storage cache backend (optional)
│   ├── oxllm-db/               # sqlx persistence: keys/teams/users/orgs/projects/spend/models
│   └── oxllm-admin/            # separate admin UI process
├── examples/                   # sample YAML configs
├── tests/                      # workspace integration / e2e tests
├── docs/                       # this documentation
└── helm/                       # Kubernetes chart
```

A **crate is a layer**, not a route or a provider. Routes live as modules
inside `oxllm-server`. Providers live as modules inside `oxllm-providers`.

---

## Crate roles

| Crate | Role | Pure / I/O |
|-------|------|------------|
| `oxllm-core` | Types, traits (`Provider`, `ChatProvider`, `Hook`, `Cache`), router, cost, tokens, `limits` (`ScopeLimits`) | **Pure** |
| `oxllm-providers` | Per-provider request/response/stream transforms, 34 providers selected by feature flags | **Pure** |
| `oxllm-server` | Axum app, handlers, hooks, orchestration, pipeline, state, auth, config, credentials, security, guardrails, cache, `spend_writer`, scheduler, otel, coordination | **I/O (host)** |
| `oxllm-cache-redis` | Redis implementation of the `Cache` trait | I/O |
| `oxllm-cache-s3` | S3-compatible object-storage implementation of the `Cache` trait | I/O |
| `oxllm-cache-gcs` | Google Cloud Storage implementation of the `Cache` trait | I/O |
| `oxllm-cache-azure` | Azure Blob Storage implementation of the `Cache` trait | I/O |
| `oxllm-db` | sqlx persistence for keys, teams, users, orgs, projects, spend, models | I/O |
| `oxllm-admin` | Browser admin UI (separate process; talks to management API) | I/O |
| `oxllm` (root) | Thin CLI that builds and runs the server (and optional admin) | I/O |

### Module organization

**`oxllm-core/src/`** — flat modules plus the router:

| Module | Contents |
|--------|----------|
| `provider.rs` | `Provider` super-trait, `ChatProvider` / `EmbeddingProvider` / `RerankProvider`, `ResolvedCredentials` |
| `hook.rs` | `Hook` trait, `CallContext`, `ResponseSummary`, `RateLimitInfo` |
| `router/` | `Router`, 12 strategies, cooldowns, stores, `RouterConfig` |
| `limits.rs` | `ScopeLimits`, `AllScopeLimits` (grouped per-scope limits) |
| `cost.rs` | Price table + cost math (`calculate_cost_with_override`) |
| `tokens.rs` | Token estimation (tiktoken) |
| `cache.rs` | `Cache` trait + `IncrResult` |
| `deployment.rs` / `endpoint.rs` / `registry.rs` / `error.rs` | Core types |

**`oxllm-providers/src/`** — one module per provider, with 34 provider Cargo
features (all enabled by default): `openai`, `anthropic`, `gemini`, `ollama`, `groq`,
`together_ai`, `openrouter`, `xai`, `mistral`, `cohere`, `azure`, `bedrock`,
`bedrock_mantle`, `vertex_ai`, `databricks`, `perplexity`, `sambanova`,
`cloudflare`, `deepseek`, `cerebras`, `ai21`, `fireworks_ai`, `deepinfra`,
`nvidia_nim`, `voyage`, `jina_ai`, `watsonx`, `huggingface`, `deepgram`,
`elevenlabs`, `dashscope`, `replicate`, `meta`, `baseten`. Most provider modules
are gated by their own feature. OpenAI-compatible provider features imply the
shared `openai_compat` feature, which gates both `openai_compat` and the reusable
`openai` module. The `vertex_ai` feature also enables `anthropic` for its Claude
transforms. `lib.rs` wires `default_registry()` from the individual provider
features: compiling the shared OpenAI module does not by itself register the
`openai` provider.

**`oxllm-server/src/`**:

| Area | Path | Responsibility |
|------|------|----------------|
| Routes | `routes/` | Wire handlers to URLs; middleware stack (`llm`, `platform`, `management`, `passthrough`, `observability`, `azure_legacy`) |
| Handlers | `handlers/` | Thin: parse, call pipeline/orchestrator (+ `management/` submodules) |
| Pipeline | `pipeline/` | Shared lifecycle: `CallContext`, hooks, deployment limits, header stamping |
| Orchestration | `orchestration/` | Chat retries, fallbacks, streaming, upstream call (`chat`, `outcomes`, `stream`, `upstream`, `usage`) |
| Hooks | `hooks/` | Budget, rate limit, cache, cost, metrics, alerting, langfuse, langsmith, datadog, helicone, mlflow, slack/pagerduty alerts, callbacks |
| State | `state/` | `AppState` / `SharedState`, `StateBuilder`, reload, router config, hook builders, DB models (see below) |
| Auth | `auth/` | Master key, virtual keys (DB), JWT |
| Config | `config/` | YAML load (`loader`), types, `os.environ/` resolve once, hot-reload (`watcher`) |
| Credentials | `credentials/` | Resolve `ResolvedCredentials` per deployment (static key, Bedrock, Vertex) |
| Security | `security.rs` | AuthBrake, body limits, smuggling guard |
| Guardrails | `guardrails/` | PII, Presidio, Injection, AIM, Lakera, Bedrock, Model Armor, Custom, Semantic, MCP (see below) |
| Coordination | `coordination.rs` | Redis-optional `Coordinator` (see below) |
| Cache | `cache/` (`mod.rs`, `disk.rs`, `semantic.rs`, `redis_semantic.rs`, `qdrant.rs`) + `hooks/cache_hook.rs` | `InMemoryCache`, `DualCache`, disk cache, semantic cache (in-memory, Redis, Qdrant) |
| Spend | `spend_writer.rs` | Batched async spend-log writer (DB) |
| Scheduler | `scheduler.rs` | `BudgetResetScheduler`, budget-window parsing |
| OTel | `otel.rs` | OpenTelemetry export (feature-gated) |

**`oxllm-db/src/`** — split by domain, all contributing `impl Database`
blocks:

- `pool.rs` — connection setup (`sqlx::Any`: Postgres + SQLite)
- `migrate.rs` — schema versioning: numbered DDL batches applied in order by
  `run_all_migrations`, with a `_schema_version` table
  recording the highest applied version. Each batch is idempotent (uses
  `CREATE TABLE IF NOT EXISTS` / `ADD COLUMN`) so it no-ops when a database
  pre-dates version tracking. `PlaceholderStyle` is threaded through so the
  same statements serve Postgres and SQLite.
- `rows.rs` — `sqlx` row → record mapping
- `repos/{keys,teams,users,orgs,projects,spend,models,budgets,customers,tags,skills}.rs` — per-entity queries
- `types.rs` — record types (`KeyRecord`, `SpendLog`, …)

**`oxllm-admin/src/`** — server-rendered admin UI:

- `pages/{auth,dashboard,keys,teams,users,orgs,projects,spend,calculator,models}.rs`
- `dates.rs`, `csv.rs` — shared formatting helpers
- `proxy_client.rs` — talks to the server's management API

### Dependency DAG (enforced)

```
oxllm-core  ←  oxllm-providers
oxllm-core  ←  oxllm-cache-redis  ←  oxllm-server
oxllm-core  ←  oxllm-cache-s3     ←  oxllm-server
oxllm-core  ←  oxllm-cache-gcs    ←  oxllm-server
oxllm-core  ←  oxllm-cache-azure  ←  oxllm-server
oxllm-core  ←  oxllm-db           ←  oxllm-server
oxllm-server  ←  oxllm (binary)
oxllm-admin   ←  oxllm (admin binary, feature-gated)
```

`tests/workspace_layering.rs` parses every crate's `Cargo.toml` and fails CI
if someone adds a forbidden edge (e.g. `oxllm-core` depending on `reqwest`).

### Core purity rules

**Allowed in `oxllm-core`:** pure transforms, shared types/traits, router
logic on `Deployment` + `Cache` trait, cost math, token estimation.

**Not allowed in `oxllm-core`:** network, filesystem, env reads, concrete DB
or Redis types, provider SDK signing.

**Provider boundary:** `oxllm-providers` imports **only** `oxllm-core`. The
host (`oxllm-server`) does all HTTP and credential resolution for the call.

---

## Request path (chat)

Happy path for `POST /v1/chat/completions`:

```
Client
  → routes (auth middleware, body limits, smuggling guard)
  → handlers/chat.rs          # parse model, stream flag, guardrails
  → EndpointPipeline          # CallContext, pre-call hooks
  → ChatOrchestrator          # cache short-circuit OR request chain
  → router.pick_deployment    # strategy + cooldowns + fallbacks
  → provider transform_request
  → upstream HTTP (reqwest)
  → provider transform_response / stream
  → post-call hooks (budget adjust, RPM/TPM, cost, cache store)
  → Response (+ x-oxllm-* headers)
```

Response-cache writes require a buffered `ResponseSummary.body`. Live upstream
streams currently finish with no buffered body, so `CacheHook::post_success`
does not store them; stream capture/reconstruction is not implemented. Existing
matching cached responses can be replayed as SSE. Exact cache keys include the
`stream` flag, so warming a non-streaming request does not seed its streaming
counterpart; semantic-cache lookup also skips streaming requests.

---

## State (`oxllm-server/src/state/`)

`AppState` is the struct every handler reads; `SharedState =
Arc<ArcSwap<AppState>>` is the hot-reload handle. The module is split into
six focused files:

| File | Contents |
|------|----------|
| `state/mod.rs` | `AppState` struct, `SharedState`, `DbOpt`, module docs |
| `state/build.rs` | `StateBuilder` with optional setters (`with_registry`, `with_cache`, `with_db`, …) and the single `assemble` function both startup and reload funnel through |
| `state/reload.rs` | `AppState::reload_from_config` + the `*_config_unchanged` helpers (`jwt_secret_unchanged`, `alerting_config_unchanged`, `semantic_cache_config_unchanged`, `langfuse_config_unchanged`) that decide which infrastructure `Arc`s survive a hot-reload |
| `state/router_config.rs` | `build_router_config`, `parse_strategy_kind`, `parse_fallback_rules` — `RouterConfig` assembly from `ProxyConfig` with `router_settings` over `oxllm_settings` precedence |
| `state/hook_builders.rs` | `build_hooks` (chain assembly), `build_alerting_hook`, langfuse wiring, `build_generic_callback` |
| `state/db_models.rs` | `refresh_db_models`, `db_record_to_deployment`, `decrypt_stored_api_key` (feature `db`) |

Design note: `AppState` keeps its fields **flat** on purpose — handlers read
`state.cache`, `state.router`, etc. directly. Cohesive groupings are
expressed at *construction* time in `build::assemble`, not as nested
sub-structs, so a new feature adds a field + one construction site.

Feature-specific constructors live next to the feature they build (guardrails
in `crate::guardrails`, semantic cache in `crate::cache`); `state/`
only *calls* them.

---

## Provider trait

Contracts live in `oxllm-core/src/provider.rs`. All transform methods are
**synchronous pure functions** — no network, no async.

```rust
pub trait Provider: Send + Sync {
    fn id(&self) -> ProviderId;
    fn supports(&self, endpoint: Endpoint) -> bool;
    fn chat(&self) -> Option<&dyn ChatProvider> { None }
    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> { None }
    fn rerank(&self) -> Option<&dyn RerankProvider> { None }
}
```

The server dispatches `(Endpoint, Provider) → handler` via `supports()` + the
typed accessors — no giant match. Each capability trait
(`ChatProvider` / `EmbeddingProvider` / `RerankProvider`) has
`endpoint_url`, `apply_auth`, `transform_request`, `transform_response` (and
`transform_stream_chunk` for chat).

Credentials are resolved by the host and passed in as
`ResolvedCredentials`:

| Variant | Used by |
|---------|---------|
| `ApiKey { bearer }` | OpenAI, Anthropic, Gemini, most providers |
| `Azure { api_key }` | Azure OpenAI (`api-key` header) |
| `AwsSigV4 { access_key, secret_key, session_token, region }` | Bedrock |
| `GcpBearer { token }` | Vertex AI (pre-exchanged OAuth2 token) |
| `PreSignedHeaders(HashMap)` | Signing done entirely in the host |

The enum is `#[non_exhaustive]`; providers pattern-match in `apply_auth` and
the host's `CredentialResolverRegistry` picks the right variant per provider
slug.

---

## Routing

Configured via `oxllm_settings.routing_strategy` (or `router_settings`).
Strategies live in `oxllm-core/src/router/` and pick among **deployments**
with the same client `model_name`:

| Strategy | File | Behavior |
|----------|------|----------|
| `simple_shuffle` | `simple_shuffle.rs` | Weighted random (default): probability proportional to `oxllm_params.weight`, falling back to `rpm`, then 1 — matching LiteLLM's default behavior |
| `lowest_cost` | `lowest_cost.rs` | Cheapest deployment by price table |
| `lowest_latency` | `lowest_latency.rs` | Lowest measured latency |
| `usage_based` | `usage_based.rs` | Least-used (RPM/TPM) deployment |
| `latency_based` | `latency_based.rs` | Latency-window based selection |
| `least_busy` | `least_busy.rs` | Fewest in-flight requests |
| `tag_based` | `tag_based.rs` | Match request tags (`metadata.tags` in the
 request body + `x-oxllm-tag` header values) against deployment
 `model_info.tags` (subset match, `&tag` require, `!tag` exclude,
 `tag_regex`, `default` deployments as fallback pool) |
| `quality_router` | `quality.rs` | Configured quality scores |
| `session_affinity` | `session_affinity.rs` | Sticky via `x-oxllm-session-id`, or `x-litellm-session-id` when that header is absent or empty (consistent hash, no TTL) |
| `complexity_router` | `complexity.rs` | Keyword-classify request complexity → tier model pool |
| `adaptive_router` | `adaptive.rs` | Thompson-sampling bandit per (request type, deployment); learns from post-call outcomes via `hooks/adaptive_hook.rs`; cold start defers to least-busy |
| `round_robin` | `round_robin.rs` | Per-model-group rotation within each replica; weighted slots use `weight`, then `rpm`, then `tpm`, then 1; counters survive config reloads |

New strategies implement `RoutingStrategy` (`strategy.rs`) and are parsed in
`state/router_config.rs::parse_strategy_kind` (accepts `snake_case` and
`kebab-case`; unknown names warn + fall back to `simple_shuffle`).
Strategies are constructed via `strategy_factory(kind)` with a shared
`StrategyDeps` bundle (`strategy.rs`), so plugging in a new strategy only
adds a factory arm — the picker's dispatch in `Router::build_inner` is
unchanged. Full checklist: [adding-a-feature.md](adding-a-feature.md) §4.

**The adaptive router learns from chat traffic only.** Only the chat
orchestration classifies the request type and stamps the bandit cell for
the learning hook; embeddings, passthrough, and realtime picks go through
the same `Router::pick_deployment` with a bare request context, classify as
`RequestType::Other` on demand, and never record outcomes. Under a
globally-configured `adaptive_router`, those endpoints therefore ride the
least-busy cold start indefinitely (or sample the uniform `Beta(1, 1)`
prior when `cold_start_min_observations` is 0) — a deliberate safe
degradation: they neither pollute chat's learned posteriors nor starve
themselves.

Also: cooldowns on failure, retries (incl. per-status `retry_policy`), model
fallbacks, context-window / content-policy fallbacks, priority-order tiers
(`oxllm_params.order` — lower order first, unset = lowest tier),
`default_fallbacks` for unknown model groups, wildcards
(`model_name: "*"`), model group aliases, and per-model `routing_groups`
overrides resolved at request time inside the router.

### Routing observability

Two response headers are stamped from the pick-time stashes on
`CallContext` (`apply_router_headers` in `orchestration/mod.rs`):

| Header | Value |
|--------|-------|
| `x-oxllm-router-strategy` | Strategy that made the serving pick (`RoutingStrategyKind::as_str()`, e.g. `simple_shuffle`) |
| `x-oxllm-router-decision` | How the serving deployment was reached: `picked` (strategy pick), `fallback` (fallback-chain leg), `default_fallback` (unknown-model default chain), `prompt_cache_pin` (pinned by the prompt-caching pre-call check) |

Both are emitted on every `Ok` response from the chat orchestrator —
streaming (at stream start) and non-streaming alike, including
fallback-exhausted responses whose body carries an upstream error.
Responses on the hard-error path (terminal failure with no fallbacks, or a
pre-routing rejection) do **not** carry them, and a cache short-circuit
pre-empts the pick entirely. Treat header absence as "no routing decision
completed", never as a routing outcome.

Routing metrics (via the `metrics` crate, rendered at `GET /metrics`):

| Metric | Kind | Labels | Emitted from |
|--------|------|--------|--------------|
| `oxllm_router_decisions_total` | counter | `strategy`, `model` (requested model group) | `hooks/metrics.rs` — once per completed request (success or failure) with a pick |
| `oxllm_budget_exceeded_total` | counter | `scope` | `hooks/budget.rs` — every budget rejection |
| `oxllm_health_check_unhealthy_total` | counter | `deployment` (non-secret `deployment_key`, `{model_name}:{provider_model}`) | `orchestration/chat.rs` + `orchestration/stream.rs` — on the healthy→unhealthy transition only |
| `oxllm_prompt_cache_hits_total` | counter | `model` | `hooks/cache_hook.rs` — exact and semantic hits share the series |
| `oxllm_complexity_tier_total` | counter | `tier` | `orchestration/chat.rs` — once per classified request |
| `oxllm_adaptive_alpha`, `oxllm_adaptive_beta` | gauges | `request_type`, `model` | `hooks/adaptive_hook.rs` (+ the orchestration failure debit) — refreshed after each learning update |

`GET /routing/state` (management API, master-key auth; handler
`handlers/management/routing.rs`) returns the live router snapshot:
effective `strategy` plus per-group `group_strategies`, registered
`pre_call_checks`, and per-deployment `key` (`{model_name}:{16-hex}`
— the stable cooldown key: a digest of the sanitized `oxllm_params`
plus a per-process keyed credential digest, never the credentials
themselves), `model_name`, `provider_model`, `cooling`,
`healthy`, `health_reason`, and `bandit_cells` (per request type `{alpha,
beta, observations}`; omitted when the deployment has no recorded cells).

---

## Hook chain

Hooks implement `oxllm_core::Hook` (`pre_call`, `post_success`,
`post_failure`) — a dyn-safe async contract, as is `Cache`. The chain is a
`Vec<Arc<dyn Hook>>` built by
`state/hook_builders.rs::build_hooks`. Current order:

```
RateLimitHook → CacheHook → BudgetHook → CostTrackingHook → MetricsHook
  → [AdaptiveHook] → [GenericHttpCallback] → [LangfuseHook] → [LangsmithHook]
  → [DatadogHook] → [HeliconeHook] → [WandbHook] → [S3LogHook] → [MlflowHook]
  → [SlackAlertingHook] → [PagerDutyAlertingHook] → [EmailAlertingHook]
  → [AlertingHook]
```

**Order matters:**

- **RateLimit before Cache** — RPM is consumed even on a cache hit (the
  request was received and processed).
- **Cache before Budget** — a cache short-circuit never reserves spend (no
  upstream cost).
- Bracketed hooks are optional, appended in the order shown when
  configured: `AdaptiveHook` (the `adaptive_router` strategy configured
  globally or for any routing group), `GenericHttpCallback`
  (`success_callback: ["generic"]` + `callback_settings.generic`),
  `LangfuseHook` (`langfuse_config`), `LangsmithHook` (`langsmith_config`),
  `DatadogHook` (`datadog_config`), `HeliconeHook` (`helicone_config`),
  `WandbHook` (`wandb_config`), `S3LogHook` (`s3_log_config`),
  `MlflowHook` (`mlflow_config`), `SlackAlertingHook`
  (`slack_alerting_config`), `PagerDutyAlertingHook` (`pagerduty_config`),
  `EmailAlertingHook` (`email_alert_config`), `AlertingHook` (alerting
  config). All of them are post-call observability/callback hooks sitting
  after the always-on five, so their relative order does not affect
  enforcement.

**Reservation / rollback semantics:** exact cost is unknowable before the
upstream responds, so `BudgetHook::pre_call` **atomically reserves** a
conservative estimate via `Cache::incr_if_below_in_window`, which also returns
an opaque per-scope counter-window receipt (`CounterWindow`). `post_success`
settles to the actual cost (`actual − estimate`) and `post_failure`
**releases** the reservation so failed requests don't consume budget — both
through `Cache::incr_in_window`, which applies only while the *original*
admission window is still live. A receipt can never adjust or refund a
replacement counter: once a budget window expires and a later request recreates
it, refunds are discarded rather than minting headroom in the wrong window.
Settlements, however, attribute spend at completion time — when the admission
window is gone, the **full actual** is carried into the live successor counter
(metered as `oxllm_budget_settle_carried_total`), so a long request that
straddles a rollover cannot settle unbounded spend unseen by any window. The
durable spend ledger is independent and records the actual cost once either
way. Top-ups after deployment selection reserve in the same
window or reject. The estimate is computed once per request and stashed on
`CallContext.budget_estimate` alongside the receipts in
`CallContext.budget_windows`; `CallContext.token_estimate` (a `OnceLock`)
shares the prompt-token estimate between the rate-limit and budget hooks.
A partial session whose complete cost is known before its terminal failure
(realtime duration-priced sessions) settles once through `Hook::settle_budget`
and disarms the later failure refund without emitting success events.

`CallContext` carries per-request fields (model, key hash, identity ids,
per-scope limits, `cache_hit`, `budget_estimate`, `deployment_pricing`,
`adaptive_cell`, `ratelimit_info`). Metering hooks skip `post_success` work
on cache hits.

---

## Config

- Shape is LiteLLM-compatible YAML (`model_list`, `oxllm_settings`,
  `general_settings`, `router_settings`).
- **`RoutingKnobs`** (`config/types/router.rs`) holds the routing fields
  (`routing_strategy`, `num_retries`, `cooldown_time`, `fallbacks`,
  `context_window_fallbacks`, `content_policy_fallbacks`,
  `default_fallbacks`, `retry_policy`) and
  is **flattened** (`#[serde(flatten)]`) into `OxllmSettings`,
  `RouterSettings`, and `RoutingGroupConfig` — one Rust definition, three
  YAML locations.
- Precedence: `RoutingKnobs::merge(global, group)` resolves field-by-field
  with the group winning; `build_router_config` merges `router_settings`
  over `oxllm_settings`, and per-model `routing_groups` win at request
  time inside the router.
- Compatibility blocks tolerate unknown keys with warnings; strict nested
  config structs and malformed known values can fail validation.
- `os.environ/VAR` is resolved **once at config load**, never on the hot path.
- Samples live in [`examples/`](../examples/). Local `config.yaml` is gitignored.
- The binary does not load `.env`. Missing string env references fail loading;
  the legacy `environment_variables` block is a no-op in shipped builds.
- Reload preserves DB/cache connections and the HTTP router. See the
  [operations reload matrix](operations.md#configuration-and-restarts) for
  startup-only middleware, retention, logging and shutdown settings.

---

## Guardrails

Ten pre-call content guardrails live in `oxllm-server/src/guardrails/`,
configured via `general_settings.guardrails` (`GuardrailsConfig`). They run in
the order listed below (fast/local first, slow/remote last); the first
`block` wins:

| Guardrail | Kind | Behavior |
|-----------|------|----------|
| `PiiGuardrail` | local regex | Masks emails, phones, credit cards, SSNs, API keys in place |
| `PresidioGuardrail` | remote API | Microsoft Presidio `/analyze` + `/anonymize`; fail-open |
| `InjectionGuardrail` | local heuristics | Prompt-injection detection; `block` (400) or `log` |
| `AimGuardrail` | remote API | AIM Security `/api/v1/prompt/scan`; blocks at threshold; fail-open |
| `LakeraGuardrail` | remote API | Lakera Guard `/v1/guard`; blocks when `flagged: true`; fail-open |
| `BedrockGuardrail` | remote API | AWS Bedrock `ApplyGuardrail` (SigV4-signed); blocks on `GUARDRAIL_INTERVENED`; fail-open |
| `ModelArmorGuardrail` | remote API | Google Cloud Model Armor; blocks on `blocked: true` / threshold; fail-open |
| `CustomGuardrail` | local + remote | User-defined regex / keyword / json_path / http classifiers from `custom_guardrails` |
| `SemanticGuardrail` | local MinHash | MinHash similarity against configured blocked phrases |
| `McpGuardrail` | remote JSON-RPC | MCP `tools/call` to a server; blocks on the server's verdict; fail-open |

Guardrails are **not Hook-based today**: the first six (PII, Presidio,
Injection, AIM, Lakera, Bedrock) are pre-built once at state construction
(`build_guardrails`), stored on `AppState` (`pii_guardrail`,
`injection_guardrail`, `presidio_guardrail`, `aim_guardrail`,
`lakera_guardrail`, `bedrock_guardrail`); Model Armor and the Custom/Semantic/MCP
chain are also built into state and preserved on reload when their config is
unchanged (`guardrails_config_unchanged`).
All ten are invoked by the **chat handler** via `apply_guardrails` after the
request body is parsed and before the orchestrator runs — so the
masked/redacted body is what gets serialized upstream. Run order: PII →
Presidio → Injection → AIM → Lakera → Bedrock → Model Armor → Custom →
Semantic → MCP (fast-local first, slow-remote last).

---

## Budget hierarchy

Six enforcement levels, checked by `BudgetHook` in this order:

```
key → team → org → project → user → model
```

Each scope has its own spend counter (`spend:{hash}`, `spend:team:{id}`,
`spend:org:{id}`, `spend:project:{id}`, `spend:user:{id}`,
`spend:model:{model}` — formats owned by `hooks/spend_keys.rs`) and is only
checked when the corresponding limit is set. The model scope is a global cap
per model name (`oxllm_settings.model_max_budget`).

The per-scope limits (`max_budget`, `budget_duration`, `rpm`, `tpm`) are
grouped in **`oxllm-core/src/limits.rs`** as `ScopeLimits` (one scope) and
`AllScopeLimits` (all five identity scopes: key, team, user, org, project).
`CallContext` stores them flat but exposes `key_limits()`,
`team_limits()`, …, `scope_limits()` and `set_scope_limits()` so
construction/conversion code has one place to expand/collapse — adding a new
scoped limit no longer means hand-editing every constructor. (Not every scope
populates every field: user carries only `max_budget`.)

Customer and request-tag budgets are also enforced when their records supply a
budget. In addition, the **routing loop** enforces two
more budget layers: per-deployment (`oxllm_params.max_budget`, counter
`spend:dep:{deployment_key}`) and per-provider
(`router_settings.provider_budget_config`, counter `spend:provider:{slug}`).
A picked deployment at/over its cap is cooled and the router tries the next
one. Chat and embeddings use atomic reservations and settlement in
`orchestration/budget.rs`; the optional `budget_limiter` candidate filter uses a
read-only spend snapshot and is not itself a hard enforcement gate. Other
endpoint paths must opt into the reservation lifecycle explicitly. Cost
estimates and provider usage still bound billing accuracy; the router-budget
implementation has a nominal unknown-cost fallback rather than treating every
unpriced request as free. Check endpoint-specific tests for disconnect/error
settlement behavior.

---

## Distributed coordination

When Redis is configured (`cache.type: redis`), several primitives share
state across replicas. Without Redis, each degrades to per-process behavior
with no functional loss for single-replica.

- **AuthBrake** (via `Coordinator` in `coordination.rs`): when distributed,
  failure counts are stored in Redis via atomic `incr` operations. An
  attacker's budget is shared across all replicas. The `Coordinator` is a
  thin `{cache, distributed}` wrapper whose ONLY consumer is `AuthBrake`.
- **Cache invalidation** (in `app.rs::invalidation_subscriber_loop`):
  management mutations publish on a Redis pub/sub channel
  (`oxllm:invalidate`). All replicas subscribe and clear their local moka
  caches on receipt. This touches the `RedisCache` directly, not the
  `Coordinator`.
- **AlertState** (in `hooks/alerting.rs`): fired thresholds are stored in
  Redis with TTLs matching the budget window. Alerts are deduplicated across
  replicas. `AlertState` carries its own `cache`/`distributed` fields rather
  than routing through the `Coordinator`.
- **JWT jti** (in `auth/jwt.rs`): optional replay protection tracks seen
  token IDs in Redis with TTL matching token lifetime, via
  `JwtValidator::jti_cache`. If the shared cache is unreachable the check
  fails closed but as a 503 server error (not a 401), and it does not count
  toward the auth brake — mirroring the DB-outage handling in
  `auth_middleware`.

> Note: despite its name, the `Coordinator` currently backs **AuthBrake
> only**. New distributed features should NOT be routed through it
> reflexively — wire them to the relevant subsystem (cache invalidation /
> AlertState / JWT validator) directly, or fold the coordination into the
> `Coordinator` so the name is earned.

---

## Optional features

| Feature / crate | Purpose |
|-----------------|---------|
| Root defaults | No root feature flags; dependency defaults still enable server `redis`/`platform-apis` and all providers |
| Server `redis` | Real Redis cache, counters and pub/sub; enables `oxllm-cache-redis/redis` and the server's Redis dependency; enabled by default |
| Server `platform-apis` | Platform routes/handlers (batches, files, assistants, threads, fine-tuning, token counter); enabled by default |
| `db` (root → server) | Virtual keys, teams, users, orgs, projects, spend logs, DB-stored models; enables the optional DB/sqlx dependencies and SQLite/Postgres backends |
| `otel` (root → server) | OpenTelemetry export and its optional dependencies |
| `s3-cache` / `gcs-cache` / `azure-cache` (root → server) | Enable the corresponding optional blob-cache backend |
| `admin` (root) | Build `oxllm-admin` against the management API; enables the separate admin crate, not proxy DB persistence |
| Provider features | 34 provider flags in `oxllm-providers` (all on by default), plus the shared `openai_compat` helper feature |

The root does not forward provider, `redis`, or `platform-apis` switches, so
root `--no-default-features` alone does not turn off those dependency defaults.
Custom hosts selecting a smaller surface must disable defaults on their server
and provider dependencies and choose features there. Workspace integration tests
enable server `db` and cache `redis` through root dev-dependencies.

---

## Tests and quality

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
cargo deny --locked check
```

- **Unit tests** live next to code (`#[cfg(test)]` modules).
- **Integration / e2e** live in workspace `tests/` (mostly httpmock
  upstreams). Files are named by topic (see
  [`tests/README.md`](../tests/README.md)), not by historical milestones.
- **Shared harness**: `tests/common/mod.rs` provides `OxllmHandle`,
  `start_oxllm` (in-process boot), and `proxy_config` helpers — new
  integration tests should opt in via `mod common;`.
- **Benchmarks**: criterion benches in `crates/oxllm-core/benches/`
  (cost, router, tokens) and `crates/oxllm-providers/benches/` (anthropic).
- **Admin UI browser e2e**: Playwright in `tests/admin-ui/` (not cargo).
- **Layering** must pass: `tests/workspace_layering.rs`.
- CI: `.github/workflows/ci.yml` (fmt, clippy, test, deny, feature matrix).

---

## Conventions (when editing)

- No `unwrap` / `expect` / `panic` on user or provider input — return typed errors.
- Match OpenAI wire format for client-facing types.
- Secrets never logged; use `tracing` with redaction.
- Prefer provider **official API docs** over copying LiteLLM. LiteLLM may live
  at `../litellm` as a **behavior reference only** — never paste its code.
- Comments only when they add non-obvious intent (project prefers little noise).

---

## Where to change what

| Goal | Start here |
|------|------------|
| New LLM provider | `oxllm-providers` + feature flag ([guide](adding-a-provider.md)) |
| New routing strategy | `oxllm-core/src/router/` + parse in `oxllm-server/src/state/router_config.rs` |
| New HTTP route / product API | `oxllm-server` routes + handlers (+ pipeline if shared lifecycle) |
| New hook (budget / RPM / cache behavior) | `oxllm-server/src/hooks/` + wire in `state/hook_builders.rs` |
| New guardrail | `oxllm-server/src/guardrails/` + `GuardrailsConfig` + `AppState` field |
| New config option | `config/types/` + wire in `state/build.rs` |
| New budget scope | `oxllm-core/src/limits.rs` + `CallContext` + `hooks/budget.rs` + `spend_keys.rs` |
| Price data | `oxllm-core/data/prices/<provider>.toml` (one file per provider) |
| Management / admin | `handlers/management/` + `routes/management.rs` and/or `oxllm-admin` |
| Persistence schema | `oxllm-db/src/migrate.rs` (`run_all_migrations`, numbered DDL batches) |

If something needs both pure logic and I/O, put the pure part in core (or a
pure module) and keep network/env/fs in the server. See
[adding-a-feature.md](adding-a-feature.md) for the full decision guide.
