# Multi-Instance Stress Test Plan (2 × oxllm + Redis)

> **Historical design and August 2026 run notes.** The original two-replica
> topology and development proposals are retained below. Use the checked-in
> [three-replica harness](../../tests/multi-instance/README.md) and
> [operations guide](../operations.md) for current instructions. Historical
> observations are not current-run results or release certification.

Status: historical plan with current implementation notes. The original
review-time restriction on application changes is part of the design history,
not an outstanding prerequisite for running today's harness.

## Current implementation and runnable harness

- `round_robin` is implemented: weighted rotation has per-model-group counters
  within each replica, preserved across config reloads. It is a separate choice
  from `simple_shuffle` and `session_affinity`. The checked-in config selects
  `session_affinity`; nginx rotates across replicas, and session-less oxllm
  requests fall back to shuffle.
- Model management writes publish `__model_list__` on the namespaced Redis
  invalidation channel. The production subscriber debounces refreshes for
  300 ms and also polls every 30 seconds. Model add/delete/burst propagation is
  covered by `tests/cross_replica_model_propagation_e2e.rs`; production local
  cache invalidation and subscriber cancellation are covered by
  `tests/redis_invalidation_e2e.rs`.
- The Compose stack has **three replicas**, Redis, Postgres, mock upstream,
  nginx and Prometheus/Grafana. Its `migrate` init service runs before replicas,
  which set `OXLLM_AUTO_MIGRATE=false`. Standalone `serve` still defaults to
  auto-migration; concurrent startup migrations serialize.
- Host-side scenarios require `httpx` from
  `tests/multi-instance/requirements.txt`. Optional `asyncpg` is declared in
  `requirements-postgres.txt` (which includes the base requirements) and is
  needed to evaluate SQL spend parity. The mock's separate `aiohttp` dependency
  does not install the scenario driver's dependencies.
- Scenario **FAIL or SKIP exits non-zero by default**. `--allow-skips` permits
  unevaluated checks but never failures. Missing `asyncpg`, incomparable routing
  signals or an invalid cost-counter measurement window can skip a check;
  configured HTTP/Postgres failures fail. Rust Redis regressions skip only when
  `OXLLM_REDIS_URL` is unset/empty; configured connection/ping/subscription or
  propagation failures fail.
- `scenarios.py` covers config parity, unary cache sharing, sticky deployment
  consistency, fleet deployment RPM caps and SQL/metric spend parity. It does
  not implement every scenario below. `make soak` is a 10,000-request unary
  chat workload, not the planned timed mixed-workload soak.
- SQLite supports concurrent processes within its locking constraints. The
  blanket corruption claims in the original proposal were too strong. A
  non-fatal startup diagnostic exists; this harness and the chart's external
  multi-replica persistence path use Postgres.
- Prometheus's built-in target `instance` label distinguishes replicas.
  Compose exports `OXLLM_INSTANCE_ID`, but the proxy does not currently consume
  it as a global log/metric label; that part of the telemetry proposal remains
  a proposal.

---

## 1. Verdict: does the application have the components needed?

The original assessment was **mostly yes**, with three development items in
§7. Round-robin, model propagation and the SQLite diagnostic are now
implemented; the historical reasoning and broader scenario matrix remain
useful for planning future stress runs.

### 1.1 What is already fleet-correct (Redis-backed)

| Component | Where | Behaviour across 2 instances |
|---|---|---|
| Response cache + counters | `cache/mod.rs` `DualCache` | Redis backs shared cache blobs and counters; memory is a per-instance front layer cleared by pub/sub invalidation |
| Rate limits (RPM/TPM) | `pipeline/mod.rs` `check_deployment_limits`, `incr_if_below` on DualCache | `dep_rpm:{dep_key}` / `dep_tpm:{dep_key}` counters are shared — a 100 RPM limit is 100 fleet-wide, not 100 per instance |
| Router-level budgets | `orchestration/budget.rs` reserve/settle on cache counters | Per-deployment and per-provider spend caps enforced atomically via shared Redis counters (`incr_if_below`) |
| Per-key/team/user budgets | `hooks/budget.rs` | Same shared-counter mechanism |
| Cache invalidation | `app.rs::invalidation_subscriber_loop`, `oxllm-cache-redis` pub/sub | Management invalidations clear peer memory caches; `redis_invalidation_e2e.rs` runs the production loop |
| Semantic cache | `cache/semantic.rs` + Redis mirror | Shared across replicas; survives restart |
| Spend persistence | `spend_writer.rs` (batched) → DB via sqlx | Both instances write to the same DB |
| Config identity | shared config file | Both instances run the same deployments (mount the same config) |
| CLI | `oxllm serve --config X --port N` | Any number of instances; port/env configurable (`OXLLM_PORT`) |

### 1.2 What is per-instance (local state — by design, but affects fleet behaviour)

- **Router state** — latency samples, in-flight counts, round-robin counters
  and adaptive posteriors are per replica. Shared Redis usage counters and
  deterministic session-affinity hashing have different coordination semantics;
  deployment round-robin is not a single fleet-wide alternating sequence.
- **`adaptive_router` bandit table** — per-process cell
  table; adaptation state is fragmented across instances (acceptable for
  short sessions, worth noting).
- **Config hot-reload** file watcher — per instance over a shared config source;
  filesystem notification delivery is platform-dependent (see the historical
  bind-mount observation in §9.4), not a synchronized fleet transaction.
- **Prometheus `/metrics`** — per instance (aggregate in Prometheus using its
  built-in scrape-target `instance` label; custom identity proposal in §7.3).
- **Realtime WebSocket sessions** — per instance (out of scope for this run).

### 1.3 Original development gaps and current status

1. **Round-robin — implemented.** The proposal distinguished random weighted
   shuffle from true rotation. `round_robin` now rotates through weighted slots
   with per-group counters and regression coverage.
2. **Model propagation — implemented.** Management model mutations publish the
   `__model_list__` sentinel; peer subscribers trigger debounced
   `refresh_db_models`, with periodic polling to recover missed notifications.
3. **SQLite fleet diagnostic — implemented.** The proposal recommended
   Postgres and requested a startup diagnostic for SQLite writers. The
   diagnostic is present; SQLite contention is a locking concern, not proof
   that every shared-file deployment corrupts data.

Custom `OXLLM_INSTANCE_ID` instrumentation (§7.3) remains a telemetry proposal.
Current Prometheus target labels already distinguish instances.

---

## 2. Original target topology (two replicas)

This diagram and the sketches below document the original proposal. The
checked-in three-replica files under `tests/multi-instance/` are authoritative
for runnable config, service ports, migration ordering and health checks.

```
                        ┌─────────────────────────────┐
   loadgen ───────────▶ │  LB (nginx round-robin)     │  :8080
    (oha / python)      └───────┬───────────┬─────────┘
                                │           │
                     ┌──────────▼──┐   ┌────▼──────────┐
                     │ oxllm A    │   │ oxllm B      │
                     │ :4000      │   │ :4001        │
                     └──┬────┬────┘   └──┬────┬──────┘
                        │    │          │    │
             ┌──────────▼┐  ┌▼──────────┐  ┌▼──────────┐
             │ redis:7   │  │ postgres:16│  │ mock LLM  │
             │ :6379     │  │ :5432     │  │ upstream  │ :9000
             └───────────┘  └───────────┘  └───────────┘
   observability: prometheus (:9090), grafana (:3000),
                  jaeger/otel-collector (optional tracing)
```

- Both oxllm instances: **same image** (local `Dockerfile`, features
  `db,otel`), **same mounted config file**, different ports.
- Mock upstream (`mock-llm`): a tiny container exposing an OpenAI-shaped
  `/v1/chat/completions` (+ `/v1/embeddings`, streaming SSE) with
  configurable latency/errors — **never hammer a real provider API for
  sustained load**.
- LB is optional for instance-level distribution (we can also drive the
  loadgen to hit both ports directly 50/50 — see §4.2 for why both are
  tested).

### 2.1 Historical docker-compose sketch (superseded by checked-in infra)

```yaml
# Original proposal; use tests/multi-instance/docker-compose.yml to run the stack.
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: ["redis-server", "--appendonly", "yes"]

  postgres:
    image: postgres:16-alpine
    environment: { POSTGRES_PASSWORD: oxllm, POSTGRES_DB: oxllm }
    ports: ["5432:5432"]

  mock-llm:
    build: ./mock-llm            # tiny OpenAI-compatible mock (Python)
    ports: ["9000:9000"]
    environment: { LATENCY_MS: "150", STREAM: "1" }

  oxllm-a:
    build: { context: ../.., dockerfile: Dockerfile }
    command: ["serve", "--config", "/etc/oxllm/config.yaml", "--port", "4000"]
    environment: { OXLLM_INSTANCE_ID: "a" }
    volumes: ["./config.yaml:/etc/oxllm/config.yaml:ro"]
    depends_on: [redis, postgres, mock-llm]
    ports: ["4000:4000"]

  oxllm-b:
    build: { context: ../.., dockerfile: Dockerfile }
    command: ["serve", "--config", "/etc/oxllm/config.yaml", "--port", "4001"]
    environment: { OXLLM_INSTANCE_ID: "b" }
    volumes: ["./config.yaml:/etc/oxllm/config.yaml:ro"]
    depends_on: [redis, postgres, mock-llm]
    ports: ["4001:4001"]

  lb:
    image: nginx:alpine
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf:ro"]
    ports: ["8080:80"]

  prometheus:
    image: prom/prometheus
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml:ro"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana-oss
    ports: ["3000:3000"]
```

Original shared-config sketch (not a current runnable config; §8 records the
cache nesting and mock-provider corrections made when building the infra):

```yaml
general_settings:
  master_key: "sk-stress"
  database_url: "postgres://postgres:oxllm@postgres:5432/oxllm"
  otel_endpoint: "http://otel-collector:4317"
  otel_service_name: "oxllm"
  log_format: "json"

cache:
  type: "redis"
  redis_url: "redis://redis:6379"
  semantic_cache:
    enabled: true
    embedding_model: "mock/mock-embed"
    similarity_threshold: 0.95
    ttl_secs: 3600
    redis_url: "redis://redis:6379"

model_list:
  - model_name: mock-model-a
    oxllm_params: { model: "mock/llama-3", api_base: "http://mock-llm:9000/v1", api_key: "mock" }
  - model_name: mock-model-b
    oxllm_params: { model: "mock/llama-3", api_base: "http://mock-llm:9000/v1", api_key: "mock", weight: 1 }
    # + a second deployment for load-balancing assertions
router_settings:
  routing_strategy: round_robin   # Proposed then; implemented now. Actual stack uses session_affinity.
  routing_groups: { mock-group: [mock-model-a, mock-model-b] }
```

### 2.2 Tooling

| Tool | Purpose | Install |
|---|---|---|
| Docker / Compose | Topology | already present |
| `oha` (or `hey`) | Raw RPS + latency percentiles, POST bodies, streaming | `brew install oha` |
| Python 3 + httpx; optional asyncpg | Host scenario driver and SQL spend verification | venv + `tests/multi-instance/requirements-postgres.txt`; base HTTP deps in `requirements.txt` |
| Prometheus + Grafana | Instance-level metrics (requests, latency, tokens, cost, errors) | via compose |
| Jaeger / otel-collector | Distributed trace across both instances (`otel` feature) | via compose (optional) |
| `redis-cli` (in redis container) | Key inspection: `oxllm:*` counters, cache TTLs, collision checks | via container |
| `psql` (in postgres container) | Spend-log verification: totals, per-instance rows, no double counting | via container |
| curl/jq | Ad-hoc probes (health, models, cache invalidation admin API) | present |

---

## 3. Correctness invariants to verify (the "modes of usage and data")

These are the planned pass/fail gates for the full matrix, not a claim that
today's short scenario driver implements or has passed every gate:

1. **Fleet-wide rate limits**: `rpm: 100` on a deployment → 100 requests per
   60s succeed across BOTH instances combined; 101st → 429 (verify the 429
   arrives even when the two instances each saw ~50).
2. **Fleet-wide budgets**: per-deployment `max_budget` counter enforced across
   instances (reserve/settle math stays consistent; no double counting).
3. **Shared cache**: a request answered by A with a cache hit (TTL>0) is then
   served as a HIT by B without calling upstream (mock counters on
   `mock-llm` prove it); admin cache-invalidate on A removes the entry on B
   (pub/sub).
4. **No cross-instance double spend**: total spend in Postgres == sum of what
   A and B recorded on their own `/metrics` cost counters (within floating
   tolerance).
5. **Deterministic routing**: with `session_id` set, the same session picks
   the same deployment no matter which instance serves it (session affinity
   is a pure fn of session_id + candidate list — assert identical routing
   decisions on A and B).
6. **Identical behaviour**: both instances serve the same `/v1/models`, same
   `/health` status, same cache settings (config parity).
7. **Graceful degradation**: Redis down → cache/limits fail in the designed
   mode (documented behaviour, no panic, no crash loop); Redis back →
   recovery without restart.
8. **Streaming integrity**: SSE chunk counts and content identical whichever
   instance streams it.
9. **Load-balancer fairness**: request distribution across instances is
   balanced (nginx round-robin) and across deployments within an instance is
   balanced per the strategy (round-robin / weighted).

---

## 4. Scenario matrix

### 4.1 Phase 0 — bring-up (30 min)

- Build image, `docker compose up`, instances healthy (`/health` checks
  liveness; `/health/readiness` checks dependencies), shared config validated.
  The checked-in stack starts three replicas and uses an init migration step.
- `oxllm validate --config` on the shared config.
- Smoke: 1 chat + 1 stream + 1 embed through the LB; through each instance
  directly.

### 4.2 Phase 1 — functional correctness, two instances + Redis (2–3 h)

Drive traffic in two ways: **(a)** through the LB (fleet view), **(b)** split
directly 50/50 A/B (isolates per-instance behaviour and proves the fleet
semantics are the instances', not the LB's).

1. **Deployment round-robin**: configure `round_robin` and send sequential
   uncached traffic directly to each replica. Equal-weight candidates must
   strictly rotate within a group; weighted slots give the configured ratio.
   Shuffle distribution needs a statistical bound instead. The default
   harness uses `session_affinity`, so its LB/shuffle observations do not prove
   strict round-robin alternation.
2. **Rate limit fleet test** (invariant 1): rpm 100, fire 250 at the LB.
3. **Budget fleet test** (invariant 2): `max_budget` small; cross the limit.
4. **Cache sharing** (invariant 3): warm on A, hit on B; TTL expiry both
   sides; invalidate from A's admin API → miss on B.
5. **Semantic cache sharing**: near-duplicate prompts → hit on the other
   instance (embedding via mock upstream).
6. **Session affinity across instances** (invariant 5).
7. **Spend double-count check** (invariant 4): run a fixed workload, compare
   `/spend` DB totals vs per-instance metrics.
8. **Config parity** (invariant 6): models/health/cache settings identical.
9. **Management API cross-instance** (implemented; separate Rust regression): add model on A →
   visible on B within the propagation window; delete on A → gone on B.

### 4.3 Phase 2 — load & stress (2–4 h)

| Scenario | Method | Accept criteria |
|---|---|---|
| Throughput sweep (50/100/250/500 RPS, 60s each) | `oha` POST chat through LB | p50/p95/p99 recorded; error rate < 1% (excluding intentional 429s); per-instance split balanced |
| Streaming under load (100 concurrent SSE) | `oha -s` / python | all streams complete `[DONE]`; no mid-stream errors; chunk integrity on sampled streams |
| Cache-heavy workload (90% repeat prompts) | python loadgen | cache hit ratio ≥ 80% on `/metrics` (both instances); upstream hits stay flat |
| Mixed workload (chat 70% / stream 20% / embed 10%) | python loadgen | invariants 3–5 hold under load; no latency cliff |
| Concurrency spike (500 concurrent, short bursts) | `oha`/python | no 5xx from either instance; queueing visible in p99 but no drops |
| Soak (30–60 min sustained mixed load) | python loadgen | flat error rate, bounded memory (RSS graphs), no counter drift between A and B, Redis memory bounded, spend total matches metrics at the end |

### 4.4 Phase 3 — failure injection (1–2 h)

1. **Redis down**: stop redis → observe request behaviour (documented
   fail-open/closed per component: counters, cache), no crash; restart redis
   → counters/cache recover without restart. Then re-run invariant checks.
2. **One instance down**: kill B → LB drains (nginx retries / health checks)
   → all requests served by A with no 5xx; restart B → rejoins cleanly.
3. **Postgres down**: spend writer must queue/drop gracefully, not crash
   handlers (handlers never touch DB on hot path by design).
4. **Mock upstream down**: deployment fails over to the other deployment in
   the group (fallback logic) — 5xx from upstream surfaced as UpstreamError,
   retries within limits.

### 4.5 Phase 4 — issue triage loop

Every finding (bug, misconfiguration, unexpected behaviour) goes through:

1. Reproduce with the smallest e2e test (extend `tests/` once the code review
   is merged — **no code changes during review**).
2. Classify: correctness (spend/limits/cache) / availability (failover) /
   performance (latency, throughput) / observability.
3. Fix + regression test → commit → re-run the affected invariant.
4. Track in a findings table (component, scenario, symptom, root cause,
   fix).

---

## 5. Observability & detection

- **Per-instance `/metrics`** (prometheus): `oxllm_requests_total` (model,
  provider, status), `oxllm_request_duration_seconds` histogram,
  `oxllm_tokens_total`, `oxllm_cost_total`, `oxllm_errors_total`. Aggregate
  in Prometheus; compare A vs B; watch for asymmetry as a smell.
- **Redis**: `INFO memory`, `KEYS oxllm:*` inventory (prefix hygiene), TTL
  audits, `redis-cli --latency`.
- **Postgres**: `spend_logs` row counts vs metrics; duplicate `request_id`
  check (double-write detector); growth rate.
- **Logs**: JSON logs (config `log_format: json`), attributed by container/service
  identity. `OXLLM_INSTANCE_ID` is exported by Compose but not currently read
  by the proxy; global identity fields remain proposed in §7.3. Inspect warnings
  for dropped spend entries and cache fallbacks.
- **Tracing (optional)**: otel → jaeger to follow one request across LB →
  instance → Redis → upstream and confirm spans on both instances.
- **Health probes**: `/health/readiness` checks DB/Redis; `/health` is liveness.
  The checked-in nginx uses passive connection-failure checks, not active
  readiness probes. Its Compose health check probes liveness through the LB.

---

## 6. Exit criteria

- All §3 invariants pass in both direct and LB-driven traffic.
- No unknown 5xx, no counter/cost drift after the soak, no Redis key
  collisions, no memory growth over the soak window.
- Every failure-injection scenario recovers as designed (documented
  behaviour, not crashes).
- Any bugs found are triaged and (post-review) fixed with regression tests.
- Docs explain the Redis counter requirement and the Postgres-backed example
  topology, SQLite locking constraints and how to run the actual Compose stack.

---

## 7. Original development proposals (with current status)

The proposals below are retained as design history. The original review-time
deferral applied when the plan was written; implemented items are no longer
prerequisites for the current harness.

### 7.1 New `round_robin` routing strategy (HIGH — user story asks for it)

**Implemented.** The current counter is per model group within each replica,
not a single per-strategy index; it survives reloads. Explicit `weight`, then
`rpm`, then `tpm`, then 1 determines weighted slots. Original proposal:

- `crates/oxllm-core/src/router/round_robin.rs`: rotating pick over the
  stably-sorted candidate list using a per-strategy atomic index (fetch_add
  % len), weight-aware option (round-robin over the weighted expansion or
  weighted rotation). Local state is fine — the LB splits traffic; each
  instance rotates its own share (same as LiteLLM per-worker behaviour).
- Wire into strategy name parsing (`router_settings.routing_strategy`),
  unit tests (alternation, weight bias, empty list), and a
  `load_balancing_e2e` extension asserting strict alternation.

### 7.2 Cross-instance management propagation (HIGH for the "everything works" story)

**Implemented.** The actual event is `__model_list__`, with a 300 ms debounce
and a 30 s fallback poll. The current Redis-opt-in regression shares an
in-memory SQLite DB between two in-process replicas. The earlier proposal
below named a different event and suggested Postgres for the test topology:

- After successful management writes (`POST /model/new`, `/model/update`,
  `/model/delete`, and the budget/key admin writes that change routing
  state), `publish` a `model-list-invalidated` event on the existing Redis
  pub/sub channel; every instance's subscription triggers
  `refresh_db_models`.
- Fallback: 30s DB poll when Redis pub/sub is unavailable.
- E2E test: two servers sharing Redis + Postgres; add model on A, assert
  visible on B.

### 7.3 Instance identity in telemetry (LOW, needed for clean soak analysis)

**Still a proposal for proxy-wide fields.** Compose exports the variable but
the proxy does not read it; Prometheus's existing target `instance` label
already separates replicas. Original proposal:

- `OXLLM_INSTANCE_ID` env (default hostname) → JSON log field on every
  record and `oxllm_instance` label on all `/metrics` series.

### 7.4 SQLite multi-instance guard (LOW)

**Implemented as a non-fatal startup diagnostic.** Its original blanket
"unsafe" framing should not be treated as evidence of universal corruption;
concurrent writers must respect SQLite's locking constraints. Original proposal:

- Startup warning when `database_url` uses `sqlite://` (and more than one
  instance is implied — or simply always, pointing at Postgres for fleets).

### 7.5 Test infra (this plan's artifacts, not app code)

**Implemented, with three replicas and an init migration service.** The
checked-in README and dependency files describe current coverage and explicit
skip policy. The full planned matrix still exceeds `scenarios.py` coverage.

- `tests/multi-instance/`: docker-compose.yml, nginx.conf, prometheus.yml,
  mock-llm container, config.yaml, scenario scripts, README with run
  instructions.

---

## 8. Findings discovered while building the infra (important context)

These were caught by validating the config and running a local wiring smoke
test (oxllm + local Redis + the mock, no Docker). They affect how the tests
must be written and are recorded here so the development phase is accurate.

1. **Cache config nests under `oxllm_settings`, NOT `general_settings` and
   NOT a top-level `cache:` key.** A top-level `cache:` is *silently ignored*
   (`unknown top-level config key, ignored`) — Redis caching would be
   silently disabled. The `cache` and `cache_ttl_secs` fields live on
   `OxllmSettings`. The infra `config.yaml` is correct; flag for the
   SQLite guard work (§7.4): consider also warning on a top-level `cache:`
   key since silent ignore is a sharp edge.

2. **Response caching is opt-in per request (LiteLLM convention).** The
   client must send a `"cache": {"insert": true, ...}` object in the request
   body (`request_wants_cache` in `hooks/cache_hook.rs`); identical prompts
   without it are NOT cached. Every scenario/loadgen that asserts cache
   behaviour must include the `cache` opt-in. Documented in the scenario
   script.

3. **Cache hits are detected via the `oxllm_prompt_cache_hits_total` metric,
   not a response-body marker.** The hit returns the stored response verbatim
   (original `id` included), so assert via a metrics counter delta, not a body
   field. The scenario script's cache invariant uses this. The current proxy
   also exposes the `x-oxllm-cache-hit` response header; neither signal proves
   live stream capture. Live streams are not written to the response cache today.

4. **At bring-up, `serve` auto-migrated Postgres on startup.** That remains the
   standalone default, with concurrent migrations serialized. The current
   Compose stack instead runs `oxllm migrate` in an init service and disables
   auto-migration on all three replicas; the earlier no-init-step observation
   describes the historical setup, not today's Compose file.

5. **Prometheus `/metrics` is master-key gated.** The scrape config must pass
   `bearer_token: "sk-stress"`. Prometheus's built-in `instance` (host:port)
   label already distinguishes the two instances, so gap §7.3 is lower
   priority than first thought.

6. **`openai/<model>` with an `api_base` override is the clean mock-entry
   path** — the `openai` provider honours `api_base` (`unwrap_or(default)`)
   and speaks the exact OpenAI shape the mock implements. Confirmed working
   end-to-end against the mock.

---

## 9. Testing results (2026-08-02)

### 9.1 Tests passed

| Invariant | Result | Method |
|---|---|---|
| Round-robin strategy | ✅ | `x-oxllm-router-strategy: round_robin` header on every response |
| Cache sharing (A→B) | ✅ | Warm on A → B logs "response cache hit"; mock not called; metric delta confirmed |
| Fleet rate limits | ✅ | 250 requests → 100×200 + 150×429 (was 404 before fix) |
| Instance failover | ✅ | Killed oxllm-b → 10/10 succeed via A; restart → rejoins |
| Cross-instance model add | ✅ | POST /model/new on A → visible on B <2s via pub/sub → request succeeds via B |
| Session affinity | ✅ | 8 sessions → 4/4 split across deployments; same session always same dep |
| Load (500 req) | ✅ | 500/500 success, 3s, 0 errors |
| Memory | ✅ | ~10–18 MiB per instance (within 512 MiB cap) |
| Prometheus | ✅ | Both instances scraped, `instance` label splits A/B |
| Redis outage fail-open | ✅ | First request succeeds in ~644ms (was infinite hang) |

The historical round-robin row establishes that the strategy header was
selected; a header alone does not demonstrate strict alternation. Current
router unit regressions test rotation, weights and per-group counter/reload
behaviour. None of these historical rows certify a fresh run of today's stack.

### 9.2 Bugs found and fixed

1. **Rate-limited requests returned 404 instead of 429** (commit `7d9640d`):
   when all deployments were rate-limited/cooled, `pick_deployment_index`
   returned None on the first attempt, `last_err` stayed None, and the
   fallback `ApiError::model_not_found` fired. Fixed to check if the model
   has deployments and return 429 when they're just unavailable.

2. **Redis outage hung requests indefinitely** (fixed this session): Redis
   operations had no per-operation timeout; the `ConnectionManager` retried
   forever. Added `REDIS_OP_TIMEOUT` (500ms) to every RedisCache operation
   (`get`, `set`, `delete`, `incr`, `incr_if_below`, `ping`).

3. **Rate limiter returned 500 on cache failure** (fixed this session):
   `check_deployment_limits` returned `ApiError::internal` when Redis was
   unavailable, rejecting all traffic. Changed to fail-open (log warning,
   allow request) — rate limits are a protection, not a correctness
   requirement.

4. **Dockerfile ENTRYPOINT/HEALTHCHECK hardcoded port 4000** (fixed this
   session): `ENTRYPOINT ["/oxllm", "serve", ...]` and `HEALTHCHECK ... :4000`
   made multi-port deployments impossible without `entrypoint:` overrides.
   Fixed: `ENTRYPOINT ["/oxllm"]`, `CMD ["serve", ...]`, `ENV OXLLM_PORT=4000`,
   HEALTHCHECK uses `${OXLLM_PORT}`.

### 9.3 Config sharp edges (documented, not bugs)

- **`rpm` serves as routing weight**: when `weight` is not explicitly set,
  `rpm` is used as the SimpleShuffle weight. A deployment with `rpm: 100`
  and no explicit `weight` gets weight 100, while a sibling with `weight: 1`
  gets weight 1 — a 100:1 distribution skew. Always set explicit `weight`
  when using `rpm` for rate limiting with multiple deployments.

### 9.4 Gap closure (round 2)

All initially-untested areas were addressed in a follow-up round:

| Gap | Result | Method |
|---|---|---|
| Budget enforcement | ✅ | DeepSeek v4 Flash, $0.01 budget: req 1→200, req 2→429 "deployment budget exceeded". Reservation mechanism blocks early. |
| Semantic cache cross-instance | ✅ | Primed on A, similar (not identical) prompt on B+C → both logged "semantic cache hit", metric delta confirmed. Redis-backed. |
| Spend logs in Postgres | ✅ | Direct SQL: 2526 entries from all instances, $0.012639 total, consistent. |
| Config hot-reload | ⚠️ | File watcher runs ("config hot-reload watcher started") but Docker bind mounts on macOS don't propagate inotify events. Works on restart and on native Linux. |
| Soak test | ✅ | 2000 requests / 4.7 min: 100% success, 0% errors. Memory stable (10.5→9.3 MiB, no leak). Distribution 663/668/669. |
| Embeddings through stack | ✅ | All 3 instances returned 200, 64-dim vectors, batch mode works. |
| `drop_params` | ✅ | `drop_params: true` silences "unsupported param" warnings (0 after, was 5+ before). |
| Claude Code integration | ✅ | Claude Code → oxllm → DeepSeek v4 Flash: full session succeeded (repo analysis), Anthropic↔OpenAI format translation, streaming, tools, $0.007 total (98% cheaper than Claude). |

---

## 10. Original execution order (historical)

The implementation prerequisites in this sequence have largely landed. Use
the current harness README to start a new run; retain this sequence as the
original rollout/triage plan rather than an instruction to redo closed work.

1. Merge code review → implement §7.1–7.4 (with tests) → commit.
2. Build §7.5 infra; `docker compose up`; Phase 0 bring-up.
3. Phase 1 functional invariants (both LB and direct traffic).
4. Phase 2 load/stress + soak; Phase 3 failure injection.
5. Triage loop; fix + regression tests; re-run.
6. Document results + findings table + docs updates (§6).
