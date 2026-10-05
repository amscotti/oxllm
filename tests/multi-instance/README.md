# Multi-instance stress test stack

Three oxllm instances behind nginx, sharing Redis + Postgres, with a mock
OpenAI upstream and Prometheus/Grafana. The plan and the full invariant list
live in [`../../docs/plans/multi-instance-stress-test-plan.md`](../../docs/plans/multi-instance-stress-test-plan.md).

This directory is test infrastructure. The original plan's `round_robin`
strategy, cross-replica DB-model propagation and SQLite startup diagnostic are
implemented. The checked-in stack uses `session_affinity` inside oxllm and
nginx round-robin across replicas; session-less requests fall back to shuffle.
Prometheus distinguishes replicas by its scrape-target `instance` label.
The historical plan and run notes are not evidence that a current run passed.

## Layout

```
tests/multi-instance/
  docker-compose.yml   topology (redis, postgres, mock-llm, oxllm-a/b/c, lb, prometheus, grafana)
  nginx.conf           round-robin LB, streaming-friendly, passive health checks
  prometheus.yml       scrapes all instances' /metrics (master-key bearer)
  config.yaml          shared oxllm config (mock upstream, redis cache, pg db, semantic cache)
  mock-llm/            tiny OpenAI-compatible async mock (chat + stream + embed)
  scenarios.py         functional-invariant driver (cache sharing, fleet rate limit, ...)
  requirements.txt     required host-side driver dependency (httpx)
  requirements-postgres.txt  optional SQL verification dependency (asyncpg) + base requirements
  Makefile             build / up / down / probes / scenario / soak
  README.md            this file
```

## Quick start

```sh
python3 -m venv tests/multi-instance/.venv
. tests/multi-instance/.venv/bin/activate
python -m pip install -r tests/multi-instance/requirements-postgres.txt
make -f tests/multi-instance/Makefile build     # one-time image build
make -f tests/multi-instance/Makefile up         # start the stack
make -f tests/multi-instance/Makefile status     # inspect service health before probes
make -f tests/multi-instance/Makefile probes     # smoke: chat / stream / models
make -f tests/multi-instance/Makefile scenario   # run the invariants
```

Endpoints after `up`:

| Service | URL | Notes |
|---|---|---|
| LB (nginx) | http://127.0.0.1:8080 | drive loadgen here |
| oxllm-a | http://127.0.0.1:4000 | direct, per-instance checks |
| oxllm-b | http://127.0.0.1:4001 | direct |
| oxllm-c | http://127.0.0.1:4002 | direct |
| Prometheus | http://localhost:9090 | all three instances scraped, `instance` label splits A/B/C |
| Grafana | http://localhost:3000 | anonymous admin; add Prometheus datasource |
| Redis | localhost:6379 | `docker compose exec redis redis-cli` |
| Postgres | localhost:5432 | `docker compose exec postgres psql -U postgres oxllm` |
| mock-llm | http://localhost:9000 | `?latency_ms=...`, `?status=503`, `?fail_after=N` for fault injection |

Master key for all oxllm calls: `sk-stress-master-key` (≥16 bytes; the production master-key length guard rejects shorter values).

## Dependencies and skip policy

`httpx` is required by `scenarios.py`; install `requirements.txt` for the HTTP
checks alone. The mock container has its own `mock-llm/requirements.txt`
(`aiohttp`), which does not supply host-side driver dependencies. Install
`requirements-postgres.txt` to include optional `asyncpg` and enforce the SQL
spend-parity check. `make scenario PYTHON=/path/to/venv/bin/python` selects an
explicit interpreter.

Every scenario reports **PASS**, **FAIL**, or **SKIP**. The default exit status
is non-zero for either FAIL or SKIP. Missing `asyncpg`, absent comparable
deployment signals, reset cost counters, or no measurable cost delta leave a
check unevaluated and produce SKIP. `--allow-skips` explicitly permits those
unevaluated checks, while FAIL always remains non-zero:

```sh
make -f tests/multi-instance/Makefile scenario SCENARIO_FLAGS=--allow-skips
```

HTTP failures and a configured but unreachable Postgres are failures, not
skips. The SQL check reads `OXLLM_POSTGRES_URL`, defaulting to
`postgres://postgres:oxllm@127.0.0.1:5432/oxllm` for this stack. The harness uses
IPv4 loopback explicitly so an unrelated IPv6 `localhost` listener cannot
shadow Docker's published ports. Run against an
idle, freshly started stack: the fleet RPM window lasts 60 seconds, and
concurrent traffic or replica restarts invalidate spend comparisons.

The separate Rust Redis integration tests opt in via `OXLLM_REDIS_URL`. Only
an unset/empty URL skips them; a configured connection, ping, subscription,
propagation or cancellation failure fails the test.

## Current harness coverage

- `scenarios.py` checks model-list parity, unary cache sharing, session-affinity
  deployment consistency, fleet deployment RPM caps, and windowed SQL/metric
  spend parity (when `asyncpg` is installed), and unary/streaming tag attribution.
- The tag scenario correlates 12 unary calls and 3 completed streams through
  the LB with exact SQL request IDs, all three normalized tag memberships,
  fixture-rate arithmetic, and identical spend/log/tag reads on every replica.
  A repeated header tag in `metadata.tags` must be deduplicated. Missing
  drop-counter samples are not used as a substitute for durable-row checks.
- The LB distributes traffic across three instances. oxllm sticky routing is
  configured separately via `session_affinity`; this suite does not assert
  strict deployment round-robin alternation.
- The mock upstream returns deterministic OpenAI-shaped responses (incl. SSE
  streaming and embeddings) so cache + spend assertions are exact.
- Prometheus collects per-instance metrics; the default `instance` label
  separates the replicas. Compose also exports `OXLLM_INSTANCE_ID`, but the
  proxy does not currently consume it as a global log/metric label.
- The `migrate` init service runs once against healthy Postgres. Replicas wait
  for its successful completion and set `OXLLM_AUTO_MIGRATE=false`. Standalone
  `serve` defaults to auto-migration when that variable is unset.

## Related production regressions

- `tests/cross_replica_model_propagation_e2e.rs` boots two in-process replicas
  sharing Redis and an in-memory SQLite DB to exercise model add/delete/burst
  propagation through the production subscriber. This is implemented and
  Redis-opt-in, not blocked on plan §7.2; it is not part of `scenarios.py`.
- `tests/redis_invalidation_e2e.rs` exercises the production cache-invalidation
  loop, its subscription-readiness probe and shutdown signal.
- `crates/oxllm-core/src/router/round_robin.rs` pins per-group rotation, weights
  and reload persistence. Counters are per replica, not a fleet-wide sequence.
- SQLite has a startup diagnostic; this stack uses Postgres for persistence.

The short `make soak` target sends 10,000 non-streaming chat requests. It does
not implement the plan's timed mixed-workload soak or its full failure matrix.

## Fault injection

The mock accepts query params on any request:

- `?latency_ms=500` — per-call latency override
- `?status=503` — force a non-200 (upstream-failure / failover scenario)
- `?fail_after=N` — start returning 503 after the Nth request (capacity model)

Redis/Postgres/instance failure is driven from the host:

```sh
docker compose -f tests/multi-instance/docker-compose.yml stop redis
# ...observe behaviour...
docker compose -f tests/multi-instance/docker-compose.yml start redis
```

## Notes

- The mock's embeddings are deterministic hash vectors (identical text →
  identical vector). This exercises the semantic-cache *plumbing* (Redis
  mirror, cross-instance sharing); true semantic similarity is out of scope
  for the mock.
- The migration service and all three replicas use the same repo `Dockerfile`
  (`db,otel`) and can reuse build layers; Compose assigns service image tags.
  Replicas share the mounted config and differ by `OXLLM_PORT` and the exported
  `OXLLM_INSTANCE_ID`.
- `mem_limit: 512m` per instance makes memory growth visible as a fraction of
  the cap during the soak.
