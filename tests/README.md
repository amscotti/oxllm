# Integration tests

Workspace-level e2e tests (`cargo test --workspace`). Each file boots oxllm
in-process against [httpmock](https://docs.rs/httpmock) upstreams unless noted.

| File | What it covers |
|------|----------------|
| `chat_e2e.rs` | Chat completions lifecycle |
| `embeddings_completions_e2e.rs` | Embeddings + legacy completions |
| `azure_ai_embeddings_e2e.rs` | Foundry Cohere image/text routing, mixed-batch order/usage, scalar inputs, cache/budget settlement and fallback/error contracts |
| `messages_responses_e2e.rs` | Anthropic Messages + OpenAI Responses |
| `request_attribution_e2e.rs` | Bounded identities, session/call-id correlation, denial/partial attribution, spend aliases, scoped model errors and free-group admission |
| `router_e2e.rs` | Routing, retries, fallbacks, wildcards, adaptive bandit routing |
| `router_strategies_e2e.rs` | Zero-cooldown loop bounds, 401 sibling failover, round-robin, health-metric label hygiene |
| `session_affinity_e2e.rs` | Sticky `x-oxllm-session-id` routing |
| `load_balancing_e2e.rs` | least_busy, weights, aliases, content-policy fallbacks |
| `deployment_limits_e2e.rs` | Per-deployment RPM/TPM after pick |
| `routing_metrics_e2e.rs` | Routing strategies, metrics, DualCache |
| `metering_e2e.rs` | Cost, budgets, RPM/TPM hooks, response cache |
| `management_e2e.rs` | Virtual keys + management REST (DB) |
| `spend_analytics_e2e.rs` | Spend logs + global analytics |
| `platform_apis_e2e.rs` | Batches, files, fine-tuning, assistants, token counter |
| `passthrough_hooks_e2e.rs` | HookedRaw / Raw passthrough tiers |
| `security_jwt_realtime_e2e.rs` | JWT, realtime WS, body limits, smuggling |
| `callbacks_ops_e2e.rs` | HTTP callbacks, IP allowlist, pass-through endpoints |
| `bedrock_e2e.rs` | Bedrock chat + streaming + Mantle pricing |
| `anthropic_e2e.rs` | Anthropic provider adapter (`/messages`, `x-api-key`, system/stop translation) |
| `openai_e2e.rs` | OpenAI provider adapter (reasoning models, org/project headers) |
| `groq_e2e.rs` | Groq provider adapter |
| `xai_e2e.rs` | xAI provider adapter |
| `deepseek_e2e.rs` | DeepSeek provider adapter |
| `meta_e2e.rs` | Meta provider adapter |
| `fireworks_e2e.rs` | Fireworks provider adapter (chat + embeddings) |
| `perplexity_e2e.rs` | Perplexity provider adapter |
| `cross_replica_model_propagation_e2e.rs` | Model add/delete propagation across replicas via Redis pub/sub (skipped without `OXLLM_REDIS_URL`) |
| `router_stream_feed_e2e.rs` | Streaming traffic feeding the router strategy stores (TTFB, in-flight slots, failure labels) |
| `concurrency_e2e.rs` | Concurrent request handling / race conditions |
| `guardrails_e2e.rs` | Pre-call guardrails (PII, injection, Presidio, AIM, Lakera, …) |
| `langfuse_e2e.rs` | Langfuse observability hook wiring |
| `redis_invalidation_e2e.rs` | Cross-replica cache/auth invalidation via Redis pub/sub |
| `semantic_cache_e2e.rs` | MinHash semantic response cache hits/misses |
| `unicode_e2e.rs` | UTF-8 / multi-byte content through transforms + SSE |
| `vertex_ai_e2e.rs` | Vertex AI (GCP bearer auth) chat + embeddings |
| `config_compat.rs` | LiteLLM YAML fixtures + example config parse |
| `workspace_layering.rs` | Crate dependency DAG (no httpmock) |

Fixtures: `tests/fixtures/`.

## Database contracts

When testing the database crate by itself, enable a backend explicitly; its
default feature set has no database implementation and runs no DB contracts:

```bash
cargo test --locked -p oxllm-db --features sqlite --lib
cargo test --locked -p oxllm-db --features postgres,sqlite --lib
```

`OXLLM_TEST_POSTGRES_URL` selects a disposable PostgreSQL fixture for the opt-in
integration cases, including isolated-schema migrations and spend-writer
regressions. Without it, PostgreSQL cases skip; SQLite contracts still run.

## Live tests (real provider APIs)

The `*_live.rs` files below call **real** upstream APIs and are not part of
the httpmock suite. Each one self-skips (early-`return`) unless its provider
API key is present in the environment — except `e2e_live.rs`, which is
compiled out entirely unless the `live-tests` feature is enabled.

| File | Covers |
|------|----------------|
| `e2e_live.rs` | Full gateway pipeline against a real provider (feature `live-tests`) |
| `anthropic_live.rs` | Live Anthropic smoke (needs `ANTHROPIC_API_KEY`) |
| `openai_live.rs` | Live OpenAI smoke (needs `OPENAI_API_KEY`) |
| `groq_live.rs` | Live Groq smoke (needs `GROQ_API_KEY`) |
| `xai_live.rs` | Live xAI smoke (needs `XAI_API_KEY`) |
| `deepseek_live.rs` | Live DeepSeek smoke (needs `DEEPSEEK_API_KEY`) |
| `meta_live.rs` | Live Meta smoke (needs `META_API_KEY`) |
| `fireworks_live.rs` | Live Fireworks smoke (needs `FIREWORKS_API_KEY`) |
| `perplexity_live.rs` | Live Perplexity smoke (needs `PERPLEXITY_API_KEY`) |
| `bedrock_live.rs` | Live Bedrock smoke (env-only AWS credentials: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION`) |

## Multi-instance stress stack (docker)

Topology-level invariants (fleet rate limits, shared cache, session
affinity, spend parity) across 3 oxllm replicas behind nginx with shared
Redis + Postgres live in `tests/multi-instance/` — see its `README.md` and
`Makefile` (`make -f tests/multi-instance/Makefile up`).

## Admin UI browser e2e (Playwright)

Browser coverage for HTMX/Alpine/CSRF lives in `tests/admin-ui/` (not cargo).
Requires a running local stack (`./scripts/dev-stack.sh`).

```bash
cd tests/admin-ui
npm install
npx playwright install chromium
npm test
```

Coverage matrix + gaps: `tests/admin-ui/COVERAGE.md`.  
Setup notes: `tests/admin-ui/README.md`.

Handler/template coverage: `cargo test -p oxllm-admin` (complements Playwright).
