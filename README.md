<p align="center">
  <img src="assets/logo.png" alt="oxllm — blue ox logo" width="120" />
</p>

<h1 align="center">oxllm</h1>

<p align="center">
A fast, type-safe Rust proxy gateway for LLM APIs.
</p>

oxllm speaks the OpenAI API to your clients and translates requests to any
provider's native format — Anthropic, Gemini, Ollama, and more. Route across
deployments, track cost, enforce budgets, and cache responses, all from a
single native binary with no Python runtime.

> **Pre-release software.** oxllm is in active 0.1.x development and still
> requires a fair amount of testing — do not treat it as production-ready yet.
> Chat, embeddings, streaming, routing, budgets, optional DB persistence, JWT
> auth and realtime WebSocket are implemented. See the
> [operations guide](docs/operations.md) for deployment and validation.

---

## Why oxllm?

- **Cross-provider.** Use the OpenAI SDK to call Claude. Use the Anthropic SDK
 to call GPT-5. oxllm translates both directions transparently.
- **Fast.** Native async Rust — no GIL, no interpreter overhead. Sub-millisecond
 transforms (benchmarks in `crates/*/benches/`).
- **Config-driven.** A single YAML file defines models, routing, fallbacks,
 budgets, and rate limits. Hot-reload without restart.
- **Type-safe.** Every fallible path returns typed errors. No panics on user
 or provider input.
- **Extensible.** Adding a provider is a provider module, a feature flag, and a
  registry entry; price-shard and passthrough entries are optional data-only
  edits. See [docs](docs/).

---

## Project philosophy

The AI landscape evolves at a pace that breaks projects trying to be
everything to everyone. oxllm takes a different approach: **maintainability
is the feature.**

**Stability over coverage.** We prioritize a rock-solid core that serves the
majority of use cases over chasing every niche provider or experimental API.
When a new model lands from an existing provider, we support it quickly. When
a new feature request would add complexity for a small audience, we weigh it
against the maintenance cost it imposes on everyone else.

**Bug fixes first.** Correctness, pricing accuracy, and provider
compatibility come before new functionality. A proxy that miscounts tokens or
mishandles a 429 is worse than one with fewer features — billing errors and
silent failures erode trust faster than a missing endpoint.

**Deprecation is healthy.** Features that are poorly adopted, hard to
maintain, or frequently broken get deprecated and eventually removed. This
keeps the codebase lean and the surface area testable. We would rather ship a
smaller, reliable product than a large, fragile one.

**Not a LiteLLM replacement.** oxllm is an independent project with its own
design priorities. We learned from LiteLLM's proxy architecture (and maintain
config compatibility where it's free), but the goal was never feature parity —
it was a sustainable, maintainable gateway that can keep up with the pace of
AI without becoming a maintenance burden.

The bet is simple: the projects that last are the ones you can still
understand, debug, and extend five years from now. Rust's type system,
compile-time checks, and explicit error handling are how we make that bet pay
off.

---

## Quick start

### 1. Configure

Copy the small quickstart config. It needs only the provider keys for its two
model entries and the proxy master key; remove an entry if you use one provider:

```bash
cp examples/config.minimal.yaml config.yaml
```

Minimal shape:

```yaml
# config.yaml
model_list:
  - model_name: chat
    oxllm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude
    oxllm_params:
      model: anthropic/claude-sonnet-4-5-20250929
      api_key: os.environ/ANTHROPIC_API_KEY

oxllm_settings:
  drop_params: true
  num_retries: 2
  request_timeout: 120

general_settings:
  master_key: os.environ/OXLLM_MASTER_KEY
```

More samples under [`examples/`](examples/): Docker, local Redis, multi-provider LB.
`config.example.yaml` is an annotated catalog, not a ready-to-run quickstart:
its enabled entries reference many provider environment variables.

### 2. Run

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export OXLLM_MASTER_KEY=sk-ox-your-secret
export OXLLM_REQUIRE_AUTH=true

cargo run --locked --release --bin oxllm -- serve --config config.yaml --port 4000
```

The binary **does not load `.env`**. Export variables in the launching shell or
load a trusted, shell-compatible file with `set -a; source .env; set +a`.
Docker Compose reads `.env` for interpolation and injects the variables listed
in its service configuration. A missing `os.environ/NAME` reference fails config
loading; it is not silently ignored.

### 3. Call any model with any SDK

**OpenAI SDK → Claude:**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-ox-your-secret")

response = client.chat.completions.create(
 model="claude",
 messages=[{"role": "user", "content": "Hello!"}]
)
```

**Anthropic SDK → OpenAI:**
```python
import anthropic
client = anthropic.Anthropic(
 base_url="http://localhost:4000",
 api_key="sk-ox-your-secret"
)

response = client.messages.create(
 model="chat",
 max_tokens=100,
 messages=[{"role": "user", "content": "Hello!"}]
)
```

**curl:**
```bash
curl http://localhost:4000/v1/chat/completions \
 -H "Authorization: Bearer sk-ox-your-secret" \
 -H "Content-Type: application/json" \
  -d '{"model": "claude", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 20}'
```

---

## Features

### Translation layer

| Provider | Chat | Streaming | Embeddings | Tool Calls | Native Anthropic Messages |
|----------|:----:|:---------:|:----------:|:----------:|:--------------:|
| OpenAI | ✅ | ✅ | ✅ | ✅ | — |
| Anthropic | ✅ | ✅ | — | ✅ | ✅ |
| Gemini | ✅ | ✅ | ✅ | ✅ | — |
| Ollama | ✅ | ✅ | ✅ | — | — |
| Groq | ✅ | ✅ | — | ✅ | — |
| Together AI | ✅ | ✅ | ✅ | ✅ | — |
| OpenRouter | ✅ | ✅ | ✅ | ✅ | — |
| xAI (Grok) | ✅ | ✅ | — | ✅ | — |
| Mistral | ✅ | ✅ | ✅ | ✅ | — |
| Cohere | ✅ | ✅ | ✅ | ✅ | — |
| Azure OpenAI | ✅ | ✅ | ✅ | ✅ | — |
| Azure AI Foundry | ✅ | ✅ | ✅ (text + Cohere images) | ✅ | ✅ (Claude) |

oxllm ships **35** provider transforms behind feature flags (all on by
default) — including the ones above plus Bedrock, Bedrock Mantle, Vertex AI,
DeepSeek, Cerebras, Fireworks AI, Databricks, Perplexity, SambaNova,
Cloudflare, Deep Infra, AI21, NVIDIA NIM, Hugging Face, Voyage, Jina AI,
Watsonx, DashScope, Deepgram, ElevenLabs, Meta, Replicate, and Baseten. See
`crates/oxllm-providers/src/` for the full list. The capability columns
above cover the most common providers; the others follow the same OpenAI-
or native-shape contracts.

Foundry uses the `azure_ai/` slug and supports Cohere image data-URI embeddings
through `/v1/embeddings`, including mixed text/image batches with ordered results
and summed usage. Native
Cohere configurations also recognize its text-embedding contract.
See [the Foundry sample](examples/config.foundry.yaml)
and [operations guide](docs/operations.md) for authentication overrides,
IPv4-only transport, Converse history repairs and background Vertex renewal.

Adding a provider follows a mechanical checklist: a provider module behind a
feature flag plus registry wiring. Price-table and optional passthrough entries
are data-only edits; no handler or route changes are required.

### Routing

| Strategy | Description |
|----------|-------------|
| `simple_shuffle` | Random selection (default); respects per-deployment `weight` |
| `lowest_latency` | Rolling p50 latency |
| `lowest_cost` | Price-table driven |
| `usage_based` | Live TPM/RPM counters |
| `latency_based` | EWMA smoothed latency |
| `tag_based` | Route by request metadata (`tag` on deployments) |
| `quality_router` | Configured quality scores |
| `least_busy` | Fewest in-flight requests |
| `session_affinity` | Sticky pick via `x-oxllm-session-id` (prompt-cache friendly) |
| `complexity_router` | Keyword-classify request complexity → tiered model pool |
| `adaptive_router` | Thompson-sampling bandit that learns from per-request outcomes |
| `round_robin` | Strict rotation across deployments |

Plus: cooldowns on failure, retries with exponential backoff, model fallbacks
(try model A, on failure try B), context-window fallbacks, and wildcard
routing (`model: "anthropic/*"`).

#### Session affinity

When several deployments share the same `model_name` (multi-provider or multi-key
load balancing), set `routing_strategy: session_affinity` and send a stable
session id so a conversation sticks to one deployment:

```yaml
model_list:
  - model_name: chat
    oxllm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
  - model_name: chat
    oxllm_params:
      model: xai/grok-4-1-fast-non-reasoning
      api_key: os.environ/XAI_API_KEY
  - model_name: chat
    oxllm_params:
      model: openrouter/openai/gpt-4o-mini
      api_key: os.environ/OPENROUTER_API_KEY

oxllm_settings:
 routing_strategy: session_affinity
```

```bash
curl http://localhost:4000/v1/chat/completions \
 -H "Authorization: Bearer sk-ox-your-secret" \
 -H "Content-Type: application/json" \
 -H "x-oxllm-session-id: conv-42" \
 -d '{"model":"chat","messages":[{"role":"user","content":"Hi"}]}'
```

Behavior:

- Same non-empty `x-oxllm-session-id` → same deployment (stable hash of the
  normalized session id modulo the stably sorted candidate list; works across
  multi-replica gateways). `x-litellm-session-id` is accepted when the oxllm
  header is absent or empty. The pin has no TTL.
- Missing or empty header → falls back to `simple_shuffle`.
- Cooldowns still apply: if the sticky deployment is cooling down, the request
 is re-hashed among the remaining available candidates.
- Deployment `weight` is not used under `session_affinity` (stickiness first).

Header is honored on chat, completions, messages, responses, embeddings,
realtime, images, audio, rerank, and moderations.

### Request / response headers

| Header | Direction | Description |
|--------|-----------|-------------|
| `x-oxllm-session-id` | request | Sticky routing key and durable session attribution, trimmed and bounded to 256 bytes. `x-litellm-session-id` is the fallback when this header is absent or empty. The routing pin has no TTL. |
| `x-oxllm-tag`, `x-oxllm-tags` | request | Tags for routing, attribution and per-tag budgets; migration aliases `x-litellm-tags` and `x-litellm-tag` are also accepted. |
| `x-oxllm-request-id` | response | Gateway-generated request id, also emitted as `x-request-id` and `x-litellm-call-id`, including errors and streams |
| `x-oxllm-response-cost` | response | Computed cost (USD) when available |
| `x-oxllm-key-spend` | response | Tracked chat/embedding budget-window spend snapshot after unary settlement, also emitted as `x-litellm-key-spend`; omitted for untracked keys and live streams |
| `x-oxllm-cache-hit` | response | `true` when served from response cache |
| `x-oxllm-fallback-used` | response | Fallback model name when a fallback served the request |
| `x-oxllm-router-strategy` | response | Strategy that picked the deployment (e.g. `simple_shuffle`) |
| `x-oxllm-router-decision` | response | Why: `picked`, `prompt_cache_pin`, `fallback`, `default_fallback` |
| `traceparent` | request | Propagated to upstream (OpenTelemetry) |

### Cost & budgets

- Per-request cost calculation from a vendored price table (input/output/cache
  token rates).
- OpenRouter and Perplexity measured `usage.cost` receipts take precedence over
  catalog estimates when no deployment price override is configured. Fully free
  overrides remain zero for receipt-only replies; local cards still support
  admission estimates, previews and missing/invalid-receipt fallback.
- Receiptless JSON/SSE billing uses the reported serving-model card before
  concrete-deployment fallback. Mantle bare aliases share their canonical
  prices, limits and service/context tiers.
- Service-specific pricing for `priority`, `flex`, and `ultrafast` requests,
  including long-context and cache rates. Billing uses the served tier reported
  by the provider, including explicit standard downgrades; `fast` uses a
  published fast rate or the priority rate, falling back to standard pricing
  when neither is published.
- Google `BATCH` traffic metadata selects published batch or existing Flex
  rates without an additional discount. A request `service_tier: batch` alone
  does not establish batch billing.
- Duration-priced deployments accept `oxllm_params.cost_per_second`. Elapsed
  time is billed once; per-token overrides take precedence.
- `x-oxllm-response-cost` metadata on metered responses when available.
- Per-key, per-team, and per-user budgets with configurable reset windows
 (`budget_duration: 30d`).
- RPM/TPM rate limits with atomic reservation (no overcommit under concurrency).
- Budgets reserve estimated cost before an upstream call. A zero budget blocks
  new spend; an omitted budget is unlimited. Response-cache hits bypass spend
  reservation. Deployment pricing overrides remain authoritative.
- Groups whose effective deployment overrides are entirely free reserve zero
  before selection. This includes inherited cache rates and every configured
  context tier; zero base-token rates alone need not make a deployment free.
  Mixed-price groups retain conservative estimates, and paid fallbacks still
  require their normal budget top-up.

### Request tagging (routing + per-application budgets)

Requests carry tags in the top-level `tags` array, `metadata.tags` array, and
headers. The merge order is body tags, metadata tags, `x-oxllm-tag`,
`x-oxllm-tags`, `x-litellm-tags`, then `x-litellm-tag`. Header values are
comma-separated and may be repeated; values are trimmed and deduplicated.
Literal attribution pairs such as `product:demo` remain whole, including when
creating a tag budget. Requests retain at most 16 tags of 128 bytes each.
Gateway tags are removed from provider bodies and are not forwarded as headers.
One shared production key can serve many applications, each
identified by its tag.

- **Routing** (`routing_strategy: tag_based`): deployments advertise tags
  via `model_info.tags` (plus regexes via `model_info.tag_regex`). Positive
  tags subset-match and are a preference: if none match, the request falls
  back to the `default`-tagged pool, else the constrained group. `&tag`
  requires a deployment tag and `!tag` excludes deployments carrying it;
  when hard requirements/exclusions leave no eligible deployment, the
  request is rejected with HTTP 400 `routing_error` instead of falling back.
- **Spend logging**: every spend row records the request's tags
  (`tag_name` primary + `tags` JSON list). Filter with
  `GET /spend/logs?tag_name=app-billing` and aggregate with
  `GET /spend/tags` (a multi-tag request counts toward each carried tag).
- **Per-application budgets**: create a tag (`POST /tag/new`) with
  `max_budget` and every tagged request reserves against that tag's own
  counter — over-budget applications 429 while the same key's other apps
  keep working.

### Authentication

| Method | Description |
|--------|-------------|
| Master key | Management access; bypasses caller-scoped ACLs/budgets/rate limits |
| Virtual keys | `sk-ox-*` format, SHA-256 hashed, stored in DB |
| JWT | HS256/HS384/HS512, with team/budget claims |

Auth brute-force protection (per-IP rate limiting on failures), constant-time
key comparison, and cache invalidation on key deletion.

### Caching

- In-memory response cache (moka, 10k entries, TTL-based).
- Optional Redis cache for multi-instance deployments (DualCache: read-through,
 write-through).
- Optional S3 / GCS / Azure blob response cache (`cache.type: s3|gcs|azure`)
for large shared caches; the corresponding `s3-cache`, `gcs-cache`, or
`azure-cache` build feature is required, otherwise the server logs a warning
and falls back to memory. An in-memory counter layer is added automatically so
rate limits and budgets keep working (blob stores have no atomic counters).
Transient errors are retried with backoff; reads stay fail-open.
- Live upstream streams are not captured or written to the response cache:
  writes require a buffered response body. Existing matching cache entries can
  be replayed as SSE for streaming requests; streaming and non-streaming
  requests have separate cache keys.
- Cached responses consume RPM (to prevent cache-storm abuse) but skip budget and TPM metering.

### Persistence

Build with **`--features db`** and set `general_settings.database_url`:
- **Keys / teams / users** — CRUD via management REST endpoints.
- **Spend logs** — one row per request (model, tokens, cost, latency, key/team).
- **Session attribution** — nullable `session_id` on successful, failed and
  partial request rows, independently of the routing strategy. Filter either
  spend-log endpoint with `session_id`; historical/older-writer rows remain null.
- **Batched spend writer** — background flush, no DB write on the hot path.
- Known transient database rollbacks retry the same event IDs up to four attempts;
  ambiguous commits and permanent errors retain their terminal behavior.
- Terminal partial-stream/realtime records can wait up to 250 ms for queue
  capacity; continued saturation/closure is counted as an observable drop.
- SQLite for zero-config local dev; Postgres for production.

The root crate has no default Cargo features: a plain `cargo build` does not
enable DB persistence. The proxy Dockerfile builds with `db,otel`. `admin` builds
the separate UI binary; it does not enable persistence in the proxy.

Virtual-key creation/update accepts `allowed_cache_controls` containing
`insert`, `use_cache`, `no-cache`, and `no-store`. On creation, an omitted/null
policy is unrestricted; `[]` disallows explicit controls. On update, omission
leaves the current policy unchanged. Use
`clear_allowed_cache_controls: true` on update to remove the restriction.
Authentication validates cached key/scope snapshots against current DB records,
so completed policy changes cannot be undone by a late cache refill.
Key creation/update also accepts `rpm_limit`/`tpm_limit` as aliases of native
`rpm`/`tpm`. Unknown or duplicate fields fail with 400 before mutation; unsupported
key `metadata`, `tags`, and permission fields are not silently discarded.
Info/list/update return `models` as a JSON array or null and omit the key hash.
Corrupt stored model policies use authentication's fail-closed deny-all policy
on read-back and remain available for explicit repair. Unsupported request
content types retain HTTP 415; malformed JSON and key fields return 400.
Incremental team model add/delete refuses unreadable or deny-all policies;
`/team/update` provides the explicit replacement path. Admin team/user edits
preserve effective deny-all for unreadable stored policies. Corrupt customer
blocklists block every model until explicitly replaced.
Full key/team/user updates preserve an identical stored literal model array,
so an unrelated save cannot trim a whitespace-bearing pattern into a broader
wildcard. Explicitly changed replacements keep native normalization semantics.
Their `spend` field is an asynchronously persisted total from retained logs,
distinct from the current-window `x-oxllm-key-spend` budget snapshot.
Key blocking/unblocking and rotation are available at `/key/block`,
`/key/unblock`, and `/key/rotate`, using `key_id`.

Native key attachments must match an owning project's team; ownerless projects
remain shareable. Key edits, bulk team moves, project reparenting and affected
scope deletions validate and commit atomically, with cache invalidation afterwards.

Spend rollups (`/spend/keys`, `/spend/by-model`, `/spend/users`, `/spend/teams`,
`/spend/organizations`, `/spend/tags`) support `limit` and `offset`. Replies
include `has_more` and `next_offset`; limits select complete aggregated groups,
rather than limiting the spend rows included in their totals.
Inference-pipeline model ACL/customer denials, admission-hook rejections, and
terminal upstream failures record `status: "failure"`, zero cost, request
identity and tags. This includes the chat/embedding API bridges and hooked
native/provider-prefixed routes. Local token-counter estimates and model
discovery do not use this spend observer. Prompts, responses and upstream
error bodies are not stored. Success,
failure and partial-stream reporting share one terminal record claim.
`/global/spend` includes the configured global `max_budget` (null when unset).
End-user attribution uses trimmed, nonblank request `user` values for both
successful requests and early denials. End-user/customer IDs are limited to 256
UTF-8 bytes after trimming; oversized inference IDs and new customer IDs return
400 rather than aliasing a shorter customer. The spend writer drops invalid
end-user attribution defensively, not the metering event or its settled cost.
Realtime partial rows retain known
reported cost; unpriced or unreported usage remains unknown.

`POST /spend/calculate` previews the same provider/tier/deployment rates used by
inference, without an upstream call. It accepts `model`, standard or native
`usage`, and optional `provider`, `service_tier`, `deployment_id`,
`pricing_override`, and `duration_ms`. Configured aliases inherit their rate
card; differing load-balanced cards need a selector. Explicit overrides inherit
unspecified published rates, and explicit zero token rates take precedence over
duration pricing. Batch previews can select `service_tier: "batches"`; this
read-only selector does not establish batch billing for ordinary inference.

### Realtime transports

Clients use the OpenAI-compatible `/v1/realtime` WebSocket endpoint. Native
Bedrock uses a signed HTTP/2 `InvokeModelWithBidirectionalStream` POST, with the
AWS SDK handling event framing and chained signatures. Explicit WS(S)
`api_base` values select a compatible bridge; HTTP(S) values select native
runtime transport. Model-specific features on opaque Bedrock inference profiles
need `model_info.base_model` (or `oxllm_params.base_model`) while the wire model
remains the profile ARN.

### Observability

- Structured `tracing` logs with request IDs and secret redaction.
- Prometheus `/metrics` (request count, latency histogram, tokens, cost, errors).
- Optional OpenTelemetry distributed tracing (`otel` feature).
- `traceparent` propagation to upstream providers.

Set `general_settings.log_format: json` for JSON output from `serve` (`text` is
the default); `RUST_LOG` sets verbosity. Changing the log format or OTel exporter
requires a process restart. There is no `RUST_LOG_FORMAT` switch.

### Security

- SSRF protection: `api_base` validated at config load (private IPs blocked).
- Body size limit (configurable, default 10 MB).
- JSON depth limit (128 levels max).
- HTTP request smuggling guard (dual Content-Length + Transfer-Encoding).
- No-redirect policy on upstream requests.
- `OXLLM_REQUIRE_AUTH` env var refuses to start without any auth configured
(master key, JWT secret, or database with virtual keys) in prod.
- Trusted proxy CIDRs for `X-Forwarded-For` validation.

---

## Configuration reference

Full annotated sample: [`examples/config.example.yaml`](examples/config.example.yaml).

```yaml
model_list:
  - model_name: my-model # name clients send
    oxllm_params:
      model: openai/gpt-4o-mini # provider/model
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1 # optional override
      timeout: 600 # request timeout (seconds)
      stream_timeout: 60 # max idle between stream chunks
      rpm: 480 # per-deployment rate limit
      tpm: 1000000 # per-deployment token limit

  # Multiple deployments → load balanced + failover
  - model_name: my-model
    oxllm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY_BACKUP

  # Wildcard: serve requested `anthropic/<model>` names through this
  # deployment. The requested provider prefix must match; this does not catch
  # bare model names or other providers.
  - model_name: "*"
    oxllm_params:
      model: anthropic/*
      api_key: os.environ/ANTHROPIC_API_KEY

oxllm_settings:
  drop_params: true # strip unsupported params
  num_retries: 2 # retries on 5xx/429/timeout
  request_timeout: 600 # global default
  # simple_shuffle | lowest_latency | lowest_cost | usage_based |
  # latency_based | tag_based | quality_router | least_busy | session_affinity |
  # complexity_router | adaptive_router | round_robin
  routing_strategy: simple_shuffle
  cooldown_time: 60 # seconds to skip failed deployments
  # Fallback targets must also be configured in model_list:
  # fallbacks:
  #   - my-model: [backup-model]
  # context_window_fallbacks:
  #   - small-model: [large-model]
  max_budget: 100.0 # default per-key budget (optional)
  budget_duration: 30d # reset window
  rpm: 1000 # default per-key RPM limit (optional)
  tpm: 1000000 # default per-key TPM limit (optional)
  cache:
    type: memory # "redis" | "s3" | "gcs" | "azure" also supported
  quality_scores: # keyed by provider/model for quality_router
    openai/gpt-4o-mini: 0.8

general_settings:
  master_key: os.environ/OXLLM_MASTER_KEY
  # database_url: os.environ/DATABASE_URL # optional; build with --features db
  # jwt_secret: os.environ/JWT_SECRET # optional
  allow_private_api_base: false # opt in for private Redis or upstream endpoints
  # trusted_proxy_cidrs: ["10.0.0.0/8"] # use your actual trusted proxy CIDRs
  # allowed_ips: ["192.0.2.0/24"] # optional caller CIDR allowlist
  max_request_body_size_mb: 10
  log_format: json
  spend_log_retention_days: 90 # optional DB pruning; unset keeps all logs
  shutdown_timeout_secs: 30 # effective drain is max(this, request_timeout)
  management_cors_origins:
    - "http://localhost:3000"
  proxy_cors_origins:
    - "http://localhost:3000"
```

`os.environ/` string references are resolved at each config load against the
process environment. They do not coerce strings into numeric/boolean fields.
Compatibility blocks tolerate many unknown keys with warnings; invalid known
values and strict substructures can fail validation.

`allowed_ips` omitted allows all caller IPs; **`allowed_ips: []` denies all protected callers**.
Forwarded client IPs are used only for trusted proxy peers. Management and proxy
CORS are separate allowlists; omitted/empty lists emit no CORS headers.

Hot reload updates models, routing, auth and config-derived hooks. DB/cache
connections, listener/middleware, retention tasks, shutdown timing and logging
have restart-only settings; see the [reload matrix](docs/operations.md#configuration-and-restarts).

---

## Management API

When the proxy is built with `--features db` and `database_url` is configured,
management endpoints are available (master-key only):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/key/generate` | POST | Create a virtual API key |
| `/key/list` | GET | List keys, up to 1000 (hashes redacted; check `truncated` for more) |
| `/key/info` | GET | Get key details |
| `/key/delete` | POST | Soft-delete a key |
| `/team/add` | POST | Create a team |
| `/team/list` | GET | List teams, up to 1000 (check `truncated` for more) |
| `/user/add` | POST | Create a user |
| `/spend/logs` | GET | Query spend logs (filter: `key_id`, `team_id`, `model`, `tag_name`, `session_id`, dates) |
| `/spend/total` | GET | Total spend summary |
| `/spend/tags` | GET | Spend aggregated by request tag (multi-tag rows overlap) |
| `/model/list` | GET | List models with metadata |

---

## Endpoints

| Path | Description |
|------|-------------|
| `POST /v1/chat/completions` | Chat completions (stream + non-stream) |
| `POST /chat/completions` | Alias |
| `POST /v1/completions` | Legacy text completions |
| `POST /v1/embeddings` | Embeddings |
| `POST /v1/messages` | Anthropic input translated through the shared OpenAI-shaped chat pipeline |
| `POST /anthropic/v1/messages` | Native Anthropic Messages forwarding |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/images/generations` | Image generation |
| `POST /v1/audio/transcriptions` | Audio transcription |
| `POST /v1/audio/speech` | Text-to-speech |
| `POST /v1/moderations` | Content moderation |
| `POST /v1/rerank` | Reranking |
| `GET /v1/realtime` | Realtime WebSocket proxy |
| `GET /v1/models` | List configured models |
| `GET /health` | Liveness check |
| `GET /health/readiness` | Readiness check (pings DB + Redis) |
| `GET /metrics` | Prometheus metrics (proxy auth applies when auth is configured) |
| `/{provider}/{path}` | Provider passthrough proxy |

`/metrics` is protected by the normal proxy auth stack when a master key,
virtual key, or JWT is configured; send the same credentials you use for
protected inference routes. `/health` and `/health/readiness` remain
unauthenticated for probes.

`/v1/messages` is the Anthropic↔OpenAI translation bridge, even when its upstream
is Anthropic. For native Anthropic content, streaming events and beta-specific
fields, use `/anthropic/v1/messages` with an Anthropic deployment (Anthropic SDK
`base_url="http://localhost:4000/anthropic"`). Native features are not automatically
portable through the translated route. See endpoint tests for exact coverage.

`/v1/responses` translates `text.format` through the shared chat pipeline,
including streams, when the selected adapter has a native representation.
JSON-object mode requires native JSON-object support; Anthropic/Converse require
a supplied JSON schema instead. Schema names, strictness and descriptions reach
the provider transform, and formats participate in cache identity. Malformed
formats return a local 400. SDK response parsers consume
the returned `output_text` content and input/output token usage.
Schema names are trimmed and must use 1–64 ASCII letters, digits, underscores
or hyphens. `text.verbosity` is not translated by the chat bridge.

Structured JSON output uses the provider's native schema API, never a synthetic
tool or a repaired substitute schema. Explicit strict function tools forward
`strict: true` in the native tool definition and stay tool calls. The selected
provider owns model/schema support checks; unsupported features remain errors.
Schema-less JSON-object mode returns 400 on Anthropic/Converse, which have no
native equivalent, rather than falling back to tool calling. See the
[native-output contract](docs/operations.md#native-structured-outputs-and-strict-tools)
for API mapping, migration guidance, cache isolation and deployment-policy safeguards.
The safeguards include native Messages/Responses relay body shaping, message-level formats,
and explicit tool choices even when the tools are not strict. Unmappable choices
fail rather than silently defaulting to automatic selection.

OpenAI-shaped inference surfaces return HTTP 400 with `code: model_not_found`
for an unconfigured model after authorized fallback handling; translated
`/v1/messages` uses the Anthropic 400 `invalid_request_error` envelope.
Native embeddings and token-counting routes also reject invalid model selectors
with 400. Configured groups with no available deployment return retryable 429s.
Resource lookups and genuine provider 404s retain HTTP 404. Gateway request
deadlines retain HTTP 504 and a timeout message.

---

## Architecture

```
oxllm-core Pure types, traits, router, cost, auth logic, hook chain
oxllm-providers Per-provider transforms (feature-gated, one module each)
oxllm-server Axum HTTP server, config, cache, hooks, management API
oxllm-cache-redis Redis cache impl (server `redis` feature, enabled by default)
oxllm-cache-s3 S3-compatible blob response cache (optional)
oxllm-cache-gcs GCS blob response cache (optional)
oxllm-cache-azure Azure blob response cache (optional)
oxllm-db sqlx persistence: keys, teams, spend logs (optional)
oxllm-admin Admin web UI (optional, separate binary)
```

Dependency direction (acyclic, enforced by `tests/workspace_layering.rs`):

```
oxllm-core ← oxllm-providers
oxllm-core ← oxllm-cache-redis ← oxllm-server
oxllm-core ← oxllm-cache-s3 ← oxllm-server
oxllm-core ← oxllm-cache-gcs ← oxllm-server
oxllm-core ← oxllm-cache-azure ← oxllm-server
oxllm-db ← oxllm-server
oxllm-server ← oxllm (binary)
```

`oxllm-db` is intentionally independent of `oxllm-core`; persistence types
stay in the database crate.

**Adding a provider** = a module in `oxllm-providers`, a feature flag, and
registry wiring. Price-table and optional passthrough entries are data-only
edits; no handler or route changes. See
[docs/adding-a-provider.md](docs/adding-a-provider.md).

### Documentation

| Doc | Description |
|-----|-------------|
| [docs/architecture.md](docs/architecture.md) | Crates, request path, where to change what |
| [docs/operations.md](docs/operations.md) | Deployment, reload, backups, migrations and cache privacy |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Startup, auth, backend and streaming diagnostics |
| [docs/adding-a-feature.md](docs/adding-a-feature.md) | Where a change belongs and the steps for each feature kind |
| [docs/adding-a-provider.md](docs/adding-a-provider.md) | Checklist for a new upstream provider |
| [helm/oxllm/README.md](helm/oxllm/README.md) | Kubernetes chart values and rollout behavior |
| [examples/](examples/) | Sample YAML configs |

---

## Deployment

> **⚠️ Multi-replica requirement:** When running multiple replicas (e.g. behind
> a load balancer or with `replicas > 1` in Helm), Redis cache
> (`cache.type: redis`) is **REQUIRED** for correct rate limiting and budget
> enforcement. Without Redis, each replica maintains independent counters and
> limits/budgets will be off by a factor of ~N. The in-memory cache is fine for
> single-replica deployments only.

### Docker

```bash
cp .env.example .env
# Set OXLLM_MASTER_KEY and the provider keys you intend to use in .env.
docker compose up --build -d --wait
bash scripts/smoke-docker.sh
```

The base stack runs Redis and persistent SQLite (`oxllm-data` volume). It boots
with dummy provider values for health/auth checks; live requests need real keys.
The admin UI uses a separate overlay so its required secrets are evaluated only
when that file is included:

```bash
# Also set ADMIN_SESSION_SECRET (16+ chars) and ADMIN_UI_PASSWORD in .env.
docker compose -f docker-compose.yml -f docker-compose.admin.yml --profile admin up --build -d --wait
```

Open `http://127.0.0.1:8080`. The sample uses local HTTP cookies; set
`ADMIN_COOKIE_SECURE=true` when serving admin over HTTPS. For a standalone proxy
without Redis/DB, build `docker build -t oxllm:local .`, mount
`examples/config.minimal.yaml` at `/config.yaml` and pass its three environment
variables. The baked-in Docker config expects the Compose Redis service.

### Helm

```bash
helm upgrade --install oxllm ./helm/oxllm -f my-values.yaml
```

Use the [chart README](helm/oxllm/README.md) to choose images you have built/pushed,
provide config-specific credentials via `envSecretRefs` or `env`, and enable
external Postgres/Redis. The chart's image defaults are local names, not a claim
of published registry artifacts. No OpenAI credential is required for a config
that uses another provider. Protect any secret-bearing values file and remember
that chart-managed secret values are also stored in Helm release metadata.

---

## Distributed Deployment

When running multiple replicas, configure Redis to enable fleet-wide
coordination:

```yaml
oxllm_settings:
  cache:
    type: redis
    host: redis
    port: 6379
general_settings:
  allow_private_api_base: true # the Redis service above is on a private network
```

### What Redis Enables

| Feature | Without Redis | With Redis |
|---------|--------------|------------|
| Rate limiting (RPM/TPM) | Per-replica counters | Fleet-wide enforcement |
| Budget tracking | Per-replica spend | Global spend tracking |
| Auth brute-force protection | Per-replica failure budget | Shared failure counter |
| Key revocation | Up to 5 min delay | Near-instant (pub/sub) |
| Budget threshold alerts | Duplicate per replica | Deduplicated fleet-wide |
| JWT replay protection | Not available | jti blacklist tracking |

### Configuration

All coordination features are **automatic** when Redis is configured.
No additional settings required — the proxy detects Redis at startup
and enables distributed coordination.

For JWT replay protection (optional, opt-in):
```yaml
general_settings:
  jwt_replay_protection: true
```

Note: a token's `jti` is consumed at authentication time, before rate-limit,
budget, and model-ACL checks. If a request is rejected after authentication
(e.g. a 429), retries with the same token are rejected as replays. Mint
short-lived tokens (or leave replay protection off) when clients must be
free to retry.

### Without Redis

Without Redis, the supported cache and counter operations are per-process.
Rate limits and budgets are enforced per-replica (effectively multiplied
by replica count). Key revocation propagates within 5 minutes via TTL
expiry. This is suitable for single-replica deployments or development.

---

## Development

### Build & test

```bash
cargo build --locked --workspace
cargo test --locked --workspace
```

### Lint

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo deny --locked check
```

### Benchmarks

```bash
cargo bench --locked --workspace
```

Historical sample results (Apple Silicon, release build; not a current-run
performance guarantee):

| Benchmark | Time |
|-----------|------|
| Cost calculation | 16 ns |
| Router pick (10 deployments) | 843 ns |
| Router pick (100 deployments) | 10 μs |
| Anthropic transform_request | 454 ns |
| Anthropic transform_response | 1.2 μs |
| Token estimation (short) | 4.7 μs |

### Coverage

```bash
cargo llvm-cov --workspace --all-features --html --output-dir target/cov
# Open target/cov/html/index.html in your browser.
```

---

## Comparison with LiteLLM

oxllm is a from-scratch Rust rewrite of the LiteLLM **proxy/gateway** (not the
Python SDK). We copy the ideas, not the code.

| LiteLLM | oxllm |
|---------|-------|
| Python proxy and SDK | Rust proxy with separate core, provider and host crates |
| Central proxy application | Thin axum handlers; shared pipeline and core logic |
| Untyped dicts everywhere | `serde_json::Value` at boundaries; typed traits |
| GIL-bound | Native async; saturate cores |
| Prisma ORM | sqlx with runtime SQL for SQLite/Postgres |
| Adding a provider touches 6 files | Provider module + feature + registry entry |

---

## Multi-instance deployments

oxllm supports running multiple instances behind a load balancer with shared
Redis (cache + rate limits + spend counters) and Postgres (spend logs + model
management). See the [operations guide](docs/operations.md) for deployment checks
and the [multi-instance harness](tests/multi-instance/README.md) for the
reproducible stress-test stack. Historical run results are not release
certification.

### Database migration

By default, each oxllm instance auto-migrates the database on startup.
Concurrent starters serialize safely (Postgres advisory lock, SQLite
transaction, idempotent version writes) — latecomers see a current schema
and no-op. For large fleets, still prefer running migration as a pre-step
so a slow or stuck migration can't stall every replica's startup.

**Solution: disable auto-migration and run it as a pre-step.**

```sh
# 1. Run migrations once (standalone CLI):
oxllm migrate --database-url postgres://user:pass@host/oxllm

# 2. Start instances with auto-migration disabled:
OXLLM_AUTO_MIGRATE=false oxllm serve --config config.yaml
```

In Docker Compose, use an init container:

```yaml
services:
  migrate:
    image: oxllm
    command: ["migrate", "--database-url", "postgres://postgres@db/oxllm"]
    depends_on:
      db: { condition: service_healthy }
    restart: "no"  # run once and exit

  oxllm-a:
    image: oxllm
    command: ["serve", "--config", "/etc/oxllm/config.yaml"]
    environment:
      OXLLM_AUTO_MIGRATE: "false"
    depends_on:
      migrate: { condition: service_completed_successfully }
```

The `OXLLM_AUTO_MIGRATE` env var defaults to `true` (backward compatible).
Set it to `false` or `0` when migrations are managed externally.

---

## License

MIT
