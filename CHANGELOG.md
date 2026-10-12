# Changelog

All notable changes to oxllm are documented here.
Format: Keep a Changelog, adheres to Semantic Versioning.

## [Unreleased]

### Provider and accounting corrections, 2026-10-11

- Close independently reproduced review gaps in receiptless JSON/SSE pricing,
  transient partial-ledger backpressure, dangling-project error status, Mantle
  alias parity, Converse file-data validation, realtime status policies and
  sanitized-zero receipt-only pricing.
- Correct published token/cache rates, context/output limits, Flex/ultrafast
  tiers and eight model cards across existing provider namespaces. Flatten
  direct Sonnet 4.5 pricing at its corrected 200k input ceiling and retire two
  bare Bedrock IDs that AWS cannot invoke.
- Preserve Converse tool-result images, documents, references and ordering;
  omit unsupported stop sequences for Grok and GPT 5.6+ families.
- Lift leading system/developer content during Claude token counting after the
  caller's system prefix, retaining cache controls and hosted authentication.
- Use OpenRouter measured cost receipts ahead of fallback catalog estimates.
  Preserve operator price overrides, including fully free receipt-only replies.
- Retry known transient spend-batch rollbacks with bounded backoff and stable
  event IDs; retain the commit-ambiguity boundary and once-only accounting.
- Serialize native key/team/project mutations and scope deletion so ownership
  validation cannot race attachment, reparenting, reassignment or removal.
- Classify exact `cyber_policy` errors as HTTP 400 policy rejections. Preserve
  pre-content fallback and post-commit termination, and honor explicit cooldown
  policies without treating ordinary client stream errors as deployment faults;
  classified context-window/content-policy errors keep counting toward
  configured cooldown policies.
- Select the MongoDB Voyage endpoint from resolved `al-` credentials on
  embeddings, rerank and native relays, preserving explicit API bases.
- Isolate real filesystem-watcher regressions and wait for directory-registration
  barriers before exercising reload behavior.

### Round-4 functional corrections

- Consume and persist top-level request tags alongside metadata/header tags,
  stripping all gateway tag carriers from provider bodies.
- Keep zero retry aliases independent of cooldown thresholds; generic request
  400/422s no longer cool healthy deployments under global/category policies.
- Normalize proven additive Vertex Gemini reasoning usage for JSON and SSE so
  completion/TPM/client counts and billing agree without double folding.
- Accept key RPM/TPM limit aliases, reject unknown/conflicting fields before
  mutation, and return typed model policies and persisted spend on key reads.
- Record attributed zero-cost failure/denial rows without bodies or duplicate
  terminal/partial entries; expose global cap and tracked unary chat/embedding
  key-spend snapshots.
- Close review gaps in embedding-cache spend headers, early-denial header tags,
  rerank/native control validation, corrupt key-policy read-back, inconsistent
  Vertex zero-output usage, and lossless admin model-policy editing. Preserve
  HTTP transport rejection statuses and selected-model failure attribution.
- Preserve corrupt team/user/customer policies across incremental/admin edits,
  normalize end-user attribution consistently, retain known realtime partial
  cost, audit pre-upgrade rejections, and apply proof-gated Gemini accounting
  on both compatible Google surfaces. Harden async ledger regression waits.

### Gateway compatibility and observability

- Qualify Vertex OpenAI-compatible Gemini selectors with `google/`, authenticate
  Foundry native Claude with `x-api-key`, and translate legacy token limits for
  identifiable Azure reasoning deployments.
- Echo requested model names for direct calls and the public serving group for
  fallbacks consistently across JSON/SSE/cache responses; normalize missing
  reasoning-token details to zero.
- Expose public deployment identities, retry/fallback counters, upstream throttle
  headers and total/upstream/overhead timings under the primary `x-oxllm-*`
  contract, with additive compatibility aliases and browser CORS exposure.
- Record/filter deployment, requested-model and serving-group spend through
  additive schema versions 15/16 while retaining serving-model pricing and
  older-writer compatibility.
- Recognize the session-affinity pre-call alias, share validation/startup
  diagnostics, and add bounded Datadog dependency-health failure callbacks.
- Deduplicate config-directory watch aliases and coalesce callback overflow
  into a rescan without blocking watcher registration or losing changes.
- Translate Azure GPT chat-family token limits without reasoning/sampling
  mutations; accept routing-budget/model-rate-limit check aliases and expose
  effective retry budgets and opt-in fallback status summaries.

### Models and provider compatibility

- Sync the refreshed model catalog's token/cache prices, context/output limits,
  service tiers and new model cards, including Azure regional GPT-6.1 Sol,
  Grok 4.7, Cohere Embed v5, Mistral Large 4 and Voyage entries.
- Stage stream metadata until content or a normal terminator arrives, allowing
  pre-content disconnects/error frames to retry or fall back with the same
  status-aware policy as failures before headers.
- Accept validated request-scoped fallback chains, disable/cap controls, retry
  budgets and fractional timeouts; strip gateway controls from provider bodies.
- Bill Azure/Azure AI Model Router's selected model plus its input-token
  infrastructure fee exactly once, including streaming spend settlement.
- Inject the OpenAI explicit prompt-cache dialect on flagged Mantle/Foundry
  deployments, preserving caller marks/options and the four-block limit.
- Add Vertex/Foundry Claude token counting with system/tools forwarding,
  counting-region overrides and optional utility-endpoint local fallback;
  normalize image parts in native Count Tokens requests.
- Remove foreign thinking-block signatures from Gemini history and use Vertex's
  multi-region `us`/`eu` HTTP hosts.
- Resolve Anthropic federation through cached OIDC file/environment grants,
  fail closed on missing federation IDs, and preserve OAuth/client beta flags.
- Keep caller W3C trace headers ahead of configured defaults across attempts.
- Correct review regressions in Fireworks priority overlays and Claude 5.5 batch
  cache/context rates; retain the nominal budget floor alongside Model Router fees.
- Bound client retry overrides to 10, classify malformed stream preludes as
  retryable 502s, honor overload-status policies, and release startup reservations
  on cancellation without repeating partially completed refunds.
- Validate gateway/federation controls before API conversion, strip standalone
  controls and configured reintroductions from native/embedding/chat bodies,
  and preserve opt-in fallback error headers through Messages/Responses bridges.
- Keep deployment unary timeouts separate from stream idle budgets; include
  thinking-signature repairs in retry-budget headers.
- Prefer deployment-owned federation sources to fleet defaults, cap raw assertion
  bytes before normalization, and deduplicate OAuth/client beta flags.
- Prevent disabled utility token counting from falling back locally when native
  counting is unsupported or unavailable.
- Added Claude Haiku 5.5 pricing, context limits and cache rates for Anthropic,
  Vertex AI, Bedrock regional aliases (including GovCloud), Bedrock Mantle,
  Azure AI, OpenRouter and Perplexity. Published >100k-token tiers are included
  where present; the Perplexity entry uses base rates only.
- Corrected Claude Sonnet 5.5 cache-read rates from the refreshed reference
  price map across Anthropic, Vertex AI, Bedrock, and Bedrock Mantle.
- Added GPT-6.1 Sol pricing and context limits for OpenAI, Azure, OpenRouter,
  Bedrock regional aliases, and Bedrock Mantle, plus OpenRouter pro/batch aliases.
- Added the Baseten chat backend and the accelerated DeepSeek-V4.1-Flash-Fast
  model's token/cache prices and limits.
- Corrected Azure, Gemini, Groq, Bedrock, and OpenRouter prices and context limits.
- Bill service tiers using the provider's served tier, including streaming,
  cache tokens, long-context prices, budget settlement, and spend records.
- Added canonical `cost_per_second` deployment pricing. Legacy input/output
  duration rates are fallbacks for the same elapsed time, which is billed once.
  Audio duration rates are not treated as request latency.
- Preserve per-message `output_config` on Claude requests and add its required
  beta tag on direct, Vertex AI, Bedrock Invoke, and native Messages routes.
- Route current GPT families on Bedrock through Converse, and select Mantle's
  model-specific API base for modern models, including wildcard deployments.
- Strip the internal client-error traceback flag from provider requests.

### Session affinity routing

- **`session_affinity` strategy** (`oxllm-core::router::session_affinity`):
  sticky deployment selection via a deterministic modulo hash of
  `RequestContext::session_id`. Clients send `x-oxllm-session-id` so multi-turn
  traffic on a multi-deployment model group stays on one provider (prompt-cache
  friendly). Falls back to `simple_shuffle` when the header is absent.
- Integration coverage in `tests/session_affinity_e2e.rs` (chat stickiness,
  spread across sessions, shuffle fallback, embeddings header plumbing).

### Post-Phase-8 — P0 fixes + benchmarks

- **P0 security fixes**:
  - Passthrough routes now require auth when `master_key` is set.
  - SSRF validation: `api_base` checked for private IPs / non-HTTP schemes at
    config load; `redirect::Policy::none()` on shared client.
  - Config hot-reload now preserves DB, Redis, spend writer, and HTTP client.
  - Budget/RPM hooks always registered (per-key limits work without global config).
  - Master key bypasses all budget/rate-limit checks.
- **Criterion benchmarks** (stable Rust):
  - `oxllm-core/benches/router_bench.rs` — deployment picking, wildcard, cooldown.
  - `oxllm-core/benches/cost_bench.rs` — cost calculation and price lookup.
  - `oxllm-core/benches/tokens_bench.rs` — tiktoken estimation at various sizes.
  - `oxllm-providers/benches/anthropic_bench.rs` — request/response/stream transforms.
  - Run with `cargo bench --workspace`.

### Phase 8 — Hardening + realtime + advanced auth

- **Realtime WebSocket** (`GET /v1/realtime`): transparent bidirectional proxy
  to a provider's realtime API (OpenAI `wss://api.openai.com/v1/realtime`).
  Forwards the `model` query param, propagates the bearer token, and tears
  down the upstream on client disconnect (and vice versa). Backed by
  `tokio-tungstenite`.
- **JWT auth** (`oxllm-server::auth::jwt`): an alternative auth method checked
  after master + virtual keys. Configured via `general_settings.jwt_secret`
  (+ optional `jwt_algorithm`, `jwt_audience`). Extracts `sub`/`team_id` and
  optional `budget`/`rpm`/`tpm` claims into an `AuthUser`. Uses the
  `jsonwebtoken` crate (HS256 default).
- **Quality router** (`oxllm-core::router::quality`): a new
  `routing_strategy: quality_router` that picks the deployment with the highest
  configured score from `oxllm_settings.quality_scores`. Falls back to
  `simple_shuffle` when no candidates have scores.
- **OpenTelemetry** (feature `otel`): OTLP/gRPC trace exporter wired into the
  global `tracing` subscriber when `general_settings.otel_endpoint` is set.
  The incoming `traceparent` header is forwarded on upstream chat requests.
- **Security hardening** (`oxllm-server::security`):
  - Configurable body size limit (`general_settings.max_request_body_size_mb`,
    default 10).
  - Request-smuggling guard (rejects requests carrying both `Content-Length`
    and `Transfer-Encoding` with 400).
  - Per-IP auth-failure brake (`AuthBrake`): 10 failures / 60s → blocked 300s
    with a `Retry-After` header.
  - `RedactionLayer` helpers (`is_secret_key`, `redact_header_value`) for
    secret-aware logging.
  - Upstream response sanitization: strips `server` / `x-request-id` headers
    and truncates error bodies to 500 chars (+ellipsis).
- **Performance**: shared `reqwest` client now sets `pool_max_idle_per_host(20)`,
  `pool_idle_timeout(90s)`, `tcp_nodelay`, and `tcp_keepalive(60s)`.
- **Deployment**: multi-stage distroless `Dockerfile`, `docker-compose.yml`,
  `.dockerignore`, and a minimal Helm chart (`helm/oxllm/` with Deployment,
  Service, ConfigMap+Secret).
- New config fields: `jwt_secret`, `jwt_algorithm`, `jwt_audience`,
  `max_request_body_size_mb`, `otel_endpoint`, `otel_service_name`,
  `quality_scores`.
- New workspace deps: `tokio-tungstenite`, `jsonwebtoken`, `opentelemetry`,
  `opentelemetry_sdk`, `opentelemetry-otlp`, `tracing-opentelemetry`.

### Phase 3 — More providers + more endpoints

- `oxllm-core`: added `EmbeddingProvider` trait, `Provider::embeddings()`
  accessor, and `ChatProvider::supported_params()` for `drop_params` handling.
- `oxllm-core::registry::ProviderRegistry`: a slug → `Arc<dyn Provider>` map
  with `resolve_for_deployment`. The server now resolves providers
  dynamically instead of hardcoding OpenAI.
- `oxllm-providers`: three new providers, all default-enabled and
  feature-gated:
  - `anthropic`: full OpenAI ⇄ Anthropic Messages API translation. Extracts
    system messages into the top-level `system` field, maps tools to
    `input_schema`, converts `tool_use` content blocks back to OpenAI
    `tool_calls`, maps `stop_reason` → `finish_reason`, requires
    `max_tokens` (defaults to 4096), and translates the Anthropic SSE event
    stream (`message_start`, `content_block_delta`, `message_delta`,
    `message_stop`) into OpenAI chat.completion.chunk deltas. Auth via
    `x-api-key` + `anthropic-version: 2023-06-01`.
  - `gemini`: uses Google's OpenAI-compatible endpoint
    (`generativelanguage.googleapis.com/v1beta/openai/chat/completions`),
    near-identity transform with bearer auth.
  - `ollama`: uses the local OpenAI-compatible server
    (`localhost:11434/v1/chat/completions`), near-identity transform, no auth.
- `oxllm-providers::default_registry()`: builds the registry from the
  enabled Cargo features — adding a provider is one `register` call.
- `oxllm-providers/src/openai`: now implements `EmbeddingProvider`
  (near-identity), so `/v1/embeddings` works for OpenAI deployments.
- `oxllm-server`: new endpoints:
  - `POST /v1/completions`, `/completions` (alias) — legacy text completions,
    wraps the prompt into a chat message and forwards to the chat pipeline.
  - `POST /v1/embeddings`, `/embeddings` (alias) — embeddings via
    `EmbeddingProvider`.
- `oxllm-server::handlers::chat`: refactored to use the `ProviderRegistry`
  instead of hardcoding `OpenAiProvider`. Resolves an `Arc<dyn Provider>`
  per deployment and threads it through the retry / fallback / streaming
  pipeline. Streaming handler now owns its provider `Arc` so it survives
  `tokio::spawn`.
- `drop_params` handling: when `oxllm_settings.drop_params` is true and a
  provider publishes `supported_params()`, unsupported top-level request keys
  are stripped before calling `transform_request`. When false, unsupported
  keys are logged at `debug` level and passed through unchanged.
- Golden-file parity tests for Anthropic (request, response, tools) under
  `crates/oxllm-providers/src/anthropic/goldens/`.
- 183 tests passing (81 new: 31 Anthropic, 12 Gemini, 12 Ollama, 6 registry,
  3 chat-handler unit, 3 completions-handler unit, 11 new E2E, 3 OpenAI
  test updates).

### Phase 2 — Routing

- `oxllm-core::router`: pure routing module with `Router`, `RoutingStrategy`
  trait, `SimpleShuffle` strategy, `CooldownStore`, and `RequestContext`.
- Cooldowns: failed deployments (5xx/429/timeout/network) are temporarily
  excluded from selection for `cooldown_time_secs` (default 60s).
- Retries: `num_retries` on retryable errors, potentially selecting a different
  deployment on each retry.
- Fallbacks: `fallbacks` and `context_window_fallbacks` config keys.
  Regular fallbacks fire on retryable errors; context-window fallbacks fire on
  context-length-exceeded errors. Both set the `x-oxllm-fallback-used` response
  header. Fallbacks only fire before the first byte is streamed; never on 4xx
  client errors.
- Wildcard routing: `model_name: "*"` deployments match unmatched
  models. `openai/*` patterns match any model starting with `openai/`.
- Client disconnect handling: `tokio::select!` aborts the upstream
  stream when the client disconnects, freeing provider capacity.
- `stream_timeout`: max idle time between stream chunks, enforced
  via `tokio::time::timeout`.
- `/v1/models` deduplicates model names when multiple deployments share a
  `model_name`.
- 101 tests passing (37 new: 23 unit tests in core, 8 integration tests, 6
  state/config tests).

### Phase 0 — Foundation
- Cargo workspace with 5 crates: `oxllm-core`, `oxllm-providers`,
  `oxllm-server`, `oxllm-cache-redis`, `oxllm-db`.
- Root binary (`oxllm`) with `clap` CLI: `serve`, `validate`, `keygen` stubs.
- `oxllm-core`: `Endpoint` enum, `Error`/`Result` types, `Provider`/
  `ChatProvider` traits, `ResolvedCredentials`.
- `oxllm-providers`: `openai` feature with `OpenAiProvider` (identity transform).
- DAG layering test (`tests/workspace_layering.rs`) enforcing the workspace
  dependency graph.
- CI: fmt, clippy, test, deny, doc build.
- Tooling: `rust-toolchain.toml`, `rustfmt.toml`, `clippy.toml`, `deny.toml`.
- Docs: `docs/architecture.md`, `docs/adding-a-provider.md`, `README.md`,
  `examples/config.example.yaml`.

## Fixed

Comprehensive bug-fix batch — all 55 findings from `REVIEW_TODO.md` (review
batches 1–8) resolved.

### Streaming / budget metering
- Streaming chat injects `stream_options.include_usage` so budget/TPM/spend
  accounting applies to streams; realtime reservations roll back on disconnect
  and partial usage is accounted on client disconnect.

### Providers
- Anthropic cache-read cost no longer double-subtracted; `-YYYYMMDD` dated
  model suffixes now resolve pricing. Cohere embed sends a valid `input_type`
  and `p` instead of `top_p`. OpenRouter mid-stream error chunks detected;
  Azure API version updated.

### Auth
- JWT budget/rate-limit/cache buckets keyed off `sub` (token refresh no longer
  resets limits); X-Forwarded-For trusted right-to-left; admin console refuses
  non-loopback bind without a password; hot reload enforces require-auth and
  preserves the auth brake.

### Routing
- Wildcard deployments are provider-scoped (`openai/*` no longer matches every
  unmatched model); tag-based routing plumbed end to end; `least_busy`
  reserves at pick to avoid races; credential failover fixed.

### Config
- Explicit-null config blocks accepted; `router_settings` precedence corrected;
  K8s ConfigMap watcher fires real debounced reloads; Postgres uses
  `DOUBLE PRECISION` and correct column aliases.

### API
- `/v1/responses` and `/v1/messages` streaming forward tool calls and report
  real usage; legacy completions schema fixed and completions streaming no
  longer 501s.

### SSE
- UTF-8 framer skips hard-invalid bytes instead of stalling; parser is
  spec-compliant (comments, event/id/retry lines, mixed framing).

### Cache
- Stream flag included in the cache key (no more malformed cached streams);
  cache entry shape confusion fixed.

### Misc
- AuthBrake/counter maps evicted periodically; passthrough paths validated
  against traversal and response body reads capped; pricing table expanded;
  tokenizer updated for newer models.
