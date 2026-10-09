# Upstream update work list, 2026-10-08

Reference: LiteLLM `b370996b9d` through `f824d11a22383abc6cb316cb51cb4e739afeecbd`.
The native `x-oxllm-*` client contract remains primary.

## Implementation checklist

- [x] Claude Haiku 5.5 rate cards, context tiers and cache prices across shipped
  providers, plus corrected Sonnet 5.5 cache-read rates.
- [x] Refresh changed catalog prices/limits and new model cards, including
  service-tier overlays and the pinned context-tier snapshot. Remove the stale
  Fireworks empty-model row. Catalog entries describe pricing; endpoint support
  continues to come from provider capabilities.
- [x] Retry pre-content stream drops/error frames and walk the fallback chain
  without re-entering failed primaries (`68d9b8bbb8`, `88f15e7572`). Hold failed
  metadata before client commitment and release attempt-level in-flight slots.
- [x] Honor request-scoped fallbacks during those failures (`328f5a720c`), including
  context/content-policy chains, empty/null controls, ACLs and server fallback caps.
- [x] Keep router-only controls off provider bodies (`a7ee038592`), including the
  final configured-body shaping pass.
- [x] Azure/Azure AI Model Router selected-model cost plus one infrastructure
  charge (`e2c55daaaf`), with JSON/SSE spend consistency and override precedence.
- [x] Mantle/Foundry explicit OpenAI cache breakpoints (`7c7b0ea85b`), strict
  provider-scoped flags, null-option defaults and caller-mark preservation.
- [x] Bedrock streaming request timeouts (`1b98748528`), covering Converse/Invoke
  header waits and per-read idle budgets without a total streaming deadline.
- [x] Vertex Claude counting system/tools forwarding (`6d73fa6b49`), dedicated
  count region, unversioned counting model IDs and optional utility local fallback.
- [x] Native Count Tokens image normalization (`6befb9ad7d`) through nested tool
  results, including provider-native auth on direct/hosted Claude routes.
- [x] Foreign thinking-signature cleanup on Gemini (`bc2df7cdcd`) while retaining
  native Google fields and tool IDs; correct Vertex multi-region HTTP hosts.
- [x] Anthropic OIDC file/env federation and missing-ID failure boundaries
  (`0ed1c08f02`, `c4b48f6b3f`), cached JSON grants and merged OAuth beta flags.
  Additional identity sources plug into the existing credential-resolver registry.
- [x] Preserve caller W3C trace headers across routing attempts and configured
  defaults; caller body trace/session IDs keep their own precedence (`02b8b5cc80`).

## Reviewed applicability

- Prompt-cache eligibility (`4c6c84afb7`): the current pin implementation already
  tokenizes only the selected prefix, not the whole conversation.
- Model-group pricing (`c13746c951`): settlement already follows the selected
  deployment, independent of alias chains.
- Stream double-wrapping (`3ca3e1c686`) and Python sync-generator changes: the Rust
  host uses one retry chain and a single stream pump.
- Terminal errors before content use the native HTTP-error contract after
  recovery is exhausted; post-commit failures retain their SSE error contract.
- Vertex Live host/TLS (`70e6a2ff24`): there is no Vertex Live transport in this
  checkout. Its shared multi-region host behavior is applied to current HTTP paths.
- Stored-body spend masking fixes: spend records contain attribution and usage,
  without stored request/tool/logprob bodies.
- New evaluation-only models were reviewed without adding an evaluation endpoint.

## Review dispositions

The review's 21 points were checked against the current implementation and pinned
source. Confirmed defects were corrected; the regression cases below cover the
behavioral boundaries rather than accepting every reported concern as a defect.

| # | Finding | Disposition and evidence |
| --- | --- | --- |
| 1 | Fireworks priority overlays grouped differently priced Flash variants | Fixed: v4p1 and 0731/vision overlays are split, including account-scoped aliases. Published per-bucket cost tests cover all six identifiers. |
| 2 | Claude 5.5 batch cache/context buckets were incomplete | Fixed: all published buckets are present on six Anthropic/Vertex Sonnet/Haiku cards, including Haiku's >100k Anthropic batch tier. Provider-specific and undiscounted 1h rates are retained. |
| 3 | Client retry overrides accepted unbounded `u32` budgets | Fixed: request overrides accept 0–10; 11 and `u32::MAX` return 400 before any upstream call across translated/native/counting/embedding surfaces. Trusted configured policies keep precedence. |
| 4 | Structured/named internal-error policies omitted 529 | Fixed: overloads share the internal-error category and named status translation. Numeric status policies still win. |
| 5 | Anthropic timeout/not-found frames fell through to 500 | Fixed: frames map to 504/404. The existing status contract remains: timeout-category aliases cover 408; 504 belongs to the internal-error category. |
| 6 | Malformed pre-content transforms aborted recovery as client 500s | Fixed: transform/serialization errors become retryable 502s with a generic client message. A wire regression recovers through fallback without leaking failed metadata or parse detail. |
| 7 | Deployment unary `timeout` changed stream idle precedence | Fixed regression: request `stream_timeout`/`timeout`, deployment `stream_timeout`, global `request_timeout`, then 60s. A short unary deployment timeout no longer shortens streams; fractional request overrides are retained. |
| 8 | Passthrough shaping missed standalone gateway controls | Fixed: presence detection and stripping share one registry, including standalone fallback-error/cache/chunk/traceback flags and `metadata.tags`. Native wire tests exercise each independently. |
| 9 | Messages/Responses dropped opt-in fallback error summaries | Fixed: bridge control copying uses the same registry. Messages/Responses/Completions preserve sanitized `x-oxllm-fallback-errors` and serving-fallback identity. |
| 10 | Model Router fee estimates lost the unknown-price floor or selected unintended models | Fixed: unknown deployment admission reserves $0.01 plus the fee; canonical fee-only cards charge the effective fee once. Detection requires recognized Azure selectors/labels or a base-model hint. All-in/zero overrides and cache-hit refunds are covered. |
| 11 | Fleet identity-file defaults overrode deployment references | Fixed: deployment file/reference sources precede environment defaults. Injected-environment tests pin the complete precedence without mutating process state. |
| 12 | Assertion normalization/JSON serialization weakened credential bounds and lifetime | Fixed: raw bytes are capped before trimming/UTF-8 conversion, assertion source buffers and the owned serialized JSON grant are zeroizing. Boundary, suffix/padding, serialization, cancellation and refresh tests pass; HTTP/TLS/header copies are outside this guarantee. |
| 13 | Deployment retry configuration had duplicate typed/extra paths | Fixed: retry resolution uses typed `OxllmParams.num_retries`; the extra-map branch and allowlist entry are removed. Deployment retry wire coverage remains in the streaming suite. |
| 14 | Foundry OpenAI requests retained incompatible Anthropic cache controls | Fixed: incompatible message/block/tool controls are removed before native injection. Native marks survive mixed/dual-mark blocks; request selector decorations remain lenient while deployment parsing stays strict. |
| 15 | Body overrides/allowed params could reintroduce cache selectors | Fixed: cache selectors remain available during provider transformation, then final wire shaping strips them and other gateway controls after merges/restoration. Chat/embedding/native wire tests cover configured reintroduction. |
| 16 | Marker census counted arbitrary schema/metadata properties | Fixed: only message/block annotations consume the marker budget. System-message blocks count; unrelated schema/tool metadata and external Anthropic controls do not reserve native OpenAI slots. Exact regional cards remain authoritative. |
| 17 | Cancellation during stream startup leaked slots/reservations | Fixed: synchronous in-flight ownership and cancellation-aware router-budget ownership span header/pre-content waits. Tests abort both startup stages and interrupt an already-applied first-scope refund; both scopes finish exactly once. |
| 18 | Thinking-signature repair omitted the effective retry-budget header | Fixed: the one-shot repair is included in `x-oxllm-max-retries`; a wire regression with zero configured retries records one started retry and budget 1. |
| 19 | Disabled local counting still ran after an unavailable native result | Fixed: disabled local counting returns 503 for missing/unsupported native results and local-only requests; native errors keep their status. Malformed boolean controls return 400. |
| 20 | Federation-disable/request credentials escaped validation through converters | Fixed: the disable flag and all federation credentials are rejected in raw bodies before conversion/normalization. Configured body overrides cannot put those fields onto a generation wire. |
| 21 | Strict Vertex location validation with a custom base was considered a regression | Intentional validation: configured locations are nonempty lowercase path-safe names even with a custom base. Tests cover Gemini/Claude, valid regions/multi-regions/global and rejected uppercase/path/query-bearing names. Operator docs state the contract. |

Additional review sub-findings were resolved as follows:

- Extreme negative cache indexes do not overflow: adding a nonnegative message
  count to a negative `i64` cannot underflow. Extreme-index/malformed-message
  regressions confirm safe no-ops.
- Broad `<region>/gpt-*` Foundry flag fallback is not a supported routing
  convention. The genuine flag regression was commercial-card inheritance by
  unflagged GovCloud cards; exact regional cards now win.
- Stable Anthropic Count Tokens needs no automatic counting beta. Caller and
  OAuth betas are preserved, filtered and deduplicated on translated/native chat
  and counting; the federation wire regression asserts each appears once.
- Missing/malformed federation configuration fails before assertion access or
  exchange. Authorized sibling/configured fallback recovery remains deliberate;
  a routing regression proves zero bad-deployment HTTP calls and blocks an
  unauthorized fallback.
- Native Messages passthrough remains single-attempt. Translated Messages uses
  chat recovery; post-commit failures never replay content. Retry budgets are
  resolved for the selected deployment rather than copied from a failed sibling.
- Prelude replay now coalesces retained bytes, bounding tiny-chunk bookkeeping
  by the existing 16 MiB byte limit without adding a separate chunk-count policy.
- Model Router custom prices remain all-in, Perplexity Haiku remains base-rate
  only, and zero reasoning-token normalization is intentional.
- Provider helpers are gated to their consumers, including the Azure-only
  reasoning helper and Anthropic/Bedrock image/schema/thinking helpers, so
  standalone Gemini/Mantle builds remain warning-free.
- Full validation caught and corrected two integration regressions: a successful
  first attempt still omits `x-oxllm-max-retries`, and bridge conversion still
  omits null cache/selectors. Started thinking-repair retries expose their budget.
- Bedrock's standalone feature now declares its actual Anthropic transform
  dependency, matching the existing hosted-provider feature relationships.
- Combined validation exposed duplicate OS-watch registrations for logical and
  canonical directory aliases. Directory watches now deduplicate those aliases;
  callback overflow uses a bounded nonblocking queue and a coalesced rescan so
  startup/restart cannot wait for a blocked callback or lose the last change.
  Dedicated queue/alias regressions and the full server suite cover the fix.

## Validation

- Post-review full workspace tests pass with isolated PostgreSQL and Redis:
  `cargo test --locked --workspace --all-features --quiet -- --test-threads=8`.
  This includes database upgrade contracts and cross-replica cache tests.
- Core: 764 tests; providers: 2,036 tests; database: 161 tests; all-feature server:
  3,084 tests. No-default-feature server: 2,748 tests.
- All 14 upstream-update wire tests, 16 round-2 compatibility tests, nine
  streaming/router feed tests and Azure JSON/SSE spend tests pass. Coverage
  includes malformed preludes, cancellation/refund ownership, authorized
  federation recovery, raw control validation and final body stripping.
- Focused pricing, credentials, cache/token-counter, native OAuth beta,
  retry-policy and watcher queue/directory-alias regressions also pass.
- Workspace all-feature/all-target Clippy and no-default server Clippy pass with
  `-D warnings`. Standalone Anthropic, Gemini, Mantle and Bedrock provider
  feature Clippy checks pass with warnings denied.
- Public library Rustdoc passes with warnings denied; stale private/broken links
  found by the check were corrected.
- `cargo fmt --all -- --check`, `git diff --check`, refreshed catalog parity,
  and the federation sample's CLI validation pass.
- Disposable PostgreSQL/Redis fixtures were removed after validation.
- Synthetic upstreams cover protocol behavior; credential-dependent live
  provider tests skip when their required environment variables are absent.
