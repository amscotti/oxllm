# oxllm Review Worklist

> **Historical worklist.** Statuses below record an earlier review. Current
> verdicts are in source/tests; they supersede these statuses. Retained for traceability.

Status: `pending` | `in_progress` | `fixed` | `dropped`

## P0 — Critical

### 1. Streaming chat silently escapes all budget/TPM/spend accounting
- **Status:** fixed
- **Root cause:** chat path never injects `stream_options: {"include_usage": true}`; most clients don't send it → usage=None → budget/TPM/spend all skipped
- **Files:** `orchestration/upstream.rs:32`, `orchestration/stream.rs:188-213`, `hooks/budget.rs:246-252`, `hooks/rate_limit.rs:165-174`
- **Fix:** inject `stream_options.include_usage` in `attempt_upstream` when `is_stream=true`
- **Tests:** unit test verifying the flag is injected; e2e test verifying budget debit on stream

### 2. Anthropic prompt-cache reads double-subtracted — cost under-reported ~386x
- **Status:** fixed
- **Root cause:** `calculate_cost` assumes cache_read ⊆ prompt_tokens (OpenAI), but Anthropic excludes cache tokens from input_tokens
- **Files:** `oxllm-core/src/cost.rs:179-196`, `providers/anthropic/mod.rs:150-165`
- **Fix:** normalize cache conventions at extraction point or branch formula by provider
- **Tests:** unit test with Anthropic usage shape (input=100, cache_read=1M, output=50)

### 3. Anthropic dated model IDs never resolve pricing — Anthropic traffic free under budgets
- **Status:** fixed
- **Root cause:** `strip_date_version_suffix` only strips `-YYYY-MM-DD`; Anthropic uses `-YYYYMMDD` (e.g. `claude-3-5-sonnet-20241022`)
- **Files:** `oxllm-core/src/cost.rs:102-128`
- **Fix:** also strip 8-digit date suffixes
- **Tests:** unit test with `claude-3-5-sonnet-20241022`

### 4. Postgres broken: REAL columns can't decode to f64, count_spend reads wrong column name
- **Status:** fixed
- **Root cause:** schema uses REAL, Rust decodes f64; `try_get("COUNT(*)")` fails on Postgres (names it `count`)
- **Files:** `oxllm-db/src/lib.rs:90-155,796-808,820-868`
- **Fix:** DOUBLE PRECISION for Postgres; alias COUNT(*); add Postgres CI matrix
- **Tests:** unit test with Postgres-compatible column aliases

### 5. Admin console unauthenticated by default in Docker/Compose/Helm
- **Status:** fixed
- **Root cause:** `login_password: None` → auto-authorized; Docker binds 0.0.0.0; compose passes empty password
- **Files:** `oxllm-admin/src/app.rs:241-244`, `config.rs:148-150`, `Dockerfile.admin`, `docker-compose.yml:63`
- **Fix:** refuse to start on non-loopback without password; honor OXLLM_REQUIRE_ADMIN_PASSWORD
- **Tests:** unit test verifying config validation rejects empty password on non-loopback

### 6. JWT principals get fresh budget/rate-limit/cache buckets per token
- **Status:** fixed
- **Root cause:** buckets key off hash of entire JWT; token refresh resets all windows to zero
- **Files:** `auth/jwt.rs:138-141`, `pipeline/mod.rs:96`, `hooks/budget.rs:21-23`
- **Fix:** key buckets off `sub` (`jwt:{sub}` already computed)
- **Tests:** unit test verifying same sub + different exp hashes to same key

### 7. Wildcard deployments match ANY unmatched model regardless of provider prefix
- **Status:** fixed
- **Root cause:** `deployments_for_model` falls back to `model_name == "*"` for any unmatched model; `openai/*` doesn't scope
- **Files:** `oxllm-core/src/router/mod.rs:153-160`
- **Fix:** provider-scope the wildcard match
- **Tests:** unit test: openai/* wildcard does NOT match anthropic models

### 8. Single-feature provider builds fail to compile
- **Status:** fixed
- **Root cause:** `pub mod openai_compat` gated on features that don't include openai/azure/openrouter which reference it
- **Files:** `oxllm-providers/src/lib.rs:10-18`
- **Fix:** add openai/azure/openrouter to the cfg gate or make features imply each other
- **Tests:** CI check with --no-default-features --features openai

### 9. Anthropic provider forwards OpenAI `name` field → Anthropic 400s
- **Status:** fixed
- **Root cause:** `normalize_message` copies `name` into Anthropic message; also `developer`/`function` roles unhandled
- **Files:** `providers/anthropic/mod.rs:354-356`
- **Fix:** drop `name`; map `developer`→system; handle `function`→tool_result
- **Tests:** golden test with name field present

### 10. /v1/messages streaming drops all tool calls + zero usage
- **Status:** fixed
- **Root cause:** AnthropicStreamConverter only handles delta.content; tool_calls ignored; usage hardcoded 0
- **Files:** `handlers/messages.rs:563-603, 548-560`
- **Fix:** handle tool_calls deltas; accumulate usage from final chunk
- **Tests:** unit test with tool_calls stream chunks

### 11. Cohere embeddings send invalid input_type "text" — every request 422s
- **Status:** fixed
- **Root cause:** DEFAULT_EMBED_INPUT_TYPE = "text" not in valid Cohere values
- **Files:** `providers/cohere/mod.rs:32-33, 237-240`
- **Fix:** default to "search_document" for embed
- **Tests:** unit test verifying valid input_type

## P1 — High

### 12. X-Forwarded-For leftmost trust bypasses IP allowlist and auth-brake
- **Status:** fixed
- **Root cause:** leftmost XFF entry is client-controlled when proxies append
- **Files:** `security.rs:217-224`
- **Fix:** use rightmost entry or right-to-left walk skipping trusted hops
- **Tests:** unit test with multi-hop XFF

### 13. Hot reload silently removes auth brake for JWT-only; can drop to open mode
- **Status:** fixed
- **Root cause:** reload condition narrower than startup; OXLLM_REQUIRE_AUTH not enforced on reload
- **Files:** `state.rs:158-164, 224-231`, `app.rs:128`
- **Fix:** mirror startup condition; enforce require-auth on reload
- **Tests:** unit test verifying brake preserved on JWT-only reload

### 14. OpenRouter mid-stream error chunks not detected
- **Status:** fixed
- **Root cause:** duplicate SSE parser lacks in-body error check
- **Files:** `providers/openrouter/mod.rs:106-118`
- **Fix:** delegate to `openai_compat::sse::parse_openai_stream_chunk`
- **Tests:** unit test with error chunk

### 15. tag_based routing dead — tag never populated
- **Status:** fixed
- **Root cause:** RequestContext.tag always None
- **Files:** `router/tag_based.rs:38-43`, `orchestration/mod.rs:98-99`
- **Fix:** plumb tag from header or remove from docs/config
- **Tests:** unit test with tag header

### 16. Anthropic mid-stream error events silently swallowed
- **Status:** fixed
- **Root cause:** transform_stream_chunk drops unknown event types including error
- **Files:** `providers/anthropic/mod.rs:212-218`
- **Fix:** add error event handling
- **Tests:** unit test with error event

### 17. /v1/responses streaming always reports zero usage
- **Status:** fixed
- **Root cause:** usage chunk arrives after finish_reason; converter emits completion before usage captured
- **Files:** `handlers/responses.rs:645-664`
- **Fix:** defer completion until usage chunk or [DONE]
- **Tests:** unit test with usage chunk after finish

### 18. Realtime WebSocket: budget reserved never settled; usage unmetered
- **Status:** fixed
- **Root cause:** pre_call reserves, no post_success/post_failure ever runs
- **Files:** `handlers/realtime.rs:62-112`
- **Fix:** call post_failure on disconnect at minimum; document the gap
- **Tests:** e2e test verifying budget released on disconnect

### 19. Invalid UTF-8 mid-stream: silent stall + unbounded memory
- **Status:** fixed
- **Root cause:** hard-invalid bytes sit at head of pending_utf8 forever
- **Files:** `sse.rs:36-45`
- **Fix:** distinguish incomplete-tail from hard-invalid; skip/error bad byte
- **Tests:** unit test with invalid byte mid-stream

### 20. Provider catch-all passthrough unmetered/router-bypassing
- **Status:** fixed
- **Root cause:** verbatim forwarding, first-match deployment, no usage/cost
- **Files:** `handlers/passthrough.rs:318-441`, `pipeline/passthrough.rs:256-274`
- **Fix:** at minimum call post_failure on non-2xx; document catch-all as unmetered
- **Tests:** e2e test verifying hooks fire

### 21. Audio input tokens double-charged
- **Status:** fixed
- **Root cause:** audio tokens subset of prompt but billed additively
- **Files:** `oxllm-core/src/cost.rs:203-208`
- **Fix:** subtract audio from regular input bucket
- **Tests:** unit test with audio tokens

### 22. drop_params strips max_completion_tokens for Anthropic
- **Status:** fixed
- **Root cause:** supported_params list omits max_completion_tokens
- **Files:** `providers/anthropic/mod.rs:221-233`
- **Fix:** add max_completion_tokens to supported_params
- **Tests:** unit test verifying param preserved with drop_params=true
- **Also:** raise DEFAULT_MAX_TOKENS to 8192

### 23. Router state wiped on hot-reload; positional cooldown keys
- **Status:** fixed
- **Root cause:** fresh Router per reload; cooldown keys by index not stable id
- **Files:** `state.rs:147-148`, `router/mod.rs:199-201`
- **Fix:** preserve router state on reload; use deployment_key for cooldowns
- **Tests:** unit test verifying cooldowns survive reload

### 24. least_busy races under concurrency
- **Status:** fixed
- **Root cause:** in-flight increment after pick; thundering herd
- **Files:** `orchestration/chat.rs:200-236`
- **Fix:** reserve-at-pick or accept as known limitation
- **Tests:** concurrent test verifying distribution

### 25. rotate_key reactivates soft-deleted keys
- **Status:** fixed
- **Root cause:** UPDATE SET is_active=1 unconditionally
- **Files:** `oxllm-db/src/lib.rs:305-317`
- **Fix:** preserve is_active; use transaction
- **Tests:** unit test verifying disabled key stays disabled after rotate

### 26. Unbounded body reads and unvalidated management writes
- **Status:** fixed
- **Root cause:** response.text() no cap; negative limits accepted; expires unvalidated
- **Files:** `orchestration/upstream.rs:121`, `handlers/management/keys.rs`, `spend.rs`
- **Fix:** cap error body reads; validate inputs at management API
- **Tests:** unit tests for validation

## P2 — Medium

### 27. AuthBrake and counter maps grow unboundedly
- **Status:** fixed
- **Fix:** add periodic sweep/eviction; cap map size

### 28. Path traversal on pass-through targets
- **Status:** fixed
- **Fix:** normalize/validate paths; add timeout; respect body limit config

### 29. HookedRaw leaks budget on non-2xx
- **Status:** fixed
- **Fix:** call post_failure on non-2xx in hooked_raw

### 30. Budget scheduler ignores per-key duration
- **Status:** fixed
- **Fix:** track per-counter window start; use each counter's own duration

### 31. Cache-hit on streaming returns malformed stream
- **Status:** fixed
- **Fix:** include stream flag in cache key or serve as non-stream

### 32. SSE parser not spec-compliant
- **Status:** fixed
- **Fix:** handle comments, event/id/retry lines, mixed framing

### 33. Client disconnect misaccounted
- **Status:** fixed
- **Fix:** partial usage accounting on disconnect

### 34. ?authorization= accepted on all routes
- **Status:** fixed
- **Fix:** restrict to realtime route

### 35. error: null false positives terminate streams
- **Status:** fixed
- **Fix:** check !error.is_null() before treating as error

### 36. Multi-segment model IDs fail pricing; tokenizer misclassifies
- **Status:** fixed
- **Fix:** iterative prefix stripping; update tokenizer_for for newer models

### 37. usage_based strategy consumes raw counts not utilization
- **Status:** fixed
- **Fix:** compute utilization percentage

### 38. Virtual-key staleness across replicas
- **Status:** fixed
- **Fix:** document; consider shorter TTL or pub/sub invalidation

### 39. Management API gaps and validation
- **Status:** fixed
- **Fix:** clear_max_budget on keys; pagination; existence checks; proper error codes

### 40. Cohere top_p sent under wrong name
- **Status:** fixed
- **Fix:** rename to `p`

## P3 — Low

### 41. Raw buffer flush forwards untransformed bytes
- **Status:** fixed
### 42. Legacy completions streaming returns chat-chunk format
- **Status:** fixed
### 43. merge_base_path_query duplicate keys + .. normalization
- **Status:** fixed
### 44. Anthropic cache fields non-standard in OpenAI usage
- **Status:** fixed
### 45. Anthropic streaming chunks omit id/created/model
- **Status:** fixed
### 46. Azure stale API version, unencoded path, sentinel host
- **Status:** fixed
### 47. openai_compat drops newer OpenAI params
- **Status:** fixed
### 48. Partial-pattern wildcards don't work
- **Status:** fixed
### 49. Config parsing rejects explicit-null blocks
- **Status:** fixed
### 50. K8s ConfigMap updates never fire reload; debounce not real
- **Status:** fixed
### 51. Benches don't compile on stable; --all-targets clippy fails
- **Status:** fixed
### 52. Price table coverage gaps
- **Status:** fixed
### 53. Token estimator undercounts tool/image/system overhead
- **Status:** fixed
### 54. Documentation and examples stale
- **Status:** fixed
### 55. Remaining low-severity nits (dead constants, SseFramer quadratic, unmetered token_counter, Prometheus cardinality, spend writer drops, RateLimitHook no post_failure, parse_fallback string shorthand, Database::connect memory match, spend_calculate GET doc, v2 cursor always next_cursor, admin query encoding, admin session irrevocable)
- **Status:** fixed
