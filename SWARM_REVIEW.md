# Swarm Review — Consolidated Issue List

> **Historical review artifact.** Findings and completion claims below describe
> an earlier review, not the current tree. They are superseded by current source
> and regression tests. Retained for traceability; not release certification.

Generated from 6 parallel review agents covering all changed areas.
Severity: CRITICAL → HIGH → MEDIUM → LOW → INFO

---

## CRITICAL (auth bypass, data loss)

### C1. Open mode bypasses virtual-key auth when DB is configured but no master key/JWT
**File:** `crates/oxllm-server/src/auth/mod.rs:348`  
**Root cause:** The open-mode gate at line 204 checks `master_key_hash.is_none() && jwt_validator.is_none()`. It does NOT consider whether a database is configured with virtual keys. A deployment with `database_url` + virtual API keys but no `master_key` or `jwt_secret` operates in open mode — every request authenticates as `AuthUser::open()`.  
**Recommendation:** Check whether a DB is configured (`state.db.is_some()`) before entering open mode. If virtual keys exist, require master key or JWT.

---

## HIGH (production impact, exploit potential)

### H1. `Algorithm::None` accepted via JWT config — signature verification can be disabled
**File:** `crates/oxllm-server/src/auth/jwt.rs:152-163`  
**Root cause:** `parse_algorithm` accepts `Algorithm::None` from config. An operator setting `jwt_algorithm: "none"` disables all signature verification. Anyone can forge tokens.  
**Recommendation:** Reject `Algorithm::None` in `parse_algorithm` with a hard error.

### H2. JWT `aud` claim included in hash derivation without validation
**File:** `crates/oxllm-server/src/auth/jwt.rs:211-214`  
**Root cause:** `claims_to_user` derives the bucket hash from `aud` when `iss` is absent, even when `jwt_audience` is unset (no validation). A client that controls `aud` could fragment budget tracking or collide with another user's bucket.  
**Recommendation:** Only include `aud` in the hash when `jwt_audience` is configured (and therefore validated). Exclude it when unvalidated.

### H3. `lookup_pricing` prefix stripping breaks for 4+ level prefixes
**File:** `crates/oxllm-core/src/cost.rs:284-298`  
**Root cause:** The iterative strip loop has a `levels >= 3` break that fires before trying the bare model name. `a/b/c/d/gpt-4o` strips to `d/gpt-4o` then exits — never tries `gpt-4o`.  
**Recommendation:** Move the break to AFTER the table.get check, or increase the limit.

### H4. Missing `#[cfg(feature = "replicate")]` on provider registration
**File:** `crates/oxllm-providers/src/lib.rs:187`  
**Root cause:** Line 187 unconditionally references `replicate::PROVIDER`, but the module is gated behind `#[cfg(feature = "replicate")]`. Building without `replicate` but with other features fails to compile.  
**Recommendation:** Add `#[cfg(feature = "replicate")]` to line 187.

---

## MEDIUM (correctness, resource leaks, edge cases)

### M1. `[DONE]` marker check before prefix stripping — `data:[DONE]` not recognized as terminal
**File:** `crates/oxllm-providers/src/openai_compat/sse.rs:27`  
**Root cause:** The check matches `"data: [DONE]"` (with space). `data:[DONE]` (no space) bypasses it and parses `"[DONE]"` as a JSON array.  
**Recommendation:** Move the terminal check AFTER prefix stripping, comparing `json_str.trim() == "[DONE]"`.

### M2. Anthropic `build_finish_chunk` sets `prompt_tokens: 0` — prompt tokens lost in stream
**File:** `crates/oxllm-providers/src/anthropic/mod.rs:2800-2803`  
**Root cause:** The finish chunk's usage only has `output_tokens`. Prompt token count (computed in `build_role_chunk`) is not stored in StreamMeta. Clients reading usage from the final chunk miss prompt tokens entirely.  
**Recommendation:** Store `input_tokens` in StreamMeta and emit them in `build_finish_chunk`.

### M3. Streaming SSE error from OpenAI upstream always maps to status 500
**File:** `crates/oxllm-providers/src/openai_compat/sse.rs:52-55`  
**Root cause:** Error `code` is read with `as_u64()`, but OpenAI uses string codes like `"rate_limit_exceeded"`. The fallback always produces 500 instead of the correct status.  
**Recommendation:** Also check `error.get("type")` for known type strings, or parse `code` as a string with status mapping.

### M4. Duplicate `[DONE]` emission on malformed SSE boundaries
**File:** `crates/oxllm-server/src/orchestration/stream.rs:237-256`  
**Root cause:** The `Ok(None)` branch emits `[DONE]` unconditionally when `buffer.trim().contains("[DONE]")`, not gated by `done_sent`.  
**Recommendation:** Gate the `[DONE]` send with `if !*done_sent`.

### M5. Converter streams silently terminate on read errors without error events
**File:** `crates/oxllm-server/src/handlers/messages.rs:155-158`, `responses.rs:116-119`  
**Root cause:** `Some(Err(e))` from the data stream logs and returns — no error event forwarded to client.  
**Recommendation:** Forward an error event + terminal marker before return.

### M6. Cohere `developer` role not extracted — potential 400
**File:** `crates/oxllm-providers/src/cohere/mod.rs:628`  
**Root cause:** `split_system_messages` only checks `role == "system"`, not `role == "developer"`.  
**Recommendation:** Add `|| role == "developer"` matching the Anthropic pattern.

### M7. Cohere legacy `function` role not handled — potential 400
**File:** `crates/oxllm-providers/src/cohere/mod.rs:663-666`  
**Root cause:** `normalize_message` only handles `role == "tool"`, not `role == "function"`.  
**Recommendation:** Add `|| role == "function"`.

### M8. `record_failure` counts requests with NO auth header against auth-brake
**File:** `crates/oxllm-server/src/auth/mod.rs:371-372`  
**Root cause:** `extract_auth_token` returning `None` triggers `record_failure`. An attacker sending empty-header requests exhausts a legitimate user's auth-brake budget from the same IP.  
**Recommendation:** Don't record auth-brake failures for requests that carry zero credentials.

### M9. `InMemoryCache::delete` does not remove counter state
**File:** `crates/oxllm-server/src/cache/mod.rs:173-175`  
**Root cause:** `delete` removes the moka entry but not the `counters: HashMap` entry. `counter_keys()` and `counter_value()` still return stale data.  
**Recommendation:** Also call `self.counters.write().await.remove(key)` in `delete`.

### M10. Unbounded upstream error body in embeddings handler
**File:** `crates/oxllm-server/src/handlers/embeddings.rs:196-206`  
**Root cause:** Non-success path reads `response.text().await` with no size limit.  
**Recommendation:** Use bounded read + `check_bytes_limit` before parsing.

### M11. `lookup_pricing` 4+ level prefix stripping breaks
**File:** `crates/oxllm-core/src/cost.rs:284-298`  
**Root cause:** The loop's `levels >= 3` break fires before trying bare model name.  
**Recommendation:** Increase limit or restructure to check AFTER stripping.

---

## LOW (nits, edge cases, test isolation)

### L1. `tokenizer_for` ordering fragile — new `gpt-4.*` models could fall into wrong tokenizer
**File:** `crates/oxllm-core/src/tokens.rs:23-38`

### L2. `session_affinity` calls itself "consistent hashing" but uses plain modulo
**File:** `crates/oxllm-core/src/router/session_affinity.rs:54`

### L3. `any_data` cannot distinguish "no data" from "zero usage" in usage_based strategy
**File:** `crates/oxllm-core/src/router/usage_based.rs:47-49`

### L4. `top_k` missing from `SUPPORTED_CHAT_PARAMS`
**File:** `crates/oxllm-providers/src/openai_compat/mod.rs:36-68`

### L5. Unknown SSE field lines produce Err instead of Ok(None)
**File:** `crates/oxllm-providers/src/openai_compat/sse.rs:33-38`

### L6. Empty `data:` line causes JSON parse error
**File:** `crates/oxllm-providers/src/openai_compat/sse.rs:40-45`

### L7. Admin login brake ignores XFF — collapses all proxy clients into one bucket
**File:** `crates/oxllm-admin/src/auth.rs:424`

### L8. `DualCache::incr` silently swallows memory increment failure
**File:** `crates/oxllm-server/src/cache/mod.rs:351`

### L9. SSE `debug_assert` can lose data in release builds
**File:** `crates/oxllm-server/src/sse.rs:43-49`

### L10. `is_blocked` creates map entries for IPs merely checking their status
**File:** `crates/oxllm-server/src/security.rs:230-233`

### L11. JTI stored verbatim without hashing — oversized cache keys possible
**File:** `crates/oxllm-server/src/auth/jwt.rs:125`

### L12. `SystemTime::now()` used for JTI TTL — non-monotonic clock can extend cache TTL
**File:** `crates/oxllm-server/src/auth/jwt.rs:130-136`

### L13. DB transient error causes 5-second negative-cache tombstone
**File:** `crates/oxllm-server/src/auth/mod.rs:499-510`

### L14. Rerank transform vs raw fallback return different error shapes
**File:** `crates/oxllm-server/src/handlers/passthrough.rs:181-198 vs 109-121`

### L15. TPM rollback of RPM increment may lose key TTL
**File:** `crates/oxllm-server/src/pipeline/mod.rs:308-311`

### L16. Streaming relay bypasses Content-Length pre-check
**File:** `crates/oxllm-server/src/security.rs:506-519`

### L17. Tests share global metrics state (rare flakes)
**File:** `tests/routing_metrics_e2e.rs:153-154`

### L18. `list_keys` silently truncates at 1000 without pagination
**File:** `crates/oxllm-server/src/handlers/management/keys.rs:152-160`

---

## Summary

| Severity | Count | Key areas |
|---|---|---|
| CRITICAL | 1 | Auth open-mode gate |
| HIGH | 4 | JWT algorithm/aud, pricing lookup, build cfg |
| MEDIUM | 11 | SSE parser (+Anthropic/Cohere), auth-brake, cache, embeddings, [DONE] |
| LOW | 18 | Tokenizer, session_affinity, error shape, TTL, tests, docs |
| **Total** | **34** | |

This file was generated from 6 parallel review agents covering:
- Streaming & Metering (Agent 1)
- Auth & Security (Agent 2)
- Provider modules (Agent 3)
- Core & Router (Agent 4)
- Config & Infrastructure (Agent 5)
- Pipeline, Passthrough & E2E (Agent 6)
