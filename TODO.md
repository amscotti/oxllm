# oxllm Unified TODO

> **Historical snapshot.** The counts, LOC, completion claims and backlog below
> are from the dated review, not current release evidence. Current source/tests
> supersede this snapshot.
> Retained as a historical artifact rather than a live issue tracker.

Last updated: 2026-07-30

---

## COMPLETED ✅

### Original Review (REVIEW_TODO.md) — 55/55 done
### Swarm Review (SWARM_REVIEW.md) — 34/34 done
### LiteLLM Upstream P0+P1 — 37/37 investigated (6 fixed, 24 safe, 7 N/A)
### Consistency Sweep — 6/6 done

**Total: 132 items completed across all review rounds.**

All critical, high, and medium-severity bugs are fixed, tested, and committed.

---

## PROJECT STATE

The project is substantially built out — **130K LOC** across the workspace:

- **33 providers** (OpenAI, Anthropic, Bedrock, Vertex AI, Bedrock Mantle, Azure, Cohere, Gemini, Groq, xAI, Together, OpenRouter, Mistral, Ollama, Databricks, Perplexity, SambaNova, Cloudflare, DeepSeek, Cerebras, AI21, Fireworks AI, DeepInfra, NVIDIA NIM, Voyage, Jina AI, WatsonX, HuggingFace, Deepgram, ElevenLabs, DashScope, Replicate)
- **10 guardrails** (AIM, Bedrock Guardrail, Custom, Injection, Lakera, MCP, Model Armor, Presidio, Semantic, + base)
- **22 hooks** (adaptive routing, alerting (email/pagerduty/slack), budget, cache, callback, cost, datadog, helicone, http telemetry, langfuse, langsmith, metrics, mlflow, rate_limit, s3 log, spend_keys, wandb)
- **4 cache backends** (Redis, Azure Blob, GCS, S3)
- **5 realtime providers** (OpenAI, Azure, Bedrock, xAI, + base)
- **3 pre-call checks** (budget limiter, image inlining, prompt caching)
- **Full admin UI** with templates, spend analytics, key/team/user management
- **26 integration test files**

---

## OPEN ITEMS

### A. Lower-priority fixes and improvements (32 items)

Low-priority fixes and feature candidates from an earlier review.

**Quick wins (small effort, high value):**
| # | Description |
|---|-------------|
| U34+U35 | Per-model prompt caching minimum tokens (hardcoded 1024 → price table field) |
| U45 | Raise /spend/logs/v2 page_size cap to 1000 |
| U46–U48 | Map cache_write_tokens for OpenAI/Responses cost tracking |
| U49 | OTel error.* attributes on failure spans |
| U52 | Return 400 instead of 500 for chat completions without messages |
| U54–U55 | Malformed/stale token limits on /v1/models |
| U57 | Honor zero fallback and aggregate-only TPM in passthrough |

**Feature parity (medium effort):**
| # | Description |
|---|-------------|
| U41–U42 | Per-key/team routing_strategy and tag_filtering override |
| U43–U44 | Tool spend rollup + 30-day cap |
| U60 | Context-window pre-call check for Responses API |
| U65–U66 | Atomic team member add / user upsert |

**N/A or skip:**
| # | Reason |
|---|--------|
| U38, U53, U58, U63, U64, U67–U69 | N/A in Rust / no equivalent feature |

### B. Future feature work (6 items)

| Feature | Prerequisite | Effort |
|---------|-------------|--------|
| ITPM/OTPM deployment rate limits | Add fields to OxllmParams | Medium |
| x-ratelimit-* response headers | None | Small |
| Anthropic adaptive thinking | Capability flags in price table | Medium |
| Anthropic native structured output | Capability flags in price table | Small |
| Independent coordination Redis | Separate config section | Medium |
| Anthropic auto prompt caching (U31) | New post-transform hook | Medium |

### C. Known gaps (partially addressed) (6 items)

| Gap | Current state | What's needed |
|-----|--------------|---------------|
| Router state wiped on hot-reload | Warning logged | Preserve Arc stores across reload |
| Partial-pattern wildcards | Only `provider/*` works | Add glob matching in wildcard_model_matches |
| Price table coverage | ~50 curated TOML models | Wholesale import from LiteLLM + generator script |
| Anthropic StreamMeta global cell | Clears on message_start | Per-stream state (needs trait-level change) |
| Admin session irrevocable | Stateless HMAC cookie | Server-side session store for revocation |
| max_parallel_requests | Stored but not enforced | Concurrency gauge (slot-based, not sliding window) |

### D. Broader project features

| Feature | Status | Effort |
|---------|--------|--------|
| MCP protocol support | Guardrail module exists, no server/client | Medium |
| SSO/SAML for admin UI | JWT auth exists | Medium |
| Per-key model ACLs | DB schema can be extended | Medium |
| Team membership table | Keys have team_id, no M:N | Medium |
| Cost optimization dashboard | Spend tracking exists | Medium |
| OTel streaming TTFT | OTel layer exists | Small |
| Responses API native dispatch | Bridge to chat works | Large |

---

## Summary

| Category | Done | Open |
|----------|------|------|
| All review rounds (P0–P1) | 132 | 0 |
| LiteLLM P2 minor fixes | 0 | 32 |
| Future ports | 0 | 6 |
| Known gaps | 0 | 6 |
| Broader features | 0 | 7 |
| **Total** | **132** | **51** |
