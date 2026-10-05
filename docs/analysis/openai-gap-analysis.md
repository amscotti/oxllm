# OpenAI Provider Gap Analysis — oxllm vs LiteLLM

Generated from 6 parallel agents scanning LiteLLM provider code, test files, and
comparing against oxllm's OpenAI provider implementation.

---

## HIGH Priority Gaps

### H1. O-Series: `max_tokens` → `max_completion_tokens` mapping
- **Severity**: HIGH
- **LiteLLM**: `o_series_transformation.py:98` — maps `max_tokens` → `max_completion_tokens` for o1/o3/o4; `gpt_5_transformation.py:256` for gpt-5.
- **oxllm**: `openai/mod.rs:89-103` — pure pass-through. No mapping.
- **Impact**: Requests with `max_tokens` to o1/o3/o4-mini/gpt-5 get HTTP 400.
- **Also applies to**: OpenAI-compatible providers (openai_compat) that reverse-map for backends.

### H2. O-Series: Unsupported params not stripped
- **Severity**: HIGH
- **LiteLLM**: `o_series_transformation.py:52-58` — strips `logprobs`, `top_p`, `presence_penalty`, `frequency_penalty`, `top_logprobs` for o-series.
- **oxllm**: `openai/mod.rs:92-101` — only handles temperature. All others pass through.
- **Impact**: `top_p`, `frequency_penalty`, `logprobs` cause 400 errors on o1/o3/o4-mini.

### H3. No reasoning-token-aware cost calculation
- **Severity**: HIGH
- **LiteLLM**: `utils.py:807-813` — `generic_cost_per_token` separates reasoning_tokens and applies `output_cost_per_reasoning_token` rate.
- **oxllm**: `cost.rs:414-415` — single `output_cost_per_token` rate for all completion tokens.
- **Impact**: o-series costs significantly undercounted (reasoning tokens are ~2-3x more expensive).

### H4. `reasoning_tokens` not extracted from usage
- **Severity**: HIGH
- **LiteLLM**: `types/utils.py:1613-1620` — reads `completion_tokens_details.reasoning_tokens`, auto-calculates `text_tokens`.
- **oxllm**: `orchestration/usage.rs:93` explicitly sets `reasoning_tokens: None`.
- **Impact**: Cost, hooks, and synthesized usage never capture reasoning tokens from OpenAI o-series, DeepSeek, or Grok.

### H5. Null token values not normalized to 0
- **Severity**: HIGH
- **LiteLLM**: `openai_like/chat/transformation.py:62-76` — `_sanitize_usage_obj` replaces null token fields with 0.
- **oxllm**: `openai/mod.rs:106` — identity pass-through. Null values reach the client.
- **Impact**: Many OpenAI-compatible providers return null tokens. Strict clients break on type mismatches.

### H6. No streaming `reasoning_content` aggregation
- **Severity**: HIGH
- **LiteLLM**: `main.py:8608-8619` — extracts `reasoning_content` chunks from streaming deltas, combines them, counts tokens.
- **oxllm**: `openai/mod.rs:109-111` — no reasoning content aggregation across chunks.
- **Impact**: Streaming o-series responses lose reasoning text and `reasoning_tokens` is always None.

### H7. Anthropic `json_schema` type not handled
- **Severity**: HIGH
- **LiteLLM**: Anthropic transforms handle both `json_object` and `json_schema` response_format types.
- **oxllm**: `anthropic/mod.rs:320-374` — only checks for `json_object`. `json_schema` with strict schema is silently ignored.
- **Impact**: Users sending `response_format: {type: "json_schema"}` to Anthropic get plain text instead of structured JSON.

### H8. O-Series: `system` → `user` role conversion missing
- **Severity**: HIGH
- **LiteLLM**: `o_series_transformation.py:148-152` — converts `system` role messages to `user` for models without system support.
- **oxllm**: `openai/mod.rs:89-103` — no message transformation.
- **Impact**: Older o1 models that reject `role: system` get HTTP 400.

### H9. No organization/project header support
- **Severity**: HIGH
- **LiteLLM**: `openai.py:350-402` — passes `organization` to OpenAI client; `audio_speech()` passes both `organization` and `project`.
- **oxllm**: `openai/mod.rs:73-87` — `apply_auth()` only sets `Authorization` header.
- **Impact**: Multi-org OpenAI users can't route to specific orgs.

---

## MEDIUM Priority Gaps

### M1. Dict-shaped `reasoning_effort` not normalized
- **LiteLLM**: `gpt_5_transformation.py:14-28` — converts dict `{"effort":"high"}` to string `"high"`.
- **oxllm**: Pass-through — dict sent as-is to Chat Completions API, which expects a string.
- **Impact**: Dict reasoning_effort (common in AI SDK interop) causes 400.

### M2. `stream_options.include_usage` default not set
- **LiteLLM**: `openai.py:1129-1139` — adds `stream_options: {include_usage: true}` by default.
- **oxllm**: Whatever is in the request passes through. No default.
- **Impact**: Streaming responses don't include usage data unless client explicitly sets it.

### M3. `supported_params` returns None for OpenAI
- **LiteLLM**: `gpt_transformation.py:141-187` — model-specific list of ~25+ params.
- **oxllm**: `openai/mod.rs:113-115` — returns `None` (all pass).
- **Impact**: `warn_unsupported_params`/`drop_params` doesn't work. Introspection is blind.

### M4. 200-response error bodies not detected
- **LiteLLM**: `convert_dict_to_response.py:588-618` — checks for `error` key in 200 response.
- **oxllm**: No error check — treats 200 with `{"error":...}` as success.
- **Impact**: Provider errors as 200 bodies (OpenRouter, etc.) treated as successful completions.

### M5. Synthesized usage chunk lacks detail fields
- **LiteLLM**: Usage model carries `completion_tokens_details` and `prompt_tokens_details` with all sub-fields.
- **oxllm**: `stream.rs:637-645` — only sets prompt/completion/total_tokens. No details.
- **Impact**: Clients relying on `include_usage` in streams see no reasoning/cache/audio tokens.

### M6. Duplicate usage chunk when `include_usage` requested
- **LiteLLM**: OpenAI SDK handles `include_usage` transparently — no synthetic chunk.
- **oxllm**: Always synthesizes a usage chunk on `[DONE]`, without checking if upstream already sent one.
- **Impact**: OpenAI clients get two usage chunks — real one (from upstream) then synthetic one.

### M7. `reasoning` → `reasoning_content` delta mapping missing
- **LiteLLM**: `gpt_transformation.py:767-784` — maps `delta.reasoning` to `delta.reasoning_content`.
- **oxllm**: No mapping in `transform_stream_chunk`.
- **Impact**: GLM-5/hosted_vllm reasoning content not visible in expected `reasoning_content` field.

### M8. Message content block transforms missing
- **LiteLLM**: `gpt_transformation.py:348-378` — image_url string→dict normalization, file content processing, PDF URL download.
- **oxllm**: Identity pass-through. No content block normalization.
- **Impact**: `file` type blocks with file_data but no filename fail. Image_url as string passes through (OpenAI is lenient).

### M9. Image_url → Anthropic `image` conversion missing (direct Anthropic)
- **LiteLLM**: Anthropic transforms convert `image_url` to `image` type.
- **oxllm**: **FIXED IN TIER 2 E2E** — the E2E vision test was broken and we fixed it.
- **Status**: Already fixed in recent commit.

### M10. Cache_control stripping for openai.com missing
- **LiteLLM**: `gpt_transformation.py:404-423` — preserves cache_control for non-openai.com hosts; strips for real OpenAI.
- **oxllm**: No cache_control handling. Anthropic-style markers forwarded to OpenAI.
- **Impact**: Clients sending `cache_control` through OpenAI provider get 400.

### M11. Anthropic/Bedrock: `strict` → `required` population missing
- **LiteLLM**: Sets both `additionalProperties: false` AND `required = properties.keys()`.
- **oxllm**: Only injects `additionalProperties: false`. No `required` population.
- **Impact**: Strict-mode schemas may fail validation on Anthropic/Bedrock.

### M12. Streaming error extraction from 200 stream body
- **LiteLLM**: `gpt_transformation.py:787-799` — extracts error payloads from stream chunks with HTTP 200.
- **oxllm`: No error extraction from stream chunks — silent data loss on vLLM/sglang errors.
- **Note**: Already handled for non-streaming via `parse_openai_stream_chunk` in SSE parser. Check if streaming path also catches this.

### M13. Fake streaming for non-streaming models
- **LiteLLM**: `openai.py:632-635` — wraps non-streaming response in SSE chunks when model doesn't support streaming.
- **oxllm**: No fake streaming.
- **Impact**: Older o1 models that don't support streaming break when clients request `stream: true`.

---

## LOW Priority Gaps

### L1. `completion_tokens_details` completely ignored in `extract_usage`
- **LiteLLM**: `Usage` model carries the full sub-object.
- **oxllm**: `usage.rs` never reads `completion_tokens_details` (audio_tokens, accepted_prediction_tokens, rejected_prediction_tokens).
- **Impact**: These fields silently dropped from cost/hook tracking.

### L2. `text_tokens` never auto-calculated
- **LiteLLM**: Auto-calculates `text_tokens = completion_tokens - reasoning_tokens - image_tokens - audio_tokens`.
- **oxllm**: No equivalent logic.

### L3. `prompt_tokens_details.image_tokens` not extracted
- **LiteLLM**: `PromptTokensDetailsWrapper` includes `image_tokens`.
- **oxllm**: `usage.rs` reads only `audio_tokens` from `prompt_tokens_details`.

### L4. `$defs`/`definitions` not walked in schema injection
- **LiteLLM**: Walks `$defs` and `definitions` keys in schema injection for Anthropic/Bedrock.
- **oxllm**: `inject_additional_properties_false` does not walk `$defs` or `definitions`.
- **Impact**: Schemas using `$defs` for reusable sub-schemas may get rejected.

### L5. `max_retries` forwarded in request body
- **LiteLLM**: Strips `max_retries` (client-side param, not API body).
- **oxllm**: Passes through if in JSON body.
- **Impact**: No functional breakage (OpenAI ignores unknown params).

### L6. `extra_headers` forwarded in request body
- **LiteLLM**: Handled as HTTP headers, not body params.
- **oxllm**: Passes through if in JSON body.
- **Impact**: No functional breakage but semantically incorrect.

### L7. `deployment_id`/organization not in auth
- **LiteLLM**: Resolves from params/env, passes to client.
- **oxllm**: Only sets Authorization header.
- **Impact**: Enterprise org features don't work. (Covered by H9 for org header.)

### L8. Responses API sub-endpoints missing
- **LiteLLM**: Full CRUD for /v1/responses (create, retrieve, update, delete, cancel, compact, list input items).
- **oxllm**: Core /v1/responses works. Missing cancel, compact, input items listing endpoints.

### L9. Model prefix not preserved in response
- **LiteLLM**: Prepends provider prefix to response model field.
- **oxllm**: Returns upstream model name as-is.

### L10. `content_filter` finish reason not normalized
- **LiteLLM**: Maps multiple variants (SAFETY, RECITATION, sensitive, guardrail_intervened) → `content_filter`.
- **oxllm**: Raw finish reason passes through. Only relevant for OpenAI-compatible providers.

### L11. Azure o-series prefix handling
- **LiteLLM**: Strips `o_series/` prefix, handles fake streaming.
- **oxllm**: Azure provider likely delegates to openai_compat passthrough.

### L12. `drop_params` error recovery
- **LiteLLM**: On HTTP 422, parses error, identifies unsupported params, retries without them.
- **oxllm**: Raw error propagates to client.

---

## Test Porting Priorities

### HIGH (immediately useful)
| LiteLLM Test | What It Tests |
|---|---|
| `test_gpt5_maps_max_tokens` | max_tokens → max_completion_tokens for gpt-5 |
| `test_gpt5_temperature_drop` | Temperature ≠ 1 dropped for reasoning models |
| `test_o1_handle_system_role` | system → user role mapping for o1 |
| `test_chunk_parser_raises_on_in_body_error_payload` | vLLM error-in-stream detection |
| `test_chunk_parser_maps_reasoning_to_reasoning_content` | reasoning → reasoning_content delta mapping |
| `test_sanitize_usage_obj_handles_null_tokens` | null token → 0 normalization |
| `test_transform_request_strips_for_real_openai` | cache_control stripping for openai.com |

### MEDIUM
| LiteLLM Test | What It Tests |
|---|---|
| `test_gpt5_normalizes_reasoning_effort_dict_with_summary` | Dict reasoning_effort → string |
| `test_gpt5_none_dict_with_tools_no_tool_drop` | effort:"none" dict preserves tools |
| `test_gpt5_none_dict_with_sampling_params_allowed` | effort:"none" dict allows logprobs/top_p |
| `test_gpt5_rejects_params_unsupported_by_openai` | Logit_bias/modalities/audio rejected for gpt-5 |
| `test_o1_handle_tool_calling_optional_params` | O1 tool calling param support |
| `test_chunk_parser_preserves_usage` | Usage field preserved in stream chunks |
| `test_chunk_parser_without_id_field` | Missing `id` field (MiniMax compat) |

### LOW
| LiteLLM Test | What It Tests |
|---|---|
| `test_gpt5_unknown_model_passes_through_minimal` | Unknown models pass minimal through |
| `test_gpt5_5_pro_rejects_reasoning_effort_low` | gpt-5.5-pro rejects low effort |
| `test_predicate_real_openai_no_api_base_strips` | cache_control predicate helpers |
| `test_predicate_lookalike_host_not_treated_as_openai` | Security: domain check |

---

## Summary by Category

| Category | HIGH | MEDIUM | LOW |
|----------|------|--------|-----|
| Request params | 3 (H1, H2, H8) | 4 (M1, M2, M3, M10) | 3 (L5, L6, L7) |
| Response + streaming + usage | 3 (H4, H5, H6) | 4 (M4, M5, M6, M7) | 4 (L1, L2, L3, L10) |
| Structured outputs | 1 (H7) | 1 (M11) | 1 (L4) |
| Cost calculation | 1 (H3) | — | — |
| Features (auth, endpoints) | 1 (H9) | 3 (M8, M12, M13) | 3 (L8, L9, L11) |
| **Total** | **9** | **12** | **11** |
