# Bedrock Provider Gap Analysis — oxllm vs LiteLLM

Generated from 4 parallel agents scanning LiteLLM Bedrock provider code, test files,
and comparing against oxllm's Bedrock provider implementation.

---

## HIGH Severity Gaps

### H1. Non-Anthropic InvokeModel request transforms are passthrough (broken)
- **LiteLLM**: Each vendor has dedicated config class with `transform_request` that converts OpenAI `messages` to vendor-native prompt format and maps params to vendor-specific keys.
- **oxllm**: Non-Anthropic branch copies the entire OpenAI-shaped body verbatim. Bedrock Invoke does NOT accept `messages` for non-Anthropic models.
- **Impact**: Every non-Anthropic model (Titan, Llama, Mistral, AI21, Cohere, DeepSeek, Qwen2/3) on the Invoke path gets 400.

### H2. Titan request params not wrapped in `textGenerationConfig`
- **LiteLLM**: Wraps params in `{"inputText": prompt, "textGenerationConfig": {maxTokenCount, stopSequences, temperature, topP}}`.
- **oxllm**: Passthrough — no wrapping. Sends `messages`, `max_tokens`, `temperature` as top-level keys.
- **Impact**: Titan InvokeModel will 400 or silently ignore params.

### H3. Missing `max_gen_len` for Llama (different key name)
- **LiteLLM**: Maps `max_tokens` → `max_gen_len`.
- **oxllm**: Passthrough — sends `max_tokens` which Llama rejects.
- **Impact**: Llama InvokeModel ignores `max_tokens` — defaults to small output.

### H4. AI21 response shape `completions[0].data.text` not handled
- **LiteLLM**: Reads `completion_response["completions"][0]["data"]["text"]`.
- **oxllm**: `extract_vendor_text` does NOT handle `completions[].data.text`.
- **Impact**: AI21 responses return raw vendor JSON instead of standard OpenAI shape.

### H5. DeepSeek R1 `<think>` tag reasoning extraction missing
- **LiteLLM**: Extracts `<think>` reasoning from response, splits into `reasoning_content` and `content`. Streaming handles `<think>`/`</think>` delimiters.
- **oxllm**: Passthrough — returns full `generation` text including `<think>` tags.
- **Impact**: DeepSeek R1 on Bedrock Invoke leaks `<think>` tags, never separates reasoning from content.

### H6. Qwen2/Qwen3 custom prompt format and response stripping missing
- **LiteLLM**: Converts messages to Qwen `|<im_start|>role\ncontent<|im_end|>` format. Strips response tokens.
- **oxllm**: Passthrough — sends raw OpenAI `messages`.
- **Impact**: Qwen2/3 on Bedrock Invoke fails entirely.

### H7. AI21 param names wrong (different casing)
- **LiteLLM**: Maps `max_tokens` → `maxTokens`, `top_p` → `topP`. Also maps `frequencyPenalty`, `presencePenalty`, `countPenalty`.
- **oxllm**: Passthrough — sends `max_tokens` (snake_case) which AI21 doesn't recognise.
- **Impact**: AI21 ignores `max_tokens` and penalty params.

### H8. Missing `outputConfig.textFormat` structured output shape (Converse)
- **LiteLLM**: Produces `outputConfig: {textFormat: {type, structure: {jsonSchema: {schema: "<json-string>"}}}}`.
- **oxllm**: Produces `outputConfig: {format: {type: "json"}, schema: <object>}`. Wrong shape.
- **Impact**: Structured outputs on Bedrock Converse either fail 400 or produce unstructured output.

### H9. Thinking budget not clamped to minimum 1024 (Converse)
- **LiteLLM**: Clamps `thinking.budget_tokens` to `BEDROCK_MIN_THINKING_BUDGET_TOKENS` (1024).
- **oxllm**: No minimum validation. A caller sending `budget_tokens: 512` gets 400.
- **Impact**: Any thinking request with `budget_tokens < 1024` fails with 400.

### H10. No `$defs`/`definitions` handling in schema normalization
- **LiteLLM**: Recurses into `$defs` and `definitions` keys for `additionalProperties: false` injection.
- **oxllm**: `inject_additional_properties_false` does NOT walk `$defs` or `definitions`.
- **Impact**: Schemas using `$defs` for shared type definitions may be rejected.

### H11. Application inference profiles not auto-routed to Converse
- **LiteLLM**: Detects `:application-inference-profile/` in model ARN, forces Converse route.
- **oxllm**: No detection. May route through Invoke or fail entirely.
- **Impact**: Application inference profiles may not work.

### H12. Throughput/context-window suffixes not stripped from model names
- **LiteLLM**: Strips `:51k`, `[1m]` patterns from model names.
- **oxllm**: No suffix stripping.
- **Impact**: URLs like `/model/anthropic.claude-3-5-sonnet-v1:0:51k/converse` may 404.

### H13. No per-vendor provider detection for InvokeModel
- **LiteLLM**: Resolves provider from model name prefix (`amazon.`, `meta.`, `mistral.`, `ai21.`, `cohere.`, etc.).
- **oxllm**: Only checks `starts_with("anthropic.")`.
- **Impact**: Can't route non-Anthropic models to vendor-specific transforms even if added.

### H14. `completion_tokens_details` entirely missing from Converse response
- **LiteLLM**: Builds `completion_tokens_details` with `reasoning_tokens` and `text_tokens`.
- **oxllm**: No `completion_tokens_details` in Converse response usage at all.
- **Impact**: Missing reasoning_tokens for Bedrock reasoning models.

### H15. Document (PDF) and video content types not supported (Converse)
- **LiteLLM**: Supports document types (pdf, csv, doc, xls) and video types (mp4, mov, mkv).
- **oxllm**: Only handles `text`, `guarded_text`, and `image_url`.
- **Impact**: PDF/video content blocks silently dropped.

### H16. Cohere request format broken on Invoke
- **LiteLLM**: Sends Cohere as `{"message": prompt, "chat_history": [...]}`.
- **oxllm**: Passthrough — sends `messages` array.
- **Impact**: Cohere on Invoke returns 400.

---

## MEDIUM Severity Gaps

### M1. Computer-use beta headers not injected (Converse)
- **LiteLLM**: Adds `computer-use-*` beta header based on model version.
- **oxllm**: `filter_converse_beta_headers` only STRIPS headers, never adds them.
- **Impact**: Computer use tools on Bedrock Converse don't work.

### M2. Missing `parallel_tool_calls` → `disable_parallel_tool_use` (Converse)
- **LiteLLM**: Maps `parallel_tool_calls: false` → `tool_choice.disable_parallel_tool_use: true`.
- **oxllm**: No handling.
- **Impact**: Clients can't disable parallel tool execution.

### M3. Context management filtering for Bedrock Converse
- **LiteLLM**: Keeps only compact edit types, adds `compact-2026-01-12` beta header.
- **oxllm**: Passes `context_management` verbatim. No beta header injection.
- **Impact**: Non-compact edits fail; compact edits miss required beta header.

### M4. Nova grounding citations not surfaced in response
- **LiteLLM**: Extracts `citationsContent` blocks, converts to URL citation annotations.
- **oxllm**: `citationsContent` blocks silently ignored.
- **Impact**: Nova grounding citations (web search sources) lost.

### M5. `guardrailIntervened` stop reason not mapped (Converse)
- **LiteLLM**: Maps `guardrail_intervened` → `content_filter`.
- **oxllm**: No entry for `guardrailIntervened` — falls through to `"stop"`.
- **Impact**: Clients see `finish_reason: "stop"` instead of `"content_filter"` when guardrail intervenes.

### M6. Service tier not surfaced from Converse response
- **LiteLLM**: Maps `serviceTier.type` → `service_tier` string.
- **oxllm**: Never reads `serviceTier` from Converse response.
- **Impact**: Clients can't determine which service tier handled the request.

### M7. Guardrail `trace` not surfaced from Converse response
- **LiteLLM**: Attaches `trace` from Converse response to `model_response.trace`.
- **oxllm**: Never reads `trace` from response.
- **Impact**: Guardrail assessment results lost.

### M8. Thinking budget auto-calculation when `max_tokens` absent
- **LiteLLM**: Sets `maxTokens = thinking.budget_tokens + DEFAULT_MAX_TOKENS` when thinking enabled but no `max_tokens`.
- **oxllm**: No auto-calculation.
- **Impact**: Request may fail when `reasoning_effort` set without `max_tokens`.

### M9. `requestMetadata` value validation missing
- **LiteLLM**: Validates key/value length (1-256), character patterns, max 16 items.
- **oxllm**: Passes through verbatim with no validation.
- **Impact**: Bad metadata causes late 400 from Bedrock.

### M10. Eventstream decoder ignores headers/errors
- **LiteLLM**: Parses typed headers, detects error events (non-200 status codes), verifies CRCs.
- **oxllm**: Skips eventstream headers. Non-200 events not distinguished.
- **Impact**: Mid-stream errors (throttling, guardrail) silently dropped.

### M11. Fake streaming for AI21 not implemented
- **LiteLLM**: Collects full response, wraps in single-chunk SSE for AI21 (non-streaming vendor).
- **oxllm**: No fake streaming concept.
- **Impact**: AI21 streaming requests hang or 404.

### M12. `bedrock_converse_models` routing missing
- **LiteLLM**: Maintains list of models that MUST use Converse (DeepSeek, Qwen3, Jamba, Llama3).
- **oxllm**: Only routes Converse when `use_converse_api: true` explicitly set.
- **Impact**: Models like `deepseek.v3` hit broken Invoke path.

### M13. Output config effort beta header not injected (Converse)
- **LiteLLM**: Adds `ANTHROPIC_EFFORT_BETA_HEADER` when `output_config.effort` set on non-adaptive model.
- **oxllm**: Normalizes effort values but never adds beta header.
- **Impact**: `output_config.effort` on pre-4.6 models may 400.

### M14. Citations in streaming not handled
- **LiteLLM**: Handles `citationsContent` in streaming deltas, stores in `provider_specific_fields`.
- **oxllm**: No handling for `citationsContent` in streaming deltas.
- **Impact**: Streaming Nova grounding citations lost.

### M15. Streaming usage lacks `completion_tokens_details`
- **LiteLLM**: Streaming metadata produces full token details including `reasoning_tokens`.
- **oxllm**: `handle_metadata` produces basic usage without `completion_tokens_details`.
- **Impact**: Reasoning token costs invisible in streaming.

---

## LOW Severity Gaps

| # | Gap | Notes |
|---|-----|-------|
| L1 | `support_strict_tools` flag not forwarded | `toolSpec.strict` from function definition |
| L2 | `extra_headers` / `stream_options` not supported | Can be forwarded |
| L3 | Cross-region prefix handling in model checks | Strip `us.`/`eu.`/`apac.` before model-family detection |
| L4 | `max_completion_tokens` not mapped to `maxTokens` | OpenAI o-series compat |
| L5 | `trace` in streaming not surfaced | Guardrail trace in stream |
| L6 | Streaming `id` not from `conversationId` | Consistent SSE `id` field |
| L7 | Signature-only delta missing `thinking_blocks` | Empty thinking block with signature |
| L8 | `cache_creation_input_tokens` field name wrong | Should be `cache_creation_input_tokens` not `cache_creation_tokens` |
| L9 | `prompt_tokens_details.text_tokens` missing | Not standard but some clients expect it |
| L10 | `guardrailIntervened` already covered by H5 | Same as guard railIntervened |
| L11 | Streaming `serviceTier`/`trace` | Same as M6/M7 for streaming |
| L12 | Dummy tool injection | Edge case for tool-use history without tools param |
| L13 | Tool search types filtering | Niche feature |

---

## Missing Provider Features

| Feature | LiteLLM Lines | oxllm Status | Priority |
|---------|--------------|--------------|----------|
| **Embeddings** (Titan, Cohere, Nova, TwelveLabs) | 7 vendor transform files | ❌ `supports(Embeddings)` returns false | HIGH |
| **Image Generation** (Nova Canvas, Stability, Titan) | 6 files + handler | ❌ Not implemented | HIGH |
| **Batches** (S3 model-invocation jobs) | handler + transformation | ❌ Not implemented | MEDIUM |
| **Rerank** (Cohere, Amazon) | handler + transformation | ❌ Not implemented | MEDIUM |
| **Audio Transcription** (Nova Sonic) | handler + transformation | ❌ Not implemented | LOW |
| **Files** (S3-based) | handler + integration | ❌ Not implemented | LOW |
| **Count Tokens** | transformation + token counter | ❌ Not implemented | LOW |
| **Agents** (InvokeAgent) | transformation | ❌ Not implemented | LOW |
| **AgentCore** (A2A protocol) | transformation | ❌ Not implemented | LOW |
| **Claude Platform Provider** | dedicated provider | ❌ Not implemented | LOW |
| **Mantle Endpoint** | handler + transformation | ❌ Not implemented | LOW |
| **Guardrail Translation** | guardrail_translation module | ❌ Not implemented | LOW |

---

## Summary by Category

| Category | HIGH | MEDIUM | LOW |
|----------|------|--------|-----|
| InvokeModel vendor transforms | 9 (H1-H7, H13, H16) | 1 (M12) | — |
| Converse request | 4 (H8-H12) | 4 (M1-M3, M9) | 5 (L1-L5) |
| Converse response + streaming | 1 (H14) | 7 (M4-M8, M10, M14, M15) | 5 (L6-L10) |
| Missing providers | 2 (embeddings, image gen) | 2 (batches, rerank) | 6 |
| **Total** | **16** | **15** | **13** |

---

## Test Porting Priorities

| LiteLLM Test | What It Tests | Priority |
|---|---|---|
| `test_base_invoke_transformation.py` | `stream_chunk_size` stripped, guardrail headers, malformed configs | HIGH |
| `test_converse_transformation.py` | reasoning token split in usage, citations, trace/serviceTier | HIGH |
| `test_bedrock_common_utils.py` | inference profile routing, throughput suffix stripping, cross-region | MEDIUM |
| `test_streaming_choice_index.py` | Choice index ≠ contentBlockIndex | MEDIUM |
| `test_service_tier.py` | ServiceTier block passthrough | MEDIUM |
| `test_amazon_qwen2_transformation.py` | Qwen2 custom format (8 tests) | MEDIUM |
