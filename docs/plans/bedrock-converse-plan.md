# Plan: Bedrock Converse API Migration

> **Historical proposal.** Converse transforms and routing now exist under
> `crates/oxllm-providers/src/bedrock/`. Explicit `use_converse_api` and the
> provider's model-selection rules choose the path. The tasks/estimates below
> are retained for context, not a live backlog. **Protocol correction:** AWS
> ConverseStream uses binary eventstream framing; the host decodes it before
> producing client-facing SSE.

## Planning-time state (July 2026)

At the time of this proposal, oxllm's Bedrock provider
(`crates/oxllm-providers/src/bedrock/mod.rs`) used **InvokeModel**
(`/model/{id}/invoke`). The current provider is substantially larger and also
implements Converse:
- Anthropic models: delegates to `AnthropicProvider` + inlines `anthropic_version`
- Non-Anthropic models: passthrough of OpenAI-shaped body + vendor-specific response parsing
- Streaming: `invoke-with-response-stream` + binary eventstream decoder

LiteLLM has moved to **Converse API** (`/model/{id}/converse`) as the primary path:
- `converse_transformation.py`: 2,271 lines — standardizes ALL model families via Converse's unified request/response format
- `converse_stream`: AWS binary eventstream framing, decoded by the host

## Why Migrate
- **Unified format**: Converse standardizes request/response across all model families (Anthropic, Meta, Mistral, Cohere, Nova, AI21, etc.) — no more vendor-specific parsing
- **Features locked behind Converse**: guardrail config, native structured outputs, web search/Nova grounding, performance config, service tiers, reasoning config, cache point blocks, computer use tools
- **Streaming**: ConverseStream still requires the host's binary eventstream decoder
- **Newer models**: Many newer models (Claude 4+, Nova 2) work better via Converse

## Design Decision: Dual-Path (Not Replacement)
Keep InvokeModel as a fallback, add Converse as the primary path. Users choose via config:
```yaml
oxllm_params:
  model: bedrock/anthropic.claude-sonnet-4-6
  use_converse_api: true   # opt-in, default false initially
```
Eventually make Converse the default once stable.

## Implementation Plan (4 batches)

### Batch 1: Core Converse Transform (oxllm-core + oxllm-providers)
**Files**: `crates/oxllm-providers/src/bedrock/converse.rs` (new), `bedrock/mod.rs`

1. Create `bedrock/converse.rs` with `BedrockConverseConfig`:
   - `transform_request`: OpenAI request → Bedrock Converse request
     - Messages → `messages[]` content blocks (text, image, tool_use, tool_result)
     - System messages → `system[]` content blocks
     - `max_tokens`/`temperature`/`top_p`/`stop` → `inferenceConfig`
     - Tools → Converse `tools[]` format (function specs, not Anthropic-native)
     - `tool_choice` → Converse `toolConfig.toolChoice`
     - `response_format` → native `outputConfig` or synthetic tool
   - `transform_response`: Converse response → OpenAI response
     - `output.message.content[]` → `choices[0].message`
     - Content blocks: text, tool_use, thinking, citations
     - `usage` → OpenAI usage + thinking tokens
     - `stopReason` → OpenAI `finish_reason`
   - `transform_stream_chunk`: decoded ConverseStream event → OpenAI chunk
     - Handle: `messageStart`, `contentBlockDelta`, `contentBlockStop`, `messageStop`, `metadata`

2. Tests: golden round-trip for each model family (Anthropic, Meta, Mistral, Nova)

**Effort**: ~800 lines + 200 lines tests

### Batch 2: Wire into BedrockProvider (oxllm-providers)
**Files**: `bedrock/mod.rs`

1. Add `use_converse_api` field to `OxllmParams` (config-driven opt-in)
2. In `endpoint_url`: if Converse → `/model/{id}/converse`; if streaming → `/converseStream`
3. In `transform_request`: if Converse → delegate to `converse.rs`; else existing InvokeModel path
4. In `transform_response`: same conditional
5. In `transform_stream_chunk`: both ConverseStream and InvokeModel use AWS binary eventstream framing; the host decodes it into client-facing SSE

**Effort**: ~200 lines

### Batch 3: Converse Features (incremental, can be separate batches)
Each feature is independent — add as needed:
- Guardrail config (`guardrailConfig` in Converse request body)
- Native structured outputs (`outputConfig`)
- Performance config (`performanceConfig`)
- Service tier (`serviceTier`)
- Reasoning effort mapping (`reasoningConfig` for Nova 2)
- Cache point blocks (`cachePoint` for prompt caching)
- Web search / Nova grounding (`systemTool`)
- Computer use tools

**Effort**: ~100-200 lines per feature

### Batch 4: Make Converse Default
After testing with live Bedrock:
1. Flip `use_converse_api` default to `true`
2. Keep InvokeModel as fallback for models that don't support Converse
3. Update docs

**Effort**: config change + docs

## Key Design Principles
- Converse transform is **pure** (no I/O) — lives in provider crate
- Auth is unchanged (SigV4 signing is endpoint-agnostic)
- Eventstream decoder stays for InvokeModel fallback
- No changes to oxllm-core types needed (Converse is a provider-internal transform)
- Pricing lookup unchanged (same model IDs)

## Risk Mitigation
- Dual-path means zero risk to existing InvokeModel users
- Converse opt-in via config means users test at their own pace
- Golden tests against provider/AWS wire examples ensure transform correctness
- The live Bedrock suites in `tests/bedrock_live.rs` cover real accounts when
  AWS credentials are available
