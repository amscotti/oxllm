# Plan: Bedrock Realtime API (Nova Sonic)

> **Historical proposal.** The server now has WebSocket infrastructure,
> `/v1/realtime`, core realtime contracts and Bedrock realtime code. The old
> "Current oxllm State" and effort estimates below describe planning-time
> assumptions, not the current tree. Protocol/provider readiness still requires
> current endpoint tests and an appropriately configured live deployment.
>
> **Current transport:** clients still use `/v1/realtime` WebSockets. Native
> Bedrock traffic uses the SDK's signed HTTP/2
> `InvokeModelWithBidirectionalStream` POST, including event signatures. Only
> explicitly configured WS(S) bridge endpoints use WebSocket upstream transport.
> Router feedback, bounded drain, and cancellation-safe settlement are implemented.

## What It Is
Bedrock enables **bidirectional speech-to-speech** streaming through its native
HTTP/2 eventstream operation, powered by Amazon Nova Sonic. The gateway exposes
OpenAI-compatible WebSocket events to clients, translating audio, tool calls, and
text transcripts at the host/provider boundary.

LiteLLM implementation: `litellm/llms/bedrock/realtime/` — 1,540 lines (handler + transformation).

## Current oxllm State
- **No WebSocket support** in the server
- **No RealtimeProvider trait** in core
- **No `/v1/realtime` route** in the server
- oxllm has a `/v1/realtime` passthrough route that proxies OpenAI's WebSocket realtime API, but this is a dumb proxy — no transform layer

## Architecture Requirements

### 1. WebSocket Infrastructure (oxllm-server)
The server needs a WebSocket proxy layer:
- Accept WebSocket upgrade from client (`/v1/realtime?model=...`)
- Open a WebSocket to OpenAI-compatible upstreams; native Bedrock requires a
  signed bidirectional HTTP/2 POST and a streaming event signer.
- Bidirectionally proxy frames with optional transform

**Effort**: ~500 lines — new `crates/oxllm-server/src/realtime/` module

### 2. RealtimeProvider Trait (oxllm-core)
```rust
pub trait RealtimeProvider: Send + Sync {
    fn transform_session_update(&self, event: &Value) -> Vec<String>;
    fn transform_input_audio(&self, event: &Value) -> Vec<String>;
    fn transform_realtime_response(&self, event: &str) -> Vec<Value>;
    // ... bidirectional event transforms
}
```

**Effort**: ~200 lines — trait + types in `oxllm-core/src/provider/realtime.rs`

### 3. Bedrock Nova Sonic Transform (oxllm-providers)
`crates/oxllm-providers/src/bedrock/realtime.rs` — converts between:
- **OpenAI Realtime API events** (`session.update`, `input_audio_buffer.append`, `conversation.item.create`, `response.create`)
- **Bedrock Nova Sonic events** (`sessionStart`, `contentStart`, `audioOutput`, `textOutput`, `contentEnd`, `sessionEnd`, `toolUse`)

Key transforms:
- Audio format: OpenAI PCM16 ↔ Bedrock PCM with sample rate mapping
- Session setup: tools, voice, turn detection settings
- Tool calls: bidirectional tool_use/tool_result mapping
- Audio output: chunked PCM with trigger audio

**Effort**: ~1,200 lines — following LiteLLM's `transformation.py` closely

### 4. OpenAI Realtime Transform (identity passthrough)
OpenAI's native realtime API is the wire format we expose, so the OpenAI transform is identity (no-op). This is the reference implementation.

**Effort**: ~50 lines

### 5. Route Registration
```rust
// crates/oxllm-server/src/routes/mod.rs
.route("/v1/realtime", get(realtime::handle_websocket_upgrade))
```

**Effort**: ~50 lines

## Implementation Plan (3 batches)

### Batch 1: WebSocket Proxy Infrastructure
1. Add `tokio-tungstenite` WebSocket support to oxllm-server
2. Create `realtime/` module with WebSocket upgrade handler
3. Implement bidirectional proxy: client WS ↔ upstream WS
4. Auth: extract API key from query param (`?authorization=...`) — existing pattern
5. Test: echo server + basic connection lifecycle

### Batch 2: Bedrock Nova Sonic Transform
1. Create `bedrock/realtime.rs` with all event transforms
2. Wire into the WebSocket proxy: Bedrock models route through the transform; OpenAI models passthrough
3. Handle: session setup, audio I/O, text I/O, tool calls, session close
4. Tests: golden event transform vectors

### Batch 3: Integration
1. Route registration with model resolution
2. Provider detection: `bedrock/*` → Bedrock transform; `openai/*` → passthrough
3. Live test with Bedrock Nova Sonic (requires WebSocket-capable client)
4. Docs

## Key Design Principles
- WebSocket proxy is a **server concern**, not core — core only defines the transform trait
- The transform is pure (input event JSON → output event JSON strings)
- Multiple providers can implement realtime (OpenAI, Bedrock, future)
- The client always speaks OpenAI Realtime API format; providers transform internally

## Prerequisites
- `tokio-tungstenite` is already a workspace dependency (used for existing realtime passthrough)
- Auth for WebSocket: existing `?authorization=` query param pattern on `/v1/realtime`

## Risk Assessment
- **Medium-high effort**: WebSocket proxying with transforms is complex
- **Niche use case**: speech-to-speech is not a common proxy gateway use case
- **Can start with Batch 1** (infrastructure) since it benefits any realtime provider
- **Bedrock-specific transform (Batch 2)** can wait until there's user demand
