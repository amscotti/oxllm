"""
Minimal OpenAI-compatible mock upstream for multi-instance stress testing.

Why this exists: sustained load tests must NOT hit a real provider API
(rate limits, cost, latency variance all poison the results). This server
is a deterministic, fast stand-in that speaks the OpenAI Chat Completions,
Embeddings, and Models shapes oxllm forwards to OpenAI-compatible upstreams.

Design goals:
- Fast (aiohttp, async, no DB) so the upstream is never the bottleneck.
- Deterministic: identical input -> identical output (cache-hit assertions
  depend on it). Content is a stable hash of the prompt.
- Streaming: emits N SSE chunks then [DONE], OpenAI shape.
- Embeddings: deterministic vector derived from the text, so identical or
  near-identical prompts produce identical/similar vectors (semantic-cache
  plumbing can be validated). Dimension defaults to 64.
- Fault injection: ?fail_after=N drops the Nth request; ?latency_ms=...
  overrides the env default per-call; ?status=503 forces a status.

Not a security boundary — listen only on the docker-internal port.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from aiohttp import web

DEFAULT_LATENCY_MS = int(os.environ.get("LATENCY_MS", "120"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "64"))
PORT = int(os.environ.get("PORT", "9000"))
# Incrementing request counter for deterministic ids and fault injection.
_req_counter = 0


def _next_id() -> int:
    global _req_counter
    _req_counter += 1
    return _req_counter


def _stable_text(messages) -> str:
    """Flatten an OpenAI messages array into one comparable string."""
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                parts.append(f"{role}:{content}")
        return "\n".join(parts)
    return str(messages)


def _content_for(prompt: str, model: str) -> str:
    """Deterministic completion so cache-hit assertions are exact."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    tok = prompt.strip().split()[:6]
    echo = " ".join(tok) if tok else "(empty)"
    return f"[mock:{model}] {echo} …({h})"


def _embed_vec(text: str) -> list[float]:
    """
    Deterministic embedding: a bag-of-tokens hashed into EMBED_DIM buckets.
    Identical text -> identical vector (cosine 1.0). Text sharing tokens ->
    correlated vectors (cosine > 0). Enough to exercise semantic-cache plumbing
    without a real embedding model.
    """
    vec = [0.0] * EMBED_DIM
    for tok in text.lower().split():
        b = hashlib.sha256(tok.encode("utf-8")).digest()
        for i in range(0, min(8, EMBED_DIM)):
            vec[b[i] % EMBED_DIM] += 1.0
    # L2-normalise so cosine similarity == dot product.
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [round(v / norm, 8) for v in vec]


async def _maybe_sleep(request):
    lat = request.query.get("latency_ms")
    ms = int(lat) if lat else DEFAULT_LATENCY_MS
    if ms > 0:
        await asyncio.sleep(ms / 1000.0)


def _maybe_inject_fault(request):
    """Return an aiohttp Response to abort with, or None to proceed."""
    force = request.query.get("status")
    if force:
        try:
            code = int(force)
            return web.Response(
                status=code,
                body=f'{{"error":{{"message":"forced {code}","type":"forced"}}}}',
                content_type="application/json",
            )
        except ValueError:
            pass
    after = request.query.get("fail_after")
    if after:
        try:
            n = int(after)
            if _req_counter >= n:
                return web.Response(
                    status=503,
                    body='{"error":{"message":"fail_after reached","type":"upstream"}}',
                    content_type="application/json",
                )
        except ValueError:
            pass
    return None


async def chat_completions(request: web.Request) -> web.StreamResponse:
    await _maybe_sleep(request)
    fault = _maybe_inject_fault(request)
    if fault is not None:
        return fault
    body = await request.json()
    model = body.get("model", "mock-model")
    messages = body.get("messages", [])
    prompt = _stable_text(messages)
    stream = body.get("stream", False)
    rid = f"chatcmpl-mock-{_next_id()}"
    created = int(time.time())

    if stream:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        first = {
            "id": rid, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        await resp.write(f"data: {_json(first)}\n\n".encode())
        content = _content_for(prompt, model)
        # Emit content in 4 chunks so streaming integrity is observable.
        step = max(1, len(content) // 4)
        for i in range(0, len(content), step):
            chunk = {
                "id": rid, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content[i:i + step]}, "finish_reason": None}],
            }
            await resp.write(f"data: {_json(chunk)}\n\n".encode())
            await asyncio.sleep(0)  # yield to event loop
        done_chunk = {
            "id": rid, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": max(1, len(prompt.split())),
                "completion_tokens": max(1, len(content.split())),
                "total_tokens": max(2, len(prompt.split()) + len(content.split())),
            },
        }
        await resp.write(f"data: {_json(done_chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    content = _content_for(prompt, model)
    ptoks = max(1, len(prompt.split()))
    ctoks = max(1, len(content.split()))
    return web.json_response({
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "refusal": None},
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": ptoks,
            "completion_tokens": ctoks,
            "total_tokens": ptoks + ctoks,
        },
    })


async def embeddings(request: web.Request) -> web.Response:
    await _maybe_sleep(request)
    fault = _maybe_inject_fault(request)
    if fault is not None:
        return fault
    body = await request.json()
    model = body.get("model", "mock-embed")
    inp = body.get("input", "")
    if isinstance(inp, list):
        inputs = inp
    else:
        inputs = [inp]
    data = []
    for i, text in enumerate(inputs):
        data.append({"object": "embedding", "index": i, "embedding": _embed_vec(str(text))})
    return web.json_response({
        "object": "list",
        "model": model,
        "data": data,
        "usage": {"prompt_tokens": max(1, len(str(inputs)) // 4), "total_tokens": max(1, len(str(inputs)) // 4)},
    })


async def models(request: web.Request) -> web.Response:
    return web.json_response({
        "object": "list",
        "data": [
            {"id": "mock-llama-3", "object": "model", "owned_by": "mock"},
            {"id": "mock-embed", "object": "model", "owned_by": "mock"},
        ],
    })


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "requests": _req_counter})


def _json(obj) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"))


async def metrics(request: web.Request) -> web.Response:
    """Prometheus exposition endpoint (text/plain).

    The compose stack's Prometheus scrapes this job; /health is JSON and
    cannot serve as a scrape target ("unsupported Content-Type").
    """
    body = (
        "# HELP mock_llm_requests_total Total requests handled.\n"
        "# TYPE mock_llm_requests_total counter\n"
        f"mock_llm_requests_total {_req_counter}\n"
    )
    return web.Response(text=body, content_type="text/plain")


if __name__ == "__main__":
    app = web.Application(client_max_size=1_048_576)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/chat/completions", chat_completions)
    app.router.add_post("/v1/embeddings", embeddings)
    app.router.add_post("/embeddings", embeddings)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/models", models)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    print(f"mock-llm listening on :{PORT} (latency={DEFAULT_LATENCY_MS}ms, embed_dim={EMBED_DIM})", flush=True)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
