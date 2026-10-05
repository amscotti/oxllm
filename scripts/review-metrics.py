#!/usr/bin/env python3
"""Parse an opencode session export JSON and emit a TSV of metrics, including
an estimated USD cost computed from per-1M-token rates.

Usage: review-metrics.py <session-export.json>
Prints (tab-separated):
    tok_in  tok_out  reasoning  cache_read  cache_write  cost_est_usd  span_seconds

Cost rates (USD per 1M tokens) come from env, defaulting to DeepSeek V4 Flash
official API rates (pay-per-token, so the estimate is the actual cost):
    PRICE_INPUT   cache-miss input   default 0.14
    PRICE_CACHE   cache-hit input    default 0.0028
    PRICE_OUTPUT  generated tokens   default 0.28   (output + reasoning)
"""
import json
import os
import sys

PRICE_INPUT = float(os.environ.get("PRICE_INPUT", "0.14"))
PRICE_CACHE = float(os.environ.get("PRICE_CACHE", "0.0028"))
PRICE_OUTPUT = float(os.environ.get("PRICE_OUTPUT", "0.28"))

d = json.load(open(sys.argv[1]))
info = d.get("info", {}) or {}
t = info.get("tokens", {}) or {}
cache = t.get("cache", {}) or {}
tm = info.get("time", {}) or {}
created = tm.get("created", 0) or 0
updated = tm.get("updated", 0) or 0
span = (updated - created) // 1000 if (updated and created and updated > created) else 0

tok_in = t.get("input", 0) or 0
tok_out = t.get("output", 0) or 0
reasoning = t.get("reasoning", 0) or 0
cache_read = cache.get("read", 0) or 0
cache_write = cache.get("write", 0) or 0

cost = (
    tok_in / 1_000_000 * PRICE_INPUT
    + cache_read / 1_000_000 * PRICE_CACHE
    + (tok_out + reasoning) / 1_000_000 * PRICE_OUTPUT
)

print("\t".join(str(v) for v in (
    tok_in,
    tok_out,
    reasoning,
    cache_read,
    cache_write,
    f"{cost:.6f}",
    span,
)))
