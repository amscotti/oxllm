#!/usr/bin/env python3
"""Backfill context-tier and service-tier keys in oxllm price shards.

Reads a JSON price map, finds every model row shipped in an
oxllm shard that is missing TIER_BASE_FIELDS tier keys, and inserts them
verbatim (glued to the end of the row block). Also writes a pinned JSON
snapshot of the tier keys for the CI parity test in cost.rs.

With --service-only, backfills just the service-tier suffixed keys
(`_batches`, `_flex`, `_priority`, ...) that cost/service_tiers.rs
consumes, and leaves context tiers and the snapshot untouched. A key is
inserted only when the row's base input rate matches the catalog (so a
renamed/repriced row is never silently tiered from a different model),
when the key normalizes to a consumed base field (mirroring
`price_field` in service_tiers.rs), and when the curated
service_tiers.toml overlay does not already price that (tier, field).
For gemini/vertex_ai models already flex-priced (shard `_flex` keys or
the overlay), `_batches` keys are skipped as a whole tier: batch
resolution falls back to flex there (Google publishes identical
Flex/Batch prices, and the curated flex entries carry long-context
rates the catalog omits), while a flat catalog batches entry would
shadow that fallback and lose the long-context buckets. Google models
with no flex coverage get the complete batches set.

Run: python3 scripts/backfill_tiers.py [--service-only] <path-to-model-prices.json>
"""

import glob
import json
import os
import sys
import tomllib

TIER_BASE_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
)
SERVICE_SUFFIXES = ("_priority", "_flex", "_fast", "_ultrafast", "_balanced", "_batches")
SERVICE_TIERS = tuple(s[1:] for s in SERVICE_SUFFIXES)
# Bases service_tiers.rs `price_field` consumes (TIER_BASE_FIELDS + these two).
SERVICE_BASE_FIELDS = TIER_BASE_FIELDS + (
    "input_cost_per_audio_token",
    "output_cost_per_reasoning_token",
)


def fmt(v: float) -> str:
    mant, exp = f"{float(v):.6e}".split("e")
    sign = ""
    if exp.startswith(("+", "-")):
        sign, exp = exp[0], exp[1:]
    exp = exp.lstrip("0") or "0"
    return f"{mant}e{sign}{exp}"


def parse_threshold(key: str):
    rest = key.rsplit("_above_", 1)[1]
    if not rest.endswith("_tokens"):
        return None
    rest = rest[: -len("_tokens")]
    scale = 1000 if rest.endswith("k") else 1
    num = rest[:-1] if rest.endswith("k") else rest
    try:
        return int(num) * scale
    except ValueError:
        return None


def std_tier_entries(v):
    out = []
    for k, val in v.items():
        if "_above_" not in k or k.endswith(SERVICE_SUFFIXES):
            continue
        base = k.rsplit("_above_", 1)[0]
        if base not in TIER_BASE_FIELDS:
            continue
        t = parse_threshold(k)
        if t is None:
            continue
        out.append((k, t, val))
    out.sort(key=lambda x: (-x[1], TIER_BASE_FIELDS.index(x[0].rsplit("_above_", 1)[0])))
    return out


def normalized_price_key(key, tier):
    """Mirror service_tiers.rs: strip `_{tier}` suffix, or rewrite
    `_{tier}_above_` to `_above_`; valid only for consumed base fields
    (including their `_above_<N>k_tokens` threshold forms)."""
    suffix = f"_{tier}"
    if key.endswith(suffix):
        normalized = key[: -len(suffix)]
    elif f"{suffix}_above_" in key:
        normalized = key.replace(f"{suffix}_above_", "_above_")
    else:
        return None
    if normalized in SERVICE_BASE_FIELDS:
        return normalized
    if "_above_" in normalized:
        base = normalized.rsplit("_above_", 1)[0]
        if base in SERVICE_BASE_FIELDS and parse_threshold(normalized) is not None:
            return normalized
    return None


def overlay_coverage(shard_dir):
    """(model, tier, normalized field) priced by service_tiers.toml."""
    covered = set()
    path = os.path.join(shard_dir, "service_tiers.toml")
    with open(path, "rb") as f:
        overlay = tomllib.load(f)
    for name, vals in overlay.items():
        names = [name, *vals.get("aliases", [])]
        for tier in SERVICE_TIERS:
            body = vals.get(tier)
            if isinstance(body, dict):
                for n in names:
                    for field in body:
                        covered.add((n, tier, field))
    return covered


def service_tier_entries(slug, model, catalog_row, shard_row, covered):
    """Catalog service-tier keys to insert for one shard row, as
    (key, value) sorted by key. Empty unless base input rates agree."""
    shard_in = shard_row.get("input_cost_per_token")
    cat_in = catalog_row.get("input_cost_per_token")
    if not isinstance(shard_in, (int, float)) or not isinstance(cat_in, (int, float)):
        return []
    if abs(float(shard_in) - float(cat_in)) > 1e-15:
        return []
    google_flex_priced = slug in {"gemini", "vertex_ai"} and (
        any(normalized_price_key(k, "flex") == "input_cost_per_token" for k in shard_row)
        or (model, "flex", "input_cost_per_token") in covered
    )
    out = []
    for key, val in catalog_row.items():
        if not isinstance(val, (int, float)) or key in shard_row:
            continue
        hit = None
        for tier in SERVICE_TIERS:
            normalized = normalized_price_key(key, tier)
            if normalized is not None:
                hit = (tier, normalized)
                break
        if hit is None:
            continue
        tier, normalized = hit
        if (model, tier, normalized) in covered:
            continue
        if tier == "batches" and google_flex_priced:
            continue
        out.append((key, val))
    out.sort()
    return out


def main():
    args = [a for a in sys.argv[1:] if a != "--service-only"]
    service_only = len(args) != len(sys.argv[1:])
    catalog_path = args[0]
    with open(catalog_path) as f:
        catalog = json.load(f)

    shard_dir = "crates/oxllm-core/data/prices"
    covered = overlay_coverage(shard_dir)
    snapshot = {}

    for path in sorted(glob.glob(os.path.join(shard_dir, "*.toml"))):
        slug = os.path.basename(path)[:-5]
        if slug in {"cost_tiers_test", "service_tiers"}:
            continue
        with open(path) as f:
            raw = f.read()
        table = tomllib.loads(raw)
        lines = raw.split("\n")
        inserts = {}  # line index (after last content line) -> [new lines]
        for model in table:
            if model not in catalog:
                continue
            entries = [] if service_only else std_tier_entries(catalog[model])
            svc_entries = service_tier_entries(
                slug, model, catalog[model], table[model], covered
            )
            if not entries and not svc_entries:
                continue
            existing = set(k for k in table[model] if "_above_" in k)
            to_add = [(k, t, v) for (k, t, v) in entries if k not in existing]
            start = None
            for i, ln in enumerate(lines):
                s = ln.strip()
                if s == f'["{model}"]' or s == f"[{model}]":
                    start = i
                    break
            if start is None:
                print(f"  !! header not found for {slug}:{model}")
                continue
            end = len(lines)
            for i in range(start + 1, len(lines)):
                s = lines[i].strip()
                if s.startswith("[") and s.endswith("]") and not s.startswith("#["):
                    end = i
                    break
            last_content = start
            for i in range(start + 1, end):
                if lines[i].strip():
                    last_content = i
            new_lines = [f"{k} = {fmt(v)}" for (k, t, v) in to_add]
            new_lines += [f"{k} = {fmt(v)}" for (k, v) in svc_entries]
            if new_lines:
                inserts.setdefault(last_content + 1, []).extend(new_lines)
            if service_only:
                continue
            snapshot.setdefault(model, {})
            for k, t, v in entries:
                snapshot[model][k] = v

        if not inserts:
            continue
        out = []
        for i, ln in enumerate(lines):
            if i in inserts:
                for extra in inserts[i]:
                    out.append(extra)
            out.append(ln)
        if len(lines) in inserts:
            while out and out[-1].strip() == "":
                out.pop()
            out.extend(inserts[len(lines)])
        text = "\n".join(out).rstrip("\n") + "\n"
        with open(path, "w") as f:
            f.write(text)
        print(f"patched {slug}: {sum(len(v) for v in inserts.values())} tier lines")

    if service_only:
        print("service-only run: snapshot untouched")
        return
    snap_path = os.path.join(shard_dir, "context_tiers_snapshot.json")
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"snapshot written: {len(snapshot)} models")


if __name__ == "__main__":
    main()
