#!/usr/bin/env python3
"""Backfill context-tier keys in oxllm price shards from a local catalog.

Reads a JSON price map, finds every model row shipped in an
oxllm shard that is missing TIER_BASE_FIELDS tier keys, and inserts them
verbatim (glued to the end of the row block). Also writes a pinned JSON
snapshot of the tier keys for the CI parity test in cost.rs.

Run: python3 scripts/backfill_tiers.py <path-to-model-prices.json>
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


def main():
    catalog_path = sys.argv[1]
    with open(catalog_path) as f:
        catalog = json.load(f)

    shard_dir = "crates/oxllm-core/data/prices"
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
            entries = std_tier_entries(catalog[model])
            if not entries:
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
            if new_lines:
                inserts.setdefault(last_content + 1, []).extend(new_lines)
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

    snap_path = os.path.join(shard_dir, "context_tiers_snapshot.json")
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"snapshot written: {len(snapshot)} models")


if __name__ == "__main__":
    main()
