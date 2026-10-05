"""
Scenario driver for the multi-instance stress invariants (plan §3/§4.2).

oha covers raw throughput; this script covers the functional invariants that
need request-level control: cache sharing across instances, fleet rate limits,
session-affinity determinism, spend parity, and unary/streaming tag attribution.

Usage (with the stack already `up`):

    python tests/multi-instance/scenarios.py --base http://127.0.0.1:8080
    python tests/multi-instance/scenarios.py \
        --a http://127.0.0.1:4000 --b http://127.0.0.1:4001 --c http://127.0.0.1:4002

Each invariant reports PASS/FAIL/SKIP. Failures and skipped invariants produce
a non-zero exit code; --allow-skips explicitly permits missing optional checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import uuid
from typing import Optional

import httpx

MASTER = "sk-stress-master-key"
HDRS = {"Authorization": f"Bearer {MASTER}", "Content-Type": "application/json"}


def _chat(model: str, content: str, session_id: Optional[str] = None, stream: bool = False, cache: bool = False) -> dict:
    # oxllm uses the LiteLLM opt-in convention: the client must send a `cache`
    # object to engage the response cache. Without it, identical prompts are
    # NOT cached (request_wants_cache in hooks/cache_hook.rs).
    body = {"model": model, "messages": [{"role": "user", "content": content}], "stream": stream}
    if cache:
        body["cache"] = {"insert": True, "use_cache": True}
    h = dict(HDRS)
    if session_id:
        h["x-oxllm-session-id"] = session_id
    return {"json": body, "headers": h}


async def _cache_hits_metric(base: str) -> int:
    """Read oxllm_prompt_cache_hits_total from /metrics (sum across labels)."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{base}/metrics", headers=HDRS)
        r.raise_for_status()
    tot = 0
    for line in r.text.splitlines():
        if line.startswith("oxllm_prompt_cache_hits_total") and not line.startswith("#"):
            tot += float(line.split()[-1])
    return int(tot)


async def invariant_cache_sharing(a: str, b: str) -> bool:
    """Warm the cache on A, assert a HIT on B for the same prompt.

    Cache hits are detected via the oxllm_prompt_cache_hits_total metric
    (there is no response-body marker), so we take a before/after delta on B.
    The client MUST opt into caching (LiteLLM convention: a `cache` object in
    the request body) or nothing is cached.
    """
    print("\n[invariant 3] shared cache: warm on A, HIT on B")
    prompt = f"cache-sharing-probe-{uuid.uuid4().hex}"
    async with httpx.AsyncClient(timeout=60) as c:
        hits_before = await _cache_hits_metric(b)
        # Prime on A (miss -> stores in shared Redis).
        r1 = await c.post(f"{a}/v1/chat/completions", **_chat("mock-unthrottled", prompt, cache=True))
        if r1.status_code != 200:
            print(f"  FAIL: prime on A -> {r1.status_code} {r1.text[:200]}")
            return False
        hits_mid = await _cache_hits_metric(b)
        # Hit on B (served from shared Redis; no upstream call).
        r2 = await c.post(f"{b}/v1/chat/completions", **_chat("mock-unthrottled", prompt, cache=True))
        if r2.status_code != 200:
            print(f"  FAIL: hit on B -> {r2.status_code} {r2.text[:200]}")
            return False
        hits_after = await _cache_hits_metric(b)
        c1 = r1.json()["choices"][0]["message"]["content"]
        c2 = r2.json()["choices"][0]["message"]["content"]
        identical = c1 == c2
        hit_on_b = hits_after > hits_mid
        print(f"  content identical: {identical} | B cache-hit delta: {hits_after - hits_mid} (baseline {hits_before})")
        return identical and hit_on_b


async def invariant_fleet_rpm_cap(lb: str) -> bool:
    """rpm:100 on each mock-group deployment is a per-DEPLOYMENT cap enforced
    against a counter shared fleet-wide via Redis (dep_rpm:{deployment}).

    mock-group has two deployments (a and b), each capped at rpm 100, so a
    250-request burst through the LB must yield ~200 200s (2 x 100, minus
    timing slack) and the rest 429. The shared-Redis property is what this
    guards: if a replica kept a LOCAL counter, the ceiling would be
    2 deployments x 100 x 3 replicas = 600 and every request would succeed
    (n429 == 0) — the upper bound + 429 floor below catches that regression.
    Per-KEY rpm (rpm on a virtual key) is a separate counter and is not
    exercised here.
    """
    print("\n[invariant 1] fleet rpm cap: 2 deployments x rpm 100, shared via Redis")
    async with httpx.AsyncClient(timeout=30) as c:
        # Reset window: the rpm counter TTL is 60s; just burst.
        sem = asyncio.Semaphore(20)

        async def one(i):
            async with sem:
                try:
                    r = await c.post(
                        f"{lb}/v1/chat/completions",
                        **_chat("mock-group", f"rl-{i}"),
                    )
                    return r.status_code
                except Exception:
                    return -1

        codes = await asyncio.gather(*[one(i) for i in range(250)])
    n200 = codes.count(200)
    n429 = codes.count(429)
    # Expect roughly 200 successes (2 deployments x the shared rpm:100 cap)
    # and the rest 429. Slack: 15 for timing + counter propagation. The
    # LOWER bound matters as much as the upper one: without it a stack that
    # rejects every request (n200=0, all 429/5xx) would still pass. A local
    # (unshared) counter ceiling of 600 would surface as n429 == 0.
    other = len(codes) - n200 - n429
    ok = 185 <= n200 <= 215 and n429 >= 30 and other == 0
    print(f"  200s={n200} 429s={n429} other={other} -> {'PASS' if ok else 'FAIL'}")
    return ok


async def invariant_session_affinity_deterministic(a: str, b: str) -> Optional[bool]:
    """Same session id picks the same deployment on both instances."""
    print("\n[invariant 5] session affinity deterministic across instances")
    async with httpx.AsyncClient(timeout=30) as c:
        # Use the response's _oxllm_deployment marker if present; otherwise
        # fall back to the deterministic content (same deployment -> same mock
        # content for same prompt).
        sid = "sess-determinism-123"
        r1 = await c.post(f"{a}/v1/chat/completions", **_chat("mock-group", "hello", sid))
        r2 = await c.post(f"{b}/v1/chat/completions", **_chat("mock-group", "hello", sid))
        r1.raise_for_status()
        r2.raise_for_status()
        j1, j2 = r1.json(), r2.json()
        d1 = j1.get("_oxllm_deployment", {})
        d2 = j2.get("_oxllm_deployment", {})
        dep1 = d1.get("key") or d1.get("model_name")
        dep2 = d2.get("key") or d2.get("model_name")
        if dep1 is None or dep2 is None:
            # The explicit marker is not exposed on both instances. Fall back
            # to the response `model` field: the stack config deliberately
            # gives each deployment a distinct upstream model name so routing
            # decisions are visible there (see config.yaml model_list).
            m1, m2 = j1.get("model"), j2.get("model")
            if m1 and m2:
                dep1, dep2 = m1, m2
        # Without comparable signals the check is unevaluated. Keep that
        # distinct from a successful routing-invariance assertion.
        if dep1 is None or dep2 is None:
            print(
                f"  SKIP: no comparable deployment signal "
                f"(A dep={dep1} B dep={dep2}); invariant not enforced"
            )
            return None
        ok = dep1 == dep2
        print(f"  A dep={dep1} B dep={dep2} -> {'PASS' if ok else 'FAIL (deployments differ)'}")
        return ok


async def invariant_no_double_spend(a: str, b: str, replica_c: str = "") -> Optional[bool]:
    """Sum of per-instance cost metrics approx equals DB spend total.

    This is the billing-parity gate (plan §3 invariant 4): each instance's
    `oxllm_cost_total` metric reflects what it believes it spent, while the
    shared Postgres `spend_logs` table is the authoritative durable record. A
    drift between the two signals a spend-logging bug — silent row loss
    (metrics > DB, e.g. spend-writer SQLITE_BUSY drops) or double-writes
    (DB > metrics). The check requires asyncpg; if it is not installed the
    invariant is explicitly skipped (and fails the run unless --allow-skips
    was supplied). A configured but unreachable DB is a failure.
    """
    print("\n[invariant 4] no double spend: metric deltas ~= windowed DB rows")
    async with httpx.AsyncClient(timeout=15) as client:
        def cost(txt):
            tot = 0.0
            for line in txt.splitlines():
                if line.startswith("oxllm_cost_total") and not line.startswith("#"):
                    tot += float(line.split()[-1])
            return tot

        ma = (await client.get(f"{a}/metrics", headers=HDRS)).raise_for_status().text
        mb = (await client.get(f"{b}/metrics", headers=HDRS)).raise_for_status().text
        ca0, cb0 = cost(ma), cost(mb)
        cc0 = 0.0
        c_url: Optional[str] = (replica_c or "").strip() or None
        if c_url:
            mc = (await client.get(f"{c_url}/metrics", headers=HDRS)).raise_for_status().text
            cc0 = cost(mc)
        print(f"  baseline cost(micro$): A={ca0:.0f} B={cb0:.0f} C={cc0:.0f}")

        try:
            import asyncpg  # type: ignore
        except ImportError:
            print(
                "  SKIP: asyncpg not installed; cannot query Postgres spend_logs. "
                "Install with `pip install asyncpg` to enforce the billing-parity gate."
            )
            return None

        pg_url = os.environ.get(
            "OXLLM_POSTGRES_URL", "postgres://postgres:oxllm@127.0.0.1:5432/oxllm"
        )
        try:
            conn = await asyncpg.connect(pg_url, timeout=10, command_timeout=10)
        except Exception as e:  # noqa: BLE001 - surface any connect failure
            print(f"  FAIL: cannot connect to configured Postgres: {e}")
            return False

        try:
            # Window the DB comparison by spend_logs row id: comparing
            # lifetime SUM(cost) against lifetime metric counters breaks when
            # any instance has restarted (counters reset to 0, Postgres keeps
            # history). Row-id bounding is immune to timestamp formats and
            # restarts alike: only rows written during this run count.
            row = await conn.fetchrow("SELECT COALESCE(MAX(id), 0) AS max_id FROM spend_logs")
            max_id = int(row["max_id"]) if row else 0

            # Generate traffic so both sides advance during the window.
            for i in range(10):
                (await client.post(f"{a}/v1/chat/completions", **_chat("mock-unthrottled", f"nods-{i}"))).raise_for_status()
                (await client.post(f"{b}/v1/chat/completions", **_chat("mock-unthrottled", f"nods-{i}"))).raise_for_status()

            # Spend rows are flushed in batches (60s writer interval); poll
            # until the windowed rows land. Break only when the row count has
            # been stable for three consecutive polls (12s) past the 64s mark
            # (one full flush tick): earlier traffic from this suite run
            # (e.g. the rpm burst) can push n past 20 before this
            # invariant's own rows flush, which would under-count the window.
            prev_n = -1
            stable = 0
            for i in range(45):
                row = await conn.fetchrow(
                    "SELECT COUNT(*) AS n, COALESCE(SUM(cost), 0) AS total "
                    "FROM spend_logs WHERE id > $1",
                    max_id,
                )
                n = int(row["n"]) if row else 0
                if i >= 16 and n >= 20 and n == prev_n and stable >= 2:
                    break
                stable = stable + 1 if n == prev_n else 0
                prev_n = n
                await asyncio.sleep(4)
            # Metric counters are micro-dollars (metrics.rs: cost * 1e6);
            # the DB `cost` column is dollars — convert before comparing.
            db_delta = float(row["total"]) * 1_000_000.0 if row else 0.0
        finally:
            await conn.close(timeout=10)

        ma = (await client.get(f"{a}/metrics", headers=HDRS)).raise_for_status().text
        mb = (await client.get(f"{b}/metrics", headers=HDRS)).raise_for_status().text
        ca1, cb1 = cost(ma), cost(mb)
        cc1 = 0.0
        if c_url:
            mc = (await client.get(f"{c_url}/metrics", headers=HDRS)).raise_for_status().text
            cc1 = cost(mc)
        # A restart invalidates this measurement window. Do not manufacture a
        # passing comparison by clamping away one replica's lost counters.
        delta_a, delta_b, delta_c = ca1 - ca0, cb1 - cb0, cc1 - cc0
        if min(delta_a, delta_b, delta_c) < 0:
            print("  SKIP: a replica's cost counter reset; comparison window is invalid")
            return None
        metric_delta = delta_a + delta_b + delta_c
        if metric_delta <= 0.0:
            print("  SKIP: no measurable metric delta (restart mid-run?); invariant not enforced")
            return None
        # Cost is recorded in micro-dollars on the metric side and dollars in
        # spend_logs (converted above); allow 1% relative tolerance plus a
        # small absolute epsilon for float rounding.
        tolerance = max(metric_delta * 0.01, 1.0)
        drift = abs(db_delta - metric_delta)
        ok = drift <= tolerance
        print(
            f"  metric_delta={metric_delta:.0f} (A +{delta_a:.0f}, B +{delta_b:.0f}, "
            f"C +{delta_c:.0f})  "
            f"DB windowed SUM(cost)={db_delta:.0f}  drift={drift:.0f} "
            f"(tol {tolerance:.0f}) -> {'PASS' if ok else 'FAIL'}"
        )
        return ok


async def invariant_config_parity(a: str, b: str) -> bool:
    print("\n[invariant 6] config parity: identical /v1/models on A and B")
    async with httpx.AsyncClient(timeout=15) as c:
        ma = (await c.get(f"{a}/v1/models", headers=HDRS)).raise_for_status().json()
        mb = (await c.get(f"{b}/v1/models", headers=HDRS)).raise_for_status().json()
        ids_a = sorted(m["id"] for m in ma.get("data", []))
        ids_b = sorted(m["id"] for m in mb.get("data", []))
        ok = bool(ids_a) and ids_a == ids_b
        print(f"  A={ids_a}\n  B={ids_b}\n  -> {'PASS' if ok else 'FAIL'}")
        return ok


async def invariant_tag_attribution(lb: str, *replicas: str) -> Optional[bool]:
    """Correlate each unary/streaming call with SQL rows and all-tag rollups.

    Unique run tags isolate this check from the persistent Postgres volume.
    Exact request IDs and memberships detect both lost rows and double writes;
    missing drop-counter samples alone are not evidence of successful writes.
    """
    print("\n[invariant 7] tags and pricing: LB -> SQL -> every replica")
    try:
        import asyncpg  # type: ignore
    except ImportError:
        print("  SKIP: asyncpg not installed; cannot verify tag memberships")
        return None
    pg_url = os.environ.get(
        "OXLLM_POSTGRES_URL", "postgres://postgres:oxllm@127.0.0.1:5432/oxllm"
    )
    conn = await asyncpg.connect(pg_url, timeout=10, command_timeout=10)
    prefix = f"fleet-tags-{uuid.uuid4().hex}"
    tags = [f"{prefix}-header", f"{prefix}-body", f"{prefix}-shared"]
    expected_tags = tags[1:] + tags[:1]  # Body order wins; header tags append/deduplicate.
    observed = {}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(15):
                stream = i >= 12
                payload = _chat("mock-unthrottled", f"{prefix}-{i}", stream=stream)
                payload["headers"]["x-oxllm-tag"] = tags[0]
                # Repeating the header tag in the body must not double-count it.
                payload["json"]["metadata"] = {"tags": tags[1:] + [tags[0]]}
                if stream:
                    payload["json"]["stream_options"] = {"include_usage": True}
                response = await client.post(f"{lb}/v1/chat/completions", **payload)
                response.raise_for_status()
                request_id = response.headers["x-oxllm-request-id"]
                if request_id in observed or response.headers.get("x-request-id") != request_id:
                    raise AssertionError("duplicate or mismatched request ID")
                if stream:
                    data = [line[5:].strip() for line in response.text.splitlines()
                            if line.startswith("data:")]
                    if not data or data[-1] != "[DONE]":
                        raise AssertionError("stream lacks [DONE] terminator")
                    chunks = [json.loads(item) for item in data[:-1]]
                    if any(chunk.get("error") for chunk in chunks):
                        raise AssertionError("in-band stream error")
                    usages = [chunk["usage"] for chunk in chunks
                              if isinstance(chunk.get("usage"), dict)]
                    content = "".join(choice.get("delta", {}).get("content") or ""
                                      for chunk in chunks for choice in chunk.get("choices", []))
                    if not usages or not content:
                        raise AssertionError("stream lacks visible content or usage")
                    usage = usages[-1]
                    if "x-oxllm-response-cost" in response.headers:
                        raise AssertionError("stream claims a premature final-cost header")
                else:
                    usage = response.json()["usage"]
                # Independent arithmetic from this fixture's configured rates.
                expected = usage["prompt_tokens"] * 5e-7 + usage["completion_tokens"] * 1.5e-6
                if not stream and not math.isclose(float(response.headers["x-oxllm-response-cost"]), expected,
                                                  rel_tol=1e-9, abs_tol=1e-12):
                    raise AssertionError("response price differs from fixture rate card")
                observed[request_id] = (usage, expected)

            deadline = time.monotonic() + 90
            while True:
                rows = await conn.fetch(
                    "SELECT request_id, model, provider, prompt_tokens, completion_tokens, "
                    "total_tokens, cost, status, tag_name, tags FROM spend_logs "
                    "WHERE request_id = ANY($1::text[])", list(observed)
                )
                if len(rows) >= len(observed):
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError(f"missing durable rows: {len(rows)}/{len(observed)}")
                await asyncio.sleep(0.5)
            if len(rows) != len(observed) or {row["request_id"] for row in rows} != set(observed):
                raise AssertionError("missing or duplicate SQL rows")
            for row in rows:
                usage, expected = observed[row["request_id"]]
                if row["status"] != "success" or row["provider"] != "openai":
                    raise AssertionError("incorrect spend status/provider")
                if row["tag_name"] != expected_tags[0] or json.loads(row["tags"]) != expected_tags:
                    raise AssertionError("missing, reordered or duplicate SQL tags")
                for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    if row[field] != usage[field]:
                        raise AssertionError(f"incorrect SQL {field}")
                if not math.isclose(row["cost"], expected, rel_tol=1e-9, abs_tol=1e-12):
                    raise AssertionError("SQL cost differs from response usage")
            expected_total = sum(cost for _, cost in observed.values())
            memberships = await conn.fetch(
                "SELECT st.tag_name, COUNT(*) AS n, SUM(s.cost) AS cost "
                "FROM spend_log_tags st JOIN spend_logs s ON s.id = st.spend_log_id "
                "WHERE st.tag_name = ANY($1::text[]) GROUP BY st.tag_name", tags
            )
            if len(memberships) != len(tags):
                raise AssertionError("missing normalized tag memberships")
            for row in memberships:
                if row["n"] != len(observed) or not math.isclose(row["cost"], expected_total,
                                                               rel_tol=1e-9, abs_tol=1e-12):
                    raise AssertionError("tag memberships double-count or lose spend")
            for replica in filter(None, replicas):
                page = (await client.get(f"{replica}/spend/logs", headers=HDRS,
                                        params={"tag_name": tags[0], "limit": 100})).raise_for_status().json()
                if page["total_requests"] != len(observed) or {row["request_id"] for row in page["data"]} != set(observed):
                    raise AssertionError(f"{replica}: cross-replica spend reads differ")
                aggregates, offset = {}, 0
                while True:
                    page = (await client.get(f"{replica}/spend/tags", headers=HDRS,
                                            params={"limit": 500, "offset": offset})).raise_for_status().json()
                    aggregates.update({row["tag_name"]: row for row in page["data"]})
                    if len(page["data"]) < 500:
                        break
                    offset += len(page["data"])
                for tag in tags:
                    row = aggregates[tag]
                    if row["total_requests"] != len(observed) or not math.isclose(row["total_spend"], expected_total,
                                                                                rel_tol=1e-9, abs_tol=1e-12):
                        raise AssertionError(f"{replica}: tag aggregate mismatch")
                metrics = (await client.get(f"{replica}/metrics", headers=HDRS)).raise_for_status().text
                if "oxllm_requests_total" not in metrics:
                    raise AssertionError(f"{replica}: metrics unavailable")
                if any(float(line.split()[-1]) > 0 for line in metrics.splitlines()
                       if line.startswith("oxllm_spend_entries_dropped_total")):
                    raise AssertionError(f"{replica}: spend-writer drops")
        print(f"  PASS: 12 unary + 3 streams, {len(tags)} tags x {len(observed)} exact memberships, "
              f"${expected_total:.8f}, identical reads across {len(tuple(filter(None, replicas)))} replicas")
        return True
    finally:
        await conn.close(timeout=10)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="single base URL (LB) for fleet-view invariants")
    ap.add_argument("--a", default="http://127.0.0.1:4000")
    ap.add_argument("--b", default="http://127.0.0.1:4001")
    ap.add_argument("--c", default="http://127.0.0.1:4002",
                    help="third instance; pass an empty string to restrict the "
                         "double-spend check to A and B only")
    ap.add_argument("--allow-skips", action="store_true",
                    help="Allow explicitly unevaluated invariants; never suppress failures")
    args = ap.parse_args()
    lb = args.base or "http://127.0.0.1:8080"

    results: dict[str, Optional[bool]] = {}
    checks = [
        ("config_parity", invariant_config_parity, (args.a, args.b)),
        ("cache_sharing", invariant_cache_sharing, (args.a, args.b)),
        ("session_affinity", invariant_session_affinity_deterministic, (args.a, args.b)),
        ("fleet_rpm_cap", invariant_fleet_rpm_cap, (lb,)),
        ("no_double_spend", invariant_no_double_spend, (args.a, args.b, args.c)),
        ("tag_attribution", invariant_tag_attribution, (lb, args.a, args.b, args.c)),
    ]
    for name, check, inputs in checks:
        try:
            results[name] = await check(*inputs)
        except Exception as exc:
            print(f"  FAIL: {name}: {type(exc).__name__}: {exc}")
            results[name] = False

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        status = "SKIP" if v is None else ("PASS" if v else "FAIL")
        print(f"  {k}: {status}")
    failed = any(value is False for value in results.values())
    skipped = any(value is None for value in results.values())
    if skipped and not args.allow_skips:
        print("  Skipped invariants require --allow-skips to permit a successful run.")
    return 1 if failed or (skipped and not args.allow_skips) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
