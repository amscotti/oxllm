#!/usr/bin/env bash
# Check a running stack. Default checks never call a provider.
# OXLLM_URL / OXLLM_MASTER_KEY select the proxy. OXLLM_LOAD_ENV=0 skips .env.
# OXLLM_DB_SMOKE=1 tests management DB access; OXLLM_ADMIN_URL tests the UI.
# OXLLM_LIVE_SMOKE=1 explicitly enables small, billable provider requests.
# Requires Python 3; useful against either Docker or a local binary.
set -euo pipefail
if [[ "${OXLLM_LOAD_ENV:-1}" == 1 && -z "${OXLLM_MASTER_KEY:-}" && -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${OXLLM_MASTER_KEY:?export OXLLM_MASTER_KEY or supply it in .env}"
export OXLLM_MASTER_KEY
python3 - <<'PY'
import json
import os
import sys
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

base = os.environ.get("OXLLM_URL", "http://127.0.0.1:4000").rstrip("/")
key = os.environ["OXLLM_MASTER_KEY"]
passed = failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


def request(path, headers=None, data=None, origin=base):
    headers = dict(headers or {})
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(origin + path, data=json.dumps(data).encode() if data is not None else None, headers=headers)
    try:
        with urlopen(req, timeout=150) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()
    except (URLError, TimeoutError) as error:
        print(f"Request failed for {path}: {error.reason if isinstance(error, URLError) else 'timeout'}")
        return 0, b""


auth = {"Authorization": f"Bearer {key}"}
print(f"Smoke checks against {base} (live calls explicitly opt-in)")
check("liveness returns 200", request("/health")[0] == 200)
check("readiness returns 200", request("/health/readiness")[0] == 200)
check("unauthenticated chat is rejected", request("/v1/chat/completions", data={})[0] == 401)
check("unauthenticated model list is rejected", request("/v1/models")[0] == 401)
for name, headers in (("Bearer", auth), ("x-api-key", {"x-api-key": key})):
    status, body = request("/v1/models", headers)
    try:
        valid = status == 200 and isinstance(json.loads(body).get("data"), list)
    except (ValueError, AttributeError):
        valid = False
    check(f"{name} authenticates without an upstream request", valid)

if os.environ.get("OXLLM_DB_SMOKE") == "1":
    check("management database is available", request("/key/list", auth)[0] == 200)

admin = os.environ.get("OXLLM_ADMIN_URL", "").rstrip("/")
if admin:
    check("admin liveness", request("/health", origin=admin)[0] == 200)
    status, body = request("/login", origin=admin)
    check("admin login page", status == 200 and b"form" in body)

    class Assets(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = {"/static/logo.png"}
            self.remote_scripts = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            for field in ("src", "href"):
                path = attrs.get(field, "")
                if path.startswith("/static/"):
                    self.paths.add(path)
                if tag == "script" and field == "src" and path.startswith(("http://", "https://", "//")):
                    self.remote_scripts.append(path)

    assets = Assets()
    assets.feed(body.decode(errors="replace"))
    check("admin browser scripts are served locally", not assets.remote_scripts)
    for path in sorted(assets.paths):
        status, body = request(path, origin=admin)
        check(f"admin asset {path}", status == 200 and len(body) > 0)

if os.environ.get("OXLLM_LIVE_SMOKE") == "1":
    for env, model in (("OPENAI_API_KEY", "gpt-4o"), ("XAI_API_KEY", "grok"), ("ANTHROPIC_API_KEY", "claude")):
        credential = os.environ.get(env, "")
        if not credential or "placeholder" in credential:
            print(f"SKIP: live {model} ({env} unset or placeholder)")
            continue
        path = "/v1/messages" if model == "claude" else "/v1/chat/completions"
        status, body = request(path, {**auth, "anthropic-version": "2023-06-01"}, {"model": model, "messages": [{"role": "user", "content": "Reply with pong"}], "max_tokens": 16})
        try:
            result = json.loads(body)
            valid = status == 200 and bool(result.get("content") if model == "claude" else result.get("choices"))
        except (ValueError, AttributeError):
            valid = False
        check(f"live {model}", valid)
print(f"Result: {passed} passed, {failed} failed")
sys.exit(bool(failed))
PY
