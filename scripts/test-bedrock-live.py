#!/usr/bin/env python3
"""Verify Bedrock through a real proxy using the AWS CLI's current login.

Credentials stay in subprocess memory/environment. Catalogs, sanitized results,
and proxy logs are retained outside the checkout. Requests incur provider usage.
Python 3.11+, standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
TEMP_ROOT = Path(os.environ.get("OXLLM_TEST_TEMP_DIR", tempfile.gettempdir()))
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
MASTER = "sk-bedrock-live-verification-master-00000000"
RATE_IN, RATE_OUT = 0.000001, 0.000002


def aws(args: list[str], profile: str) -> dict:
    result = subprocess.run(["aws", *args, "--profile", profile, "--no-cli-pager"],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"AWS {args[0]} failed: {result.stderr[-1200:]}")
    return json.loads(result.stdout)


def http(base: str, path: str, body: dict | None = None, token: str = MASTER) -> tuple[int, dict, str]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
    })
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read().decode()


def parser_module():
    spec = importlib.util.spec_from_file_location("live_model_helpers", ROOT / "scripts/test-live-models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_cost(headers: dict, body: dict) -> float:
    usage = body.get("usage") or {}
    prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    if prompt <= 0 or completion < 0:
        raise AssertionError(f"missing/invalid metering: {usage}")
    cost = float(headers["x-oxllm-response-cost"])
    write = usage.get("cache_creation_input_tokens", 0)
    expected = (prompt + write) * RATE_IN + completion * RATE_OUT
    if not math.isfinite(cost) or not math.isclose(cost, expected, rel_tol=1e-8, abs_tol=1e-12):
        raise AssertionError(f"cost {cost} does not match declared verification rates {expected}")
    return cost


def unmerged_converse_body(payload: dict, model: str) -> dict:
    """Build a direct control request without role grouping or history repair."""
    messages = []
    for message in payload["messages"]:
        role = message["role"]
        if role == "tool":
            blocks = [{"toolResult": {"toolUseId": message["tool_call_id"], "content": [{"text": message["content"]}]}}]
            role = "user"
        else:
            blocks = [{"text": message["content"]}] if message.get("content") else []
            for call in message.get("tool_calls", []):
                blocks.append({"toolUse": {"toolUseId": call["id"], "name": call["function"]["name"],
                    "input": json.loads(call["function"]["arguments"])}})
        messages.append({"role": role, "content": blocks})
    body = {"modelId": model, "messages": messages, "inferenceConfig": {"maxTokens": payload["max_tokens"]}}
    if payload.get("tools"):
        body["toolConfig"] = {"tools": [{"toolSpec": {"name": tool["function"]["name"],
            "description": tool["function"]["description"], "inputSchema": {"json": tool["function"]["parameters"]}}} for tool in payload["tools"]]}
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/oxllm")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--phase", choices=["embeddings", "chat", "operations", "compatibility", "all"], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="*", help="Limit chat checks to these exact catalog/profile ids")
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="oxllm-bedrock-live-", dir=TEMP_ROOT))
    output.mkdir(parents=True, exist_ok=True)
    credentials = aws(["configure", "export-credentials", "--format", "process"], args.profile)
    env = {**os.environ, "AWS_ACCESS_KEY_ID": credentials["AccessKeyId"],
           "AWS_SECRET_ACCESS_KEY": credentials["SecretAccessKey"], "AWS_REGION": "us-east-1", "AWS_REGION_NAME": "us-east-1",
           "RUST_LOG": "info", "OXLLM_AUTO_MIGRATE": "true", "OXLLM_ENVIRONMENT": "development"}
    if credentials.get("SessionToken"):
        env["AWS_SESSION_TOKEN"] = credentials["SessionToken"]
    else:
        env.pop("AWS_SESSION_TOKEN", None)
    secrets = [credentials.get(field, "") for field in ["AccessKeyId", "SecretAccessKey", "SessionToken"]] + [MASTER]

    def safe(value: str) -> str:
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        return value

    catalogs, profiles = {}, {}
    for region in ["us-east-1", "us-west-2"]:
        catalogs[region] = aws(["bedrock", "list-foundation-models", "--region", region], args.profile)
        profiles[region] = aws(["bedrock", "list-inference-profiles", "--region", region,
                                "--type-equals", "SYSTEM_DEFINED"], args.profile)
    (output / "catalogs.json").write_text(json.dumps(catalogs, indent=2))
    (output / "profiles.json").write_text(json.dumps(profiles, indent=2))
    model_lists = {region: {m["modelId"]: m for m in data["modelSummaries"]}
                   for region, data in catalogs.items()}
    profile_ids = {region: {m["inferenceProfileId"] for m in data["inferenceProfileSummaries"]}
                   for region, data in profiles.items()}
    deployments, embedding_cases, chat_cases, compatibility_cases = [], [], [], []

    def deployment(alias: str, model: str, region: str, extra: dict | None = None) -> None:
        params = {"model": f"bedrock/{model}", "timeout": 180, "stream_timeout": 180,
                  "input_cost_per_token": RATE_IN, "output_cost_per_token": RATE_OUT,
                  "cache_read_input_token_cost": RATE_IN, "cache_creation_input_token_cost": RATE_IN,
                  "cache_creation_input_token_cost_above_1hr": RATE_IN, **(extra or {})}
        if region != "us-east-1":
            params["aws_region_name"] = region
        deployments.append({"model_name": alias, "oxllm_params": params})

    if args.phase in {"embeddings", "all"}:
        for region in model_lists:
            for model in ["us.cohere.embed-v4:0", "global.cohere.embed-v4:0", "cohere.embed-v4:0",
                          "cohere.embed-english-v3", "cohere.embed-multilingual-v3",
                          "amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v1"]:
                info = model_lists[region].get(model)
                if model not in profile_ids[region] and (not info or "ON_DEMAND" not in info.get("inferenceTypesSupported", [])):
                    continue
                alias = f"embed-{region}-{model}"
                deployment(alias, model, region, {"additional_drop_params": ["encoding_format"]})
                inputs = ["A small live embedding verification."] if "titan" in model else ["A small live embedding verification.", "A second sentence."]
                dims = 1024 if "titan-embed-text-v2" in model or "v3" in model else 1536
                embedding_cases.append((alias, model, region, inputs, None, dims, "float"))
                if "embed-v4" in model or "titan-embed-text-v2" in model:
                    embedding_cases.append((alias, model, region, inputs, 512, 512, "float"))
                    embedding_cases.append((alias, model, region, inputs, 256, 256, "base64"))

        for region in model_lists:
            info = model_lists[region].get("cohere.embed-v4:0")
            if info and "ON_DEMAND" in info.get("inferenceTypesSupported", []):
                arn = info["modelArn"]
                alias = f"embed-foundation-arn-{region}"
                deployment(alias, arn, region)
                embedding_cases.append((alias, arn, region, ["ARN embedding verification."], None, 1536, "float"))

    if args.phase in {"chat", "all"}:
        desired = args.models or [
            "global.openai.gpt-5.6-luna", "global.openai.gpt-6-luna", "global.openai.gpt-6.1-sol",
            "global.anthropic.claude-sonnet-5-5", "global.anthropic.claude-opus-5-5",
            "global.anthropic.claude-sonnet-4-6", "global.anthropic.claude-opus-4-6-v1",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0", "global.amazon.nova-2-lite-v1:0",
            "global.xai.grok-4.7", "global.zai.glm-5.3", "global.moonshotai.kimi-k3",
            "mistral.mistral-large-3-675b-instruct", "qwen.qwen3-coder-next",
            "deepseek.v3.2", "openai.gpt-oss-120b-1:0",
        ]
        for region in model_lists:
            for model in desired:
                info = model_lists[region].get(model)
                if model not in profile_ids[region] and (not info or "ON_DEMAND" not in info.get("inferenceTypesSupported", [])):
                    continue
                alias = f"chat-{region}-{model}"
                extra = {"use_converse_api": True}
                if ".openai." in model or model.startswith("openai."):
                    extra["additional_drop_params"] = ["reasoning_effort"]
                deployment(alias, model, region, extra)
                chat_cases.append((alias, model, region))
                if "haiku" in model:
                    invoke_alias = f"invoke-{region}-{model}"
                    deployment(invoke_alias, model, region)
                    chat_cases.append((invoke_alias, model, region))

    if args.phase in {"operations", "all"}:
        haiku = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        deployment("ops-chat", haiku, "us-east-1", {"use_converse_api": True})
        deployment("ops-fallback", haiku, "us-west-2", {"use_converse_api": True})
        deployment("ops-bad-primary", haiku, "us-east-1", {"aws_secret_access_key": "synthetic-invalid-signing-secret", "use_converse_api": True})
        deployment("ops-siblings", haiku, "us-east-1", {"order": 1, "aws_secret_access_key": "synthetic-invalid-signing-secret", "use_converse_api": True})
        deployment("ops-siblings", haiku, "us-west-2", {"order": 2, "use_converse_api": True})
        deployment("ops-embed", "us.cohere.embed-v4:0", "us-west-2", {"additional_drop_params": ["encoding_format"]})
        deployment("ops-bad-embed", "us.cohere.embed-v4:0", "us-east-1", {"aws_secret_access_key": "synthetic-invalid-signing-secret"})
        deployment("*", "*", "us-east-1")
        deployment("ops-cache", "global.anthropic.claude-sonnet-4-6", "us-east-1", {"cache_control_injection_points": [
            {"location": "message", "role": "system"}, {"location": "message", "index": -1}]})
        deployment("ops-mantle-grok", "xai.grok-4.3", "us-east-1")
        deployments[-1]["oxllm_params"].update(model="bedrock_mantle/xai.grok-4.3",
            api_base="https://bedrock-mantle.us-west-2.api.aws/openai/v1")
        deployment("ops-mantle-oss", "openai.gpt-oss-120b", "us-east-1")
        deployments[-1]["oxllm_params"].update(model="bedrock_mantle/openai.gpt-oss-120b",
            api_base="https://bedrock-mantle.us-west-2.api.aws/v1")
        chat_cases.extend([("ops-mantle-grok", "xai.grok-4.3", "us-west-2"),
                           ("ops-mantle-oss", "openai.gpt-oss-120b", "us-west-2")])

    if args.phase in {"compatibility", "all"}:
        tools = [{"type": "function", "function": {"name": "echo", "description": "Echo the supplied marker.", "parameters": {
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]

        def history(markers):
            return [{"role": "user", "content": "The tools have already returned their markers. Summarize the results without calling a tool."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": f"call_{marker.lower()}", "type": "function",
                    "function": {"name": "echo", "arguments": json.dumps({"text": marker})}} for marker in markers]},
                *[{"role": "tool", "tool_call_id": f"call_{marker.lower()}", "content": marker} for marker in markers]]

        for index, model in enumerate(["us.anthropic.claude-haiku-4-5-20251001-v1:0", "global.amazon.nova-2-lite-v1:0", "openai.gpt-oss-120b-1:0"]):
            alias = f"compat-{index}"
            deployment(alias, model, "us-east-1", {"use_converse_api": True})
            for label, messages, has_tools in [
                ("consecutive-users", [{"role": "user", "content": "The first marker is ALPHA."}, {"role": "user", "content": "The second marker is BETA. Reply with both markers."}], False),
                ("consecutive-assistants", [{"role": "user", "content": "Record two markers."}, {"role": "assistant", "content": "The first marker is ALPHA."},
                    {"role": "assistant", "content": "The second marker is BETA."}, {"role": "user", "content": "Reply with both markers."}], False),
                ("tool-result-followup", [*history(["ALPHA"]), {"role": "user", "content": "Reply with the returned marker as text."}], True),
                ("parallel-tool-results", history(["ALPHA", "BETA"]), True),
                ("history-without-declarations", history(["ALPHA"]), False),
            ]:
                payload = {"model": alias, "messages": messages, "max_tokens": 512}
                if has_tools:
                    payload["tools"] = tools
                compatibility_cases.append((alias, model, label, payload))
            payload = {"model": alias, "messages": [{"role": "user", "content": "Be brief."}, {"role": "user", "content": "Reply with exactly: stream-ok"}],
                       "max_tokens": 512, "stream": True, "stream_options": {"include_usage": True}}
            compatibility_cases.append((alias, model, "consecutive-users-stream", payload))
        deployment("compat-no-repairs", "us.anthropic.claude-haiku-4-5-20251001-v1:0", "us-east-1", {"use_converse_api": True, "modify_params": False})

    config = {"model_list": [], "include": ["models.json"],
              "general_settings": {"master_key": MASTER, "database_url": "os.environ/DATABASE_URL", "log_format": "json"},
              "oxllm_settings": {"drop_params": True, "request_timeout": 180, "cache_ttl_secs": 300},
              "router_settings": {"num_retries": 0, "cooldown_time": 0,
                   "fallbacks": [{"ops-bad-primary": ["ops-fallback"]}, {"ops-bad-embed": ["ops-embed"]}]}}
    if compatibility_cases:
        config["oxllm_settings"]["modify_params"] = True
    models_path = output / "models.json"
    models_path.write_text(json.dumps({"model_list": deployments}))
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config))
    env["DATABASE_URL"] = f"sqlite://{output / 'spend.db'}?mode=rwc"
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    results = []
    token = MASTER
    helpers = parser_module()
    log = (output / "proxy.log").open("w")
    proxy = subprocess.Popen([str(args.binary.resolve()), "serve", "--config", str(config_path), "--port", str(port)],
                             env=env, stdout=log, stderr=subprocess.STDOUT)

    def record(label: str, model: str, region: str, run) -> None:
        started = time.monotonic()
        row = {"case": label, "model": model, "region": region}
        try:
            row.update(run())
            row["status"] = "pass"
        except Exception as error:
            row.update(status="fail", error=safe(str(error)))
        row["seconds"] = round(time.monotonic() - started, 3)
        results.append(row)
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)

    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if proxy.poll() is not None:
                print(safe((output / "proxy.log").read_text()[-4000:]), file=sys.stderr)
                raise RuntimeError("proxy exited during startup; inspect sanitized proxy.log")
            try:
                if http(base, "/health/readiness")[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("proxy did not become ready")
        status, _, text = http(base, "/key/generate", {"models": ["*"], "max_budget": 5.0})
        if status != 200:
            raise RuntimeError(f"virtual key creation failed: {text[:1000]}")
        token = json.loads(text)["key"]
        secrets.append(token)

        for alias, model, label, payload in compatibility_cases:
            def run(payload=payload, model=model):
                direct_body = unmerged_converse_body(payload, model)
                comparison = {"direct_raw_roles": [message["role"] for message in direct_body["messages"]]}
                if not payload.get("stream"):
                    try:
                        direct_response = aws(["bedrock-runtime", "converse", "--region", "us-east-1", "--cli-input-json", json.dumps(direct_body),
                            "--cli-connect-timeout", "5", "--cli-read-timeout", "60"], args.profile)
                        comparison.update(direct_raw_accepted=True, direct_raw_usage=direct_response.get("usage"))
                    except RuntimeError as error:
                        comparison.update(direct_raw_accepted=False, direct_raw_error=safe(str(error)))
                status, headers, text = http(base, "/v1/chat/completions", payload, token)
                if status != 200:
                    raise AssertionError(f"HTTP {status}: {text[:1200]}")
                body = helpers.parse_stream(text) if payload.get("stream") else json.loads(text)
                message = body["choices"][0]["message"]
                if not (message.get("content") or "").strip() and not message.get("tool_calls"):
                    raise AssertionError(f"empty completion: {body}")
                metered = headers if not payload.get("stream") else {"x-oxllm-response-cost": str(
                    body["usage"]["prompt_tokens"] * RATE_IN + body["usage"]["completion_tokens"] * RATE_OUT)}
                return {"http": status, "usage": body["usage"], "cost": check_cost(metered, body),
                    "request_id": headers.get("x-oxllm-request-id"), **comparison}
            record(label, model, "us-east-1", run)

        if compatibility_cases:
            def disabled_repairs():
                payload = next(payload.copy() for alias, _, label, payload in compatibility_cases if alias == "compat-0" and label == "history-without-declarations")
                payload["model"] = "compat-no-repairs"
                status, _, text = http(base, "/v1/chat/completions", payload, token)
                if status != 400 or "tool" not in text.lower():
                    raise AssertionError(f"expected tool-history validation failure, got HTTP {status}: {text[:1200]}")
                return {"http": status, "expected_validation_failure": True}
            record("deployment-disables-global-history-repairs", "haiku-converse", "us-east-1", disabled_repairs)

        for alias, model, region, inputs, dimensions, expected_dimensions, encoding in embedding_cases:
            def run(alias=alias, inputs=inputs, dimensions=dimensions, expected_dimensions=expected_dimensions, encoding=encoding):
                payload = {"model": alias, "input": inputs, "encoding_format": encoding}
                if dimensions:
                    payload["dimensions"] = dimensions
                status, headers, text = http(base, "/v1/embeddings", payload, token)
                if status != 200:
                    raise AssertionError(f"HTTP {status}: {text[:1200]}")
                body = json.loads(text)
                if len(body["data"]) != len(inputs):
                    raise AssertionError("embedding cardinality mismatch")
                for index, item in enumerate(body["data"]):
                    if item["index"] != index or len(item["embedding"]) != expected_dimensions:
                        raise AssertionError(f"embedding index/dimensions mismatch: index {item['index']}, dims {len(item['embedding'])}, wanted {expected_dimensions}")
                    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in item["embedding"]):
                        raise AssertionError("embedding contains invalid numbers")
                cost = check_cost(headers, body)
                return {"http": status, "dimensions": expected_dimensions, "vectors": len(inputs), "usage": body["usage"], "cost": cost,
                        "request_id": headers.get("x-oxllm-request-id")}
            record(f"embedding-{dimensions or 'default'}-{encoding}", model, region, run)

        for alias, model, region in chat_cases:
            for mode in ["plain", "stream", "tools"]:
                def run(alias=alias, mode=mode, model=model):
                    payload = {"model": alias, "messages": [{"role": "user", "content": "Reply with exactly: bedrock-live-ok"}], "max_tokens": 1024 if "mantle" in alias else 512}
                    if "mantle" in alias:
                        payload["reasoning_effort"] = "low"
                    if ".openai." in model or model.startswith("openai."):
                        payload["reasoning_effort"] = "low"
                    if mode == "stream":
                        payload["stream"] = True
                        payload["stream_options"] = {"include_usage": True}
                    if mode == "tools":
                        payload["messages"][0]["content"] = "Call the echo function with text exactly bedrock-live-ok. Do not answer with plain text."
                        payload["tools"] = [{"type": "function", "function": {"name": "echo", "description": "Echo the supplied text.",
                            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
                        payload["tool_choice"] = "auto"
                    status, headers, text = http(base, "/v1/chat/completions", payload, token)
                    if status != 200:
                        raise AssertionError(f"HTTP {status}: {text[:1200]}")
                    body = helpers.parse_stream(text) if mode == "stream" else json.loads(text)
                    (output / f"response-{len(results)}.json").write_text(safe(json.dumps(body)))
                    message = body["choices"][0]["message"]
                    if mode == "tools":
                        calls = message.get("tool_calls") or []
                        if not calls or calls[0]["function"]["name"] != "echo":
                            raise AssertionError(f"missing echo tool call: {message}")
                        json.loads(calls[0]["function"]["arguments"])
                    elif not (message.get("content") or "").strip():
                        raise AssertionError("empty visible completion")
                    cost = check_cost(headers, body) if mode != "stream" else check_cost({"x-oxllm-response-cost": str(
                        body["usage"]["prompt_tokens"] * RATE_IN + body["usage"]["completion_tokens"] * RATE_OUT)}, body)
                    return {"http": status, "mode": mode, "usage": body["usage"], "cost": cost,
                            "request_id": headers.get("x-oxllm-request-id")}
                record(mode, model, region, run)

        if args.phase in {"operations", "all"}:
            def chat(alias, messages=None, extra=None):
                payload = {"model": alias, "messages": messages or [{"role": "user", "content": "Reply: bedrock-live-ok"}], "max_tokens": 256, **(extra or {})}
                status, headers, text = http(base, "/v1/chat/completions", payload, token)
                if status != 200:
                    raise AssertionError(f"HTTP {status}: {text[:1200]}")
                body = json.loads(text)
                return body, headers

            for alias in ["ops-bad-primary", "ops-siblings"]:
                def run(alias=alias):
                    body, headers = chat(alias)
                    if alias == "ops-bad-primary" and headers.get("x-oxllm-fallback-used") != "ops-fallback":
                        raise AssertionError(f"missing fallback attribution: {headers}")
                    return {"http": 200, "cost": check_cost(headers, body), "usage": body["usage"], "request_id": headers["x-oxllm-request-id"]}
                record(alias, "real-signature-failure-to-real-haiku", "us-east-1/us-west-2", run)

            def embed_fallback():
                status, headers, text = http(base, "/v1/embeddings", {"model": "ops-bad-embed", "input": ["Real embedding failover."], "encoding_format": "float"}, token)
                if status != 200:
                    raise AssertionError(f"HTTP {status}: {text[:1200]}")
                body = json.loads(text)
                if len(body["data"][0]["embedding"]) != 1536:
                    raise AssertionError("fallback embedding dimensions mismatch")
                return {"http": status, "cost": check_cost(headers, body), "usage": body["usage"], "request_id": headers["x-oxllm-request-id"]}
            record("embedding-failover", "us.cohere.embed-v4:0", "us-east-1/us-west-2", embed_fallback)

            def wildcard_embed():
                model = "bedrock/us.cohere.embed-v4:0"
                status, headers, text = http(base, "/v1/embeddings", {"model": model, "input": ["Wildcard embedding."]}, token)
                if status != 200:
                    raise AssertionError(f"HTTP {status}: {text[:1200]}")
                body = json.loads(text)
                if body["model"] != model or len(body["data"][0]["embedding"]) != 1536:
                    raise AssertionError(f"wildcard response identity/dimensions mismatch: {body.get('model')}")
                return {"http": status, "usage": body["usage"], "cost": check_cost(headers, body), "request_id": headers["x-oxllm-request-id"]}
            record("wildcard-embedding", "us.cohere.embed-v4:0", "us-east-1", wildcard_embed)

            def tool_round_trip():
                tools = [{"type": "function", "function": {"name": "echo", "description": "Echo a value.", "parameters": {
                    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
                messages = [{"role": "user", "content": "Call echo with bedrock-live-ok."}]
                body, headers = chat("ops-chat", messages, {"tools": tools, "tool_choice": "auto"})
                calls = body["choices"][0]["message"].get("tool_calls") or []
                if not calls:
                    raise AssertionError("missing tool call")
                results.append({"case": "tool-roundtrip-first-turn", "model": "ops-chat", "region": "us-east-1", "status": "pass",
                    "usage": body["usage"], "cost": check_cost(headers, body), "request_id": headers["x-oxllm-request-id"]})
                messages.extend([body["choices"][0]["message"], {"role": "tool", "tool_call_id": calls[0]["id"], "content": "bedrock-live-ok"},
                    {"role": "user", "content": "Now return the echoed value as text."}])
                final, headers = chat("ops-chat", messages, {"tools": tools})
                if not (final["choices"][0]["message"].get("content") or "").strip():
                    raise AssertionError("empty tool-result continuation")
                return {"http": 200, "cost": check_cost(headers, final), "usage": final["usage"], "request_id": headers["x-oxllm-request-id"]}
            record("tool-result-roundtrip", "haiku-converse", "us-east-1", tool_round_trip)

            def consecutive_users():
                body, headers = chat("ops-chat", [{"role": "user", "content": "Be brief."}, {"role": "user", "content": "Say ok."}])
                return {"http": 200, "cost": check_cost(headers, body), "usage": body["usage"], "request_id": headers["x-oxllm-request-id"]}
            record("consecutive-user-turns", "haiku-converse", "us-east-1", consecutive_users)

            def prompt_cache():
                system = output.name + "\n" + "\n".join(f"Memo {index}: The verified record contains the entry number {index}." for index in range(1600))
                messages = [{"role": "system", "content": system}, {"role": "user", "content": "Reply with exactly: cache-ok"}]
                first, headers = chat("ops-cache", messages)
                results.append({"case": "prompt-cache-write", "model": "ops-cache", "region": "us-east-1", "status": "pass",
                    "usage": first["usage"], "cost": check_cost(headers, first), "request_id": headers["x-oxllm-request-id"]})
                time.sleep(1)
                second, headers = chat("ops-cache", messages)
                read = (second["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", second["usage"].get("cache_read_input_tokens", 0))
                if read <= 0:
                    raise AssertionError(f"configured breakpoints did not produce cache reads: {second['usage']}")
                return {"http": 200, "cache_read_tokens": read, "usage": second["usage"], "cost": check_cost(headers, second), "request_id": headers["x-oxllm-request-id"]}
            record("prompt-cache-read", "sonnet-4.6-invoke", "us-east-1", prompt_cache)

            def response_cache():
                extra = {"cache": {"insert": True}}
                first, headers = chat("ops-chat", extra=extra)
                results.append({"case": "response-cache-first", "model": "ops-chat", "region": "us-east-1", "status": "pass",
                    "usage": first["usage"], "cost": check_cost(headers, first), "request_id": headers["x-oxllm-request-id"]})
                second, headers = chat("ops-chat", extra=extra)
                if headers.get("x-oxllm-cache-hit") != "true":
                    raise AssertionError("missing response-cache hit")
                return {"http": 200, "cache_hit": True, "usage": second["usage"], "cost": check_cost(headers, second), "request_id": headers["x-oxllm-request-id"]}
            record("response-cache-hit", "haiku-converse", "us-east-1", response_cache)

            def hot_reload():
                alias = "ops-after-include-reload"
                deployment(alias, "us.cohere.embed-v4:0", "us-west-2")
                models_path.write_text(json.dumps({"model_list": deployments}))
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    status, _, text = http(base, "/v1/models")
                    if status == 200 and any(model["id"] == alias for model in json.loads(text)["data"]):
                        return {"http": status, "model_added_without_restart": alias}
                    time.sleep(0.2)
                raise AssertionError("included-file change was not reloaded by the serving binary")
            record("included-config-hot-reload", "real-serving-binary", "local", hot_reload)
    finally:
        proxy.send_signal(signal.SIGTERM)
        try:
            proxy.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.wait()
        log.close()
        log_path = output / "proxy.log"
        log_path.write_text(safe(log_path.read_text()))
    with sqlite3.connect(output / "spend.db") as database:
        for result in list(results):
            if result["status"] != "pass" or not result.get("request_id"):
                continue
            rows = database.execute("SELECT cost, prompt_tokens, completion_tokens FROM spend_logs WHERE request_id = ?", (result["request_id"],)).fetchall()
            if result.get("cache_hit"):
                valid = not rows
            else:
                valid = len(rows) == 1 and math.isclose(rows[0][0], result["cost"], rel_tol=1e-8, abs_tol=1e-12)
            if not valid:
                results.append({"case": "durable-spend", "model": result["model"], "region": result["region"], "status": "fail",
                    "error": f"spend mismatch for {result['request_id']}: {rows}"})
    (output / "results.json").write_text(json.dumps(results, indent=2))
    passed = sum(row["status"] == "pass" for row in results)
    print(json.dumps({"passed": passed, "failed": len(results) - passed, "evidence": str(output)}))
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
