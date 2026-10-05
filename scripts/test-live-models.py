#!/usr/bin/env python3
"""Exercise current models through a real proxy, including durable billing.

Python 3.11+; standard library only. Loads .env without printing credentials.
Examples:
  cargo build --locked --features db --bin oxllm
  python3 scripts/test-live-models.py --binary target/debug/oxllm --tools
  python3 scripts/test-live-models.py --image oxllm:live --all-providers --tools
  python3 scripts/test-live-models.py --image oxllm:live --replicas 2 --tools

The Docker multi-replica mode creates disposable Postgres and Redis services.
All managed processes, containers and configuration files are removed on exit.
Evidence logs and the JSON result are retained in the printed output directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent.parent
MASTER = "sk-live-model-matrix-master-00000000"
TIMEOUT = 180
CATALOGS = {
    "openai": ("https://api.openai.com/v1/models", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/models?limit=100", "ANTHROPIC_API_KEY"),
    "xai": ("https://api.x.ai/v1/models", "XAI_API_KEY"),
    "meta": ("https://api.meta.ai/v1/models", "META_API_KEY"),
    "fireworks_ai": ("https://api.fireworks.ai/inference/v1/models", "FIREWORKS_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/models", "OPENROUTER_API_KEY"),
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # shlex handles quoted values and trailing comments without evaluating
        # shell substitutions. Explicit process env takes precedence over .env.
        parts = shlex.split(value, comments=True)
        os.environ.setdefault(key.strip(), " ".join(parts))


def safe(text: str) -> str:
    for key, value in os.environ.items():
        if value and (key.endswith("API_KEY") or key == "OXLLM_MASTER_KEY"):
            text = text.replace(value, "[REDACTED]")
    return text.replace(MASTER, "[REDACTED]")


def request(url: str, body: dict | None = None, headers: dict | None = None,
            timeout: float = TIMEOUT) -> tuple[int, dict, str]:
    hdrs = {"Accept": "application/json", **(headers or {})}
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs)
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024:
            raise AssertionError("response exceeded live-test size limit")
        return response.status, dict(response.headers.items()), raw.decode("utf-8")


def api(base: str, path: str, body: dict | None = None,
        extra_headers: dict | None = None) -> tuple[dict, dict]:
    status, headers, text = request(base + path, body, {
        "Authorization": f"Bearer {MASTER}", **(extra_headers or {}),
    })
    if status != 200:
        raise AssertionError(safe(f"{path}: HTTP {status}: {text[:800]}"))
    return json.loads(text), {key.lower(): value for key, value in headers.items()}


def catalog(provider: str) -> tuple[str, dict]:
    url, env = CATALOGS[provider]
    key = os.environ.get(env, "")
    if not key:
        raise AssertionError(f"{env} is required; this run does not silently skip models")
    headers = ({"x-api-key": key, "anthropic-version": "2023-06-01"}
               if provider == "anthropic" else {"Authorization": f"Bearer {key}"})
    status, _, text = request(url, headers=headers, timeout=45)
    if status != 200:
        raise AssertionError(safe(f"{provider} catalog: HTTP {status}: {text[:400]}"))
    body = json.loads(text)
    if body.get("has_more") or (body.get("links") or {}).get("next"):
        raise AssertionError(f"{provider} catalog is paginated; extend the catalog reader")
    return provider, {model["id"]: model for model in body["data"]}


@dataclass(frozen=True)
class Model:
    alias: str
    provider: str
    model: str
    key_env: str
    reference: str | None
    max_tokens: int = 1024
    tools: bool = True
    response_aliases: tuple[str, ...] = ()


def models(catalogs: dict, all_providers: bool) -> list[Model]:
    # Select a text-capable, paid, non-batch MiniMax model by publication time.
    # Never replace a requested model after a 403/404 with an older success.
    candidates = [model for model in catalogs["openrouter"].values()
                  if model["id"].startswith("minimax/") and ":" not in model["id"]
                  and "text" in model.get("architecture", {}).get("output_modalities", [])
                  and float(model["pricing"]["completion"]) > 0
                  and (not model.get("expiration_date")
                       or model["expiration_date"] > date.today().isoformat())]
    if not candidates:
        raise AssertionError("OpenRouter has no current paid MiniMax text model")
    minimax = max(candidates, key=lambda model: (model["created"], model["id"]))["id"]
    selected = [
        Model("gpt-6-sol", "openai", "gpt-6-sol", "OPENAI_API_KEY", "openai/gpt-6-sol"),
        Model("gpt-6.1-sol", "openai", "gpt-6.1-sol", "OPENAI_API_KEY", "openai/gpt-6.1-sol"),
        Model("gpt-6-luna", "openai", "gpt-6-luna", "OPENAI_API_KEY", "openai/gpt-6-luna"),
        Model("sonnet-5.5", "anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY", "anthropic/claude-sonnet-5.5"),
        Model("grok-4.7", "xai", "grok-4.7", "XAI_API_KEY", "x-ai/grok-4.7"),
        Model("muse-spark-1.3", "meta", "muse-spark-1.3", "META_API_KEY", "meta/muse-spark-1.3", 2048),
        Model("minimax-current", "openrouter", minimax, "OPENROUTER_API_KEY", minimax, 2048),
        Model("glm-5.3-flash", "fireworks_ai", "accounts/fireworks/models/glm-5p3-flash", "FIREWORKS_API_KEY", "z-ai/glm-5.3-flash", 2048),
        Model("glm-5.3-flash-openrouter", "openrouter", "z-ai/glm-5.3-flash", "OPENROUTER_API_KEY", "z-ai/glm-5.3-flash", 2048),
    ]
    if all_providers:
        # Additional providers retain their own serving-provider rate cards;
        # an OpenRouter reseller price is not an independent native price.
        selected.extend([
            Model("gemini", "gemini", "gemini-3.8-flash", "GEMINI_API_KEY", None, tools=False),
            Model("deepseek", "deepseek", "deepseek-v4-flash", "DEEPSEEK_API_KEY", None,
                  tools=False, response_aliases=("deepseek-flash",)),
            # Keep the short smoke below the account's 1,000 output-TPM cap;
            # an oversized requested budget can be rejected before generation.
            Model("groq", "groq", "qwen/qwen3.8-27b", "GROQ_API_KEY", None, max_tokens=128, tools=False),
            Model("perplexity", "perplexity", "sonar", "PERPLEXITY_API_KEY", None, tools=False),
        ])
    for model in selected:
        if not os.environ.get(model.key_env):
            raise AssertionError(f"{model.key_env} required for {model.alias}")
        if model.provider in catalogs and model.model not in catalogs[model.provider]:
            raise AssertionError(f"{model.provider} catalog does not list {model.model}")
    return selected


def close_cost(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-12):
        raise AssertionError(f"{label}: ${actual:.12g} != ${expected:.12g}")


def rate_card(model: Model, catalogs: dict) -> dict:
    table = tomllib.loads((ROOT / "crates/oxllm-core/data/prices" / f"{model.provider}.toml").read_text())
    row = table.get(f"{model.provider}/{model.model}", table.get(model.model))
    if row is None:
        raise AssertionError(f"missing embedded price for {model.provider}/{model.model}")
    if model.reference:
        live = catalogs["openrouter"].get(model.reference)
        if live is None:
            raise AssertionError(f"missing live reference price for {model.reference}")
        prices = live["pricing"]
        for field, remote in [("input_cost_per_token", "prompt"),
                              ("output_cost_per_token", "completion"),
                              ("cache_read_input_token_cost", "input_cache_read")]:
            if remote in prices:
                close_cost(float(row.get(field, row["input_cost_per_token"])),
                           float(prices[remote]), f"{model.alias} {field} vs live catalog")
    return row


def normalized_usage(usage: dict) -> dict:
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    read = details.get("cached_tokens", usage.get("cache_read_input_tokens", 0))
    if "cached_tokens" not in details:
        prompt += usage.get("cache_read_input_tokens", 0)
    write = usage.get("cache_creation_input_tokens", details.get("cache_write_tokens", details.get("cache_creation_tokens", 0)))
    if "cache_creation_input_tokens" not in usage:
        prompt = max(0, prompt - write)
    output_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    return {"prompt": prompt, "completion": completion, "total": prompt + completion,
            "read": read, "write": write,
            "reasoning": output_details.get("reasoning_tokens", output_details.get("thinking_tokens", 0)),
            "write_1h": details.get("ephemeral_1h_input_tokens", (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0)),
            "search": details.get("web_search_requests", usage.get("num_search_queries", usage.get("search_queries", 0))),
            "citation": usage.get("citation_tokens", 0)}


def expected_cost(usage: dict, prices: dict) -> float:
    u = normalized_usage(usage)
    read = min(u["read"], u["prompt"])
    one_hour = min(u["write_1h"], u["write"])
    input_rate, output_rate = prices["input_cost_per_token"], prices["output_cost_per_token"]
    reasoning = min(u["reasoning"], u["completion"])
    return (max(0, u["prompt"] - read) * input_rate
            + read * prices.get("cache_read_input_token_cost", input_rate)
            + (u["write"] - one_hour) * prices.get("cache_creation_input_token_cost", input_rate)
            + one_hour * prices.get("cache_creation_input_token_cost_above_1hr", 0)
            + (u["completion"] - reasoning) * output_rate
            + reasoning * prices.get("output_cost_per_reasoning_token", output_rate)
            + u["search"] * prices.get("search_queries_cost_per_query", 0)
            + u["citation"] * prices.get("citation_cost_per_token", 0))


def receipt_cost(usage: dict) -> float | None:
    value = usage.get("cost")
    if isinstance(value, dict):
        value = value.get("total_cost")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def parse_stream(text: str) -> dict:
    chunks = []
    done = False
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
        if not data:
            continue
        if done:
            raise AssertionError("SSE data after [DONE]")
        if data == "[DONE]":
            done = True
            continue
        chunk = json.loads(data)
        if chunk.get("error"):
            raise AssertionError(safe(f"in-band stream error: {chunk['error']}"))
        chunks.append(chunk)
    if not done or not chunks:
        raise AssertionError("stream lacks parsed data or [DONE]")
    usage_chunks = [chunk["usage"] for chunk in chunks if isinstance(chunk.get("usage"), dict)]
    if not usage_chunks:
        raise AssertionError("stream lacks a usage object")
    content = "".join(choice.get("delta", {}).get("content") or ""
                      for chunk in chunks for choice in chunk.get("choices", []))
    model_ids = {chunk["model"] for chunk in chunks if chunk.get("model")}
    if len(model_ids) != 1:
        raise AssertionError(f"stream model identity is missing or inconsistent: {model_ids}")
    return {"model": model_ids.pop(), "usage": usage_chunks[-1],
            "choices": [{"message": {"content": content}}]}


def parse_native_responses(text: str, streamed: bool) -> dict:
    if streamed:
        completed = []
        for frame in text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if event.get("error") or event.get("type") in {"error", "response.failed", "response.incomplete"}:
                raise AssertionError(safe(f"native Responses stream failure: {event}"))
            if event.get("type") == "response.completed":
                completed.append(event["response"])
        if len(completed) != 1:
            raise AssertionError("native Responses stream lacks exactly one completed response")
        response = completed[0]
    else:
        response = json.loads(text)
    if response.get("status") != "completed" or response.get("error"):
        raise AssertionError(safe(f"native Responses did not complete: {response}"))
    calls = [{"id": item["call_id"], "type": "function", "function": {
        "name": item["name"], "arguments": item["arguments"],
    }} for item in response.get("output", []) if item.get("type") == "function_call"]
    return {"model": response["model"], "usage": response["usage"],
            "choices": [{"message": {"tool_calls": calls}}]}


def command(args: list[str], timeout: int = TIMEOUT) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(safe(f"{args[0]} failed ({result.returncode}): {result.stderr[-1000:]}"))
    return result.stdout.strip()


class Runtime:
    def __init__(self, args, selected: list[Model], output: Path):
        self.args, self.output = args, output
        self.selected = selected
        self.containers: list[str] = []
        self.proxies: list[str] = []
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.network = None
        self.temp = tempfile.TemporaryDirectory(prefix=".live-models-", dir=ROOT)
        self.directory = Path(self.temp.name)
        config = {
            "model_list": [{"model_name": model.alias, "oxllm_params": {
                "model": f"{model.provider}/{model.model}", "api_key": f"os.environ/{model.key_env}",
                "timeout": TIMEOUT, "stream_timeout": TIMEOUT,
            }} for model in selected],
            "general_settings": {"master_key": MASTER, "database_url": "os.environ/DATABASE_URL",
                                 "log_format": "json", "allow_private_api_base": True},
            "oxllm_settings": {"drop_params": True, "num_retries": 0,
                                 "request_timeout": TIMEOUT, "cache_ttl_secs": 300},
        }
        if args.replicas > 1:
            config["oxllm_settings"]["cache"] = {"type": "redis", "redis_url": "redis://redis:6379"}
        self.config = self.directory / "config.yaml"
        # JSON is valid YAML and avoids another host-side dependency.
        self.config.write_text(json.dumps(config))

    def start(self) -> list[str]:
        env = {**os.environ, "RUST_LOG": "info,oxllm_server=info", "OXLLM_MASTER_KEY": MASTER}
        if self.args.image:
            db_url = "sqlite:///data/oxllm.db?mode=rwc"
            if self.args.replicas > 1:
                self.network = f"oxllm-live-{uuid.uuid4().hex[:10]}"
                command(["docker", "network", "create", self.network])
                pg = command(["docker", "run", "-d", "--network", self.network, "--network-alias", "postgres",
                              "-e", "POSTGRES_PASSWORD=live-test", "-e", "POSTGRES_DB=oxllm",
                              "--health-cmd", "pg_isready -U postgres -d oxllm", "--health-interval", "1s",
                              "--health-retries", "40", "postgres:16-alpine"])
                self.containers.append(pg)
                redis = command(["docker", "run", "-d", "--network", self.network,
                                 "--network-alias", "redis", "redis:7-alpine"])
                self.containers.append(redis)
                deadline = time.monotonic() + 60
                while command(["docker", "inspect", "--format", "{{.State.Health.Status}}", pg]) != "healthy":
                    if time.monotonic() >= deadline:
                        raise AssertionError("disposable Postgres did not become healthy")
                    time.sleep(1)
                db_url = "postgres://postgres:live-test@postgres:5432/oxllm"
                command(["docker", "run", "--rm", "--network", self.network, self.args.image,
                         "migrate", "--database-url", db_url])
            env["DATABASE_URL"] = db_url
            env["OXLLM_AUTO_MIGRATE"] = "false" if self.network else "true"
            bases = []
            for _ in range(self.args.replicas):
                cmd = ["docker", "run", "-d", "-p", "127.0.0.1::4000",
                       "-v", f"{self.config}:/config.yaml:ro"]
                if self.network:
                    cmd += ["--network", self.network]
                for name in sorted({model.key_env for model in self.selected} | {
                    "DATABASE_URL", "RUST_LOG", "OXLLM_AUTO_MIGRATE",
                }):
                    cmd += ["-e", name]
                # Docker inherits keys by env name; they are never command-line arguments.
                result = subprocess.run(cmd + [self.args.image, "serve", "--config", "/config.yaml"],
                                        env=env, capture_output=True, text=True, timeout=TIMEOUT)
                if result.returncode:
                    raise RuntimeError(safe(result.stderr))
                cid = result.stdout.strip()
                self.containers.append(cid)
                self.proxies.append(cid)
                port = command(["docker", "inspect", "--format", '{{(index (index .NetworkSettings.Ports "4000/tcp") 0).HostPort}}', cid])
                bases.append(f"http://127.0.0.1:{port}")
        else:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            env["DATABASE_URL"] = f"sqlite:///{self.directory}/spend.db?mode=rwc"
            self.log_file = (self.output / "proxy-0.jsonl").open("w")
            self.process = subprocess.Popen([str(Path(self.args.binary).resolve()), "serve", "--config",
                                             str(self.config), "--port", str(port)], env=env,
                                            stdout=self.log_file, stderr=subprocess.STDOUT)
            bases = [f"http://127.0.0.1:{port}"]
        for base in bases:
            deadline = time.monotonic() + 60
            while True:
                try:
                    status, _, _ = request(base + "/health", timeout=3)
                    if status == 200:
                        break
                except (urllib.error.URLError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError(f"proxy did not become ready: {base}")
                time.sleep(0.25)
        return bases

    def stop(self) -> list[Path]:
        logs = []
        try:
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
                    raise AssertionError("proxy required forced termination")
                self.log_file.close()
                logs.append(self.output / "proxy-0.jsonl")
            for index, cid in enumerate(self.proxies):
                command(["docker", "stop", "--time", "30", cid], timeout=45)
                path = self.output / f"proxy-{index}.jsonl"
                result = subprocess.run(["docker", "logs", cid], capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError(safe(result.stderr))
                path.write_text(result.stdout + result.stderr)
                logs.append(path)
        finally:
            for cid in reversed(self.containers):
                subprocess.run(["docker", "rm", "-f", "-v", cid], capture_output=True, timeout=30)
            if self.network:
                subprocess.run(["docker", "network", "rm", self.network], capture_output=True, timeout=30)
            self.temp.cleanup()
        return logs


def chat_body(model: Model, run_tag: str, prompt: str = "Reply with exactly OK.") -> dict:
    return {"model": model.alias, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": model.max_tokens,
            "metadata": {"tags": [run_tag, f"{run_tag}:{model.alias}", "live-model-matrix"]}}


def exercise(base: str, model: Model, mode: str, run_tag: str, prices: dict,
             prompt: str = "Reply with exactly OK.", cache: bool = False) -> dict:
    tags = [run_tag, f"{run_tag}:{model.alias}", "live-model-matrix"]
    body = chat_body(model, run_tag, prompt)
    if cache:
        body["cache"] = {"insert": True, "use_cache": True}
    native = model.model == "gpt-6.1-sol" and mode in {"tools", "responses-tools-stream"}
    streamed = mode in {"stream", "responses-tools-stream"}
    tool_mode = mode in {"tools", "responses-tools-stream"}
    endpoint = "/v1/chat/completions"
    if mode == "stream":
        body.update(stream=True, stream_options={"include_usage": True})
    if tool_mode:
        body["messages"][0]["content"] = "Call record_result with value exactly OK."
        body["tools"] = [{"type": "function", "function": {"name": "record_result",
                          "description": "Record the test result", "parameters": {
                              "type": "object", "properties": {"value": {"type": "string"}},
                              "required": ["value"], "additionalProperties": False}}}]
        body["tool_choice"] = {"type": "function", "function": {"name": "record_result"}}
        if model.provider == "openai":
            # GPT-6 Chat Completions function tools require disabled reasoning;
            # reasoning+tools is supported through the native Responses API.
            body["reasoning_effort"] = "none"
        elif model.provider in {"anthropic", "meta"}:
            # Sonnet 5.5 and Muse Spark 1.3 reject forced tool choice. Exercise
            # their supported auto contract and still require a real tool call.
            body["tool_choice"] = "auto"
    if native:
        # GPT-6.1 Sol cannot disable reasoning and rejects Chat tools. Use the
        # provider-native route, not /v1/responses' Chat compatibility bridge.
        endpoint = "/openai/v1/responses"
        body = {"model": model.alias, "input": "Call record_result with value exactly OK.",
                "max_output_tokens": 2048, "reasoning": {"effort": "low"}, "store": False,
                "tools": [{"type": "function", **body["tools"][0]["function"]}],
                "tool_choice": {"type": "function", "name": "record_result"},
                "metadata": {"tags": tags}, "stream": streamed}
    status, headers, text = request(base + endpoint, body, {
        "Authorization": f"Bearer {MASTER}", "x-oxllm-tag": run_tag,
    })
    if status != 200:
        raise AssertionError(safe(f"{model.alias}/{mode}: HTTP {status}: {text[:800]}"))
    headers = {key.lower(): value for key, value in headers.items()}
    response = (parse_native_responses(text, streamed) if native
                else parse_stream(text) if streamed else json.loads(text))
    if response.get("error"):
        raise AssertionError(safe(f"{model.alias}/{mode}: {response['error']}"))
    served = response.get("model", "")
    if served != model.model and served not in model.response_aliases and not served.startswith(model.model + "-20"):
        raise AssertionError(f"{model.alias}/{mode}: unexpected served model {served}")
    message = response["choices"][0]["message"]
    if tool_mode:
        calls = message.get("tool_calls") or []
        if not calls or not calls[0].get("id") or calls[0]["function"]["name"] != "record_result":
            raise AssertionError(f"{model.alias}: invalid forced tool call")
        if json.loads(calls[0]["function"]["arguments"]).get("value") != "OK":
            raise AssertionError(f"{model.alias}: unexpected tool arguments")
    elif not (message.get("content") or "").strip():
        raise AssertionError(f"{model.alias}/{mode}: no visible assistant text")
    usage = response["usage"]
    normalized = normalized_usage(usage)
    if normalized["prompt"] <= 0 or normalized["completion"] <= 0:
        raise AssertionError(f"{model.alias}/{mode}: no actual token usage")
    request_id = headers.get("x-oxllm-request-id")
    if not request_id or headers.get("x-request-id") != request_id:
        raise AssertionError(f"{model.alias}/{mode}: missing/mismatched request-id headers")
    uuid.UUID(request_id)
    expected = expected_cost(usage, prices)
    reported = receipt_cost(usage)
    if model.provider == "perplexity" and reported is not None:
        expected = reported
    cost_header = headers.get("x-oxllm-response-cost")
    if streamed:
        # Final usage is not known when stream headers are committed. Verify
        # settlement in the ledger/log below, never invent header parity.
        if cost_header is not None:
            raise AssertionError(f"{model.alias}: streaming header claims a premature final cost")
    else:
        if cost_header is None:
            raise AssertionError(f"{model.alias}/{mode}: missing priced-response cost header")
        close_cost(float(cost_header), expected, f"{model.alias}/{mode} response cost")
    preview, _ = api(base, "/spend/calculate", {"model": model.alias, "provider": model.provider, "usage": usage})
    close_cost(preview["cost"], expected, f"{model.alias}/{mode} preview cost")
    print(f"PASS {model.provider}/{model.model} {mode}: tokens={normalized['prompt']}+{normalized['completion']} ${expected:.9f}", flush=True)
    reported_mode = "responses-tools" if native and not streamed else mode
    return {"alias": model.alias, "provider": model.provider, "model": served, "mode": reported_mode,
            "endpoint": endpoint,
            "request_id": request_id, "usage": usage, "cost": expected, "tags": tags,
            "tag_name": run_tag, "reference": model.reference}


def verify_cache_and_rejections(bases: list[str], model: Model, prices: dict,
                                run_tag: str, records: list[dict]) -> None:
    prompt = f"Cache probe {uuid.uuid4().hex}. Reply with exactly OK."
    prime = exercise(bases[0], model, "cache-prime", run_tag, prices, prompt, cache=True)
    records.append(prime)
    body = chat_body(model, run_tag, prompt)
    body["cache"] = {"insert": True, "use_cache": True}
    cached, headers = api(bases[-1], "/v1/chat/completions", body, {"x-oxllm-tag": run_tag})
    if headers.get("x-oxllm-cache-hit") != "true":
        raise AssertionError("repeat request was not served from the response cache")
    # Cache-hit headers describe the original response's cost; they are not a
    # new debit. The request must produce neither a ledger row nor a spend log.
    close_cost(float(headers["x-oxllm-response-cost"]), prime["cost"], "cached response cost")
    if cached["model"] != prime["model"]:
        raise AssertionError("cached serving-model identity changed")
    cached_id = headers["x-oxllm-request-id"]
    if cached_id == prime["request_id"]:
        raise AssertionError("cache hit reused the original request ID")
    for base in bases:
        for auth, expected in [(None, 401), ("Bearer sk-live-definitely-invalid", 401)]:
            status, _, _ = request(base + "/v1/chat/completions", chat_body(model, run_tag),
                                   {} if auth is None else {"Authorization": auth})
            if status != expected:
                raise AssertionError(f"unauthorized request returned {status} instead of {expected}")
        missing = chat_body(model, run_tag)
        missing["model"] = "__live_model_not_configured__"
        status, _, _ = request(base + "/v1/chat/completions", missing,
                               {"Authorization": f"Bearer {MASTER}"})
        if status != 404:
            raise AssertionError(f"unconfigured model returned {status} instead of 404")
    print("PASS response cache and auth/model rejections (checked for extra ledger rows below)", flush=True)


def rollup(base: str, endpoint: str) -> list[dict]:
    rows, offset = [], 0
    while True:
        page, _ = api(base, f"{endpoint}?limit=500&offset={offset}")
        data = page["data"]
        rows.extend(data)
        if len(data) < 500:
            return rows
        offset += len(data)


def verify_ledger(bases: list[str], records: list[dict], run_tag: str) -> None:
    deadline = time.monotonic() + 90
    query = "/spend/logs?" + urllib.parse.urlencode({"tag_name": run_tag, "limit": 500})
    while True:
        page, _ = api(bases[0], query)
        if page["total_requests"] >= len(records):
            break
        if time.monotonic() >= deadline:
            raise AssertionError(f"durable spend rows missing: {page['total_requests']}/{len(records)}")
        time.sleep(0.5)
    expected_ids = {record["request_id"] for record in records}
    if len(expected_ids) != len(records):
        raise AssertionError("duplicate request IDs across live calls")
    by_id = {record["request_id"]: record for record in records}
    totals = defaultdict(lambda: [0, 0.0])
    model_totals = defaultdict(lambda: [0, 0.0])
    for record in records:
        model_totals[(record["model"], record["provider"])][0] += 1
        model_totals[(record["model"], record["provider"])][1] += record["cost"]
        for tag in record["tags"]:
            totals[tag][0] += 1
            totals[tag][1] += record["cost"]
    for base in bases:
        page, _ = api(base, query)
        if page["total_requests"] != len(records) or len(page["data"]) != len(records):
            raise AssertionError(f"{base}: missing or duplicate ledger rows")
        if {row["request_id"] for row in page["data"]} != expected_ids:
            raise AssertionError(f"{base}: ledger/request ID mismatch")
        for row in page["data"]:
            record = by_id[row["request_id"]]
            if row["status"] != "success" or row["model"] != record["model"] or row["provider"] != record["provider"]:
                raise AssertionError(f"{base}: incorrect served-model/provider/status attribution")
            tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
            if tags != record["tags"] or row["tag_name"] != run_tag:
                raise AssertionError(f"{base}: lost, reordered or duplicate tags")
            usage = normalized_usage(record["usage"])
            for field, source in [("prompt_tokens", "prompt"), ("completion_tokens", "completion"), ("total_tokens", "total")]:
                if row[field] != usage[source]:
                    raise AssertionError(f"{base}: incorrect logged {field}")
            close_cost(row["cost"], record["cost"], "ledger cost")
        close_cost(page["total_cost"], sum(record["cost"] for record in records), "ledger total")
        aggregates = {row["tag_name"]: row for row in rollup(base, "/spend/tags")}
        for tag, (count, cost) in totals.items():
            row = aggregates.get(tag)
            if row is None or row["total_requests"] != count:
                raise AssertionError(f"{base}: incorrect aggregate for {tag}")
            close_cost(row["total_spend"], cost, f"{base} aggregate {tag}")
            # This also checks filtering on a non-primary body tag; a filter
            # that only examined legacy tag_name would silently lose these rows.
            if tag.startswith(run_tag + ":"):
                filtered, _ = api(base, "/spend/logs?" + urllib.parse.urlencode({"tag_name": tag, "limit": 500}))
                if filtered["total_requests"] != count:
                    raise AssertionError(f"{base}: non-primary tag filter loses requests")
                close_cost(filtered["total_cost"], cost, "non-primary tag filter spend")
        by_model = {(row["model"], row["provider"]): row for row in rollup(base, "/spend/by-model")}
        for key, (count, cost) in model_totals.items():
            row = by_model.get(key)
            if row is None or row["total_requests"] != count:
                raise AssertionError(f"{base}: incorrect served model/provider rollup {key}")
            close_cost(row["total_spend"], cost, "served model/provider rollup")
        status, _, metrics = request(base + "/metrics", headers={"Authorization": f"Bearer {MASTER}"})
        if status != 200 or "oxllm_requests_total" not in metrics:
            raise AssertionError(f"{base}: missing metrics exposition")
        drops = [float(line.split()[-1]) for line in metrics.splitlines()
                 if line.startswith("oxllm_spend_entries_dropped_total")]
        if any(drops):
            raise AssertionError(f"{base}: dropped spend entries")
    print(f"PASS ledger and all-tag aggregates: {len(records)} rows across {len(bases)} replica(s)", flush=True)


def verify_logs(paths: list[Path], records: list[dict]) -> None:
    logs = "\n".join(path.read_text() for path in paths)
    secrets = [value for key, value in os.environ.items()
               if value and (key.endswith("API_KEY") or key == "OXLLM_MASTER_KEY")]
    if any(secret in logs for secret in secrets + [MASTER]):
        raise AssertionError("a credential appeared in server logs")
    events = []
    for line in logs.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if "panic" in line.lower():
                raise AssertionError("panic in server logs")
            continue
        if event.get("level") == "ERROR":
            raise AssertionError(safe(f"server ERROR: {event.get('fields', {})}"))
        if event.get("fields", {}).get("event") == "spend":
            events.append(event["fields"])
    by_id = defaultdict(list)
    for event in events:
        by_id[event.get("request_id")].append(event)
    if len(events) != len(records) or set(by_id) != {record["request_id"] for record in records}:
        raise AssertionError("extra/missing spend events (including a cache hit or rejected request)")
    for record in records:
        matches = by_id[record["request_id"]]
        if len(matches) != 1:
            raise AssertionError(f"expected one structured spend event for {record['alias']}/{record['mode']}")
        event = matches[0]
        if event["model"] != record["model"] or event["provider"] != record["provider"] or not event["cost_known"]:
            raise AssertionError("incorrect structured spend attribution")
        if event.get("tag_name") != record["tag_name"]:
            raise AssertionError("structured primary tag differs from the ledger")
        if json.loads(event["tags"]) != record["tags"]:
            raise AssertionError("structured spend tags differ from durable tags")
        close_cost(event["cost"], record["cost"], "structured spend cost")
    if "duplicate pricing key" in logs or "request has no peer address" in logs:
        raise AssertionError("stale/misleading startup or auth-brake diagnostic")
    print(f"PASS structured spend logs: {len(records)} correlated events; no credential leaks/errors", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=str(ROOT / "target/debug/oxllm"))
    parser.add_argument("--image", help="Docker image built from the current checkout")
    parser.add_argument("--replicas", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, help="directory for retained evidence")
    parser.add_argument("--all-providers", action="store_true", help="also cover Gemini, DeepSeek, Groq and Perplexity")
    parser.add_argument("--tools", action="store_true", help="verify function calls using each model's supported tool-choice contract")
    parser.add_argument("--model", action="append", help="restrict to one or more test aliases (repeatable)")
    parser.add_argument("--catalog-only", action="store_true", help="verify model IDs and current price references without inference")
    args = parser.parse_args()
    if args.replicas > 1 and not args.image:
        parser.error("--replicas > 1 requires --image (managed shared Postgres/Redis)")
    load_env(args.env_file)
    output = args.output or Path(tempfile.mkdtemp(prefix="oxllm-live-models-"))
    output.mkdir(parents=True, exist_ok=True)
    print(f"Evidence: {output}", flush=True)
    runtime = None
    records, failures = [], []
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            catalogs = dict(pool.map(catalog, CATALOGS))
        selected = models(catalogs, args.all_providers)
        if args.model:
            unknown = set(args.model) - {model.alias for model in selected}
            if unknown:
                raise AssertionError(f"unknown requested test aliases: {sorted(unknown)}")
            selected = [model for model in selected if model.alias in args.model]
        prices = {model.alias: rate_card(model, catalogs) for model in selected}
        for model in selected:
            print(f"CATALOG {model.provider}/{model.model}: priced ({'live reference' if model.reference else 'embedded serving rate'})", flush=True)
        if args.catalog_only:
            return 0
        runtime = Runtime(args, selected, output)
        bases = runtime.start()
        run_tag = f"live-models-{uuid.uuid4().hex[:16]}"
        for model in selected:
            modes = ["plain", "stream"] + (["tools"] if args.tools and model.tools else [])
            if args.tools and model.model == "gpt-6.1-sol":
                modes.append("responses-tools-stream")
            for mode in modes:
                try:
                    records.append(exercise(bases[len(records) % len(bases)], model, mode, run_tag, prices[model.alias]))
                except Exception as exc:
                    failure = safe(f"{model.alias}/{mode}: {type(exc).__name__}: {exc}")
                    failures.append(failure)
                    print(f"FAIL {failure}", flush=True)
        if not failures:
            cache_model = next((model for model in selected if model.alias == "gpt-6-luna"), selected[0])
            verify_cache_and_rejections(bases, cache_model, prices[cache_model.alias], run_tag, records)
            verify_ledger(bases, records, run_tag)
        paths = runtime.stop()
        runtime = None
        if not failures:
            verify_logs(paths, records)
    except Exception as exc:
        failures.append(safe(f"{type(exc).__name__}: {exc}"))
    finally:
        if runtime is not None:
            try:
                runtime.stop()
            except Exception as exc:
                failures.append(safe(f"cleanup failed: {exc}"))
        (output / "result.json").write_text(json.dumps({"records": records, "failures": failures}, indent=2))
    for failure in failures:
        print(f"FAIL {failure}", flush=True)
    print(f"RESULT: {len(records)} verified live responses, {len(failures)} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
