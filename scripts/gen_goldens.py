#!/usr/bin/env python3
"""Compare LiteLLM transforms with oxllm provider fixture cases.

Reads LiteLLM's provider transform classes (one directory up) and prints
input/expected pairs. The current Anthropic and OpenAI goldens are manually
curated (see their VERSION files); this script must not overwrite them or
claim they were generated from a LiteLLM commit.

``--write`` is allowed only for an existing fixture directory whose VERSION
already records ``litellm-commit: <40-hex SHA>``. This provenance check runs
for every selected provider before any file is written. CI does not run it.

Usage (from oxllm repo root)::

    # Dry-run: print expected JSON to stdout
    uv run --directory ../litellm python ../oxllm/scripts/gen_goldens.py --provider anthropic

    # Only for explicitly LiteLLM-generated fixtures; current manual fixtures refuse this
    uv run --directory ../litellm python ../oxllm/scripts/gen_goldens.py --provider anthropic --write

    uv run --directory ../litellm python ../oxllm/scripts/gen_goldens.py --provider all --write

Requires: LiteLLM repo at ``../litellm`` with ``uv`` environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

OXLLM_ROOT = Path(__file__).resolve().parent.parent
LITELLM_ROOT = OXLLM_ROOT.parent / "litellm"
PROVIDERS_ROOT = OXLLM_ROOT / "crates" / "oxllm-providers" / "src"


def _litellm_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=LITELLM_ROOT,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _split_openai_request(body: dict[str, Any]) -> tuple[str, list, dict]:
    """OpenAI chat body → (model, messages, optional_params)."""
    model = body["model"]
    messages = body["messages"]
    optional = {k: v for k, v in body.items() if k not in ("model", "messages")}
    return model, messages, optional


def _anthropic_cases() -> list[tuple[str, dict, Callable[..., dict]]]:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    cfg = AnthropicConfig()

    def transform(body: dict) -> dict:
        model, messages, optional = _split_openai_request(body)
        optional.setdefault("max_tokens", 4096)
        return cfg.transform_request(
            model=model,
            messages=messages,
            optional_params=optional,
            litellm_params={"custom_llm_provider": "anthropic"},
            headers={},
        )

    return [
        (
            "chat_request_001",
            {
                "model": "claude-3-5-sonnet",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                "temperature": 0.7,
                "max_tokens": 256,
                "stream": False,
            },
            transform,
        ),
        (
            "chat_request_002_tools",
            {
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "What's the weather in Boston?"}],
                "max_tokens": 1024,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
            transform,
        ),
    ]


def _openai_cases() -> list[tuple[str, dict, Callable[..., dict]]]:
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

    cfg = OpenAIGPTConfig()

    def transform(body: dict) -> dict:
        model, messages, optional = _split_openai_request(body)
        return cfg.transform_request(
            model=model,
            messages=messages,
            optional_params=optional,
            litellm_params={"custom_llm_provider": "openai"},
            headers={},
        )

    return [
        (
            "chat_request_001",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.5,
            },
            transform,
        ),
        (
            "chat_request_002",
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 16,
            },
            transform,
        ),
    ]


PROVIDERS: dict[str, Callable[[], list]] = {
    "anthropic": _anthropic_cases,
    "openai": _openai_cases,
}


def _validate_write_target(provider: str) -> None:
    version = PROVIDERS_ROOT / provider / "goldens" / "VERSION"
    try:
        provenance = version.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"refusing to write {provider}: no generated-fixture VERSION") from exc
    if not re.fullmatch(r"litellm-commit: [0-9a-f]{40}", provenance):
        raise ValueError(
            f"refusing to overwrite {provider} fixtures: VERSION does not identify "
            "LiteLLM-generated fixtures; use the dry-run output for manual review"
        )


def _write_golden(provider: str, stem: str, payload: dict, write: bool) -> None:
    out_dir = PROVIDERS_ROOT / provider / "goldens"
    path = out_dir / f"{stem}.json"
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if write:
        _validate_write_target(provider)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(OXLLM_ROOT)}")
    else:
        print(f"--- {provider}/{stem}.json ---")
        print(text)


def _write_version(provider: str, write: bool) -> None:
    sha = _litellm_git_sha()
    line = f"litellm-commit: {sha}\n"
    path = PROVIDERS_ROOT / provider / "goldens" / "VERSION"
    if write:
        _validate_write_target(provider)
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("refusing to write fixtures without a known LiteLLM source commit")
        path.write_text(line, encoding="utf-8")
        print(f"wrote {path.relative_to(OXLLM_ROOT)}")
    else:
        print(f"VERSION would be: {line.strip()}")


def generate(provider: str, write: bool) -> int:
    if provider not in PROVIDERS:
        print(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}", file=sys.stderr)
        return 1

    if write:
        _validate_write_target(provider)
        if not re.fullmatch(r"[0-9a-f]{40}", _litellm_git_sha()):
            raise ValueError("refusing to write fixtures without a known LiteLLM source commit")
    cases = PROVIDERS[provider]()
    for stem, input_body, transform in cases:
        # Some upstream transforms mutate messages/optional values in place.
        # The fixture's input must remain the original client request.
        expected = transform(copy.deepcopy(input_body))
        payload = {"input": input_body, "expected_request": expected}
        _write_golden(provider, stem, payload, write)

    _write_version(provider, write)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        required=True,
        choices=[*PROVIDERS.keys(), "all"],
        help="Provider slug or 'all'",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write only to an existing LiteLLM-generated fixture directory (default: dry-run)",
    )
    args = parser.parse_args()

    providers = sorted(PROVIDERS) if args.provider == "all" else [args.provider]
    if args.write:
        try:
            for provider in providers:
                _validate_write_target(provider)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if not LITELLM_ROOT.is_dir():
        print(f"LiteLLM repo not found at {LITELLM_ROOT}", file=sys.stderr)
        return 1

    # Ensure litellm is importable when invoked outside its venv cwd.
    sys.path.insert(0, str(LITELLM_ROOT))

    rc = 0
    try:
        for name in providers:
            rc |= generate(name, args.write)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
