# Fix Prompt (Phase 2)

> This is a template. The driver script (`scripts/review-loop.sh`) renders it
> with `{{...}}` replaced and passes it to `opencode run`. Do not edit
> placeholders by hand.

You are a senior Rust engineer fixing real issues in the **oxllm** gateway. Work
autonomously with `--auto`; do not ask questions.

## Component
- **ID:** {{COMPONENT_ID}}
- **Name:** {{COMPONENT_NAME}}
- **Primary paths:** {{COMPONENT_PATHS}}

## Input
Read the findings report produced by the review phase:

    {{FINDINGS_FILE}}

Work through every issue in that report, **most severe first**.

## Reference: LiteLLM (upstream source of truth)
oxllm is a Rust rewrite of **LiteLLM**, which is the authoritative reference for
intended behavior. When a fix is about *semantics* (cost/usage math, retry
conventions, streaming framing, error mapping, param handling), consult how
LiteLLM does it before implementing:
- **Upstream source:** `{{LITELLM_PATH}}` (e.g. `litellm/cost_calculator.py`,
  `litellm/router.py`, `litellm/llms/<provider>/`).
- **Project guidance:** `docs/architecture.md`, `docs/operations.md`, and the
  component's regression tests.
Match LiteLLM's *behavior* — do **not** port Python patterns verbatim. Keep the
fix Rust-idiomatic and within this project's conventions (below).

## Rules (the project enforces these — violations break CI)
- `forbid(unsafe_code)` in shipped code. Never add `unsafe`.
- Edition 2024. Do not bump any crate versions.
- No `//` comments unless absolutely necessary; the codebase is comment-light.
  Doc-comments (`///`) on public items are encouraged.
- Errors flow through typed `ApiError` / `oxllm_core::Error`. **Never** `panic!` /
  `unwrap` / `expect` / `todo!` on user or provider input. (`unwrap` on a value
  the code itself just constructed is fine.)
- Prefer the smallest change that is correct. Follow existing patterns in the
  surrounding module; look at neighbours before inventing helpers.
- Do not touch unrelated files. Keep changes within this component unless an
  issue's root cause genuinely lives elsewhere (then minimal, surgical edit).

## For each issue
1. Implement the fix described in the report (adjust if, on closer reading, the
   report is wrong — but say what you changed and why in your final summary).
2. Add the test the report asked for (or a better one), placed consistently with
   existing test layout (`#[cfg(test)] mod tests` in the file, or the matching
   `tests/` integration suite). Golden-file providers: update goldens only if a
   test legitimately changed shape, never to mask a bug.

## Before you finish — MANDATORY verification
Run these from the repo root and make sure they pass. Fix any breakage you
introduce before stopping:

    cargo fmt --all
    cargo clippy --workspace --lib --bins --tests -- -D warnings
    cargo test --workspace

If the full workspace test run is too slow, at minimum run the crate(s) you
touched: `cargo test -p <crate>`. Report what you ran.

## When all issues are handled — stop
Do **not** edit `.review/checklist.md`. The driver script owns the checklist and
the commits: once you finish, it commits your changes and marks the component
done itself. If you cannot fully resolve a CRITICAL/HIGH issue, say so plainly
in your final summary and leave the code in the best state you can — the driver
still records your work, and the gap is visible in your summary for a human to
follow up.

## Final output
Print a concise summary: for each issue, one line — `issue title → fixed (file)` or
`issue title → DEFERRED (reason)`, plus the exact `cargo` commands you ran and
their pass/fail status.
