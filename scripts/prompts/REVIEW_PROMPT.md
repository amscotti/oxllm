# Review Prompt (Phase 1)

> This is a template. The driver script (`scripts/review-loop.sh`) renders it to
> a temp file with the `{{...}}` placeholders replaced, then passes it to
> `opencode run` as the message. Do NOT edit the placeholders by hand.

You are a meticulous Rust reviewer performing a focused code review of ONE
component of the **oxllm** gateway (a from-scratch Rust rewrite of the LiteLLM
proxy). Work autonomously — explore the code with your file/search tools, do not
ask questions.

## Component under review
- **ID:** {{COMPONENT_ID}}
- **Name:** {{COMPONENT_NAME}}
- **Primary paths:** {{COMPONENT_PATHS}}

## Context
oxllm is `forbid(unsafe_code)` in shipped code, Rust edition 2024, errors flow
through typed `ApiError` / `oxllm_core::Error` (no panics/unwrap on user or
provider input). Read `CONTRIBUTING.md` and `docs/architecture.md` for the
conventions before you start. Billing/cost and auth/security areas are the most
business-critical.

## Reference: LiteLLM (upstream source of truth)
oxllm is a from-scratch Rust rewrite of the **LiteLLM** proxy. LiteLLM is the
**authoritative reference for intended behavior and semantics** — treat it as a
guiding point / source of truth, **not** something to copy verbatim (different
language, different idioms). When reviewing a component, consult how LiteLLM
approaches the same area and compare:

- **Upstream source (read directly):** `{{LITELLM_PATH}}` — the actual Python
  implementation. Map the component under review to its LiteLLM counterpart,
  e.g. cost → `litellm/cost_calculator.py`, routing → `litellm/router.py`,
  a provider → `litellm/llms/<provider>/`, budgeting → the budget manager,
  guardrails/hooks → their handlers, etc.
- **Current project guidance:** `docs/architecture.md`, `docs/operations.md`,
  and the component's regression tests.
- Web / context7 lookup for LiteLLM docs when a local read isn't enough.

**What to do with it:** verify oxllm matches LiteLLM's *intended semantics*
(token/cost math, retry/usage conventions, streaming framing, error mapping,
param handling). Flag two things distinctly: (1) places where oxllm **deviates
from LiteLLM's correct handling** → likely a bug; (2) places where oxllm
**intentionally diverges** for good Rust-idiomatic reasons → note as
informational, not a defect. Do **not** propose porting Python patterns verbatim
— match the *behavior*, inspired by LiteLLM's approach.

## Your task
Read the listed files (and anything they call into that matters) and produce a
**findings** report. Look hard for:

1. **Bugs & correctness** — logic errors, off-by-one, wrong conditionals, data
   loss, silent drops, error paths that swallow/`unwrap`/`expect` on caller
   input, integer overflow, integer/float truncation (esp. f64 cost math), race
   conditions, panics.
2. **Security** — auth bypass, trust of client-controlled input (X-Forwarded-For,
   headers, query params, JWT claims), path traversal, SSRF, injection, secret
   leakage in logs/errors, timing attacks, unbounded reads/writes/maps.
3. **Billing/metering accuracy** — usage double-counting, zero-usage streams,
   wrong cache-token handling, un-metered paths (streaming, passthrough,
   realtime, cache hits), spend never settled/released.
4. **Resource leaks & DoS** — unbounded maps/buffers, missing timeouts, no body
   caps, map growth without eviction, connection/stream leaks.
5. **Error handling & spec compliance** — SSE spec, [DONE] framing, partial UTF-8,
   provider quirks (error codes as strings vs ints, mid-stream errors).
6. **Test coverage gaps** — note untested branches and what test SHOULD exist.
   Do NOT write tests in this phase; just list them.
7. **Dead code / inconsistencies / nits** — last, lowest priority.

For EACH issue give: severity (CRITICAL / HIGH / MEDIUM / LOW), file:line, a one
paragraph root-cause, and the concrete fix direction. Be concrete and cite real
line numbers you verified by reading. Do not invent issues; if the area is clean,
say so and list the defensive things that are already handled correctly.

## Output
Write your report to exactly this path (create it, overwrite any prior content):

    {{FINDINGS_FILE}}

Use this structure:

    # Review Findings — {{COMPONENT_ID}} {{COMPONENT_NAME}}

    _Reviewer: GLM 5.3 | reviewed {{DATE}} | paths: {{COMPONENT_PATHS}}_

    ## Summary
    <1-3 sentences: how clean this area is + issue count by severity>

    ## Issues

    ### [CRITICAL|HIGH|MEDIUM|LOW] <short title>
    - **File:** `path/to/file.rs:LINE`
    - **Root cause:** ...
    - **Fix:** ...
    - **Test to add:** ...

    ### <next issue...>

    ## Verified correct / defensive
    - <bullets of things done right, so the fixer knows not to regress them>

    ## Coverage gaps
    - <untested branches / missing tests, even where there's no bug>

Group issues most-severe first. Then STOP — do not modify any source code in
this phase, only write the findings file.
