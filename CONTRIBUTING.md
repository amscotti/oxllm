# Contributing to oxllm

Thanks for considering a contribution! oxllm is a from-scratch Rust rewrite
of the LiteLLM proxy gateway. The codebase is intentionally small and
opinionated; this guide lays out the conventions.

> See [docs/architecture.md](docs/architecture.md) for how the crates fit
> together, and [docs/adding-a-provider.md](docs/adding-a-provider.md) for the
> provider extension checklist.

---

## Development environment

Requirements:

- Rust **1.96** (see [rust-toolchain.toml](rust-toolchain.toml) for the
  pinned channel; Docker builder images use `rust:1.96-bookworm`). Criterion
  benchmarks compile on this stable toolchain; nightly is an extra CI check.
- (optional) `cargo-deny` for licence/CVE checks.
- (optional) `cargo-llvm-cov` for coverage.
- (optional, for the `admin` feature) a local stack via
  [scripts/dev-stack.sh](scripts/dev-stack.sh).

Initial build & test:

```bash
cargo build --locked --workspace
cargo test --locked --workspace
```

The first build downloads and compiles all dependencies (including `tiktoken-rs`
data files); expect a few minutes.

---

## Code style

- **Rust edition 2024** across every crate. Do not bump individual crates.
- **Explain non-obvious intent.** Prefer clear names and small functions;
  preserve comments that explain protocol behavior, concurrency invariants or
  recovery semantics. Public items should have useful doc-comments (`///`).
- **Follow existing patterns.** Before adding a new helper, look at how the
  neighbouring module does it. In particular:
  - Providers are pure transforms; the host owns all I/O (see
    [docs/architecture.md](docs/architecture.md)).
  - Errors flow through typed `ApiError` / `oxllm_core::Error`; no panics on
    user or provider input.
  - `serde_json::Value` lives at boundaries; inside the pipeline, prefer typed
    structs.
- **`forbid(unsafe_code)`** is enforced in `oxllm-core` and the wider
  workspace. Do not add `unsafe`.
- **Format & lint before pushing:**

  ```bash
  cargo fmt --all -- --check
   cargo clippy --locked --workspace --all-targets -- -D warnings
   cargo deny --locked check    # optional locally, licences + advisories
  ```

   CI runs these checks on PRs; the primary jobs use the pinned toolchain.

---

## Adding a new provider

Adding a provider is mechanical and intentionally scoped — **a provider
module, a feature flag, and registry wiring**. A new price shard and optional
passthrough support are data-only edits; no handler or route changes are
needed. Follow the full checklist in
[docs/adding-a-provider.md](docs/adding-a-provider.md).

The short version:

1. Add `crates/oxllm-providers/src/<provider>/mod.rs` implementing the
   relevant traits (`ChatProvider`, `EmbeddingProvider`, …).
2. Add `"<provider>" = []` to `crates/oxllm-providers/Cargo.toml`
   `[features]`.
3. Wire the module in `crates/oxllm-providers/src/lib.rs` and register it
   in the default registry builder.
4. Add unit tests + golden files. If the provider is OpenAI-compatible,
   reuse `openai_compat` by declaring `"<provider>" = ["openai_compat"]` in
   the provider crate's features; no separate `cfg(any(...))` list is needed.

---

## Running tests

```bash
# Everything
cargo test --locked --workspace

# Just the doc tests for the core crate
cargo test --locked --doc -p oxllm-core

# A specific integration suite
cargo test --locked --test chat_e2e
cargo test --locked --test router_e2e
cargo test --locked --test management_e2e    # DB is enabled by the root dev-dependency

# Feature combinations (mirrors .github/workflows/ci.yml)
cargo test --locked --workspace --features db
cargo test --locked --workspace --features otel
cargo test --locked --workspace --features admin
cargo test --locked --workspace --no-default-features
cargo bench --locked --workspace --no-run
```

Lint, format, and doc build run as separate CI jobs:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo doc --locked --no-deps --workspace
```

## Coverage

Coverage is tracked with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov)
(`cargo install cargo-llvm-cov`). CI does not enforce a fixed percentage.
Measure the current tree and include the commit and date when reporting a
number; uncovered lines are typically concentrated in startup/shutdown wiring
and rare error branches.

```bash
# Fast signal: library/unit tests only, per-file JSON + terminal summary.
cargo llvm-cov --locked --features db --workspace --lib --no-fail-fast \
  --ignore-filename-regex '(^|/)(tests|benches)(/|$)'

# Full signal: unit + integration suites (needs the Redis/Postgres env vars
# above for the gated tests; takes substantially longer).
cargo llvm-cov --locked --features db --workspace --tests --no-fail-fast \
  --ignore-filename-regex '(^|/)(tests|benches)(/|$)'

# Merge coverage across separate runs (e.g. lib first, then a few e2e suites):
cargo llvm-cov --locked --features db --workspace --tests --no-clean \
  --test management_e2e --test router_e2e --no-fail-fast
```

Prefer adding tests for behavior that can regress silently — money paths
(streaming settlement, budgets, spend accounting), security checks, provider
wire contracts, and routing constraints. Tests that only mirror the
implementation or assert constants are worse than no test.

Most integration tests boot the proxy in-process against
[`httpmock`](https://docs.rs/httpmock) upstreams. Explicit Redis/Postgres fixture
tests also use local services, and live-provider tests have their own opt-ins.
The shared harness lives in [tests/common/mod.rs](tests/common/mod.rs) —
use `common::start_oxllm(...)` when adding a new integration test rather than
re-rolling the Axum boilerplate.

Redis CI sets `OXLLM_REDIS_URL=redis://127.0.0.1:6379`; without it, the gated
cross-replica tests skip. Postgres CI supplies `OXLLM_TEST_POSTGRES_URL` and runs
`oxllm-db` with `postgres,sqlite`. Use disposable local services for these tests.
Deployment CI also renders Helm/Compose, builds both images, and checks runtime
health/auth/SQLite/admin assets with synthetic credentials and no provider calls.

### Current-model live verification

`scripts/test-live-models.py` (Python 3.11+, standard library only) loads provider
keys from `.env` or the process environment, verifies exact model IDs against
live catalogs, selects the newest paid MiniMax text model on OpenRouter, and
checks the requested newer models' embedded rates against live reference prices.
It exercises plain and completed SSE responses, optional function-tool calls,
request-ID/model/provider attribution, deduplicated header/body tags (body-first
primary-tag ordering), response-cache behavior,
auth/model rejection, cost previews, durable spend rows, per-tag aggregates, and
correlated structured spend logs. Configured failures are failures, not skips.

```sh
cargo build --locked --features db --bin oxllm
python3 scripts/test-live-models.py --binary target/debug/oxllm --tools

docker build -t oxllm:live .
python3 scripts/test-live-models.py --image oxllm:live --all-providers --tools
python3 scripts/test-live-models.py --image oxllm:live --replicas 2 --tools
```

Docker replica runs create disposable shared Postgres/Redis services. Managed
resources are removed on exit; evidence remains in the printed output directory
(or `--output /path/outside/the/repository`). `--catalog-only` makes no inference
calls. Stream cost is checked after settlement in the ledger/log: initial SSE
headers cannot claim a final usage-dependent price. Cache-hit headers retain
the original response cost; the test verifies that no new spend row is created.
The four additional provider smoke models selected by `--all-providers` use
their embedded serving-provider cards rather than reseller prices.
Perplexity's complete receipts retain dynamic request/search fees in settlement
and previews. OpenRouter's published rate-card cost can differ from account
discounts in its `usage.cost` receipt; explicit deployment overrides still win.
Tool choice follows the verified model contract: GPT-6 Sol/Luna Chat tools use
`reasoning_effort: none`; Sonnet 5.5 and Muse Spark 1.3 use automatic choice.
GPT-6.1 Sol cannot disable reasoning, so its tool cases exercise unary and SSE
native Responses through `/openai/v1/responses`. `/v1/responses` is the Chat
compatibility bridge. Native cached/write/reasoning buckets and served snapshots
are checked against previews and ledger rows too.

This supplements the provider-specific live suites and the deterministic
`tests/multi-instance/scenarios.py` fixture; it does not claim live coverage of
providers without credentials or of every modality/model capability. Perplexity
live tests serialize their calls automatically to avoid harness-induced 429s.

---

## Pull request process

1. **Branch from `main`.** Keep PRs focused; one logical change per PR.
2. **Squash commits on merge.** Write each commit message as if it were the
   only one that would land. Use the
   [Conventional Commits](https://www.conventionalcommits.org/) style —
   `fix:`, `feat:`, `docs:`, `refactor:`, `test:`, `chore:`.
3. **Title & description.** The PR title becomes the squash-merge commit
   subject. Describe *what* changed and *why*; link any issues/PRs the change
   closes. Call out breaking API or config changes explicitly.
4. **CI must be green.** Every check in
   [.github/workflows/ci.yml](.github/workflows/ci.yml) must pass on
    `ubuntu-latest` (the pinned-toolchain matrix).
    `--all-features` clippy is required; only the extra nightly leg is
    allowed to fail.
5. **No secrets in code or history.** Virtual keys, master keys, JWT
   secrets, and provider credentials belong in env vars or `.env` files
   (never committed). See [SECURITY.md](SECURITY.md) for the disclosure
   path if you find a real secret in the tree.
6. **Don't update `Cargo.lock`** unless you are intentionally bumping a
   transitive dep or adding a new crate. Reviewers will ask otherwise.

### Review expectations

- New public types carry doc-comments. Doc-tests on `oxllm-core` public APIs
  are strongly encouraged — `cargo test --doc -p oxllm-core` runs them.
- New error paths return a typed error (`ApiError` / `oxllm_core::Error`),
  not `panic!` / `unwrap` / `expect` on caller input.
- New HTTP behaviour gets an integration test under `tests/` against
  `httpmock`.
- New config knobs get a line in the relevant
  [examples/config.example.yaml](examples/config.example.yaml) block and a
  note in [README.md](README.md) when user-facing.

---

## Feature flags

The top-level `oxllm` crate gates optional behaviour behind Cargo features.
Most crates are unconditional; the flags below control binary-level surface
area only.

| Flag | Crate(s) enabled | What it does |
|------|------------------|--------------|
| `db` | `oxllm-server/db` | Enables SQLite/Postgres persistence: virtual keys, teams, users, spend logs, DB-stored model config and DB-backed management endpoints. Off → no DB; master-key/JWT auth still works. |
| `otel` | `oxllm-server/otel` | Enables OpenTelemetry OTLP export (`otel_endpoint`, `otel_service_name`). **Requires a full process restart** — these settings are NOT hot-reloadable. |
| `redis` *(server-internal)* | `oxllm-cache-redis`, `oxllm-server` | DualCache (in-memory + Redis) for multi-replica rate-limit / budget accuracy. Required when `replicas > 1`. |
| `s3-cache` / `gcs-cache` / `azure-cache` | corresponding cache crate + server | Enables the S3, GCS, or Azure Blob response-cache backend. Without the matching feature, that `cache.type` warns and falls back to memory. |
| `live-tests` (test config) | root tests | Enables real-provider `e2e_live.rs` cases; without it they compile but are ignored. |
| `admin` | `oxllm-admin` | Builds the `oxllm-admin` binary and the HTMX admin UI server. Use for the local UI; production deployments usually expose the management REST API directly. |
| `platform-apis` *(server-internal)* | `oxllm-server/platform-apis` | OpenAI platform endpoints (Batches, Files, Fine-tuning, Assistants, Threads). Off → those routes return 404. |

The CI feature matrix (`db`, `admin`, `otel`, `--no-default-features`,
`--all-features`) is exercised in
[.github/workflows/ci.yml](.github/workflows/ci.yml). When adding or
removing a feature flag, update the matrix there and this table.

The root defaults are empty. A source-built proxy needs `--features db` plus a
configured database URL for persistence; the proxy Dockerfile includes `db,otel`.
`--no-default-features` on the root is not a provider-pruning switch: dependency
defaults remain enabled unless explicitly changed in their dependency entries.
