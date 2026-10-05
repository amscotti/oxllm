# Security policy

## Reporting a vulnerability

**Please do NOT open a public GitHub issue for security problems.** Instead,
report vulnerabilities privately so we can ship a fix before details are
public.

Use a private channel verified by the maintainer from whom you obtained this
checkout. A hosted repository, reporting mailbox, GPG key and response SLA have
not been verified here; the earlier draft's contact/SLA claims are not a
confirmed reporting service. Before public distribution, maintainers should
publish a working private contact or enable Private Vulnerability Reporting on
the actual hosting repository.

Include a minimal reproduction, affected commit/version, impact and any
mitigation already tried. Use synthetic credentials and redact customer data.
Coordinate disclosure timing directly with the maintainer handling the report.

## Supported versions

This source tree identifies itself as `0.1.0` and is under active development.
Include the commit SHA in reports; a version string alone does not identify
unreleased fixes. Historical review completion counts are not a support or
security certification. Release/backport commitments must be stated by the
maintainer of the distributed build.

## Security features overview

Deployment safety depends on the configured auth and network boundary. The
binary permits open mode when no auth is configured unless the require-auth
guard is enabled. See
[README.md](README.md#security) for the full list and the
[example config](examples/config.example.yaml) for the relevant keys.

### Authentication & authorization

- **Master key** bypasses caller-scoped ACL/budget/rate-limit checks; configured via `general_settings.master_key`
  (or `os.environ/OXLLM_MASTER_KEY`). `OXLLM_REQUIRE_AUTH=true` refuses to
  start without a usable master-key, JWT or DB-backed virtual-key auth path.
- **Virtual keys** (`sk-ox-*`) are SHA-256 hashed at rest and compared in
  constant time. Soft-delete + immediate cache invalidation prevents stale
  keys from being used.
- **JWT auth** (HS256/HS384/HS512) is checked after master + virtual keys,
  with optional audience claim enforcement.
- **Per-IP auth brute-force rate limiting** prevents key guessing.
- **Trusted proxy CIDRs** gate `X-Forwarded-For` / `X-Real-IP` so the
  per-IP limiter cannot be spoofed from outside the load balancer.
- **Model ACLs** (`models` on keys/teams/users) and per-key/team/project/org
  budgets are enforced before any upstream call.

### Request safety

- **SSRF protection.** `oxllm_params.api_base` is validated at config
  load — private / loopback / link-local addresses are rejected unless
  `general_settings.allow_private_api_base: true` is set explicitly.
- **Body size limit.** Default 10 MB, configurable via
  `general_settings.max_request_body_size_mb`.
- **JSON depth limit** of 128 levels rejects pathologically nested payloads
  (DoS against `serde_json`).
- **HTTP request smuggling guard.** Requests with both `Content-Length` and
  `Transfer-Encoding` are rejected.
- **No-redirect policy.** Upstream requests do not follow redirects — the
  proxy never silently follows an upstream to an attacker-controlled host.
- **Upstream error body cap.** Error responses from upstreams are capped at
  10 KB so a malicious upstream cannot OOM the gateway with a huge error
  payload.

### Operational safety

- **Graceful shutdown** drains for the maximum of configured shutdown time
  (default 30 s) and global request timeout (default 600 s). The orchestrator
  needs additional grace for background flushing; see [operations](docs/operations.md).
- **Per-deployment cooldowns** quarantine failing deployments so one bad
  upstream cannot poison the request chain.
- **`traceparent` propagation** keeps OpenTelemetry traces intact across
  the proxy → upstream hop, which is critical for incident forensics.
- **Structured `tracing` logs** redact secrets (API keys, JWT bearer tokens)
  before they reach the log sink.

### Storage and cache privacy

Response caches hold model output, and semantic caches can hold prompt-derived
data. Principal-scoped cache keys isolate distinct keys; users sharing a key
share its cache scope. Hashes of lookup keys are not encryption of cached
payloads. Configure retention, storage access, TLS and managed at-rest encryption
according to your deployment's data policy. See the
[cache/storage guide](docs/operations.md#cache-privacy-and-storage).

DB-managed provider credentials have separate opt-in encryption via
`ENCRYPTION_KEY` / `ENCRYPTION_KEY_PREVIOUS`. This does not encrypt all response
caches or spend records. Retain the needed key ring for backups and verify
restores. JWT and admin-session secrets are not substitutes for that key ring.

### What is intentionally *not* a security boundary

- The **`/metrics` Prometheus endpoint** is unauthenticated. Put it behind
  your ingress / network policy (mTLS, sidecar, security group) — never
  expose it to the public internet.
- The **`/health` and `/health/readiness`** endpoints are unauthenticated
  by design (kubelet probes). They return no sensitive data.
- **Master-key traffic bypasses ACLs and rate limits.** Reserve the master
  key for bootstrap and management tasks; give every application its own
  virtual key with a tight `max_budget` and `models` allow-list.

See [docs/troubleshooting.md](docs/troubleshooting.md) for common
auth / rate-limit / connectivity problems and their resolutions.
