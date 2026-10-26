# Security policy

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.** Report
vulnerabilities privately so a fix can ship before details become public.

Use GitHub's private vulnerability reporting for this repository:

<https://github.com/amscotti/oxllm/security/advisories/new>

If that form is unavailable, open a public issue containing only a request for
a private channel — do not include reproduction steps, credentials, customer
data, or exploit details in the public issue.

Include a minimal reproduction, the affected commit/version, impact, and any
mitigation already tried. Use synthetic credentials and redact customer data.
We will coordinate disclosure timing with you and ask that you allow time for a
fix before publishing details.

## Supported versions

oxllm is pre-release software. The `0.1.x` line is the only line receiving
fixes, and support is best-effort with no guaranteed response or backport SLA.
Include the commit SHA in reports; a version string alone does not identify
unreleased fixes.

## Security features overview

Deployment safety depends on the configured auth and network boundary. The
binary permits open mode when no auth is configured unless the require-auth
guard is enabled. See [README.md](README.md#security) for the full list and the
[example config](examples/config.example.yaml) for the relevant keys.

### Authentication and authorization

- **Master key** bypasses caller-scoped ACL/budget/rate-limit checks; configured
  via `general_settings.master_key` (or `os.environ/OXLLM_MASTER_KEY`).
  `OXLLM_REQUIRE_AUTH=true` refuses to start without a usable master-key, JWT,
  or DB-backed virtual-key auth path.
- **Virtual keys** (`sk-ox-*`) are SHA-256 hashed at rest and compared in
  constant time. Soft-delete and cache invalidation prevent stale keys from
  being used after revocation has propagated.
- **JWT auth** (HS256/HS384/HS512) is checked after master and virtual keys,
  with optional audience enforcement and optional replay protection.
- **Per-IP auth brute-force rate limiting** slows key guessing. Trusted proxy
  CIDRs gate `X-Forwarded-For` / `X-Real-IP` so the limiter cannot be spoofed
  from outside the load balancer.
- **Model ACLs** (`models` on keys/teams/users) and per-scope budgets are
  enforced before any upstream call.

### Request safety

- **SSRF protection.** `oxllm_params.api_base` is validated at config load —
  private, loopback, and link-local addresses are rejected unless
  `general_settings.allow_private_api_base: true` is set explicitly.
- **Body size limit.** Default 10 MB, configurable via
  `general_settings.max_request_body_size_mb`.
- **JSON depth limit** of 128 levels rejects pathologically nested payloads.
- **HTTP request smuggling guard.** Requests with both `Content-Length` and
  `Transfer-Encoding` are rejected.
- **No-redirect policy.** Upstream requests do not follow redirects, so the
  proxy cannot be steered to an attacker-controlled host by a redirect.
- **Upstream error body cap.** Error responses from upstreams are capped so a
  malicious upstream cannot exhaust gateway memory with a huge error payload.

### Operational safety

- **Graceful shutdown** drains for the maximum of configured shutdown time
  (default 30 s) and the global request timeout (default 600 s), plus additional
  time for background flushing. See [operations](docs/operations.md).
- **Per-deployment cooldowns** quarantine failing deployments so one bad
  upstream cannot poison the request chain.
- **`traceparent` propagation** keeps OpenTelemetry traces intact across the
  proxy-to-upstream hop.
- **Structured `tracing` logs** redact secrets (API keys, JWT bearer tokens)
  before they reach the log sink.

### Storage and cache privacy

Response caches hold model output, and semantic caches can hold prompt-derived
data. Principal-scoped cache keys isolate distinct callers; users sharing a key
share its cache scope. Hashes of lookup keys are not encryption of cached
payloads. Configure retention, storage access, TLS, and managed at-rest
encryption according to your deployment's data policy. See the
[cache/storage guide](docs/operations.md#cache-privacy-and-storage).

DB-managed provider credentials have separate opt-in encryption via
`ENCRYPTION_KEY` / `ENCRYPTION_KEY_PREVIOUS`. This does not encrypt all response
caches or spend records. Retain the needed key ring for backups and verify
restores. JWT and admin-session secrets are not substitutes for that key ring.

### What is intentionally not a security boundary

- **`/metrics`** is protected by the normal proxy auth stack when auth is
  configured. It can still expose operational data to any principal allowed to
  reach it, so keep it behind your ingress, network policy, or mTLS boundary
  and do not expose it to the public internet.
- **`/health` and `/health/readiness`** are unauthenticated by design for
  kubelet probes. They return no sensitive data.
- **Master-key traffic bypasses ACLs and rate limits.** Reserve the master key
  for bootstrap and management tasks; give every application its own virtual
  key with a tight `max_budget` and `models` allow-list.

See [docs/troubleshooting.md](docs/troubleshooting.md) for common auth,
rate-limit, and connectivity problems.
