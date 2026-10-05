# Troubleshooting

Start with [operations](operations.md) for deployment, configuration and recovery
procedures. Use `RUST_LOG=oxllm=debug,oxllm_server=debug` for targeted diagnostics;
`general_settings.log_format: json` selects JSON for `serve` after a restart.
Treat detailed request/response logs as potentially sensitive.

## Proxy won't start

1. **Validate the actual config:** `oxllm validate --config config.yaml`.
   `cargo test --test config_compat` tests repository fixtures, not the file you
   intend to deploy. Config validation can perform DNS checks but does not prove
   DB/cache/upstream readiness.
2. **Export every active env reference.** `os.environ/NAME` is resolved against
   the process environment; missing values are errors. The binary does not read
   `.env`. Use your supervisor, Compose or a trusted shell-compatible env file.
   Start from `examples/config.minimal.yaml`, not the all-provider catalog.
3. **Check auth configuration.** `OXLLM_REQUIRE_AUTH=true` requires a usable auth
   path (master key, JWT or DB-backed virtual keys). A provider API key is not a
   proxy master key. A DB-configured deployment also needs the `db` feature and
   an available, migrated database.
4. **Check the bind port.** CLI `--port` overrides `OXLLM_PORT`, then the built-in
   default is 4000. Inspect existing listeners before choosing another port;
   the proxy binds `0.0.0.0` on that port.
5. **Check private targets.** Private Redis, Ollama, local callbacks and OTLP
   collectors need `general_settings.allow_private_api_base: true`. This is an
   explicit outbound-network opt-in; keep the sample Redis allowance when
   deploying on a private Compose/cluster network.

## Compose demands admin secrets without starting admin

Required `${VAR:?message}` interpolation happens before profile filtering.
Use only `docker-compose.yml` for the base stack. Include
`docker-compose.admin.yml` when enabling the admin profile, with both
`ADMIN_SESSION_SECRET` and `ADMIN_UI_PASSWORD` set. This retains the credential
checks rather than replacing them with empty values.

To validate syntax with synthetic credentials and no local `.env`, run
`bash scripts/check-deployment-artifacts.sh --require-tools`.

## 401 or 403 responses

- `/health` is public; it can prove process liveness independently of auth.
  Test proxy auth without a provider call using `/v1/models` with
  `Authorization: Bearer <proxy key>` or `x-api-key: <proxy key>`.
- Master keys, DB virtual keys and JWTs are distinct from upstream provider
  credentials. Check the returned error/source before concluding a provider's
  401 is a proxy login failure.
- Model ACLs and caller IP allowlists can cause 403. `allowed_ips: []` denies
  all protected callers; omit it to allow all. Behind a proxy, forwarded IPs
  are honored only from `trusted_proxy_cidrs`.
- Rotating an environment-backed master/JWT secret requires a process restart
  with the new environment. Exporting a value in another shell does not modify
  an already-running process. In Helm, bump `rolloutRevision` for external
  Secret rotations.
- Deleted keys are invalidated locally and published via Redis when enabled.
  Pub/sub is best-effort; a disconnected peer can retain an entry until its
  positive-cache TTL. Test revocation on each replica when diagnosing drift.

## 429 responses

Check whether the response is a proxy limit/budget rejection or an upstream
provider 429. Caller RPM/TPM and budgets can come from virtual keys and associated
scope records; per-deployment limits are in `oxllm_params`. Inspect management
records, response rate-limit headers and `/metrics` rather than assuming every
429 means an upstream outage.

Budget hooks reserve estimated cost before the call and settle against usage
after it; they are not merely post-call counters. Unknown pricing or upstream
usage affects metering accuracy. Cache hits still consume RPM but skip upstream
TPM/budget charges. Master-key smoke traffic bypasses caller-scoped limits, so
use a scoped virtual key when testing enforcement.

Multiple replicas need shared Redis counters. The chart rejects `replicas > 1`
with memory/blob-only configuration to prevent independently multiplied limits.
Configure retry/fallback targets deliberately; not every client-caused 4xx is
retryable.

## Redis unavailable or readiness degraded

- Verify reachability **from the proxy's network**, not just from your laptop.
  The base Compose Redis intentionally has no published host port; a native
  proxy needs a loopback-accessible Redis fixture (for example, run Redis on
  127.0.0.1:6379; `scripts/dev-stack.sh` expects one there).
- Configure `oxllm_settings.cache.redis_url` for auth, TLS and database index.
  It takes precedence over `host`/`port`. `cache.password` and `cache.pool_size`
  are not supported connection knobs in this config type.
- Use `rediss://` for a TLS Redis endpoint and check certificate trust. Include
  Redis's port and DNS in any Kubernetes egress policy; the chart's default
  NetworkPolicy permits HTTPS/DNS, not arbitrary backend ports.
- Response-cache reads/writes can tolerate errors. **Atomic Redis counter
  operations do not silently fall back to local counters:** they return errors
  to the enforcement hooks. Inspect the actual response and readiness status
  rather than assuming every request remains successful during an outage.
- If Redis construction failed at boot and logs report a memory fallback,
  restore Redis and restart the proxy. Reload preserves the primary cache
  handle and does not recreate a failed startup connection. An existing Redis
  connection manager can reconnect after a later transient outage.

## Database unavailable

The proxy Dockerfile includes `db`; a plain source build does not. Set
`general_settings.database_url` explicitly. Merely exporting `DATABASE_URL` or
`OXLLM_DATABASE_URL` does not activate it for `serve`; the YAML must reference
the intended variable. The latter is also accepted directly by `oxllm migrate`.

For SQLite, make the parent directory and WAL location writable. The Compose
named volume uses UID/GID 10001. For Postgres, verify the URL, TLS trust, network
policy and role permissions. Automatic migrations need DDL rights. If migrations
run separately, set `OXLLM_AUTO_MIGRATE=false` for the runtime role and verify
the schema was provisioned using the intended image.

Connection pools are configured in `crates/oxllm-db/src/pool.rs` (Postgres max
20 connections, file SQLite max 4, in-memory SQLite 1). There is no
`DATABASE_POOL_SIZE` environment setting. Account for all replicas and migration
containers when allocating DB connection capacity.

The chart's `persistence.enabled` wires both the migrate init container and
`serve` to one Secret. See [backup/migration recovery](operations.md#migrations-and-recovery)
before repairing schema or rolling back an upgrade.

## Config edits have no effect

Check for the watcher log `config reloaded and state swapped successfully`.
Validation failures keep the previous state. Read the
[reload matrix](operations.md#configuration-and-restarts): DB/cache connections,
startup middleware, CORS, retention, logging and shutdown timing require a
restart. `GET /config` alone is not proof that every infrastructure setting has
been rebuilt.

ConfigMap `subPath` mounts do not receive updates. The chart mounts a directory
and changes pod-template checksums on Helm upgrades for deterministic rollouts.
Roll pods for out-of-band config/Secret changes when required. There is no
`--watch-config-dir` CLI flag. Docker single-file bind mounts can also retain
the old inode after an editor atomically replaces the host file; recreate the
container or use a directory mount and an explicit config path.

## Admin UI login or assets fail

- Set the same proxy master key on both processes and ensure
  `OXLLM_PROXY_URL` is reachable from the admin container.
- A non-loopback admin bind requires `ADMIN_UI_PASSWORD`. Explicit session
  secrets must be at least 16 characters; a stable secret preserves sessions
  across restarts. Secure cookies require HTTPS; opt out only for local HTTP.
- DB-backed keys/teams/users/spend pages need persistence on the proxy; a
  healthy admin process does not prove those management APIs are available.
- Admin templates and browser assets are embedded in the binary. If `/login`
  cannot load its local script/image URLs, check the running binary/version and
  `/health`; there is no runtime `/static` directory to populate.

## Anthropic features or streaming differ

`/v1/messages` translates Anthropic input through the shared OpenAI-shaped chat
pipeline. For native Anthropic Messages forwarding, use
`/anthropic/v1/messages` and an Anthropic deployment; the Anthropic SDK's base
URL is then `http://host:4000/anthropic`. The native route and translated route
have separate tests and capability limits.

For stream truncation, check per-deployment `stream_timeout` (idle time between
chunks), request timeouts, client disconnects and ingress idle/buffering policy.
Use `curl -N` to compare against the SDK. After SSE response headers have been
sent, an upstream error cannot change the HTTP status; inspect stream events and
server logs too. Do not infer an HTTP/2 bug from a missing `Transfer-Encoding`
header—HTTP/2 does not use that HTTP/1.1 framing mechanism.

During shutdown, allow the proxy's effective drain plus background-flush time
before an orchestrator sends SIGKILL. See [shutdown and probes](operations.md#shutdown-and-probes).
