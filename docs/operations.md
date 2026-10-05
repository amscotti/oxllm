# Operating oxllm

This guide describes the source tree's deployment paths. Image builds, a
registry push, successful cluster admission, and live-provider calls are
separate checks; a rendered manifest alone does not establish those outcomes.

## Choose the runtime

| Runtime | Included behavior |
|---|---|
| `cargo build --locked --release --bin oxllm` | Proxy with root default features; no DB persistence |
| `cargo build --locked --release --bin oxllm --features db,otel` | SQLite/Postgres and optional OTLP export |
| `cargo build --locked --release --bin oxllm-admin --features admin` | Separate admin UI; connect it to a DB-enabled proxy for management |
| `Dockerfile` | Proxy with `db,otel`, CA trust, HTTP health check and writable `/data` owned by UID/GID 10001 |
| `Dockerfile.admin` | Admin binary, local static assets, CA trust and HTTP health check; UID/GID 10001 |

The root Cargo feature list is empty by default. Dependency features such as
Redis and providers have their own defaults; `--no-default-features` on the root
does not disable all dependency features.

## Compose

1. Copy `.env.example` to a private `.env`, set a nonempty master key, and add
   provider keys for the models you intend to call.
2. Run `docker compose up --build -d --wait`.
3. Run `bash scripts/smoke-docker.sh`. Its default checks exercise local health
   and authentication only. `OXLLM_LIVE_SMOKE=1` opts into small billable requests
   for non-placeholder OpenAI/xAI/Anthropic credentials available to the script.

Redis is private to the Compose network. SQLite lives at `/data/oxllm.db` in the
`oxllm-data` named volume. `DATABASE_URL` can select external Postgres instead.
Keep `allow_private_api_base: true` while using the private Redis service; the
setting also permits private upstream/telemetry targets, so use appropriate
network access controls for that deployment.

The optional admin service is in `docker-compose.admin.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.admin.yml --profile admin up --build -d --wait
```

Set `ADMIN_SESSION_SECRET` (at least 16 characters) and `ADMIN_UI_PASSWORD` before
including this file. Compose interpolates required variables before filtering
profiles; the separate file preserves strict checks without requiring admin
credentials for the base stack. The UI is published on `127.0.0.1:8080` and waits
for proxy health. `ADMIN_COOKIE_SECURE=false` in the sample is for local HTTP;
use `true` with TLS at a reverse proxy. Restart policies recover exited
containers; a Docker health check alone does not restart an unhealthy process.

For optional JWT auth, set a strong `JWT_SECRET` and uncomment
`general_settings.jwt_secret: os.environ/JWT_SECRET` in the mounted config.
Passing an environment variable alone does not add a YAML setting. Use
`OXLLM_CONFIG_FILE` to mount your own config rather than editing the sample.

`docker compose down` retains named volumes. `down --volumes` removes them and
is appropriate only for disposable fixtures or after a deliberate backup.

## Configuration and restarts

The binary reads the process environment, not `.env` files. A trusted
shell-compatible file can be exported before a direct launch:

```bash
set -a
source .env
set +a
oxllm validate --config config.yaml
oxllm serve --config config.yaml
```

`validate` checks YAML, env references, typed config and the require-auth guard;
it does not establish database/Redis/upstream readiness. Missing env references
are errors. `os.environ/NAME` substitutes strings only, so use YAML numbers and
booleans directly. The legacy top-level `environment_variables` block does not
set process variables in shipped builds.

| Setting | Apply method |
|---|---|
| `model_list`, aliases, routing/retry/fallback knobs, request-time limits, cache TTL, config-derived hooks | File reload |
| Master/JWT settings, `trusted_proxy_cidrs`, `allowed_ips`, HTTP pool/connect settings, guardrails | File reload |
| DB URL, primary cache backend/Redis connection, infrastructure credentials | Restart; existing connections survive file reload |
| Port, body-size middleware, management/proxy CORS, declarative passthrough route registration | Restart; HTTP router is built at startup |
| `log_format`, OTel endpoint/service, `spend_log_retention_days`, shutdown drain timing | Restart |
| Exported provider/master/JWT values, Kubernetes env-backed Secrets, admin configuration | Restart with the new environment |

`request_timeout` affects requests through live state, but the shutdown budget
is captured at startup. Restart and adjust the orchestrator's grace when
changing it. The watcher logs `config reloaded and state swapped successfully`
after applying an edit; invalid edits retain the previous state. Long-lived
requests can finish using the old state.

For Kubernetes, the chart mounts the config directory and uses checksums to
roll pods when Helm-managed config or secrets change. Do not rely on file
watching alone for projected-volume updates or startup-only settings. A
ConfigMap mounted with `subPath` does not receive updates. For externally
managed Secret rotations, bump `rolloutRevision` or restart the Deployments.

### Logs, CORS and client IPs

Set `general_settings.log_format: json` for structured `serve` logs and use
`RUST_LOG` to select verbosity. `validate`, `keygen`, `migrate` and the admin
binary use their own text log setup. CORS is off when its allowlist is omitted
or empty. `management_cors_origins` and `proxy_cors_origins` govern different
route groups; the admin UI normally calls the proxy server-to-server, so it
does not need browser CORS access to the proxy.

`allowed_ips` omitted permits all caller IPs; an empty list denies all callers
on the protected route groups. Use real CIDRs. Forwarded IP headers are trusted
only when the connection peer belongs to `trusted_proxy_cidrs`; otherwise the
socket peer is used. Behind a load balancer, an allowlist of only the load
balancer's IP is not an allowlist of its end users.

## Shutdown and probes

The proxy drains active requests for
`max(general_settings.shutdown_timeout_secs or 30, oxllm_settings.request_timeout or 600)`
seconds after SIGTERM/SIGINT, then shuts down background work and flushes spend
and telemetry. This is a shutdown budget, not a process lifetime limit.

Leave additional cleanup time in the orchestrator: the bundled Compose config
uses a 120-second request timeout and a 150-second stop grace. Helm computes the
drain plus 30 seconds (630 seconds with the application defaults). A larger
per-deployment timeout does not automatically enlarge the startup drain; set
`shutdown_timeout_secs` accordingly. Align ingress/client stream timeouts too.

`/health` is liveness. `/health/readiness` reports configured DB/cache health
and shutdown readiness; use readiness to remove an instance from traffic.
These probes do not contact every provider or certify provider API keys.

## Migrations and recovery

Set `general_settings.database_url` and build with `db`. A configured DB is a
startup dependency: connection or migration failures must be resolved before
serving. With `OXLLM_AUTO_MIGRATE=false`, provision the schema first:

```bash
OXLLM_DATABASE_URL="$DATABASE_URL" oxllm migrate
OXLLM_AUTO_MIGRATE=false oxllm serve --config config.yaml
```

`OXLLM_DATABASE_URL` is the **migrate CLI** environment variable; `serve` needs a
YAML reference such as `database_url: os.environ/DATABASE_URL`. The chart wires
both explicitly to the same Secret. Its migrate init container runs **per pod**,
not once per release. Database locks serialize migrations. For separate DDL and
runtime DML roles, run the migration CLI externally and use
`persistence.migrate=false` in Helm. See [chart settings](../helm/oxllm/README.md).

PostgreSQL upgrades automatically prepare existing legacy spend tables online:
additive columns commit in a short transaction, then missing spend indexes are
built with `CREATE INDEX CONCURRENTLY` before schema version 14 and its read
fallbacks become ready. A dedicated migration session coordinates replicas;
public startup allows a bounded 30-minute preparation budget, including waiting
for another replica, while ordinary DDL lock waits remain short. Allow enough
startup/init-container grace for a large index build. Session locks/settings are
discarded with that connection on completion or cancellation. Wrong or invalid
same-named indexes and inconsistent event IDs produce a typed readiness error
without replacement or data deletion; an interrupted concurrent build can leave
an invalid index that must be inspected and recovered before retrying.

Before an upgrade, retain the current image/config and take a database backup.
Use a backup method appropriate to the database:

- **SQLite:** use SQLite's online backup facility (`sqlite3 /path/oxllm.db
  ".backup '/backup/oxllm.db'"`) or stop all writers before copying the database
  and any WAL state. Copying only a live `.db` file can omit committed WAL data.
  The runtime image does not include the `sqlite3` CLI; run backup tooling with
  access to the volume. Test the backup using `PRAGMA integrity_check` and a
  restore into a disposable volume owned by UID/GID 10001.
- **Postgres:** use your managed backup/PITR service or `pg_dump --format=custom`
  with a protected connection configuration. Test `pg_restore` into a new
  database, run migrations using the intended image, and verify management
  reads before switching production traffic.

Retain encryption keys separately from backups when DB-managed provider
credentials use `ENCRYPTION_KEY`. It accepts 32 random bytes encoded as 64 hex
or 44 base64 characters. `ENCRYPTION_KEY_PREVIOUS` retains old keys during
rotation; do not discard a key while any retained backup or row requires it.
JWT/session secrets serve different purposes. Restoring a DB without its
credential key ring does not recover encrypted provider credentials.

Migration rollback is not automatic. An older binary may not understand a new
schema. Restore the pre-upgrade DB into a separate target and use its matching
image/config when a downgrade needs schema rollback. Stop conflicting writers
and check `/health/readiness`, management data, key authentication and spend
records after recovery.

Set `spend_log_retention_days` to your retention policy; omitted means no pruning.
Pruning runs daily and begins after startup, rather than deleting immediately.
Database backups and external logs have independent retention lifetimes.

## Cache privacy and storage

Response caching retains model output; semantic caching can additionally retain
prompt text/embeddings. Cache keys include the authenticated principal's hash,
but applications sharing one key share that principal's cache scope. Use
distinct keys for independent tenants and review cache-control policies when
payload retention is restricted. Hashing cache keys does not encrypt the cached
values.

For Redis and object storage, choose access controls, transport security,
managed encryption at rest and lifecycle/backup retention appropriate to your
data. Blob TTL checks turn expired objects into misses; physical deletion
depends on your bucket/container lifecycle rules. `ENCRYPTION_KEY` protects
DB-managed provider credentials, not all cached responses or spend data. A
managed storage encryption policy is an operator choice; this repository does
not claim every deployment lacks it or must use a particular KMS.

For multiple replicas, use a shared Redis counter store. Blob response caches
do not replace Redis's atomic fleet-wide counter behavior. Do not flush a live
Redis database as though it contained only disposable responses: it also holds
budget/rate-limit, revocation and coordination state.
The bundled Redis uses `noeviction` for this reason. Monitor its capacity and
errors: reaching `maxmemory` rejects writes instead of silently resetting live
enforcement counters.

## Validation and image publication

`bash scripts/check-deployment-artifacts.sh --require-tools` performs nonbuilding
Helm/Compose and YAML checks with synthetic credentials and `/dev/null` as the
Compose env file. It needs Helm, Docker Compose, Python 3.11+ and PyYAML. CI also
builds both images and runs health/auth/SQLite/admin-asset smoke checks with no
provider keys. Rust CI separately runs the pinned toolchain, Criterion bench
compilation, Redis integration and Postgres integration.

`.github/workflows/release.yml` is a parameterized container pipeline. Configure
repository variables `CONTAINER_REGISTRY` and `CONTAINER_NAMESPACE` and secrets
`REGISTRY_USERNAME` / `REGISTRY_PASSWORD` in the repository where you host it.
A pushed `v*` tag runs Rust/deployment checks and publishes both images using the tag
without its leading `v`. Manual runs accept destination/tag inputs and default
to build-only (`publish=false`). The job publishes Linux images for its runner
architecture; it is not a multi-architecture release job.

No registry namespace, hosted GitHub repository, security contact or released
artifact is established by this local checkout. Configure and verify those
destinations before relying on automated releases. Version tags in Dockerfiles
are mutable; use a verified digest if your release policy requires immutable
base images. No fabricated digest or mandatory signing/SBOM service is supplied.
