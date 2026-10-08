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
are errors. `os.environ/NAME` is substituted as a string. If the typed config
then fails to parse, only substituted values rejected by numeric/boolean fields
are coerced as integers, floats, or `true`/`false`. String fields keep their
values, including numeric-looking API keys and header values. Invalid scalar
values fail the load with the field path and variable name. Ordinary YAML
strings are not coerced. The legacy top-level
`environment_variables` block does not set process variables in shipped builds.

`include` is expanded before that substitution. Each entry is a path relative
to the file that declares it. Lists, including `model_list`, concatenate with
the declaring file's items first. Every other value is replaced by the included
file, so `oxllm_settings`, `general_settings`, and `router_settings` are
replaced as whole blocks and are not deep-merged. Nested includes are applied
after the siblings already queued. A cycle fails the load and names the chain.
Repeated files merge once; cycles are rejected even across shared includes.
Legacy and native gateway-settings names refer to the same replaceable block.
The hot-reload watcher follows contributing files and their logical symlink
paths, including files added by later edits. `POST /config/update` validates
includes and env-driven scalars before atomically writing the root file.

Deployments with `model_info.supported_environments` load only in matching
environments. The first non-empty `OXLLM_ENVIRONMENT` or `LITELLM_ENVIRONMENT`
value wins; otherwise oxllm defaults to `development`. An absent/null list
allows every environment; an empty list allows none. Filtering precedes
credential/env resolution, so production does not require dev-only credentials.
The same scope is honored by programmatic state construction and DB models;
inactive DB definitions can be stored and activated later.

`general_settings.allow_short_master_key: true` permits a non-blank master key
below 16 bytes only in development, with a warning at boot. Other environments
reject the flag even with a long key. JWT secrets retain the 16-byte minimum.

Deployment `drop_params` overrides the global flag. `allowed_openai_params`
preserves explicitly permitted extras without undoing known provider mappings.
`additional_drop_params` removes top-level keys before transforms can rename
them and again after body overrides. `extra_body` shallowly overrides the
native outbound JSON; configured values, including `store: false`, win over
client/provider values, while routing retains control of `model`. Drops win
over allows/body overrides. JSON passthrough routes apply the same final pass.
Nested dotted drop paths are not supported.

Deployment `auth_header` replaces native credential headers; `auth_scheme`
sets the value prefix. `auth_header: api-key` defaults to a raw key, while
`auth_header: authorization` defaults to `Bearer`. Explicit `raw`, `none` or an
empty scheme sends no prefix. Header names/schemes are validated, and
`extra_headers` is applied afterwards. These overrides work on transformed and
provider-authenticated passthrough paths with API-key/bearer credentials; AWS
signing remains owned by the signer.

`azure_ai/` serves Foundry native Claude messages, compatible vendor chat and
text embeddings. Configure the resource or full endpoint in `api_base`; existing
query strings are preserved and explicit `api_version` fills an absent
`api-version` query. Unified `.services.ai.azure.com` resources use `api-key`;
explicit `azure_ad_token` uses Bearer authentication. Native Cohere embedding
deployments also recognize unified Foundry hosts, or `api_type: azure_ai` for
custom gateways, and send `input` to `/models/embeddings`. Azure ML
`llm_provider-num_tokens` headers are honored for embedding usage and cost.
See [the Foundry configuration sample](../examples/config.foundry.yaml).

`general_settings.http_force_ipv4: true` restricts the shared upstream and
image-download clients to IPv4 through DNS filtering and IPv4 local binding.
The general-settings `force_ipv4` alias and legacy
`oxllm_settings.force_ipv4` are accepted; an explicit native setting, including
false, wins. The image client keeps its private-address restrictions even when
operator-controlled upstreams opt into private bases.

`cache_control_injection_points` supports Anthropic message-role/index selectors
on transformed chat requests. Indices refer to source messages before system
splitting. Existing controls/TTLs are preserved; missing targets are skipped
and existing tool/system/message controls count toward the four-block cap.
Malformed selectors and points combined with explicitly disabled global
prompt caching are rejected during validation. Automatic cache insertion from
the global prompt-routing switch remains a separate feature.

Deployment `*_above_<N>k_tokens`/`*_above_<N>_tokens` prices apply to all tokens
after the prompt crosses the threshold. At 272,000 tokens the base applies;
at 272,001 the tier applies. `tier_threshold_is_inclusive: true` opts into an
inclusive boundary. Cache-creation tokens count toward context length,
including standalone one-hour details. Malformed tier keys/rates are warned.

Named `retry_policy` entries map onto status counts and structured categories.
Timeouts map to 408; actual upstream 504s belong to the 5xx category.
`BadRequestErrorRetries` and `AuthenticationErrorRetries` accept only zero.
Their zero allowed-fails mappings opt those statuses into immediate fallback;
other request-content errors remain terminal unless explicitly opted in.
Explicit numeric retry entries win over aliases. Nonzero content-policy
retry parity is not established by the zero-retry compatibility tests.
Exhausted deployments are excluded for the remainder of the current request,
even with `cooldown_time: 0`; future requests may reconsider them immediately.

Vertex renewable auth supports service-account and authorized-user JSON,
with precedence: static deployment `api_key`, deployment `vertex_credentials`,
explicit access-token env, `GOOGLE_APPLICATION_CREDENTIALS`, then inline
`VERTEX_AI_CREDENTIALS_JSON` (or `LITELLM_VERTEX_AI_CREDENTIALS`). Omit the static
deployment key/access-token override to select renewable mode. Initial token
acquisition performs a coalesced exchange; afterwards a background worker renews
at half TTL, signing a fresh assertion when needed. Valid cached tokens remain
available during renewal. Failed renewal retries with backoff, and an expired
token is rejected rather than forwarded. The bounded identity cache has a
two-hour idle policy; eviction or resolver shutdown cancels workers and releases
refresh material. Full ADC discovery/federation/impersonation is not implemented.

Bedrock Cohere/Titan text embeddings use InvokeModel with SigV4 over the exact
sent bytes. Runtime URLs use the resolved signing region. Usage comes from
the native body, then the invocation token-count header, then text estimation
when metering is absent. Mantle token scopes follow an official custom base's
host region so an ambient region cannot create a mismatched token.

A database model row whose `model_name` matches a YAML deployment is skipped;
the YAML deployment is the one that is served. Spend logs flush every 60
seconds. The Postgres pool allows at most 20 connections. Session affinity is
a stateless hash of the session id and has no TTL. Datadog receives logs for
successful requests and terminal failures, excluding cache hits/master-key
requests; failure messages redact credential-shaped values. Redis/PG service
failure callbacks are not supplied by this request hook.

Bedrock Converse always merges adjacent user/assistant roles and parallel tool
results while retaining content metadata and signed reasoning. Enable
`oxllm_settings.modify_params: true` for undeclared historical tools,
tool-result/follow-up continuation turns, and unsigned tool-history thinking
repairs. A deployment's `modify_params` explicitly overrides the global value.

| Setting | Apply method |
|---|---|
| `model_list`, aliases, routing/retry/fallback knobs, request-time limits, cache TTL, config-derived hooks | File reload |
| Master/JWT settings, `trusted_proxy_cidrs`, `allowed_ips`, HTTP pool/connect/IPv4 settings, guardrails | File reload |
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
