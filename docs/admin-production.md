# Admin UI operations

The admin binary is a separate server-side client of the proxy management API.
Its browser session does not contain the proxy master key. Every management
page — keys, users, teams, models, and spend — requires a DB-enabled proxy
(`--features db` plus `general_settings.database_url`); without `db`, only the
login and health shell is usable.

## Current deployment path

- **Compose:** start the base Redis/SQLite stack, then include
  `docker-compose.admin.yml` with `--profile admin`. Set `ADMIN_SESSION_SECRET`
  (16+ characters), `ADMIN_UI_PASSWORD` and the same `OXLLM_MASTER_KEY` as the
  proxy. Templates and browser assets are embedded in the binary; the image
  has its own health probe; Compose waits for proxy health and restarts exited
  services.
- **Helm:** configure `admin.enabled`, the image you published and session/
  password secrets. The admin connects to the proxy Service port. See the
  [chart README](../helm/oxllm/README.md) for external Secret references.
- **Local source:** `bash scripts/dev-stack.sh` builds the proxy with DB support
  and the admin binary. Supply Redis on loopback or choose another config.
  Existing listeners are not stopped implicitly. This script is for local
  development, not service supervision in production.

Non-loopback binds require a password. `ADMIN_COOKIE_SECURE` defaults to true
in the binary/chart. Set it false only for local HTTP; production TLS termination
should retain secure cookies. Admin env/config changes require a restart.

`OXLLM_PROXY_URL`/`proxy_url` must be an absolute `http(s)` URL with a host.
Credentials in the URL, query strings and fragments are rejected; base paths
are allowed for reverse-proxied deployments. Plain `http` to a non-loopback
host is accepted but logs a cleartext-credentials warning — use `https` for
remote endpoints.

List pages load up to 1000 rows; when the proxy reports more, the UI marks the
list “incomplete” (`truncated`). The notice means only the loaded rows are
displayed: client-side search and pick lists cover those rows, so narrow the
filters to bring the rest into view — a missing row is not proof it does not
exist.

Check both process and dependency health: admin `/health` tests the UI process,
proxy `/health` tests the proxy process, and proxy `/health/readiness` tests
configured DB/cache readiness. A healthy admin process alone does not prove
management access. For packaged-runtime checks:

```bash
OXLLM_ADMIN_URL=http://127.0.0.1:8080 OXLLM_DB_SMOKE=1 bash scripts/smoke-docker.sh
```

The default smoke checks make no provider requests. See
[operations](operations.md) for backup/restore, credentials, rotation and
restart behavior.
