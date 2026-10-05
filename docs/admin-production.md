# Admin UI operations

The admin binary is a separate server-side client of the proxy management API.
Its browser session does not contain the proxy master key. Use a DB-enabled
proxy (`--features db` plus `general_settings.database_url`) for keys, users,
teams, DB-managed models and spend pages.

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

## Historical readiness plan

> The original MVP plan below is retained for design history. Its "today",
> priority/status and suggested-next-PR statements are historical and superseded
> by current source/tests and validation reports. They are not current defects
> or release readiness evidence.

Goal: make **oxllm-admin** a reliable MVP console for operating the proxy —
keys, users/teams, spend/billing visibility, models, and proxy health — with a
clear UI and stable local/prod deploy.

---

## Why “Unreachable” shows up today

Dashboard **Proxy Status** is driven by `ProxyClient::health()` →  
`GET {proxy}/health/readiness` with the master key.

| Cause | What you see |
|--------|----------------|
| **Proxy process stopped** | Unreachable (connection error). Local nohup sessions have been dying after ~120s when the launcher receives SIGTERM / graceful shutdown. |
| **Readiness 503** (Redis/DB ping fail) | Treated as failure → Unreachable, even if process is alive. |
| **Wrong `OXLLM_PROXY_URL` / master key** | Connection or 401 → Unreachable or login-ish failures. |

**MVP fix (P0):**
1. Treat **liveness** (`GET /health`) as Online/Offline.
2. Call **readiness** separately; show checks (db/redis) as subtext, not as “Unreachable” for the whole proxy.
3. Surface the last error string in the UI (“connection refused”, “status 503”, …).
4. Ship a **`scripts/dev-stack.sh`** (or document systemd/docker) so proxy+admin stay up for local use.
5. Docker: `depends_on` + healthchecks already exist; ensure admin waits on proxy health.

---

## Current capability matrix

### Proxy management API (exists)

| Area | Endpoints |
|------|-----------|
| Keys | generate, list, info, update, rotate, delete (soft), purge, block/unblock |
| Teams | add, list, info, update, delete, block/unblock |
| Users | add, list, info, update, delete |
| Spend | logs, logs/v2, total (key/team-scoped), calculate, global/spend, global/activity, by-key/team/user/org/model |
| Models | list (incl. DB `model_id` for dynamic rows), info, new, update, delete |
| Config | `GET /config` (secrets redacted), `GET /routes` |
| Routing | `GET /routing/state` (live strategy, pre-call checks, per-deployment cooling/health/bandit snapshot) |

### Admin UI today

| Page | Works | Gaps |
|------|--------|------|
| Login / session / CSRF | Yes | Optional password only; no SSO |
| Dashboard | Partial | Health false-negatives; spend needs DB |
| Keys | Create / list / edit / rotate / disable / re-enable / purge | Copy-expiry UX polish; no dedicated key detail drawer |
| Teams | Create / list / edit / delete | No detail view |
| Users | Create / list / edit / delete | No detail view; no user-scoped spend bar (the proxy has no such rollup endpoint) |
| Spend | List + filters + pagination + CSV export | No team/user rollups charts |
| Models | List + costs + create/delete (DB-backed rows; config-managed rows are read-only by design) | — |

**MVP definition of done:** an operator can run the proxy day-to-day without `curl`:
create/disable keys, attach budgets/RPM, manage teams/users, inspect spend and
daily cost, confirm proxy + backends healthy, see configured models.

---

## Workstreams

### P0 — Make it usable with a running proxy (this week)

| # | Item | Notes |
|---|------|--------|
| P0.1 | Fix health semantics | Liveness vs readiness; never label “alive but redis down” as Unreachable only — show “Degraded”. |
| P0.2 | Dev stack script | Start redis (if needed), proxy (`--features db`), admin; trap signals; log to `/tmp` or `logs/`. |
| P0.3 | Models page robustness | Already fixed empty-cost template crash; keep regression tests. |
| P0.4 | Logo in UI | Brand mark in sidebar + favicon from `assets/logo.png` / `static/logo.png`. |
| P0.5 | Dashboard clarity | Status pill + last check time + short error; spend empty state explains “needs DB + traffic”. |

### P1 — Complete admin CRUD for keys / teams / users

| # | Item | API support |
|---|------|-------------|
| P1.1 | Key detail drawer | `GET /key/info` — budgets, RPM/TPM, team, created |
| P1.2 | Key edit | **Needs proxy API** `POST /key/update` (budget, rpm, tpm, alias, team, active) |
| P1.3 | Team delete + detail | `DELETE`/`POST /team/delete`, `GET /team/info` |
| P1.4 | User detail (+ delete if product wants) | `GET /user/info`; delete endpoint if required |
| P1.5 | Confirm dialogs + flash toasts | Consistent success/error after mutations |
| P1.6 | Empty states with CTAs | “Create first key” → open form |

### P2 — Billing / cost / spend (ops-ready)

| # | Item | Notes |
|---|------|--------|
| P2.1 | Spend filters UX | Date range presets, model, key, team; clear filters |
| P2.2 | Totals strip | Page-level total from `/spend/total` or global spend |
| P2.3 | Export CSV | Server-side stream or client build from current page + “export all” via API |
| P2.4 | Dashboard charts | Daily spend (existing bars); optional top models by cost |
| P2.5 | Cost calculator | UI for `POST /spend/calculate` (model + tokens → $) |
| P2.6 | Require DB for admin | Document: admin without `database_url` is view-models-only |

### P3 — UI polish (clearer, calmer)

| # | Item |
|---|------|
| P3.1 | Logo + product name in sidebar; page titles consistent |
| P3.2 | Nav grouping: Overview / Access / Billing |
| P3.3 | Loading skeletons / HTMX indicators on all panels |
| P3.4 | Tables: sticky header, denser type, mono for IDs |
| P3.5 | Dark mode toggle (already started) — verify contrast |
| P3.6 | Mobile: sidebar overlay already sketched — test + fix |
| P3.7 | Prefer local/static assets over CDN for prod hardening (optional) |

### P4 — Production ops for admin service

| # | Item |
|---|------|
| P4.1 | `ADMIN_UI_PASSWORD` required to bind a non-loopback address (the admin service refuses to start on a public bind without it) |
| P4.2 | `cookie_secure=true` behind TLS; document reverse-proxy headers |
| P4.3 | Structured logs + request IDs on admin |
| P4.4 | Docker Compose `admin` profile healthcheck + restart policy |
| P4.5 | Helm values for admin Deployment/Service (chart already partial) |
| P4.6 | Rate-limit login (LoginBrake exists) — metrics/alerts |
| P4.7 | Audit log of admin actions (optional; proxy already has spend logs) |

### Out of scope for this MVP (explicit)

- Editing `model_list` / live config from UI (keep YAML + hot-reload)
- Multi-tenant admin RBAC (admin = master-key BFF only)
- Full SSO/OIDC (can follow after password gate)
- Replacing LiteLLM’s entire UI surface

---

## Suggested PR sequence

1. **PR-A — Health + stack + logo**  
   P0.1–P0.5; no new proxy APIs.

2. **PR-B — Admin CRUD polish**  
   P1 using existing APIs; add `/key/update` in proxy if missing.

3. **PR-C — Spend / billing UX**  
   P2.

4. **PR-D — UI polish + prod flags**  
   P3 + P4.

Each PR: unit tests for templates/handlers, e2e against httpmock proxy where
routes exist, manual pass on `examples/config.local-ui.yaml`.

---

## Local verification checklist

```bash
# Terminal 1 — proxy (DB required for keys/spend)
cargo run --features db -- serve --config examples/config.local-ui.yaml --port 4000

# Terminal 2 — admin
export OXLLM_MASTER_KEY=...   # same as proxy
export OXLLM_PROXY_URL=http://127.0.0.1:4000
cargo run --features admin --bin oxllm-admin -- serve --listen 127.0.0.1:8080
```

- [ ] Dashboard shows Online when proxy up  
- [ ] Degraded when Redis down but process up (after P0.1)  
- [ ] Create key → appears in list → delete works  
- [ ] Create team/user  
- [ ] Spend page empty state, then logs after a chat call  
- [ ] Models list with costs when price table has entries  

---

## Success criteria (MVP ship)

1. Operator can administer **keys, teams, users** without curl.  
2. Operator can see **spend totals, activity, and filtered logs**.  
3. Operator can see **models + pricing** and **proxy health** (honest states).  
4. Admin runs as separate process; master key never in the browser.  
5. Docker/Helm path documented; password required in prod.

---

## Immediate next step

Implement **PR-A** (health semantics, logo, dashboard clarity, dev-stack script)
so “Unreachable” stops lying and the UI feels like one product with the proxy.
