# oxllm Helm chart

Installs the proxy and, optionally, the admin UI. It does not provision Redis,
Postgres or an ingress. Use images you have built and made available to your
cluster; defaults `oxllm:0.1.0` / `oxllm-admin:0.1.0` are local image names.
Empty image tags resolve to `Chart.appVersion`.

## Basic install

Create the Secrets referenced below in the target namespace, then save a values
file. Replace the example registry with your own published image location:

```yaml
image:
  repository: registry.example.com/your-team/oxllm
  tag: "0.1.0"
envSecretRefs:
  OXLLM_MASTER_KEY: {name: proxy-auth, key: master-key}
  OPENAI_API_KEY: {name: provider-auth, key: openai-api-key}
```

```bash
helm lint ./helm/oxllm -f my-values.yaml
helm upgrade --install oxllm ./helm/oxllm -f my-values.yaml
```

For private registries, attach image-pull credentials to the ServiceAccount
selected by `serviceAccount.name` (set `create=false` to use an existing one).
The default config exposes `gpt-4o` via OpenAI. Replace `config.model_list` to
use another provider. Only variables actually referenced in `config` (and the
admin master key when enabled) are checked at render time. With `envFrom`,
values cannot be inspected until the pod starts; the config loader catches
missing references then.

`env` is a generic name/value map stored in a chart-managed Secret. Use
`envSecretRefs` for explicit per-variable Secret references, or `envFrom` for
Kubernetes SecretRef/ConfigMapRef sources. Avoid duplicate names. `OXLLM_PORT`
and `OXLLM_REQUIRE_AUTH` are controlled by chart values; persistence also owns
`OXLLM_DATABASE_URL` and `OXLLM_AUTO_MIGRATE`. If using `env`, remember its values
also become part of Helm release metadata.

## Persistence and multiple replicas

```yaml
replicas: 3
config:
  oxllm_settings:
    request_timeout: 120
    cache:
      type: redis
      redis_url: os.environ/REDIS_URL
  general_settings:
    allow_private_api_base: true
envSecretRefs:
  OXLLM_MASTER_KEY: {name: proxy-auth, key: master-key}
  OPENAI_API_KEY: {name: provider-auth, key: openai-api-key}
  REDIS_URL: {name: redis-auth, key: url}
persistence:
  enabled: true
  existingSecret: postgres-auth
  databaseUrlSecretKey: url
```

Use an external Postgres URL in that Secret. No SQLite PVC is created by this
chart. `persistence.databaseUrl` can instead create a chart-managed connection
Secret; supply exactly one source. Custom `databaseUrlSecretKey` values work for
either source. Persistence adds `general_settings.database_url:
os.environ/OXLLM_DATABASE_URL` to the mounted config, passes the same env to the
main and migration containers, and sets `OXLLM_AUTO_MIGRATE=false` for `serve`.

`persistence.migrate=true` (default) runs a migrate init container **per pod**.
The DB migration lock coordinates concurrent starts. For an externally managed
migration step or separate DDL role, migrate with the same image before the
rollout and set `persistence.migrate=false`. Database availability and schema
compatibility remain prerequisites. Enabling persistence does not grant DDL
rights or create the target database.

`replicas > 1` fails rendering unless `config.oxllm_settings.cache.type=redis`;
otherwise rate limits/budgets would use per-process counters. Redis on a private
cluster network also needs `allow_private_api_base: true`. Use `redis_url` for
Redis auth/TLS; plain host/port configuration is unauthenticated plaintext.

## Admin UI

```yaml
admin:
  enabled: true
  image: registry.example.com/your-team/oxllm-admin
  tag: "0.1.0"
  sessionSecretRef: {name: admin-auth, key: session-secret}
  passwordSecret: admin-password # Secret key must be "password"
  cookieSecure: true
```

The proxy master key comes from `env.OXLLM_MASTER_KEY`, the matching
`envSecretRefs` entry, or `envFrom`. Set the same key in proxy config. The admin
uses the proxy Service's port, independently of the proxy container port.
Alternatively, `admin.sessionSecret` (16+ characters) and `admin.uiPassword`
create chart-managed secrets. Every management page — keys, teams, users,
models, and spend — needs a DB-enabled proxy; without persistence only the
login/health shell is usable. Terminate TLS before exposing the UI and keep
secure cookies on.
For local HTTP port-forward testing only, use `admin.cookieSecure=false`.

## Updates, health and security context

- Config is mounted at `/etc/oxllm/config.yaml` as a directory volume. Helm
  changes to config or chart-managed Secrets change pod-template checksums,
  triggering rollouts. Externally managed Secret content is not hashed: bump
  `rolloutRevision` when rotating it, or restart both Deployments.
- Proxy grace defaults to `max(shutdown_timeout_secs or 30, request_timeout or
  600) + 30` seconds. An explicit `terminationGracePeriodSeconds` must exceed
  the drain budget; leave time for background flushes. There is no sleep-only
  preStop hook consuming that grace. Align any load-balancer drain policy.
- Probes use the named `http` port. Changing `port` also changes `OXLLM_PORT`;
  `service.port` is independent. Admin probes check its own process; proxy
  readiness checks configured backend health.
- Proxy, admin and migration containers default to nonroot UID/GID 10001,
  `allowPrivilegeEscalation=false`, dropped capabilities and `RuntimeDefault`
  seccomp. The root filesystem is read-only; `/tmp` is a writable `emptyDir`.
  These settings support the restricted Pod Security Standard. Adapt the
  context for an alternate image rather than relying only on its Docker USER.
- API token automount is disabled at pod level even with an existing or default
  ServiceAccount.
- If enabling `networkPolicy`, the default egress permits provider HTTPS (443)
  plus cluster DNS. Add explicit egress for your Redis/Postgres/OTLP ports.
  `ingress.from` restricts sources when set; an empty list permits any source
  on the service port. `denyMetadataEndpoint` excludes metadata IPs only in
  rules without a custom `to`; custom destinations and other policies remain
  the operator's responsibility. Kubernetes NetworkPolicy rules are additive.

See [operations](../../docs/operations.md) for backups, migrations, config reload
limits, cache privacy and release configuration. Run
`bash scripts/check-deployment-artifacts.sh --require-tools` from the repo root
for nonbuilding render checks.
