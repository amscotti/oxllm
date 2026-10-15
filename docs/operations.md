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
Directory aliases share one OS watch registration. The bounded event queue
coalesces overflow into a rescan, preserving changes without blocking callback
threads during watcher startup or restart.

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
`api-version` query. Native Claude Messages routes use `x-api-key`. Compatible
vendor routes on unified `.services.ai.azure.com` resources use `api-key`;
explicit `azure_ad_token` uses Bearer authentication. Native Cohere embedding
deployments also recognize unified Foundry hosts, or `api_type: azure_ai` for
custom gateways, and send `input` to `/models/embeddings`. Azure ML
`llm_provider-num_tokens` headers are honored for embedding usage and cost.
See [the Foundry configuration sample](../examples/config.foundry.yaml).

Azure OpenAI deployments identified by GPT-5/GPT-6/o-series names
translate `max_tokens` to `max_completion_tokens`; an explicit modern limit
wins. This includes GPT-5/GPT-6 chat variants, deployment labels containing
those family names, and `gpt5_series` labels. Token-limit translation does not
enable reasoning-specific sampling restrictions for chat models.
Set `model_info.base_model` for opaque deployment aliases that need a
capability hint. An explicit nonreasoning base-model hint takes precedence over
deployment-name inference.

Azure/Azure AI Model Router selectors (`model_router/<deployment>` or
`model-router/<deployment>`) retain their configured identity for routing and
send only the deployment name upstream. Settlement prices the model reported
by Azure, then adds the canonical Model Router infrastructure rate once
($0.14 per million input tokens in the 2026-10-08 catalog). Cached-read prompt
tokens still incur that fee. A canonical fee-only response or an unpriced
selected model charges the fee once; explicit deployment/request cost overrides
are all-in and take precedence, including explicit zero rates. Opaque deployment
names can use `model_info.base_model: model-router`; merely containing the words
`model-router` does not select this billing mode. Budget admission for an unpriced
deployment reserves the nominal $0.01 floor plus the known infrastructure fee;
canonical fee-only cards reserve their effective fee once.
JSON headers, SSE settlement, spend records and exporters use the same result.

OpenRouter and Perplexity finite, nonnegative `usage.cost` receipts are
authoritative over local catalog rates when no deployment price override is
configured; a measured zero is a valid charge. Explicit rate overrides and
pre-computed cost overrides retain precedence. An override with every effective
rate and context tier set to zero remains free when only a receipt is reported,
without inventing token counts. Nonzero cache/tier prices still require their
usage counters; existing best-effort receipt fallback remains available when
local cost cannot be computed. Static OpenRouter cards remain available for
admission estimates, cost previews and missing/invalid-receipt fallback.
Only measured receipts enter the receipt channel. Receiptless JSON/SSE requests
use the serving-model card before the concrete-deployment fallback, so a locally
computed fallback cannot override another reported model's price.

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

Flagged OpenAI-shaped deployments also accept these selectors. Mantle GPT-5.6+
and flagged Foundry GPT deployments inject `prompt_cache_breakpoint:
{mode: explicit}` on the last eligible content block instead of Anthropic
`cache_control`. Missing/null `prompt_cache_options` becomes `{mode: implicit}`
when a new mark is inserted. Existing marks and explicit options win; assistant
messages are skipped. Request-level point lists override deployment points;
malformed request entries are ignored, while deployment selectors remain typed
and validated. Request decorations such as Anthropic TTL controls do not alter
OpenAI selectors. The four-block cap counts message/content annotations, not
schema properties or metadata fields with matching names. Foundry OpenAI bodies
remove Anthropic cache annotations before inserting native breakpoints.
`supports_prompt_cache_breakpoint: false` disables the dialect;
only boolean `true` opts an unmodeled deployment in. Provider-scoped catalog
flags never license another provider. Native OpenAI defaults require the OpenAI
API host or explicit caller cache options on a compatible custom host. An exact
regional hosted-model card is authoritative, including unflagged GovCloud cards;
commercial flags do not opt those deployments in.

Deployment `*_above_<N>k_tokens`/`*_above_<N>_tokens` prices apply to all tokens
after the prompt crosses the threshold. At 272,000 tokens the base applies;
at 272,001 the tier applies. `tier_threshold_is_inclusive: true` opts into an
inclusive boundary. Cache-creation tokens count toward context length,
including standalone one-hour details. Malformed tier keys/rates are warned.

Named `retry_policy` entries map onto status counts and structured categories.
Timeouts map to 408; actual upstream 504s belong to the 5xx category.
The internal-server-error category also covers Anthropic overload status 529;
explicit numeric status policies retain precedence. Streamed Anthropic timeout
and not-found frames preserve HTTP 504 and 404 respectively.
`BadRequestErrorRetries` and `AuthenticationErrorRetries` accept only zero.
Their aliases set retry counts only and never inject allowed-fails thresholds.
Zero disables same-deployment retries. Deployment-fault 401/403/404 recovery
still uses authorized siblings/configured fallback groups. Generic 400/422
request errors do not cool deployments under global thresholds or retry
categories; an explicit per-status `allowed_fails_policy` remains an operator
opt-in to that cooldown/failover behavior.
Upstream-classified context-window and content-policy errors retain their
specific retry/fallback paths. Exact `cyber_policy` codes/types are policy
rejections (400); generic invalid-prompt errors are not promoted to that class.
Committed native error frames retain their wire bytes, including nested
`response.failed` errors. Terminal relayed SSE errors use the same explicit
per-status cooldown opt-in as other client failures; classified
context-window/content-policy errors keep their dedicated cooldown path.
Realtime upstream 409 rejections count as retryable faults; actual upstream
400/422s count only with an explicit status policy. Local malformed-client,
configuration and transform errors remain outside those policies. Realtime 424
failed-dependency observations retain their existing treatment.
Explicit numeric retry entries win over aliases. Nonzero content-policy
retry parity is not established by the zero-retry compatibility tests.
Exhausted deployments are excluded for the remainder of the current request,
even with `cooldown_time: 0`; future requests may reconsider them immediately.

Chat, Messages and Responses requests can override `fallbacks`,
`context_window_fallbacks` and `content_policy_fallbacks` with model-name lists
or the configured rule shape. Null inherits defaults; an empty list disables
that chain. `disable_fallbacks: true` disables all model fallback, including
unknown-model defaults. `max_fallbacks` can lower the server cap; zero disables
fallback walks. Every target retains key/team/user ACL and budget checks.
Embeddings share request-scoped generic fallback and retry controls.

An explicit status/exception retry policy wins over request `num_retries`, then
deployment `num_retries`, then the model-group/global default. Zero prevents
same-deployment retries. Request `num_retries` must be an integer from 0 through
10; larger values return 400 before provider calls. Trusted server/deployment
retry policies retain their configured budgets. The one-shot thinking-signature
repair is included in `x-oxllm-max-retries` and the attempted-retry count.
Gateway controls, including cache selectors, top-level `tags` and `metadata.tags`, are removed
from final provider bodies after configured overrides and allowed-parameter
restoration. Raw bodies are validated before Messages/Responses/counting
conversion; opt-in fallback status summaries survive API bridges.

Tag headers accept native `x-oxllm-tag` / `x-oxllm-tags` and migration aliases
`x-litellm-tags` / `x-litellm-tag`. Merge order is top-level body tags, metadata
tags, native singular/plural headers, then legacy plural/singular headers.
Repeated and comma-separated values are trimmed, deduplicated and bounded to
16 tags of 128 bytes each. Colons within attribution pairs are literal, including
tag-management names, budget counters and alert-reset scopes.

Responses `text.format` supports `text`, `json_object` and `json_schema` on the
chat bridge. JSON-schema names, descriptions, strictness and schemas reach the
provider's existing structured-output transform and remain part of exact and
semantic cache identity. Invalid formats return 400 before provider calls.
Schema names are trimmed, bounded to 64 characters and restricted to ASCII
letters/digits/underscores/hyphens. The chat bridge does not translate
`text.verbosity`; schema objects still pass through the provider's own supported
JSON-schema subset checks.
Converted Anthropic unary tool calls carry indexes into the visible tool-call
array after hidden JSON-helper removal.

Vertex Anthropic deployments serving schema requests require the appropriate
`structured_outputs` permission under
`constraints/vertexai.allowedPartnerModelFeatures`. Verify the model feature
allowlist and a schema request on each intended deployment. Provider policy
errors retain their status; output constraints are not removed to evade them.

Locally unresolved inference models return 400 after fallback handling, including
native embeddings, token counting and realtime setup. OpenAI-shaped errors retain
`code: model_not_found`; translated Messages errors use Anthropic's
`invalid_request_error` envelope. Configured groups whose deployments are
temporarily unavailable return 429 `proxy_rate_limited`, preserving the
distinction between an invalid selector and retryable unavailability.
Missing resources and genuine upstream 404s retain 404; local upstream deadlines
remain 504 with a timeout message.

Streaming startup holds role/usage metadata until content or a normal terminator
arrives. A pre-content disconnect, timeout or provider error frame re-enters the
same retry/fallback chain; malformed provider preludes are retryable 502s and
failed-attempt metadata never reaches clients. Startup retains at most 16 MiB
of coalesced upstream bytes. Cancellation before commitment releases the
deployment in-flight slot and acquired router-budget estimates; refunds have
one owner even when cancellation interrupts a multi-scope release. Once
the stream is committed, later errors terminate it without restarting output.
Header waits and stream reads use the same idle timeout precedence: request
`stream_timeout`, request `timeout`, deployment `stream_timeout`, global
`request_timeout`, then 60 seconds. Deployment `timeout` remains a unary
setting and does not shorten streams. Request timeouts accept finite positive
fractional seconds. Healthy chunks/keepalives do not trigger a total streaming
deadline. Native `/anthropic/v1/messages` passthrough remains single-attempt;
translated `/v1/messages` participates in chat retry/fallback orchestration.

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

Vertex Gemini chat and OpenAI-compatible embeddings send `google/<model>` as
the publisher-qualified selector. Configured bare IDs, `google/<model>` and
`publishers/google/models/<model>` are accepted. Native Vertex Claude routes
use an Anthropic publisher URL with a single model-id segment.
Direct Gemini and Vertex Gemini JSON/SSE usage is normalized when the endpoint proves that reasoning
was counted additively: `total = prompt + completion + reasoning`. Reasoning
is folded into completion once, retaining its detail count, so client usage,
TPM and output billing agree. Integer prompt/total counters are required; a
missing completion counter is recoverable when the remaining counters prove
its value. Already-normalized, inconsistent or unknown usage is not folded
speculatively. Completion zero is inserted for measured zero output only when
reasoning is absent or zero.

Vertex multi-regions `us` and `eu` use `aiplatform.us.rep.googleapis.com` and
`aiplatform.eu.rep.googleapis.com`; `global` and regional hosts retain their
normal forms. `vertex_location` and counting-region overrides use nonempty
lowercase letters/digits/hyphens, including with a custom API base; uppercase
or path/query-bearing names are rejected. Gemini requests discard foreign
`thinking_blocks`, retaining
the caller's single `reasoning_content` representation and Google's native
provider fields/tool-call IDs.

### Provider token counting

`POST /v1/messages/count_tokens` supports direct Anthropic, Vertex Claude and
Foundry Claude deployments. System prompts and native tool definitions are
forwarded, null fields are omitted, and malformed tool lists fail before any
provider request. OpenAI `image_url` blocks are normalized to Anthropic image
sources, including images nested in tool results. Counts incur no generation
spend. A leading system/developer message run follows the caller's top-level
system prefix, preserving text order and cache controls. Later system reminders
retain their position. Vertex removes `@default`/snapshot suffixes only for
counting and calls the publisher's `count-tokens:rawPredict` endpoint. Set
`vertex_count_tokens_location` when counting uses a different region; otherwise
the deployment location applies, with `us-east5` when no location is configured.
The stable Anthropic counting API does not need a token-counting beta header;
explicit caller/OAuth feature betas are preserved.

`POST /utils/token_counter` remains local by default. `call_endpoint: true`
uses a supported Claude counting deployment and falls back to local estimation
if the provider is unavailable. `disable_token_counter: true` prohibits local
estimation on every path, including default/local-only calls and unsupported or
missing native counters. A missing native result returns 503 with code
`token_counting_disabled`; native errors retain their status. Both flags must be
boolean or null. Model ACLs and the utility work limit apply to both paths.

### Anthropic workload identity federation

See [the federation configuration example](../examples/config.anthropic-wif.yaml).
An explicit static `api_key` takes precedence. Otherwise configure
`anthropic_identity_token_file`, or `anthropic_identity_token` as
`oidc/file//absolute/path` or `oidc/env/NAME`. The standard
`ANTHROPIC_IDENTITY_TOKEN_FILE`/`ANTHROPIC_IDENTITY_TOKEN` environment sources are
also supported. Source precedence is deployment file, deployment reference,
environment file, `OXLLM_ANTHROPIC_IDENTITY_TOKEN_REF`, then the raw assertion in
`ANTHROPIC_IDENTITY_TOKEN`. Raw assertions are not accepted in the deployment
reference parameter. `anthropic_identity_source` or an ambient
`ANTHROPIC_IDENTITY_SOURCE` requires a custom registered resolver; the built-in
resolver does not implement Keycloak/internal issuer acquisition.

Federation requires `anthropic_federation_rule_id` and
`anthropic_organization_id`, with vendor environment fallbacks
`ANTHROPIC_FEDERATION_RULE_ID` and `ANTHROPIC_ORGANIZATION_ID`. Missing IDs return
a named 401 before file access or an exchange. Optional
`anthropic_service_account_id` and `anthropic_federation_workspace_id` select the
grant scope. `anthropic_disable_workload_identity_federation: true` selects static
key resolution. These credentials belong to deployments and cannot be supplied
through generation request bodies.
Validation includes the federation-disable flag and runs before body conversion.
A credential 401 can still cool a bad deployment and select an authorized
sibling/configured fallback; failing locally means no assertion read/exchange
for malformed configuration, not disabling normal authorized routing recovery.

The built-in resolver exchanges JSON JWT-bearer grants for expiry-checked Bearer
tokens, coalesces concurrent acquisition, and reuses unexpired grants. File reads
and token responses are capped at 64 KiB. The assertion cap applies to raw bytes
before UTF-8 conversion or whitespace trimming, so padding cannot bypass it.
Identity assertions must rotate before
the minted token expires; no expired grant is forwarded. OAuth beta flags are
merged without empty segments or duplicates on translated/native chat and
counting paths. Assertion source buffers and the owned serialized JSON grant
buffer are zeroized when dropped, including after transport clones release the
last owner; this does not cover copies inside HTTP/TLS libraries or headers.
Additional issuer/identity-source implementations use the credential-resolver registry.

Exchange URLs require HTTPS. `api.anthropic.com` is trusted by default;
`OXLLM_ANTHROPIC_WIF_ALLOWED_HOSTS` adds comma-separated exact authorities,
optionally pinning a port. Identity files default to `/var/run/secrets` and
`/run/secrets`; `OXLLM_OIDC_ALLOWED_CREDENTIAL_DIRS` selects allowed roots using
the platform path separator (`:` on Unix). These are server-owned environment
settings. Restart after changing them.

Bedrock Cohere/Titan text embeddings use InvokeModel with SigV4 over the exact
sent bytes. Runtime URLs use the resolved signing region. Usage comes from
the native body, then the invocation token-count header, then text estimation
when metering is absent. Mantle token scopes follow an official custom base's
host region so an ambient region cannot create a mismatched token.

Converse omits `stopSequences` for Grok and numeric GPT 5.6+ families, using
known regional/resource IDs or declared underlying-model hints for opaque
profiles. Tool-result images remain nested on supported families and become
sibling image blocks on GPT; tool IDs, status, cache boundaries and ordering
are retained. Image-only/empty results receive nonempty text, and tool references
carry their tool names. Unsupported remote or malformed image sources return
an error rather than losing content.

Voyage credentials whose resolved bearer starts with `al-` select
`https://ai.mongodb.com/v1` for embeddings/rerank and the corresponding origin
for native relays. Endpoint selection uses the authentication snapshot; explicit
operator `api_base` values always win and native paths/queries are retained.

A database model row whose `model_name` matches a YAML deployment is skipped;
the YAML deployment is the one that is served. The background spend writer
drains received entries in bounded batches and flushes remaining work at
shutdown; its periodic flush tick defaults to 60 seconds. Known pre-commit
PostgreSQL serialization/deadlock/lock-timeout and SQLite BUSY/LOCKED failures
reuse the same event IDs across four total attempts, with 25/50/100 ms backoff
(175 ms maximum added sleep per batch). Permanent errors and commit-time
ambiguity bypass those retries. The Postgres pool allows at most 20 connections;
`database_connection_pool_limit` does not tune this pool. Session affinity is
a stateless hash of the session id and has no TTL; changes to group membership
can remap sessions. `optional_pre_call_checks: [session_affinity]` selects this
strategy when the global strategy is omitted. An explicitly configured strategy
takes precedence. Unknown/unconfigured check diagnostics are shared by
`validate` and `serve`, and only enabled filters enter the routing pipeline.
`router_budget_limiting` is accepted as an alias for the native `budget_limiter`
check. `enforce_model_rate_limits` is accepted as a declaration of deployment
RPM/TPM enforcement, which runs for every selected deployment independently of
the optional-filter list.

`oxllm_settings.datadog_config` sends successful-request and terminal-failure
logs, excluding cache-hit/master-key request events. Failure messages redact
credential-shaped values. Adding `service_callback: [datadog]` enables separate
dependency-health failure logs using that same intake configuration. The server
checks configured DB/cache backends every 15 seconds with the readiness probe's
500 ms deadlines. It logs an outage once, repeats after 60 seconds, and rearms
after recovery. These events are independent of callers and billing. Hot reload
updates the callback configuration; shutdown stops the monitor. Missing intake
credentials and unsupported service backends produce validation/startup warnings.

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

### Response metadata and deployment attribution

Direct chat JSON/SSE responses and embeddings echo the client's requested model
name. When a configured/default fallback serves the request, both JSON and SSE
identify its public model group. `x-oxllm-requested-model` records the original
alias and `x-oxllm-model-group` records the serving group, making fallback
reporting consistent across response shapes.
Billing and spend telemetry use the serving provider/model and deployment's
rate card. Completion usage includes
`completion_tokens_details.reasoning_tokens: 0` when no reasoning count was
reported; reported nonzero counts are retained.

`x-oxllm-*` is the primary client contract, including for browser clients using
the configured proxy CORS policy. Selected `x-litellm-*` response names are
additive compatibility aliases with the same values; native clients use only
`x-oxllm-*` headers and native configuration names.

| Primary header | Meaning |
|---|---|
| `x-oxllm-request-id` | Gateway-generated correlation id; `x-request-id` and `x-litellm-call-id` contain the same value on success, errors, cache hits and streams |
| `x-oxllm-deployment-id`, `x-oxllm-model-id` | Selected deployment's `model_info.id` |
| `x-oxllm-model-name` | Selected deployment's provider/model selector, independently of its pricing hint |
| `x-oxllm-model-group` | Public group that served, including a fallback group |
| `x-oxllm-requested-model` | Original client-requested alias |
| `x-oxllm-attempted-retries` | Same-deployment retries across attempted groups, including history repair |
| `x-oxllm-attempted-fallbacks` | Configured/default fallback groups attempted |
| `x-oxllm-max-retries` | Effective retry budget when the serving deployment needed a policy retry |
| `x-oxllm-fallback-errors` | Bounded status summaries when `include_fallback_errors: true` was requested |
| `x-oxllm-response-duration-ms` | Total time from HTTP request admission to response-header construction |
| `x-oxllm-key-spend` | Tracked chat/embedding key's current budget-window counter after unary settlement; may include other in-flight reservations |
| `x-oxllm-upstream-duration-ms` | Sum of upstream attempts, including failed attempts |
| `x-oxllm-overhead-duration-ms` | Total time outside upstream attempts, including backoff and response conversion |

Streaming timing headers stop at response commitment (first content for
transformed streams); complete generation
duration is available from final telemetry. Upstream `x-ratelimit-*` headers
are passed through from the serving leg. Local per-principal quota headers take
precedence when both provide the same header.

Key-spend snapshots use the same counter backend as budget enforcement. Redis
counters are shared across replicas; local in-memory counters are per replica,
including the counter layer used with blob response caches. Cache hits expose
the tracked counter without adding spend. Removing both key and global caps
omits the header even if a previous counter remains. An absent counter during
reset or a failed counter read also omits it. Live upstream streams omit the
snapshot; buffered cached SSE replays can expose the tracked counter.
The migration alias `x-litellm-key-spend` has the identical value and omission
rules. Both correlation and spend aliases are exposed by the proxy CORS policy.
Upstream-provided key-spend snapshots are stripped from native relays; response
cost and token receipts remain available for the chained-billing contract.

Entirely free primary groups use a zero admission estimate only when every
matching deployment's effective override, inherited bucket and context tier
proves a zero price. Concrete deployment selection clears the admission hint;
paid fallback pricing must top up the reservation before reaching the provider.
Zero budgets retain block-all semantics, and cached responses still bypass spend
reservation through the normal cache-first hook order.

Configure a unique `model_info.id` for durable identities across restarts and
replicas. Otherwise oxllm generates a UUID and preserves it across in-process
reloads and DB refreshes. DB-managed deployments default to their stored model
row id. IDs are public identifiers independent of internal routing fingerprints.
A response-cache hit has no new deployment selection and omits deployment-id
headers.
Cache hits retain the original public response model and serving group;
retry/fallback counters describe the current request and are zero. Cached
rate-limit snapshots and deployment selections are not replayed. Cache-hit
overhead excludes the response-cache lookup time.

Spend logs include nullable `deployment_id`; historical records retain null.
Both `/spend/logs?deployment_id=region-a` and
`/spend/logs/v2?deployment_id=region-a` filter rows to one leg, including when
multiple deployments share the same provider model.
Schema version 16 also records nullable `requested_model` and `model_group`.
Both log endpoints accept `requested_model=requested-alias` or
`model_group=serving-alias`, with filtered totals on the v1 endpoint. These
identities remain separate from the provider-model field used to compute spend.
Schema version 17 adds nullable `session_id` and a session/timestamp index.
Native session headers take precedence over their legacy alias, with existing
Anthropic metadata fallback. Attribution is independent of routing strategy and
includes failure and partial-stream/realtime records.

Session identities are trimmed and bounded to 256 bytes on a UTF-8 boundary
before routing, logging or persistence. Oversized sessions use this bounded
prefix consistently; the writer also normalizes direct producer entries.
Both spend-log endpoints accept exact, case-sensitive `session_id` filters and
return totals over the full matching scope. Request-log reads expose the same
identity. Historical/older-writer rows remain null, and cache hits create no
additional spend rows. Structured logs retain correlation without prompts or
responses.

End-user/customer identities have a separate 256-byte limit after trimming.
Oversized request `user` values and new customer IDs return local 400s; they are
never truncated into another customer's policy or spend. Existing customer
records are not renamed. The spend writer omits invalid end-user attribution
from direct producer entries while preserving the event's other fields and
settled cost. Its indexed `model` label is bounded to 512 bytes; longer provider
or failure-model labels include a SHA-256 suffix so distinct values remain
distinct. This storage-only safeguard does not change routing or pricing; the
original `requested_model` remains available for correlation.

Inference-pipeline model ACL/customer denials, tag-budget lookup rejections,
admission-hook rejections and terminal upstream failures queue a zero-cost
`status: failure` row with request/model/key
identity and tags. The error body, prompt and response are not persisted; use
`request_id` to correlate structured logs. Partial streams retain their existing
cost and `partial` status. Shared terminal claims prevent duplicate rows across
request/stream context clones. API totals count these recorded requests, with
no monetary charge for denied/failed requests.
The integrated ACL boundaries include chat, embeddings, their translated API
bridges, rerank, hooked-native routes and provider-prefixed endpoints, including
nested batch model checks. Local token-counter estimates, model discovery,
unauthenticated middleware/extractor errors and explicit unmetered raw routes
are outside this spend-audit scope. Before deployment selection, provider
identity is unknown; selected deployments retain their native provider/model
identity even when configured without a provider prefix. Known local ACL and
customer rejections are labeled as pre-call checks in structured logs.
Realtime routing/concrete-selector/pricing rejections before upgrade are also
observed before any provider connection. Reported partial realtime cost is
preserved consistently with router settlement; missing usage is not converted
to invented tokens or cost. End-user IDs are trimmed before every attribution
path, and whitespace-only IDs are absent.

Native key/project creation and updates, bulk user-key team moves, project
reparenting and affected scope deletions share a database graph lock. Owned
projects require matching key teams; ownerless projects permit sharing. A
conflicting bulk move or reparent rejects the entire mutation. PostgreSQL uses
an exclusive transaction-scoped advisory lock with fresh `READ COMMITTED`
reads; SQLite reserves its writer with `BEGIN IMMEDIATE`, including independent
pools. Metadata edits load their current row inside the lock, and authentication
cache invalidation uses identities returned after commit. Low-level import and
fixture primitives retain their existing unchecked scope-link behavior.

`/key/info`, `/key/list`, and `/key/update` return typed `models` arrays (null
for unrestricted keys) and a `spend` total from retained, persisted logs. The
total can lag a response until the asynchronous writer inserts its record, and
retention can remove older history. It is not the resetting budget-window
counter. Invalid stored model JSON returns the same nonempty deny-all policy
used by authentication, so a damaged row can still be inspected and repaired
without breaking key listing or returning an error after an unrelated update.
`/global/spend` also returns the configured `max_budget`, or null when unset.
Team model add/delete refuses unreadable stored policies and the reserved
deny-all marker until an explicit `/team/update` replacement. Incremental
edits preserve existing literal patterns rather than re-normalizing them into
a different policy. Admin team/user forms keep corrupt stored policy read-back
as escaped deny-all JSON, preserving enforcement during unrelated edits.
Corrupt customer blocklists are interpreted as block-all, including for master
traffic and fallback groups; `/customer/update` can explicitly replace them.
Full key/team/user updates compare a supplied array with the decoded stored
literal policy before normalization. Identical arrays preserve the original
column, including whitespace-bearing patterns; intentionally changed
replacements retain the existing validation/normalization and explicit-clear
contracts.

#### Compatibility boundaries

The optional compatibility headers preserve the meanings listed above. Retry
counters cover the whole gateway request, including earlier failed groups;
some upstream routers report only the winning group's retries. Provider
error frames that fail before stream commitment become ordinary HTTP errors
after recovery is exhausted, preserving their status; errors after commitment
remain terminal SSE events. Request-content
4xx require an explicit failover policy, as described in the retry section.
Session affinity uses the documented stateless strategy, and Datadog service
callbacks are sampled dependency-health notifications rather than per-operation
latency/error telemetry. Vertex Gemini uses the OpenAI-compatible API surface;
native Gemini-specific safety/grounding/thinking fields need capability-specific
validation. The gateway's native request/cost/group contract is the stable
interface for clients.

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
built with `CREATE INDEX CONCURRENTLY` before publishing the current schema
and its read fallbacks. A dedicated migration session coordinates replicas;
public startup allows a bounded 30-minute preparation budget, including waiting
for another replica, while ordinary DDL lock waits remain short. Allow enough
startup/init-container grace for a large index build. Session locks/settings are
discarded with that connection on completion or cancellation. Wrong or invalid
same-named indexes and inconsistent event IDs produce a typed readiness error
without replacement or data deletion; an interrupted concurrent build can leave
an invalid index that must be inspected and recovered before retrying.

Schema version 15 adds nullable deployment attribution and updates the spend
read view. Existing rows and older writers that omit the column remain readable;
no historical deployment identity is guessed or backfilled.
Version 16 adds original-request and serving-group identities and refreshes the
read view again. Historical records keep null group fields. Upgrading from v15
is additive on SQLite and PostgreSQL, including mixed-version writers.

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

Redis-backed deployments already use an in-process Moka value-cache tier ahead
of Redis. This local tier is per replica; there is no shared memcached service.
Redis counters remain authoritative across replicas. The Redis driver shares a
managed multiplexed connection, with a separate pub/sub connection for
invalidation; `cache_kwargs.max_connections` does not configure a connection
pool. Configure Redis through `oxllm_settings.cache.redis_url` or `host`/`port`;
legacy `router_settings.redis_host`/`redis_port` are ignored.

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
