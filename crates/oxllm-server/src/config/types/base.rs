//! Top-level config containers: [`ProxyConfig`], [`OxllmSettings`],
//! [`GeneralSettings`], [`ConfigModel`], and the [`PassThroughEndpoint`]
//! declaration referenced by [`GeneralSettings`].

use oxllm_core::deployment::{ModelInfo, OxllmParams};
use serde::{Deserialize, Serialize};

use super::{alerting::*, cache::CacheConfig, guardrails::GuardrailsConfig, telemetry::*};

/// Deserialize `T` from `null` as `T::default()`. Some YAML emitters (and
/// hand-written configs that have a top-level key whose body is entirely
/// comments) produce an explicit `null` for a struct-valued field; without
/// this we'd reject it as `invalid type: null`.
fn deserialize_or_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    let opt: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match opt {
        None | Some(serde_json::Value::Null) => Ok(T::default()),
        Some(value) => T::deserialize(value).map_err(serde::de::Error::custom),
    }
}

/// Top-level proxy config, matching the LiteLLM config shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// The model list — one entry per upstream deployment.
    pub model_list: Vec<ConfigModel>,
    /// Global gateway settings. Uses `deserialize_or_default` so an explicit
    /// `null` body (e.g. when the block contains only comments) still parses
    /// (Bug #49). The legacy `litellm_settings` key is accepted as an alias
    /// for configs written before the rename.
    #[serde(
        default,
        alias = "litellm_settings",
        deserialize_with = "deserialize_or_default"
    )]
    pub oxllm_settings: OxllmSettings,
    /// General proxy settings (auth, db, etc.). Uses `deserialize_or_default`
    /// so an explicit `null` body still parses (Bug #49).
    #[serde(default, deserialize_with = "deserialize_or_default")]
    pub general_settings: GeneralSettings,
    /// Router settings (model-group alias, routing groups, etc.). Uses
    /// `deserialize_or_default` so an explicit `null` body (e.g. when the
    /// block contains only comments) still parses.
    #[serde(default, deserialize_with = "deserialize_or_default")]
    pub router_settings: super::router::RouterSettings,
    /// Unknown top-level keys captured for forward compatibility.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One entry in the `model_list`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigModel {
    /// The name clients send (e.g. `"gpt-4o"`).
    pub model_name: String,
    /// Provider config for this deployment.
    #[serde(alias = "litellm_params")]
    pub oxllm_params: OxllmParams,
    /// Optional metadata.
    #[serde(default)]
    pub model_info: Option<ModelInfo>,
    /// Unknown keys captured for forward compatibility.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl From<ConfigModel> for oxllm_core::Deployment {
    fn from(cm: ConfigModel) -> Self {
        if warns_wildcard_with_fully_qualified_model(&cm.model_name, &cm.oxllm_params.model) {
            tracing::warn!(
                event = "config",
                model_name = %cm.model_name,
                model = %cm.oxllm_params.model,
                "wildcard deployment configured with a fully-qualified model without `*`; \
                 it will match only exact requests for that provider/model, not every model \
                 of its provider"
            );
        }
        let mut oxllm_params = cm.oxllm_params;
        // Older proxy YAML puts `rpm` / `tpm` on the model_list row, not
        // inside the params block. Promote them when the params left them
        // unset so deployment limits actually apply.
        promote_row_limit(&mut oxllm_params.rpm, cm.extra.get("rpm"));
        promote_row_limit(&mut oxllm_params.tpm, cm.extra.get("tpm"));
        Self {
            model_name: cm.model_name,
            oxllm_params,
            model_info: cm.model_info,
        }
    }
}

fn promote_row_limit(slot: &mut Option<u32>, raw: Option<&serde_json::Value>) {
    if slot.is_some() {
        return;
    }
    let Some(v) = raw else {
        return;
    };
    *slot = v
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .or_else(|| v.as_i64().and_then(|n| u32::try_from(n).ok()));
}

/// Predicate for the [`From<ConfigModel> for Deployment`] wildcard footgun
/// warning. Returns `true` when a wildcard `model_name` (`"*"`) is paired
/// with a **fully-qualified** provider/model string that contains no `*`
/// anywhere (e.g. `"openai/gpt-4o"`) — such a deployment matches only exact
/// requests for that one provider/model rather than expanding across every
/// model of its provider. A `*` anywhere in the model string (e.g.
/// `"openai/*"` or `"*/gpt-4o"`, a provider wildcard) means the deployment
/// is genuinely wildcard and must NOT trip the warning.
fn warns_wildcard_with_fully_qualified_model(model_name: &str, model: &str) -> bool {
    model_name == "*" && model.contains('/') && !model.contains('*')
}

/// The `oxllm_settings` block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OxllmSettings {
    /// Strip unsupported params instead of erroring.
    #[serde(default)]
    pub drop_params: Option<bool>,
    /// Routing knobs (strategy, retries, cooldown, fallback chains, retry
    /// policy). Flattened so the YAML keys stay at the top level of
    /// `oxllm_settings`; `router_settings` values take precedence when a
    /// knob is set in both places (see [`super::router::RoutingKnobs::merge`]).
    #[serde(flatten)]
    pub routing: super::router::RoutingKnobs,
    /// Global default request timeout (seconds).
    #[serde(default)]
    pub request_timeout: Option<u64>,
    /// Global default per-key budget (USD).
    #[serde(default)]
    pub max_budget: Option<f64>,
    /// How long a budget window lasts (e.g. `"30d"`, `"1mo"`, `"1h"`).
    #[serde(default)]
    pub budget_duration: Option<String>,
    /// Global default per-key requests-per-minute limit.
    #[serde(default)]
    pub rpm: Option<u32>,
    /// Global default per-key tokens-per-minute limit.
    #[serde(default)]
    pub tpm: Option<u32>,
    /// TTL in seconds for response cache entries.
    #[serde(default)]
    pub cache_ttl_secs: Option<u64>,
    /// Cache configuration (type, host, port).
    #[serde(default)]
    pub cache: Option<CacheConfig>,
    /// Quality scores per deployment model (used by `quality_router`).
    /// Keyed by `oxllm_params.model` (e.g. `"openai/gpt-4o"`).
    #[serde(default)]
    pub quality_scores: Option<std::collections::HashMap<String, f64>>,
    /// Global default minimum estimated prompt-prefix tokens before the
    /// prompt-caching deployment check engages. Per-deployment
    /// `oxllm_params.prompt_caching_min_tokens` overrides; defaults to
    /// 1024 when neither is set.
    #[serde(default)]
    pub prompt_caching_min_tokens: Option<u32>,
    /// Master switch for the prompt-caching deployment check. The check is
    /// opt-in: when unset or `false`, no pin lookup or write happens and
    /// routing is unaffected.
    #[serde(default)]
    pub prompt_caching_enabled: Option<bool>,
    /// Per-model global spend caps (USD). Keys are model names (e.g.
    /// `"gpt-4o"`); values are the maximum total spend on that model across
    /// ALL keys/teams/users within the configured `budget_duration` window.
    /// Enforced via the `spend:model:{model}` counter.
    #[serde(default)]
    pub model_max_budget: Option<std::collections::HashMap<String, f64>>,
    /// List of success-callback slugs to invoke after a successful upstream
    /// response. Currently only `"generic"` is supported.
    #[serde(default)]
    pub success_callback: Option<Vec<String>>,
    /// List of failure-callback slugs to invoke after a failed upstream
    /// response (LiteLLM `failure_callback`). A slug listed here fires the
    /// corresponding hook's `post_failure` in addition to (or instead of)
    /// `post_success` when it is also in `success_callback`.
    #[serde(default)]
    pub failure_callback: Option<Vec<String>>,
    /// Dual-fire callback slugs (LiteLLM `callbacks`). A slug listed here
    /// fires the corresponding hook on BOTH success and failure — equivalent
    /// to adding the slug to both `success_callback` and
    /// `failure_callback`.
    #[serde(default)]
    pub callbacks: Option<Vec<String>>,
    /// Per-slug configuration for `success_callback`. Keys are slugs (e.g.
    /// `"generic"`); values are arbitrary YAML objects consumed by the
    /// corresponding hook.
    #[serde(default)]
    pub callback_settings: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub langfuse_config: Option<LangfuseConfig>,
    #[serde(default)]
    pub langsmith_config: Option<LangsmithConfig>,
    #[serde(default)]
    pub datadog_config: Option<DatadogConfig>,
    #[serde(default)]
    pub slack_alerting_config: Option<SlackAlertConfig>,
    #[serde(default)]
    pub helicone_config: Option<HeliconeConfig>,
    #[serde(default)]
    pub mlflow_config: Option<MlflowConfig>,
    #[serde(default)]
    pub pagerduty_config: Option<PagerDutyConfig>,
    /// Weights & Biases spend logging. Wired into the hook chain when
    /// present; logs model/tokens/cost per successful request.
    #[serde(default)]
    pub wandb_config: Option<WandbConfig>,
    /// S3 log storage. Writes one JSON line per successful request to the
    /// configured bucket as `{prefix}/{date}/{request_id}.json`.
    #[serde(default)]
    pub s3_log_config: Option<S3LogConfig>,
    /// Email (SMTP) budget-threshold alerting. The email analogue of
    /// `slack_alerting_config`.
    #[serde(default)]
    pub email_alert_config: Option<EmailAlertConfig>,
    #[serde(default)]
    pub alerting: Option<Vec<String>>,
    #[serde(default)]
    pub alerting_config: Option<AlertingConfig>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// The `general_settings` block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneralSettings {
    /// Master API key (bypasses all checks).
    #[serde(default)]
    pub master_key: Option<String>,
    /// Optional database URL.
    #[serde(default)]
    pub database_url: Option<String>,
    /// JWT signing secret (enables JWT auth when set).
    #[serde(default)]
    pub jwt_secret: Option<String>,
    /// JWT algorithm (default `HS256`).
    #[serde(default)]
    pub jwt_algorithm: Option<String>,
    /// Optional JWT audience (`aud`) claim to validate.
    #[serde(default)]
    pub jwt_audience: Option<String>,
    /// Optional JWT issuer (`iss`) claim to validate. When set, tokens whose
    /// `iss` claim does not match are rejected.
    #[serde(default)]
    pub jwt_issuer: Option<String>,
    /// Maximum request body size in megabytes (default 10).
    #[serde(default)]
    pub max_request_body_size_mb: Option<u64>,
    /// Log output format: `"json"` for structured JSON logs, `"text"` (or
    /// unset) for the default human-readable format.
    #[serde(default)]
    pub log_format: Option<String>,
    /// OpenTelemetry OTLP endpoint (e.g. `http://localhost:4317`). When set,
    /// distributed tracing is enabled (requires the `otel` feature).
    #[serde(default)]
    pub otel_endpoint: Option<String>,
    /// OpenTelemetry service name (default `oxllm`).
    #[serde(default)]
    pub otel_service_name: Option<String>,
    /// Allow deployment `api_base` URLs that resolve to private / loopback /
    /// link-local IP addresses. Defaults to `false`. Set to `true` for local
    /// development (e.g. pointing at a local Ollama instance).
    #[serde(default)]
    pub allow_private_api_base: Option<bool>,
    /// Allowed CORS origins for the management API
    /// (`/key/*`, `/team/*`, `/user/*`, `/spend/*`, `/model/*`). When `None`
    /// or empty, CORS is disabled entirely (no `Access-Control-*` headers).
    /// Otherwise only the listed origins are allowed. Defaults to `None` in
    /// production builds; tests/dev typically set
    /// `["http://localhost:3000"]`.
    #[serde(default)]
    pub management_cors_origins: Option<Vec<String>>,
    /// Allowed CORS origins for the LLM proxy routes (`/v1/*`,
    /// `/openai/v1/*`, metrics). When `None` or empty, no CORS layer is
    /// applied to the LLM routes. Otherwise only the listed origins are
    /// allowed and the layer permits `GET`, `POST`, and `OPTIONS` with
    /// `Authorization`, `Content-Type`, and `X-CSRF-Token` headers.
    #[serde(default)]
    pub proxy_cors_origins: Option<Vec<String>>,
    /// CIDR ranges of trusted reverse proxies (e.g. `["10.0.0.0/8",
    /// "192.168.0.0/16"]`). When set, the `X-Forwarded-For` / `X-Real-IP`
    /// headers are honored only when the request's peer (socket) address
    /// falls inside one of these ranges. When unset (default), forwarded
    /// headers are never trusted and the peer IP is used directly — this
    /// prevents spoofing of the per-IP auth-failure rate limiter.
    #[serde(default)]
    pub trusted_proxy_cidrs: Option<Vec<String>>,
    /// CIDR allowlist for client IPs (e.g. `["10.0.0.0/8", "192.168.1.0/24"]`).
    /// When set, requests whose resolved client IP is not inside any of the
    /// listed ranges are rejected with 403. When unset (default), all client
    /// IPs are allowed. An explicitly empty list (`allowed_ips: []`) denies
    /// ALL requests (fail-closed, matching LiteLLM's `_check_valid_ip`) —
    /// remove the setting to allow everyone. The client IP is resolved via
    /// the same `trusted_proxy_cidrs` logic used by the auth brake.
    #[serde(default)]
    pub allowed_ips: Option<Vec<String>>,
    /// Declarative pass-through routes. Each entry registers a catch-all
    /// route at `path` that forwards the request (with `headers` applied) to
    /// `target`.
    #[serde(default)]
    pub pass_through_endpoints: Option<Vec<PassThroughEndpoint>>,
    /// Graceful shutdown deadline in seconds. Defaults to 30. Active requests
    /// that have not completed by the deadline are force-closed. This is
    /// intentionally shorter than the streaming request timeout so that pod
    /// termination is not delayed by long-lived streams.
    #[serde(default)]
    pub shutdown_timeout_secs: Option<u64>,
    /// Pre-call content guardrails (PII masking, prompt-injection detection).
    /// When unset, no guardrails run.
    #[serde(default)]
    pub guardrails: Option<GuardrailsConfig>,
    /// Spend-log retention window in days. When `Some(n)`, a background task
    /// prunes `spend_logs` rows older than `n` days once per day. When
    /// `None` (default), no pruning is performed and the table grows
    /// without bound — appropriate for short-lived dev/test setups but not
    /// for long-running production deployments. The default recommended
    /// value is 90 days.
    #[serde(default)]
    pub spend_log_retention_days: Option<u64>,
    /// Enable JWT replay protection via `jti` (JWT ID) claim tracking.
    ///
    /// When `true` **and** a distributed cache (Redis) is configured, each
    /// validated JWT's `jti` is recorded in the cache with a TTL equal to the
    /// token's remaining lifetime. A token presenting an already-seen `jti`
    /// is rejected as a replay. This makes JWT validation stateful when
    /// enabled — every replica must share the same Redis instance.
    ///
    /// When `false` (default) or when no Redis cache is configured, JWT
    /// validation remains purely stateless (current behavior).
    #[serde(default)]
    pub jwt_replay_protection: Option<bool>,
    /// Maximum number of idle connections kept alive per host in the shared
    /// upstream HTTP client pool. Defaults to `20` when unset.
    #[serde(default)]
    pub http_pool_max_idle_per_host: Option<usize>,
    /// How long an idle connection stays in the pool before being closed
    /// (seconds). Defaults to `90` when unset.
    #[serde(default)]
    pub http_pool_idle_timeout_secs: Option<u64>,
    /// TCP connect timeout for the upstream HTTP client (seconds). When unset
    /// (default) no connect timeout is applied.
    #[serde(default)]
    pub http_connect_timeout_secs: Option<u64>,
    /// When `true`, the opt-in value-pattern redaction pass
    /// ([`redact_secret_values_in_place`]) is applied to every telemetry /
    /// exporter payload in addition to the unconditional key-name denylist.
    /// This scans string **values** for well-known secret shapes
    /// (`sk-...`, `AKIA...`, `Bearer ...`, `ghp_...`, `xoxb-...`) and masks
    /// matching substrings with `"[redacted]"`. Defaults to `false` so
    /// prompt content that legitimately discusses these patterns is
    /// preserved unless the operator opts in.
    ///
    /// [`redact_secret_values_in_place`]: crate::hooks::redact::redact_secret_values_in_place
    #[serde(default)]
    pub redact_secret_values: Option<bool>,
    /// Unknown keys captured for forward compatibility.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One declarative pass-through route. Requests whose path begins with
/// `path` are forwarded to `target` with `headers` merged on top of the
/// inbound headers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PassThroughEndpoint {
    /// Path prefix to match (e.g. `/v1/custom-endpoint`).
    pub path: String,
    /// Fully-qualified upstream target URL (e.g.
    /// `https://custom-api.example.com`).
    pub target: String,
    /// Extra headers to attach to the forwarded request (e.g. credentials).
    #[serde(default)]
    pub headers: Option<std::collections::HashMap<String, String>>,
}

#[cfg(test)]
mod tests {
    use super::warns_wildcard_with_fully_qualified_model as warn_pred;

    #[test]
    fn warn_fires_for_wildcard_name_with_fully_qualified_model() {
        // The original footgun: `model_name: "*"` paired with a specific
        // provider/model — the wildcard never expands.
        assert!(warn_pred("*", "openai/gpt-4o"));
        assert!(warn_pred("*", "anthropic/claude-3-opus"));
    }

    #[test]
    fn warn_does_not_fire_for_provider_wildcard_model() {
        // `*/gpt-4o` is a valid provider-wildcard deployment that matches
        // gpt-4o across all providers. Previously this tripped the warning
        // because the check only inspected the post-`/` model part.
        assert!(
            !warn_pred("*", "*/gpt-4o"),
            "provider wildcard must not trip the fully-qualified-model warning"
        );
    }

    #[test]
    fn warn_does_not_fire_for_full_provider_wildcard() {
        // `openai/*` is a genuine wildcard across all of openai's models.
        assert!(!warn_pred("*", "openai/*"));
        assert!(!warn_pred("*", "*/*"));
    }

    #[test]
    fn warn_does_not_fire_for_non_wildcard_model_name() {
        // Only `model_name == "*"` is the footgun; a specific name paired
        // with a fully-qualified model is the normal case.
        assert!(!warn_pred("gpt-4o", "openai/gpt-4o"));
    }

    #[test]
    fn warn_does_not_fire_for_bare_model_without_slash() {
        // The warning is scoped to provider-prefixed models; a bare model
        // string (no `/`) is not the fully-qualified case being flagged.
        assert!(!warn_pred("*", "gpt-4o"));
    }
}
