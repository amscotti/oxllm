//! Config loader — reads YAML and resolves `os.environ/` references.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::config::types::ProxyConfig;
use crate::security::{
    CLOUD_METADATA_EXCEPTIONS, PRIVATE_CIDRS, normalize_ip_for_ssrf, sanitize_url,
};

/// The path the proxy most recently loaded its config from (set by
/// [`load_config`]). The `/config/update` handler reads this to know where to
/// write back live mutations so the hot-reload watcher picks them up.
///
/// Process-global on purpose: the config is loaded exactly once at startup
/// (and on each hot-reload, which re-sets the same path). A `Mutex` rather
/// than a `OnceLock` lets tests in a single binary point at distinct temp
/// files without reusing a stale path.
static CONFIG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Every file that contributed to the most recent successful load, starting
/// with the write-back path. The hot-reload watcher observes all of them.
/// `current_config_path` stays the file `/config/update` writes.
static CONFIG_CONTRIBUTORS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Serialize any test that reads or writes [`CONFIG_PATH`] so the
/// `/config/update` handler test and the loader's config-path tests do not
/// race each other when `cargo test` runs them in parallel. Held across the
/// handler invocation in the management-config test so `current_config_path`
/// stays stable for the duration of the write.
#[cfg(test)]
pub(crate) static CONFIG_PATH_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Load and validate a proxy config from a YAML file.
///
/// This reads the file, parses it as JSON (via YAML), resolves all
/// `os.environ/` references, applies any `environment_variables` block, and
/// deserializes into [`ProxyConfig`].
///
/// `os.environ/` is resolved **once** at load time; downstream code never
/// reads environment variables.
pub fn load_config(path: &str) -> Result<ProxyConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {path}"))?;
    finish_load(&content, path)
}

/// Async variant of [`load_config`] backed by `tokio::fs`, with the parse +
/// validation (which performs blocking DNS resolution for the SSRF checks)
/// offloaded to the blocking pool. Use this on the hot runtime (e.g. the
/// hot-reload `Rebuild` closure) so neither a config read on a slow NFS mount
/// nor `validate_host_port`'s `to_socket_addrs()` blocks a tokio worker
/// thread. The synchronous
/// [`load_config`] remains for CLI entrypoints (`validate`, `keygen`) that run
/// before the runtime is shared with request handling.
pub async fn load_config_async(path: &str) -> Result<ProxyConfig> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read config file: {path}"))?;
    // Parsing + validation run on the blocking pool: `validate_proxy_config`
    // performs `to_socket_addrs()` DNS resolution (fail-closed SSRF checks),
    // which would otherwise block the tokio worker this future runs on —
    // defeating the point of the async variant during hot-reload.
    let path_owned = path.to_string();
    tokio::task::spawn_blocking(move || finish_load(&content, &path_owned))
        .await
        .context("config parse/validation task panicked")?
}

fn finish_load(content: &str, path: &str) -> Result<ProxyConfig> {
    let mut value: Value = serde_yml::from_str(content)
        .with_context(|| format!("failed to parse YAML from: {path}"))?;
    let mut contributors = Vec::new();
    expand_includes(&mut value, std::path::Path::new(path), &mut contributors)
        .with_context(|| format!("failed to expand includes from: {path}"))?;

    let config =
        value_to_config(value).with_context(|| format!("failed to parse config from: {path}"))?;

    // Record where we loaded from so `/config/update` can write back here.
    // Updated on every load (initial + hot-reload); the path is stable across
    // reloads in production but a `Mutex` (not `OnceLock`) keeps tests honest
    // when several temp-file configs are loaded in one binary.
    // Contributors include the root file plus every nested include. The
    // write-back path stays the root.
    if let Ok(mut guard) = CONFIG_PATH.lock() {
        *guard = Some(PathBuf::from(path));
    }
    if let Ok(mut guard) = CONFIG_CONTRIBUTORS.lock() {
        let mut recorded = vec![PathBuf::from(path)];
        recorded.extend(contributors);
        *guard = recorded;
    }

    Ok(config)
}

/// Files watched for hot reload after the latest successful load.
pub fn config_contributor_paths() -> Vec<PathBuf> {
    CONFIG_CONTRIBUTORS
        .lock()
        .ok()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

/// Returns the path the proxy most recently loaded its config from, if any.
/// Used by the `/config/update` handler to write live mutations to the same
/// file the hot-reload watcher observes.
pub fn current_config_path() -> Option<PathBuf> {
    CONFIG_PATH.lock().ok().and_then(|g| g.clone())
}

/// Parse a config from a YAML string (useful for tests).
pub fn parse_config_str(yaml: &str) -> Result<ProxyConfig> {
    let value: Value = serde_yml::from_str(yaml).context("failed to parse YAML")?;
    if value.get("include").is_some() {
        bail!("include requires a config file path; use load_config");
    }
    value_to_config(value)
}

fn value_to_config(mut value: Value) -> Result<ProxyConfig> {
    filter_raw_supported_environments(&mut value)?;
    let substitutions = resolve_env_vars(&mut value)?;

    // Apply the legacy `environment_variables` block (a top-level mapping of
    // NAME: value) BEFORE any other initialization: hooks, the cache, and DB
    // setup all read env vars during state construction, so they must be in
    // the process environment first. Runs at config-load time only; hot-reload
    // re-applies current values but never removes ones deleted from the YAML.
    apply_environment_variables(&value)?;

    let mut config = deserialize_with_env_coercion(value, &substitutions)?;

    log_unknown_keys(&config);
    validate_proxy_config(&mut config)?;
    Ok(config)
}

/// Validate a proposed root config without publishing its path or changing
/// process variables. Includes resolve relative to the file being replaced.
#[cfg(any(feature = "db", test))]
pub(crate) fn validate_config_value(
    mut value: Value,
    path: &std::path::Path,
) -> Result<ProxyConfig> {
    expand_includes(&mut value, path, &mut Vec::new())?;
    filter_raw_supported_environments(&mut value)?;
    let mut config: ProxyConfig = deserialize_env_value(value)?;
    log_unknown_keys(&config);
    validate_proxy_config(&mut config)?;
    Ok(config)
}

/// Resolve and deserialize env references using the target field's scalar type.
#[cfg(any(feature = "db", test))]
pub(crate) fn deserialize_env_value<T: serde::de::DeserializeOwned>(mut value: Value) -> Result<T> {
    let substitutions = resolve_env_vars(&mut value)?;
    deserialize_with_env_coercion(value, &substitutions)
}

/// Post-deserialize checks shared by file load and `POST /config/update`.
/// Does not resolve env vars or mutate process environment.
pub fn validate_proxy_config(config: &mut ProxyConfig) -> Result<()> {
    normalize_retry_aliases(config)?;
    crate::auth::validate_auth_config(&config.general_settings)?;
    filter_supported_environments(config);
    crate::pre_call_checks::warn_configured_checks(config);
    crate::hooks::datadog::warn_service_callback_config(config);
    for model in &config.model_list {
        model
            .oxllm_params
            .validate_auth_overrides()
            .with_context(|| {
                format!(
                    "invalid authentication override for model '{}'",
                    model.model_name
                )
            })?;
        if let Some(points) = &model.oxllm_params.cache_control_injection_points {
            for point in points {
                point.validate().with_context(|| {
                    format!(
                        "invalid cache_control_injection_points for model '{}'",
                        model.model_name
                    )
                })?;
            }
        }
    }
    if config.oxllm_settings.prompt_caching_enabled == Some(false) {
        for model in &config.model_list {
            if model
                .oxllm_params
                .cache_control_injection_points
                .as_ref()
                .is_some_and(|points| !points.is_empty())
            {
                bail!(
                    "model '{}' sets cache_control_injection_points while oxllm_settings.prompt_caching_enabled is false",
                    model.model_name
                );
            }
        }
    }
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    for model in &config.model_list {
        if let Some(api_base) = model.oxllm_params.api_base.as_deref() {
            validate_api_base(api_base, allow_private).with_context(|| {
                format!(
                    "invalid api_base for model '{}': {}",
                    model.model_name,
                    sanitize_url(api_base)
                )
            })?;
        }
    }
    if let Some(endpoints) = config.general_settings.pass_through_endpoints.as_ref() {
        for ep in endpoints {
            validate_api_base(&ep.target, allow_private).with_context(|| {
                format!(
                    "invalid pass_through_endpoints target for path '{}': {}",
                    ep.path,
                    sanitize_url(&ep.target)
                )
            })?;
        }
    }

    if let Some(callbacks) = config.oxllm_settings.success_callback.as_ref() {
        let has_generic = callbacks.iter().any(|s| s == "generic");
        if has_generic {
            if let Some(settings) = config.oxllm_settings.callback_settings.as_ref() {
                if let Some(block) = settings.get("generic") {
                    if let Some(url) = block.get("url").and_then(|v| v.as_str()) {
                        if !url.is_empty() {
                            validate_api_base(url, allow_private).with_context(|| {
                                format!(
                                    "invalid generic success_callback URL: {}",
                                    sanitize_url(url)
                                )
                            })?;
                        }
                    }
                }
            }
        }
    }

    // `failure_callback` and `callbacks` can also request the `generic`
    // callback. The same URL must pass SSRF validation regardless of which
    // list it was registered under.
    let generic_in_failure_or_callbacks = config
        .oxllm_settings
        .failure_callback
        .iter()
        .flatten()
        .chain(config.oxllm_settings.callbacks.iter().flatten())
        .any(|s| s == "generic");
    if generic_in_failure_or_callbacks {
        if let Some(url) = config
            .oxllm_settings
            .callback_settings
            .as_ref()
            .and_then(|s| s.get("generic"))
            .and_then(|b| b.get("url"))
            .and_then(|v| v.as_str())
        {
            if !url.is_empty() {
                validate_api_base(url, allow_private).with_context(|| {
                    format!("invalid generic callback URL: {}", sanitize_url(url))
                })?;
            }
        }
    }

    if let Some(alerting) = config.oxllm_settings.alerting.as_ref() {
        let has_webhook = alerting.iter().any(|s| s == "webhook");
        if has_webhook {
            if let Some(webhook) = config
                .oxllm_settings
                .alerting_config
                .as_ref()
                .and_then(|c| c.webhook.as_ref())
            {
                if !webhook.url.is_empty() {
                    validate_api_base(&webhook.url, allow_private).with_context(|| {
                        format!(
                            "invalid alerting webhook URL: {}",
                            sanitize_url(&webhook.url)
                        )
                    })?;
                }
                validate_thresholds("alerting_config.webhook.thresholds", &webhook.thresholds)?;
                validate_soft_budget_pct("alerting_config.webhook", webhook.soft_budget_pct)?;
            }
        }
    }

    if let Some(cfg) = config.oxllm_settings.slack_alerting_config.as_ref() {
        if !cfg.webhook_url.is_empty() {
            validate_api_base(&cfg.webhook_url, allow_private).with_context(|| {
                format!(
                    "invalid slack_alerting_config.webhook_url: {}",
                    sanitize_url(&cfg.webhook_url)
                )
            })?;
        }
        validate_thresholds("slack_alerting_config.thresholds", &cfg.thresholds)?;
        validate_soft_budget_pct("slack_alerting_config", cfg.soft_budget_pct)?;
    }
    if let Some(cfg) = config.oxllm_settings.pagerduty_config.as_ref() {
        validate_optional_url(
            "pagerduty_config.endpoint",
            cfg.endpoint.as_deref(),
            allow_private,
        )?;
        validate_thresholds("pagerduty_config.thresholds", &cfg.thresholds)?;
        validate_soft_budget_pct("pagerduty_config", cfg.soft_budget_pct)?;
    }
    if let Some(cfg) = config.oxllm_settings.email_alert_config.as_ref() {
        validate_host_port(&cfg.smtp_host, cfg.smtp_port, allow_private).with_context(|| {
            format!(
                "invalid email_alert_config.smtp_host/smtp_port: {}:{}",
                cfg.smtp_host, cfg.smtp_port
            )
        })?;
        validate_thresholds("email_alert_config.thresholds", &cfg.thresholds)?;
        validate_soft_budget_pct("email_alert_config", cfg.soft_budget_pct)?;
    }

    // OTLP trace-export endpoint — validated like every other outbound URL
    // so a private/loopback target requires `allow_private_api_base`.
    validate_optional_url(
        "general_settings.otel_endpoint",
        config.general_settings.otel_endpoint.as_deref(),
        allow_private,
    )?;

    // Telemetry outbound endpoints — validated at load for fail-fast parity
    // with `api_base` / pass-through / webhook URLs above. These are also
    // checked at hook-build time, but a build-time failure only warn-skips the
    // hook rather than failing startup; load-time validation surfaces
    // misconfiguration (e.g. a private outbound target) immediately.
    if let Some(cfg) = config.oxllm_settings.langfuse_config.as_ref() {
        if cfg.flush_interval_secs == Some(0) {
            bail!("langfuse_config.flush_interval_secs must be positive");
        }
        validate_optional_url("langfuse_config.host", Some(&cfg.host), allow_private)?;
    }
    validate_optional_url(
        "langsmith_config.endpoint",
        config
            .oxllm_settings
            .langsmith_config
            .as_ref()
            .and_then(|c| c.endpoint.as_deref()),
        allow_private,
    )?;
    validate_optional_url(
        "datadog_config.endpoint",
        config
            .oxllm_settings
            .datadog_config
            .as_ref()
            .and_then(|c| c.endpoint.as_deref()),
        allow_private,
    )?;
    validate_optional_url(
        "helicone_config.endpoint",
        config
            .oxllm_settings
            .helicone_config
            .as_ref()
            .and_then(|c| c.endpoint.as_deref()),
        allow_private,
    )?;
    if let Some(cfg) = config.oxllm_settings.mlflow_config.as_ref() {
        validate_optional_url(
            "mlflow_config.tracking_uri",
            Some(&cfg.tracking_uri),
            allow_private,
        )?;
    }
    validate_optional_url(
        "wandb_config.endpoint",
        config
            .oxllm_settings
            .wandb_config
            .as_ref()
            .and_then(|c| c.endpoint.as_deref()),
        allow_private,
    )?;
    // S3 log-storage endpoint: the hook build (`build_s3_log_hook_opt`) also
    // validates it, but a failure there only warn-skips the hook; validating
    // here fails fast at load and keeps `POST /config/update` honest.
    validate_optional_url(
        "s3_log_config.endpoint",
        config
            .oxllm_settings
            .s3_log_config
            .as_ref()
            .and_then(|c| c.endpoint.as_deref()),
        allow_private,
    )?;

    // Cache outbound endpoints — top-level Redis connection (rate-limit /
    // budget coordination + response cache) and semantic-cache backends.
    if let Some(cache) = config.oxllm_settings.cache.as_ref() {
        validate_redis_url_opt(cache.redis_url.as_deref(), allow_private)?;
        if let Some(host) = cache.host.as_deref() {
            let port = cache.port.unwrap_or(6379);
            validate_host_port(host, port, allow_private)
                .with_context(|| format!("invalid cache.host/cache.port: {host}:{port}"))?;
        }
        // S3-compatible cache endpoint (MinIO / R2 / …): `build_s3_cache`
        // hands this URL verbatim to the backend's plain HTTP client, so it
        // must pass the same SSRF rules as every other outbound URL at load
        // time (also covers remotely-supplied `POST /config/update` bodies).
        validate_optional_url(
            "cache.s3.endpoint",
            cache.s3.as_ref().and_then(|s| s.endpoint.as_deref()),
            allow_private,
        )?;
        if let Some(sc) = cache.semantic_cache.as_ref() {
            validate_optional_url(
                "semantic_cache.api_base",
                sc.api_base.as_deref(),
                allow_private,
            )?;
            validate_optional_url(
                "semantic_cache.qdrant_url",
                sc.qdrant_url.as_deref(),
                allow_private,
            )?;
            validate_redis_url_opt(sc.redis_url.as_deref(), allow_private)?;
            if let Some(t) = sc.similarity_threshold {
                if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
                    bail!(
                        "semantic_cache.similarity_threshold must be a finite \
                         cosine-similarity cutoff in [0.0, 1.0], got {t}"
                    );
                }
            }
        }
    }

    // Guardrail outbound endpoints — validated at load for fail-fast parity
    // with telemetry / alerting URLs. These are also checked at guardrail-
    // build time (where a failure warn-skips the guardrail), but load-time
    // validation surfaces a misconfigured or private target immediately.
    if let Some(gr) = config.general_settings.guardrails.as_ref() {
        if let Some(threshold) = gr.semantic_guard.as_ref().and_then(|guard| guard.threshold) {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                bail!("guardrails.semantic_guard.threshold must be finite and in [0, 1]");
            }
        }
        if let Some(presidio) = gr.presidio.as_ref() {
            validate_api_base(&presidio.url, allow_private).with_context(|| {
                format!(
                    "invalid guardrails.presidio.url: {}",
                    sanitize_url(&presidio.url)
                )
            })?;
        }
        if let Some(aim) = gr.aim.as_ref() {
            validate_api_base(&aim.url, allow_private).with_context(|| {
                format!("invalid guardrails.aim.url: {}", sanitize_url(&aim.url))
            })?;
        }
        if let Some(lakera) = gr.lakera.as_ref() {
            validate_optional_url(
                "guardrails.lakera.endpoint",
                lakera.endpoint.as_deref(),
                allow_private,
            )?;
        }
        if let Some(ma) = gr.model_armor.as_ref() {
            validate_api_base(&ma.endpoint, allow_private).with_context(|| {
                format!(
                    "invalid guardrails.model_armor.endpoint: {}",
                    sanitize_url(&ma.endpoint)
                )
            })?;
        }
        if let Some(customs) = gr.custom_guardrails.as_ref() {
            for cg in customs {
                if cg.guardrail_type == "http" {
                    if let Some(url) = cg
                        .config
                        .get("url")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        validate_api_base(url, allow_private).with_context(|| {
                            format!(
                                "invalid guardrails.custom_guardrails[\"{}\"].config.url: {}",
                                cg.name,
                                sanitize_url(url)
                            )
                        })?;
                    }
                }
            }
        }
        if let Some(mcps) = gr.mcp_guardrails.as_ref() {
            for mcp in mcps {
                validate_api_base(&mcp.server_url, allow_private).with_context(|| {
                    format!(
                        "invalid guardrails.mcp_guardrails[\"{}\"].server_url: {}",
                        mcp.name,
                        sanitize_url(&mcp.server_url)
                    )
                })?;
            }
        }
    }

    Ok(())
}

/// Validate an optional, possibly-empty outbound URL against the same SSRF
/// rules as [`validate_api_base`]. `None` and empty strings are no-ops.
fn validate_optional_url(label: &str, url: Option<&str>, allow_private: bool) -> Result<()> {
    if let Some(u) = url.filter(|s| !s.is_empty()) {
        validate_api_base(u, allow_private)
            .with_context(|| format!("invalid {label}: {}", sanitize_url(u)))?;
    }
    Ok(())
}

/// Validate an optional `redis://` / `rediss://` URL. Uses
/// [`validate_host_port`] since the `redis` scheme is rejected by
/// [`validate_api_base`]. `None` and empty strings are no-ops.
fn validate_redis_url_opt(url: Option<&str>, allow_private: bool) -> Result<()> {
    let Some(u) = url.filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let parsed =
        url::Url::parse(u).with_context(|| format!("invalid redis_url: {}", sanitize_url(u)))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("redis_url missing host: {}", sanitize_url(u)))?;
    let port = parsed.port().unwrap_or(6379);
    validate_host_port(host, port, allow_private)
        .with_context(|| format!("invalid redis_url: {}", sanitize_url(u)))?;
    Ok(())
}

/// Alert thresholds are spend fractions and must lie in `(0.0, 1.0]`. Values
/// outside that range are nonsensical (a negative or `>1` fraction never/always
/// fires) and, critically, a negative threshold collides with the soft-budget
/// sentinel key used by [`crate::hooks::alerting`]. Reject them at load so the
/// misconfiguration is visible rather than silently misbehaving.
fn validate_thresholds(field: &str, thresholds: &[f64]) -> Result<()> {
    for &t in thresholds {
        if !(t > 0.0 && t <= 1.0) {
            bail!(
                "{field} contains out-of-range threshold {t}; \
 each threshold must be a fraction in (0.0, 1.0]"
            );
        }
    }
    Ok(())
}

/// Validate an optional `soft_budget_pct`. Like [`validate_thresholds`] this is
/// a fraction of `max_budget`, so it must lie in `[0.0, 1.0]` and be finite. A
/// negative value makes every (always `>= 0`) request spend cross the threshold
/// and fires `"soft_budget_exceeded"` on every call — an alerting DoS; a `>1.0`
/// value places the soft threshold above the hard cap so the alert never fires;
/// `NaN` silently disables the alert (every `>=` comparison is false). `None`
/// (the default, which disables soft-budget alerts) is accepted.
fn validate_soft_budget_pct(field: &str, pct: Option<f64>) -> Result<()> {
    if let Some(p) = pct {
        if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
            bail!("{field}.soft_budget_pct must be a finite fraction in [0.0, 1.0], got {p}");
        }
    }
    Ok(())
}

/// Recursively walk the JSON tree and resolve `os.environ/FOO` strings.
///
/// If the referenced env var is unset, returns an error naming the config
/// **path** (not the raw env var name, to avoid env enumeration).
///
/// # Limitations
///
/// - **Coercion is limited to substituted values.** Substitution writes the
///   env value back as a string. If typed deserialization then fails, each
///   value rejected by a numeric/bool field is coerced and deserialization is
///   retried. String fields and ordinary YAML strings are not coerced. An
///   unparseable substituted value fails the load and names the field path
///   and the variable without disclosing its value.
/// - **Ordering.** This runs *before* `apply_environment_variables`, so
///   `os.environ/` references resolve against the pre-existing process
///   environment — not against values just declared in an
///   `environment_variables` block. (That block is a no-op in shipped builds
///   anyway; see its doc-comment.)
pub fn resolve_env_vars(value: &mut Value) -> Result<Vec<EnvSubstitution>> {
    let mut errors = Vec::new();
    let mut substitutions = Vec::new();
    resolve_inner(value, "", "", &mut errors, &mut substitutions);
    if errors.is_empty() {
        Ok(substitutions)
    } else {
        bail!(
            "config references unset environment variable at: {}",
            errors.join(", ")
        );
    }
}

/// One `os.environ/` substitution, remembered so a later typed-parse failure
/// can coerce the string or name the field and variable.
#[derive(Debug, Clone)]
pub struct EnvSubstitution {
    /// JSON pointer of the substituted node.
    pub pointer: String,
    /// Dotted config path used in error text.
    pub path: String,
    /// Environment variable name.
    pub var_name: String,
    /// Substituted value.
    pub value: String,
}

fn resolve_inner(
    value: &mut Value,
    path: &str,
    pointer: &str,
    errors: &mut Vec<String>,
    substitutions: &mut Vec<EnvSubstitution>,
) {
    match value {
        Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                let child_path = child_object_path(path, key);
                let child_pointer = format!("{pointer}/{}", pointer_escape(key));
                resolve_inner(val, &child_path, &child_pointer, errors, substitutions);
            }
        }
        Value::Array(arr) => {
            for (i, val) in arr.iter_mut().enumerate() {
                let child_path = format!("{path}[{i}]");
                let child_pointer = format!("{pointer}/{i}");
                resolve_inner(val, &child_path, &child_pointer, errors, substitutions);
            }
        }
        Value::String(s) => {
            if let Some(env_name) = s.strip_prefix("os.environ/") {
                match std::env::var(env_name) {
                    Ok(val) => {
                        substitutions.push(EnvSubstitution {
                            pointer: pointer.to_string(),
                            path: path.to_string(),
                            var_name: env_name.to_string(),
                            value: val.clone(),
                        });
                        *s = val;
                    }
                    Err(_) => {
                        errors.push(path.to_string());
                    }
                }
            }
        }
        _ => {}
    }
}

fn pointer_escape(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

fn child_object_path(parent: &str, key: &str) -> String {
    if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    }
}

/// Expand `include:` before env substitution.
///
/// Each entry is a path relative to the file that declares it. The declaring
/// file is the base. Each included file is merged on top: lists concatenate
/// base-then-included, and every other value, including `oxllm_settings`,
/// `general_settings`, and `router_settings`, is replaced as a whole. Nested
/// includes are queued after the siblings already waiting, so they overlay
/// those siblings. A file already merged through another path is skipped. A
/// cycle is an error that names the chain.
fn expand_includes(
    value: &mut Value,
    path: &std::path::Path,
    contributors: &mut Vec<PathBuf>,
) -> Result<()> {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::path::{Path, PathBuf};

    fn identity(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    let root_key = identity(path);
    canonicalize_settings_key(value)?;
    let mut loaded = HashSet::from([root_key.clone()]);
    let mut graph: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    let mut pending = VecDeque::new();
    for entry in take_include_list(value, path)? {
        graph.entry(root_key.clone()).or_default().push(identity(
            &path.parent().unwrap_or_else(|| Path::new(".")).join(&entry),
        ));
        pending.push_back((entry, path.to_path_buf(), vec![root_key.clone()]));
    }

    while let Some((entry, declared_in, chain)) = pending.pop_front() {
        let parent = declared_in.parent().unwrap_or_else(|| Path::new("."));
        let child_path = parent.join(&entry);
        let child_key = identity(&child_path);
        if chain.iter().any(|seen| seen == &child_key) {
            let mut names: Vec<String> = chain.iter().map(|p| p.display().to_string()).collect();
            names.push(child_key.display().to_string());
            bail!("include cycle: {}", names.join(" -> "));
        }
        if !loaded.insert(child_key.clone()) {
            continue;
        }
        let child_content = std::fs::read_to_string(&child_path).with_context(|| {
            format!(
                "failed to read include '{}' from {}",
                entry,
                declared_in.display()
            )
        })?;
        let mut child: Value = serde_yml::from_str(&child_content)
            .with_context(|| format!("failed to parse include {}", child_path.display()))?;
        canonicalize_settings_key(&mut child)?;
        let nested = take_include_list(&mut child, &child_path)?;
        graph.insert(
            child_key.clone(),
            nested
                .iter()
                .map(|entry| {
                    identity(
                        &child_path
                            .parent()
                            .unwrap_or_else(|| Path::new("."))
                            .join(entry),
                    )
                })
                .collect(),
        );
        let current = std::mem::take(value);
        *value = merge_included_config(current, child);
        contributors.push(child_path.clone());
        let mut child_chain = chain;
        child_chain.push(child_key);
        for nested_entry in nested {
            pending.push_back((nested_entry, child_path.clone(), child_chain.clone()));
        }
    }
    // A shared include may have been merged before another branch reaches it.
    // Check all edges, including skipped duplicates, so those cycles still fail.
    let mut complete = HashSet::new();
    let mut chain = Vec::new();
    let mut stack = vec![(root_key, false)];
    while let Some((node, exiting)) = stack.pop() {
        if exiting {
            chain.pop();
            complete.insert(node);
            continue;
        }
        if let Some(start) = chain.iter().position(|seen| seen == &node) {
            let names: Vec<_> = chain[start..]
                .iter()
                .chain(std::iter::once(&node))
                .map(|path: &PathBuf| path.display().to_string())
                .collect();
            bail!("include cycle: {}", names.join(" -> "));
        }
        if complete.contains(&node) {
            continue;
        }
        chain.push(node.clone());
        stack.push((node.clone(), true));
        if let Some(children) = graph.get(&node) {
            stack.extend(children.iter().rev().cloned().map(|child| (child, false)));
        }
    }
    Ok(())
}

/// Normalize the legacy settings alias before whole-block include merging.
fn canonicalize_settings_key(value: &mut Value) -> Result<()> {
    let Some(object) = value.as_object_mut() else {
        return Ok(());
    };
    if let Some(settings) = object.remove("litellm_settings") {
        if object.contains_key("oxllm_settings") {
            bail!("configure only one of oxllm_settings and its legacy alias in each file");
        }
        object.insert("oxllm_settings".into(), settings);
    }
    Ok(())
}

fn take_include_list(value: &mut Value, path: &std::path::Path) -> Result<Vec<String>> {
    let Some(obj) = value.as_object_mut() else {
        bail!("config {} must be a mapping", path.display());
    };
    let Some(raw) = obj.remove("include") else {
        return Ok(Vec::new());
    };
    if raw.is_null() {
        return Ok(Vec::new());
    }
    let Some(items) = raw.as_array() else {
        bail!("include in {} must be a list of paths", path.display());
    };
    let mut paths = Vec::with_capacity(items.len());
    for item in items {
        let Some(path_str) = item.as_str() else {
            bail!("include entry in {} must be a string path", path.display());
        };
        paths.push(path_str.to_string());
    }
    Ok(paths)
}

/// Merge `overlay` onto `base`. Lists concatenate base then overlay. Every
/// other value is replaced by the overlay, so a settings object from an
/// included file replaces the declaring file's block entirely.
fn merge_included_config(base: Value, overlay: Value) -> Value {
    let (Some(base_obj), Some(overlay_obj)) = (base.as_object(), overlay.as_object()) else {
        return overlay;
    };
    let mut out = base_obj.clone();
    for (key, overlay_value) in overlay_obj {
        match (out.get(key).cloned(), overlay_value) {
            (Some(Value::Array(mut base_list)), Value::Array(overlay_list)) => {
                base_list.extend(overlay_list.iter().cloned());
                out.insert(key.clone(), Value::Array(base_list));
            }
            (_, overlay_value) => {
                out.insert(key.clone(), overlay_value.clone());
            }
        }
    }
    Value::Object(out)
}

fn deserialize_with_env_coercion<T: serde::de::DeserializeOwned>(
    mut value: Value,
    substitutions: &[EnvSubstitution],
) -> Result<T> {
    let mut coerced = Vec::new();
    loop {
        let err = match serde_json::from_value::<T>(value.clone()) {
            Ok(config) => return Ok(config),
            Err(err) => err,
        };
        let message = err.to_string();
        // Probe only a rejected string. The unique replacement identifies the
        // exact field even through serde flattening and repeated env values;
        // numeric-looking secrets in string fields keep their original type.
        let target = substitutions.iter().enumerate().find(|(index, sub)| {
            if coerced.contains(index)
                || value.pointer(&sub.pointer).and_then(Value::as_str) != Some(&sub.value)
            {
                return false;
            }
            let rejected = format!("invalid type: string {:?}, expected ", sub.value);
            if !message.starts_with(&rejected) {
                return false;
            }
            let marker = format!("__oxllm_env_scalar_probe_{index}__");
            let mut probe = value.clone();
            *probe
                .pointer_mut(&sub.pointer)
                .expect("substitution path exists") = Value::String(marker.clone());
            serde_json::from_value::<T>(probe).is_err_and(|probe_err| {
                probe_err
                    .to_string()
                    .starts_with(&format!("invalid type: string {marker:?}, expected "))
            })
        });
        let Some((index, sub)) = target else {
            let context = coerced
                .last()
                .map(|index| {
                    let sub = &substitutions[*index];
                    format!(
                        "failed to deserialize config; check {} from os.environ/{}",
                        sub.path, sub.var_name
                    )
                })
                .unwrap_or_else(|| "failed to deserialize config into typed struct".into());
            return Err(err).context(context);
        };
        let parsed = coerce_env_scalar(&sub.value).ok_or_else(|| {
            anyhow::anyhow!(
                "os.environ value could not be parsed as a number or bool: {} from os.environ/{}",
                sub.path,
                sub.var_name,
            )
        })?;
        *value
            .pointer_mut(&sub.pointer)
            .expect("substitution path exists") = parsed;
        coerced.push(index);
    }
}

fn coerce_env_scalar(raw: &str) -> Option<Value> {
    match raw.to_ascii_lowercase().as_str() {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        _ => {}
    }
    if is_strict_int(raw) {
        if let Ok(n) = raw.parse::<i64>() {
            return Some(Value::Number(n.into()));
        }
        if let Ok(n) = raw.parse::<u64>() {
            return Some(Value::Number(n.into()));
        }
    }
    if is_strict_float(raw) {
        let parsed = raw.parse::<f64>().ok()?;
        return serde_json::Number::from_f64(parsed).map(Value::Number);
    }
    None
}

fn is_strict_int(raw: &str) -> bool {
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    digits == "0" || !digits.starts_with('0')
}

fn is_strict_float(raw: &str) -> bool {
    let body = raw.strip_prefix('-').unwrap_or(raw);
    if !(body.contains('.') || body.contains('e') || body.contains('E')) {
        return false;
    }
    body.chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
        && body.parse::<f64>().is_ok()
}

/// Resolve the primary environment, then its legacy alias, then development.
pub fn resolved_environment() -> String {
    ["OXLLM_ENVIRONMENT", "LITELLM_ENVIRONMENT"]
        .into_iter()
        .filter_map(|name| std::env::var(name).ok())
        .map(|value| value.trim().to_string())
        .find(|value| !value.is_empty())
        .unwrap_or_else(|| "development".to_string())
}

pub fn environment_is_development() -> bool {
    resolved_environment().eq_ignore_ascii_case("development")
}

/// Resolve the scope before reading credentials for an inactive deployment.
pub(crate) fn raw_model_info_supports_environment(
    info: Option<&Value>,
    environment: &str,
) -> Result<bool> {
    let Some(allowed) = info
        .and_then(|info| info.get("supported_environments"))
        .filter(|value| !value.is_null())
    else {
        return Ok(true);
    };
    let mut allowed = allowed.clone();
    resolve_env_vars(&mut allowed)
        .context("invalid model_info.supported_environments env reference")?;
    let allowed: Vec<String> = serde_json::from_value(allowed)
        .context("model_info.supported_environments must be a list of strings")?;
    Ok(allowed.iter().any(|name| name == environment))
}

fn filter_raw_supported_environments(value: &mut Value) -> Result<()> {
    let Some(models) = value.get_mut("model_list").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    let environment = resolved_environment();
    let mut retained = Vec::with_capacity(models.len());
    for model in std::mem::take(models) {
        if raw_model_info_supports_environment(model.get("model_info"), &environment)? {
            retained.push(model);
        } else {
            tracing::info!(model_name = model.get("model_name").and_then(|value| value.as_str()).unwrap_or("unknown"),
                environment = %environment, "dropping inactive deployment before env resolution");
        }
    }
    *models = retained;
    Ok(())
}

pub(crate) fn filter_supported_environments(config: &mut ProxyConfig) {
    let environment = resolved_environment();
    config.model_list.retain(|model| {
        let Some(info) = model.model_info.as_ref() else {
            return true;
        };
        let Some(allowed) = info.supported_environments.as_ref() else {
            return true;
        };
        if allowed.iter().any(|name| name == &environment) {
            return true;
        }
        tracing::info!(
            model_name = %model.model_name,
            environment = %environment,
            "dropping deployment: supported_environments excludes the current environment"
        );
        false
    });
}

#[cfg(feature = "db")]
pub(crate) fn deployment_supports_environment(
    deployment: &oxllm_core::Deployment,
    environment: &str,
) -> bool {
    deployment
        .model_info
        .as_ref()
        .and_then(|info| info.supported_environments.as_ref())
        .is_none_or(|allowed| allowed.iter().any(|name| name == environment))
}

fn normalize_retry_aliases(config: &mut ProxyConfig) -> Result<()> {
    let mut notes = Vec::new();
    notes.extend(translate_retry_block(
        "router_settings",
        &mut config.router_settings.routing,
    )?);
    notes.extend(translate_retry_block(
        "oxllm_settings",
        &mut config.oxllm_settings.routing,
    )?);
    if let Some(groups) = config.router_settings.routing_groups.as_mut() {
        for (name, group) in groups {
            notes.extend(translate_retry_block(
                &format!("router_settings.routing_groups.{name}"),
                &mut group.routing,
            )?);
        }
    }
    if !notes.is_empty() {
        tracing::info!(
            mapping = %notes.join("; "),
            "resolved named retry_policy entries"
        );
    }
    Ok(())
}

fn translate_retry_block(
    label: &str,
    knobs: &mut crate::config::types::RoutingKnobs,
) -> Result<Vec<String>> {
    knobs
        .translate_named_retry_policy()
        .map_err(|err| anyhow::anyhow!("{label}: {err}"))
        .map(|notes| {
            notes
                .into_iter()
                .map(|note| format!("{label}: {note}"))
                .collect()
        })
}

/// Apply the legacy `environment_variables` block (a top-level YAML mapping
/// of `NAME: value`) to the process environment via `std::env::set_var`.
///
/// This runs at config-load time, before hooks/cache/DB initialization, so
/// features that read env vars during construction see the configured values.
/// Hot-reload re-applies current values on each reload but never *removes*
/// entries deleted from the YAML (there is no record of which vars the
/// process originally lacked).
///
/// # Limitation: no-op in shipped builds
///
/// `std::env::set_var` is `unsafe` under the Rust 2024 edition, and the crate
/// is `#![cfg_attr(not(test), forbid(unsafe_code))]`. Because `forbid` cannot
/// be relaxed by an inner `allow`, the actual write is delegated to a
/// `cfg`-gated helper ([`apply_env_entries`]).
///
/// **`cfg(test)` is only set for the crate currently being compiled** (see the
/// Rust Reference), so the writing arm runs *exclusively* inside this crate's
/// own `#[cfg(test)]` modules. It does **not** run in:
/// - the production `oxllm` server binary,
/// - integration tests (`tests/*.rs`, which compile this crate as a
///   dependency, so `cfg(test)` is false for it),
/// - or any consumer binary.
///
/// In all of those the block is still parsed and validated, but the writes are
/// skipped with a `warn!`. Operators must set such variables in the actual
/// process environment (e.g. the systemd unit, container env, or shell)
/// rather than relying on this block. This keeps the crate `unsafe`-free in
/// shipped builds (see the workspace `AGENTS`/`CONTRIBUTING` rules). A full
/// fix would require an in-process env overlay consulted by every
/// `std::env::var` call site, which is a cross-cutting change outside this
/// component.
/// Environment-variable names that the `environment_variables` block must
/// never override, mirroring the reference `_BLOCKED_ENV_KEYS`. These are
/// process-identity / runtime-discovery variables whose silent overwrite
/// would break the process (e.g. clobbering `PATH` or `HOME`), skew
/// operator observability (`RUST_LOG`), or — for the dynamic-linker and
/// TLS-trust entries (`LD_PRELOAD`, `DYLD_INSERT_LIBRARIES`, `SSL_CERT_*`)
/// — open a code-injection / trust-bypass vector. Python-specific entries
/// from the reference list (`PYTHONSTARTUP`, `PYTHONHOME`) are omitted since
/// oxllm is a Rust binary.
const BLOCKED_ENV_KEYS: &[&str] = &[
    "HOME",
    "USER",
    "SHELL",
    "LOGNAME",
    "PATH",
    "NO_PROXY",
    "no_proxy",
    "RUST_LOG",
    "RUST_BACKTRACE",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
];

fn apply_environment_variables(value: &Value) -> Result<()> {
    let Some(env_block) = value.get("environment_variables") else {
        return Ok(());
    };
    let map = env_block
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("environment_variables must be a mapping of NAME: value"))?;

    // Validate every value is a string up front so application is
    // all-or-nothing (no partial application on a malformed block).
    let mut entries: Vec<(String, String)> = Vec::with_capacity(map.len());
    for (name, val) in map {
        let s = val.as_str().ok_or_else(|| {
            anyhow::anyhow!("environment_variables.{name} must be a string value")
        })?;
        if BLOCKED_ENV_KEYS.contains(&name.as_str()) {
            tracing::warn!(
                key = %name,
                "environment_variables entry on the blocked-keys list is skipped \
                 to avoid clobbering process-identity / runtime variables"
            );
            continue;
        }
        entries.push((name.clone(), s.to_string()));
    }

    apply_env_entries(&entries);
    Ok(())
}

#[cfg(test)]
fn apply_env_entries(entries: &[(String, String)]) {
    // SAFETY: `std::env::set_var` is `unsafe` in edition 2024 because a
    // concurrent `std::env::var` read on another thread would be a data
    // race. This arm only compiles for this crate's own `#[cfg(test)]`
    // modules (see `apply_environment_variables` for why), and tests
    // serialize env access behind the module-level `ENV_LOCK`. No concurrent
    // reader runs here.
    #[allow(unsafe_code)]
    unsafe {
        for (name, val) in entries {
            std::env::set_var(name, val);
            tracing::debug!(var = %name, "applied environment_variables entry from config");
        }
    }
}

#[cfg(not(test))]
fn apply_env_entries(entries: &[(String, String)]) {
    if entries.is_empty() {
        return;
    }
    // See `apply_environment_variables`: this non-test arm is the ONLY arm
    // compiled into shipped builds (production binary, integration tests that
    // compile this crate as a dependency, and consumer binaries), because
    // `cfg(test)` is set solely for the crate currently being compiled. It
    // cannot call `std::env::set_var` (the crate is `forbid(unsafe_code)` and
    // `set_var` is `unsafe` in edition 2024), so the block has already been
    // parsed and validated but the writes are skipped. Surface the skip so
    // operators notice and set the variables in the real process environment.
    tracing::warn!(
        count = entries.len(),
        keys = ?entries.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        "environment_variables parsed but not applied: this is a \
         `forbid(unsafe_code)` build and `std::env::set_var` is `unsafe` in \
         edition 2024. The entries were validated but NOT written to the \
         process environment. Set them in the real environment (e.g. \
         container/shell env) for downstream consumers to observe them.",
    );
}

/// Log unknown keys at the top level (forward-compat warning, never a failure).
fn log_unknown_keys(config: &ProxyConfig) {
    for key in config.extra.keys() {
        tracing::warn!(key = %key, "unknown top-level config key, ignored");
    }
    for model in &config.model_list {
        for key in model.extra.keys() {
            tracing::warn!(
            model = %model.model_name,
            key = %key,
            "unknown config key in model entry, ignored"
            );
        }
    }
    for key in config.oxllm_settings.extra.keys() {
        tracing::warn!(key = %key, "unknown key in oxllm_settings, ignored");
    }
    for key in config.general_settings.extra.keys() {
        tracing::warn!(key = %key, "unknown key in general_settings, ignored");
    }
    for key in config.router_settings.extra.keys() {
        tracing::warn!(key = %key, "unknown key in router_settings, ignored");
    }
    for model in &config.model_list {
        for (key, value) in &model.oxllm_params.extra {
            if oxllm_core::deployment::is_known_oxllm_param_extra(key) {
                continue;
            }
            match oxllm_core::cost::classify_extra_tier_key(key) {
                oxllm_core::cost::ExtraTierKeyClass::Parsed => {
                    if oxllm_core::cost::extra_tier_rate_is_non_numeric(key, value) {
                        tracing::warn!(
                            model = %model.model_name,
                            key = %key,
                            "context-tier price key has a non-numeric rate and will not bill"
                        );
                    }
                }
                oxllm_core::cost::ExtraTierKeyClass::Malformed => {
                    tracing::warn!(
                        model = %model.model_name,
                        key = %key,
                        "malformed context-tier price key in oxllm_params, ignored"
                    );
                }
                oxllm_core::cost::ExtraTierKeyClass::NotTier => {
                    tracing::warn!(
                        model = %model.model_name,
                        key = %key,
                        "unknown key in oxllm_params, ignored"
                    );
                }
            }
        }
    }
}

/// Private / loopback / link-local CIDR ranges that a deployment `api_base`
/// must not resolve to unless `allow_private_api_base` is set. Re-exported
/// from [`crate::security::PRIVATE_CIDRS`] so the config-load-time and
/// runtime SSRF checks cannot drift apart.
///
/// Validate a deployment `api_base` URL to prevent SSRF.
///
/// - Rejects non-`http(s)`/`ws(s)` schemes.
/// - When `allow_private` is `false`, rejects URLs whose host resolves to a
///   private, loopback, or link-local IP. Hosts that fail to resolve are
///   **rejected** (fail-closed) to prevent DNS rebinding: a name that is
///   unresolvable at load time could later resolve to a private address,
///   bypassing the static private-IP check.
pub fn validate_api_base(api_base: &str, allow_private: bool) -> Result<()> {
    let parsed = url::Url::parse(api_base)
        .with_context(|| format!("invalid URL: {}", sanitize_url(api_base)))?;

    match parsed.scheme() {
        "http" | "https" | "ws" | "wss" => {}
        other => bail!("unsupported api_base scheme '{other}'; expected http(s)/ws(s)"),
    }

    if allow_private {
        return Ok(());
    }

    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("api_base is missing a host"))?;
    let port = parsed.port_or_known_default().unwrap_or(80);

    validate_host_port(host, port, allow_private)
}

/// Validate a bare `host:port` pair (e.g. an SMTP relay) against the same
/// private-CIDR / DNS-rebinding protections as [`validate_api_base`], without
/// requiring an HTTP URL. When `allow_private` is `false`, rejects hosts that
/// resolve to a private/loopback/link-local IP and hosts that fail to resolve
/// (fail-closed). This is the shared host-resolution core used by every
/// outbound transport — HTTP webhooks via [`validate_api_base`] and non-HTTP
/// transports (e.g. the SMTP email alert hook) directly.
pub fn validate_host_port(host: &str, port: u16, allow_private: bool) -> Result<()> {
    if allow_private {
        return Ok(());
    }

    let candidates: Vec<std::net::IpAddr> = if let Some(ip) = parse_ip(host) {
        vec![ip]
    } else {
        match (host, port).to_socket_addrs() {
            Ok(addrs) => {
                let collected: Vec<_> = addrs.map(|sa| sa.ip()).collect();
                if collected.is_empty() {
                    bail!(
                        "host '{host}' did not resolve to any address; \
 refusing to allow an unresolvable host (DNS-rebinding protection). \
 Set general_settings.allow_private_api_base: true to permit."
                    );
                }
                collected
            }
            Err(e) => {
                bail!(
                    "host '{host}' did not resolve ({e}); \
 refusing to allow an unresolvable host (DNS-rebinding protection). \
 Set general_settings.allow_private_api_base: true to permit."
                );
            }
        }
    };

    for ip in candidates {
        let check_ip = normalize_ip_for_ssrf(ip);
        for cidr in blocked_cidrs() {
            if cidr.contains(&check_ip) {
                bail!(
                    "host '{host}' resolves to private IP {check_ip} (in {cidr}); \
 set general_settings.allow_private_api_base: true to permit"
                );
            }
        }
    }

    Ok(())
}

/// Parsed private / cloud-metadata CIDRs used by [`validate_host_port`].
/// Parsing ~15 `IpNet`s is cheap but non-zero, and the result is fully
/// static, so it is computed once via [`std::sync::OnceLock`] instead of on
/// every call for every candidate IP (config load and hot-reload invoke this
/// per model/callback/guardrail).
static BLOCKED_CIDRS: std::sync::OnceLock<Vec<ipnet::IpNet>> = std::sync::OnceLock::new();

fn blocked_cidrs() -> &'static [ipnet::IpNet] {
    BLOCKED_CIDRS.get_or_init(|| {
        PRIVATE_CIDRS
            .iter()
            .chain(CLOUD_METADATA_EXCEPTIONS.iter())
            .filter_map(|c| c.parse().ok())
            .collect()
    })
}

/// Parse a host string as a literal IPv4/IPv6 address. Handles the bracketed
/// form returned by `url::Url::host_str()` for IPv6 literals (e.g. `"[::1]"`).
fn parse_ip(host: &str) -> Option<std::net::IpAddr> {
    let trimmed = host.trim_start_matches('[').trim_end_matches(']');
    trimmed.parse::<std::net::IpAddr>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[allow(unsafe_code)]
    fn set_env(name: &str, value: &str) {
        unsafe { std::env::set_var(name, value) };
    }

    #[allow(unsafe_code)]
    fn unset_env(name: &str) {
        unsafe { std::env::remove_var(name) };
    }

    #[test]
    fn resolve_plain_string() {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut val = Value::String("hello".to_string());
        resolve_env_vars(&mut val).expect("plain string should pass through");
        assert_eq!(val, Value::String("hello".to_string()));
    }

    #[test]
    fn resolve_env_var_in_nested_object() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_TEST_KEY", "sk-resolved-123");
        let mut val: Value = serde_json::json!({
        "model_list": [
        {
        "oxllm_params": {
        "api_key": "os.environ/OXLLM_TEST_KEY"
        }
        }
        ]
        });
        resolve_env_vars(&mut val).expect("env var should resolve");
        assert_eq!(
            val["model_list"][0]["oxllm_params"]["api_key"],
            Value::String("sk-resolved-123".to_string())
        );
        unset_env("OXLLM_TEST_KEY");
    }

    #[test]
    fn unset_env_var_reports_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_env("OXLLM_NONEXISTENT_VAR_XYZ");
        let mut val: Value = serde_json::json!({
        "model_list": [
        {
        "oxllm_params": {
        "api_key": "os.environ/OXLLM_NONEXISTENT_VAR_XYZ"
        }
        }
        ]
        });
        let err = resolve_env_vars(&mut val).expect_err("unset env var should error");
        let msg = err.to_string();
        assert!(
            msg.contains("model_list[0].oxllm_params.api_key"),
            "error should name the config path, got: {msg}"
        );
        assert!(
            !msg.contains("OXLLM_NONEXISTENT_VAR_XYZ"),
            "error must NOT expose the raw env var name, got: {msg}"
        );
    }

    #[test]
    fn load_config_with_env_resolution() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_TEST_API_KEY", "sk-from-env");
        let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
      api_key: os.environ/OXLLM_TEST_API_KEY
";
        let cfg = parse_config_str(yaml).expect("config should load");
        assert_eq!(
            cfg.model_list[0].oxllm_params.api_key.as_deref(),
            Some("sk-from-env")
        );
        unset_env("OXLLM_TEST_API_KEY");
    }

    #[test]
    fn load_config_missing_file() {
        let result = load_config("/nonexistent/path/to/config.yaml");
        assert!(result.is_err(), "missing file should error");
    }

    /// C60 MEDIUM: the hot-reload `Rebuild` closure must use the async variant
    /// so a config read on a slow NFS mount does not block a tokio worker
    /// thread. This pins that the async path reads and parses identically to
    /// the sync [`load_config`] used at startup. Path recording is shared via
    /// `finish_load` and is covered by `current_config_path_tracks_load_config`.
    #[tokio::test]
    async fn load_config_async_reads_and_parses() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxllm_test_async_load_{}.yaml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, "model_list: []\n").expect("write temp config");
        let path_str = path.to_string_lossy().to_string();

        let cfg = load_config_async(&path_str)
            .await
            .expect("async load should succeed");
        assert!(
            cfg.model_list.is_empty(),
            "async-loaded config should parse identically"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn load_config_async_missing_file_errors() {
        let result = load_config_async("/nonexistent/path/to/config.yaml").await;
        assert!(
            result.is_err(),
            "missing file should error on the async path"
        );
    }

    #[test]
    fn unknown_keys_captured_not_error() {
        let _guard = ENV_LOCK.lock().unwrap();
        let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
    future_field: value

unknown_top_level:
  whatever: true
";
        let cfg = parse_config_str(yaml).expect("unknown keys should not fail");
        assert!(cfg.extra.contains_key("unknown_top_level"));
        assert!(cfg.model_list[0].extra.contains_key("future_field"));
    }

    #[test]
    fn unknown_router_settings_keys_captured_for_warning() {
        // `log_unknown_keys` iterates `router_settings.extra`; a typo here
        // must land in `extra` so the forward-compat warning fires (matching
        // the behavior for top-level / model / gateway / general keys).
        let _guard = ENV_LOCK.lock().unwrap();
        let yaml = "\
model_list: []

router_settings:
  routing_strategyy: typoed-key
";
        let cfg = parse_config_str(yaml).expect("unknown router key should not fail");
        assert!(
            cfg.router_settings.extra.contains_key("routing_strategyy"),
            "unknown router_settings key must be captured in extra"
        );
    }

    #[test]
    fn resolve_non_string_values_ignored() {
        let _guard = ENV_LOCK.lock().unwrap();
        let mut val: Value = serde_json::json!({
        "port": 4000,
        "enabled": true,
        "models": null
        });
        resolve_env_vars(&mut val).expect("non-string values should pass through");
    }

    #[test]
    fn resolve_env_var_with_special_chars() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_SPECIAL", "key-with-dashes_and_underscores");
        let mut val: Value = serde_json::json!({
        "key": "os.environ/OXLLM_SPECIAL"
        });
        resolve_env_vars(&mut val).expect("should resolve");
        assert_eq!(val["key"], "key-with-dashes_and_underscores");
        unset_env("OXLLM_SPECIAL");
    }

    #[test]
    fn resolve_multiple_unset_vars_reports_all() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_env("OXLLM_UNSET_A");
        unset_env("OXLLM_UNSET_B");
        let mut val: Value = serde_json::json!({
        "a": "os.environ/OXLLM_UNSET_A",
        "b": "os.environ/OXLLM_UNSET_B"
        });
        let err = resolve_env_vars(&mut val).expect_err("should error");
        let msg = err.to_string();
        assert!(msg.contains("a"), "should report path 'a': {msg}");
        assert!(msg.contains("b"), "should report path 'b': {msg}");
    }

    #[test]
    fn validate_api_base_accepts_https_url() {
        validate_api_base("https://api.openai.com/v1", false)
            .expect("public HTTPS URL should pass");
    }

    #[test]
    fn validate_api_base_rejects_non_http_scheme() {
        let err = validate_api_base("file:///etc/passwd", false)
            .expect_err("file scheme should be rejected");
        let msg = err.to_string();
        assert!(msg.contains("scheme"), "expected scheme error, got: {msg}");
    }

    #[test]
    fn validate_api_base_rejects_zero_address() {
        let err = validate_api_base("http://0.0.0.0:8080/v1", false)
            .expect_err("0.0.0.0 should be blocked");
        assert!(
            err.to_string().contains("private"),
            "error should mention private IP, got: {err}"
        );
    }

    #[test]
    fn validate_api_base_rejects_ipv4_mapped_loopback() {
        let err = validate_api_base("http://[::ffff:127.0.0.1]/v1", false)
            .expect_err("IPv4-mapped loopback should be blocked");
        assert!(
            err.to_string().contains("private"),
            "error should mention private IP, got: {err}"
        );
    }

    #[test]
    fn validate_api_base_rejects_localhost() {
        let err = validate_api_base("http://127.0.0.1:11434/v1", false)
            .expect_err("loopback IP should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("private"),
            "expected private-IP error, got: {msg}"
        );
    }

    #[test]
    fn validate_api_base_rejects_private_v4() {
        let err = validate_api_base("http://10.0.0.5/v1", false)
            .expect_err("private 10/8 should be rejected");
        assert!(err.to_string().contains("private"));
    }

    #[test]
    fn validate_api_base_rejects_link_local() {
        let err = validate_api_base("http://169.254.169.254/latest/meta-data", false)
            .expect_err("link-local metadata IP must be rejected");
        assert!(err.to_string().contains("private"));
    }

    #[test]
    fn validate_api_base_rejects_ipv6_loopback() {
        let err = validate_api_base("http://[::1]/v1", false)
            .expect_err("IPv6 loopback should be rejected");
        assert!(err.to_string().contains("private"));
    }

    #[test]
    fn validate_api_base_allows_localhost_when_opted_in() {
        validate_api_base("http://127.0.0.1:11434/v1", true)
            .expect("allow_private_api_base=true should permit loopback");
    }

    #[test]
    fn validate_api_base_rejects_malformed_url() {
        let err = validate_api_base("not a url", false).expect_err("malformed URL should fail");
        assert!(err.to_string().contains("invalid URL"));
    }

    #[test]
    fn validate_api_base_rejects_unresolvable_host_when_private_blocked() {
        let err = validate_api_base("https://nonexistent-host-xyz-123.invalid/v1", false)
            .expect_err("unresolvable host should be rejected (fail-closed)");
        let msg = err.to_string();
        assert!(
            msg.contains("did not resolve") || msg.contains("unresolvable"),
            "expected DNS-resolution error, got: {msg}"
        );
    }

    #[test]
    fn validate_api_base_allows_unresolvable_host_when_private_allowed() {
        validate_api_base("https://nonexistent-host-xyz-123.invalid/v1", true)
            .expect("allow_private_api_base=true should permit unresolvable hosts");
    }

    #[test]
    fn validate_host_port_rejects_loopback() {
        let err = validate_host_port("127.0.0.1", 25, false).expect_err("loopback rejected");
        assert!(err.to_string().contains("private"));
    }

    #[test]
    fn validate_host_port_rejects_link_local() {
        let err =
            validate_host_port("169.254.169.254", 80, false).expect_err("link-local rejected");
        assert!(err.to_string().contains("private"));
    }

    #[test]
    fn validate_host_port_rejects_private_v4() {
        let err = validate_host_port("10.0.0.5", 587, false).expect_err("10/8 rejected");
        assert!(err.to_string().contains("private"));
    }

    #[test]
    fn validate_host_port_allows_loopback_when_opted_in() {
        validate_host_port("127.0.0.1", 25, true).expect("allow_private permits loopback");
    }

    #[test]
    fn validate_host_port_rejects_ipv6_loopback() {
        let err = validate_host_port("::1", 25, false).expect_err("IPv6 loopback rejected");
        assert!(err.to_string().contains("private"));
    }

    #[test]
    fn parse_config_rejects_private_api_base_by_default() {
        let yaml = "\
model_list:
  - model_name: ollama
    oxllm_params:
      model: ollama/llama3
      api_key: sk-test
      api_base: http://127.0.0.1:11434
";
        let err = parse_config_str(yaml).expect_err("private api_base should fail by default");
        assert!(err.to_string().contains("api_base"));
    }

    #[test]
    fn parse_config_allows_private_api_base_when_opted_in() {
        let yaml = "\
model_list:
  - model_name: ollama
    oxllm_params:
      model: ollama/llama3
      api_key: sk-test
      api_base: http://127.0.0.1:11434

general_settings:
 allow_private_api_base: true
";
        parse_config_str(yaml).expect("allow_private_api_base should permit loopback");
    }

    #[test]
    fn parse_config_accepts_public_api_base() {
        let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
      api_key: sk-test
      api_base: https://api.openai.com/v1
";
        parse_config_str(yaml).expect("public api_base should pass");
    }

    #[test]
    fn parse_config_rejects_private_callback_url() {
        let yaml = "\
model_list: []

oxllm_settings:
  success_callback:
    - generic
  callback_settings:
    generic:
      url: http://127.0.0.1:9090/spend
";
        let err = parse_config_str(yaml)
            .expect_err("private callback URL should be rejected at config load");
        let full = format!("{err:#}");
        assert!(
            full.contains("callback") || full.contains("url"),
            "error should mention the callback URL, got: {full}"
        );
        assert!(
            full.contains("private"),
            "error should mention private IP, got: {full}"
        );
    }

    #[test]
    fn parse_config_allows_private_callback_url_when_opted_in() {
        let yaml = "\
model_list: []

oxllm_settings:
  success_callback:
    - generic
  callback_settings:
    generic:
      url: http://127.0.0.1:9090/spend

general_settings:
  allow_private_api_base: true
";
        parse_config_str(yaml).expect("allow_private_api_base should permit private callback URL");
    }

    #[test]
    fn parse_config_accepts_public_callback_url() {
        let yaml = "\
model_list: []

oxllm_settings:
  success_callback:
    - generic
  callback_settings:
    generic:
      url: https://api.openai.com/v1/spend
";
        parse_config_str(yaml).expect("public callback URL should pass");
    }

    #[test]
    fn parse_config_rejects_private_alerting_webhook_url() {
        let yaml = "\
model_list: []

oxllm_settings:
  alerting:
    - webhook
  alerting_config:
    webhook:
      url: http://127.0.0.1:9090/alert
";
        let err = parse_config_str(yaml).expect_err("private alerting webhook must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("alerting") || full.contains("webhook"),
            "error should mention the alerting webhook, got: {full}"
        );
        assert!(
            full.contains("private"),
            "error should mention private IP, got: {full}"
        );
    }

    #[test]
    fn parse_config_allows_private_alerting_webhook_when_opted_in() {
        let yaml = "\
model_list: []

oxllm_settings:
  alerting:
    - webhook
  alerting_config:
    webhook:
      url: http://127.0.0.1:9090/alert
      thresholds: [0.5, 1.0]

general_settings:
  allow_private_api_base: true
";
        let cfg =
            parse_config_str(yaml).expect("allow_private_api_base should permit private webhook");
        let webhook = cfg
            .oxllm_settings
            .alerting_config
            .as_ref()
            .unwrap()
            .webhook
            .as_ref()
            .unwrap();
        assert_eq!(webhook.url, "http://127.0.0.1:9090/alert");
        assert_eq!(webhook.thresholds, vec![0.5, 1.0]);
    }

    #[test]
    fn parse_config_rejects_negative_alerting_threshold() {
        let yaml = "\
model_list: []

oxllm_settings:
  alerting:
    - webhook
  alerting_config:
    webhook:
      url: http://127.0.0.1:9090/alert
      thresholds: [-0.5]

general_settings:
  allow_private_api_base: true
";
        let err =
            parse_config_str(yaml).expect_err("negative threshold must be rejected at config load");
        assert!(
            err.to_string().contains("threshold"),
            "error should mention threshold, got: {err}"
        );
    }

    #[test]
    fn parse_config_rejects_threshold_above_one() {
        let yaml = "\
model_list: []

oxllm_settings:
  slack_alerting_config:
    webhook_url: https://hooks.slack.com/services/T/B/X
    thresholds: [1.5]

general_settings:
  allow_private_api_base: true
";
        let err =
            parse_config_str(yaml).expect_err("threshold > 1.0 must be rejected at config load");
        assert!(err.to_string().contains("threshold"));
    }

    #[test]
    fn parse_config_alerting_uses_default_thresholds_when_omitted() {
        let yaml = "\
model_list: []

oxllm_settings:
  alerting:
    - webhook
  alerting_config:
    webhook:
      url: http://127.0.0.1:9090/alert

general_settings:
  allow_private_api_base: true
";
        let cfg = parse_config_str(yaml).expect("parses with default thresholds");
        let webhook = cfg
            .oxllm_settings
            .alerting_config
            .as_ref()
            .unwrap()
            .webhook
            .as_ref()
            .unwrap();
        assert_eq!(webhook.thresholds, vec![0.8, 0.9, 1.0]);
    }

    // ---- M9: environment_variables ----------------------------------------
    //
    // NOTE: these tests exercise the `cfg(test)` writing arm of
    // `apply_env_entries`, which compiles ONLY for this crate's own
    // `#[cfg(test)]` modules. The block is a no-op in shipped builds — see
    // `apply_environment_variables`. The integration test in
    // `tests/environment_variables_noop_in_shipped_build.rs` pins that
    // production behavior.

    #[allow(unsafe_code)]
    fn remove_test_env(name: &str) {
        unsafe { std::env::remove_var(name) };
    }

    #[test]
    fn environment_variables_are_applied_to_process_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        remove_test_env("OXLLM_TEST_ENV_APPLIED");
        let yaml = "\
model_list: []

environment_variables:
  OXLLM_TEST_ENV_APPLIED: from-config
";
        let _cfg = parse_config_str(yaml).expect("config with env block should parse");
        assert_eq!(
            std::env::var("OXLLM_TEST_ENV_APPLIED").as_deref(),
            Ok("from-config"),
            "environment_variables entry must be visible via std::env::var after load"
        );
        remove_test_env("OXLLM_TEST_ENV_APPLIED");
    }

    #[test]
    fn environment_variables_absent_block_is_noop() {
        let _guard = ENV_LOCK.lock().unwrap();
        remove_test_env("OXLLM_TEST_ENV_NOOP");
        let yaml = "model_list: []\n";
        let _cfg = parse_config_str(yaml).expect("plain config should parse");
        assert!(
            std::env::var("OXLLM_TEST_ENV_NOOP").is_err(),
            "no environment_variables block → var must stay unset"
        );
    }

    #[test]
    fn environment_variables_overwrites_existing_value() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_TEST_ENV_OVERWRITE", "initial");
        let yaml = "\
model_list: []

environment_variables:
  OXLLM_TEST_ENV_OVERWRITE: replaced
";
        let _cfg = parse_config_str(yaml).expect("config should parse");
        assert_eq!(
            std::env::var("OXLLM_TEST_ENV_OVERWRITE").as_deref(),
            Ok("replaced"),
            "environment_variables must overwrite a pre-existing env value"
        );
        remove_test_env("OXLLM_TEST_ENV_OVERWRITE");
    }

    #[test]
    fn environment_variables_non_mapping_is_rejected() {
        let _guard = ENV_LOCK.lock().unwrap();
        let yaml = "\
model_list: []

environment_variables: not-a-mapping
";
        let err = parse_config_str(yaml).expect_err("non-mapping env block must error");
        assert!(
            err.to_string().contains("environment_variables"),
            "error should mention environment_variables, got: {err}"
        );
    }

    #[test]
    fn environment_variables_non_string_value_is_rejected() {
        let _guard = ENV_LOCK.lock().unwrap();
        let yaml = "\
model_list: []

environment_variables:
  SOME_VAR: 12345
";
        let err = parse_config_str(yaml).expect_err("non-string env value must error");
        assert!(
            err.to_string().contains("SOME_VAR"),
            "error should name the offending key, got: {err}"
        );
    }

    // ---- M12: config path tracking ----------------------------------------

    #[test]
    fn current_config_path_tracks_load_config() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("oxllm_test_cfgpath_{}.yaml", uuid::Uuid::new_v4()));
        std::fs::write(&path, "model_list: []\n").expect("write temp config");
        let path_str = path.to_string_lossy().to_string();

        // load_config should record the path for /config/update.
        let _cfg = load_config(&path_str).expect("load temp config");
        assert_eq!(
            current_config_path().as_deref(),
            Some(path.as_path()),
            "current_config_path must reflect the most recent load_config call"
        );

        let _ = std::fs::remove_file(&path);
        // Reset the global so other tests don't see a stale path.
        if let Ok(mut g) = CONFIG_PATH.lock() {
            *g = None;
        }
    }

    #[test]
    fn current_config_path_unset_until_load() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        if let Ok(mut g) = CONFIG_PATH.lock() {
            *g = None;
        }
        assert!(
            current_config_path().is_none(),
            "current_config_path must be None until load_config records one"
        );
    }

    #[test]
    fn parse_config_rejects_private_otel_endpoint() {
        let yaml = "\
model_list: []

general_settings:
  otel_endpoint: http://127.0.0.1:4318
";
        let err = parse_config_str(yaml).expect_err("private otel_endpoint must be SSRF-validated");
        let full = format!("{err:#}");
        assert!(
            full.contains("private") || full.contains("otel_endpoint"),
            "expected private-IP error for otel_endpoint, got: {full}"
        );
    }

    #[test]
    fn parse_config_allows_private_otel_endpoint_when_opted_in() {
        let yaml = "\
model_list: []

general_settings:
  otel_endpoint: http://127.0.0.1:4318
  allow_private_api_base: true
";
        parse_config_str(yaml).expect("allow_private_api_base must permit loopback otel_endpoint");
    }

    // ---- M8: callback URL validation across all three lists ---------------

    #[test]
    fn parse_config_validates_generic_callback_url_in_failure_callback() {
        let yaml = "\
model_list: []

oxllm_settings:
  failure_callback:
    - generic
  callback_settings:
    generic:
      url: http://127.0.0.1:9090/spend
";
        let err = parse_config_str(yaml)
            .expect_err("generic in failure_callback must still be SSRF-validated");
        let full = format!("{err:#}");
        assert!(
            full.contains("private"),
            "expected private-IP error for generic callback, got: {full}"
        );
    }

    #[test]
    fn parse_config_validates_generic_callback_url_in_callbacks() {
        let yaml = "\
model_list: []

oxllm_settings:
  callbacks:
    - generic
  callback_settings:
    generic:
      url: http://127.0.0.1:9090/spend
";
        let err =
            parse_config_str(yaml).expect_err("generic in callbacks must still be SSRF-validated");
        let full = format!("{err:#}");
        assert!(
            full.contains("private"),
            "expected private-IP error for generic callback, got: {full}"
        );
    }

    // ---- outbound URL SSRF validation parity (slack/telemetry/cache/smtp) --

    #[test]
    fn parse_config_rejects_private_slack_webhook_url() {
        let yaml = "\
model_list: []

oxllm_settings:
  slack_alerting_config:
    webhook_url: http://127.0.0.1:9090/slack
    thresholds: [0.8, 1.0]
";
        let err = parse_config_str(yaml).expect_err("private slack webhook must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("slack") && full.contains("private"),
            "expected private-IP error naming slack, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_langfuse_host() {
        let yaml = "\
model_list: []

oxllm_settings:
  langfuse_config:
    host: http://127.0.0.1:3000
    public_key: pk
    secret_key: sk
";
        let err = parse_config_str(yaml).expect_err("private langfuse host must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("langfuse") && full.contains("private"),
            "expected private-IP error naming langfuse, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_mlflow_tracking_uri() {
        let yaml = "\
model_list: []

oxllm_settings:
  mlflow_config:
    tracking_uri: http://10.0.0.5:5000
";
        let err = parse_config_str(yaml).expect_err("private mlflow tracking_uri must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("mlflow") && full.contains("private"),
            "expected private-IP error naming mlflow, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_semantic_cache_qdrant_url() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      qdrant_url: http://127.0.0.1:6333
";
        let err = parse_config_str(yaml).expect_err("private qdrant url must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("qdrant") && full.contains("private"),
            "expected private-IP error naming qdrant, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_semantic_cache_redis_url() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      redis_url: redis://127.0.0.1:6379
";
        let err = parse_config_str(yaml).expect_err("private redis url must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("redis") && full.contains("private"),
            "expected private-IP error naming redis, got: {full}"
        );
    }

    #[test]
    fn parse_config_accepts_outbound_urls_when_private_allowed() {
        let yaml = "\
model_list: []

oxllm_settings:
  slack_alerting_config:
    webhook_url: http://127.0.0.1:9090/slack
    thresholds: [1.0]
  langfuse_config:
    host: http://127.0.0.1:3000
    public_key: pk
    secret_key: sk
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      qdrant_url: http://127.0.0.1:6333

general_settings:
  allow_private_api_base: true
";
        parse_config_str(yaml).expect("allow_private_api_base should permit all outbound targets");
    }

    // ---- credential redaction in SSRF error messages -----------------------

    #[test]
    fn parse_config_redacts_redis_password_in_ssrf_error() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      redis_url: redis://:leaked-password@127.0.0.1:6379
";
        let err =
            parse_config_str(yaml).expect_err("private redis URL with password must be rejected");
        let full = format!("{err:#}");
        assert!(
            !full.contains("leaked-password"),
            "SSRF error must NOT contain the Redis password, got: {full}"
        );
        assert!(
            full.contains("redis") && full.contains("private"),
            "expected private-IP error naming redis, got: {full}"
        );
    }

    #[test]
    fn parse_config_redacts_api_base_credentials_in_ssrf_error() {
        let yaml = "\
model_list:
  - model_name: leaky
    oxllm_params:
      model: openai/gpt-4o
      api_key: sk-test
      api_base: https://user:super-secret@127.0.0.1:8080/v1
";
        let err =
            parse_config_str(yaml).expect_err("private api_base with credentials must be rejected");
        let full = format!("{err:#}");
        assert!(
            !full.contains("super-secret"),
            "SSRF error must NOT contain the URL password, got: {full}"
        );
    }

    // ---- top-level cache.redis_url / cache.host SSRF validation -----------

    #[test]
    fn parse_config_rejects_private_top_level_cache_redis_url() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: redis
    redis_url: redis://127.0.0.1:6379
";
        let err = parse_config_str(yaml)
            .expect_err("private top-level cache.redis_url must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("redis") && full.contains("private"),
            "expected private-IP error for cache.redis_url, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_top_level_cache_host() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: redis
    host: 127.0.0.1
    port: 6379
";
        let err = parse_config_str(yaml).expect_err("private cache.host must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("private"),
            "expected private-IP error for cache.host, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_s3_cache_endpoint() {
        // `build_s3_cache` passes cache.s3.endpoint verbatim to the backend's
        // plain HTTP client, so it must be SSRF-validated at load like every
        // other outbound URL (MinIO on loopback requires opt-in).
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: s3
    s3:
      bucket: my-cache
      endpoint: http://127.0.0.1:9000
";
        let err =
            parse_config_str(yaml).expect_err("private cache.s3.endpoint must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("cache.s3.endpoint") && full.contains("private"),
            "expected private-IP error naming cache.s3.endpoint, got: {full}"
        );
    }

    #[test]
    fn parse_config_allows_private_s3_cache_endpoint_when_opted_in() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: s3
    s3:
      bucket: my-cache
      endpoint: http://127.0.0.1:9000

general_settings:
  allow_private_api_base: true
";
        parse_config_str(yaml).expect("allow_private_api_base should permit private s3 endpoint");
    }

    #[test]
    fn parse_config_rejects_private_s3_log_config_endpoint() {
        let yaml = "\
model_list: []

oxllm_settings:
  s3_log_config:
    bucket: my-llm-logs
    region: us-east-1
    access_key_id: AKIA...
    secret_access_key: sk
    endpoint: http://169.254.169.254:9000
";
        let err = parse_config_str(yaml)
            .expect_err("private s3_log_config.endpoint must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("s3_log_config.endpoint") && full.contains("private"),
            "expected private-IP error naming s3_log_config.endpoint, got: {full}"
        );
    }

    #[test]
    fn parse_config_allows_private_s3_log_config_endpoint_when_opted_in() {
        let yaml = "\
model_list: []

oxllm_settings:
  s3_log_config:
    bucket: my-llm-logs
    region: us-east-1
    access_key_id: AKIA...
    secret_access_key: sk
    endpoint: http://127.0.0.1:9000

general_settings:
  allow_private_api_base: true
";
        parse_config_str(yaml)
            .expect("allow_private_api_base should permit private s3 log endpoint");
    }

    // ---- PagerDuty endpoint + SMTP host SSRF validation -------------------

    #[test]
    fn parse_config_rejects_private_pagerduty_endpoint() {
        let yaml = "\
model_list: []

oxllm_settings:
  pagerduty_config:
    routing_key: rk-test
    endpoint: http://127.0.0.1:9090
";
        let err = parse_config_str(yaml)
            .expect_err("private pagerduty endpoint must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("pagerduty") && full.contains("private"),
            "expected private-IP error for pagerduty endpoint, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_smtp_host() {
        let yaml = "\
model_list: []

oxllm_settings:
  email_alert_config:
    smtp_host: 127.0.0.1
    smtp_port: 587
    from: alerts@example.com
    to: [oncall@example.com]
";
        let err = parse_config_str(yaml).expect_err("private smtp_host must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("smtp") && full.contains("private"),
            "expected private-IP error for smtp_host, got: {full}"
        );
    }

    // ---- soft_budget_pct range validation --------------------------------

    #[test]
    fn parse_config_rejects_out_of_range_soft_budget_pct_webhook() {
        // `allow_private_api_base: true` lets the unresolvable webhook host
        // pass SSRF validation so we reach the soft_budget_pct check. NaN is
        // also rejected by the validator's `is_finite()` guard, but YAML's
        // `.nan` collapses to `None` during deserialize so it never reaches
        // the check via config.
        for (label, value) in [
            ("negative", "soft_budget_pct: -0.5"),
            ("over_one", "soft_budget_pct: 1.5"),
        ] {
            let yaml = format!(
                "\
model_list: []

general_settings:
  allow_private_api_base: true

oxllm_settings:
  alerting:
    - webhook
  alerting_config:
    webhook:
      url: https://hooks.example.com/alert
      thresholds: [0.8, 0.9, 1.0]
      {value}
"
            );
            let res = parse_config_str(&yaml);
            let err = res.expect_err(&format!(
                "webhook soft_budget_pct ({label}) must be rejected"
            ));
            let full = format!("{err:#}");
            assert!(
                full.contains("soft_budget_pct"),
                "error should name the offending field for {label}, got: {full}"
            );
        }
    }

    #[test]
    fn parse_config_rejects_out_of_range_soft_budget_pct_slack() {
        let yaml = "\
model_list: []

general_settings:
  allow_private_api_base: true

oxllm_settings:
  slack_alerting_config:
    webhook_url: https://hooks.slack.com/services/T0/B0/XXX
    soft_budget_pct: 1.5
";
        let err = parse_config_str(yaml).expect_err("slack soft_budget_pct > 1.0 must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("slack_alerting_config") && full.contains("soft_budget_pct"),
            "error should name slack + soft_budget_pct, got: {full}"
        );
    }
    #[test]
    fn parse_config_rejects_out_of_range_soft_budget_pct_email() {
        let yaml = "\
model_list: []

general_settings:
  allow_private_api_base: true

oxllm_settings:
  email_alert_config:
    smtp_host: smtp.example.com
    smtp_port: 587
    from: alerts@example.com
    to: [oncall@example.com]
    soft_budget_pct: 2.0
";
        let err = parse_config_str(yaml).expect_err("email soft_budget_pct > 1.0 must be rejected");
        let full = format!("{err:#}");
        assert!(
            full.contains("email_alert_config") && full.contains("soft_budget_pct"),
            "error should name email + soft_budget_pct, got: {full}"
        );
    }

    #[test]
    fn parse_config_accepts_in_range_soft_budget_pct() {
        // Boundary values 0.0 and 1.0 are valid (unlike alert thresholds,
        // which require (0.0, 1.0]); `None` disables the feature.
        for value in ["0.0", "1.0", "0.8"] {
            let yaml = format!(
                "\
model_list: []

general_settings:
  allow_private_api_base: true

oxllm_settings:
  slack_alerting_config:
    webhook_url: https://hooks.slack.com/services/T0/B0/XXX
    soft_budget_pct: {value}
"
            );
            parse_config_str(&yaml)
                .unwrap_or_else(|e| panic!("in-range soft_budget_pct={value} must parse: {e}"));
        }
    }

    // ---- nested config deny_unknown_fields (typo detection) ---------------

    #[test]
    fn parse_config_rejects_typo_in_semantic_cache_threshold() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      similarit_threshold: 0.9
";
        let err = parse_config_str(yaml)
            .expect_err("typo in semantic_cache must be rejected by deny_unknown_fields");
        let full = format!("{err:#}");
        assert!(
            full.contains("unknown field") && full.contains("similarit_threshold"),
            "error should name the unknown field, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_out_of_range_similarity_threshold() {
        // Out-of-range values: a percentage-style typo (95.0) would make the
        // threshold unreachable (cosine similarity caps at 1.0, so the cache
        // never matches); a negative value matches everything. Reject both at
        // load, mirroring alerting thresholds. (NaN is also rejected by the
        // `is_finite()` guard in the validator, but YAML's `.nan` collapses
        // to `None` for `Option<f64>` during deserialize, so it can never
        // reach the check via config — it's covered by the unit-level guard
        // rather than this parse-level test.)
        for (label, raw) in [
            ("over_one", "1.5"),
            ("percentage_typo", "95.0"),
            ("negative", "-0.1"),
        ] {
            let yaml = format!(
                "\
model_list: []

oxllm_settings:
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      similarity_threshold: {raw}
"
            );
            let err = parse_config_str(&yaml)
                .expect_err(&format!("similarity_threshold ({label}) must be rejected"));
            let full = format!("{err:#}");
            assert!(
                full.contains("similarity_threshold"),
                "error should name similarity_threshold for {label}, got: {full}"
            );
        }
    }

    #[test]
    fn parse_config_accepts_in_range_similarity_threshold() {
        // 0.0 and 1.0 are valid boundary values.
        for raw in ["0.0", "1.0", "0.95"] {
            let yaml = format!(
                "\
model_list: []

oxllm_settings:
  cache:
    type: memory
    semantic_cache:
      enabled: true
      embedding_model: text-embedding-3-small
      similarity_threshold: {raw}
"
            );
            parse_config_str(&yaml)
                .unwrap_or_else(|e| panic!("in-range similarity_threshold={raw} must parse: {e}"));
        }
    }

    #[test]
    fn parse_config_rejects_typo_in_webhook_threshold() {
        let yaml = "\
model_list: []

oxllm_settings:
  alerting:
    - webhook
  alerting_config:
    webhook:
      url: https://hooks.example.com/alert
      treshold: 0.8
";
        let err = parse_config_str(yaml)
            .expect_err("typo in webhook config must be rejected by deny_unknown_fields");
        let full = format!("{err:#}");
        assert!(
            full.contains("unknown field") && full.contains("treshold"),
            "error should name the unknown field, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_typo_in_langfuse_flush_interval() {
        // `flush_intervall_secs` (double-l) is a plausible typo that, without
        // deny_unknown_fields, would be silently dropped so the operator
        // believes they configured the flush interval while the default runs.
        let yaml = "\
model_list: []

oxllm_settings:
  langfuse_config:
    host: https://cloud.langfuse.com
    public_key: pk
    secret_key: sk
    flush_intervall_secs: 5
";
        let err = parse_config_str(yaml)
            .expect_err("typo in langfuse_config must be rejected by deny_unknown_fields");
        let full = format!("{err:#}");
        assert!(
            full.contains("unknown field") && full.contains("flush_intervall_secs"),
            "error should name the unknown field, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_typo_in_presidio_config() {
        let yaml = "\
model_list: []

general_settings:
  guardrails:
    presidio:
      enabled: true
      url: http://presidio.example.com:5000
      languge: en
";
        let err = parse_config_str(yaml)
            .expect_err("typo in presidio config must be rejected by deny_unknown_fields");
        let full = format!("{err:#}");
        assert!(
            full.contains("unknown field") && full.contains("languge"),
            "error should name the unknown field, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_typo_in_s3_cache_config() {
        let yaml = "\
model_list: []

oxllm_settings:
  cache:
    type: s3
    s3:
      bucket: my-cache
      buckets: typo-extra
";
        let err = parse_config_str(yaml)
            .expect_err("typo in s3 cache config must be rejected by deny_unknown_fields");
        let full = format!("{err:#}");
        assert!(
            full.contains("unknown field") && full.contains("buckets"),
            "error should name the unknown field, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_typo_in_pass_through_endpoint() {
        let yaml = "\
model_list: []

general_settings:
  pass_through_endpoints:
    - path: /v1/custom
      target: https://custom.example.com
      headerz: {Authorization:Bearer x}
";
        let err = parse_config_str(yaml)
            .expect_err("typo in pass_through_endpoints must be rejected by deny_unknown_fields");
        let full = format!("{err:#}");
        assert!(
            full.contains("unknown field") && full.contains("headerz"),
            "error should name the unknown field, got: {full}"
        );
    }

    // ---- guardrail URL SSRF validation at load ---------------------------

    #[test]
    fn parse_config_rejects_private_guardrail_presidio_url() {
        let yaml = "\
model_list: []

general_settings:
  guardrails:
    presidio:
      enabled: true
      url: http://127.0.0.1:5000
";
        let err =
            parse_config_str(yaml).expect_err("private presidio url must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("presidio") && full.contains("private"),
            "expected private-IP error for presidio url, got: {full}"
        );
    }

    #[test]
    fn parse_config_rejects_private_guardrail_mcp_server_url() {
        let yaml = "\
model_list: []

general_settings:
  guardrails:
    mcp_guardrails:
      - name: test-mcp
        server_url: http://127.0.0.1:8765
";
        let err =
            parse_config_str(yaml).expect_err("private mcp server_url must be rejected at load");
        let full = format!("{err:#}");
        assert!(
            full.contains("mcp") && full.contains("private"),
            "expected private-IP error for mcp server_url, got: {full}"
        );
    }

    // ---- environment_variables blocked keys ------------------------------

    #[test]
    fn environment_variables_skips_blocked_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        let yaml = "\
model_list: []

environment_variables:
  PATH: /evil/bin
  HOME: /evil/home
  OXLLM_TEST_NOT_BLOCKED: should-still-work
";
        let _cfg = parse_config_str(yaml).expect("config with blocked keys should parse");
        // Blocked keys must NOT be applied.
        assert_ne!(
            std::env::var("PATH").unwrap_or_default(),
            "/evil/bin",
            "PATH must not be overridden by environment_variables"
        );
        // Non-blocked keys must still be applied.
        assert_eq!(
            std::env::var("OXLLM_TEST_NOT_BLOCKED").as_deref(),
            Ok("should-still-work"),
            "non-blocked env var must still be applied"
        );
        remove_test_env("OXLLM_TEST_NOT_BLOCKED");
    }

    /// Dynamic-linker injection variables (`LD_PRELOAD`,
    /// `DYLD_INSERT_LIBRARIES`, …) and TLS-trust overrides
    /// (`SSL_CERT_FILE` / `SSL_CERT_DIR`) must be on the blocked list just
    /// like `PATH`/`HOME`: a future in-process env overlay would otherwise
    /// turn `environment_variables` into a code-injection / trust-bypass
    /// vector via config.
    #[test]
    fn environment_variables_skips_dynamic_linker_keys() {
        let _guard = ENV_LOCK.lock().unwrap();
        let yaml = "\
model_list: []

environment_variables:
  LD_PRELOAD: /evil/libinject.so
  LD_LIBRARY_PATH: /evil/lib
  DYLD_LIBRARY_PATH: /evil/lib
  DYLD_INSERT_LIBRARIES: /evil/libinject.dylib
  SSL_CERT_FILE: /evil/ca-bundle.pem
  SSL_CERT_DIR: /evil/certs
  OXLLM_TEST_NOT_BLOCKED: should-still-work
";
        let _cfg = parse_config_str(yaml).expect("config with blocked keys should parse");
        for key in [
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        ] {
            // The dynamic-linker / TLS vars are process-identity sensitive:
            // they must not be present in the test environment at all (we
            // never set them), and the block must ensure the values from the
            // YAML never land in the process env.
            assert_ne!(
                std::env::var(key).unwrap_or_default(),
                match key {
                    "LD_PRELOAD" => "/evil/libinject.so",
                    "LD_LIBRARY_PATH" => "/evil/lib",
                    "DYLD_LIBRARY_PATH" => "/evil/lib",
                    "DYLD_INSERT_LIBRARIES" => "/evil/libinject.dylib",
                    "SSL_CERT_FILE" => "/evil/ca-bundle.pem",
                    _ => "/evil/certs",
                },
                "{key} must not be overridden by environment_variables"
            );
        }
        assert_eq!(
            std::env::var("OXLLM_TEST_NOT_BLOCKED").as_deref(),
            Ok("should-still-work"),
            "non-blocked env var must still be applied"
        );
        remove_test_env("OXLLM_TEST_NOT_BLOCKED");
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "oxllm-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn write_file(dir: &std::path::Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).expect("write fixture");
    }

    #[test]
    fn include_concatenates_models_and_replaces_settings_blocks() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = temp_dir("include");
        write_file(
            &dir,
            "c.yaml",
            "\
model_list:
  - model_name: from-c
    oxllm_params:
      model: openai/gpt-4o
router_settings:
  num_retries: 9
  routing_strategy: simple_shuffle
",
        );
        write_file(
            &dir,
            "b.yaml",
            "\
include:
  - c.yaml
model_list:
  - model_name: from-b
    oxllm_params:
      model: openai/gpt-4o
router_settings:
  num_retries: 1
",
        );
        write_file(
            &dir,
            "a.yaml",
            "\
include:
  - b.yaml
model_list:
  - model_name: from-a
    oxllm_params:
      model: openai/gpt-4o
",
        );
        let cfg = load_config(dir.join("a.yaml").to_str().expect("utf8")).expect("include load");
        let names: Vec<_> = cfg
            .model_list
            .iter()
            .map(|m| m.model_name.as_str())
            .collect();
        assert_eq!(names, ["from-a", "from-b", "from-c"]);
        assert_eq!(cfg.router_settings.routing.num_retries, Some(9));
        assert_eq!(
            cfg.router_settings.routing.routing_strategy.as_deref(),
            Some("simple_shuffle")
        );
        assert_eq!(
            current_config_path().as_ref(),
            Some(&dir.join("a.yaml")),
            "write-back path stays the root file"
        );
        assert!(config_contributor_paths().len() >= 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn include_cycle_names_the_chain() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = temp_dir("cycle");
        write_file(&dir, "a.yaml", "include:\n  - b.yaml\nmodel_list: []\n");
        write_file(&dir, "b.yaml", "include:\n  - a.yaml\nmodel_list: []\n");
        let err =
            load_config(dir.join("a.yaml").to_str().expect("utf8")).expect_err("cycle must fail");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("include cycle"), "{rendered}");
        assert!(rendered.contains("->"), "{rendered}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn include_shallow_settings_replacement() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = temp_dir("shallow");
        write_file(
            &dir,
            "b.yaml",
            "\
model_list: []
router_settings:
  num_retries: 1
",
        );
        write_file(
            &dir,
            "a.yaml",
            "\
include:
  - b.yaml
model_list: []
router_settings:
  num_retries: 3
  routing_strategy: simple_shuffle
",
        );
        let cfg = load_config(dir.join("a.yaml").to_str().expect("utf8")).expect("load");
        assert_eq!(cfg.router_settings.routing.num_retries, Some(1));
        assert!(cfg.router_settings.routing.routing_strategy.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn env_refs_coerce_int_float_and_bool_after_typed_failure() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_TEST_RPM", "30");
        set_env("OXLLM_TEST_TIMEOUT", "1.5");
        set_env("OXLLM_TEST_DROP", "true");
        set_env("OXLLM_TEST_KEY", "sk-test");
        let yaml = "\
model_list:
  - model_name: m
    oxllm_params:
      model: openai/gpt-4o
      api_key: os.environ/OXLLM_TEST_KEY
      rpm: os.environ/OXLLM_TEST_RPM
      timeout: os.environ/OXLLM_TEST_TIMEOUT
oxllm_settings:
  drop_params: os.environ/OXLLM_TEST_DROP
";
        let cfg = parse_config_str(yaml).expect("coerced env refs");
        assert_eq!(
            cfg.model_list[0].oxllm_params.api_key.as_deref(),
            Some("sk-test")
        );
        assert_eq!(cfg.model_list[0].oxllm_params.rpm, Some(30));
        assert_eq!(cfg.model_list[0].oxllm_params.timeout, Some(1.5));
        assert_eq!(cfg.oxllm_settings.drop_params, Some(true));
        unset_env("OXLLM_TEST_RPM");
        unset_env("OXLLM_TEST_TIMEOUT");
        unset_env("OXLLM_TEST_DROP");
        unset_env("OXLLM_TEST_KEY");
    }

    #[test]
    fn ordinary_yaml_string_is_not_coerced() {
        let yaml = "\
model_list:
  - model_name: m
    oxllm_params:
      model: openai/gpt-4o
      rpm: \"30\"
";
        let err = parse_config_str(yaml).expect_err("quoted rpm stays a string");
        assert!(format!("{err:#}").contains("rpm") || format!("{err:#}").contains("invalid type"));
    }

    #[test]
    fn unparseable_env_ref_names_the_field_and_variable() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_TEST_BANANA", "banana");
        let yaml = "\
model_list:
  - model_name: m
    oxllm_params:
      model: openai/gpt-4o
      rpm: os.environ/OXLLM_TEST_BANANA
";
        let err = parse_config_str(yaml).expect_err("banana is not an int");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("rpm"), "{rendered}");
        assert!(rendered.contains("OXLLM_TEST_BANANA"), "{rendered}");
        unset_env("OXLLM_TEST_BANANA");
    }

    #[test]
    fn supported_environments_filter_after_validate() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_env("OXLLM_ENVIRONMENT");
        unset_env("LITELLM_ENVIRONMENT");
        let yaml = "\
model_list:
  - model_name: always
    oxllm_params:
      model: openai/gpt-4o
  - model_name: dev-only
    oxllm_params:
      model: openai/gpt-4o
    model_info:
      supported_environments: [development]
  - model_name: empty-list
    oxllm_params:
      model: openai/gpt-4o
    model_info:
      supported_environments: []
  - model_name: prod-only
    oxllm_params:
      model: openai/gpt-4o
    model_info:
      supported_environments: [production]
";
        let cfg = parse_config_str(yaml).expect("default environment");
        let names: Vec<_> = cfg
            .model_list
            .iter()
            .map(|m| m.model_name.as_str())
            .collect();
        assert_eq!(names, ["always", "dev-only"]);

        set_env("LITELLM_ENVIRONMENT", "production");
        set_env("OXLLM_ENVIRONMENT", "staging");
        let cfg = parse_config_str(yaml).expect("oxllm env wins");
        let names: Vec<_> = cfg
            .model_list
            .iter()
            .map(|m| m.model_name.as_str())
            .collect();
        assert_eq!(names, ["always"]);
        unset_env("OXLLM_ENVIRONMENT");
        unset_env("LITELLM_ENVIRONMENT");
    }

    #[test]
    fn cache_injection_rejected_only_when_prompt_caching_is_disabled() {
        let disabled = "\
model_list:
  - model_name: m
    oxllm_params:
      model: anthropic/claude-3-5-sonnet
      cache_control_injection_points:
        - location: message
          role: system
oxllm_settings:
  prompt_caching_enabled: false
";
        let err = parse_config_str(disabled).expect_err("disabled caching rejects points");
        assert!(format!("{err:#}").contains("cache_control_injection_points"));

        let unset = "\
model_list:
  - model_name: m
    oxllm_params:
      model: anthropic/claude-3-5-sonnet
      cache_control_injection_points:
        - location: message
          index: -1
";
        let cfg = parse_config_str(unset).expect("unset caching allows points");
        assert_eq!(
            cfg.model_list[0]
                .oxllm_params
                .cache_control_injection_points
                .as_ref()
                .map(|p| p.len()),
            Some(1)
        );
    }

    #[test]
    fn named_retry_policy_maps_onto_status_and_exception_fields() {
        let yaml = "\
model_list: []
oxllm_settings:
  retry_policy:
    TimeoutErrorRetries: 4
router_settings:
  retry_policy:
    TimeoutErrorRetries: 1
    RateLimitErrorRetries: 1
    ContentPolicyViolationErrorRetries: 0
    InternalServerErrorRetries: 2
    BadRequestErrorRetries: 0
    AuthenticationErrorRetries: 0
    \"408\": 7
  routing_groups:
    gpt-4o:
      retry_policy:
        RateLimitErrorRetries: 3
";
        let cfg = parse_config_str(yaml).expect("named retries");
        assert_eq!(
            cfg.oxllm_settings
                .routing
                .exception_retry_policy
                .as_ref()
                .and_then(|p| p.timeout_retries),
            Some(4)
        );
        let router = &cfg.router_settings.routing;
        assert_eq!(router.retry_policy.as_ref().unwrap().get("408"), Some(&7));
        assert_eq!(router.retry_policy.as_ref().unwrap().get("429"), Some(&1));
        assert_eq!(router.retry_policy.as_ref().unwrap().get("400"), Some(&0));
        assert_eq!(router.retry_policy.as_ref().unwrap().get("504"), Some(&2));
        assert_eq!(router.retry_policy.as_ref().unwrap().get("529"), Some(&2));
        assert_eq!(router.retry_policy.as_ref().unwrap().get("422"), Some(&0));
        assert_eq!(router.retry_policy.as_ref().unwrap().get("401"), Some(&0));
        assert!(
            !router
                .retry_policy
                .as_ref()
                .unwrap()
                .contains_key("TimeoutErrorRetries")
        );
        let policy = router.exception_retry_policy.as_ref().expect("exception");
        assert_eq!(policy.timeout_retries, Some(7));
        assert_eq!(policy.rate_limit_retries, Some(1));
        assert_eq!(policy.content_policy_retries, Some(0));
        assert_eq!(policy.internal_server_error_retries, Some(2));
        let allowed = router.allowed_fails_policy.as_ref().expect("allowed");
        assert_eq!(allowed.get("400"), Some(&0));
        assert_eq!(allowed.get("422"), Some(&0));
        assert_eq!(allowed.get("401"), Some(&0));
        assert_eq!(allowed.get("403"), Some(&0));
        let group = cfg
            .router_settings
            .routing_groups
            .as_ref()
            .unwrap()
            .get("gpt-4o")
            .unwrap();
        assert_eq!(
            group.routing.retry_policy.as_ref().unwrap().get("429"),
            Some(&3)
        );
    }

    #[test]
    fn nonzero_bad_request_retry_alias_fails_validate() {
        let yaml = "\
model_list: []
router_settings:
  retry_policy:
    BadRequestErrorRetries: 2
";
        let err = parse_config_str(yaml).expect_err("nonzero bad request");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("no retry translation"), "{rendered}");
        assert!(rendered.contains("BadRequestErrorRetries"), "{rendered}");
    }

    #[test]
    fn short_master_key_hatch_is_development_only() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_env("OXLLM_ENVIRONMENT");
        unset_env("LITELLM_ENVIRONMENT");
        let yaml = "\
model_list: []
general_settings:
  master_key: short-key
  allow_short_master_key: true
";
        let cfg = parse_config_str(yaml).expect("development accepts a short key");
        assert!(!crate::app::auth_is_configured(
            &cfg.general_settings.master_key,
            &cfg.general_settings.jwt_secret,
            &cfg.general_settings.database_url,
        ));
        assert!(crate::app::settings_auth_is_configured(
            &cfg.general_settings
        ));

        set_env("OXLLM_ENVIRONMENT", "production");
        let err = parse_config_str(yaml).expect_err("production rejects the hatch");
        assert!(format!("{err:#}").contains("allow_short_master_key"));
        let long = "\
model_list: []
general_settings:
  master_key: a-long-enough-master
  allow_short_master_key: true
";
        assert!(parse_config_str(long).is_err());
        unset_env("OXLLM_ENVIRONMENT");
        unset_env("LITELLM_ENVIRONMENT");
    }

    #[test]
    fn env_coercion_preserves_numeric_string_fields_and_coerces_each_numeric_target() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_TEST_SHARED_SCALAR", "30");
        set_env("OXLLM_TEST_BOOLEAN_SCALAR", "true");
        let config = parse_config_str(
            "\
model_list:
  - model_name: os.environ/OXLLM_TEST_SHARED_SCALAR
    oxllm_params:
      model: openai/gpt-4o
      api_key: os.environ/OXLLM_TEST_SHARED_SCALAR
      rpm: os.environ/OXLLM_TEST_SHARED_SCALAR
      tpm: os.environ/OXLLM_TEST_SHARED_SCALAR
      extra_headers:
        X-Boolean-String: os.environ/OXLLM_TEST_BOOLEAN_SCALAR
oxllm_settings:
  drop_params: os.environ/OXLLM_TEST_BOOLEAN_SCALAR
",
        )
        .expect("only numeric/bool fields coerce");
        unset_env("OXLLM_TEST_SHARED_SCALAR");
        unset_env("OXLLM_TEST_BOOLEAN_SCALAR");
        let params = &config.model_list[0].oxllm_params;
        assert_eq!(config.model_list[0].model_name, "30");
        assert_eq!(params.api_key.as_deref(), Some("30"));
        assert_eq!(params.rpm, Some(30));
        assert_eq!(params.tpm, Some(30));
        assert_eq!(
            params.extra_headers.as_ref().unwrap()["X-Boolean-String"],
            "true"
        );
        assert_eq!(config.oxllm_settings.drop_params, Some(true));
    }

    #[test]
    fn blank_primary_environment_cannot_bypass_the_legacy_production_environment() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_ENVIRONMENT", "  ");
        set_env("LITELLM_ENVIRONMENT", "production");
        let environment = resolved_environment();
        let result = parse_config_str(
            "model_list: []\ngeneral_settings:\n  master_key: short\n  allow_short_master_key: true\n",
        );
        unset_env("OXLLM_ENVIRONMENT");
        unset_env("LITELLM_ENVIRONMENT");
        assert_eq!(environment, "production");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("allow_short_master_key")
        );
    }

    #[test]
    fn include_cycle_is_detected_when_both_files_are_already_queued() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "root.yaml",
            "model_list: []\ninclude: [a.yaml, b.yaml]\n",
        );
        write_file(dir.path(), "a.yaml", "model_list: []\ninclude: [b.yaml]\n");
        write_file(dir.path(), "b.yaml", "model_list: []\ninclude: [a.yaml]\n");
        let error = load_config(dir.path().join("root.yaml").to_str().unwrap()).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("include cycle"), "{message}");
        assert!(
            message.contains("a.yaml") && message.contains("b.yaml"),
            "{message}"
        );
    }

    #[test]
    fn nested_includes_overlay_after_siblings_and_duplicate_files_merge_once() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "root.yaml",
            "model_list: []\ninclude: [a.yaml, b.yaml]\n",
        );
        write_file(
            dir.path(),
            "a.yaml",
            "model_list: []\ninclude: [nested.yaml]\nrouter_settings:\n  num_retries: 1\n",
        );
        write_file(
            dir.path(),
            "b.yaml",
            "model_list: []\ninclude: [nested.yaml]\nrouter_settings:\n  num_retries: 2\n",
        );
        write_file(
            dir.path(),
            "nested.yaml",
            "model_list:\n  - model_name: nested\n    oxllm_params:\n      model: openai/gpt-4o\nrouter_settings:\n  num_retries: 3\n",
        );
        let config = load_config(dir.path().join("root.yaml").to_str().unwrap()).unwrap();
        assert_eq!(config.router_settings.routing.num_retries, Some(3));
        assert_eq!(config.model_list.len(), 1);
        assert_eq!(config.model_list[0].model_name, "nested");
    }

    #[test]
    fn includes_replace_the_same_settings_block_across_legacy_and_native_keys() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_file(
            dir.path(),
            "root.yaml",
            "model_list: []\ninclude: [included.yaml]\noxllm_settings:\n  drop_params: true\n  request_timeout: 10\n",
        );
        write_file(
            dir.path(),
            "included.yaml",
            "model_list: []\nlitellm_settings:\n  request_timeout: 20\n",
        );
        let config = load_config(dir.path().join("root.yaml").to_str().unwrap()).unwrap();
        assert_eq!(config.oxllm_settings.request_timeout, Some(20));
        assert_eq!(config.oxllm_settings.drop_params, None);
    }

    #[test]
    fn proposed_include_config_is_validated_without_publishing_contributors() {
        let _guard = CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "root.yaml", "model_list: []\n");
        let root = dir.path().join("root.yaml");
        load_config(root.to_str().unwrap()).unwrap();
        let contributors = config_contributor_paths();
        write_file(
            dir.path(),
            "models.yaml",
            "model_list:\n  - model_name: included\n    oxllm_params:\n      model: openai/gpt-4o\n",
        );
        let config =
            validate_config_value(serde_json::json!({"include": ["models.yaml"]}), &root).unwrap();
        assert_eq!(config.model_list[0].model_name, "included");
        assert_eq!(config_contributor_paths(), contributors);
        assert!(
            validate_config_value(serde_json::json!({"include": ["missing.yaml"]}), &root).is_err()
        );
    }

    #[test]
    fn invalid_cache_selectors_are_rejected_at_load() {
        for point in [
            serde_json::json!({"location": "tool", "index": 0}),
            serde_json::json!({"location": "message"}),
            serde_json::json!({"location": "message", "index": 0, "role": "system"}),
            serde_json::json!({"location": "message", "role": "bogus"}),
        ] {
            let config = serde_json::json!({"model_list": [{"model_name": "m", "oxllm_params": {
                "model": "anthropic/claude-test", "cache_control_injection_points": [point]
            }}]});
            let error = parse_config_str(&serde_yml::to_string(&config).unwrap()).unwrap_err();
            assert!(format!("{error:#}").contains("cache_control_injection_points"));
        }
    }

    #[test]
    fn inactive_deployments_do_not_require_their_credential_or_limit_env_variables() {
        let config = parse_config_str(
            "\
model_list:
  - model_name: active
    oxllm_params:
      model: openai/gpt-4o
  - model_name: inactive
    oxllm_params:
      model: xai/grok-test
      api_key: os.environ/OXLLM_INACTIVE_PROVIDER_SECRET
      rpm: os.environ/OXLLM_INACTIVE_PROVIDER_RPM
    model_info:
      supported_environments: []
",
        )
        .expect("inactive credentials are not resolved");
        assert_eq!(config.model_list.len(), 1);
        assert_eq!(config.model_list[0].model_name, "active");
    }

    #[test]
    fn programmatic_state_construction_also_filters_inactive_deployments() {
        let config: ProxyConfig = serde_json::from_value(serde_json::json!({"model_list": [
            {"model_name": "active", "oxllm_params": {"model": "openai/gpt-4o"}},
            {"model_name": "inactive", "oxllm_params": {"model": "openai/gpt-4o"},
                "model_info": {"supported_environments": []}}
        ]}))
        .unwrap();
        let state = crate::state::AppState::new(config).unwrap();
        assert_eq!(state.deployments.len(), 1);
        assert_eq!(state.deployments[0].model_name, "active");
    }
}
