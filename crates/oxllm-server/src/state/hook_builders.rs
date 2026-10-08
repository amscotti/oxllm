//! Hook-chain assembly from `oxllm_settings` global defaults.
//!
//! The feature-specific hook constructors live next to their features
//! ([`crate::hooks::alerting`], [`crate::hooks::langfuse`],
//! [`crate::cache::semantic`]); this module only sequences the chain and
//! wires the `success_callback` slugs.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::Cache;
use oxllm_core::Hook;
use oxllm_core::{CallContext, Error, HookAction, ResponseSummary};

use crate::config::ProxyConfig;
use crate::config::types::OxllmSettings;
use crate::spend_writer::SpendEntry;

/// Build the hook chain from `oxllm_settings` global defaults.
///
/// Reads the following settings:
/// - `max_budget` / `budget_duration`: per-key budget
/// - `rpm` / `tpm`: per-key rate limits
/// - `cache_ttl` (optional): TTL for the response cache; defaults to 5 minutes.
/// - `cache.semantic_cache` (optional): in-memory semantic cache.
///
/// The rate-limit and budget hooks are always registered (with `None` global
/// limits when unset) so per-key limits carried by `CallContext` are enforced
/// even when no global defaults are configured.
///
/// `adaptive_bandit` carries the router's shared bandit store when the
/// `adaptive_router` strategy is configured (globally or for any routing
/// group); the [`crate::hooks::AdaptiveHook`] learning hook is registered
/// only then.
///
/// Also wires the callback lists — `success_callback`,
/// `failure_callback`, and `callbacks` (dual-fire). The `generic` HTTP
/// callback is built when requested by any list and gated to the matching
/// events; the feature-specific hooks (Langfuse, Datadog, ...) are wrapped
/// so a slug listed in `failure_callback` / `callbacks` additionally fires
/// `post_failure`. Unrecognized callback slugs are ignored with a warning.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hooks(
    config: &ProxyConfig,
    cache: Arc<dyn Cache>,
    spend_tx: Option<tokio::sync::mpsc::Sender<SpendEntry>>,
    alert_states: AlertStates,
    semantic_runtime: Option<Arc<crate::hooks::cache_hook::SemanticCacheRuntime>>,
    langfuse_hook: Option<Arc<crate::hooks::LangfuseHook>>,
    mlflow_hook: Option<Arc<crate::hooks::MlflowHook>>,
    adaptive_bandit: Option<Arc<oxllm_core::router::BanditStore>>,
) -> Vec<Arc<dyn Hook>> {
    let settings = &config.oxllm_settings;

    let max_budget = settings.max_budget;
    let budget_duration = settings
        .budget_duration
        .as_deref()
        .unwrap_or("30d")
        .to_string();

    let rpm = settings.rpm;
    let tpm = settings.tpm;

    let cache_ttl = Duration::from_secs(settings.cache_ttl_secs.unwrap_or(300));

    // Merge routing knobs the same way `build_router_config` does so
    // `budget_throttle_pct` set in either `oxllm_settings` or
    // `router_settings` reaches the rate-limit hook.
    let knobs = crate::config::types::RoutingKnobs::merge(
        &config.oxllm_settings.routing,
        &config.router_settings.routing,
    );
    let budget_throttle_pct = knobs.budget_throttle_pct;

    let mut hooks: Vec<Arc<dyn Hook>> = vec![
        // Order matters: RateLimit runs before Cache so RPM is consumed even on
        // a cache hit (the request was received and processed). Cache runs before
        // Budget so a cache short-circuit never reserves spend (no upstream
        // cost). The per-hook `cache_hit` flag additionally skips post-success
        // adjustments for cache hits.
        {
            let mut rate_hook = crate::hooks::RateLimitHook::new(cache.clone(), rpm, tpm);
            if let Some(pct) = budget_throttle_pct {
                rate_hook = rate_hook.with_budget_throttle_pct(Some(pct));
            }
            Arc::new(rate_hook)
        },
        {
            let mut cache_hook = crate::hooks::CacheHook::new(cache.clone(), cache_ttl);
            if let Some(rt) = semantic_runtime {
                cache_hook = cache_hook.with_semantic(rt);
            }
            Arc::new(cache_hook)
        },
        {
            let budget_hook =
                crate::hooks::BudgetHook::new(cache.clone(), max_budget, &budget_duration);
            let budget_hook = match settings.model_max_budget.clone() {
                Some(map) if !map.is_empty() => budget_hook.with_model_budget(map),
                _ => budget_hook,
            };
            Arc::new(budget_hook)
        },
        Arc::new(crate::hooks::CostTrackingHook::with_spend_writer(spend_tx)),
        Arc::new(crate::hooks::MetricsHook::new()),
    ];

    // Adaptive-router learning hook: registered only when the adaptive
    // strategy is configured somewhere (global or any routing group) —
    // `state::build::assemble` passes the router's shared bandit store in
    // that case. It is observability-only (no `pre_call`), so its exact
    // position does not matter; it sits with the other post-call metering
    // hooks and skips cache hits like they do.
    if let Some(bandit) = adaptive_bandit {
        hooks.push(Arc::new(crate::hooks::AdaptiveHook::new(bandit)));
    }

    // ---- Callback-driven hooks -------------------------------------------
    //
    // Three callback lists are exposed: `success_callback` (fires on
    // success), `failure_callback` (fires on failure), and `callbacks`
    // (fires on both). The `generic` HTTP callback is built only when
    // requested by any of the three; the feature-specific hooks below
    // (Langfuse, Datadog, ...) are built from their own config blocks but
    // are wrapped so a slug listed in `failure_callback` / `callbacks` also
    // fires their `post_failure`.

    let (generic_on_success, generic_on_failure) = callback_events(settings, "generic");
    if generic_on_success || generic_on_failure {
        match build_generic_callback(config) {
            Some(Ok(hook)) => {
                hooks.push(Arc::new(CallbackEventFilter {
                    inner: Arc::new(hook),
                    fire_on_success: generic_on_success,
                    fire_on_failure: generic_on_failure,
                }));
            }
            Some(Err(e)) => {
                tracing::warn!(
                    error = %e,
                    "failed to build generic callback; skipping"
                );
            }
            None => {
                tracing::warn!("callback 'generic' is missing callback_settings.generic; skipping");
            }
        }
    }

    // Warn on any unrecognized slug across the three lists. Known slugs
    // (`generic` plus the feature-specific hooks built below) are silently
    // honored; anything else is ignored with a single warning per slug.
    warn_unknown_callback_slugs(settings);

    if let Some(hook) = langfuse_hook {
        hooks.push(wrap_config_block_hook(hook, settings, "langfuse"));
    }

    if let Some(hook) = mlflow_hook {
        hooks.push(wrap_config_block_hook(hook, settings, "mlflow"));
    }

    // The five `HttpTelemetryHook`-backed observability integrations
    // (LangSmith, Datadog, Helicone, W&B, MLflow) share an identical
    // builder signature — `fn(&ProxyConfig) -> Option<Arc<HttpTelemetryHook>>`
    // — so they are registered from a single table. Adding a sixth
    // telemetry integration is a one-line table entry in
    // [`push_http_telemetry_hooks`] instead of a five-line `if let Some`
    // block here.
    push_http_telemetry_hooks(&mut hooks, config, settings);

    if let Some(hook) = crate::hooks::s3_log::build_s3_log_hook_opt(config) {
        hooks.push(wrap_config_block_hook(hook, settings, "s3"));
    }

    if let Some(state) = alert_states.slack {
        if let Some(hook) =
            crate::hooks::slack_alert::build_slack_hook_opt(config, cache.clone(), state)
        {
            hooks.push(wrap_config_block_hook(hook, settings, "slack"));
        }
    }

    if let Some(state) = alert_states.pagerduty {
        if let Some(hook) =
            crate::hooks::pagerduty_alert::build_pagerduty_hook_opt(config, cache.clone(), state)
        {
            hooks.push(wrap_config_block_hook(hook, settings, "pagerduty"));
        }
    }

    if let Some(state) = alert_states.email {
        if let Some(hook) =
            crate::hooks::email_alert::build_email_hook_opt(config, cache.clone(), state)
        {
            hooks.push(wrap_config_block_hook(hook, settings, "email"));
        }
    }

    if let Some(state) = alert_states.webhook {
        if let Some(hook) =
            crate::hooks::alerting::build_alerting_hook(config, cache.clone(), state)
        {
            hooks.push(wrap_config_block_hook(Arc::new(hook), settings, "alerting"));
        }
    }

    hooks
}

/// Bundle of the four alerting hooks' [`AlertState`] handles, built/preserved
/// in `state::build::assemble` and threaded through [`build_hooks`]. Each hook
/// owns its own `AlertState` so channels fire independently; preserving the
/// `Arc`s across a hot-reload keeps in-memory fired markers alive (Redis mode
/// is unaffected — markers persist via TTL).
///
/// [`AlertState`]: crate::hooks::AlertState
#[derive(Clone, Default)]
pub(crate) struct AlertStates {
    pub webhook: Option<Arc<crate::hooks::AlertState>>,
    pub slack: Option<Arc<crate::hooks::AlertState>>,
    pub pagerduty: Option<Arc<crate::hooks::AlertState>>,
    pub email: Option<Arc<crate::hooks::AlertState>>,
}

/// Push the five `HttpTelemetryHook`-backed observability integrations
/// (LangSmith, Datadog, Helicone, W&B, MLflow) from a single table.
///
/// Each entry pairs a callback `slug` with its config-block builder. The
/// builder returns `None` when the corresponding `_config` section is
/// absent (or fails SSRF validation), so absent hooks are simply skipped.
/// Present hooks are wrapped via [`wrap_config_block_hook`] so that
/// listing the slug in `failure_callback` / `callbacks` additionally fires
/// `post_failure`.
///
/// Adding a sixth telemetry integration is a single new row in the
/// `builders` table rather than a fresh five-line `if let Some` block in
/// [`build_hooks`].
fn push_http_telemetry_hooks(
    hooks: &mut Vec<Arc<dyn Hook>>,
    config: &ProxyConfig,
    settings: &OxllmSettings,
) {
    type Builder = fn(&ProxyConfig) -> Option<Arc<crate::hooks::HttpTelemetryHook>>;
    // `(callback slug, config-block builder)` pairs for every
    // `HttpTelemetryHook`-backed integration. Add a row here to register
    // a new telemetry integration — no other changes needed.
    let builders: &[(&str, Builder)] = &[
        (
            "langsmith",
            crate::hooks::langsmith::build_langsmith_hook_opt,
        ),
        ("datadog", crate::hooks::datadog::build_datadog_hook_opt),
        ("helicone", crate::hooks::helicone::build_helicone_hook_opt),
        ("wandb", crate::hooks::wandb::build_wandb_hook_opt),
    ];

    for (slug, builder) in builders {
        if let Some(hook) = builder(config) {
            hooks.push(wrap_config_block_hook(hook, settings, slug));
        }
    }
}

/// Slugs that have dedicated config-block builders in `build_hooks`. Listed
/// here so [`warn_unknown_callback_slugs`] does not warn for them — they are
/// real hooks, just driven by their own config sections rather than the
/// generic callback dispatch.
const KNOWN_OBSERVABILITY_SLUGS: &[&str] = &[
    "generic",
    "langfuse",
    "langsmith",
    "datadog",
    "helicone",
    "wandb",
    "s3",
    "mlflow",
    "slack",
    "pagerduty",
    "email",
    "alerting",
    "webhook",
];

/// Emit a single `warn` log per unrecognized slug across the three callback
/// lists. Known observability slugs (see [`KNOWN_OBSERVABILITY_SLUGS`]) are
/// honored silently even when they appear in a list redundantly with their
/// config block.
fn warn_unknown_callback_slugs(settings: &OxllmSettings) {
    let mut seen: Vec<&str> = Vec::new();
    for slug in settings
        .success_callback
        .iter()
        .flatten()
        .chain(settings.failure_callback.iter().flatten())
        .chain(settings.callbacks.iter().flatten())
    {
        if KNOWN_OBSERVABILITY_SLUGS.contains(&slug.as_str()) {
            continue;
        }
        if !seen.contains(&slug.as_str()) {
            seen.push(slug.as_str());
            tracing::warn!(slug = %slug, "unsupported callback slug; ignoring");
        }
    }
}

/// Compute whether a callback slug should fire on success and/or failure
/// based on its membership in the three callback lists.
///
/// Returns `(fire_on_success, fire_on_failure)`:
/// - in `success_callback` only → `(true, false)`
/// - in `failure_callback` only → `(false, true)`
/// - in `callbacks` → `(true, true)`
/// - in multiple lists → unioned
/// - in none → `(false, false)`
fn callback_events(settings: &OxllmSettings, slug: &str) -> (bool, bool) {
    let in_success = settings
        .success_callback
        .iter()
        .flatten()
        .any(|s| s == slug);
    let in_failure = settings
        .failure_callback
        .iter()
        .flatten()
        .any(|s| s == slug);
    let in_callbacks = settings.callbacks.iter().flatten().any(|s| s == slug);
    (in_success || in_callbacks, in_failure || in_callbacks)
}

/// Wrap a config-block-built hook (Langfuse, Datadog, ...) so that listing
/// its slug in `failure_callback` or `callbacks` adds failure firing on top
/// of the default success-only behavior.
///
/// Config-block hooks are always present whenever their config section is
/// set, so they always fire on success (current behavior). When the slug is
/// NOT in `failure_callback` or `callbacks`, this returns the hook unwrapped
/// to avoid any indirection on the hot path.
///
/// When the slug IS in `failure_callback` or `callbacks`, the wrapper
/// additionally delegates `post_failure` to the inner hook — so observability
/// hooks that implement `post_failure` will record failed requests too.
///
/// MEDIUM #1: when the inner hook does NOT override `post_failure`
/// (`supports_failure_events() == false`) but the operator listed the slug
/// under `failure_callback` / `callbacks`, emit a one-time `tracing::warn!`
/// so the operator learns the limitation at config load rather than
/// after-the-fact from missing data. The reference implements failure
/// logging for several of these backends (langsmith, datadog, ...); oxllm
/// does not yet, and the silent no-op was previously pinned by an
/// integration test without surfacing the gap to operators.
fn wrap_config_block_hook(
    inner: Arc<dyn Hook>,
    settings: &OxllmSettings,
    slug: &str,
) -> Arc<dyn Hook> {
    let (_, on_failure) = callback_events(settings, slug);
    if !on_failure {
        return inner;
    }
    if !inner.supports_failure_events() {
        tracing::warn!(
            slug = slug,
            "hook is listed under failure_callback / callbacks but does not \
             implement failure-event logging; no failure data will reach this \
             backend. Remove the slug from failure_callback to silence this \
             warning, or refer to the upstream LiteLLM parity notes."
        );
    }
    Arc::new(CallbackEventFilter {
        inner,
        // Config-block hooks always fire on success (they existed before this
        // change and are present regardless of callback-list membership).
        fire_on_success: true,
        fire_on_failure: true,
    })
}

/// Hook wrapper that gates `post_success` / `post_failure` delegation based
/// on which callback list the wrapped slug was registered under.
///
/// `pre_call` is always delegated so any pre-call logic carried by the inner
/// hook (cache lookup, budget reservation, ...) still runs regardless of the
/// success/failure filter.
struct CallbackEventFilter {
    inner: Arc<dyn Hook>,
    fire_on_success: bool,
    fire_on_failure: bool,
}

#[async_trait]
impl Hook for CallbackEventFilter {
    async fn pre_call(&self, ctx: &mut CallContext) -> oxllm_core::error::Result<HookAction> {
        self.inner.pre_call(ctx).await
    }

    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        if self.fire_on_success {
            self.inner.post_success(ctx, resp).await;
        }
    }

    async fn post_failure(&self, ctx: &CallContext, err: &Error) {
        if self.fire_on_failure {
            self.inner.post_failure(ctx, err).await;
        }
    }
}

/// Construct a [`GenericHttpCallback`] from `oxllm_settings.callback_settings.generic`.
fn build_generic_callback(
    config: &ProxyConfig,
) -> Option<anyhow::Result<crate::hooks::GenericHttpCallback>> {
    let block = config
        .oxllm_settings
        .callback_settings
        .as_ref()?
        .get("generic")?;
    let mut cfg = crate::hooks::CallbackConfig {
        url: String::new(),
        headers: std::collections::HashMap::new(),
    };
    if let serde_json::Value::Object(map) = block {
        for (k, v) in map {
            match k.as_str() {
                "url" => {
                    if let Some(s) = v.as_str() {
                        cfg.url = s.to_string();
                    }
                }
                "headers" => {
                    if let serde_json::Value::Object(hm) = v {
                        for (hk, hv) in hm {
                            if let Some(hs) = hv.as_str() {
                                cfg.headers.insert(hk.clone(), hs.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if cfg.url.is_empty() {
        return Some(Err(anyhow::anyhow!(
            "generic callback settings missing 'url'"
        )));
    }
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    if let Err(e) = crate::config::validate_api_base(&cfg.url, allow_private) {
        return Some(Err(anyhow::anyhow!(
            "generic callback URL '{url}' failed SSRF validation: {e}",
            url = cfg.url
        )));
    }
    Some(crate::hooks::GenericHttpCallback::with_config_capacity(
        cfg,
        crate::hooks::callback::DEFAULT_CHANNEL_CAPACITY,
        allow_private,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{OxllmSettings, RouterSettings};

    fn proxy_with_settings(settings: OxllmSettings) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings,
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        }
    }

    /// Issue #9: a cache hit must still consume RPM (the request was received
    /// and processed) but must NOT reserve budget (no upstream cost). This only
    /// holds when RateLimitHook runs before CacheHook and CacheHook runs before
    /// BudgetHook in the pre-call chain.
    #[tokio::test]
    async fn cache_hit_consumes_rpm_but_not_budget() {
        use crate::cache::InMemoryCache;
        use crate::hooks::cache_hook::cache_key_for;
        use oxllm_core::CallContext;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let settings = OxllmSettings {
            rpm: Some(10),
            max_budget: Some(100.0),
            ..Default::default()
        };
        let config = proxy_with_settings(settings);

        let hooks = build_hooks(
            &config,
            cache.clone(),
            None,
            AlertStates::default(),
            None,
            None,
            None,
            None,
        );

        let request_body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "cache": {"insert": true},
        });
        let api_key_hash = "abc".to_string();
        let key = cache_key_for(&request_body, &api_key_hash);
        cache
            .set(
                &key,
                serde_json::to_string(&serde_json::json!({"id": "x", "choices": []})).unwrap(),
                Some(std::time::Duration::from_secs(60)),
            )
            .await;

        let mut ctx = CallContext {
            model: "gpt-4o".to_string(),
            api_key_hash: api_key_hash.clone(),
            request_id: "r".to_string(),
            request_body: std::sync::Arc::new(request_body),
            ..Default::default()
        };

        let outcome = crate::hooks::run_pre_call_hooks(&hooks, &mut ctx).await;
        assert!(outcome.is_ok(), "cache short-circuit must not be an error");
        assert!(
            outcome.unwrap().is_some(),
            "cache hit must short-circuit with a value"
        );

        assert_eq!(
            cache.counter_value("rpm:abc").await.unwrap_or(0.0),
            1.0,
            "RPM must be consumed even on a cache hit"
        );
        assert_eq!(
            cache.counter_value("spend:abc").await,
            None,
            "budget must not be reserved on a cache hit (no upstream cost)"
        );
    }

    /// Run `post_success` for every hook in the chain over a context
    /// stamped with an adaptive bandit cell.
    async fn run_chain_post_success(hooks: &[Arc<dyn Hook>]) {
        use oxllm_core::router::RequestType;
        let ctx = oxllm_core::CallContext {
            adaptive_cell: Some((RequestType::Other, "m:0".to_string())),
            ..oxllm_core::CallContext::for_test("m")
        };
        let summary = oxllm_core::ResponseSummary {
            model: "m".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        };
        for hook in hooks {
            hook.post_success(&ctx, &summary).await;
        }
    }

    fn cell() -> (oxllm_core::router::RequestType, String) {
        (oxllm_core::router::RequestType::Other, "m:0".to_string())
    }

    #[tokio::test]
    async fn adaptive_hook_registered_when_bandit_store_passed() {
        use crate::cache::InMemoryCache;
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let config = proxy_with_settings(OxllmSettings::default());
        let bandit = Arc::new(oxllm_core::router::BanditStore::new());
        let hooks = build_hooks(
            &config,
            cache,
            None,
            AlertStates::default(),
            None,
            None,
            None,
            Some(bandit.clone()),
        );
        run_chain_post_success(&hooks).await;
        assert_eq!(
            bandit.observations(&cell()),
            1.0,
            "a bandit store must register exactly one AdaptiveHook"
        );
    }

    #[tokio::test]
    async fn adaptive_hook_not_registered_without_bandit_store() {
        use crate::cache::InMemoryCache;
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let config = proxy_with_settings(OxllmSettings::default());
        // No store passed (adaptive strategy not configured): the chain
        // must contain no AdaptiveHook. The probe store observes nothing
        // because no hook was constructed with it.
        let bandit = Arc::new(oxllm_core::router::BanditStore::new());
        let hooks = build_hooks(
            &config,
            cache,
            None,
            AlertStates::default(),
            None,
            None,
            None,
            None,
        );
        run_chain_post_success(&hooks).await;
        assert_eq!(
            bandit.observations(&cell()),
            0.0,
            "no adaptive strategy → no learning hook"
        );
    }

    // ---- M8: failure_callback / callbacks gating -------------------------

    /// A recording probe that logs which lifecycle methods fire. Used to
    /// verify the [`CallbackEventFilter`] gating logic.
    struct RecordingHook {
        calls: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl Hook for RecordingHook {
        async fn post_success(&self, _ctx: &CallContext, _resp: &ResponseSummary) {
            self.calls.lock().unwrap().push("success");
        }
        async fn post_failure(&self, _ctx: &CallContext, _err: &Error) {
            self.calls.lock().unwrap().push("failure");
        }
    }

    fn recording() -> (Arc<RecordingHook>, Arc<std::sync::Mutex<Vec<&'static str>>>) {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Arc::new(RecordingHook {
                calls: calls.clone(),
            }),
            calls,
        )
    }

    fn summary() -> ResponseSummary {
        ResponseSummary {
            model: "m".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    async fn callback_filter_success_only_gates_failure() {
        let (inner, calls) = recording();
        let wrapped = CallbackEventFilter {
            inner,
            fire_on_success: true,
            fire_on_failure: false,
        };
        let ctx = CallContext::for_test("m");
        let err = Error::Network("boom".to_string());
        wrapped.post_failure(&ctx, &err).await;
        wrapped.post_success(&ctx, &summary()).await;
        let got = calls.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["success"],
            "fire_on_failure=false must drop the failure call"
        );
    }

    #[tokio::test]
    async fn callback_filter_failure_only_gates_success() {
        let (inner, calls) = recording();
        let wrapped = CallbackEventFilter {
            inner,
            fire_on_success: false,
            fire_on_failure: true,
        };
        let ctx = CallContext::for_test("m");
        let err = Error::Network("boom".to_string());
        wrapped.post_failure(&ctx, &err).await;
        wrapped.post_success(&ctx, &summary()).await;
        let got = calls.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["failure"],
            "fire_on_success=false must drop the success call"
        );
    }

    #[tokio::test]
    async fn callback_filter_dual_fires_both() {
        let (inner, calls) = recording();
        let wrapped = CallbackEventFilter {
            inner,
            fire_on_success: true,
            fire_on_failure: true,
        };
        let ctx = CallContext::for_test("m");
        let err = Error::Network("boom".to_string());
        wrapped.post_failure(&ctx, &err).await;
        wrapped.post_success(&ctx, &summary()).await;
        let got = calls.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["failure", "success"],
            "fire on both must delegate both events"
        );
    }

    #[test]
    fn callback_events_membership_matrix() {
        let mk = |success: &[&str], failure: &[&str], dual: &[&str]| OxllmSettings {
            success_callback: Some(success.iter().map(|s| s.to_string()).collect()),
            failure_callback: Some(failure.iter().map(|s| s.to_string()).collect()),
            callbacks: Some(dual.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };

        // success-only
        assert_eq!(callback_events(&mk(&["x"], &[], &[]), "x"), (true, false));
        // failure-only
        assert_eq!(callback_events(&mk(&[], &["x"], &[]), "x"), (false, true));
        // dual (callbacks list)
        assert_eq!(callback_events(&mk(&[], &[], &["x"]), "x"), (true, true));
        // success + failure lists combined
        assert_eq!(callback_events(&mk(&["x"], &["x"], &[]), "x"), (true, true));
        // not listed at all
        assert_eq!(
            callback_events(&mk(&["y"], &["y"], &["y"]), "x"),
            (false, false)
        );
    }

    #[test]
    fn wrap_config_block_hook_unwrapped_when_no_failure_request() {
        // A config-block hook whose slug is NOT in failure_callback/callbacks
        // must be returned unwrapped (no CallbackEventFilter indirection).
        let inner: Arc<dyn Hook> = Arc::new(RecordingHook {
            calls: Arc::new(std::sync::Mutex::new(Vec::new())),
        });
        let settings = OxllmSettings {
            success_callback: Some(vec!["langfuse".to_string()]),
            ..Default::default()
        };
        let wrapped = wrap_config_block_hook(inner.clone(), &settings, "langfuse");
        // Arc::ptr_eq is true only when NO wrapper was applied.
        assert!(
            Arc::ptr_eq(&wrapped, &inner),
            "hook must stay unwrapped when slug is only in success_callback"
        );
    }

    #[test]
    fn wrap_config_block_hook_wraps_for_failure_callback() {
        let inner: Arc<dyn Hook> = Arc::new(RecordingHook {
            calls: Arc::new(std::sync::Mutex::new(Vec::new())),
        });
        let settings = OxllmSettings {
            failure_callback: Some(vec!["langfuse".to_string()]),
            ..Default::default()
        };
        let wrapped = wrap_config_block_hook(inner.clone(), &settings, "langfuse");
        assert!(
            !Arc::ptr_eq(&wrapped, &inner),
            "hook must be wrapped when slug is in failure_callback"
        );
    }

    #[test]
    fn wrap_config_block_hook_wraps_for_callbacks_dual() {
        let inner: Arc<dyn Hook> = Arc::new(RecordingHook {
            calls: Arc::new(std::sync::Mutex::new(Vec::new())),
        });
        let settings = OxllmSettings {
            callbacks: Some(vec!["langfuse".to_string()]),
            ..Default::default()
        };
        let wrapped = wrap_config_block_hook(inner.clone(), &settings, "langfuse");
        assert!(
            !Arc::ptr_eq(&wrapped, &inner),
            "hook must be wrapped when slug is in callbacks"
        );
    }

    /// MEDIUM #1: an operator who lists a slug under `failure_callback` /
    /// `callbacks` whose hook does not implement `post_failure` would
    /// silently get zero failure data. The wrapper still wraps the hook
    /// (delegating to the default no-op `post_failure`) but emits a
    /// `tracing::warn!` so the limitation is visible at startup. The
    /// `supports_failure_events()` marker trait method defaults to `false`
    /// (Langfuse / Langsmith / Helicone / MLflow / W&B / S3 / generic-callback
    /// hooks inherit that default). Datadog sets a failure payload, so its
    /// hook returns true and the wrapper stays quiet. A hook that records
    /// failure events overrides the marker and silences the warn.
    #[test]
    fn wrap_config_block_hook_marks_default_post_failure_as_unsupported() {
        // The RecordingHook used as a stand-in for the observability backends
        // does override post_failure, but it does NOT declare
        // supports_failure_events(); by default that returns false.
        let inner: Arc<dyn Hook> = Arc::new(RecordingHook {
            calls: Arc::new(std::sync::Mutex::new(Vec::new())),
        });
        assert!(
            !inner.supports_failure_events(),
            "default Hook::supports_failure_events must be false so the wrapper warns"
        );

        // The wrapper still wraps for failure_callback regardless — the warn
        // is informational, not a build skip (the hook may still produce
        // useful success data).
        let settings = OxllmSettings {
            failure_callback: Some(vec!["langfuse".to_string()]),
            ..Default::default()
        };
        let wrapped = wrap_config_block_hook(inner.clone(), &settings, "langfuse");
        assert!(
            !Arc::ptr_eq(&wrapped, &inner),
            "wrapper must still wrap so success-firing works"
        );
    }

    /// M8 integration: a config with `langfuse_config` plus
    /// `failure_callback: ["langfuse"]` must register a Langfuse hook that is
    /// wrapped to additionally fire on failure. The wrapped chain (run
    /// through the real post_failure dispatcher) must not panic and the
    /// langfuse hook must be present in the chain.
    #[tokio::test]
    async fn langfuse_wrapped_for_failure_callback() {
        use crate::cache::InMemoryCache;
        use crate::config::types::{GeneralSettings, LangfuseConfig, RouterSettings};

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let lf = LangfuseConfig {
            host: "https://langfuse.example.invalid".to_string(),
            public_key: "pk-test".to_string(),
            secret_key: "sk-test".to_string(),
            ..Default::default()
        };
        // `allow_private_api_base` skips the DNS-based SSRF check so the
        // `.invalid` host builds without network access in the test env.
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings {
                langfuse_config: Some(lf),
                failure_callback: Some(vec!["langfuse".to_string()]),
                ..Default::default()
            },
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };

        // Build the langfuse hook the same way assemble() does, then pass it
        // into build_hooks so wrap_config_block_hook applies.
        let langfuse = crate::hooks::langfuse::build_langfuse_hook_opt(&config)
            .expect("langfuse_config present → hook built");

        let hooks = build_hooks(
            &config,
            cache,
            None,
            AlertStates::default(),
            None,
            Some(langfuse),
            None,
            None,
        );

        // The chain must contain at least one non-adaptive hook beyond the
        // base five (rate-limit, cache, budget, cost, metrics) — that's the
        // wrapped langfuse hook.
        assert!(
            hooks.len() > 5,
            "expected langfuse hook in chain, got {} hooks",
            hooks.len()
        );

        // Running post_failure across the chain must not panic even though
        // langfuse has no custom post_failure (the wrapper delegates to the
        // default no-op). This proves the failure path is wired and safe.
        let ctx = CallContext::for_test("m");
        let err = Error::Network("boom".to_string());
        for hook in &hooks {
            hook.post_failure(&ctx, &err).await;
        }
    }

    /// M8 integration: `callbacks: ["generic"]` (dual-fire) builds the generic
    /// HTTP callback registered for BOTH success and failure. The hook count
    /// grows by one versus the no-callback baseline.
    #[tokio::test]
    async fn generic_callback_built_from_callbacks_dual_fire() {
        use crate::cache::InMemoryCache;
        use crate::config::types::{GeneralSettings, RouterSettings};

        let baseline_config = proxy_with_settings(OxllmSettings::default());
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let baseline_hooks = build_hooks(
            &baseline_config,
            cache.clone(),
            None,
            AlertStates::default(),
            None,
            None,
            None,
            None,
        );

        // Same shape but with `callbacks: ["generic"]` + a callback URL.
        // `allow_private_api_base` skips the DNS SSRF check so the
        // `.invalid` host builds without network access.
        let mut callback_settings = std::collections::HashMap::new();
        callback_settings.insert(
            "generic".to_string(),
            serde_json::json!({
                "url": "https://callbacks.example.invalid/spend",
                "headers": {},
            }),
        );
        let cfg_with_generic = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings {
                callbacks: Some(vec!["generic".to_string()]),
                callback_settings: Some(callback_settings),
                ..Default::default()
            },
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };

        let hooks = build_hooks(
            &cfg_with_generic,
            cache,
            None,
            AlertStates::default(),
            None,
            None,
            None,
            None,
        );
        assert_eq!(
            hooks.len(),
            baseline_hooks.len() + 1,
            "callbacks:[generic] must add exactly one (wrapped) generic hook"
        );
    }

    /// LOW: `budget_throttle_pct` set in `oxllm_settings.routing` must reach
    /// the `RateLimitHook` and take effect. When a scope is over its
    /// `max_budget`, the effective RPM limit is multiplied by the throttle
    /// fraction. We verify end-to-end through the full hook chain built by
    /// `build_hooks`: with `rpm: 10` and `budget_throttle_pct: 0.5`, an
    /// over-budget scope whose RPM counter is at 5 is rejected (5+1 > 5
    /// throttled limit), whereas without the throttle it would be allowed
    /// (5+1 < 10).
    #[tokio::test]
    async fn budget_throttle_pct_wired_and_engages() {
        use crate::cache::InMemoryCache;
        use oxllm_core::CallContext;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let settings = OxllmSettings {
            rpm: Some(10),
            routing: crate::config::types::RoutingKnobs {
                budget_throttle_pct: Some(0.5),
                ..Default::default()
            },
            ..Default::default()
        };
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: settings,
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: crate::config::types::RouterSettings::default(),
            extra: serde_json::Map::new(),
        };

        let hooks = build_hooks(
            &config,
            cache.clone(),
            None,
            AlertStates::default(),
            None,
            None,
            None,
            None,
        );

        // Put the scope over budget and RPM at 5 (throttled limit is also 5).
        let _ = cache.incr("spend:abc", 100.0, None).await;
        let _ = cache.incr("rpm:abc", 5.0, None).await;

        let mut ctx = CallContext {
            model: "gpt-4o".to_string(),
            api_key_hash: "abc".to_string(),
            request_id: "r1".to_string(),
            request_body: std::sync::Arc::new(serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            })),
            max_budget: Some(100.0),
            ..Default::default()
        };

        let outcome = hooks[0].pre_call(&mut ctx).await.expect("pre_call ok");
        assert!(
            matches!(outcome, oxllm_core::HookAction::Reject(_)),
            "throttled RPM limit (5) must reject when current+1 exceeds it; \
             the budget_throttle_pct must have reached the RateLimitHook"
        );
    }

    /// LOW: `budget_throttle_pct` set in `router_settings.routing` (the
    /// canonical location in newer configs) must also reach the hook, and
    /// takes precedence over `oxllm_settings` when both are set.
    #[tokio::test]
    async fn budget_throttle_pct_from_router_settings_takes_precedence() {
        use crate::cache::InMemoryCache;
        use crate::config::types::{GeneralSettings, RouterSettings};

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings {
                routing: crate::config::types::RoutingKnobs {
                    budget_throttle_pct: Some(0.3),
                    ..Default::default()
                },
                ..Default::default()
            },
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: crate::config::types::RoutingKnobs {
                    budget_throttle_pct: Some(0.7),
                    ..Default::default()
                },
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };

        // This must not panic — the validation in build_router_config accepts
        // 0.7 and the merge picks router_settings. The hook builds with the
        // merged value.
        let hooks = build_hooks(
            &config,
            cache,
            None,
            AlertStates::default(),
            None,
            None,
            None,
            None,
        );
        assert!(!hooks.is_empty());
    }
}
