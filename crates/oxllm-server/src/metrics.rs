//! Prometheus metrics.
//!
//! Registers counters and histograms and exposes a `/metrics` endpoint.

use std::collections::HashSet;
use std::sync::OnceLock;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use parking_lot::RwLock;

/// Metric name constants.
pub const REQUESTS_TOTAL: &str = "oxllm_requests_total";
pub const REQUEST_DURATION: &str = "oxllm_request_duration_seconds";
pub const TOKENS_TOTAL: &str = "oxllm_tokens_total";
pub const COST_TOTAL: &str = "oxllm_cost_total";
pub const ERRORS_TOTAL: &str = "oxllm_errors_total";
/// Additional same-deployment chat attempts started by the retry loop. Does
/// not count the first attempt, deployment rotation, or fallback-chain hops.
/// Labels identify the attempted routing group and provider, which may be a
/// fallback/default target rather than the client's original model name.
pub const UPSTREAM_RETRIES_TOTAL: &str = "oxllm_upstream_retries_total";
/// Optional shadow calls skipped because all shadow-work slots are occupied.
pub const SILENT_MODEL_SKIPPED_TOTAL: &str = "oxllm_silent_model_skipped_total";
/// Spend entries dropped before reaching the DB. Label `reason` is one of
/// `full` (channel at capacity), `closed` (channel shut down) or `batch_error`
/// (a batch-insert failure that is not safely retriable per-row: a
/// non-connection DB error, a pool-down error, or an ambiguous commit-time
/// failure).
pub const SPEND_ENTRIES_DROPPED_TOTAL: &str = "oxllm_spend_entries_dropped_total";
/// Cost samples that were `NaN` / negative / non-finite and so could not be
/// recorded into [`COST_TOTAL`] without silently saturating to 0.
pub const COST_INVALID_TOTAL: &str = "oxllm_cost_invalid_total";
/// Telemetry events dropped before reaching a backend. Label `backend` is the
/// exporter slug (`langfuse`, `langsmith`, `helicone`, `datadog`, `mlflow`,
/// `wandb`, `callback`, `s3_log`, `http_telemetry`); `reason` is `full`
/// (queue at capacity) or `post_failed` (the POST itself failed). Lets
/// operators alert on silent telemetry loss from any exporter.
pub const TELEMETRY_DROPPED_TOTAL: &str = "oxllm_telemetry_dropped_total";
/// Startup gauge set to `1` when the process boots with SQLite persistence
/// co-configured with Redis — an almost-certainly-multi-instance shape that
/// causes silent spend-log loss (the batched spend writer swallows
/// `SQLITE_BUSY`). A log line alone is easy to miss in production aggregation;
/// this gauge lets a dashboard/alert fire on the misconfiguration for as long
/// as the process runs. Label `level` is `error` (SQLite + Redis) or `warn`
/// (single-instance SQLite).
pub const SQLITE_MULTI_INSTANCE_GUARD: &str = "oxllm_sqlite_multi_instance_guard";

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Maximum bytes in a client/provider-controlled metric label. This bounds
/// individual values; [`LabelRegistry`] separately bounds their cardinality.
const MAX_LABEL_LEN: usize = 64;
const MAX_MODEL_LABELS: usize = 1024;
const MAX_PROVIDER_LABELS: usize = 64;

static MODEL_LABELS: OnceLock<LabelRegistry> = OnceLock::new();
static PROVIDER_LABELS: OnceLock<LabelRegistry> = OnceLock::new();

/// Admit a bounded set of labels for the lifetime of the process. Entries
/// must never be evicted: evicting a local cache does not remove the series
/// already registered in Prometheus, so churn would still grow it forever.
/// Existing labels keep their identity after saturation; only new values
/// are aggregated into `other`.
struct LabelRegistry {
    labels: RwLock<HashSet<String>>,
    capacity: usize,
}

impl LabelRegistry {
    fn new(capacity: usize) -> Self {
        Self {
            labels: RwLock::new(HashSet::new()),
            capacity,
        }
    }

    fn label(&self, value: &str) -> String {
        let value = bounded_label(value);
        // Keep sentinel meanings stable even when the registry is full.
        if matches!(value.as_str(), "" | "unknown" | "other") {
            return value;
        }
        {
            let labels = self.labels.read();
            if labels.contains(&value) {
                return value;
            }
            // The set never shrinks, so full registries can reject unknown
            // values without taking the exclusive lock on every request.
            if labels.len() >= self.capacity {
                return "other".to_string();
            }
        }
        let mut labels = self.labels.write();
        // Another thread may have admitted this value before the write lock.
        if labels.contains(&value) {
            return value;
        }
        if labels.len() < self.capacity {
            labels.insert(value.clone());
            value
        } else {
            "other".to_string()
        }
    }
}

/// Bound model labels shared by request/routing/cache helpers, including raw requested
/// names accepted by wildcard/default-fallback routing. Up to 1024 distinct
/// length-capped values retain their identity per process lifetime, plus the
/// fixed empty/unknown/other sentinels.
pub(crate) fn model_label(value: &str) -> String {
    MODEL_LABELS
        .get_or_init(|| LabelRegistry::new(MAX_MODEL_LABELS))
        .label(value)
}

fn provider_label(value: &str) -> String {
    PROVIDER_LABELS
        .get_or_init(|| LabelRegistry::new(MAX_PROVIDER_LABELS))
        .label(value)
}

/// Optionally seed configured model groups/aliases/upstream names before
/// accepting traffic so they are admitted ahead of client-controlled names.
/// Reloads share the same lifetime cap; this never resets the registry.
pub fn register_model_labels<'a>(models: impl IntoIterator<Item = &'a str>) {
    for model in models {
        let _ = model_label(model);
    }
}

/// Test-only membership probe: whether `value` is already admitted to the
/// process-wide model-label registry. Unlike [`model_label`], this never
/// admits the value, so a state-build test can assert configured names were
/// *pre-registered* rather than lazily admitted on first lookup.
#[cfg(test)]
pub(crate) fn model_label_registered(value: &str) -> bool {
    let value = bounded_label(value);
    MODEL_LABELS
        .get_or_init(|| LabelRegistry::new(MAX_MODEL_LABELS))
        .labels
        .read()
        .contains(&value)
}

/// Truncate an untrusted label value to [`MAX_LABEL_LEN`] on a UTF-8 char
/// boundary so multi-byte sequences are never split.
pub(crate) fn bounded_label(value: &str) -> String {
    if value.len() <= MAX_LABEL_LEN {
        return value.to_string();
    }
    let mut end = MAX_LABEL_LEN;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

/// Install the Prometheus exporter globally. Safe to call multiple times
/// only the first call takes effect.
pub fn install() -> &'static PrometheusHandle {
    HANDLE.get_or_init(|| {
        let handle = PrometheusBuilder::new()
            .install_recorder()
            .unwrap_or_else(|e| {
                tracing::error!(error = %e, "failed to install prometheus recorder");
                // Fall back to manually registering a fresh recorder with the
                // `metrics` facade so counters still land in the recorder this
                // handle renders — otherwise /metrics would show empty output
                // forever while recorded values go to an orphaned recorder.
                let recorder = PrometheusBuilder::new().build_recorder();
                let handle = recorder.handle();
                if let Err(set_err) = metrics::set_global_recorder(recorder) {
                    tracing::error!(
                        error = %set_err,
                        "failed to set global metrics recorder; /metrics will render empty"
                    );
                }
                handle
            });
        describe_metrics();
        handle
    })
}

/// Register metadata for these helpers and the routing/cache hook counters.
fn describe_metrics() {
    use metrics::Unit;
    metrics::describe_counter!(
        REQUESTS_TOTAL,
        Unit::Count,
        "Completed LLM pipeline requests, including response-cache hits"
    );
    metrics::describe_histogram!(
        REQUEST_DURATION,
        Unit::Seconds,
        "Request latency (time-to-first-byte for streaming responses)"
    );
    metrics::describe_counter!(
        TOKENS_TOTAL,
        Unit::Count,
        "Upstream tokens; excludes response-cache hits"
    );
    metrics::describe_counter!(
        COST_TOTAL,
        Unit::Count,
        "Upstream cost in micro-dollars; excludes response-cache hits"
    );
    metrics::describe_counter!(ERRORS_TOTAL, Unit::Count, "Terminal LLM pipeline errors");
    metrics::describe_counter!(
        crate::hooks::metrics::ROUTER_DECISIONS_TOTAL,
        Unit::Count,
        "Completed routed requests by final strategy and requested model; excludes cache hits and pre-routing rejections"
    );
    metrics::describe_counter!(
        crate::hooks::cache_hook::PROMPT_CACHE_HITS_TOTAL,
        Unit::Count,
        "Response-cache hits (exact or semantic); excludes provider-side prompt caching"
    );
    metrics::describe_counter!(
        UPSTREAM_RETRIES_TOTAL,
        Unit::Count,
        "Additional same-deployment chat attempts started; excludes deployment rotation and fallback hops"
    );
    metrics::describe_counter!(
        SILENT_MODEL_SKIPPED_TOTAL,
        Unit::Count,
        "Shadow calls skipped before spawning because the concurrency limit was reached"
    );
    metrics::describe_counter!(
        SPEND_ENTRIES_DROPPED_TOTAL,
        Unit::Count,
        "Spend entries dropped before DB write"
    );
    metrics::describe_counter!(
        COST_INVALID_TOTAL,
        Unit::Count,
        "Cost samples that were NaN/negative/non-finite"
    );
    metrics::describe_counter!(
        TELEMETRY_DROPPED_TOTAL,
        Unit::Count,
        "Telemetry events dropped before reaching a backend"
    );
    metrics::describe_gauge!(
        SQLITE_MULTI_INSTANCE_GUARD,
        Unit::Count,
        "1 when the process started with a SQLite multi-instance misconfiguration"
    );
}

/// Get the Prometheus handle, rendering the current metrics in text format.
pub fn render() -> String {
    install().render()
}

/// Record a terminal LLM pipeline outcome (including cache hits), once per
/// request. Auth/parse rejections before the pipeline and raw passthrough
/// endpoints are outside this counter's scope; HTTP latency is separate.
pub fn record_request(model: &str, provider: &str, status: &str) {
    let model = model_label(model);
    let provider = provider_label(provider);
    let counter = metrics::counter!(REQUESTS_TOTAL, "model" => model, "provider" => provider, "status" => status.to_string());
    counter.increment(1);
}

/// Record request duration.
pub fn record_duration(model: &str, provider: &str, secs: f64) {
    let model = model_label(model);
    let provider = provider_label(provider);
    metrics::histogram!(REQUEST_DURATION, "model" => model, "provider" => provider).record(secs);
}

/// Record the wall-clock duration of a completed HTTP request, labelled by
/// HTTP status family (`"2xx"`, `"4xx"`, `"5xx"`, …). This is the dedicated
/// entry point for the global request-timing middleware; it avoids overloading
/// the `model` / `provider` label slots of [`record_duration`] with a
/// semantically unrelated value. The value captures time-to-first-byte for
/// streaming (SSE) responses — see `crate::routes::middleware`.
pub fn record_request_duration(status_family: &str, secs: f64) {
    metrics::histogram!(REQUEST_DURATION, "status_family" => status_family.to_string())
        .record(secs);
}

/// Record token usage.
pub fn record_tokens(model: &str, token_type: &str, count: u64) {
    let model = model_label(model);
    let counter =
        metrics::counter!(TOKENS_TOTAL, "model" => model, "type" => token_type.to_string());
    counter.increment(count);
}

/// Record cost. Stored as micro-dollars (cost * 1_000_000) since counters
/// are u64.
///
/// A `NaN`, negative or infinite cost (refund, upstream bug, `0.0 * inf` in
/// the f64 price math) is rejected and counted under [`COST_INVALID_TOTAL`]
/// instead: `f64 as u64` saturates, so `NaN`/`NEG_INFINITY` would record
/// `0` and `INFINITY` would record `u64::MAX`, making the Prometheus cost
/// metric diverge from the `spend_logs` table. A finite cost whose
/// micro-dollar product overflows `u64` (cost > ~1.8e13) is likewise
/// rejected — saturating to `u64::MAX` would record a single giant spike
/// rather than flagging the bad sample. Such values are implausible for a
/// per-request cost but defended against for the same reason.
pub fn record_cost(model: &str, cost: f64) {
    if !cost.is_finite() || cost < 0.0 || cost * 1_000_000.0 > u64::MAX as f64 {
        tracing::warn!(
            cost,
            "ignoring non-finite, negative or overflowing cost metric"
        );
        metrics::counter!(COST_INVALID_TOTAL, "model" => model_label(model)).increment(1);
        return;
    }
    let micros = (cost * 1_000_000.0) as u64;
    let counter = metrics::counter!(COST_TOTAL, "model" => model_label(model));
    counter.increment(micros);
}

/// Record an error.
pub fn record_error(error_type: &str) {
    let counter = metrics::counter!(ERRORS_TOTAL, "type" => error_type.to_string());
    counter.increment(1);
}

/// Record that `count` spend entries were dropped before reaching the DB.
/// `reason` is one of `full`, `closed` or `batch_error` (see
/// [`SPEND_ENTRIES_DROPPED_TOTAL`]). Lets operators alert on silent spend
/// telemetry loss.
pub fn record_spend_dropped(reason: &str, count: u64) {
    metrics::counter!(SPEND_ENTRIES_DROPPED_TOTAL, "reason" => reason.to_string()).increment(count);
}

/// Record a routing decision under [`crate::hooks::metrics::ROUTER_DECISIONS_TOTAL`],
/// labelled by effective strategy and the cardinality-bounded requested model.
pub fn record_router_decision(strategy: &str, model: &str) {
    let model = model_label(model);
    metrics::counter!(
        crate::hooks::metrics::ROUTER_DECISIONS_TOTAL,
        "strategy" => strategy.to_string(),
        "model" => model
    )
    .increment(1);
}

/// Call immediately before starting each additional same-deployment attempt
/// in the chat retry loop, including a thinking-signature repair retry. This
/// records attempts actually started, not configured retry budgets or sleeps.
/// `model` is the attempted routing group for this leg.
pub fn record_upstream_retry(model: &str, provider: &str) {
    metrics::counter!(
        UPSTREAM_RETRIES_TOTAL,
        "model" => model_label(model),
        "provider" => provider_label(provider)
    )
    .increment(1);
}

pub(crate) fn record_silent_model_skipped() {
    metrics::counter!(SILENT_MODEL_SKIPPED_TOTAL, "reason" => "concurrency_limit").increment(1);
}

/// Record that `count` telemetry events were dropped before reaching a
/// backend (queue full or POST failure). Mirrors
/// [`record_spend_dropped`] for the telemetry/exporter family so silent
/// exporter loss is observable from the same dashboards.
pub fn record_telemetry_dropped(backend: &str, reason: &str, count: u64) {
    metrics::counter!(
        TELEMETRY_DROPPED_TOTAL,
        "backend" => backend.to_string(),
        "reason" => reason.to_string()
    )
    .increment(count);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_registry_bounds_churn_without_forgetting_existing_models() {
        // A private registry avoids exhausting the process-wide registry used
        // by other tests. Admission is the same for all model metric families.
        let registry = LabelRegistry::new(2);
        assert_eq!(registry.label("gpt-4o"), "gpt-4o");
        assert_eq!(registry.label("claude"), "claude");
        for i in 0..10_000 {
            assert_eq!(registry.label(&format!("wildcard-client-{i}")), "other");
        }
        assert_eq!(registry.labels.read().len(), 2);
        assert_eq!(registry.label("gpt-4o"), "gpt-4o");
        assert_eq!(registry.label("unknown"), "unknown");
        assert_eq!(registry.label(""), "");
    }

    #[test]
    fn label_registry_admission_is_bounded_under_concurrency() {
        let registry = LabelRegistry::new(4);
        std::thread::scope(|scope| {
            for i in 0..32 {
                let registry = &registry;
                scope.spawn(move || {
                    for j in 0..16 {
                        registry.label(&format!("model-{i}-{j}"));
                    }
                });
            }
        });
        assert_eq!(registry.labels.read().len(), 4);
    }

    /// `register_model_labels` admits operator-configured names into the
    /// process-wide registry (asserted by membership, not by a lookup that
    /// would admit lazily), and the bounded-label fallback applies once a
    /// registry is full while the length cap applies before admission.
    #[test]
    #[serial_test::serial]
    fn register_model_labels_admits_configured_names_and_bounds_unknown_values() {
        let _ = install();
        let configured = "register-labels-probe-configured";
        assert!(
            !model_label_registered(configured),
            "precondition: the probe name must not already be registered"
        );
        register_model_labels([configured]);
        assert!(
            model_label_registered(configured),
            "register_model_labels must admit the configured name"
        );
        assert_eq!(model_label(configured), configured);

        // Saturation is exercised on a private registry: the global 1024-label
        // budget other tests depend on can never shrink or reset, so filling it
        // here would corrupt unrelated metric assertions.
        let registry = LabelRegistry::new(2);
        assert_eq!(registry.label("first-configured"), "first-configured");
        assert_eq!(registry.label("second-configured"), "second-configured");
        assert_eq!(registry.label("unregistered-unknown"), "other");
        assert_eq!(
            registry.label("first-configured"),
            "first-configured",
            "existing labels keep their identity after saturation"
        );
        assert_eq!(registry.label("unknown"), "unknown");
        assert_eq!(registry.label(""), "");

        // The length cap applies before admission and before the fallback.
        let long = "l".repeat(MAX_LABEL_LEN * 3 + 8);
        assert_eq!(bounded_label(&long), "l".repeat(MAX_LABEL_LEN));
        assert_eq!(model_label(&long), "l".repeat(MAX_LABEL_LEN));
    }

    #[test]
    fn render_includes_type_and_help_for_described_metrics() {
        let _ = install();
        record_request("help-type-probe", "openai", "success");
        let output = render();
        assert!(
            output.contains("# TYPE oxllm_requests_total counter"),
            "missing # TYPE line, got:\n{output}"
        );
        assert!(
            output.contains("# HELP oxllm_requests_total "),
            "missing # HELP line, got:\n{output}"
        );
    }

    #[test]
    fn record_does_not_panic() {
        let _ = install();
        record_request("gpt-4o", "openai", "success");
        record_tokens("gpt-4o", "prompt", 100);
        record_cost("gpt-4o", 0.001);
        record_error("timeout");
        record_duration("gpt-4o", "openai", 0.5);
        record_request_duration("2xx", 0.5);
    }

    /// MEDIUM: provider-controlled `model` strings are truncated so a single
    /// pathological value cannot blow up per-series memory.
    #[test]
    #[serial_test::serial]
    fn record_request_truncates_long_provider_model_label() {
        let _ = install();
        let long_model: String = "z".repeat(MAX_LABEL_LEN * 3 + 8);
        record_request(&long_model, "openai", "success");
        let output = render();
        // The full (untruncated) value must never appear as a label...
        assert!(
            !output.contains(&format!("model=\"{long_model}\"")),
            "untruncated provider model leaked into a metric label"
        );
        // ...and the truncated form (exactly MAX_LABEL_LEN chars) must.
        assert!(
            output.contains(&format!("model=\"{}\"", "z".repeat(MAX_LABEL_LEN))),
            "expected a model label truncated to {MAX_LABEL_LEN} chars in:\n{output}"
        );
    }

    /// MEDIUM: NaN / negative / infinite cost must not panic, must not move
    /// the cost counter, and must bump the invalid counter once per call.
    #[test]
    #[serial_test::serial]
    fn record_cost_rejects_non_finite_and_negative() {
        let _ = install();
        let model = "cost-invalid-probe";
        // Establish a baseline valid cost (micro-dollars).
        record_cost(model, 0.5);
        let baseline = counter_value(COST_TOTAL, model).unwrap_or(0);

        record_cost(model, f64::NAN);
        record_cost(model, -1.0);
        record_cost(model, f64::INFINITY);
        record_cost(model, f64::NEG_INFINITY);

        // Cost counter unchanged for this model.
        assert_eq!(
            counter_value(COST_TOTAL, model).unwrap_or(baseline),
            baseline,
            "non-finite/negative cost must not increment the cost counter"
        );
        // Exactly four invalid samples for this model.
        assert_eq!(
            counter_value(COST_INVALID_TOTAL, model),
            Some(4),
            "expected 4 invalid cost samples for {model}"
        );
    }

    /// LOW (C16): a finite cost whose micro-dollar product overflows `u64`
    /// (cost > ~1.8e13) must be rejected under `COST_INVALID_TOTAL` rather
    /// than saturating to `u64::MAX` — saturating would record a single
    /// giant spike instead of flagging the bad sample. Such values are
    /// implausible for a per-request cost but defended against here for
    /// the same reason NaN / negative / infinite are.
    #[test]
    #[serial_test::serial]
    fn record_cost_rejects_overflowing_finite_cost() {
        let _ = install();
        let model = "cost-overflow-probe";
        record_cost(model, 0.5);
        let baseline = counter_value(COST_TOTAL, model).unwrap_or(0);
        let invalid_before = counter_value(COST_INVALID_TOTAL, model).unwrap_or(0);

        record_cost(model, 1.0e20);
        record_cost(model, (u64::MAX as f64) / 1_000_000.0 * 2.0);

        assert_eq!(
            counter_value(COST_TOTAL, model).unwrap_or(baseline),
            baseline,
            "overflowing finite cost must not increment the cost counter"
        );
        assert_eq!(
            counter_value(COST_INVALID_TOTAL, model).unwrap_or(0) - invalid_before,
            2,
            "expected 2 invalid cost samples for {model}"
        );
    }

    /// LOW (C16): `record_duration` routes `model` / `provider` through
    /// `bounded_label` so a future caller passing a provider-controlled
    /// model string cannot blow up per-series registry memory. Mirrors
    /// `record_request_truncates_long_provider_model_label`.
    #[test]
    #[serial_test::serial]
    fn record_duration_truncates_long_provider_model_label() {
        let _ = install();
        let long_model: String = "d".repeat(MAX_LABEL_LEN * 3 + 8);
        let long_provider: String = "p".repeat(MAX_LABEL_LEN * 3 + 8);
        record_duration(&long_model, &long_provider, 0.5);
        let output = render();
        assert!(
            !output.contains(&format!("model=\"{long_model}\"")),
            "untruncated model leaked into request_duration series"
        );
        assert!(
            !output.contains(&format!("provider=\"{long_provider}\"")),
            "untruncated provider leaked into request_duration series"
        );
        assert!(
            output.contains(&format!("model=\"{}\"", "d".repeat(MAX_LABEL_LEN))),
            "expected a model label truncated to {MAX_LABEL_LEN} chars in:\n{output}"
        );
        assert!(
            output.contains(&format!("provider=\"{}\"", "p".repeat(MAX_LABEL_LEN))),
            "expected a provider label truncated to {MAX_LABEL_LEN} chars in:\n{output}"
        );
    }

    /// LOW (C25): `record_request_duration` emits the histogram with a single
    /// `status_family` label rather than overloading the `model` / `provider`
    /// label slots of `record_duration`. The status family value must appear
    /// verbatim and no spurious `model=""` / `provider="…"` label should be
    /// emitted for the request-timing series.
    #[test]
    #[serial_test::serial]
    fn record_request_duration_uses_status_family_label() {
        let _ = install();
        let family = "2xx";
        record_request_duration(family, 0.25);
        let output = render();
        assert!(
            output.contains(&format!("status_family=\"{family}\"")),
            "expected status_family label in request_duration series:\n{output}"
        );
    }

    /// Parse a `oxllm_<name>{... "model"="<model>" ...}` counter value out of
    /// the rendered exposition text, keyed by model label.
    fn counter_value(name: &str, model: &str) -> Option<u64> {
        let output = render();
        for line in output.lines() {
            if !line.starts_with(name) {
                continue;
            }
            if line.contains(&format!("model=\"{model}\"")) {
                return line.rsplit(' ').next().and_then(|s| s.parse::<u64>().ok());
            }
        }
        None
    }
}
