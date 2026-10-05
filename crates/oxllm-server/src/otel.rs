//! OpenTelemetry distributed tracing (feature-gated under `otel`).
//!
//! When `general_settings.otel_endpoint` is configured, [`init`] installs an
//! OTLP trace exporter and wires it into the global `tracing` subscriber so
//! spans flow to a collector. When the feature is off or no endpoint is set,
//! tracing works normally (just `tracing-subscriber`).

use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;

/// The W3C `traceparent` header name.
pub const TRACEPARENT_HEADER: &str = "traceparent";

/// Guard that flushes pending spans on shutdown.
///
/// The explicit [`shutdown`](Self::shutdown) method is the preferred exit
/// path: it moves the synchronous `provider.shutdown()` call (which blocks
/// until the batch exporter flushes, bounded by the exporter's 5s timeout)
/// onto the tokio blocking pool so it cannot stall an async worker. Drop is
/// a fallback for the early-return path where the explicit call was not
/// reached; there it falls back to a direct blocking call (acceptable
/// because process exit is already in flight).
pub struct OtelGuard {
    provider: Option<SdkTracerProvider>,
}

impl OtelGuard {
    /// Flush pending spans and shut down the exporter, bounded by the
    /// exporter's 5s timeout. The synchronous `provider.shutdown()` runs
    /// on a tokio blocking-pool thread via `spawn_blocking` so a slow
    /// flush cannot stall the async worker that drives `run()`'s shutdown
    /// sequence (e.g. the concurrent spend-writer flush).
    ///
    /// Consumes `self` so the `Drop` fallback does not double-flush.
    pub async fn shutdown(mut self) {
        let Some(provider) = self.provider.take() else {
            return;
        };
        match tokio::task::spawn_blocking(move || provider.shutdown()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(error = %e, "otel tracer shutdown failed"),
            Err(e) => tracing::warn!(error = %e, "otel tracer shutdown task join failed"),
        }
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        // Fallback: explicit `shutdown()` was not called (early-return
        // path), or was called and consumed the inner provider (in which
        // case `take()` left `None` and this is a true no-op).
        // `spawn_blocking` cannot be awaited from `Drop` (which is
        // synchronous), so when a provider is still present this falls
        // back to a direct blocking call. Process exit is already in
        // flight on that path so the brief stall is acceptable; the
        // explicit `shutdown()` method is the non-blocking path used by
        // the normal `run()` exit.
        if let Some(provider) = self.provider.as_ref() {
            if let Err(e) = provider.shutdown() {
                tracing::warn!(error = %e, "otel tracer shutdown failed");
            }
        }
    }
}

/// Outcome of [`init`]: distinguishes "otel disabled" from "an embedder's
/// pre-installed global tracing subscriber swallowed the OTel layer", so the
/// caller can surface the latter instead of silently running without traces.
pub enum InitOutcome {
    /// No `otel_endpoint` configured; tracing runs on the plain fmt
    /// subscriber.
    Disabled,
    /// The OTel layer was attached to the freshly installed global
    /// subscriber. The guard flushes spans on shutdown.
    Attached(OtelGuard),
    /// A global tracing subscriber was already installed before `init` ran
    /// (e.g. by an embedder), so the OTel layer could never receive spans.
    /// The provider we built has been shut down; spans will NOT be exported.
    SubscriberConflict,
}

impl std::fmt::Debug for InitOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitOutcome::Disabled => f.write_str("Disabled"),
            InitOutcome::Attached(_) => f.write_str("Attached(OtelGuard)"),
            InitOutcome::SubscriberConflict => f.write_str("SubscriberConflict"),
        }
    }
}

/// Initialize OpenTelemetry tracing. Returns
/// [`InitOutcome::Disabled`] when `endpoint` is `None` (otel disabled) and
/// [`InitOutcome::SubscriberConflict`] when a global tracing subscriber was
/// already installed before this call.
///
/// Only installs the OpenTelemetry tracer provider + returns the layer to
/// attach; the caller composes it with the fmt subscriber and installs a
/// single global subscriber (the W3C `set_global_default` API can only be
/// called once per process, so each layer must not try to install its own
/// global).
pub fn init(
    endpoint: Option<&str>,
    service_name: Option<&str>,
    log_format: Option<&str>,
) -> anyhow::Result<InitOutcome> {
    let Some(endpoint) = endpoint else {
        return Ok(InitOutcome::Disabled);
    };
    let service_name_owned = service_name.unwrap_or("oxllm").to_string();

    // The exporter is hardcoded to OTLP/gRPC below; if the operator pointed
    // `otel_endpoint` at the OTLP/HTTP standard port (4318, vs 4317 for
    // gRPC) the tonic dial will succeed lazily but no span will ever
    // export — a silent trace-loss footgun. Warn loudly at startup so the
    // misconfiguration surfaces immediately.
    if looks_like_http_otlp_endpoint(endpoint) {
        tracing::warn!(
            endpoint = %endpoint,
            "OTLP endpoint appears to target OTLP/HTTP (port 4318). \
             oxllm currently exports only via OTLP/gRPC (port 4317); \
             traces will NOT be exported. Point otel_endpoint at the \
             collector's gRPC port (typically 4317)."
        );
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_protocol(Protocol::Grpc)
        .with_timeout(Duration::from_secs(5))
        .build()?;

    let resource = Resource::builder()
        .with_service_name(service_name_owned.clone())
        .with_attributes([KeyValue::new("service.version", env!("CARGO_PKG_VERSION"))])
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    // Compose a single global subscriber from the registry + fmt + OTel layer.
    // `set_global_default` can only succeed once per process, so OTel must
    // own subscriber installation rather than fighting a fmt subscriber that
    // was installed earlier by `main`.
    use tracing_subscriber::layer::SubscriberExt;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(false);
    // The JSON and text fmt layers have different concrete types, and the
    // OTel layer is parameterized by the subscriber type it is attached to,
    // so the subscriber must be composed in two separate code paths rather
    // than conditionally swapping the layer mid-construction.
    let installed = if log_format == Some("json") {
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer.json())
                .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("oxllm"))),
        )
    } else {
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("oxllm"))),
        )
    };
    if installed.is_err() {
        // A global subscriber is already installed (e.g. by an embedder), so
        // our OTel layer can never receive spans. Shut down the provider we
        // built (flushing + releasing its batch exporter) and return `None`
        // so the caller falls back to the plain fmt subscriber rather than
        // holding a live exporter that can never emit.
        //
        // The global tracer provider is deliberately NOT registered in this
        // branch: `set_tracer_provider` can only succeed once per process,
        // and the shut-down provider is now useless. Registering it eagerly
        // before the subscriber attach would leave the global slot pointing
        // at a dead `SdkTracerProvider` that a later init could not replace.
        tracing::warn!(
            "a global tracing subscriber is already installed; \
             otel layer not attached, spans will not be exported"
        );
        if let Err(e) = provider.shutdown() {
            tracing::warn!(error = %e, "otel tracer shutdown failed");
        }
        return Ok(InitOutcome::SubscriberConflict);
    }

    // Subscriber attach succeeded: now register the provider as the global
    // tracer so `tracing_opentelemetry`'s layer (which reads via the global
    // tracer provider) and any out-of-band OTel consumers emit through the
    // same exporter. Done after the success branch so a failed attach never
    // leaves a dead provider in the one-shot global slot.
    global::set_tracer_provider(provider.clone());

    tracing::info!(endpoint = %endpoint, service = %service_name_owned, "opentelemetry tracing initialized");
    Ok(InitOutcome::Attached(OtelGuard {
        provider: Some(provider),
    }))
}

/// U49: stamp the OpenTelemetry semantic-convention error attributes
/// `error.type` and `error.message` on the **current** span and mark it
/// errored (`Status::Error`).
///
/// Call this on every request failure path (see `run_post_failure_hooks`) so
/// collector dashboards can slice by error type and surface the failure
/// message on the request span. The OTel layer bridges the active `tracing`
/// span to an OTel span, so this must be called while the request span is
/// entered (e.g. inside the tower-http `TraceLayer` that wraps the handler —
/// which it always is on the request path).
///
/// Safe to call when OTel is not initialised: with no tracer provider the
/// current span is the no-op span whose `set_attribute` / `set_status` are
/// cheap no-ops (the `SpanRef` inner is `None`). This function additionally
/// short-circuits before any `to_string()` / `KeyValue` allocation when the
/// current span is the no-op span (the common case when no OTel collector is
/// configured) so the failure hot path pays no allocation cost when OTel is
/// off.
pub fn record_span_error(error_type: &str, message: &str) {
    use opentelemetry::trace::{Status, TraceContextExt};
    let cx = opentelemetry::Context::current();
    let span = cx.span();
    // The no-op span (no OTel provider, or no active OTel span) has an
    // invalid trace id; bail out before allocating `error_type.to_string()`
    // / `message.to_string()` and the two `KeyValue`s. The `SpanRef` inner
    // is `None` in this case so the `set_attribute` / `set_status` calls
    // below would also be no-ops — this just makes the no-op cheap rather
    // than allocating-then-discarding.
    if !span.span_context().is_valid() {
        return;
    }
    span.set_attribute(opentelemetry::KeyValue::new(
        "error.type",
        error_type.to_string(),
    ));
    span.set_attribute(opentelemetry::KeyValue::new(
        "error.message",
        message.to_string(),
    ));
    span.set_status(Status::Error {
        description: message.to_string().into(),
    });
}

/// Whether `endpoint` looks like an OTLP/HTTP receiver rather than the
/// OTLP/gRPC receiver this exporter actually targets.
///
/// The OTLP/HTTP standard port is `4318`; OTLP/gRPC is `4317`. If the
/// endpoint's port component is `4318`, the operator almost certainly
/// configured an HTTP receiver and the tonic (gRPC) exporter will silently
/// produce zero spans. The check is intentionally simple (substring match
/// on `:4318`) — false positives require a literally `:4318`-suffixed host
/// on a different port, which is pathological for real OTLP endpoints.
fn looks_like_http_otlp_endpoint(endpoint: &str) -> bool {
    endpoint.contains(":4318")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without an OTel provider installed, the current span is the no-op
    /// span; `record_span_error` must be a safe no-op (no panic).
    #[test]
    fn record_span_error_is_safe_without_provider() {
        record_span_error("rate_limited", "rpm limit exceeded");
        record_span_error("upstream_error", "upstream returned 503");
    }

    /// LOW (C16): the OTLP/HTTP endpoint detector should flag the standard
    /// OTLP/HTTP port (4318) across the URI shapes tonic accepts, while
    /// leaving the gRPC standard port (4317) and non-port-bearing URIs
    /// alone. A false negative is a missed warning (operator-visible only
    /// via absent traces); a false positive on a pathological hostname
    /// like `host:4318` (without a scheme) is acceptable.
    #[test]
    fn looks_like_http_otlp_endpoint_detects_port_4318() {
        // Standard OTLP/HTTP forms.
        assert!(looks_like_http_otlp_endpoint("http://collector:4318"));
        assert!(looks_like_http_otlp_endpoint("https://collector:4318"));
        assert!(looks_like_http_otlp_endpoint("collector:4318"));
        assert!(looks_like_http_otlp_endpoint(
            "http://collector:4318/v1/traces"
        ));

        // Standard OTLP/gRPC forms — must NOT match.
        assert!(!looks_like_http_otlp_endpoint("http://collector:4317"));
        assert!(!looks_like_http_otlp_endpoint("collector:4317"));
        assert!(!looks_like_http_otlp_endpoint("https://collector:4317"));

        // Pathological port-less forms — must NOT match (no silent warning
        // when the endpoint is fine).
        assert!(!looks_like_http_otlp_endpoint("collector"));
        assert!(!looks_like_http_otlp_endpoint("http://collector"));
    }
}
