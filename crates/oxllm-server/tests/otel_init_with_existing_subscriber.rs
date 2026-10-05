//! When a global `tracing` subscriber is already installed before
//! [`oxllm_server::otel::init`] runs, `init` must NOT register the OTel
//! tracer provider in the one-shot global slot before realising its layer
//! cannot attach. The failure path returns
//! [`otel::InitOutcome::SubscriberConflict`] (distinct from `Disabled`) and
//! shuts down the provider it built, leaving the global slot untouched so
//! the embedder can detect that its subscriber swallowed the OTel layer.
//!
//! This is an integration test (separate binary) so the process-global
//! subscriber slot is isolated from the crate's unit tests.

#![cfg(feature = "otel")]

use oxllm_server::otel;
use tracing_subscriber::layer::SubscriberExt;

#[tokio::test]
async fn init_returns_subscriber_conflict_when_global_subscriber_already_installed() {
    // Pre-install a global subscriber, simulating an embedder that owns
    // tracing before oxllm starts. `set_global_default` succeeds exactly
    // once per process; this binary only runs this one test.
    let installed = tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_subscriber::EnvFilter::new("info")),
    );
    assert!(
        installed.is_ok(),
        "test precondition: pre-install the global subscriber"
    );

    // `init` must observe the occupied slot, fall through to the conflict
    // branch, shut down the provider it built, and return
    // `SubscriberConflict` — without panicking and without leaving a dead
    // provider in the global tracer slot.
    let outcome = otel::init(Some("http://127.0.0.1:4317"), Some("oxllm-test"), None)
        .expect("init must not error on the already-installed-subscriber path");
    assert!(
        matches!(outcome, otel::InitOutcome::SubscriberConflict),
        "init must return SubscriberConflict when its layer cannot attach \
         to a global subscriber, got {outcome:?}"
    );
}
