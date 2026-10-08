//! Observability routes: liveness/readiness probes (public) and the metrics
//! endpoint (auth-protected). The health and metrics routers are returned
//! separately so that `build_router` can apply the auth stack to metrics
//! while leaving the health probes public.

use axum::Router;
use axum::routing::get;

use crate::handlers::{health, passthrough};
use crate::state::SharedState;

/// Register the public health probes (`/health`, `/health/readiness`).
/// These routes never go through auth.
///
/// `/health/liveness` and `/health/liveliness` are aliases for `/health`,
/// matching the reference k8s-friendly liveness route variants
/// (`_health_endpoints.py:1651-1705`). A deployment migrating from an existing proxy
/// with k8s probes pointed at either path must not 404.
pub(crate) fn health_routes() -> Router<SharedState> {
    Router::new()
        .route("/health", get(health::health))
        .route("/health/liveness", get(health::health))
        .route("/health/liveliness", get(health::health))
        .route("/health/readiness", get(health::readiness))
}

/// Register the metrics endpoint (`/metrics`). The returned router carries
/// no auth layers — `build_router` applies the standard proxy auth stack.
pub(crate) fn metrics_routes() -> Router<SharedState> {
    Router::new().route("/metrics", get(passthrough::metrics_handler))
}
