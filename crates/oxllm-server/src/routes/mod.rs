//! Route registration, split into focused groups.
//!
//! [`build_router`] is the single entry point: it merges every route group,
//! applies the standard auth stacks to the protected groups (leaving the
//! health probes public), attaches the management routes (when `db` is
//! enabled) and the configurable pass-through endpoints, and finally wraps
//! everything in the global layers.

mod azure_legacy;
mod llm;
mod management;
mod middleware;
mod observability;
mod passthrough;
mod platform;

use axum::Router;

use crate::security::body_limit_bytes;
use crate::state::SharedState;

/// Build the application router from a shared state.
///
/// This is a pure refactor of the original `app::build_router`: the route
/// table, middleware ordering, and feature gating are identical. Each route
/// group is registered in its own module; this function composes them.
pub fn build_router(shared: SharedState) -> Router {
    let body_limit = body_limit_bytes(
        shared
            .load_full()
            .config
            .general_settings
            .max_request_body_size_mb,
    );

    let v1_routes = Router::new()
        .merge(llm::routes())
        .merge(azure_legacy::routes());
    #[cfg(feature = "platform-apis")]
    let v1_routes = v1_routes.merge(platform::routes());

    // Apply the auth stack BEFORE the CORS layer so CORS sits outermost over
    // the v1 group (mirrors the management ordering). With auth innermost, a
    // 401/403/429 produced by the auth/ip/brake layers traverses the CORS
    // layer on the way out and picks up `Access-Control-Allow-Origin`, so
    // browser clients see the real status/code instead of a generic CORS
    // error. `auth_with_preflight` remains necessary for bare `OPTIONS`
    // requests that pass through CORS without preflight headers (CORS only
    // short-circuits genuine preflights).
    let v1_routes = middleware::apply_proxy_auth(v1_routes, shared.clone());
    let v1_routes = match middleware::build_proxy_cors(
        &shared
            .load_full()
            .config
            .general_settings
            .proxy_cors_origins,
    ) {
        Some(cors) => v1_routes.layer(cors),
        None => v1_routes,
    };

    let metrics_routes =
        middleware::apply_proxy_auth(observability::metrics_routes(), shared.clone());

    let router = Router::new()
        .merge(observability::health_routes())
        .merge(metrics_routes)
        .merge(v1_routes);

    #[cfg(feature = "db")]
    let router = management::attach(router, shared.clone());

    let router = passthrough::attach_endpoints(router, shared.clone());

    let passthrough_routes =
        middleware::apply_proxy_auth_no_preflight(passthrough::catch_all_routes(), shared.clone());

    middleware::apply_global_layers(router.merge(passthrough_routes), body_limit).with_state(shared)
}
