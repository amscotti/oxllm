//! Management route group — key/team/user/spend/model management endpoints.
//!
//! Only mounted when the `db` feature is enabled. These routes sit behind a
//! master-key-only auth layer (`management_auth_middleware`) plus the
//! auth-brake guard, and an optional CORS layer driven by
//! `general_settings.management_cors_origins`.

#[cfg(feature = "db")]
use axum::Router;
#[cfg(feature = "db")]
use axum::middleware;
#[cfg(feature = "db")]
use axum::response::IntoResponse;
#[cfg(feature = "db")]
use axum::routing::{get, post};

#[cfg(feature = "db")]
use crate::state::SharedState;

/// Attach the management endpoints behind a master-key-only auth layer and
/// an explicit CORS layer.
///
/// CORS is driven by `general_settings.management_cors_origins`. When unset
/// or empty, no CORS layer is added (the browser will block all cross-origin
/// requests). Otherwise each entry must be a fully-qualified origin (scheme,
/// host, optional port) such as `https://admin.example.com` or
/// `http://localhost:3000`.
#[cfg(feature = "db")]
pub(crate) fn attach(router: Router<SharedState>, shared: SharedState) -> Router<SharedState> {
    use crate::handlers::management;

    let cors = build_management_cors(
        &shared
            .load_full()
            .config
            .general_settings
            .management_cors_origins,
    );

    let mgmt = Router::new()
        .route("/key/generate", post(management::keys::generate_key))
        .route("/key/list", get(management::keys::list_keys))
        .route("/key/info", get(management::keys::key_info))
        .route("/key/delete", post(management::keys::delete_key))
        .route("/key/purge", post(management::keys::purge_key))
        .route("/key/update", post(management::keys::update_key))
        .route("/key/rotate", post(management::keys::rotate_key))
        .route("/key/block", post(management::keys::block_key))
        .route("/key/unblock", post(management::keys::unblock_key))
        .route("/team/add", post(management::teams::add_team))
        .route("/team/list", get(management::teams::list_teams))
        .route("/team/info", get(management::teams::team_info))
        .route("/team/delete", post(management::teams::delete_team))
        .route("/team/update", post(management::teams::update_team))
        .route("/team/block", post(management::teams::block_team))
        .route("/team/unblock", post(management::teams::unblock_team))
        .route("/team/member/add", post(management::teams::member_add))
        .route(
            "/team/member/update",
            post(management::teams::member_update),
        )
        .route(
            "/team/member/delete",
            post(management::teams::member_delete),
        )
        .route("/team/model/add", post(management::teams::team_model_add))
        .route(
            "/team/model/delete",
            post(management::teams::team_model_delete),
        )
        .route("/customer/new", post(management::customers::new_customer))
        .route(
            "/customer/update",
            post(management::customers::update_customer),
        )
        .route(
            "/customer/delete",
            post(management::customers::delete_customer),
        )
        .route("/customer/info", get(management::customers::customer_info))
        .route("/customer/list", get(management::customers::list_customers))
        .route(
            "/customer/block",
            post(management::customers::block_customer),
        )
        .route(
            "/customer/unblock",
            post(management::customers::unblock_customer),
        )
        .route("/budget/new", post(management::budgets::new_budget))
        .route("/budget/update", post(management::budgets::update_budget))
        .route("/budget/delete", post(management::budgets::delete_budget))
        .route(
            "/budget/info",
            get(management::budgets::budget_info).post(management::budgets::budget_info_bulk),
        )
        .route("/budget/list", get(management::budgets::list_budgets))
        .route("/tag/new", post(management::tags::new_tag))
        .route("/tag/update", post(management::tags::update_tag))
        .route("/tag/delete", post(management::tags::delete_tag))
        .route("/tag/info", get(management::tags::tag_info))
        .route("/tag/list", get(management::tags::list_tags))
        .route("/skill/new", post(management::skills::new_skill))
        .route("/skill/list", get(management::skills::list_skills))
        .route("/skill/{skill_id}", get(management::skills::skill_info))
        .route("/skill/update", post(management::skills::update_skill))
        .route("/skill/delete", post(management::skills::delete_skill))
        .route(
            "/cache/settings",
            get(management::cache_settings::get_cache_settings)
                .post(management::cache_settings::update_cache_settings),
        )
        .route("/user/add", post(management::users::add_user))
        .route("/user/list", get(management::users::list_users))
        .route("/user/info", get(management::users::user_info))
        .route("/user/update", post(management::users::update_user))
        .route("/user/delete", post(management::users::delete_user))
        .route("/organization/add", post(management::orgs::add_org))
        .route("/organization/list", get(management::orgs::list_orgs))
        .route("/organization/info", get(management::orgs::org_info))
        .route("/organization/update", post(management::orgs::update_org))
        .route("/organization/delete", post(management::orgs::delete_org))
        .route("/project/add", post(management::projects::add_project))
        .route("/project/list", get(management::projects::list_projects))
        .route("/project/info", get(management::projects::project_info))
        .route(
            "/project/update",
            post(management::projects::update_project),
        )
        .route(
            "/project/delete",
            post(management::projects::delete_project),
        )
        .route("/spend/logs", get(management::spend::spend_logs))
        .route("/spend/logs/v2", get(management::spend::spend_logs_v2))
        .route("/spend/total", get(management::spend::spend_total))
        .route("/spend/calculate", post(management::spend::spend_calculate))
        .route("/spend/keys", get(management::spend::spend_by_key))
        .route("/spend/by-model", get(management::spend::spend_by_model))
        .route("/spend/tags", get(management::spend::spend_by_tag))
        .route("/spend/users", get(management::spend::spend_by_user))
        .route("/spend/teams", get(management::spend::spend_by_team))
        .route("/spend/organizations", get(management::spend::spend_by_org))
        .route("/spend/models", get(management::spend::spend_by_model))
        .route(
            "/spend/regenerate",
            post(management::spend::spend_regenerate),
        )
        .route("/global/spend", get(management::spend::global_spend))
        .route("/global/activity", get(management::spend::global_activity))
        .route("/audit/logs", get(management::audit::list_audit_logs))
        .route(
            "/request/logs/{request_id}",
            get(management::audit::get_request_log),
        )
        .route("/model/list", get(management::models_mgmt::list_models))
        .route("/model/info", get(management::models_mgmt::model_info))
        .route("/model/new", post(management::models_mgmt::add_model))
        .route("/model/update", post(management::models_mgmt::update_model))
        .route("/model/delete", post(management::models_mgmt::delete_model))
        .route("/config", get(management::config::get_config))
        .route("/config/update", post(management::config::update_config))
        .route("/routes", get(management::config::get_routes))
        .route(
            "/routing/state",
            get(management::routing::get_routing_state),
        )
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            management_auth_with_preflight,
        ))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            crate::security::allowed_ips_guard,
        ))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            crate::security::auth_brake_guard,
        ));

    let mgmt = match cors {
        Some(cors) => mgmt.layer(cors),
        None => mgmt,
    };

    router.merge(mgmt)
}

/// [`crate::auth::management_auth_middleware`] wrapped with a CORS preflight
/// (OPTIONS) bypass. Browsers send preflight `OPTIONS` requests without an
/// `Authorization` header by design; this lets the CORS layer produce the
/// preflight response instead of returning 401. Only `OPTIONS` bypasses the
/// master-key check — all real methods still require it.
///
/// The bypass is gated on a management CORS layer actually being configured:
/// when `management_cors_origins` is unset there is no CORS layer to answer
/// the preflight, and forwarding the bare `OPTIONS` would land it in a
/// handler whose `Extension<AuthUser>` extractor is missing → 500. Return 405
/// instead.
#[cfg(feature = "db")]
async fn management_auth_with_preflight(
    state: axum::extract::State<SharedState>,
    headers: axum::http::HeaderMap,
    request: axum::extract::Request,
    next: middleware::Next,
) -> axum::response::Response {
    if request.method() == axum::http::Method::OPTIONS {
        if management_cors_enabled(
            &state
                .load_full()
                .config
                .general_settings
                .management_cors_origins,
        ) {
            // Let the CORS layer handle the preflight — no auth required.
            return next.run(request).await;
        }
        return axum::http::StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    crate::auth::management_auth_middleware(state, headers, request, next).await
}

/// Whether a management CORS layer would be built from the configured
/// allow-list (non-empty and at least one parseable origin). Kept separate so
/// the preflight bypass and the layer builder agree on one condition.
#[cfg(feature = "db")]
fn management_cors_enabled(origins: &Option<Vec<String>>) -> bool {
    let Some(list) = origins.as_ref() else {
        return false;
    };
    !list.is_empty()
        && list
            .iter()
            .any(|o| axum::http::HeaderValue::from_str(o).is_ok())
}

/// Build the management CORS layer from the configured allow-list. Returns
/// `None` when CORS should be disabled entirely (no allow-list configured).
#[cfg(feature = "db")]
fn build_management_cors(origins: &Option<Vec<String>>) -> Option<tower_http::cors::CorsLayer> {
    use tower_http::cors::{AllowOrigin, CorsLayer};

    if !management_cors_enabled(origins) {
        return None;
    }
    let list = origins.as_ref().expect("cors enabled implies a list");
    let parsed: Vec<axum::http::HeaderValue> = list
        .iter()
        .filter_map(|o| axum::http::HeaderValue::from_str(o).ok())
        .collect();
    Some(
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(parsed))
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers([
                axum::http::header::AUTHORIZATION,
                axum::http::header::CONTENT_TYPE,
            ]),
    )
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::*;
    use crate::config::types::{
        ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings,
    };
    use crate::routes::build_router;
    use crate::state::SharedState;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn config_with(master_key: Option<String>, allowed_ips: Option<Vec<String>>) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: crate::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                master_key,
                allowed_ips,
                ..Default::default()
            },
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        }
    }

    fn shared_for(config: ProxyConfig) -> SharedState {
        crate::app::build_shared_state(config).expect("state")
    }

    async fn send(router: axum::Router, bearer: Option<&str>) -> axum::http::Response<Body> {
        let mut builder = Request::builder().method("GET").uri("/key/list");
        if let Some(tok) = bearer {
            builder = builder.header("authorization", format!("Bearer {tok}"));
        }
        router
            .oneshot(builder.body(Body::empty()).expect("request"))
            .await
            .expect("response")
    }

    /// Like `send` but attaches a connection peer address. The auth brake only
    /// records/enforces failures for requests with an attributable peer;
    /// peerless requests are skipped to avoid a shared "unknown" bucket.
    async fn send_with_peer(
        router: axum::Router,
        bearer: Option<&str>,
    ) -> axum::http::Response<Body> {
        let peer: std::net::SocketAddr = "8.8.8.8:5555".parse().expect("peer");
        let mut builder = Request::builder()
            .method("GET")
            .uri("/key/list")
            .extension(axum::extract::ConnectInfo(peer));
        if let Some(tok) = bearer {
            builder = builder.header("authorization", format!("Bearer {tok}"));
        }
        router
            .oneshot(builder.body(Body::empty()).expect("request"))
            .await
            .expect("response")
    }

    #[test]
    fn management_cors_disabled_when_unset() {
        assert!(build_management_cors(&None).is_none());
    }

    #[test]
    fn management_cors_disabled_when_empty() {
        assert!(build_management_cors(&Some(vec![])).is_none());
    }

    #[test]
    fn management_cors_enabled_when_origins_listed() {
        let layer = build_management_cors(&Some(vec![
            "http://localhost:3000".to_string(),
            "https://admin.example.com".to_string(),
        ]));
        assert!(layer.is_some());
    }

    #[test]
    fn management_cors_disabled_when_origins_unparseable() {
        assert!(build_management_cors(&Some(vec!["invalid\norigin".to_string()])).is_none());
    }

    /// Issue #12: the management router must be guarded by `allowed_ips_guard`.
    /// A request whose resolved client IP is outside the configured allowlist
    /// is rejected with 403 before reaching the auth or handler layers.
    #[tokio::test]
    async fn management_route_blocked_by_ip_allowlist() {
        let shared = shared_for(config_with(
            Some("sk-master-test-key".to_string()),
            Some(vec!["10.0.0.0/8".to_string()]),
        ));
        let router = build_router(shared);
        let resp = send(router, Some("sk-master-test-key")).await;
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::FORBIDDEN,
            "management route must be blocked by allowed_ips guard \
 (no peer info → 'unknown' is not in 10.0.0.0/8)"
        );
    }

    /// Issue #12: repeated failed master-key auth against management routes
    /// must trip the auth brake. After the configured failure threshold the
    /// `auth_brake_guard` returns 429 instead of 401.
    #[tokio::test]
    async fn management_auth_brake_fires_on_repeated_failures() {
        let shared = shared_for(config_with(Some("sk-master-test-key".to_string()), None));
        // AuthBrake::default_policy(): 10 fails / 60s → block 300s.
        let threshold: u32 = 10;

        for _ in 0..threshold {
            let router = build_router(shared.clone());
            let resp = send_with_peer(router, Some("wrong-key")).await;
            assert_eq!(
                resp.status(),
                axum::http::StatusCode::UNAUTHORIZED,
                "failed master-key auth must return 401 and record a failure"
            );
        }

        let router = build_router(shared.clone());
        let resp = send_with_peer(router, Some("wrong-key")).await;
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            "auth brake must block after {threshold} failed management auth attempts"
        );
    }

    /// C05/LOW: the management auth path must only brake *presented-but-wrong*
    /// credentials — a keyless probe (monitoring agents, scanners) must not
    /// inflate the per-IP failure counter (matching the LLM `auth_middleware`
    /// rationale).
    #[tokio::test]
    async fn management_missing_token_does_not_count_failure() {
        let shared = shared_for(config_with(Some("sk-master-test-key".to_string()), None));
        let router = build_router(shared.clone());

        // No Authorization header → 401 but NO brake increment.
        let resp = send_with_peer(router.clone(), None).await;
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
        let brake = shared.load_full().auth_brake.clone().expect("auth brake");
        assert_eq!(
            brake.tracked_ips(),
            0,
            "missing-credential probes must not inflate the auth-brake counter"
        );

        // A presented-but-wrong token → 401 AND a brake increment.
        let resp = send_with_peer(router, Some("wrong-key")).await;
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
        assert_eq!(
            brake.tracked_ips(),
            1,
            "a wrong-token attempt must count as a real auth failure"
        );
    }

    /// An unauthenticated `OPTIONS` on a management route must never yield a
    /// 500 when no management CORS layer is configured. Previously the
    /// preflight wrapper bypassed auth unconditionally, so the bare `OPTIONS`
    /// reached a handler extracting `Extension<AuthUser>` → `MissingExtension`
    /// → 500. With no CORS layer to answer the preflight, 405 is correct.
    #[tokio::test]
    async fn options_without_management_cors_returns_405_not_500() {
        let shared = shared_for(config_with(Some("sk-master-test-key".to_string()), None));
        let router = build_router(shared);
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/key/list")
            .body(Body::empty())
            .expect("request");
        let resp = router.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::METHOD_NOT_ALLOWED,
            "OPTIONS with management_cors_origins unset must return 405, never 500"
        );
    }

    /// When management CORS IS configured, a browser preflight `OPTIONS` is
    /// answered by the (outermost) CORS layer — the preflight bypass exists so
    /// this works without an Authorization header.
    #[tokio::test]
    async fn options_with_management_cors_answered_by_cors_layer() {
        let mut config = config_with(Some("sk-master-test-key".to_string()), None);
        config.general_settings.management_cors_origins =
            Some(vec!["http://localhost:3000".to_string()]);
        let shared = shared_for(config);
        let router = build_router(shared);
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/key/list")
            .header("origin", "http://localhost:3000")
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .expect("request");
        let resp = router.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        assert!(
            resp.headers().contains_key("access-control-allow-origin"),
            "the CORS layer must answer the preflight"
        );
    }
}
