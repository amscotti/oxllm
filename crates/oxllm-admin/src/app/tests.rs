use super::*;
use crate::auth::{LoginBrake, SESSION_COOKIE, create_session};
use crate::config::AdminConfig;
use crate::dates::{
    civil_date_from_unix_days, days_before_iso, parse_iso_date_to_unix_days, today_iso_date,
};
use crate::pages::spend::{
    SpendQuery, budget_spend_tier, enrich_spend_logs, enrich_with_budget_spend,
    enrich_with_budget_spend_bounded, sort_spend_logs, spend_total_from_response,
    validate_export_date_range,
};
use axum::body::Body;
use axum::http::Request;
use httpmock::MockServer;
use tower::ServiceExt;

#[path = "review_tests.rs"]
mod review;

fn default_brake() -> Arc<LoginBrake> {
    Arc::new(LoginBrake::new(5, 60, 300))
}

fn test_state_with_proxy(server: &MockServer) -> AppState {
    let config = AdminConfig {
        listen: "127.0.0.1:0".to_string(),
        proxy_url: server.base_url(),
        master_key: "sk-test".to_string(),
        session_secret: "unit-test-secret-value".to_string(),
        session_ttl_hours: 1,
        login_password: None,
        trusted_proxies: Vec::new(),
        cookie_secure: false,
        login_max_fails: 5,
        login_window_secs: 60,
        login_block_secs: 300,
    };
    AppState {
        config: Arc::new(config),
        proxy: Arc::new(ProxyClient::new(&server.base_url(), "sk-test")),
        templates: Arc::new(Templates::new()),
        login_brake: default_brake(),
    }
}

fn authed_get(uri: &str) -> Request<Body> {
    let session = create_session("unit-test-secret-value", 1);
    Request::builder()
        .uri(uri)
        .header(
            "cookie",
            format!("{SESSION_COOKIE}={}", session.cookie_value),
        )
        .body(Body::empty())
        .unwrap()
}

fn authed_htmx_get(uri: &str) -> Request<Body> {
    let session = create_session("unit-test-secret-value", 1);
    Request::builder()
        .uri(uri)
        .header(
            "cookie",
            format!("{SESSION_COOKIE}={}", session.cookie_value),
        )
        .header("HX-Request", "true")
        .body(Body::empty())
        .unwrap()
}

fn authed_post(uri: &str, body: &str) -> Request<Body> {
    let session = create_session("unit-test-secret-value", 1);
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(
            "cookie",
            format!("{SESSION_COOKIE}={}", session.cookie_value),
        )
        .header("X-CSRF-Token", &session.csrf_token)
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn authed_post_without_csrf(uri: &str, body: &str) -> Request<Body> {
    let session = create_session("unit-test-secret-value", 1);
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(
            "cookie",
            format!("{SESSION_COOKIE}={}", session.cookie_value),
        )
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Mint a valid pre-login CSRF pair against the unit-test secret:
/// `(form_field_value, cookie_header_value)`. The form field is
/// percent-encoded because the token is base64 (contains `+`, `/`, `=`).
fn login_csrf_pair() -> (String, String) {
    let (token, cookie) = crate::auth::create_login_csrf("unit-test-secret-value", false);
    let value = cookie.split(';').next().unwrap().to_string();
    (crate::urlencode::urlencoding_minimal(&token), value)
}

async fn body_string(resp: Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    String::from_utf8(bytes.to_vec()).unwrap_or_default()
}

#[tokio::test]
async fn health_returns_200_json() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn protected_route_redirects_without_session() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/dashboard")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    let loc = resp
        .headers()
        .get("location")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(loc, "/login");
}

#[tokio::test]
async fn login_form_renders() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/login")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn keys_page_full_renders_table() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(
            r#"{"keys":[{"key_id":"key-1","key_alias":"alpha","is_active":true,"created_at":"2024-01-01"}]}"#,
        );
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200)
            .body(r#"{"teams":[{"team_id":"team-eng","team_name":"Engineering"}]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/user/list");
        then.status(200)
            .body(r#"{"users":[{"user_id":"user-1","user_name":"Ada"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/keys")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("API Keys"));
    assert!(html.contains("key-1"));
    assert!(html.contains("alpha"));
    assert!(html.contains("/keys/key-1/delete"));
    // Team/user selects (LiteLLM-style)
    assert!(html.contains("<select"));
    assert!(html.contains("team-eng"));
    assert!(html.contains("Engineering"));
    assert!(html.contains("user-1"));
}

#[tokio::test]
async fn keys_page_htmx_returns_partial_only() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[{"key_id":"k9"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_htmx_get("/keys")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("k9"));
    assert!(
        !html.contains("<html"),
        "HTMX response should not include base layout"
    );
}

#[tokio::test]
async fn keys_page_proxy_error_shows_alert() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(500).body(r#"{"error":"boom"}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_htmx_get("/keys")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("alert"));
    assert!(html.contains("500"));
}

#[tokio::test]
async fn keys_page_proxy_401_returns_htmx_redirect() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(401).body(r#"{"error":"unauthorized"}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_htmx_get("/keys")).await.unwrap();
    let redirect = resp
        .headers()
        .get("HX-Redirect")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(redirect, "/login");
}

/// Regression guard for the typed `ProxyStatusError` switch in
/// `is_proxy_unauthorized`: a 5xx body that happens to contain the literal
/// `"status 401"` must NOT trigger a spurious login redirect. The old
/// substring match (`msg.contains("status 401")`) would have redirected here.
#[tokio::test]
async fn keys_page_proxy_500_with_401_in_body_does_not_redirect() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(500)
            .body(r#"{"error":"upstream returned status 401 from inner call"}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_htmx_get("/keys")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let redirect = resp
        .headers()
        .get("HX-Redirect")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        redirect.is_empty(),
        "500-with-401-in-body must NOT redirect to /login, got HX-Redirect={redirect:?}"
    );
    let html = body_string(resp).await;
    assert!(
        html.contains("500"),
        "expected the 500 status surfaced: {html}"
    );
}

#[tokio::test]
async fn keys_create_returns_modal_with_key() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/key/generate");
        then.status(200)
            .body(r#"{"key":"sk-ox-secret","key_id":"key-new"}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/keys/create", "key_alias=my-key"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // Secret-bearing response must not be stored in HTTP caches.
    let cc = resp
        .headers()
        .get(axum::http::header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        cc.contains("no-store"),
        "expected Cache-Control: no-store on key_created, got {cc:?}"
    );
    let html = body_string(resp).await;
    assert!(html.contains("sk-ox-secret"));
    assert!(html.contains("key-new"));
    assert!(html.contains("Copy"));
    assert!(
        !html.contains("x-cloak"),
        "the one-time secret must remain visible without Alpine"
    );
    assert!(
        html.contains("<a href=\"/keys\""),
        "Done must work without Alpine"
    );
}

#[tokio::test]
async fn keys_rotate_renders_new_key_with_no_store() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/key/rotate");
        then.status(200)
            .body(r#"{"key":"sk-rotated","key_id":"key-9"}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/keys/key-9/rotate", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let cc = resp
        .headers()
        .get(axum::http::header::CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        cc.contains("no-store"),
        "expected Cache-Control: no-store on rotated key, got {cc:?}"
    );
    let html = body_string(resp).await;
    assert!(html.contains("sk-rotated"));
}

#[tokio::test]
async fn keys_delete_refreshes_table_as_disabled() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/delete")
            .json_body_partial(r#"{"key_id":"key-9"}"#);
        then.status(200).body(r#"{"status":"deleted"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-9","key_alias":"gone","is_active":false}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/keys/key-9/delete", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
    let body = body_string(resp).await;
    assert!(body.contains("keys-panel"));
    assert!(body.contains("key-9"));
    assert!(body.contains("Disabled"));
    assert!(!body.contains("hx-post=\"/keys/key-9/delete\""));
    assert!(body.contains("hx-post=\"/keys/key-9/enable\""));
    assert!(body.contains("hx-post=\"/keys/key-9/purge\""));
}

#[tokio::test]
async fn keys_enable_refreshes_table_as_active() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/update")
            .json_body_partial(r#"{"key_id":"key-9","is_active":true}"#);
        then.status(200)
            .body(r#"{"key_id":"key-9","key_alias":"back","is_active":true}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-9","key_alias":"back","is_active":true}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/keys/key-9/enable", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
    let body = body_string(resp).await;
    assert!(body.contains("Active"));
    assert!(body.contains("hx-post=\"/keys/key-9/delete\""));
    assert!(body.contains("re-enabled"));
}

#[tokio::test]
async fn keys_purge_removes_row_from_table() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/purge")
            .json_body_partial(r#"{"key_id":"key-9"}"#);
        then.status(200).body(r#"{"status":"purged"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/keys/key-9/purge", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
    let body = body_string(resp).await;
    assert!(body.contains("keys-panel"));
    assert!(!body.contains("id=\"key-row-key-9\""));
    assert!(body.contains("permanently removed"));
}

#[tokio::test]
async fn teams_page_full_renders_table() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200)
            .body(r#"{"teams":[{"team_id":"team-1","team_name":"eng","max_budget":100.0}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/teams")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("team-1"));
    assert!(html.contains("eng"));
}

#[tokio::test]
async fn teams_create_refreshes_table() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/team/add");
        then.status(200).body(r#"{"team_id":"team-2"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200)
            .body(r#"{"teams":[{"team_id":"team-2","team_name":"new"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/teams/create", "team_name=new"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("team-2"));
    assert!(html.contains("new"));
}

#[tokio::test]
async fn users_page_full_renders_table() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/user/list");
        then.status(200)
            .body(r#"{"users":[{"user_id":"u1","user_name":"alice"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/users")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("u1"));
    assert!(html.contains("alice"));
}

#[tokio::test]
async fn users_create_refreshes_table() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/user/add");
        then.status(200).body(r#"{"user_id":"u2"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/user/list");
        then.status(200)
            .body(r#"{"users":[{"user_id":"u2","user_name":"bob"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/users/create", "user_name=bob"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("u2"));
}

#[tokio::test]
async fn spend_page_full_renders_table() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/logs/v2");
        then.status(200).body(
            r#"{"data":[{"timestamp":"2024-01-01","model":"gpt-4o","cost":0.01,"key_id":"key-1","team_id":"team-1"}],"next_cursor":42}"#,
        );
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-1","key_alias":"prod","is_active":true}]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200)
            .body(r#"{"teams":[{"team_id":"team-1","team_name":"Platform"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/spend")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("gpt-4o"));
    assert!(html.contains("id=\"spend-panel\""));
    assert!(html.contains("Load more") || html.contains("Load More"));
    assert!(html.contains("$0.01000000") || html.contains("0.01"));
    assert!(html.contains("prod") || html.contains("key-1"));
    assert!(html.contains("Platform") || html.contains("team-1"));
    assert!(html.contains("Export CSV") || html.contains("export.csv"));
    assert!(html.contains("sort_by=model") || html.contains("sort_by=cost"));
}

#[tokio::test]
async fn spend_export_csv_requires_date_range() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_get("/spend/export.csv?model=gpt-4o"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_string(resp).await;
    assert!(body.contains("date range") || body.contains("start_date"));
}

#[tokio::test]
async fn spend_export_csv_returns_attachment_for_range() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("start_date", "2024-01-01")
            .query_param("end_date", "2024-01-31");
        then.status(200).body(
            r#"{"data":[{"timestamp":"2024-01-15T00:00:00Z","model":"gpt-4o","provider":"openai","cost":0.01,"key_id":"key-1","prompt_tokens":10,"completion_tokens":5}],"next_cursor":null}"#,
        );
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-1","key_alias":"prod"}]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200).body(r#"{"teams":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_get(
            "/spend/export.csv?start_date=2024-01-01&end_date=2024-01-31&model=gpt-4o",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/csv"), "content-type={ct}");
    let cd = resp
        .headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(cd.contains("attachment"));
    assert!(cd.contains("2024-01-01") && cd.contains("2024-01-31"));
    let body = body_string(resp).await;
    assert!(body.contains("timestamp,request_id,model"));
    assert!(body.contains("gpt-4o"));
    assert!(body.contains("key-1"));
    assert!(body.contains("prod"));
}

#[test]
fn iso_date_roundtrip_helpers() {
    assert_eq!(parse_iso_date_to_unix_days("1970-01-01"), Some(0));
    assert_eq!(civil_date_from_unix_days(0), "1970-01-01");
    assert_eq!(
        days_before_iso("2024-01-31", 30).as_deref(),
        Some("2024-01-01")
    );
    let q = SpendQuery::default().with_default_date_range(30);
    assert!(q.has_date_range());
    assert!(validate_export_date_range(&q).is_ok());
    let bad = SpendQuery {
        start_date: Some("2024-02-01".into()),
        end_date: Some("2024-01-01".into()),
        ..Default::default()
    };
    assert!(validate_export_date_range(&bad).is_err());
}

#[test]
fn sort_spend_logs_by_cost_desc() {
    let mut logs = vec![
        json!({"cost": 0.01, "model": "a", "timestamp": "1"}),
        json!({"cost": 0.05, "model": "b", "timestamp": "2"}),
        json!({"cost": 0.02, "model": "c", "timestamp": "3"}),
    ];
    sort_spend_logs(&mut logs, "cost", "desc");
    assert_eq!(logs[0]["model"], "b");
    assert_eq!(logs[2]["model"], "a");
}

/// LOW: the default (timestamp) sort must order rows chronologically even
/// when timestamps are Unix-epoch numbers, not ISO 8601 strings. Pure
/// lexicographic comparison of numbers-as-strings mis-sorts (e.g. "10" < "9").
#[test]
fn sort_spend_logs_sorts_numeric_timestamps_chronologically() {
    let mut logs = vec![
        json!({"timestamp": 1700000002, "id": "b"}),
        json!({"timestamp": 1700000000, "id": "a"}),
        json!({"timestamp": 1700000001, "id": "c"}),
    ];
    sort_spend_logs(&mut logs, "timestamp", "asc");
    assert_eq!(logs[0]["id"], "a");
    assert_eq!(logs[1]["id"], "c");
    assert_eq!(logs[2]["id"], "b");
    // Desc reverses the order.
    sort_spend_logs(&mut logs, "timestamp", "desc");
    assert_eq!(logs[0]["id"], "b");
    assert_eq!(logs[2]["id"], "a");
}

/// LOW (cont.): numeric timestamps expressed as strings (epoch seconds) must
/// also sort numerically — lexicographic order would put "17000000002" before
/// "1700000001" (10-char vs 11-char).
#[test]
fn sort_spend_logs_sorts_numeric_string_timestamps_by_value() {
    let mut logs = vec![
        json!({"timestamp": "1700000002", "id": "b"}),
        json!({"timestamp": "1700000000", "id": "a"}),
        json!({"timestamp": "1700000001", "id": "c"}),
    ];
    sort_spend_logs(&mut logs, "timestamp", "asc");
    assert_eq!(logs[0]["id"], "a");
    assert_eq!(logs[1]["id"], "c");
    assert_eq!(logs[2]["id"], "b");
}

/// LOW (cont.): ISO 8601 strings (the proxy's actual format) must still sort
/// correctly via the lexicographic fallback.
#[test]
fn sort_spend_logs_sorts_iso8601_timestamps_chronologically() {
    let mut logs = vec![
        json!({"timestamp": "2024-01-03T00:00:00Z", "id": "c"}),
        json!({"timestamp": "2024-01-01T00:00:00Z", "id": "a"}),
        json!({"timestamp": "2024-01-02T00:00:00Z", "id": "b"}),
    ];
    sort_spend_logs(&mut logs, "timestamp", "asc");
    assert_eq!(logs[0]["id"], "a");
    assert_eq!(logs[1]["id"], "b");
    assert_eq!(logs[2]["id"], "c");
}

#[test]
fn enrich_spend_logs_adds_aliases() {
    let logs = vec![json!({"key_id": "key-1", "team_id": "team-1", "model": "m"})];
    let keys = vec![json!({"key_id": "key-1", "key_alias": "prod"})];
    let teams = vec![json!({"team_id": "team-1", "team_name": "Eng"})];
    let out = enrich_spend_logs(logs, &keys, &teams);
    assert_eq!(out[0]["key_alias"], "prod");
    assert_eq!(out[0]["team_name"], "Eng");
}

#[test]
fn budget_spend_tier_thresholds() {
    assert_eq!(budget_spend_tier(0.0), "ok");
    assert_eq!(budget_spend_tier(49.9), "ok");
    assert_eq!(budget_spend_tier(50.0), "warn");
    assert_eq!(budget_spend_tier(79.9), "warn");
    assert_eq!(budget_spend_tier(80.0), "danger");
    assert_eq!(budget_spend_tier(150.0), "danger");
}

#[test]
fn spend_total_from_response_handles_variants() {
    assert_eq!(
        spend_total_from_response(&json!({"total_spend": 1.23})),
        1.23
    );
    assert_eq!(
        spend_total_from_response(&json!({"total_cost": 4.56})),
        4.56
    );
    assert_eq!(
        spend_total_from_response(&json!({
            "results": [{"total_spend": 7.89}]
        })),
        7.89
    );
    assert_eq!(spend_total_from_response(&json!({})), 0.0);
}

/// `/spend/total` has no `user_id` filter: querying it with one returns the
/// GLOBAL total, which every user row would then display against its own
/// budget. Enrichment must skip unsupported scopes entirely (zero calls).
#[tokio::test]
async fn enrich_with_budget_spend_skips_unsupported_scopes() {
    let server = MockServer::start();
    // No /spend/total mock: any call would register as a hit below.
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![
        json!({"user_id": "u1", "max_budget": 50.0}),
        json!({"team_id": "t-no-budget"}),
    ];
    enrich_with_budget_spend(&proxy, &mut rows, "user_id").await;
    assert!(
        rows[0].get("budget_spend").is_none(),
        "no user-scoped total exists"
    );
}

/// Teams ARE supported (`/spend/total?team_id=`), so team rows must still be
/// enriched with a team-scoped figure, not a global one.
#[tokio::test]
async fn enrich_with_budget_spend_queries_team_scope() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("team_id", "t1")
            .query_param_exists("start_date");
        then.status(200)
            .body(r#"{"total_cost": 12.5, "total_requests": 4}"#);
    });
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![json!({"team_id": "t1", "max_budget": 100.0})];
    enrich_with_budget_spend(&proxy, &mut rows, "team_id").await;
    m.assert();
    assert_eq!(rows[0]["budget_spend"], 12.5);
}

#[tokio::test]
async fn enrich_with_budget_spend_skips_entities_without_budget() {
    let server = MockServer::start();
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![
        json!({"key_id": "k1", "max_budget": serde_json::Value::Null}),
        json!({"key_id": "k2"}),
        json!({"key_id": "k3", "max_budget": 0.0}),
    ];
    // Should make ZERO spend/total calls (no budget on any row).
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    assert!(rows[0].get("budget_spend").is_none());
    assert!(rows[1].get("budget_spend").is_none());
    assert!(rows[2].get("budget_spend").is_none());
}

#[tokio::test]
async fn enrich_with_budget_spend_attaches_pct_and_tier() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("key_id", "k1");
        then.status(200).body(r#"{"total_spend": 45.0}"#);
    });
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![json!({"key_id": "k1", "max_budget": 100.0})];
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    assert_eq!(rows[0]["budget_spend"], 45.0);
    assert_eq!(rows[0]["budget_spend_pct"], 45.0);
    assert_eq!(rows[0]["budget_spend_tier"], "ok");
}

#[tokio::test]
async fn enrich_with_budget_spend_tolerates_proxy_failure() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/total");
        then.status(500).body(r#"{"error":"boom"}"#);
    });
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![json!({"key_id": "k1", "max_budget": 100.0})];
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    assert_eq!(rows[0]["budget_spend"], 0.0);
    assert_eq!(rows[0]["budget_spend_pct"], 0.0);
    assert_eq!(rows[0]["budget_spend_tier"], "ok");
}

#[tokio::test]
async fn spend_export_csv_passes_snapshot_filter() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("start_date", "2024-01-01")
            .query_param("end_date", "2024-01-31")
            .query_param_exists("created_at_max");
        then.status(200)
            .body(r#"{"data":[{"timestamp":"2024-01-15","model":"gpt-4o","cost":0.01}],"next_cursor":null}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200).body(r#"{"teams":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_get(
            "/spend/export.csv?start_date=2024-01-01&end_date=2024-01-31",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

#[tokio::test]
async fn spend_export_csv_caps_future_end_date_at_today() {
    let server = MockServer::start();
    let today = today_iso_date();
    let yesterday = days_before_iso(&today, 1).unwrap_or_else(|| today.clone());
    let today_clone = today.clone();
    // Request an end_date in the future; the snapshot should cap it at today.
    // Use a 1-day window so it stays within the export range cap.
    let m = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("end_date", today_clone.clone())
            .query_param("created_at_max", today_clone);
        then.status(200).body(r#"{"data":[],"next_cursor":null}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200).body(r#"{"teams":[]}"#);
    });

    let future = days_before_iso(&today, -7).unwrap_or_else(|| today.clone());
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_get(&format!(
            "/spend/export.csv?start_date={yesterday}&end_date={future}"
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

#[tokio::test]
async fn spend_htmx_refresh_returns_single_panel_root() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/logs/v2");
        then.status(200).body(
            r#"{"data":[{"timestamp":"2024-01-01","model":"gpt-4o","cost":0.01}],"next_cursor":null}"#,
        );
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_htmx_get("/spend")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    let panel_count = html.matches("id=\"spend-panel\"").count();
    assert_eq!(
        panel_count, 1,
        "HTMX spend partial must have exactly one #spend-panel root"
    );
    assert!(
        !html.contains("<html"),
        "HTMX response should not include base layout"
    );
    // Header "Spend Logs" appears once (not nested on refresh)
    assert_eq!(html.matches(">Spend Logs<").count(), 1);
}

#[tokio::test]
async fn keys_page_includes_create_form_controls() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/keys")).await.unwrap();
    let html = body_string(resp).await;
    assert!(html.contains("hx-post=\"/keys/create\""));
    assert!(html.contains("name=\"key_alias\""));
    assert!(html.contains("htmx:configRequest"));
    assert!(html.contains("document.addEventListener('htmx:configRequest'"));
    assert!(!html.contains("document.body.addEventListener('htmx:configRequest'"));
}

#[tokio::test]
async fn spend_pagination_returns_rows_and_oob_button() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("cursor", "10");
        then.status(200).body(
 r#"{"data":[{"timestamp":"2024-01-02","model":"claude","cost":0.02}],"next_cursor":null}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_htmx_get("/spend?cursor=10"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("claude"));
    assert!(html.contains("hx-swap-oob"));
    assert!(html.contains("End of results") || html.contains("spend-load-more"));
}

#[tokio::test]
async fn dashboard_renders_summary() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/global/spend");
        then.status(200)
            .body(r#"{"total_cost":12.34,"total_requests":100}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/global/activity");
        then.status(200).body(
            r#"{"results":[{"day":"2024-01-01","spend":1.5},{"day":"2024-01-02","spend":3.0}],"days":30,"total_spend":4.5}"#,
        );
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/health");
        then.status(200).body(r#"{"status":"healthy"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/health/readiness");
        then.status(200)
            .body(r#"{"status":"healthy","checks":{"redis":"ok","db":"ok"}}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/dashboard")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("$12.3400"));
    assert!(html.contains("100 requests"));
    assert!(html.contains("Online"));
    // bar_pct for max day should be 100%
    assert!(
        html.contains("width: 100%") || html.contains("width: 100.0%"),
        "expected non-zero activity bar width, got snippet without 100%"
    );
    assert!(html.contains("width: 50%") || html.contains("width: 50.0%"));
}

#[tokio::test]
async fn logout_clears_session_cookie() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_post("/logout", "")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    let cookie = resp
        .headers()
        .get("set-cookie")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(cookie.contains("Max-Age=0") || cookie.contains("max-age=0"));
    assert!(resp.headers().get("location").and_then(|v| v.to_str().ok()) == Some("/login"));
}

/// HIGH: `/logout` must be POST-only. A cross-site `<img src="/logout">`
/// could otherwise force a logout, since GET requests are exempt from the
/// CSRF middleware.
#[tokio::test]
async fn logout_get_is_not_routed() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/logout")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn logout_post_without_csrf_is_forbidden() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post_without_csrf("/logout", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn spend_filter_query_is_forwarded_to_proxy() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("model", "gpt-4o");
        then.status(200).body(r#"{"data":[],"next_cursor":null}"#);
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_get("/spend?model=gpt-4o"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
    let html = body_string(resp).await;
    assert!(html.contains("No spend records") || html.contains("spend-panel"));
}

#[tokio::test]
async fn spend_page_clamps_negative_offset() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("offset", "0");
        then.status(200).body(r#"{"data":[],"next_cursor":null}"#);
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/spend?offset=-5")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

#[test]
fn with_default_date_range_fills_either_missing_side() {
    let today = today_iso_date();
    // Only Start provided: End must be filled so the export link is complete.
    let start = days_before_iso(&today, 10).unwrap();
    let q = SpendQuery {
        start_date: Some(start.clone()),
        ..Default::default()
    }
    .with_default_date_range(30);
    assert!(validate_export_date_range(&q).is_ok());
    assert_eq!(q.end_date.as_deref(), Some(today.as_str()));

    // Only End provided: Start must be filled.
    let q = SpendQuery {
        end_date: Some("2024-01-31".into()),
        ..Default::default()
    }
    .with_default_date_range(30);
    assert!(validate_export_date_range(&q).is_ok());
    assert_eq!(q.start_date.as_deref(), Some("2024-01-02"));
}

#[tokio::test]
async fn models_page_lists_enriched_models() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/model/list");
        then.status(200)
            .body(r#"{"data":[{"id":"gpt-4o","provider":"openai"}]}"#);
    });
    server.mock(|when, then| {
 when.method(httpmock::Method::GET)
 .path("/model/info")
 .query_param("model", "gpt-4o");
 then.status(200).body(
 r#"{"model":"gpt-4o","provider":"openai","max_input_tokens":128000,"max_output_tokens":16384,"input_cost_per_token":0.000005,"output_cost_per_token":0.000015}"#);
 });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("gpt-4o"));
    assert!(html.contains("128000"));
}

/// Model delete must send the proxy's DB `model_id` (the old `{"id": …}`
/// body 422'd on every delete), and the URL the template generates must be
/// percent-encoded so an id containing `/` cannot split the route path.
#[tokio::test]
async fn models_delete_posts_proxy_model_id() {
    let server = MockServer::start();
    let delete_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/model/delete")
            .json_body_partial(r#"{"model_id":"model-abc/123"}"#);
        then.status(200)
            .body(r#"{"model_id":"model-abc/123","status":"deleted"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/model/list");
        then.status(200)
            .body(r#"{"data":[{"id":"db-model","model_id":"model-abc/123","provider":"openai"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/models/model-abc%2F123/delete", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    delete_mock.assert();
    let html = body_string(resp).await;
    assert!(
        html.contains("removed"),
        "expected removal flash, got: {html}"
    );
}

/// Only DB-backed models (those carrying a `model_id` from `/model/list`)
/// get a delete button; YAML-configured models cannot be deleted through the
/// management API and must not offer a button that would always fail.
#[tokio::test]
async fn models_page_hides_delete_for_config_managed_models() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/model/list");
        then.status(200)
            .body(r#"{"data":[{"id":"yaml-model","provider":"openai"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/models")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("yaml-model"));
    assert!(
        !html.contains("/delete"),
        "config-managed model must not render a delete action: {html}"
    );
}

fn cookie_value_of<'a>(header: &'a str, name: &str) -> Option<&'a str> {
    for part in header.split(';') {
        let trimmed = part.trim();
        if let Some(rest) = trimmed.strip_prefix(&format!("{name}=")) {
            return Some(rest);
        }
    }
    None
}

#[tokio::test]
async fn login_with_correct_password_sets_session_and_redirects() {
    let server = MockServer::start();
    let config = AdminConfig {
        listen: "127.0.0.1:0".to_string(),
        proxy_url: server.base_url(),
        master_key: "sk-test".to_string(),
        session_secret: "unit-test-secret-value".to_string(),
        session_ttl_hours: 1,
        login_password: Some("hunter2".to_string()),
        trusted_proxies: Vec::new(),
        cookie_secure: false,
        login_max_fails: 5,
        login_window_secs: 60,
        login_block_secs: 300,
    };
    let state = AppState {
        config: Arc::new(config),
        proxy: Arc::new(ProxyClient::new(&server.base_url(), "sk-test")),
        templates: Arc::new(Templates::new()),
        login_brake: default_brake(),
    };
    let app = build(state);
    let (csrf, csrf_cookie) = login_csrf_pair();
    let body = format!("password=hunter2&csrf_token={csrf}");
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/login")
                .header("cookie", &csrf_cookie)
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    assert_eq!(
        resp.headers().get("location").and_then(|v| v.to_str().ok()),
        Some("/dashboard")
    );
    let cookie = resp
        .headers()
        .get("set-cookie")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(cookie.contains("oxllm_admin_session="));
    assert!(cookie.contains("HttpOnly"));
    assert!(cookie.contains("SameSite=Strict"));

    let value = cookie_value_of(cookie, SESSION_COOKIE).unwrap();
    assert!(crate::auth::verify_session("unit-test-secret-value", value));
}

#[tokio::test]
async fn login_with_secure_cookie_emits_secure_attribute() {
    let server = MockServer::start();
    let config = AdminConfig {
        listen: "127.0.0.1:0".to_string(),
        proxy_url: server.base_url(),
        master_key: "sk-test".to_string(),
        session_secret: "unit-test-secret-value".to_string(),
        session_ttl_hours: 1,
        login_password: Some("hunter2".to_string()),
        trusted_proxies: Vec::new(),
        cookie_secure: true,
        login_max_fails: 5,
        login_window_secs: 60,
        login_block_secs: 300,
    };
    let state = AppState {
        config: Arc::new(config),
        proxy: Arc::new(ProxyClient::new(&server.base_url(), "sk-test")),
        templates: Arc::new(Templates::new()),
        login_brake: default_brake(),
    };
    let app = build(state);
    let (csrf, csrf_cookie) = login_csrf_pair();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/login")
                .header("cookie", &csrf_cookie)
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from(format!("password=hunter2&csrf_token={csrf}")))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    let cookie = resp
        .headers()
        .get("set-cookie")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(cookie.contains("Secure"), "cookie must carry Secure flag");
}

#[tokio::test]
async fn login_without_pre_login_csrf_token_is_rejected() {
    let server = MockServer::start();
    let config = AdminConfig {
        listen: "127.0.0.1:0".to_string(),
        proxy_url: server.base_url(),
        master_key: "sk-test".to_string(),
        session_secret: "unit-test-secret-value".to_string(),
        session_ttl_hours: 1,
        login_password: Some("hunter2".to_string()),
        trusted_proxies: Vec::new(),
        cookie_secure: false,
        login_max_fails: 5,
        login_window_secs: 60,
        login_block_secs: 300,
    };
    let state = AppState {
        config: Arc::new(config),
        proxy: Arc::new(ProxyClient::new(&server.base_url(), "sk-test")),
        templates: Arc::new(Templates::new()),
        login_brake: default_brake(),
    };
    let app = build(state);
    // Correct password but no pre-login CSRF cookie/field: must NOT log in
    // (blocks cross-site forced login).
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/login")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("password=hunter2"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert!(
        resp.headers().get("location").is_none(),
        "must not redirect to /dashboard without a CSRF token"
    );
}

#[tokio::test]
async fn keys_update_with_non_numeric_budget_returns_error_not_success() {
    let server = MockServer::start();
    let update_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/key/update");
        then.status(200).body("{}");
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/keys/key-1/update",
            "key_alias=a&max_budget=not-a-number",
        ))
        .await
        .unwrap();
    // The invalid value must be surfaced, not silently dropped while the
    // proxy keeps the old budget and the UI reports "Key updated.".
    let body = body_string(resp).await;
    assert!(
        body.contains("max_budget"),
        "expected a max_budget validation error, got: {body}"
    );
    assert_eq!(
        update_mock.hits(),
        0,
        "invalid form must not reach the proxy"
    );
}

/// Blank fields in the key edit drawer mean "clear". The proxy treats
/// null/omitted as leave-unchanged, so the UI must send `""` for association
/// ids and the explicit `clear_*` flags for caps — otherwise it reports
/// "Key updated." while the old caps persist.
#[tokio::test]
async fn keys_update_sends_clear_contract_not_nulls() {
    let server = MockServer::start();
    let update_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/update")
            .json_body_partial(
                r#"{"key_id":"key-1","team_id":"","user_id":"","org_id":"","project_id":"",
                    "clear_max_budget":true,"clear_rpm":true,"clear_tpm":true}"#,
            );
        then.status(200).body("{}");
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/keys/key-1/update",
            "key_alias=a&team_id=&user_id=&org_id=&project_id=&max_budget=&rpm=&tpm=&budget_duration=&models=",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    update_mock.assert();
    let html = body_string(resp).await;
    assert!(
        html.contains("Key updated."),
        "expected success flash, got: {html}"
    );
}

/// Same contract for teams: blanked caps/budget must clear via flags, and a
/// blank budget_duration must be sent as `""` (omission leaves it unchanged).
#[tokio::test]
async fn teams_update_sends_clear_contract_not_nulls() {
    let server = MockServer::start();
    let update_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/team/update")
            .json_body_partial(
                r#"{"team_id":"team-1","clear_max_budget":true,"clear_rpm":true,
                    "clear_tpm":true,"budget_duration":""}"#,
            );
        then.status(200).body("{}");
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200).body(r#"{"teams":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/teams/team-1/update",
            "team_name=T&max_budget=&rpm=&tpm=&budget_duration=&models=",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    update_mock.assert();
}

/// Blank associations on a project must clear via the proxy's
/// `clear_org_id`/`clear_team_id` flags, and every re-rendered projects
/// table must still carry the org/team feeds for the edit drawer's selects
/// (previously only the page render fetched them, so after any create /
/// update / delete swap the selects degraded to "(none)" and a subsequent
/// save silently stripped the association).
#[tokio::test]
async fn projects_update_clears_associations_and_feeds_selects() {
    let server = MockServer::start();
    let update_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/project/update")
            .json_body_partial(
                r#"{"project_id":"p-1","clear_org_id":true,"clear_team_id":true,
                    "clear_max_budget":true,"clear_rpm":true,"clear_tpm":true}"#,
            );
        then.status(200).body("{}");
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/project/list");
        then.status(200)
            .body(r#"{"projects":[{"project_id":"p-1","project_name":"P"}]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/organization/list");
        then.status(200)
            .body(r#"{"organizations":[{"org_id":"org-1","org_name":"Acme"}]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200)
            .body(r#"{"teams":[{"team_id":"team-1","team_name":"Platform"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/projects/p-1/update",
            "project_name=P&org_id=&team_id=&max_budget=&rpm=&tpm=&budget_duration=",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    update_mock.assert();
    let html = body_string(resp).await;
    assert!(
        html.contains("Acme") && html.contains("Platform"),
        "re-rendered table must populate org/team selects: {html}"
    );
}

#[tokio::test]
async fn csrf_protected_post_without_token_is_forbidden() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/key/delete");
        then.status(200).body(r#"{"status":"deleted"}"#);
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post_without_csrf("/keys/key-9/delete", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn csrf_protected_post_with_correct_token_succeeds() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/key/delete");
        then.status(200).body(r#"{"status":"deleted"}"#);
    });
    // The handler refreshes the keys table after the delete.
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[]}"#);
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/keys/key-9/delete", ""))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn login_rate_limit_blocks_after_threshold() {
    let server = MockServer::start();
    let brake = Arc::new(LoginBrake::new(3, 60, 300));
    let config = AdminConfig {
        listen: "127.0.0.1:0".to_string(),
        proxy_url: server.base_url(),
        master_key: "sk-test".to_string(),
        session_secret: "unit-test-secret-value".to_string(),
        session_ttl_hours: 1,
        login_password: Some("hunter2".to_string()),
        trusted_proxies: Vec::new(),
        cookie_secure: false,
        login_max_fails: 3,
        login_window_secs: 60,
        login_block_secs: 300,
    };
    let state = AppState {
        config: Arc::new(config),
        proxy: Arc::new(ProxyClient::new(&server.base_url(), "sk-test")),
        templates: Arc::new(Templates::new()),
        login_brake: brake,
    };
    let app = build(state);

    fn login_req(ip: &str) -> Request<Body> {
        let (csrf, csrf_cookie) = login_csrf_pair();
        let mut req = Request::builder()
            .method("POST")
            .uri("/login")
            .header("cookie", &csrf_cookie)
            .header("content-type", "application/x-www-form-urlencoded")
            // Spoofed headers must be ignored — peer is authoritative.
            .header("x-real-ip", "9.9.9.9")
            .body(Body::from(format!("password=wrong&csrf_token={csrf}")))
            .unwrap();
        let addr: std::net::SocketAddr = format!("{ip}:443").parse().unwrap();
        req.extensions_mut()
            .insert(axum::extract::ConnectInfo(addr));
        req
    }

    for _ in 0..3 {
        let resp = app.clone().oneshot(login_req("203.0.113.7")).await.unwrap();
        assert!(
            resp.status() == StatusCode::UNAUTHORIZED
                || resp.status() == StatusCode::TOO_MANY_REQUESTS,
            "first failures should be 401/429, got {}",
            resp.status()
        );
    }

    let resp = app.clone().oneshot(login_req("203.0.113.7")).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "after threshold the IP must be blocked"
    );
    let retry = resp
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        retry.parse::<u64>().unwrap_or(0) > 0,
        "Retry-After must be a positive number"
    );

    let other = app.oneshot(login_req("198.51.100.42")).await.unwrap();
    assert_eq!(
        other.status(),
        StatusCode::UNAUTHORIZED,
        "other IPs must not be affected"
    );
}

#[tokio::test]
async fn calculator_run_handles_token_overflow_without_panic() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/spend/calculate");
        then.status(200).body(r#"{"total_cost":0.0}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/model/list");
        then.status(200).body(r#"{"data":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/calculator/run",
            // prompt_tokens = u64::MAX; without saturating_add this panics the
            // handler in debug builds on the arithmetic overflow check.
            "model=gpt-4o&prompt_tokens=18446744073709551615&completion_tokens=1",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

#[tokio::test]
async fn dashboard_reports_degraded_when_readiness_is_broken() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/global/spend");
        then.status(200)
            .body(r#"{"total_cost":0.0,"total_requests":0}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/global/activity");
        then.status(200).body(r#"{"results":[],"days":30}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/health");
        then.status(200).body(r#"{"status":"healthy"}"#);
    });
    // Readiness returns 503 with a non-JSON body, as a fronting proxy might
    // emit when the proxy's backends (Redis/DB) are down.
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/health/readiness");
        then.status(503).body("<html>Bad Gateway</html>");
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_htmx_get("/dashboard")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(
        !html.contains("Online"),
        "broken readiness must not render Online: {html}"
    );
    assert!(
        html.contains("Degraded") || html.contains("degraded"),
        "expected a degraded rendering, got: {html}"
    );
}

#[tokio::test]
async fn dashboard_runs_proxy_calls_concurrently() {
    // Each of the four proxy endpoints has a 60ms delay. Serial fetch would
    // take ~4x60ms=240ms; concurrent (tokio::join!) should be ~60ms.
    let server = MockServer::start();
    let per_call = std::time::Duration::from_millis(60);
    for (path, body) in [
        ("/global/spend", r#"{"total_cost":0.0,"total_requests":0}"#),
        ("/global/activity", r#"{"results":[],"days":30}"#),
        ("/health", r#"{"status":"healthy"}"#),
        ("/health/readiness", r#"{"status":"ok"}"#),
    ] {
        let body = body.to_string();
        server.mock(move |when, then| {
            when.method(httpmock::Method::GET).path(path);
            then.status(200).delay(per_call).body(body);
        });
    }

    let app = build(test_state_with_proxy(&server));
    let start = std::time::Instant::now();
    let resp = app.oneshot(authed_htmx_get("/dashboard")).await.unwrap();
    let elapsed = start.elapsed();
    assert_eq!(resp.status(), StatusCode::OK);
    let serial_bound = per_call * 4;
    assert!(
        elapsed < serial_bound * 3 / 4,
        "dashboard took {elapsed:?} — looks serial (bound {serial_bound:?})"
    );
}

#[test]
fn parse_iso_date_rejects_invalid_calendar_dates() {
    assert!(parse_iso_date_to_unix_days("2024-02-30").is_none());
    assert!(parse_iso_date_to_unix_days("2024-13-01").is_none());
    assert!(parse_iso_date_to_unix_days("2024-00-01").is_none());
    assert!(parse_iso_date_to_unix_days("2024-04-31").is_none());
    assert!(parse_iso_date_to_unix_days("2023-02-29").is_none());
    assert!(parse_iso_date_to_unix_days("2024-02-00").is_none());
    // Valid dates (including leap day) still parse.
    assert!(parse_iso_date_to_unix_days("2024-02-29").is_some());
    assert!(parse_iso_date_to_unix_days("2024-01-01").is_some());
    assert!(parse_iso_date_to_unix_days("2024-12-31").is_some());
}

#[tokio::test]
async fn keys_create_drops_negative_or_non_finite_budget_and_rate_limits() {
    // For each bad input, a leak-detector mock (matched first) fires only if
    // the distinctive bad value reaches the proxy body. Since the handler
    // drops invalid values, the leak mock must never hit.
    let cases: &[(&str, &str)] = &[
        ("key_alias=x&max_budget=-5", "-5"),
        ("key_alias=x&max_budget=NaN", "NaN"),
        ("key_alias=x&max_budget=inf", "inf"),
        ("key_alias=x&max_budget=Infinity", "Infinity"),
        ("key_alias=x&rpm=-10", "-10"),
        ("key_alias=x&tpm=-3", "-3"),
    ];
    for (form, needle) in cases {
        let server = MockServer::start();
        let leak = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/key/generate")
                .body_contains(*needle);
            then.status(200).body(r#"{"leaked":true}"#);
        });
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/key/generate")
                .header("authorization", "Bearer sk-test");
            then.status(200).body(r#"{"key":"sk-ox-1","key_id":"k1"}"#);
        });

        let app = build(test_state_with_proxy(&server));
        let resp = app
            .oneshot(authed_post("/keys/create", form))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "bad input {form:?} should be a full-page 400 error, not a 200"
        );
        let html = body_string(resp).await;
        assert!(
            !html.contains("leaked"),
            "value {needle:?} from {form:?} was forwarded to the proxy"
        );
        assert_eq!(leak.hits(), 0, "negative value leaked for {form:?}");
    }
}

#[tokio::test]
async fn enrich_with_budget_spend_fans_out_across_many_rows() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/total");
        then.status(200).body(r#"{"total_spend": 1.0}"#);
    });
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows: Vec<Value> = (0..12)
        .map(|i| json!({"key_id": format!("k{i}"), "max_budget": 10.0}))
        .collect();
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    m.assert_hits(12);
    for r in &rows {
        assert_eq!(r["budget_spend"], json!(1.0));
        assert_eq!(r["budget_spend_pct"], json!(10.0));
        assert_eq!(r["budget_spend_tier"], json!("ok"));
    }
}

#[tokio::test]
async fn keys_update_clears_rpm_tpm_on_empty_input() {
    // The proxy treats null/omitted rpm/tpm as leave-unchanged; clearing a
    // cap requires the explicit clear_rpm/clear_tpm flags. A blanked field
    // in the edit drawer means "clear", not "keep".
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/update")
            .json_body_partial(r#"{"key_id":"key-1","clear_rpm":true,"clear_tpm":true}"#);
        then.status(200).body(r#"{"key_id":"key-1"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-1","key_alias":"x"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post("/keys/key-1/update", "key_alias=x&rpm=&tpm="))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

#[tokio::test]
async fn keys_update_round_trips_nonempty_rpm_tpm() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/update")
            .json_body_partial(r#"{"key_id":"key-1","rpm":120,"tpm":240}"#);
        then.status(200).body(r#"{"key_id":"key-1"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-1","key_alias":"x"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/keys/key-1/update",
            "key_alias=x&rpm=120&tpm=240",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

/// CSV export pagination must honour an overall wall-clock deadline so a wide
/// date range can't tie up a worker for many minutes. The proxy returns
/// non-empty pages slowly with an always-advancing cursor; with a short
/// deadline the loop must stop and surface the truncation marker rather than
/// grind toward `MAX_ROWS`.
#[tokio::test]
async fn spend_export_csv_fetch_stops_at_overall_deadline() {
    use crate::pages::spend::{ExportFetch, fetch_spend_pages};
    use std::time::Duration;

    let server = MockServer::start();
    // Set up a chain of pages with strictly-advancing cursors so the
    // cursor-equality stall guard never trips — without the deadline the loop
    // would paginate until MAX_ROWS (10_000), i.e. ~10 minutes at this rate.
    // Each page has a 60 ms delay; with a 200 ms deadline we expect ~3 pages
    // before the deadline trips.
    // First page: no cursor yet.
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("cursor", "1");
        then.status(200)
            .delay(Duration::from_millis(60))
            .body(r#"{"data":[{"id":2,"model":"gpt-4o","cost":0.01}],"next_cursor":3}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("cursor", "3");
        then.status(200)
            .delay(Duration::from_millis(60))
            .body(r#"{"data":[{"id":4,"model":"gpt-4o","cost":0.01}],"next_cursor":5}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("cursor", "5");
        then.status(200)
            .delay(Duration::from_millis(60))
            .body(r#"{"data":[{"id":6,"model":"gpt-4o","cost":0.01}],"next_cursor":7}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/logs/v2")
            .query_param("cursor", "7");
        then.status(200)
            .delay(Duration::from_millis(60))
            .body(r#"{"data":[{"id":8,"model":"gpt-4o","cost":0.01}],"next_cursor":9}"#);
    });
    // First page (no cursor param) is the fall-through: define it last so
    // httpmock evaluates the cursor-specific mocks first.
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/logs/v2");
        then.status(200)
            .delay(Duration::from_millis(60))
            .body(r#"{"data":[{"id":0,"model":"gpt-4o","cost":0.01}],"next_cursor":1}"#);
    });

    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let q = crate::pages::spend::SpendQuery {
        start_date: Some("2024-01-01".to_string()),
        end_date: Some("2024-01-31".to_string()),
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let ExportFetch {
        rows,
        truncated_by_deadline,
        ..
    } = fetch_spend_pages(
        &proxy,
        &q,
        "2024-01-31",
        200,
        10_000,
        Duration::from_millis(200),
    )
    .await
    .expect("fetch should not error");
    let elapsed = started.elapsed();

    assert!(truncated_by_deadline, "expected deadline to trip");
    assert!(!rows.is_empty(), "should have collected at least one page");
    assert!(
        elapsed < Duration::from_secs(2),
        "fetch ran {elapsed:?} — deadline did not bound the loop"
    );
}

#[tokio::test]
async fn spend_export_csv_fetch_flags_row_cap_truncation() {
    use crate::pages::spend::fetch_spend_pages;
    use std::time::Duration;

    let server = MockServer::start();
    // Page of 2 rows with an advancing cursor; a max_rows of 1 trips the cap
    // while more pages remain.
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/logs/v2");
        then.status(200)
            .body(r#"{"data":[{"id":1,"model":"a","cost":0.01},{"id":2,"model":"b","cost":0.02}],"next_cursor":3}"#);
    });
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let q = SpendQuery {
        start_date: Some("2024-01-01".to_string()),
        end_date: Some("2024-01-31".to_string()),
        ..Default::default()
    };
    let f = fetch_spend_pages(&proxy, &q, "2024-01-31", 200, 1, Duration::from_secs(5))
        .await
        .expect("fetch should not error");
    assert_eq!(f.rows.len(), 1);
    assert!(!f.truncated_by_deadline);
    assert!(f.truncated_by_row_cap, "row-cap truncation must be flagged");
}

/// MEDIUM #1: `enrich_with_budget_spend` must derive the `/spend/total`
/// `start_date` from each entity's `budget_duration` so the bar reflects the
/// same window the proxy enforces. A `1d` budget queries today only; a `7d`
/// budget queries today-6 (inclusive 7-day window). The previous code hard-
/// coded a 30-day window for every entity, which made daily budgets look
/// wildly over-utilized.
#[tokio::test]
async fn enrich_with_budget_spend_window_follows_budget_duration() {
    let today = today_iso_date();
    let six_days_ago = days_before_iso(&today, 6).unwrap_or_else(|| today.clone());

    let server = MockServer::start();
    let today_for_daily = today.clone();
    let m_daily = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("key_id", "k1")
            .query_param("start_date", today_for_daily.clone());
        then.status(200).body(r#"{"total_spend": 5.0}"#);
    });
    let six_days_ago_clone = six_days_ago.clone();
    let m_weekly = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("key_id", "k2")
            .query_param("start_date", six_days_ago_clone.clone());
        then.status(200).body(r#"{"total_spend": 7.0}"#);
    });
    // A 30d budget should still hit the historical 30-day window so the
    // common case is unchanged.
    let thirty_days_ago = days_before_iso(&today, 29).unwrap_or_else(|| today.clone());
    let thirty_days_ago_clone = thirty_days_ago.clone();
    let m_monthly = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("key_id", "k3")
            .query_param("start_date", thirty_days_ago_clone.clone());
        then.status(200).body(r#"{"total_spend": 30.0}"#);
    });

    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![
        json!({"key_id": "k1", "max_budget": 10.0, "budget_duration": "1d"}),
        json!({"key_id": "k2", "max_budget": 10.0, "budget_duration": "7d"}),
        json!({"key_id": "k3", "max_budget": 10.0, "budget_duration": "30d"}),
    ];
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    m_daily.assert();
    m_weekly.assert();
    m_monthly.assert();
}

/// LOW: the `daily` word alias must map to `24h` (LiteLLM's
/// `_BUDGET_DURATION_WORD_ALIASES`) so the budget window matches the proxy's
/// scheduler and LiteLLM's duration parser. Both `daily` and `24h` collapse
/// to a sub-day window (today only); `hourly` and `weekly` round out the
/// alias set.
#[tokio::test]
async fn budget_window_word_aliases_match_canonical_durations() {
    let today = today_iso_date();

    let server = MockServer::start();
    let today_clone = today.clone();
    let m_daily = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("key_id", "k-daily")
            .query_param("start_date", today_clone.clone());
        then.status(200).body(r#"{"total_spend": 1.0}"#);
    });
    let today_clone = today.clone();
    let m_24h = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("key_id", "k-24h")
            .query_param("start_date", today_clone.clone());
        then.status(200).body(r#"{"total_spend": 1.0}"#);
    });
    let today_clone = today.clone();
    let m_hourly = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("key_id", "k-hourly")
            .query_param("start_date", today_clone.clone());
        then.status(200).body(r#"{"total_spend": 1.0}"#);
    });

    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![
        json!({"key_id": "k-daily", "max_budget": 10.0, "budget_duration": "daily"}),
        json!({"key_id": "k-24h", "max_budget": 10.0, "budget_duration": "24h"}),
        json!({"key_id": "k-hourly", "max_budget": 10.0, "budget_duration": "hourly"}),
    ];
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    // `daily` must hit the same (today) window as `24h`, not be mis-mapped.
    m_daily.assert();
    m_24h.assert();
    m_hourly.assert();
}

/// MEDIUM #1 (cont.): a missing/unparseable `budget_duration` must fall back
/// to the historical 30-day window so existing rows render unchanged.
#[tokio::test]
async fn enrich_with_budget_spend_falls_back_to_30d_when_duration_missing() {
    let today = today_iso_date();
    let thirty_days_ago = days_before_iso(&today, 29).unwrap_or_else(|| today.clone());

    let server = MockServer::start();
    let thirty_days_ago_clone = thirty_days_ago.clone();
    let m = server.mock(move |when, then| {
        when.method(httpmock::Method::GET)
            .path("/spend/total")
            .query_param("start_date", thirty_days_ago_clone.clone());
        then.status(200).body(r#"{"total_spend": 1.0}"#);
    });

    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows = vec![
        json!({"key_id": "k-missing", "max_budget": 10.0}),
        json!({"key_id": "k-empty", "max_budget": 10.0, "budget_duration": ""}),
        json!({"key_id": "k-garbage", "max_budget": 10.0, "budget_duration": "soon"}),
    ];
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    // All three rows should use the 30-day fallback start_date.
    m.assert_hits(3);
}

/// MEDIUM #3: `enrich_with_budget_spend` must cap the number of enriched rows
/// (matching `enrich_models`) so a deployment with many budget-bearing keys
/// cannot tie up a worker for minutes on every page render. Rows past the cap
/// render without `budget_spend`.
#[tokio::test]
async fn enrich_with_budget_spend_caps_enriched_row_count() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/total");
        then.status(200).body(r#"{"total_spend": 1.0}"#);
    });
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    // 300 budget-bearing rows; only the first 200 should be enriched.
    let mut rows: Vec<Value> = (0..300)
        .map(|i| json!({"key_id": format!("k{i}"), "max_budget": 10.0}))
        .collect();
    enrich_with_budget_spend(&proxy, &mut rows, "key_id").await;
    m.assert_hits(200);
    let enriched = rows
        .iter()
        .filter(|r| r.get("budget_spend").is_some())
        .count();
    assert_eq!(
        enriched, 200,
        "exactly ENRICHMENT_CAP rows should be enriched"
    );
}

/// MEDIUM #2: sort/pagination links must URL-encode filter values so a model
/// filter containing `&`, `=`, or spaces does not inject extra query params or
/// produce a malformed URL. The template receives a pre-encoded `filters_qs`.
#[test]
fn spend_load_more_encodes_filter_values_in_url() {
    let t = Templates::new();
    let html = t
        .render(
            "partials/spend_load_more.html",
            &serde_json::json!({
                "next_cursor": 42,
                "filters_qs": "model=a%26b%20c",
                "sort_by": "timestamp",
                "sort_order": "desc",
            }),
        )
        .expect("render");
    assert!(
        html.contains("model=a%26b%20c"),
        "expected URL-encoded filter value in hx-get, got: {html}"
    );
    assert!(
        !html.contains("model=a&b c"),
        "raw unencoded filter value leaked into URL: {html}"
    );
}

/// MEDIUM #2 (cont.): the sort header links must also use the encoded
/// `filters_qs` rather than interpolating raw filter values.
#[test]
fn spend_table_sort_links_use_encoded_filters_qs() {
    let t = Templates::new();
    let html = t
        .render(
            "partials/spend_table.html",
            &serde_json::json!({
                "logs": [{"timestamp": "2024-01-01", "model": "gpt-4o", "cost": 0.01}],
                "filters": {"model": "a&b c"},
                "filters_qs": "model=a%26b%20c",
                "sort_by": "timestamp",
                "sort_order": "desc",
                "list_query": "sort_by=timestamp&sort_order=desc",
                "export_query": "sort_by=timestamp&sort_order=desc",
            }),
        )
        .expect("render");
    // The encoded form must appear in every sort header href/hx-get.
    assert!(
        html.contains("model=a%26b%20c"),
        "expected encoded filter in sort links, got: {html}"
    );
    // The raw value must not leak into any URL (it's still fine inside the
    // <input> value attribute where HTML-escaping protects it).
    let href_count = html.matches("model=a&b c").count();
    assert_eq!(href_count, 0, "raw filter value leaked into a URL: {html}");
}

/// LOW #4: the Alpine `@click` edit-button binding must not interpolate entity
/// IDs into a JavaScript string literal. The templates now use the numeric
/// `loop.index` so a quote-bearing ID can never break out of the JS
/// expression. Render a key whose ID contains a single quote and assert the
/// `@click` value contains only the numeric index (no raw ID, no quote inside
/// a JS string literal).
#[test]
fn keys_table_edit_click_does_not_interpolate_id_into_js() {
    let t = Templates::new();
    let html = t
        .render(
            "partials/keys_table.html",
            &serde_json::json!({
                "keys": [{
                    "key_id": "k'evil",
                    "key_alias": "alpha",
                    "is_active": true,
                }],
            }),
        )
        .expect("render");
    // The @click must use the numeric loop index, not the raw ID.
    assert!(
        html.contains("@click=\"editing = 1;"),
        "expected numeric loop.index in @click, got: {html}"
    );
    // The raw quote-bearing ID must not appear inside a JS string literal
    // in any @click / x-if expression. (It may still appear elsewhere in the
    // HTML body, where auto-escaping protects it.)
    assert!(
        !html.contains("editing = 'k'evil"),
        "raw ID interpolated into JS string literal: {html}"
    );
    assert!(
        !html.contains("editing === 'k'evil"),
        "raw ID interpolated into x-if JS string literal: {html}"
    );
}

/// LOW #5: `keys_update` must send `""` for an empty `team_id` (and
/// `user_id`/`org_id`/`project_id`) so the proxy clears the association:
/// the proxy's update contract treats null/omitted as leave-unchanged and
/// only an explicit empty string clears a string association id.
#[tokio::test]
async fn keys_update_clears_team_user_org_project_on_empty_input() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/update")
            .json_body_partial(
                r#"{"key_id":"key-1","team_id":"","user_id":"","org_id":"","project_id":""}"#,
            );
        then.status(200).body(r#"{"key_id":"key-1"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-1","key_alias":"x"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/keys/key-1/update",
            "key_alias=x&team_id=&user_id=&org_id=&project_id=",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

/// LOW #5 (cont.): `projects_update` must clear `team_id`/`org_id` via the
/// proxy's explicit `clear_team_id`/`clear_org_id` flags — null/omitted
/// leaves the association unchanged.
#[tokio::test]
async fn projects_update_clears_team_org_on_empty_input() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/project/update")
            .json_body_partial(r#"{"project_id":"p1","clear_team_id":true,"clear_org_id":true}"#);
        then.status(200).body(r#"{"project_id":"p1"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/project/list");
        then.status(200).body(r#"{"projects":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/projects/p1/update",
            "project_name=untitled&org_id=&team_id=",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

/// LOW #6: `keys_update` must accept and forward `org_id`/`project_id` so an
/// operator can move a key between orgs/projects without deleting and
/// recreating it. Previously these fields existed on the create form but were
/// absent from the edit form / `KeyUpdateForm`.
#[tokio::test]
async fn keys_update_forwards_org_and_project_ids() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/key/update")
            .json_body_partial(r#"{"key_id":"key-1","org_id":"org-2","project_id":"proj-3"}"#);
        then.status(200).body(r#"{"key_id":"key-1"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200)
            .body(r#"{"keys":[{"key_id":"key-1","key_alias":"x"}]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/keys/key-1/update",
            "key_alias=x&org_id=org-2&project_id=proj-3",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

/// LOW #7: non-numeric token input must surface an inline error rather than
/// silently coercing to 0 (which would render a misleading "$0.00000000"
/// estimate). The handler now validates both fields up front.
#[tokio::test]
async fn calculator_run_rejects_non_numeric_token_input() {
    let server = MockServer::start();
    // If the handler silently coerced to 0, it would call /spend/calculate;
    // the leak mock must never hit.
    let leak = server.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/spend/calculate");
        then.status(200).body(r#"{"total_cost":0.0}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/model/list");
        then.status(200).body(r#"{"data":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/calculator/run",
            "model=gpt-4o&prompt_tokens=abc&completion_tokens=500",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(
        html.contains("Invalid input") || html.contains("non-negative"),
        "expected an inline error for non-numeric tokens, got: {html}"
    );
    assert_eq!(
        leak.hits(),
        0,
        "handler must not call /spend/calculate on invalid input"
    );
}

/// LOW #8: CSV export is triggered via `window.location = ...` (a full-page
/// navigation, not an HTMX swap), so an `HX-Redirect` would be silently
/// ignored on a 401 and leave the user on a blank page. The handler must
/// return a real HTTP redirect (3xx with `Location`) so the browser lands on
/// /login.
#[tokio::test]
async fn spend_export_csv_returns_http_redirect_on_401() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/logs/v2");
        then.status(401).body(r#"{"error":"unauthorized"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(200).body(r#"{"keys":[]}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200).body(r#"{"teams":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_get(
            "/spend/export.csv?start_date=2024-01-01&end_date=2024-01-31",
        ))
        .await
        .unwrap();
    // Must be a real 3xx redirect, NOT a 200 with HX-Redirect.
    assert!(
        resp.status().is_redirection(),
        "expected 3xx redirect on 401, got {}",
        resp.status()
    );
    assert_eq!(
        resp.headers().get("location").and_then(|v| v.to_str().ok()),
        Some("/login"),
        "expected Location: /login"
    );
    assert!(
        resp.headers().get("HX-Redirect").is_none(),
        "CSV export must not use HX-Redirect (not honoured by full navigations)"
    );
}

/// LOW #9: the model create form's `api_key` input must be rendered as
/// `type="password"` (with a show/hide toggle) so the upstream provider
/// credential is not shoulder-surfable while the form is open.
#[test]
fn models_form_renders_api_key_as_password() {
    let t = Templates::new();
    let html = t
        .render(
            "pages/models.html",
            &serde_json::json!({"session_active": true, "models": []}),
        )
        .expect("render");
    assert!(
        html.contains("name=\"api_key\""),
        "api_key input missing: {html}"
    );
    // The api_key input must use a password-type binding, not a plain
    // type="text". The Alpine :type toggle satisfies this.
    assert!(
        html.contains(":type=\"showKey ? 'text' : 'password'\""),
        "expected password-type binding on api_key, got: {html}"
    );
}

/// HIGH: the team edit form must include a `models` input pre-populated with
/// the team's current model restriction (as CSV). Without it, every edit
/// silently submits an empty `models` field and the update handler clears the
/// restriction. The keys edit form already does this; teams (and users) were
/// missed.
#[test]
fn teams_table_edit_form_includes_models_input_with_current_value() {
    let t = Templates::new();
    let html = t
        .render(
            "partials/teams_table.html",
            &serde_json::json!({
                "teams": [{
                    "team_id": "team-1",
                    "team_name": "Eng",
                    "models": "gpt-4o, claude-*",
                }],
            }),
        )
        .expect("render");
    assert!(
        html.contains("name=\"models\""),
        "team edit form is missing a models input: {html}"
    );
    assert!(
        html.contains("value=\"gpt-4o, claude-*\""),
        "team edit form models input must be pre-populated with the current CSV value: {html}"
    );
}

/// HIGH (cont.): the user edit form must include the same `models` input.
#[test]
fn users_table_edit_form_includes_models_input_with_current_value() {
    let t = Templates::new();
    let html = t
        .render(
            "partials/users_table.html",
            &serde_json::json!({
                "users": [{
                    "user_id": "u1",
                    "user_name": "Ada",
                    "models": "gpt-4o, claude-*",
                }],
            }),
        )
        .expect("render");
    assert!(
        html.contains("name=\"models\""),
        "user edit form is missing a models input: {html}"
    );
    assert!(
        html.contains("value=\"gpt-4o, claude-*\""),
        "user edit form models input must be pre-populated with the current CSV value: {html}"
    );
}

/// HIGH (cont.): `teams_update` must forward the submitted `models` list to the
/// proxy rather than clearing it. This mirrors the keys handler's contract. A
/// non-empty `models` form field must arrive at the proxy as a JSON array.
#[tokio::test]
async fn teams_update_forwards_nonempty_models() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/team/update")
            .json_body_partial(r#"{"team_id":"team-1","models":["gpt-4o","claude-*"]}"#);
        then.status(200).body(r#"{"team_id":"team-1"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200).body(r#"{"teams":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/teams/team-1/update",
            "team_name=Eng&models=gpt-4o%2C%20claude-*",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

/// HIGH (cont.): `users_update` must forward the submitted `models` list too.
#[tokio::test]
async fn users_update_forwards_nonempty_models() {
    let server = MockServer::start();
    let m = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/user/update")
            .json_body_partial(r#"{"user_id":"u1","models":["gpt-4o"]}"#);
        then.status(200).body(r#"{"user_id":"u1"}"#);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/user/list");
        then.status(200).body(r#"{"users":[]}"#);
    });

    let app = build(test_state_with_proxy(&server));
    let resp = app
        .oneshot(authed_post(
            "/users/u1/update",
            "user_name=Ada&models=gpt-4o",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    m.assert();
}

/// MEDIUM #1: `enrich_with_budget_spend` must bound the aggregate wall-clock
/// time of its `/spend/total` fan-out so a degraded proxy cannot tie up an
/// admin worker for `ceil(CAP / CONCURRENCY) * PROXY_REQUEST_TIMEOUT` (~6
/// min) during a page render. With a short deadline and a slow proxy, the
/// loop must stop early: only a subset of rows get `budget_spend`, and the
/// handler returns in well under the unbounded worst case.
#[tokio::test]
async fn enrich_with_budget_spend_stops_at_overall_deadline() {
    use std::time::{Duration, Instant};

    let server = MockServer::start();
    let n: usize = 32;
    let per_call = Duration::from_millis(500);
    for i in 0..n {
        server.mock(move |when, then| {
            when.method(httpmock::Method::GET)
                .path("/spend/total")
                .query_param("key_id", format!("k{i}"));
            then.status(200)
                .delay(per_call)
                .body(r#"{"total_spend": 1.0}"#);
        });
    }

    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let mut rows: Vec<Value> = (0..n)
        .map(|i| json!({"key_id": format!("k{i}"), "max_budget": 10.0}))
        .collect();

    let started = Instant::now();
    enrich_with_budget_spend_bounded(&proxy, &mut rows, "key_id", Duration::from_millis(150)).await;
    let elapsed = started.elapsed();

    let enriched = rows
        .iter()
        .filter(|r| r.get("budget_spend").is_some())
        .count();
    // The deadline (150 ms) is well under one batch delay (500 ms), so at
    // most the first batch's worth of results are pulled before the loop
    // breaks. Some rows must be left un-enriched.
    assert!(
        enriched >= 1 && enriched < n,
        "expected some-but-not-all rows enriched, got {enriched}/{n}"
    );
    // Unbounded worst case: ceil(32/8) * 500 ms = 2000 ms. The deadline must
    // keep the handler well clear of that.
    assert!(
        elapsed < Duration::from_millis(1500),
        "enrich ran {elapsed:?} — deadline did not bound the fan-out"
    );
}

/// MEDIUM #2: `keys_page` must fan its five independent list calls out
/// concurrently (tokio::join!) rather than awaiting them serially, so a slow
/// proxy can't multiply page latency by the call count. Mirrors
/// `dashboard_runs_proxy_calls_concurrently`.
#[tokio::test]
async fn keys_page_runs_proxy_calls_concurrently() {
    let server = MockServer::start();
    let per_call = std::time::Duration::from_millis(60);
    for (path, body) in [
        ("/key/list", r#"{"keys":[]}"#),
        ("/team/list", r#"{"teams":[]}"#),
        ("/user/list", r#"{"users":[]}"#),
        ("/organization/list", r#"{"organizations":[]}"#),
        ("/project/list", r#"{"projects":[]}"#),
    ] {
        let body = body.to_string();
        server.mock(move |when, then| {
            when.method(httpmock::Method::GET).path(path);
            then.status(200).delay(per_call).body(body);
        });
    }

    let app = build(test_state_with_proxy(&server));
    let start = std::time::Instant::now();
    let resp = app.oneshot(authed_htmx_get("/keys")).await.unwrap();
    let elapsed = start.elapsed();
    assert_eq!(resp.status(), StatusCode::OK);
    // 5 serial calls of 60 ms = 300 ms; concurrent should be ~60 ms.
    let serial_bound = per_call * 5;
    assert!(
        elapsed < serial_bound * 3 / 4,
        "keys_page took {elapsed:?} — looks serial (bound {serial_bound:?})"
    );
}

/// MEDIUM #2 (cont.): `spend_page` must run spend_logs + the two dropdown
/// feeds concurrently.
#[tokio::test]
async fn spend_page_runs_proxy_calls_concurrently() {
    let server = MockServer::start();
    let per_call = std::time::Duration::from_millis(60);
    for (path, body) in [
        ("/spend/logs/v2", r#"{"data":[],"total_cost":0.0}"#),
        ("/key/list", r#"{"keys":[]}"#),
        ("/team/list", r#"{"teams":[]}"#),
    ] {
        let body = body.to_string();
        server.mock(move |when, then| {
            when.method(httpmock::Method::GET).path(path);
            then.status(200).delay(per_call).body(body);
        });
    }

    let app = build(test_state_with_proxy(&server));
    let start = std::time::Instant::now();
    let resp = app.oneshot(authed_htmx_get("/spend")).await.unwrap();
    let elapsed = start.elapsed();
    assert_eq!(resp.status(), StatusCode::OK);
    let serial_bound = per_call * 3;
    assert!(
        elapsed < serial_bound * 3 / 4,
        "spend_page took {elapsed:?} — looks serial (bound {serial_bound:?})"
    );
}

/// MEDIUM #2 (cont.): `projects_page` must run its three list calls
/// concurrently.
#[tokio::test]
async fn projects_page_runs_proxy_calls_concurrently() {
    let server = MockServer::start();
    let per_call = std::time::Duration::from_millis(60);
    for (path, body) in [
        ("/project/list", r#"{"projects":[]}"#),
        ("/organization/list", r#"{"organizations":[]}"#),
        ("/team/list", r#"{"teams":[]}"#),
    ] {
        let body = body.to_string();
        server.mock(move |when, then| {
            when.method(httpmock::Method::GET).path(path);
            then.status(200).delay(per_call).body(body);
        });
    }

    let app = build(test_state_with_proxy(&server));
    let start = std::time::Instant::now();
    let resp = app.oneshot(authed_htmx_get("/projects")).await.unwrap();
    let elapsed = start.elapsed();
    assert_eq!(resp.status(), StatusCode::OK);
    let serial_bound = per_call * 3;
    assert!(
        elapsed < serial_bound * 3 / 4,
        "projects_page took {elapsed:?} — looks serial (bound {serial_bound:?})"
    );
}

/// HIGH: a proxy 401 during a non-HTMX browser navigation must return a real
/// 3xx redirect to /login — an HX-Redirect header alone would be ignored.
#[tokio::test]
async fn keys_page_proxy_401_non_htmx_returns_http_redirect() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(401).body(r#"{"error":"unauthorized"}"#);
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/keys")).await.unwrap();
    assert!(
        resp.status().is_redirection(),
        "expected 3xx on non-htmx 401, got {}",
        resp.status()
    );
    assert_eq!(
        resp.headers().get("location").and_then(|v| v.to_str().ok()),
        Some("/login")
    );
}

/// HIGH: full-page proxy failures must surface a non-200 status so
/// monitoring can see the breakage (HTMX fragments stay 200 to swap).
#[tokio::test]
async fn keys_page_proxy_500_non_htmx_returns_bad_gateway() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/key/list");
        then.status(500).body(r#"{"error":"boom"}"#);
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_get("/keys")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    let html = body_string(resp).await;
    assert!(html.contains("500"));
}

/// HIGH: a full batch without `next_cursor` and without integer row ids
/// must be flagged as truncated — otherwise the CSV export silently drops
/// the remaining pages.
#[tokio::test]
async fn spend_export_csv_flags_full_page_without_cursor() {
    use crate::pages::spend::fetch_spend_pages;
    use std::time::Duration;

    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/logs/v2");
        then.status(200)
            .body(r#"{"data":[{"key_id":"a","cost":0.01},{"key_id":"b","cost":0.02}]}"#);
    });
    let proxy = ProxyClient::new(&server.base_url(), "sk-test");
    let q = SpendQuery {
        start_date: Some("2024-01-01".to_string()),
        end_date: Some("2024-01-31".to_string()),
        ..Default::default()
    };
    let f = fetch_spend_pages(&proxy, &q, "2024-01-31", 2, 100, Duration::from_secs(5))
        .await
        .expect("fetch should not error");
    assert_eq!(f.rows.len(), 2);
    assert!(
        f.truncated_by_missing_cursor,
        "full page without a cursor must be flagged"
    );
    assert!(!f.truncated_by_row_cap);
}
