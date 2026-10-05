//! Behavioral regressions for the admin release review. No external services or
//! real accounts are used; all management traffic terminates in httpmock.

use super::*;

fn assert_private_response(resp: &Response) {
    let headers = resp.headers();
    assert_eq!(headers["cache-control"], "no-store, private");
    assert_eq!(headers["x-frame-options"], "DENY");
    assert_eq!(headers["x-content-type-options"], "nosniff");
    assert_eq!(headers["referrer-policy"], "no-referrer");
    let csp = headers["content-security-policy"].to_str().unwrap();
    assert!(csp.contains("frame-ancestors 'none'"));
    assert!(csp.contains("form-action 'self'"));
    // Standard Alpine must remain able to evaluate its attribute expressions.
    assert!(csp.contains("'unsafe-eval'"));
}

#[tokio::test]
async fn expired_missing_and_invalid_htmx_sessions_redirect_the_document() {
    let server = MockServer::start();
    let upstream = server.mock(|_, then| {
        then.status(500).body("unexpected proxy call");
    });
    let app = build(test_state_with_proxy(&server));
    let expired = create_session("unit-test-secret-value", 0);
    for cookie in [None, Some("invalid"), Some(expired.cookie_value.as_str())] {
        for (method, uri) in [("GET", "/teams"), ("POST", "/keys/create")] {
            let mut req = Request::builder()
                .method(method)
                .uri(uri)
                .header("HX-Request", "true")
                .header("X-CSRF-Token", "stale-token");
            if let Some(value) = cookie {
                req = req.header("cookie", format!("{SESSION_COOKIE}={value}"));
            }
            let resp = app
                .clone()
                .oneshot(req.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            assert_eq!(resp.headers()["HX-Redirect"], "/login");
            assert!(resp.headers().get("location").is_none());
            assert!(
                resp.headers()["set-cookie"]
                    .to_str()
                    .unwrap()
                    .contains("Max-Age=0")
            );
            assert_private_response(&resp);
            assert!(
                body_string(resp).await.is_empty(),
                "must not swap a login page"
            );
        }
    }
    upstream.assert_hits(0);
}

#[tokio::test]
async fn false_htmx_header_uses_normal_navigation_redirect() {
    let server = MockServer::start();
    let resp = build(test_state_with_proxy(&server))
        .oneshot(
            Request::builder()
                .uri("/keys")
                .header("HX-Request", "false")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SEE_OTHER);
    assert_eq!(resp.headers()["location"], "/login");
    assert!(resp.headers().get("HX-Redirect").is_none());
}

#[tokio::test]
async fn headers_cover_login_pages_fragments_redirects_and_errors() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200).json_body(json!({"teams": []}));
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/user/list");
        then.status(500).body("upstream unavailable");
    });
    let app = build(test_state_with_proxy(&server));
    for (req, status) in [
        (
            Request::builder()
                .uri("/login")
                .body(Body::empty())
                .unwrap(),
            StatusCode::OK,
        ),
        (authed_get("/teams"), StatusCode::OK),
        (authed_htmx_get("/teams"), StatusCode::OK),
        (
            Request::builder().uri("/keys").body(Body::empty()).unwrap(),
            StatusCode::SEE_OTHER,
        ),
        (
            authed_post_without_csrf("/keys/create", ""),
            StatusCode::FORBIDDEN,
        ),
        (authed_get("/users"), StatusCode::BAD_GATEWAY),
        (authed_get("/missing"), StatusCode::NOT_FOUND),
        (authed_get("/static/missing.js"), StatusCode::NOT_FOUND),
    ] {
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), status);
        assert_private_response(&resp);
    }
}

#[tokio::test]
async fn scripts_and_licenses_are_served_from_the_binary_without_a_session() {
    let server = MockServer::start();
    let app = build(test_state_with_proxy(&server));
    for (name, expected, content_type) in [
        (
            "htmx-2.0.4.min.js",
            include_bytes!("../../static/htmx-2.0.4.min.js").as_slice(),
            "text/javascript; charset=utf-8",
        ),
        (
            "alpine-3.14.8.min.js",
            include_bytes!("../../static/alpine-3.14.8.min.js").as_slice(),
            "text/javascript; charset=utf-8",
        ),
        (
            "htmx-2.0.4.LICENSE.txt",
            include_bytes!("../../static/htmx-2.0.4.LICENSE.txt").as_slice(),
            "text/plain; charset=utf-8",
        ),
        (
            "alpine-3.14.8.LICENSE.txt",
            include_bytes!("../../static/alpine-3.14.8.LICENSE.txt").as_slice(),
            "text/plain; charset=utf-8",
        ),
        (
            "vue-reactivity.LICENSE.txt",
            include_bytes!("../../static/vue-reactivity.LICENSE.txt").as_slice(),
            "text/plain; charset=utf-8",
        ),
        (
            "PROVENANCE.txt",
            include_bytes!("../../static/PROVENANCE.txt").as_slice(),
            "text/plain; charset=utf-8",
        ),
        (
            "logo.png",
            include_bytes!("../../static/logo.png").as_slice(),
            "image/png",
        ),
    ] {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/static/{name}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{name}");
        assert_eq!(resp.headers()["content-type"], content_type);
        assert_eq!(resp.headers()["x-content-type-options"], "nosniff");
        assert!(
            resp.headers()["cache-control"]
                .to_str()
                .unwrap()
                .starts_with("public,")
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), expected, "{name}");
    }
    let resp = app
        .oneshot(
            Request::builder()
                .method("HEAD")
                .uri("/static/htmx-2.0.4.min.js")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(body_string(resp).await.is_empty());
}

#[tokio::test]
async fn secure_cookie_diagnostic_does_not_trust_forwarded_proto() {
    let server = MockServer::start();
    let mut state = test_state_with_proxy(&server);
    Arc::make_mut(&mut state.config).cookie_secure = true;
    let app = build(state);
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/login")
                .header("X-Forwarded-Proto", "http")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.headers()["set-cookie"]
            .to_str()
            .unwrap()
            .contains("; Secure")
    );
    let html = body_string(resp).await;
    assert!(html.contains("ADMIN_COOKIE_SECURE=false"));
    assert!(html.contains("use HTTPS"));

    // Model the browser dropping the Secure pre-login cookie on remote HTTP.
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/login")
                .header("content-type", "application/x-www-form-urlencoded")
                .header("X-Forwarded-Proto", "http")
                .body(Body::from("password=&csrf_token=missing-cookie"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert!(
        resp.headers()["set-cookie"]
            .to_str()
            .unwrap()
            .contains("; Secure")
    );
    let html = body_string(resp).await;
    assert!(html.contains("cookie was blocked"));
    assert!(html.contains("ADMIN_COOKIE_SECURE=false"));
    assert!(html.contains("cookie_secure: false"));
}

fn mock_lists(server: &MockServer, truncated: bool) {
    for (path, field, rows) in [
        (
            "/key/list",
            "keys",
            json!([{"key_id":"k1"}, {"key_id":"k2"}]),
        ),
        (
            "/team/list",
            "teams",
            json!([{"team_id":"t1"}, {"team_id":"t2"}]),
        ),
        (
            "/user/list",
            "users",
            json!([{"user_id":"u1"}, {"user_id":"u2"}]),
        ),
        (
            "/organization/list",
            "organizations",
            json!([{"org_id":"o1"}, {"org_id":"o2"}]),
        ),
        (
            "/project/list",
            "projects",
            json!([{"project_id":"p1"}, {"project_id":"p2"}]),
        ),
        ("/model/list", "data", json!([{"id":"m1"}, {"id":"m2"}])),
    ] {
        let mut body = json!({"truncated": truncated});
        body[field] = rows;
        server.mock(move |when, then| {
            when.method(httpmock::Method::GET).path(path);
            then.status(200).json_body(body);
        });
    }
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/model/info");
        then.status(200).json_body(json!({}));
    });
}

#[tokio::test]
async fn list_pages_and_fragments_mark_only_explicitly_incomplete_results() {
    for truncated in [false, true] {
        let server = MockServer::start();
        mock_lists(&server, truncated);
        let app = build(test_state_with_proxy(&server));
        for label in [
            "keys",
            "teams",
            "users",
            "organizations",
            "projects",
            "models",
        ] {
            for hx in [false, true] {
                let uri = format!("/{label}");
                let req = if hx {
                    authed_htmx_get(&uri)
                } else {
                    authed_get(&uri)
                };
                let resp = app.clone().oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK, "{label}");
                let html = body_string(resp).await;
                assert!(
                    html.contains(&format!("2 {label}")),
                    "{label}: loaded count lost"
                );
                assert_eq!(
                    html.contains("Incomplete list:"),
                    truncated,
                    "{label}, hx={hx}"
                );
                assert_eq!(
                    html.contains(&format!("2 {label} shown (incomplete)")),
                    truncated
                );
            }
        }
    }
}

#[tokio::test]
async fn mutation_table_refreshes_keep_incomplete_markers() {
    let server = MockServer::start();
    mock_lists(&server, true);
    let mutation = server.mock(|when, then| {
        when.method(httpmock::Method::POST);
        then.status(200).json_body(json!({"status":"ok"}));
    });
    let app = build(test_state_with_proxy(&server));
    for (uri, body, label) in [
        ("/keys/k1/delete", "", "keys"),
        ("/teams/create", "team_name=New", "teams"),
        ("/users/u1/update", "user_name=New", "users"),
        ("/organizations/create", "org_name=New", "organizations"),
        ("/projects/create", "project_name=New", "projects"),
        (
            "/models/create",
            "model_name=New&litellm_model=openai%2Fnew",
            "models",
        ),
    ] {
        let resp = app.clone().oneshot(authed_post(uri, body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{uri}");
        let html = body_string(resp).await;
        assert!(
            html.contains(&format!("2 {label} shown (incomplete)")),
            "{uri}"
        );
    }
    mutation.assert_hits(6);
}

#[tokio::test]
async fn missing_dropdown_records_do_not_clear_project_associations() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/project/list");
        then.status(200).json_body(json!({"projects":[{
            "project_id":"p1", "project_name":"Project", "org_id":"org-omitted", "team_id":"team-omitted"
        }]}));
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/organization/list");
        then.status(200)
            .json_body(json!({"organizations":[{"org_id":"other-org"}], "truncated":true}));
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(503).body("unavailable");
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app.oneshot(authed_htmx_get("/projects")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("Incomplete list: 1 organizations shown."));
    assert!(html.contains("Teams list unavailable."));
    assert!(html.contains("value=\"org-omitted\" selected"));
    assert!(html.contains("value=\"team-omitted\" selected"));
    assert!(html.contains("value=\"other-org\""));
}

#[tokio::test]
async fn spend_filters_keep_ids_outside_the_loaded_lists() {
    let server = MockServer::start();
    mock_lists(&server, true);
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/spend/logs/v2");
        then.status(200)
            .json_body(json!({"data":[], "total_cost":0, "total_requests":0}));
    });
    let resp = build(test_state_with_proxy(&server))
        .oneshot(authed_htmx_get(
            "/spend?key_id=key-omitted&team_id=team-omitted",
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let html = body_string(resp).await;
    assert!(html.contains("value=\"key-omitted\" selected"));
    assert!(html.contains("value=\"team-omitted\" selected"));
    assert!(html.contains("Incomplete list: 2 keys shown."));
    assert!(html.contains("Incomplete list: 2 teams shown."));
}

#[tokio::test]
async fn all_mutation_routes_enforce_session_and_matching_csrf_before_proxy_calls() {
    let server = MockServer::start();
    let upstream = server.mock(|_, then| {
        then.status(500).body("unexpected proxy call");
    });
    let app = build(test_state_with_proxy(&server));
    for uri in [
        "/logout",
        "/keys/create",
        "/keys/k/delete",
        "/keys/k/enable",
        "/keys/k/purge",
        "/keys/k/rotate",
        "/keys/k/update",
        "/teams/create",
        "/teams/t/update",
        "/teams/t/delete",
        "/users/create",
        "/users/u/update",
        "/users/u/delete",
        "/organizations/create",
        "/organizations/o/update",
        "/organizations/o/delete",
        "/projects/create",
        "/projects/p/update",
        "/projects/p/delete",
        "/models/create",
        "/models/m/delete",
        "/calculator/run",
    ] {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("X-CSRF-Token", "forged")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SEE_OTHER, "{uri}");
        for token in [None, Some("forged")] {
            let mut req = authed_post_without_csrf(uri, "");
            if let Some(token) = token {
                req.headers_mut()
                    .insert("X-CSRF-Token", token.parse().unwrap());
            }
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::FORBIDDEN, "{uri}");
        }
    }
    upstream.assert_hits(0);
}

#[tokio::test]
async fn numeric_form_errors_do_not_reach_the_management_api() {
    let server = MockServer::start();
    let upstream = server.mock(|_, then| {
        then.status(500).body("unexpected proxy call");
    });
    let app = build(test_state_with_proxy(&server));
    for uri in [
        "/keys/create",
        "/keys/k/update",
        "/teams/create",
        "/teams/t/update",
        "/users/create",
        "/users/u/update",
        "/organizations/create",
        "/organizations/o/update",
        "/projects/create",
        "/projects/p/update",
    ] {
        for budget in ["NaN", "inf", "-1", "not-a-number"] {
            for hx in [false, true] {
                let mut req = authed_post(uri, &format!("max_budget={budget}"));
                if hx {
                    req.headers_mut()
                        .insert("HX-Request", "true".parse().unwrap());
                }
                let resp = app.clone().oneshot(req).await.unwrap();
                let status = if hx {
                    StatusCode::OK
                } else {
                    StatusCode::BAD_REQUEST
                };
                assert_eq!(resp.status(), status, "{uri}, {budget}");
                if hx {
                    // Preserve the table, edit form and entered values so the
                    // operator can correct the error and resubmit.
                    assert_eq!(resp.headers()["HX-Retarget"], "#flash");
                    assert_eq!(resp.headers()["HX-Reswap"], "innerHTML");
                    assert_eq!(resp.headers()["X-Oxllm-Error"], "true");
                }
                assert!(body_string(resp).await.contains("max_budget"));
            }
        }
    }
    upstream.assert_hits(0);
}

#[tokio::test]
async fn list_action_urls_roundtrip_custom_ids_as_one_path_segment() {
    let server = MockServer::start();
    let id = "scope/a+?#";
    server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/team/list");
        then.status(200)
            .json_body(json!({"teams":[{"team_id":id, "team_name":"Custom ID"}]}));
    });
    let delete = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/team/delete")
            .json_body(json!({"team_id":id}));
        then.status(200).json_body(json!({"status":"deleted"}));
    });
    let app = build(test_state_with_proxy(&server));
    let resp = app
        .clone()
        .oneshot(authed_htmx_get("/teams"))
        .await
        .unwrap();
    let html = body_string(resp).await;
    let url = "/teams/scope%2Fa%2B%3F%23/delete";
    assert!(html.contains(&format!("hx-post=\"{url}\"")));
    let resp = app.oneshot(authed_post(url, "")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    delete.assert_hits(1);
}
