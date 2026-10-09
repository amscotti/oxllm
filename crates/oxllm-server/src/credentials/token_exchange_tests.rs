use super::*;
use httpmock::{Method::POST, MockServer};
use serde_json::json;

fn exchange(server: &MockServer) -> TokenExchange {
    TokenExchange::for_test(server.url("/token"), Duration::from_secs(2))
}

fn form() -> Result<Form, Error> {
    Ok(vec![("apikey", "fake-key".into())])
}

struct JsonAssertion {
    assertion: Zeroizing<String>,
    source_drops: Arc<std::sync::atomic::AtomicUsize>,
    fail_serialization: bool,
}

impl Serialize for JsonAssertion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut body = serializer.serialize_map(Some(1))?;
        body.serialize_entry("assertion", self.assertion.as_str())?;
        if self.fail_serialization {
            return Err(serde::ser::Error::custom(
                "secret-marker serialization detail",
            ));
        }
        body.end()
    }
}

impl Drop for JsonAssertion {
    fn drop(&mut self) {
        self.source_drops
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

fn json_assertion(
    assertion: &str,
    source_drops: &Arc<std::sync::atomic::AtomicUsize>,
    fail_serialization: bool,
) -> JsonAssertion {
    JsonAssertion {
        assertion: Zeroizing::new(assertion.into()),
        source_drops: source_drops.clone(),
        fail_serialization,
    }
}

#[tokio::test]
async fn json_grants_release_the_assertion_source_before_http_and_refresh_lazily() {
    use axum::{Json, Router, http::HeaderMap, routing::post};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let source_drops = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/token",
        post({
            let source_drops = source_drops.clone();
            let requests = requests.clone();
            move |headers: HeaderMap, Json(body): Json<serde_json::Value>| {
                let source_drops = source_drops.clone();
                let requests = requests.clone();
                async move {
                    let number = requests.fetch_add(1, Ordering::SeqCst) + 1;
                    assert_eq!(source_drops.load(Ordering::SeqCst), number);
                    assert_eq!(headers["accept"], "application/json");
                    assert_eq!(headers["content-type"], "application/json");
                    let (assertion, token) = match number {
                        1 => (r#"secret-marker"\assertion"#, "first"),
                        2 => ("rotated-assertion", "second"),
                        _ => panic!("unexpected credential exchange"),
                    };
                    assert_eq!(body, json!({"assertion":assertion}));
                    Json(json!({"access_token":token, "expires_in":10}))
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/token", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let exchange = TokenExchange::for_test(endpoint, Duration::from_secs(2));
    assert_eq!(
        exchange
            .token_json(b"identity", async {
                Ok(json_assertion(
                    r#"secret-marker"\assertion"#,
                    &source_drops,
                    false,
                ))
            })
            .await
            .unwrap(),
        "first"
    );
    assert_eq!(
        exchange
            .token_json(b"identity", async {
                Err::<serde_json::Value, _>(Error::Config(
                    "cached grant must not read its source".into(),
                ))
            })
            .await
            .unwrap(),
        "first"
    );
    assert_eq!(source_drops.load(Ordering::SeqCst), 1);
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(9)).await;
    tokio::time::resume();
    assert_eq!(
        exchange
            .token_json(b"identity", async {
                Ok(json_assertion("rotated-assertion", &source_drops, false))
            })
            .await
            .unwrap(),
        "second"
    );
    assert_eq!(source_drops.load(Ordering::SeqCst), 2);
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn json_serialization_failure_releases_its_source_and_never_sends_partial_assertions() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let source_drops = Arc::new(AtomicUsize::new(0));
    let server = MockServer::start_async().await;
    let grant = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .json_body(json!({"access_token":"minted", "expires_in":3600}));
        })
        .await;
    let exchange = exchange(&server);
    let error = exchange
        .token_json(b"identity", async {
            Ok(json_assertion("secret-marker", &source_drops, true))
        })
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Upstream { status: 502, .. }));
    assert!(!error.to_string().contains("secret-marker"));
    assert_eq!(source_drops.load(Ordering::SeqCst), 1);
    assert!(matches!(
        exchange
            .token_json(b"identity", async {
                Err::<serde_json::Value, _>(Error::Config("backoff must not rebuild".into()))
            })
            .await,
        Err(Error::Upstream { status: 401, .. })
    ));
    grant.assert_hits_async(0).await;
}

#[tokio::test]
async fn tokens_refresh_before_expiry_without_extending_short_lifetimes() {
    let server = MockServer::start_async().await;
    let first = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .json_body(json!({"access_token": "first", "expires_in": 10}));
        })
        .await;
    let exchange = exchange(&server);
    assert_eq!(exchange.token(b"identity", form).await.unwrap(), "first");
    assert_eq!(exchange.token(b"identity", form).await.unwrap(), "first");
    first.assert_hits_async(1).await;
    first.delete_async().await;
    let second = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .json_body(json!({"access_token": "second", "expires_in": 10}));
        })
        .await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(9)).await;
    tokio::time::resume();
    assert_eq!(exchange.token(b"identity", form).await.unwrap(), "second");
    second.assert_hits_async(1).await;
}

#[tokio::test]
async fn invalid_or_expired_token_responses_are_rejected_and_failures_are_coalesced() {
    for body in [
        json!({"access_token": "secret-marker", "expires_in": 0}),
        json!({"access_token": "secret-marker", "expires_in": -1}),
        json!({"access_token": "secret-marker"}),
        json!({"access_token": "secret-marker", "expires_in": 3600, "expiration": 1}),
        json!({"access_token": "secret-marker", "expires_in": 3600, "token_type": "MAC"}),
        json!({"access_token": "\r\nsecret-marker", "expires_in": 3600}),
        json!({"access_token": " ", "expires_in": 3600}),
    ] {
        let server = MockServer::start_async().await;
        let token = server
            .mock_async(|when, then| {
                when.method(POST).path("/token");
                then.status(200).json_body(body);
            })
            .await;
        let exchange = exchange(&server);
        for _ in 0..2 {
            let error = exchange.token(b"identity", form).await.unwrap_err();
            assert!(matches!(error, Error::Upstream { status: 502, .. }));
            assert!(!error.to_string().contains("secret-marker"));
        }
        token.assert_hits_async(1).await;
    }
}

#[tokio::test]
async fn absolute_ibm_expiration_is_supported_and_bounds_expires_in() {
    let server = MockServer::start_async().await;
    let token = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .json_body(json!({"access_token": "ibm", "expiration": unix_now().unwrap() + 10}));
        })
        .await;
    let exchange = exchange(&server);
    assert_eq!(exchange.token(b"identity", form).await.unwrap(), "ibm");
    token.assert_hits_async(1).await;
}

#[tokio::test]
async fn token_endpoints_never_follow_redirects_or_echo_error_bodies() {
    let server = MockServer::start_async().await;
    let target = server
        .mock_async(|when, then| {
            when.path("/stolen");
            then.status(200)
                .json_body(json!({"access_token": "wrong", "expires_in": 3600}));
        })
        .await;
    let redirect = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(307).header("location", server.url("/stolen"));
        })
        .await;
    let exchange = exchange(&server);
    assert!(exchange.token(b"identity", form).await.is_err());
    target.assert_hits_async(0).await;
    redirect.delete_async().await;
    let rejected = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(401)
                .body("secret-marker echoed by token endpoint");
        })
        .await;
    let error = exchange.token(b"another-identity", form).await.unwrap_err();
    assert!(matches!(error, Error::Upstream { status: 401, .. }));
    assert!(!error.to_string().contains("secret-marker"));
    rejected.assert_hits_async(1).await;
}

#[tokio::test]
async fn content_length_and_credential_material_are_bounded() {
    let server = MockServer::start_async().await;
    let token = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200).body("x".repeat(MAX_TOKEN_BODY_BYTES + 1));
        })
        .await;
    let exchange = exchange(&server);
    assert!(matches!(
        exchange
            .token(&vec![b'x'; MAX_CREDENTIAL_BYTES + 1], form)
            .await,
        Err(Error::Config(_))
    ));
    token.assert_hits_async(0).await;
    let error = exchange.token(b"identity", form).await.unwrap_err();
    assert!(error.to_string().contains("64 KiB"));
    token.assert_hits_async(1).await;
}

#[tokio::test]
async fn chunked_body_is_size_capped_and_body_read_has_a_total_deadline() {
    use axum::{Router, body::Body, routing::post};
    use futures_util::stream;
    let app = Router::new()
        .route(
            "/large",
            post(|| async {
                Body::from_stream(stream::iter([Ok::<_, std::io::Error>(vec![
                    b'x';
                    MAX_TOKEN_BODY_BYTES
                        + 1
                ])]))
            }),
        )
        .route(
            "/slow",
            post(|| async {
                Body::from_stream(stream::once(async {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    Ok::<_, std::io::Error>(r#"{"access_token":"late","expires_in":60}"#)
                }))
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let large = TokenExchange::for_test(format!("{base}/large"), Duration::from_secs(2));
    assert!(
        large
            .token(b"identity", form)
            .await
            .unwrap_err()
            .to_string()
            .contains("64 KiB")
    );
    let slow = TokenExchange::for_test(format!("{base}/slow"), Duration::from_millis(100));
    assert!(matches!(
        slow.token(b"identity", form).await,
        Err(Error::Upstream { status: 504, .. })
    ));
    task.abort();
}

#[tokio::test]
async fn cancelled_refresh_leaves_backoff_instead_of_restarting_for_each_waiter() {
    use axum::{Router, routing::post};
    use std::sync::atomic::{AtomicUsize, Ordering};
    for use_json in [false, true] {
        let entered = Arc::new(tokio::sync::Notify::new());
        let requests = Arc::new(AtomicUsize::new(0));
        let app = Router::new().route(
            "/token",
            post({
                let entered = entered.clone();
                let requests = requests.clone();
                move || {
                    let entered = entered.clone();
                    let requests = requests.clone();
                    async move {
                        requests.fetch_add(1, Ordering::SeqCst);
                        entered.notify_one();
                        std::future::pending::<&'static str>().await
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/token", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let exchange = Arc::new(TokenExchange::for_test(endpoint, Duration::from_secs(2)));
        let caller = tokio::spawn({
            let exchange = exchange.clone();
            async move {
                if use_json {
                    exchange
                        .token_json(b"identity", async { Ok(json!({"assertion":"identity"})) })
                        .await
                } else {
                    exchange.token(b"identity", form).await
                }
            }
        });
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        caller.abort();
        let _ = caller.await;
        let result = if use_json {
            exchange
                .token_json(b"identity", async { Ok(json!({"assertion":"identity"})) })
                .await
        } else {
            exchange.token(b"identity", form).await
        };
        assert!(matches!(result, Err(Error::Upstream { status: 503, .. })));
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        server.abort();
    }
}

/// Once the refresh deadline (expiry minus skew) passes, a failed re-exchange
/// must surface the upstream error. Serving the cached `Ready` token would
/// hand the caller a bearer the upstream already rejects (silent 401) and
/// mask the credential problem. Removing the refresh-after-success path or
/// adding a stale-token fallback fails this test.
#[tokio::test]
async fn expired_token_refresh_failure_surfaces_error_instead_of_stale_token() {
    let server = MockServer::start_async().await;
    let first = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .json_body(json!({"access_token": "stale-token", "expires_in": 10}));
        })
        .await;
    let exchange = exchange(&server);
    assert_eq!(
        exchange.token(b"identity", form).await.unwrap(),
        "stale-token"
    );
    first.assert_hits_async(1).await;
    first.delete_async().await;
    // Lifetime 10 s → 1 s skew, so the token is stale after 9 s; advance past
    // both so the refresh is attempted, not served from cache.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::time::resume();
    let failing = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(500).body("token endpoint detail");
        })
        .await;
    let error = exchange.token(b"identity", form).await.unwrap_err();
    assert!(matches!(error, Error::Upstream { status: 500, .. }));
    failing.assert_hits_async(1).await;
}

struct RenewalFixture {
    exchange: Arc<TokenExchange>,
    requests: Arc<std::sync::atomic::AtomicUsize>,
    blocked: Arc<std::sync::atomic::AtomicBool>,
    failing: Arc<std::sync::atomic::AtomicBool>,
    release: Arc<tokio::sync::Notify>,
    server: tokio::task::JoinHandle<()>,
}

impl RenewalFixture {
    async fn new(lifetime: u64) -> Self {
        use axum::{Json, Router, http::StatusCode, response::IntoResponse, routing::post};
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let requests = Arc::new(AtomicUsize::new(0));
        let blocked = Arc::new(AtomicBool::new(false));
        let failing = Arc::new(AtomicBool::new(false));
        let release = Arc::new(tokio::sync::Notify::new());
        let app = Router::new().route("/token", post({
            let requests = requests.clone();
            let blocked = blocked.clone();
            let failing = failing.clone();
            let release = release.clone();
            move || {
                let (requests, blocked, failing, release) = (requests.clone(), blocked.clone(), failing.clone(), release.clone());
                async move {
                    let number = requests.fetch_add(1, Ordering::SeqCst) + 1;
                    if blocked.load(Ordering::SeqCst) {
                        release.notified().await;
                    }
                    if failing.load(Ordering::SeqCst) {
                        (StatusCode::BAD_GATEWAY, "secret-marker").into_response()
                    } else {
                        Json(json!({"access_token": format!("token-{number}"), "expires_in": lifetime})).into_response()
                    }
                }
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/token", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            exchange: Arc::new(TokenExchange::for_test(endpoint, Duration::from_secs(5))),
            requests,
            blocked,
            failing,
            release,
            server,
        }
    }

    async fn wait_for_requests(&self, expected: usize) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while self.requests.load(std::sync::atomic::Ordering::SeqCst) < expected {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
    }

    async fn wait_for_token(&self, expected: &str) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if self
                    .exchange
                    .background_token(b"identity", form)
                    .await
                    .unwrap()
                    == expected
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
    }
}

impl Drop for RenewalFixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn advance_renewal_time(seconds: u64) {
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(seconds)).await;
    tokio::time::resume();
}

#[tokio::test]
async fn half_ttl_background_renewal_does_not_block_readers_or_duplicate_exchanges() {
    use std::sync::atomic::Ordering;
    let fixture = RenewalFixture::new(10).await;
    assert_eq!(
        fixture
            .exchange
            .background_token(b"identity", form)
            .await
            .unwrap(),
        "token-1"
    );
    fixture.blocked.store(true, Ordering::SeqCst);
    advance_renewal_time(5).await;
    fixture.wait_for_requests(2).await;
    // Renewal is in flight, with no subsequent request needed to start it.
    for _ in 0..8 {
        let token = tokio::time::timeout(
            Duration::from_millis(100),
            fixture.exchange.background_token(b"identity", form),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(token, "token-1");
    }
    assert_eq!(fixture.requests.load(Ordering::SeqCst), 2);
    fixture.blocked.store(false, Ordering::SeqCst);
    fixture.release.notify_one();
    fixture.wait_for_token("token-2").await;
    assert_eq!(fixture.requests.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn background_renewal_failure_keeps_a_valid_token_but_never_serves_an_expired_one() {
    use std::sync::atomic::Ordering;
    let fixture = RenewalFixture::new(10).await;
    assert_eq!(
        fixture
            .exchange
            .background_token(b"identity", form)
            .await
            .unwrap(),
        "token-1"
    );
    fixture.failing.store(true, Ordering::SeqCst);
    advance_renewal_time(5).await;
    fixture.wait_for_requests(2).await;
    let key: [u8; 32] = Sha256::digest(b"identity").into();
    let slot = fixture.exchange.renewable_cache.get(&key).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if matches!(
                *slot.entry.lock().await,
                Some(RenewableEntry::Ready {
                    failure: Some(_),
                    ..
                })
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        fixture
            .exchange
            .background_token(b"identity", form)
            .await
            .unwrap(),
        "token-1"
    );
    advance_renewal_time(5).await;
    let error = fixture
        .exchange
        .background_token(b"identity", form)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Upstream { status: 502, .. }));
    assert!(!error.to_string().contains("secret-marker"));
    assert!(!error.to_string().contains("token-1"));
    fixture.failing.store(false, Ordering::SeqCst);
    advance_renewal_time(35).await;
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(token) = fixture.exchange.background_token(b"identity", form).await {
                assert_ne!(token, "token-1");
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn renewable_tokens_remain_ready_after_a_controlled_seventy_minute_soak() {
    let fixture = RenewalFixture::new(3600).await;
    assert_eq!(
        fixture
            .exchange
            .background_token(b"identity", form)
            .await
            .unwrap(),
        "token-1"
    );
    advance_renewal_time(1800).await;
    fixture.wait_for_requests(2).await;
    fixture.wait_for_token("token-2").await;
    advance_renewal_time(1800).await;
    fixture.wait_for_requests(3).await;
    fixture.wait_for_token("token-3").await;
    advance_renewal_time(600).await;
    assert_eq!(
        fixture
            .exchange
            .background_token(b"identity", form)
            .await
            .unwrap(),
        "token-3"
    );
}

#[tokio::test]
async fn evicting_a_renewable_identity_cancels_its_worker_and_releases_refresh_material() {
    let server = MockServer::start_async().await;
    let token = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .json_body(json!({"access_token": "token", "expires_in": 3600}));
        })
        .await;
    let exchange = Arc::new(exchange(&server));
    let material = Arc::new(());
    let weak = Arc::downgrade(&material);
    assert_eq!(
        exchange
            .background_token(b"identity", move || {
                let _keep_alive = &material;
                form()
            })
            .await
            .unwrap(),
        "token"
    );
    let key: [u8; 32] = Sha256::digest(b"identity").into();
    exchange.renewable_cache.invalidate(&key).await;
    exchange.renewable_cache.run_pending_tasks().await;
    tokio::time::timeout(Duration::from_secs(2), async {
        while weak.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    token.assert_hits_async(1).await;
}
