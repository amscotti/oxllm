use super::*;
use httpmock::{Method::POST, MockServer};
use serde_json::json;

fn exchange(server: &MockServer) -> TokenExchange {
    TokenExchange::for_test(server.url("/token"), Duration::from_secs(2))
}

fn form() -> Result<Form, Error> {
    Ok(vec![("apikey", "fake-key".into())])
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
        async move { exchange.token(b"identity", form).await }
    });
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    caller.abort();
    let _ = caller.await;
    assert!(matches!(
        exchange.token(b"identity", form).await,
        Err(Error::Upstream { status: 503, .. })
    ));
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    server.abort();
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
