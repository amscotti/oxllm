use super::*;
use httpmock::{Method::POST, MockServer};
use oxllm_core::deployment::OxllmParams;
use serde_json::json;
use std::time::Duration;

fn deployment(key: Option<&str>) -> Deployment {
    Deployment {
        oxllm_params: OxllmParams {
            model: "watsonx/ibm/granite".into(),
            api_key: key.map(String::from),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn resolver(server: &MockServer) -> WatsonXResolver {
    let mut resolver = WatsonXResolver::with_env(|_| None);
    resolver.exchange = Arc::new(TokenExchange::for_test(
        server.url("/identity/token"),
        Duration::from_secs(2),
    ));
    resolver
}

#[tokio::test]
async fn iam_exchange_encodes_form_and_coalesces_concurrent_requests() {
    let server = MockServer::start_async().await;
    let iam = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/identity/token")
                .header("content-type", "application/x-www-form-urlencoded")
                .body_contains("grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey")
                .body_contains("apikey=tenant-a%2B%26");
            then.status(200).json_body(
                json!({"access_token": "iam-a", "token_type": "Bearer", "expires_in": 3600}),
            );
        })
        .await;
    let resolver = resolver(&server);
    let dep = deployment(Some("tenant-a+&"));
    let results = futures_util::future::join_all((0..8).map(|_| resolver.resolve(&dep))).await;
    for result in results {
        assert_eq!(result.unwrap().best_effort_bearer(), Some("iam-a"));
    }
    iam.assert_hits_async(1).await;
}

#[tokio::test]
async fn iam_cache_isolates_rotated_keys_and_ignores_inference_base_for_token_exchange() {
    let server = MockServer::start_async().await;
    let a = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/identity/token")
                .body_contains("apikey=key-a");
            then.status(200)
                .json_body(json!({"access_token": "iam-a", "expires_in": 3600}));
        })
        .await;
    let b = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/identity/token")
                .body_contains("apikey=key-b");
            then.status(200)
                .json_body(json!({"access_token": "iam-b", "expires_in": 3600}));
        })
        .await;
    let mut resolver = resolver(&server);
    resolver.env_lookup =
        Arc::new(|name| (name == "WATSONX_TOKEN").then(|| "ambient-token".into()));
    let mut dep = deployment(Some("key-a"));
    dep.oxllm_params.api_base = Some("http://169.254.169.254/inference".into());
    dep.oxllm_params
        .extra
        .insert("iam_url".into(), json!("http://169.254.169.254/token"));
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("iam-a")
    );
    dep.oxllm_params.api_key = Some("key-b".into());
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("iam-b")
    );
    a.assert_hits_async(1).await;
    b.assert_hits_async(1).await;
}

#[tokio::test]
async fn explicit_and_environment_token_modes_do_not_exchange_a_token_as_an_api_key() {
    let resolver =
        WatsonXResolver::with_env(|name| (name == "WATSONX_TOKEN").then(|| "ambient".into()));
    let mut dep = deployment(Some("key-must-not-be-used"));
    dep.oxllm_params
        .extra
        .insert("watsonx_token".into(), json!("configured"));
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("configured")
    );
    assert_eq!(
        resolver
            .resolve(&deployment(None))
            .await
            .unwrap()
            .best_effort_bearer(),
        Some("ambient")
    );
    assert!(!format!("{resolver:?}").contains("ambient"));
}

#[tokio::test]
async fn api_key_environment_aliases_are_exchanged() {
    let server = MockServer::start_async().await;
    let iam = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/identity/token")
                .body_contains("apikey=env-key");
            then.status(200)
                .json_body(json!({"access_token": "iam-env", "expires_in": 60}));
        })
        .await;
    for alias in ["WATSONX_APIKEY", "WATSONX_API_KEY"] {
        let mut resolver = resolver(&server);
        resolver.env_lookup = Arc::new(move |name| (name == alias).then(|| "env-key".into()));
        assert_eq!(
            resolver
                .resolve(&deployment(None))
                .await
                .unwrap()
                .best_effort_bearer(),
            Some("iam-env")
        );
    }
    iam.assert_hits_async(2).await;
}

#[tokio::test]
async fn missing_credentials_are_actionable_without_leaking_model_names() {
    let err = WatsonXResolver::with_env(|_| None)
        .resolve(&deployment(None))
        .await
        .unwrap_err();
    assert!(
        matches!(err, Error::Config(ref message) if message.contains("WATSONX_TOKEN") && !message.contains("granite"))
    );
}

#[tokio::test]
async fn malformed_explicit_credentials_do_not_fall_back_to_an_ambient_identity() {
    let resolver = WatsonXResolver::with_env(|_| Some("ambient-identity".into()));
    for field in ["watsonx_token", "watsonx_api_key"] {
        let mut dep = deployment(None);
        dep.oxllm_params
            .extra
            .insert(field.into(), json!({"invalid": "secret-marker"}));
        let error = resolver.resolve(&dep).await.unwrap_err();
        assert!(matches!(error, Error::Config(_)));
        assert!(!error.to_string().contains("secret-marker"));
    }
}

/// The IAM bearer is cached until the skew-adjusted refresh deadline and then
/// re-exchanged. IBM IAM tokens live ~1 h; the skew is `min(60 s, life/10)`,
/// i.e. 60 s here, so the boundary is observable at 540 s. A regression that
/// refreshed on every call (cache miss) or never refreshed past the deadline
/// fails the hit-count/bearer assertions.
#[tokio::test]
async fn iam_token_is_cached_until_the_refresh_skew_then_refreshed() {
    let server = MockServer::start_async().await;
    let first = server
        .mock_async(|when, then| {
            when.method(POST).path("/identity/token");
            then.status(200)
                .json_body(json!({"access_token": "iam-first", "expires_in": 600}));
        })
        .await;
    let resolver = resolver(&server);
    let dep = deployment(Some("tenant-key"));
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("iam-first")
    );
    // One second before the 540 s boundary the cached token is still served.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(539)).await;
    tokio::time::resume();
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("iam-first")
    );
    first.assert_hits_async(1).await;
    first.delete_async().await;
    let second = server
        .mock_async(|when, then| {
            when.method(POST).path("/identity/token");
            then.status(200)
                .json_body(json!({"access_token": "iam-second", "expires_in": 600}));
        })
        .await;
    // Past the skew the resolver must exchange again and use the new bearer.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::time::resume();
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("iam-second")
    );
    second.assert_hits_async(1).await;
}

/// Past the refresh deadline a failing IAM exchange must propagate the error;
/// returning the cached "iam-old" bearer would silently 401 downstream.
#[tokio::test]
async fn expired_iam_token_is_not_served_stale_when_the_refresh_fails() {
    let server = MockServer::start_async().await;
    let first = server
        .mock_async(|when, then| {
            when.method(POST).path("/identity/token");
            then.status(200)
                .json_body(json!({"access_token": "iam-old", "expires_in": 10}));
        })
        .await;
    let resolver = resolver(&server);
    let dep = deployment(Some("tenant-key"));
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("iam-old")
    );
    first.assert_hits_async(1).await;
    first.delete_async().await;
    // Lifetime 10 s → 1 s skew; advance past expiry so a refresh is required.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::time::resume();
    let failing = server
        .mock_async(|when, then| {
            when.method(POST).path("/identity/token");
            then.status(500);
        })
        .await;
    let error = resolver.resolve(&dep).await.unwrap_err();
    assert!(matches!(error, Error::Upstream { status: 500, .. }));
    failing.assert_hits_async(1).await;
}
