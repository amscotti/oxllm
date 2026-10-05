use super::*;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use httpmock::{Method::POST, MockServer};
use oxllm_core::deployment::OxllmParams;
use serde_json::json;

// Synthetic RSA fixture, copied from the existing GCS cache tests. It has no
// Google identity or privileges; only local signing/mocked exchanges use it.
const DUMMY_RSA: &str = include_str!("testdata/dummy_rsa_key.pem");

fn authorized_user(refresh: &str) -> Value {
    json!({"type": "authorized_user", "client_id": "client+id", "client_secret": "client&secret", "refresh_token": refresh})
}

fn service_account() -> Value {
    json!({"type": "service_account", "client_email": "test@example.iam.gserviceaccount.com",
        "private_key": DUMMY_RSA, "private_key_id": "dummy-key-id", "token_uri": GOOGLE_TOKEN_URL})
}

fn deployment(source: Option<Value>) -> Deployment {
    let mut dep = Deployment {
        oxllm_params: OxllmParams {
            model: "vertex_ai/gemini-test".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    if let Some(source) = source {
        dep.oxllm_params
            .extra
            .insert("vertex_credentials".into(), source);
    }
    dep
}

fn resolver(server: &MockServer) -> VertexResolver {
    let mut resolver = VertexResolver::with_env(|_| None);
    resolver.exchange = Arc::new(TokenExchange::for_test(
        server.url("/token"),
        Duration::from_secs(2),
    ));
    resolver
}

#[test]
fn service_account_assertion_has_google_audience_scope_and_rs256_header() {
    let credentials: GoogleCredentials = serde_json::from_value(service_account()).unwrap();
    credentials.validate().unwrap();
    let before = unix_now().unwrap();
    let form = credentials.form().unwrap();
    assert_eq!(
        form[0],
        (
            "grant_type",
            "urn:ietf:params:oauth:grant-type:jwt-bearer".into()
        )
    );
    assert_eq!(form[1].0, "assertion");
    let parts: Vec<_> = form[1].1.split('.').collect();
    assert_eq!(parts.len(), 3);
    let header: Value = serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[0]).unwrap()).unwrap();
    let claims: Value = serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[1]).unwrap()).unwrap();
    assert_eq!(header["alg"], "RS256");
    assert_eq!(header["kid"], "dummy-key-id");
    assert_eq!(claims["iss"], "test@example.iam.gserviceaccount.com");
    assert_eq!(claims["aud"], GOOGLE_TOKEN_URL);
    assert_eq!(claims["scope"], CLOUD_PLATFORM_SCOPE);
    let issued = claims["iat"].as_u64().unwrap();
    assert!(issued >= before && issued <= unix_now().unwrap());
    assert_eq!(claims["exp"].as_u64().unwrap() - issued, 3600);
    assert_eq!(URL_SAFE_NO_PAD.decode(parts[2]).unwrap().len(), 256);
    assert!(
        claims.get("sub").is_none(),
        "no implicit user impersonation"
    );
}

#[tokio::test]
async fn service_account_is_exchanged_and_cached_instead_of_forwarding_its_private_key() {
    let server = MockServer::start_async().await;
    let token = server.mock_async(|when, then| {
        when.method(POST).path("/token")
            .body_contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer")
            .body_contains("assertion=eyJ");
        then.status(200).json_body(json!({"access_token": "google-access", "expires_in": 3600, "token_type": "Bearer"}));
    }).await;
    let resolver = resolver(&server);
    let dep = deployment(Some(service_account()));
    for _ in 0..2 {
        assert!(
            matches!(resolver.resolve(&dep).await.unwrap(), ResolvedCredentials::GcpBearer { token } if token == "google-access")
        );
    }
    token.assert_hits_async(1).await;
}

#[tokio::test]
async fn authorized_user_refresh_is_coalesced_and_deployment_credentials_beat_ambient_tokens() {
    let server = MockServer::start_async().await;
    let token = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .header("content-type", "application/x-www-form-urlencoded")
                .body_contains("grant_type=refresh_token")
                .body_contains("client_id=client%2Bid")
                .body_contains("client_secret=client%26secret")
                .body_contains("refresh_token=refresh%2Fone");
            then.status(200)
                .json_body(json!({"access_token": "user-access", "expires_in": 3600}));
        })
        .await;
    let mut resolver = resolver(&server);
    resolver.env_lookup = Arc::new(|name| (name == "GCP_ACCESS_TOKEN").then(|| "ambient".into()));
    let mut dep = deployment(Some(authorized_user("refresh/one")));
    dep.oxllm_params.api_base = Some("http://169.254.169.254/ignored-for-auth".into());
    for result in futures_util::future::join_all((0..8).map(|_| resolver.resolve(&dep))).await {
        assert_eq!(result.unwrap().best_effort_bearer(), Some("user-access"));
    }
    token.assert_hits_async(1).await;
}

#[tokio::test]
async fn standard_credential_file_is_reread_so_rotation_invalidates_cached_identity() {
    let server = MockServer::start_async().await;
    let first = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_contains("refresh_token=one");
            then.status(200)
                .json_body(json!({"access_token": "first", "expires_in": 3600}));
        })
        .await;
    let second = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_contains("refresh_token=two");
            then.status(200)
                .json_body(json!({"access_token": "second", "expires_in": 3600}));
        })
        .await;
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), authorized_user("one").to_string()).unwrap();
    let path = file.path().to_str().unwrap().to_string();
    let mut resolver = resolver(&server);
    resolver.env_lookup =
        Arc::new(move |name| (name == "GOOGLE_APPLICATION_CREDENTIALS").then(|| path.clone()));
    let dep = deployment(None);
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("first")
    );
    std::fs::write(file.path(), authorized_user("two").to_string()).unwrap();
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("second")
    );
    first.assert_hits_async(1).await;
    second.assert_hits_async(1).await;
}

#[tokio::test]
async fn credential_file_paths_and_inline_json_strings_resolve_without_full_adc_discovery() {
    let server = MockServer::start_async().await;
    let token = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(200)
                .json_body(json!({"access_token": "access", "expires_in": 60}));
        })
        .await;
    let credentials = authorized_user("one").to_string();
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), &credentials).unwrap();
    let resolver = resolver(&server);
    for source in [json!(file.path().to_str().unwrap()), json!(credentials)] {
        assert_eq!(
            resolver
                .resolve(&deployment(Some(source)))
                .await
                .unwrap()
                .best_effort_bearer(),
            Some("access")
        );
    }
    token.assert_hits_async(1).await;
}

#[tokio::test]
async fn unsafe_token_uris_and_unsupported_adc_types_fail_before_network_access() {
    let server = MockServer::start_async().await;
    let token = server
        .mock_async(|when, then| {
            when.path("/token");
            then.status(500);
        })
        .await;
    let resolver = resolver(&server);
    for uri in [
        "http://169.254.169.254/token",
        "https://127.0.0.1/token",
        "https://oauth2.googleapis.com.evil.example/token",
        "https://oauth2.googleapis.com/token#fragment",
        "https://oauth2.googleapis.com/token?redirect=evil",
        "https://user@oauth2.googleapis.com/token",
    ] {
        let mut source = authorized_user("secret-marker");
        source["token_uri"] = json!(uri);
        let error = resolver
            .resolve(&deployment(Some(source)))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Config(_)));
        assert!(!error.to_string().contains(uri));
        assert!(!error.to_string().contains("secret-marker"));
    }
    for kind in ["external_account", "impersonated_service_account"] {
        let source = json!({"type": kind, "credential_source": {"url": server.url("/token")}});
        assert!(matches!(
            resolver.resolve(&deployment(Some(source))).await,
            Err(Error::Config(_))
        ));
    }
    let mut source = authorized_user("secret-marker");
    source["universe_domain"] = json!("other.example");
    assert!(matches!(
        resolver.resolve(&deployment(Some(source))).await,
        Err(Error::Config(_))
    ));
    token.assert_hits_async(0).await;
}

#[tokio::test]
async fn malformed_oversized_and_nonregular_files_fail_with_redacted_errors() {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), "{secret-marker").unwrap();
    let resolver = VertexResolver::with_env(|_| None);
    let source = json!(file.path().to_str().unwrap());
    let error = resolver
        .resolve(&deployment(Some(source.clone())))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Config(_)));
    assert!(!error.to_string().contains("secret-marker"));
    assert!(!error.to_string().contains(file.path().to_str().unwrap()));
    std::fs::write(file.path(), vec![b'x'; MAX_CREDENTIAL_BYTES + 1]).unwrap();
    assert!(
        resolver
            .resolve(&deployment(Some(source)))
            .await
            .unwrap_err()
            .to_string()
            .contains("64 KiB")
    );
    let dir = tempfile::tempdir().unwrap();
    assert!(
        read_credential_file(dir.path().to_str().unwrap().into())
            .await
            .is_err()
    );
}

/// Renewable Google credentials are cached until the skew-adjusted refresh
/// deadline, re-exchanged once past it, and a failed re-exchange surfaces the
/// upstream error. Returning the previous bearer after a failed refresh would
/// silently 401 downstream; refreshing early or never both fail the
/// hit-count/bearer assertions.
#[tokio::test]
async fn renewable_token_is_cached_then_refreshed_and_a_failed_refresh_surfaces() {
    let server = MockServer::start_async().await;
    let first = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_contains("grant_type=refresh_token");
            then.status(200)
                .json_body(json!({"access_token": "first-access", "expires_in": 10}));
        })
        .await;
    let resolver = resolver(&server);
    let dep = deployment(Some(authorized_user("refresh/one")));
    for _ in 0..2 {
        assert_eq!(
            resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
            Some("first-access")
        );
    }
    first.assert_hits_async(1).await;
    first.delete_async().await;
    // Lifetime 10 s → 1 s skew; advance past the deadline to force a refresh.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::time::resume();
    let second = server
        .mock_async(|when, then| {
            when.method(POST)
                .path("/token")
                .body_contains("grant_type=refresh_token");
            then.status(200)
                .json_body(json!({"access_token": "second-access", "expires_in": 10}));
        })
        .await;
    assert_eq!(
        resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
        Some("second-access")
    );
    second.assert_hits_async(1).await;
    second.delete_async().await;
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::time::resume();
    let failing = server
        .mock_async(|when, then| {
            when.method(POST).path("/token");
            then.status(401);
        })
        .await;
    let error = resolver.resolve(&dep).await.unwrap_err();
    assert!(matches!(error, Error::Upstream { status: 401, .. }));
    assert!(!error.to_string().contains("second-access"));
    failing.assert_hits_async(1).await;
}
