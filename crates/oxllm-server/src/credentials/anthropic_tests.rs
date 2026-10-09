use super::*;
use httpmock::{Method::POST, MockServer};
use std::io::Write;
use std::path::Path;

fn resolver(roots: &[&Path]) -> AnthropicResolver {
    let mut resolver = AnthropicResolver::new();
    resolver.use_env = false;
    resolver.allow_http = true;
    resolver.trusted_hosts = vec![("127.0.0.1".into(), None)];
    resolver.credential_dirs = roots.iter().map(|root| root.to_path_buf()).collect();
    resolver
}

fn deployment(server: &MockServer, path: &Path) -> Deployment {
    serde_json::from_value(json!({"model_name":"wif", "oxllm_params":{
        "model":"anthropic/claude-haiku-5-5", "api_base":server.url("/v1"),
        "anthropic_identity_token_file":path, "anthropic_federation_rule_id":"rule",
        "anthropic_organization_id":"org"
    }}))
    .unwrap()
}

fn assert_denied(error: &Error, message: &str) {
    assert!(
        matches!(error, Error::Upstream { status: 401, .. }),
        "{error}"
    );
    assert!(error.to_string().contains(message), "{error}");
    assert!(!error.to_string().contains("secret-marker"), "{error}");
}

fn source_deployment(extra: Value) -> Deployment {
    Deployment {
        oxllm_params: oxllm_core::deployment::OxllmParams {
            model: "anthropic/claude-haiku-5-5".into(),
            extra: extra.as_object().unwrap().clone(),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn deployment_identity_sources_precede_all_fleet_defaults() {
    for (extra, expected) in [
        (
            json!({"anthropic_identity_token_file":"/deployment/token", "anthropic_identity_token":"oidc/env/DEPLOYMENT"}),
            IdentitySource::Reference("oidc/file//deployment/token".into()),
        ),
        (
            json!({"anthropic_identity_token":"oidc/env/DEPLOYMENT"}),
            IdentitySource::Reference("oidc/env/DEPLOYMENT".into()),
        ),
        (
            json!({"anthropic_identity_token":"oidc/file//deployment/token"}),
            IdentitySource::Reference("oidc/file//deployment/token".into()),
        ),
        (
            json!({"anthropic_identity_token_file":" \t", "anthropic_identity_token":"oidc/env/DEPLOYMENT"}),
            IdentitySource::Reference("oidc/env/DEPLOYMENT".into()),
        ),
        (
            json!({"anthropic_identity_token":"raw-secret-marker"}),
            IdentitySource::Reference("raw-secret-marker".into()),
        ),
        (
            json!({"anthropic_identity_source":"keycloak", "anthropic_identity_token_file":"/deployment/token", "anthropic_identity_token":"oidc/env/DEPLOYMENT"}),
            IdentitySource::Custom,
        ),
    ] {
        let deployment = source_deployment(extra);
        for fleet_kind in [None, Some("internal_issuer")] {
            let selected = select_identity_source(&deployment, |name| {
                match name {
                    "ANTHROPIC_IDENTITY_SOURCE" => fleet_kind,
                    "ANTHROPIC_IDENTITY_TOKEN_FILE" => Some("/fleet/token"),
                    "OXLLM_ANTHROPIC_IDENTITY_TOKEN_REF" => Some("oidc/env/FLEET"),
                    "ANTHROPIC_IDENTITY_TOKEN" => Some("fleet-assertion"),
                    _ => panic!("unexpected environment lookup: {name}"),
                }
                .map(str::to_owned)
            });
            assert_eq!(selected.as_ref(), Some(&expected));
        }
        assert_eq!(
            select_identity_source(&deployment, |_| panic!(
                "deployment source must not read fleet defaults"
            )),
            Some(expected)
        );
    }
}

#[test]
fn fleet_identity_source_precedence_and_presence_are_explicit() {
    for (environment, expected) in [
        (
            vec![
                ("ANTHROPIC_IDENTITY_TOKEN_FILE", "/fleet/token"),
                ("OXLLM_ANTHROPIC_IDENTITY_TOKEN_REF", "oidc/env/FLEET"),
                ("ANTHROPIC_IDENTITY_TOKEN", "fleet-assertion"),
            ],
            Some(IdentitySource::Reference("oidc/file//fleet/token".into())),
        ),
        (
            vec![
                ("ANTHROPIC_IDENTITY_TOKEN_FILE", " \t"),
                ("OXLLM_ANTHROPIC_IDENTITY_TOKEN_REF", "oidc/env/FLEET"),
                ("ANTHROPIC_IDENTITY_TOKEN", "fleet-assertion"),
            ],
            Some(IdentitySource::Reference("oidc/env/FLEET".into())),
        ),
        (
            vec![("ANTHROPIC_IDENTITY_TOKEN", "fleet-assertion")],
            Some(IdentitySource::Reference(
                "oidc/env/ANTHROPIC_IDENTITY_TOKEN".into(),
            )),
        ),
        (
            vec![
                ("OXLLM_ANTHROPIC_IDENTITY_TOKEN_REF", ""),
                ("ANTHROPIC_IDENTITY_TOKEN", ""),
            ],
            Some(IdentitySource::Reference(
                "oidc/env/ANTHROPIC_IDENTITY_TOKEN".into(),
            )),
        ),
        (
            vec![
                ("ANTHROPIC_IDENTITY_SOURCE", "internal_issuer"),
                ("ANTHROPIC_IDENTITY_TOKEN_FILE", "/fleet/token"),
                ("OXLLM_ANTHROPIC_IDENTITY_TOKEN_REF", "oidc/env/FLEET"),
            ],
            Some(IdentitySource::Custom),
        ),
        (
            vec![("ANTHROPIC_IDENTITY_SOURCE", "")],
            Some(IdentitySource::Custom),
        ),
        (Vec::new(), None),
    ] {
        let environment: HashMap<_, _> = environment.into_iter().collect();
        let selected = select_identity_source(&source_deployment(json!({})), |name| {
            environment.get(name).map(|value| (*value).to_owned())
        });
        assert_eq!(selected, expected);
    }
}

#[tokio::test]
async fn invalid_identity_files_are_rejected_before_http_exchange() {
    let server = MockServer::start_async().await;
    let grant = server
        .mock_async(|when, then| {
            when.method(POST).path("/v1/oauth/token");
            then.status(200)
                .json_body(json!({"access_token":"minted", "expires_in":3600}));
        })
        .await;
    let directory = tempfile::tempdir().unwrap();
    let resolver = resolver(&[directory.path()]);
    let mut capped_token_with_whitespace = vec![b'a'; MAX_CREDENTIAL_BYTES];
    capped_token_with_whitespace.extend_from_slice(b"\nignored-suffix");
    let mut short_token_with_whitespace = b"secret-marker".to_vec();
    short_token_with_whitespace.resize(MAX_CREDENTIAL_BYTES + 100, b' ');
    for (label, contents) in [
        ("token over cap", vec![b'a'; MAX_CREDENTIAL_BYTES + 1]),
        ("capped token with suffix", capped_token_with_whitespace),
        ("short token with padding", short_token_with_whitespace),
        ("oversized whitespace", vec![b' '; MAX_CREDENTIAL_BYTES * 2]),
        ("empty", Vec::new()),
        ("whitespace", b" \t\r\n".to_vec()),
        ("invalid UTF-8", b"secret-marker\xff".to_vec()),
        (
            "embedded newline",
            b"secret-marker\r\nheader: value".to_vec(),
        ),
        ("embedded NUL", b"secret-marker\0".to_vec()),
    ] {
        let mut file = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        file.write_all(&contents).unwrap();
        let result = resolver.resolve(&deployment(&server, file.path())).await;
        assert!(result.is_err(), "{label} unexpectedly reached the exchange");
        let error = result.unwrap_err();
        assert_denied(&error, "Anthropic identity token");
        if contents.len() > MAX_CREDENTIAL_BYTES {
            assert!(error.to_string().contains("64 KiB"), "{label}: {error}");
        }
        grant.assert_hits_async(0).await;
    }
}

#[tokio::test]
async fn valid_identity_files_normalize_only_within_the_raw_size_bound() {
    let directory = tempfile::tempdir().unwrap();
    for (contents, expected) in [
        (
            vec![b'a'; MAX_CREDENTIAL_BYTES],
            "a".repeat(MAX_CREDENTIAL_BYTES),
        ),
        (
            b" \tidentity.jwt.signature\r\n".to_vec(),
            "identity.jwt.signature".into(),
        ),
        (
            {
                let mut token = b"identity.jwt.signature".to_vec();
                token.resize(MAX_CREDENTIAL_BYTES, b' ');
                token
            },
            "identity.jwt.signature".into(),
        ),
    ] {
        let mut file = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        file.write_all(&contents).unwrap();
        let token = Identity::File(file.path().into())
            .read(&[directory.path().into()])
            .await
            .unwrap();
        assert_eq!(token.as_str(), expected);
    }
}

#[tokio::test]
async fn disallowed_missing_relative_and_directory_paths_fail_before_exchange() {
    let server = MockServer::start_async().await;
    let grant = server
        .mock_async(|when, then| {
            when.method(POST).path("/v1/oauth/token");
            then.status(200)
                .json_body(json!({"access_token":"minted", "expires_in":3600}));
        })
        .await;
    let directory = tempfile::tempdir().unwrap();
    let allowed = directory.path().join("allowed");
    let adjacent = directory.path().join("allowed-sibling");
    std::fs::create_dir(&allowed).unwrap();
    std::fs::create_dir(&adjacent).unwrap();
    let mut outside = tempfile::NamedTempFile::new_in(&adjacent).unwrap();
    outside.write_all(b"secret-marker").unwrap();
    let empty_roots = resolver(&[]);
    let resolver = resolver(&[&allowed]);
    for (path, expected) in [
        (outside.path().to_path_buf(), CREDENTIAL_DIRS_ENV),
        (
            allowed
                .join("..")
                .join("allowed-sibling")
                .join(outside.path().file_name().unwrap()),
            CREDENTIAL_DIRS_ENV,
        ),
        (allowed.join("missing-secret-marker"), "could not be read"),
        (
            PathBuf::from("relative-secret-marker"),
            "anthropic_identity_token",
        ),
        (allowed.clone(), "could not be read"),
    ] {
        let error = resolver
            .resolve(&deployment(&server, &path))
            .await
            .unwrap_err();
        assert_denied(&error, expected);
        grant.assert_hits_async(0).await;
    }
    let error = empty_roots
        .resolve(&deployment(&server, outside.path()))
        .await
        .unwrap_err();
    assert_denied(&error, CREDENTIAL_DIRS_ENV);
    grant.assert_hits_async(0).await;
}

#[cfg(unix)]
#[tokio::test]
async fn symlink_targets_must_remain_in_an_allowed_credential_directory() {
    let server = MockServer::start_async().await;
    let grant = server
        .mock_async(|when, then| {
            when.method(POST).path("/v1/oauth/token");
            then.status(200)
                .json_body(json!({"access_token":"minted", "expires_in":3600}));
        })
        .await;
    let allowed = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let mut outside_token = tempfile::NamedTempFile::new_in(outside.path()).unwrap();
    outside_token.write_all(b"secret-marker").unwrap();
    let escaping_link = allowed.path().join("escaping-secret-marker");
    std::os::unix::fs::symlink(outside_token.path(), &escaping_link).unwrap();
    let error = resolver(&[allowed.path()])
        .resolve(&deployment(&server, &escaping_link))
        .await
        .unwrap_err();
    assert_denied(&error, CREDENTIAL_DIRS_ENV);
    grant.assert_hits_async(0).await;
    let mut inside_token = tempfile::NamedTempFile::new_in(allowed.path()).unwrap();
    inside_token
        .write_all(b" \tidentity.jwt.signature\n")
        .unwrap();
    let inside_link = allowed.path().join("inside");
    std::os::unix::fs::symlink(inside_token.path(), &inside_link).unwrap();
    assert_eq!(
        Identity::File(inside_link)
            .read(&[allowed.path().into()])
            .await
            .unwrap()
            .as_str(),
        "identity.jwt.signature"
    );
}

#[tokio::test]
async fn required_ids_and_unsupported_sources_are_named_before_file_or_http_access() {
    let server = MockServer::start_async().await;
    let grant = server
        .mock_async(|when, then| {
            when.method(POST).path("/v1/oauth/token");
            then.status(200)
                .json_body(json!({"access_token":"minted", "expires_in":3600}));
        })
        .await;
    let resolver = resolver(&[]);
    for (field, value) in [
        ("anthropic_identity_token_file", "/missing-secret-marker"),
        ("anthropic_identity_token", "raw-secret-marker"),
        ("anthropic_identity_source", "keycloak"),
    ] {
        let mut dep = deployment(&server, Path::new("/missing-secret-marker"));
        dep.oxllm_params.extra.clear();
        dep.oxllm_params.extra.insert(field.into(), json!(value));
        let error = resolver.resolve(&dep).await.unwrap_err();
        assert_denied(&error, "anthropic_federation_rule_id");
        assert_denied(&error, "anthropic_organization_id");
        for (present, missing) in [
            ("anthropic_federation_rule_id", "anthropic_organization_id"),
            ("anthropic_organization_id", "anthropic_federation_rule_id"),
        ] {
            let mut partial = dep.clone();
            partial
                .oxllm_params
                .extra
                .insert(present.into(), json!("id"));
            assert_denied(&resolver.resolve(&partial).await.unwrap_err(), missing);
        }
    }
    for kind in ["keycloak", "internal_issuer", "unknown-secret-marker", " "] {
        let mut dep = deployment(&server, Path::new("/missing-secret-marker"));
        dep.oxllm_params
            .extra
            .insert("anthropic_identity_source".into(), json!(kind));
        let error = resolver.resolve(&dep).await.unwrap_err();
        assert_denied(&error, "anthropic_identity_source");
        assert!(
            error
                .to_string()
                .contains("custom registered credential resolver")
        );
        dep.oxllm_params.api_key = Some("static-key".into());
        assert_eq!(
            resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
            Some("static-key")
        );
    }
    for reference in [
        "raw-secret-marker",
        "oidc/google/audience-secret-marker",
        "oidc/env_path/SECRET",
        "oidc/env/INVALID-NAME",
    ] {
        let mut dep = deployment(&server, Path::new("/missing-secret-marker"));
        dep.oxllm_params
            .extra
            .remove("anthropic_identity_token_file");
        dep.oxllm_params
            .extra
            .insert("anthropic_identity_token".into(), json!(reference));
        assert_denied(
            &resolver.resolve(&dep).await.unwrap_err(),
            "anthropic_identity_token",
        );
    }
    grant.assert_hits_async(0).await;
}

#[tokio::test]
async fn exchange_authorities_enforce_https_userinfo_and_explicit_port_pins_before_reading() {
    let server = MockServer::start_async().await;
    let grant = server
        .mock_async(|when, then| {
            when.method(POST).path("/v1/oauth/token");
            then.status(200)
                .json_body(json!({"access_token":"minted", "expires_in":3600}));
        })
        .await;
    let mut resolver = resolver(&[]);
    resolver.allow_http = false;
    resolver.trusted_hosts = [
        "Gateway.Example:8443",
        "https://secure.example:443/v1",
        "unpinned.example",
    ]
    .into_iter()
    .map(|entry| allowlisted_authority(entry).unwrap())
    .collect();
    assert_eq!(
        allowlisted_authority("https://secure.example:443/v1"),
        Some(("secure.example".into(), Some(443)))
    );
    assert_eq!(
        allowlisted_authority("https://[::1]:443"),
        Some(("[::1]".into(), Some(443)))
    );
    for entry in [
        "gateway.example:65536",
        "gateway.example:not-a-port",
        "https://",
    ] {
        assert!(allowlisted_authority(entry).is_none());
    }
    let mut dep = deployment(&server, Path::new("/missing-secret-marker"));
    for base in [
        "https://gateway.example:8443/v1/messages",
        "https://secure.example/v1",
        "https://secure.example:443/v1",
        "https://unpinned.example:9443/v1",
    ] {
        dep.oxllm_params.api_base = Some(base.into());
        assert!(
            resolver
                .endpoint(&dep)
                .unwrap()
                .ends_with("/v1/oauth/token")
        );
    }
    for (base, expected) in [
        ("http://gateway.example:8443/v1", "HTTPS"),
        (
            "https://secret-marker@gateway.example:8443/v1",
            "embedded credentials",
        ),
        (
            "https://:secret-marker@gateway.example:8443/v1",
            "embedded credentials",
        ),
        ("https://gateway.example/v1", TRUSTED_HOSTS_ENV),
        ("https://gateway.example:8444/v1", TRUSTED_HOSTS_ENV),
        ("https://secure.example:8443/v1", TRUSTED_HOSTS_ENV),
        (
            "https://gateway.example.evil.test:8443/v1",
            TRUSTED_HOSTS_ENV,
        ),
    ] {
        dep.oxllm_params.api_base = Some(base.into());
        assert_denied(&resolver.resolve(&dep).await.unwrap_err(), expected);
    }
    resolver.trusted_hosts = vec![("api.anthropic.com".into(), None)];
    dep.oxllm_params.api_base = Some("https://api.anthropic.com:8443/v1".into());
    assert_eq!(
        resolver.endpoint(&dep).unwrap(),
        "https://api.anthropic.com:8443/v1/oauth/token"
    );
    grant.assert_hits_async(0).await;
}
