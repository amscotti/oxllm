use super::*;
use crate::config::types::{GeneralSettings, OxllmSettings, RouterSettings};

fn config(force: Option<bool>) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![],
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            http_force_ipv4: force,
            ..Default::default()
        },
        oxllm_settings: OxllmSettings::default(),
        router_settings: RouterSettings::default(),
        extra: serde_json::Map::new(),
    }
}

#[test]
fn legacy_ipv4_switch_is_typed_and_native_setting_takes_precedence() {
    let mut parsed = crate::config::parse_config_str("model_list: []\nlitellm_settings:\n  force_ipv4: true\ngeneral_settings:\n  http_force_ipv4: false\n").unwrap();
    assert!(!http_ipv4_only(&parsed));
    parsed.general_settings.http_force_ipv4 = None;
    assert!(http_ipv4_only(&parsed));
    let alias =
        crate::config::parse_config_str("model_list: []\ngeneral_settings:\n  force_ipv4: true\n")
            .unwrap();
    assert_eq!(alias.general_settings.http_force_ipv4, Some(true));
    assert!(http_ipv4_only(&alias));
}

#[tokio::test]
async fn ipv4_resolver_filters_dual_stack_localhost_and_keeps_private_policy() {
    use reqwest::dns::Resolve;
    let resolver = crate::security::SsrfResolver::with_options(true, true);
    let addresses: Vec<_> = resolver
        .resolve("localhost".parse().unwrap())
        .await
        .unwrap()
        .collect();
    assert!(!addresses.is_empty());
    assert!(addresses.iter().all(|address| address.is_ipv4()));
    let strict = crate::security::SsrfResolver::with_options(false, true);
    assert!(strict.resolve("localhost".parse().unwrap()).await.is_err());
}

#[tokio::test]
async fn ipv4_client_connects_to_ipv4_and_rejects_literal_ipv6_without_contacting_it() {
    use axum::{Router, routing::get};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let v4 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let v4_url = format!("http://localhost:{}", v4.local_addr().unwrap().port());
    let v6 = tokio::net::TcpListener::bind("[::1]:0").await.unwrap();
    let v6_url = format!("http://{}", v6.local_addr().unwrap());
    let hits = Arc::new(AtomicUsize::new(0));
    let v6_app = Router::new().route(
        "/",
        get({
            let hits = hits.clone();
            move || {
                let hits = hits.clone();
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    "v6"
                }
            }
        }),
    );
    let v4_task = tokio::spawn(async move {
        axum::serve(v4, Router::new().route("/", get(|| async { "v4" })))
            .await
            .unwrap()
    });
    let v6_task = tokio::spawn(async move { axum::serve(v6, v6_app).await.unwrap() });
    let normal = build_http_client(&config(Some(false))).unwrap();
    assert_eq!(
        normal
            .get(&v6_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "v6"
    );
    let forced = build_http_client(&config(Some(true))).unwrap();
    assert_eq!(
        forced
            .get(&v4_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "v4"
    );
    assert!(
        forced
            .get(&v6_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .is_err()
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    // Image downloads retain their stricter DNS policy even with the operator
    // private-base opt-in used by the upstream client.
    let images = build_image_http_client(&config(Some(true))).unwrap();
    assert!(
        images
            .get(&v4_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .is_err()
    );
    v4_task.abort();
    v6_task.abort();
}
