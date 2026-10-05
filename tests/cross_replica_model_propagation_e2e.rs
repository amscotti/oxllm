//! Integration test for cross-replica model-config propagation (C65 core).
//!
//! Boots two in-process oxllm replicas sharing one Redis (pub/sub) and one
//! in-memory SQLite DB. A model added via replica A's management API must
//! appear on replica B's live deployment list within a bounded timeout, driven
//! by the `__model_list__` pub/sub invalidation + debounced `refresh_db_models`
//! path. A delete on A must likewise vanish from B.
//!
//! Skipped only when `OXLLM_REDIS_URL` is unset/empty. A configured Redis
//! that cannot connect or ping fails the test. Run locally with:
//!
//! ```sh
//! docker run -d -p 6379:6379 redis:7-alpine
//! OXLLM_REDIS_URL=redis://127.0.0.1:6379 \
//!   cargo test --test cross_replica_model_propagation_e2e -- --nocapture
//! ```

mod common;

use std::sync::Arc;
use std::time::Duration;

use oxllm_cache_redis::RedisCache;
use oxllm_core::Cache as _;
use oxllm_server::app::{build_router, invalidation_subscriber_loop};
use oxllm_server::cache::{DualCache, InMemoryCache};
use oxllm_server::config::types::{GeneralSettings, OxllmSettings, ProxyConfig};
use oxllm_server::{Database, ModelConfigRecord, OxllmParams, SharedState, now_iso};

const MASTER_KEY: &str = "sk-xrep-test-key-0123";
const PREFIX: &str = "oxllm-xrep";

async fn connect_redis() -> Option<RedisCache> {
    common::connect_redis(PREFIX).await
}

async fn fresh_db() -> Arc<Database> {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect sqlite memory");
    db.migrate().await.expect("migrate");
    db
}

fn base_config() -> ProxyConfig {
    ProxyConfig {
        model_list: vec![],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

/// One replica: a live `SharedState` (DualCache over the shared Redis + shared
/// DB) plus an HTTP router bound to a free port. Replica B additionally runs
/// the cross-replica invalidation subscriber task so it picks up peer
/// publishes; replica A refreshes itself synchronously inside the management
/// handler, so it does not need its own subscriber for the add/delete paths.
struct Replica {
    shared: SharedState,
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
    subscriber_shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    subscriber_join: Option<tokio::task::JoinHandle<()>>,
}

impl Replica {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        if let Some(tx) = self.subscriber_shutdown {
            let _ = tx.send(());
        }
        if let Some(join) = self.subscriber_join {
            let _ = tokio::time::timeout(Duration::from_secs(2), join).await;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

/// Wait until the invalidation subscriber has actually landed its SUBSCRIBE
/// on Redis.
///
/// Redis pub/sub has no replay buffer: anything published before the SUBSCRIBE
/// completes is silently dropped. The previous fixed 150ms sleep was a race —
/// on a slow CI the first `__model_list__` publish could beat B's subscription
/// and the propagation test would flake at its 5s deadline. This probe seeds a
/// sentinel key in the subscriber's memory cache, publishes an invalidation
/// for it, and waits for the subscriber to delete it — end-to-end proof that
/// the subscription is live before the caller proceeds.
async fn wait_until_subscribed(redis: &RedisCache, memory: &Arc<InMemoryCache>) {
    const PROBE_KEY: &str = "__xrep_subscriber_ready__";
    memory.set(PROBE_KEY, "1".to_string(), None).await;
    let channel = redis.pubsub_channel(oxllm_server::cache::INVALIDATION_CHANNEL_BASE);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        redis
            .publish(&channel, PROBE_KEY)
            .await
            .expect("publish readiness probe");
        if memory.get(PROBE_KEY).await.is_none() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "invalidation subscriber never became ready within 5s \
             (probe key '{PROBE_KEY}' was never invalidated)"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Boot one replica. `run_subscriber` controls whether the invalidation
/// subscriber task is spawned (B subscribes; A does not need to for these
/// tests, since A mutates and refreshes synchronously).
async fn boot_replica(
    config: ProxyConfig,
    redis: RedisCache,
    db: Arc<Database>,
    run_subscriber: bool,
) -> Replica {
    let memory = Arc::new(InMemoryCache::default());
    let cache: Arc<dyn oxllm_core::Cache> =
        Arc::new(DualCache::new(memory.clone(), Some(redis.clone())));
    let registry = oxllm_providers::default_registry();
    let state =
        oxllm_server::AppState::with_registry_cache_db(config, registry, cache, None, Some(db))
            .expect("build state");
    let shared: SharedState = Arc::new(arc_swap::ArcSwap::from_pointee(state));

    let (subscriber_shutdown, subscriber_join) = if run_subscriber {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let sub_shared = shared.clone();
        let sub_memory = memory.clone();
        let sub_redis = redis.clone();
        let join = tokio::spawn(async move {
            invalidation_subscriber_loop(sub_redis, sub_memory, sub_shared, rx).await;
        });
        (Some(tx), Some(join))
    } else {
        (None, None)
    };

    // Confirm the subscriber's SUBSCRIBE landed on Redis before anything
    // publishes, so the first management mutation is not missed (pub/sub has
    // no replay buffer — a fixed sleep here was a startup race).
    if run_subscriber {
        wait_until_subscribed(&redis, &memory).await;
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let router = build_router(shared.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });

    Replica {
        shared,
        addr,
        shutdown_tx,
        subscriber_shutdown,
        subscriber_join,
    }
}

/// Wait up to `timeout` for `model_name` to appear in `replica`'s live
/// deployment list. Returns true on success.
async fn wait_until_visible(replica: &Replica, model_name: &str, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let has = replica
            .shared
            .load_full()
            .deployments
            .iter()
            .any(|d| d.model_name == model_name);
        if has {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn auth_header() -> String {
    format!("Bearer {MASTER_KEY}")
}

/// A model added on replica A must propagate to replica B's live deployment
/// list via the `__model_list__` pub/sub invalidation + debounced refresh.
#[tokio::test]
async fn add_on_a_appears_on_b() {
    let Some(redis) = connect_redis().await else {
        eprintln!("SKIP: OXLLM_REDIS_URL is unset/empty");
        return;
    };
    let db = fresh_db().await;

    let a = boot_replica(base_config(), redis.clone(), db.clone(), false).await;
    let b = boot_replica(base_config(), redis, db.clone(), true).await;

    let client = common::http_client();
    let resp = client
        .post(a.url("/model/new"))
        .header("authorization", auth_header())
        .json(&serde_json::json!({
            "model_name": "xrep-added",
            "oxllm_params": OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                ..Default::default()
            }
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        200,
        "add on A must succeed: {}",
        resp.text().await.unwrap_or_default()
    );

    let appeared = wait_until_visible(&b, "xrep-added", Duration::from_secs(5)).await;
    assert!(
        appeared,
        "model added on A must propagate to B within the bounded timeout"
    );

    a.shutdown().await;
    b.shutdown().await;
}

/// A model deleted on replica A must vanish from replica B's live list.
#[tokio::test]
async fn delete_on_a_disappears_on_b() {
    let Some(redis) = connect_redis().await else {
        eprintln!("SKIP: OXLLM_REDIS_URL is unset/empty");
        return;
    };
    let db = fresh_db().await;

    // Seed the shared DB with a row before either replica boots, so both load
    // it on their first refresh.
    db.create_model(&ModelConfigRecord {
        model_id: "model-xrep-seed".to_string(),
        model_name: "xrep-seeded".to_string(),
        oxllm_params: serde_json::json!({"model": "openai/gpt-4o", "api_key": "sk-test"}),
        model_info: None,
        created_at: now_iso(),
        updated_at: now_iso(),
    })
    .await
    .expect("seed");

    let a = boot_replica(base_config(), redis.clone(), db.clone(), false).await;
    let b = boot_replica(base_config(), redis, db.clone(), true).await;

    // Both replicas must start with the seeded model visible (A refreshes on
    // boot via refresh_db_models is not automatic in the harness, so force it).
    oxllm_server::state::refresh_db_models(&a.shared)
        .await
        .expect("refresh a");
    oxllm_server::state::refresh_db_models(&b.shared)
        .await
        .expect("refresh b");
    assert!(
        wait_until_visible(&b, "xrep-seeded", Duration::from_secs(2)).await,
        "seeded model must be visible on B before delete"
    );

    let client = common::http_client();
    let resp = client
        .post(a.url("/model/delete"))
        .header("authorization", auth_header())
        .json(&serde_json::json!({"model_id": "model-xrep-seed"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        200,
        "delete on A must succeed: {}",
        resp.text().await.unwrap_or_default()
    );

    // Wait for B to drop the model.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let still_has = b
            .shared
            .load_full()
            .deployments
            .iter()
            .any(|d| d.model_name == "xrep-seeded");
        if !still_has {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("model deleted on A must vanish from B within the bounded timeout");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    a.shutdown().await;
    b.shutdown().await;
}

/// A burst of adds on A must leave every model visible on B after the
/// debounced refresh path runs. This observes final propagation; it does not
/// instrument the number of refreshes (a slow burst may span debounce windows).
#[tokio::test]
async fn burst_of_adds_coalesces_but_all_land_on_b() {
    let Some(redis) = connect_redis().await else {
        eprintln!("SKIP: OXLLM_REDIS_URL is unset/empty");
        return;
    };
    let db = fresh_db().await;

    let a = boot_replica(base_config(), redis.clone(), db.clone(), false).await;
    let b = boot_replica(base_config(), redis, db.clone(), true).await;

    let client = common::http_client();
    let auth = auth_header();
    let mut tasks = Vec::new();
    for i in 0..5 {
        let url = a.url("/model/new");
        let auth = auth.clone();
        let client = client.clone();
        tasks.push(tokio::spawn(async move {
            let resp = client
                .post(&url)
                .header("authorization", &auth)
                .json(&serde_json::json!({
                    "model_name": format!("xrep-burst-{i}"),
                    "oxllm_params": OxllmParams {
                        model: "openai/gpt-4o".to_string(),
                        api_key: Some("sk-test".to_string()),
                        ..Default::default()
                    }
                }))
                .send()
                .await
                .expect("ok");
            assert_eq!(resp.status(), 200, "burst add {i} must succeed");
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }

    // Allow a full debounce + refresh + slack for every model to arrive.
    for i in 0..5 {
        let name = format!("xrep-burst-{i}");
        let appeared = wait_until_visible(&b, &name, Duration::from_secs(6)).await;
        assert!(
            appeared,
            "bursted model {name} must propagate to B after debounce coalescing"
        );
    }

    a.shutdown().await;
    b.shutdown().await;
}
