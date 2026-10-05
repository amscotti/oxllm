//! Integration test for cross-replica cache invalidation via Redis pub/sub.
//!
//! Skipped only when `OXLLM_REDIS_URL` is unset/empty. A configured Redis
//! that cannot connect or ping fails the test. Run locally with:
//!
//! ```sh
//! docker run -d -p 6379:6379 redis:7-alpine
//! OXLLM_REDIS_URL=redis://127.0.0.1:6379 cargo test --test redis_invalidation_e2e -- --nocapture
//! ```

mod common;

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use oxllm_cache_redis::RedisCache;
use oxllm_core::Cache as _;
use oxllm_server::app::invalidation_subscriber_loop;
use oxllm_server::cache::{DualCache, InMemoryCache};
use oxllm_server::config::types::{GeneralSettings, ProxyConfig};
use oxllm_server::{AppState, SharedState};

async fn connect_redis() -> Option<RedisCache> {
    common::connect_redis("oxllm-test").await
}

struct Subscriber {
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl Subscriber {
    fn start(redis: RedisCache, memory: Arc<InMemoryCache>) -> Self {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: GeneralSettings {
                master_key: Some("sk-invalidation-test-key".to_string()),
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let cache: Arc<dyn oxllm_core::Cache> =
            Arc::new(DualCache::new(memory.clone(), Some(redis.clone())));
        let state = AppState::with_registry_cache_db(
            config,
            oxllm_providers::default_registry(),
            cache,
            None,
            None,
        )
        .expect("build subscriber state");
        let shared: SharedState = Arc::new(arc_swap::ArcSwap::from_pointee(state));
        let (shutdown, receiver) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(invalidation_subscriber_loop(
            redis, memory, shared, receiver,
        ));
        Self {
            shutdown: Some(shutdown),
            task,
        }
    }

    async fn stop(mut self) {
        self.shutdown
            .take()
            .expect("shutdown sender")
            .send(())
            .expect("subscriber must still be running before cancellation");
        tokio::time::timeout(Duration::from_secs(2), &mut self.task)
            .await
            .expect("production subscriber did not honor cancellation")
            .expect("production subscriber panicked");
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        // Clean up even when a readiness or invalidation assertion panics.
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.abort();
    }
}

/// Match the cross-replica model test's sentinel probe: Redis pub/sub cannot
/// replay messages published before SUBSCRIBE, so a fixed startup sleep races.
async fn wait_until_subscribed(redis: &RedisCache, memory: &InMemoryCache) {
    const PROBE_KEY: &str = "vk:invalidation-subscriber-ready";
    memory.set(PROBE_KEY, "probe".to_string(), None).await;
    let channel = redis.pubsub_channel(oxllm_server::cache::INVALIDATION_CHANNEL_BASE);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            redis
                .publish(&channel, PROBE_KEY)
                .await
                .expect("publish subscription readiness probe");
            if memory.get(PROBE_KEY).await.is_none() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("production invalidation subscriber never became ready");
}

async fn wait_until_invalidated(memory: &InMemoryCache, key: &str) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while memory.get(key).await.is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("production subscriber did not clear the local entry");
}

#[tokio::test]
async fn publish_delivers_to_subscriber() {
    let Some(redis) = connect_redis().await else {
        eprintln!("SKIP: OXLLM_REDIS_URL is unset/empty");
        return;
    };
    let channel = redis.pubsub_channel("test-publish");

    let mut stream = redis
        .subscribe(&channel)
        .await
        .expect("subscribe must succeed on a live redis");

    // SUBSCRIBE has acknowledged before publishing; no scheduler delay is
    // needed to guess whether the receiver is ready.
    redis
        .publish(&channel, "vk:publish-test")
        .await
        .expect("publish");

    let received = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("timed out waiting for pub/sub message")
        .expect("stream ended unexpectedly")
        .get_payload::<String>()
        .expect("payload decode");

    assert_eq!(received, "vk:publish-test");
}

#[tokio::test]
async fn dual_cache_publish_invalidation_reaches_subscriber() {
    let Some(redis) = connect_redis().await else {
        eprintln!("SKIP: OXLLM_REDIS_URL is unset/empty");
        return;
    };

    let memory = Arc::new(InMemoryCache::default());
    let dual = Arc::new(DualCache::new(memory.clone(), Some(redis.clone())));

    memory
        .set("vk:sub", "warm".to_string(), Some(Duration::from_secs(300)))
        .await;
    assert_eq!(memory.get("vk:sub").await.as_deref(), Some("warm"));

    // DualCache::publish_invalidation now namespaces the channel with the
    // configured prefix, so the subscriber must listen on the prefixed
    // channel too.
    let channel = redis.pubsub_channel("invalidate");
    let mut stream = redis.subscribe(&channel).await.expect("subscribe");

    dual.delete("vk:sub").await;
    dual.publish_invalidation("vk:sub").await;

    let received = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("timed out waiting for invalidation message")
        .expect("stream ended unexpectedly")
        .get_payload::<String>()
        .expect("payload decode");
    assert_eq!(received, "vk:sub");

    assert!(
        memory.get("vk:sub").await.is_none(),
        "publisher-side memory must be cleared by delete()"
    );
}

#[tokio::test]
async fn subscriber_clears_local_memory_on_invalidation_message() {
    let Some(redis) = connect_redis().await else {
        eprintln!("SKIP: OXLLM_REDIS_URL is unset/empty");
        return;
    };

    let replica_memory = Arc::new(InMemoryCache::default());
    let subscriber = Subscriber::start(redis.clone(), replica_memory.clone());
    wait_until_subscribed(&redis, &replica_memory).await;
    replica_memory
        .set("vk:untouched", "keep".to_string(), None)
        .await;
    let publisher = DualCache::new(Arc::new(InMemoryCache::default()), Some(redis));

    // More than one message must be handled by the long-running production
    // loop; the readiness probe already exercised its first invalidation.
    for key in ["vk:replica", "vk:second-invalidation"] {
        replica_memory.set(key, "stale".to_string(), None).await;
        assert_eq!(replica_memory.get(key).await.as_deref(), Some("stale"));
        publisher.publish_invalidation(key).await;
        wait_until_invalidated(&replica_memory, key).await;
        assert_eq!(
            replica_memory.get("vk:untouched").await.as_deref(),
            Some("keep")
        );
    }
    // Cancellation is tested while Redis is subscribed and no events are
    // pending, rather than relying on a one-shot fake receiver terminating.
    subscriber.stop().await;
}
