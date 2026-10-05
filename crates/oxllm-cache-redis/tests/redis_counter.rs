//! Integration tests for Redis counter semantics: fractional precision, the
//! atomic fixed-window TTL on `incr`, sub-second TTL precision, and pub/sub
//! prefix isolation.
//!
//! These compile only with the `redis` feature. Each test silently skips when
//! no Redis server is reachable via `OXLLM_REDIS_URL` (defaults to
//! `redis://127.0.0.1:6379`). Run locally with:
//!
//! ```sh
//! docker run -d -p 6379:6379 redis:7-alpine
//! OXLLM_REDIS_URL=redis://127.0.0.1:6379 \
//!   cargo test -p oxllm-cache-redis --features redis --test redis_counter -- --nocapture
//! ```

#![cfg(feature = "redis")]

use std::time::Duration;

use futures_util::StreamExt;
use oxllm_cache_redis::RedisCache;
use oxllm_core::Cache as _;
use oxllm_core::IncrResult;
use redis::cmd;

/// Distinct prefix so these tests do not collide with the data keys of the
/// pub/sub invalidation e2e suite (which uses `oxllm-test`).
const PREFIX: &str = "oxllm-counter-test";

fn redis_url() -> Option<String> {
    let raw = std::env::var("OXLLM_REDIS_URL").ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Connect a `RedisCache` plus a raw client for direct TTL/value verification.
/// Returns `None` (so the caller can skip) when no Redis is reachable.
async fn connect() -> Option<(RedisCache, redis::Client)> {
    let url = redis_url()?;
    let connect = RedisCache::new(&url, PREFIX);
    let cache = match tokio::time::timeout(Duration::from_secs(2), connect).await {
        Ok(Ok(c)) if c.ping().await => c,
        Ok(Ok(_)) => {
            eprintln!("skipping: redis ping failed at {url}");
            return None;
        }
        Ok(Err(e)) => {
            eprintln!("skipping: cannot connect to redis at {url}: {e}");
            return None;
        }
        Err(_) => {
            eprintln!("skipping: redis connect timed out at {url}");
            return None;
        }
    };
    let raw = redis::Client::open(url.as_str()).ok()?;
    Some((cache, raw))
}

async fn del(client: &redis::Client, raw_key: &str) {
    let Ok(mut conn) = client.get_multiplexed_async_connection().await else {
        return;
    };
    let _: redis::RedisResult<()> = cmd("DEL").arg(raw_key).query_async(&mut conn).await;
}

/// Server-side remaining TTL in milliseconds for `raw_key`.
async fn pttl_ms(client: &redis::Client, raw_key: &str) -> Option<i64> {
    let mut conn = client.get_multiplexed_async_connection().await.ok()?;
    cmd("PTTL")
        .arg(raw_key)
        .query_async::<i64>(&mut conn)
        .await
        .ok()
}

/// Read the raw stored string for `raw_key` (bypassing the cache's prefixing).
async fn raw_get(client: &redis::Client, raw_key: &str) -> Option<String> {
    let mut conn = client.get_multiplexed_async_connection().await.ok()?;
    cmd("GET")
        .arg(raw_key)
        .query_async::<Option<String>>(&mut conn)
        .await
        .ok()?
}

#[tokio::test]
async fn zero_ttl_expires_values_and_new_counter_windows_immediately() {
    let Some((cache, client)) = connect().await else {
        return;
    };
    let key = "zero-ttl:value";
    cache.set(key, "old".to_string(), None).await;
    assert_eq!(cache.get(key).await.as_deref(), Some("old"));
    cache
        .set(key, "expired".to_string(), Some(Duration::ZERO))
        .await;
    assert!(raw_get(&client, &format!("{PREFIX}:{key}")).await.is_none());

    let counter = "zero-ttl:counter";
    cache.delete(counter).await;
    assert_eq!(
        cache
            .incr(counter, 2.0, Some(Duration::ZERO))
            .await
            .unwrap(),
        2.0
    );
    assert!(cache.counter_value(counter).await.is_none());
    assert_eq!(
        cache
            .incr_if_below(counter, 2.0, 3.0, Some(Duration::ZERO))
            .await
            .unwrap(),
        IncrResult::Allowed(2.0)
    );
    assert!(cache.counter_value(counter).await.is_none());

    cache
        .incr(counter, 1.0, Some(Duration::from_secs(60)))
        .await
        .unwrap();
    cache
        .incr(counter, 1.0, Some(Duration::ZERO))
        .await
        .unwrap();
    cache
        .incr_if_below(counter, 1.0, 4.0, Some(Duration::ZERO))
        .await
        .unwrap();
    assert_eq!(cache.counter_value(counter).await, Some(3.0));
    cache.delete(counter).await;
}

#[tokio::test]
async fn get_with_ttl_distinguishes_expiring_unbounded_and_missing_values() {
    let Some((cache, _client)) = connect().await else {
        return;
    };
    let key = "read-with-ttl:value";
    cache.delete(key).await;
    assert!(cache.get_with_ttl(key).await.is_none());
    cache.set(key, "unbounded".to_string(), None).await;
    assert_eq!(
        cache.get_with_ttl(key).await,
        Some(("unbounded".to_string(), None))
    );
    cache
        .set(key, "bounded".to_string(), Some(Duration::from_secs(60)))
        .await;
    let (value, ttl) = cache.get_with_ttl(key).await.unwrap();
    assert_eq!(value, "bounded");
    let ttl = ttl.expect("bounded value must carry its remaining TTL");
    assert!(!ttl.is_zero() && ttl <= Duration::from_secs(60));
    cache.delete(key).await;
}

#[tokio::test]
async fn corrupted_multibyte_counter_does_not_panic_during_logging() {
    let Some((cache, _client)) = connect().await else {
        return;
    };
    let key = "corrupt-counter:utf8";
    cache
        .set(key, "米".repeat(300), Some(Duration::from_secs(60)))
        .await;
    assert_eq!(cache.try_counter_value(key).await.unwrap(), None);
    cache.delete(key).await;
}

#[tokio::test]
async fn incr_if_below_returns_precise_fractional_allowed_value() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let k = "frac:allowed";
    del(&client, &format!("{PREFIX}:{k}")).await;

    // current = 0, delta = 0.25. Without the string-return fix, Lua's bare
    // number reply would be integer-truncated to Allowed(0.0).
    let res = cache.incr_if_below(k, 0.25, 10.0, None).await.unwrap();
    assert_eq!(res, IncrResult::Allowed(0.25));

    // The stored value must match the returned value exactly.
    let stored = raw_get(&client, &format!("{PREFIX}:{k}")).await.unwrap();
    assert_eq!(stored.parse::<f64>().unwrap(), 0.25);
}

#[tokio::test]
async fn incr_if_below_returns_precise_current_on_zero_delta() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let k = "frac:current";
    del(&client, &format!("{PREFIX}:{k}")).await;

    cache.incr_if_below(k, 0.25, 10.0, None).await.unwrap();
    // A zero-delta probe must carry back the real (fractional) current value,
    // not the integer-truncated 0.
    let res = cache.incr_if_below(k, 0.0, 10.0, None).await.unwrap();
    assert_eq!(res, IncrResult::Allowed(0.25));
}

#[tokio::test]
async fn incr_if_below_rejects_with_precise_current() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let k = "frac:reject";
    del(&client, &format!("{PREFIX}:{k}")).await;

    cache.incr_if_below(k, 10.5, 10.5, None).await.unwrap();
    // At the ceiling (10.5); a further 0.5 must be rejected with the precise
    // current (10.5), not 10.
    let res = cache.incr_if_below(k, 0.5, 10.5, None).await.unwrap();
    assert_eq!(res, IncrResult::Rejected(10.5));
}

#[tokio::test]
async fn incr_sets_ttl_on_first_creation() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let k = "incr:ttl";
    let raw = format!("{PREFIX}:{k}");
    del(&client, &raw).await;

    let new_val = cache
        .incr(k, 1.0, Some(Duration::from_secs(60)))
        .await
        .unwrap();
    assert_eq!(new_val, 1.0);

    let ttl = pttl_ms(&client, &raw).await.expect("pttl must be readable");
    assert!(
        ttl > 0 && ttl <= 60_000,
        "expected a fixed-window TTL in (0, 60000] ms, got {ttl}"
    );
}

#[tokio::test]
async fn incr_does_not_refresh_existing_window_ttl() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let k = "incr:fixed";
    let raw = format!("{PREFIX}:{k}");
    del(&client, &raw).await;

    cache
        .incr(k, 1.0, Some(Duration::from_secs(60)))
        .await
        .unwrap();
    // Let enough of the window elapse that a refresh would be detectable.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let ttl_before = pttl_ms(&client, &raw).await.unwrap();

    // Second incr must NOT slide the window back out to 60s (non-sliding
    // semantics): it should leave the remaining TTL essentially unchanged.
    cache
        .incr(k, 1.0, Some(Duration::from_secs(60)))
        .await
        .unwrap();
    let ttl_after = pttl_ms(&client, &raw).await.unwrap();

    let drift = ttl_before.abs_diff(ttl_after);
    assert!(
        drift < 750,
        "fixed-window TTL was refreshed by a subsequent incr \
         (before={ttl_before}, after={ttl_after})"
    );

    let val = cache.incr(k, 0.0, None).await.unwrap();
    assert_eq!(val, 2.0);
}

#[tokio::test]
async fn incr_concurrent_all_land_in_one_window() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let k = "incr:concurrent";
    let raw = format!("{PREFIX}:{k}");
    del(&client, &raw).await;

    let n = 16usize;
    let mut handles = Vec::with_capacity(n);
    for _ in 0..n {
        let cache = cache.clone();
        handles.push(tokio::spawn(async move {
            cache.incr(k, 1.0, Some(Duration::from_secs(60))).await
        }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }

    let total = raw_get(&client, &raw)
        .await
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert_eq!(total, n as f64);
    let ttl = pttl_ms(&client, &raw).await.unwrap();
    assert!(ttl > 0 && ttl <= 60_000);
}

#[tokio::test]
async fn set_uses_sub_second_ttl_precision() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let k = "set:subsecond";
    let raw = format!("{PREFIX}:{k}");
    del(&client, &raw).await;

    cache
        .set(k, "v".to_string(), Some(Duration::from_millis(500)))
        .await;
    let ttl = pttl_ms(&client, &raw).await.expect("pttl readable");
    assert!(
        ttl > 0 && ttl <= 1000,
        "sub-second TTL should keep millisecond precision (<=1000ms); got {ttl}"
    );
}

#[tokio::test]
async fn pubsub_channels_are_isolated_by_prefix() {
    let Some((cache_a, _)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    let Some(url) = redis_url() else {
        return;
    };
    let cache_b =
        match tokio::time::timeout(Duration::from_secs(2), RedisCache::new(&url, "oxllm-other"))
            .await
        {
            Ok(Ok(c)) => c,
            _ => return,
        };

    let chan_a = cache_a.pubsub_channel("invalidate");
    let chan_b = cache_b.pubsub_channel("invalidate");
    assert_eq!(chan_a, format!("{PREFIX}:invalidate"));
    assert_eq!(chan_b, "oxllm-other:invalidate");
    assert_ne!(chan_a, chan_b);

    let mut stream_b = cache_b.subscribe(&chan_b).await.expect("subscribe b");

    let publisher = cache_a.clone();
    let chan_for_task = chan_a.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        publisher.publish(&chan_for_task, "leak?").await.unwrap();
    });

    // B's subscriber on its own prefixed channel must NOT receive A's message
    // published on a different channel.
    let leaked = tokio::time::timeout(Duration::from_millis(500), stream_b.next()).await;
    assert!(
        leaked.is_err(),
        "cross-prefix pub/sub leak: B received a message published on A's channel"
    );

    // Positive control: B publishing on its own channel reaches its subscriber.
    cache_b.publish(&chan_b, "own").await.unwrap();
    let msg = tokio::time::timeout(Duration::from_secs(2), stream_b.next())
        .await
        .expect("timed out waiting for own-channel message")
        .expect("stream ended unexpectedly")
        .get_payload::<String>()
        .expect("payload decode");
    assert_eq!(msg, "own");
}

#[tokio::test]
async fn incr_if_below_preserves_full_f64_precision_like_incr() {
    let Some((cache, client)) = connect().await else {
        eprintln!("skipping: no redis available (set OXLLM_REDIS_URL)");
        return;
    };
    // 1.0/3.0 needs 16 significant digits to round-trip exactly; Lua's default
    // `tostring` formats with `%.14g` and would store/return a value that
    // parses to a *different* f64 than the `incr` (INCRBYFLOAT) path.
    let delta = 1.0_f64 / 3.0;
    assert_ne!(
        format!("{delta:.14e}").parse::<f64>().unwrap(),
        delta,
        "test premise: %.14g must not round-trip this value"
    );

    let k_gate = "frac:precise:gate";
    let k_incr = "frac:precise:incr";
    let raw_gate = format!("{PREFIX}:{k_gate}");
    let raw_incr = format!("{PREFIX}:{k_incr}");
    del(&client, &raw_gate).await;
    del(&client, &raw_incr).await;

    let res = cache
        .incr_if_below(k_gate, delta, 10.0, None)
        .await
        .unwrap();
    assert_eq!(res, IncrResult::Allowed(delta));

    let incr_val = cache.incr(k_incr, delta, None).await.unwrap();
    assert_eq!(incr_val, delta);

    let stored_gate = raw_get(&client, &raw_gate).await.unwrap();
    let stored_incr = raw_get(&client, &raw_incr).await.unwrap();

    // Both paths must store a representation that parses back to the exact f64.
    assert_eq!(stored_gate.parse::<f64>().unwrap(), delta);
    assert_eq!(stored_incr.parse::<f64>().unwrap(), delta);
    // The two primitives on the same counter key must agree bit-for-bit.
    assert_eq!(
        stored_gate.parse::<f64>().unwrap().to_bits(),
        delta.to_bits(),
        "incr_if_below lost precision (stored {stored_gate:?})"
    );
}
