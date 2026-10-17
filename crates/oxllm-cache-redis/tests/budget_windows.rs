//! Run against a disposable, explicitly supplied loopback Redis only. No
//! default endpoint, DB flush, user keys, or wall-clock expiry sleeps.
#![cfg(feature = "redis")]

use std::time::Duration;

use oxllm_cache_redis::RedisCache;
use oxllm_core::{Cache, CounterWindow, WindowedIncrResult};

async fn fixture() -> Option<(RedisCache, redis::aio::MultiplexedConnection, String)> {
    let Ok(url) = std::env::var("OXLLM_REDIS_URL") else {
        eprintln!("SKIP: set OXLLM_REDIS_URL to a disposable loopback Redis fixture");
        return None;
    };
    // Tests never discover or fall back to an existing Redis installation.
    assert!(
        url.contains("@127.0.0.1:")
            || url.starts_with("redis://127.0.0.1:")
            || url.contains("@localhost:")
            || url.starts_with("redis://localhost:"),
        "a disposable loopback Redis URL is required"
    );
    let prefix = format!("budget-windows:{}", CounterWindow::fresh().as_str());
    let cache = RedisCache::new(&url, &prefix)
        .await
        .expect("explicit Redis fixture must connect");
    let client = redis::Client::open(url.as_str()).unwrap();
    let connection = client.get_multiplexed_async_connection().await.unwrap();
    Some((cache, connection, prefix))
}

fn allowed(result: WindowedIncrResult) -> (f64, CounterWindow) {
    match result {
        WindowedIncrResult::Allowed { value, window } => (value, window),
        other => panic!("expected atomic receipt, got {other:?}"),
    }
}

#[tokio::test]
async fn budget_window_real_expiry_recreation_never_credits_or_debits_a_new_window() {
    let Some((cache, mut connection, prefix)) = fixture().await else {
        return;
    };
    let key = "spend:counter";
    let raw = format!("{prefix}:{key}");
    let (_, original) = allowed(
        cache
            .incr_if_below_in_window(key, 0.1, 10.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    // A joins an already nearly expired window. No request-relative clock can
    // establish its ownership or remaining lifetime correctly.
    let _: i64 = redis::cmd("PEXPIRE")
        .arg(&raw)
        .arg(1000)
        .query_async(&mut connection)
        .await
        .unwrap();
    let (_, a) = allowed(
        cache
            .incr_if_below_in_window(key, 0.8, 10.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    assert_eq!(a, original);
    let ttl: i64 = redis::cmd("PTTL")
        .arg(&raw)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!((0..=1000).contains(&ttl));
    let _: i64 = redis::cmd("PEXPIREAT")
        .arg(&raw)
        .arg(1)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!(cache.counter_value(key).await.is_none());
    // The companion identity is deliberately NOT manually deleted here.
    let (_, b) = allowed(
        cache
            .incr_if_below_in_window(key, 0.01, 10.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    assert_ne!(a, b);
    for delta in [-0.8, 0.5, 0.0] {
        assert!(!cache.incr_in_window(key, delta, &a).await.unwrap());
        assert_eq!(cache.counter_value(key).await, Some(0.01));
    }
    assert_eq!(
        cache
            .incr_if_below_in_window(key, 0.2, 10.0, Some(Duration::from_secs(600)), Some(&a))
            .await
            .unwrap(),
        WindowedIncrResult::WindowChanged
    );
    assert!(cache.incr_in_window(key, 0.02, &b).await.unwrap());
    assert!((cache.counter_value(key).await.unwrap() - 0.03).abs() < 1e-12);
    let ttl: i64 = redis::cmd("PTTL")
        .arg(&raw)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!((0..=60_000).contains(&ttl));
    cache.delete(key).await;
}

#[tokio::test]
async fn budget_window_all_legacy_replacements_invalidate_old_receipts_atomically() {
    let Some((cache, mut connection, prefix)) = fixture().await else {
        return;
    };
    for replacement in ["set", "delete", "expire-incr", "expire-reserve"] {
        let key = format!("spend:{replacement}");
        let raw = format!("{prefix}:{key}");
        let (_, a) = allowed(
            cache
                .incr_if_below_in_window(&key, 0.8, 10.0, Some(Duration::from_secs(60)), None)
                .await
                .unwrap(),
        );
        match replacement {
            "set" => {
                cache
                    .set(&key, "0.01".into(), Some(Duration::from_secs(60)))
                    .await
            }
            "delete" => {
                cache.delete(&key).await;
                cache
                    .incr(&key, 0.01, Some(Duration::from_secs(60)))
                    .await
                    .unwrap();
            }
            _ => {
                let _: i64 = redis::cmd("PEXPIREAT")
                    .arg(&raw)
                    .arg(1)
                    .query_async(&mut connection)
                    .await
                    .unwrap();
                if replacement == "expire-incr" {
                    cache
                        .incr(&key, 0.01, Some(Duration::from_secs(60)))
                        .await
                        .unwrap();
                } else {
                    cache
                        .incr_if_below(&key, 0.01, 10.0, Some(Duration::from_secs(60)))
                        .await
                        .unwrap();
                }
            }
        }
        assert!(
            !cache.incr_in_window(&key, -0.8, &a).await.unwrap(),
            "{replacement}"
        );
        assert!(
            !cache.incr_in_window(&key, 0.5, &a).await.unwrap(),
            "{replacement}"
        );
        assert_eq!(cache.counter_value(&key).await, Some(0.01));
        let (_, b) = allowed(
            cache
                .incr_if_below_in_window(&key, 0.01, 10.0, Some(Duration::from_secs(60)), None)
                .await
                .unwrap(),
        );
        assert_ne!(a, b);
        cache.delete(&key).await;
    }
}

#[tokio::test]
async fn budget_window_conditional_topups_keep_the_original_ttl_and_fail_without_writing() {
    let Some((cache, mut connection, prefix)) = fixture().await else {
        return;
    };
    let key = "spend:topup";
    let (_, a) = allowed(
        cache
            .incr_if_below_in_window(key, 0.25, 1.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    let (value, topped) = allowed(
        cache
            .incr_if_below_in_window(key, 0.25, 1.0, Some(Duration::from_secs(600)), Some(&a))
            .await
            .unwrap(),
    );
    assert_eq!(a, topped);
    assert_eq!(value, 0.5);
    let ttl: i64 = redis::cmd("PTTL")
        .arg(format!("{prefix}:{key}"))
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!((0..=60_000).contains(&ttl));
    assert_eq!(
        cache
            .incr_if_below_in_window(key, 1.0, 1.0, None, Some(&a))
            .await
            .unwrap(),
        WindowedIncrResult::Rejected(0.5)
    );
    assert_eq!(cache.counter_value(key).await, Some(0.5));
    cache.delete(key).await;
    assert_eq!(
        cache
            .incr_if_below_in_window(key, 0.25, 1.0, None, Some(&a))
            .await
            .unwrap(),
        WindowedIncrResult::WindowChanged
    );
    assert!(cache.counter_value(key).await.is_none());
    assert!(!cache.incr_in_window(key, 0.5, &a).await.unwrap());
    assert!(cache.counter_value(key).await.is_none());
}

#[tokio::test]
async fn budget_window_concurrent_stale_refunds_do_not_erase_new_request_spend() {
    let Some((cache, mut connection, prefix)) = fixture().await else {
        return;
    };
    let key = "spend:race";
    let (_, a) = allowed(
        cache
            .incr_if_below_in_window(key, 0.8, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    let _: i64 = redis::cmd("PEXPIREAT")
        .arg(format!("{prefix}:{key}"))
        .arg(1)
        .query_async(&mut connection)
        .await
        .unwrap();
    let (_, b) = allowed(
        cache
            .incr_if_below_in_window(key, 0.01, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(33));
    let mut tasks = Vec::new();
    for index in 0..32 {
        let cache = cache.clone();
        let window = if index % 2 == 0 { a.clone() } else { b.clone() };
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            let delta = if index % 2 == 0 { -0.8 } else { 0.01 };
            assert_eq!(
                cache.incr_in_window(key, delta, &window).await.unwrap(),
                index % 2 != 0
            );
        }));
    }
    barrier.wait().await;
    for task in tasks {
        task.await.unwrap();
    }
    assert!((cache.counter_value(key).await.unwrap() - 0.17).abs() < 1e-12);
    cache.delete(key).await;
}

#[tokio::test]
async fn budget_window_corrupt_counters_fail_closed_without_writing() {
    let Some((cache, mut connection, prefix)) = fixture().await else {
        return;
    };
    let key = "spend:corrupt";
    let raw = format!("{prefix}:{key}");
    let (_, window) = allowed(
        cache
            .incr_if_below_in_window(key, 0.5, 10.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    // Corrupt only the numeric counter, keeping the live companion identity,
    // so the Lua corruption guards are the only thing standing between the
    // receipt and a NaN-poisoned budget scope.
    let _: () = redis::cmd("SET")
        .arg(&raw)
        .arg("nan")
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!(
        cache
            .incr_if_below_in_window(key, 0.1, 10.0, Some(Duration::from_secs(60)), None)
            .await
            .is_err()
    );
    assert!(
        cache
            .incr_if_below_in_window(key, 0.1, 10.0, Some(Duration::from_secs(60)), Some(&window))
            .await
            .is_err()
    );
    for delta in [-0.5, 0.5] {
        assert!(cache.incr_in_window(key, delta, &window).await.is_err());
    }
    let stored: String = redis::cmd("GET")
        .arg(&raw)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert_eq!(stored, "nan", "corruption guards must not write");
    cache.delete(key).await;
}

#[tokio::test]
async fn budget_window_adjustments_keep_the_companion_deadline_locked() {
    let Some((cache, mut connection, prefix)) = fixture().await else {
        return;
    };
    let key = "spend:drift";
    let raw = format!("{prefix}:{key}");
    let companion = format!("{prefix}:__oxllm_budget_window__:{key}");
    let (_, window) = allowed(
        cache
            .incr_if_below_in_window(key, 0.5, 10.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    for delta in [0.25, -0.1, 0.25, 0.0] {
        assert!(cache.incr_in_window(key, delta, &window).await.unwrap());
    }
    let ttl: i64 = redis::cmd("PTTL")
        .arg(&raw)
        .query_async(&mut connection)
        .await
        .unwrap();
    let companion_ttl: i64 = redis::cmd("PTTL")
        .arg(&companion)
        .query_async(&mut connection)
        .await
        .unwrap();
    assert!((0..=60_000).contains(&ttl));
    // Each adjustment re-arms both deadlines from the same captured PTTL, so
    // the companion can no longer outlive its counter by a settle's worth of
    // drift — a stale-receipt window edge cannot reopen.
    assert!(
        (ttl - companion_ttl).abs() <= 50,
        "counter ttl {ttl} vs companion ttl {companion_ttl}"
    );
    // The receipt still settles in that window afterwards.
    assert!(cache.incr_in_window(key, 0.1, &window).await.unwrap());
    assert!((cache.counter_value(key).await.unwrap() - 1.0).abs() < 1e-12);
    cache.delete(key).await;
}
