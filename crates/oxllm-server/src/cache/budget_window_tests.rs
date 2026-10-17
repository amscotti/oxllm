//! Store-level proofs: identity is captured atomically at reservation, not
//! inferred from a request's age or a later TTL read.

use super::*;

fn allowed(result: WindowedIncrResult) -> (f64, CounterWindow) {
    match result {
        WindowedIncrResult::Allowed { value, window } => (value, window),
        other => panic!("expected a window receipt, got {other:?}"),
    }
}

#[tokio::test]
async fn budget_window_joining_a_nearly_expired_counter_does_not_grant_a_new_window() {
    let cache = InMemoryCache::default();
    let key = "spend:near-expiry";
    let (_, original) = allowed(
        cache
            .incr_if_below_in_window(key, 0.1, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    // A joins a counter created 59 seconds earlier. Its own request can be
    // brand new while the shared window is already about to expire.
    let expires_at = Instant::now() + Duration::from_secs(1);
    {
        let mut counters = cache.counters.write().await;
        let counter = counters.get_mut(key).unwrap();
        counter.window_start = Instant::now() - Duration::from_secs(59);
        counter.expires_at = Some(expires_at);
    }
    let (_, a) = allowed(
        cache
            .incr_if_below_in_window(key, 0.8, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    assert_eq!(a, original);
    assert_eq!(cache.counter_expires_at(key).await, Some(expires_at));
    cache.expire_counter_for_test(key).await;
    assert!(cache.counter_value(key).await.is_none());
    let (_, b) = allowed(
        cache
            .incr_if_below_in_window(key, 0.01, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    assert_ne!(a, b);
    let b_expiry = cache.counter_expires_at(key).await;
    for delta in [-0.8, 0.5, 0.0] {
        assert!(!cache.incr_in_window(key, delta, &a).await.unwrap());
        assert_eq!(cache.counter_value(key).await, Some(0.01));
    }
    assert_eq!(
        cache
            .incr_if_below_in_window(key, 0.2, 100.0, Some(Duration::from_secs(600)), Some(&a))
            .await
            .unwrap(),
        WindowedIncrResult::WindowChanged
    );
    assert_eq!(cache.counter_value(key).await, Some(0.01));
    assert_eq!(cache.counter_expires_at(key).await, b_expiry);
    assert!(cache.incr_in_window(key, 0.02, &b).await.unwrap());
    assert!((cache.counter_value(key).await.unwrap() - 0.03).abs() < 1e-12);
    assert_eq!(cache.counter_expires_at(key).await, b_expiry);
}

#[tokio::test]
async fn budget_window_missing_reset_deleted_and_legacy_recreated_counters_reject_old_receipts() {
    for replacement in [
        "expired",
        "reset",
        "delete",
        "legacy-incr",
        "legacy-reserve",
    ] {
        let cache = InMemoryCache::default();
        let key = "spend:replaced";
        let (_, a) = allowed(
            cache
                .incr_if_below_in_window(key, 0.8, 100.0, Some(Duration::from_secs(60)), None)
                .await
                .unwrap(),
        );
        match replacement {
            "expired" => cache.expire_counter_for_test(key).await,
            "reset" => cache.reset_counter(key).await,
            _ => cache.delete(key).await,
        }
        if replacement == "legacy-incr" {
            cache
                .incr(key, 0.01, Some(Duration::from_secs(60)))
                .await
                .unwrap();
        } else if replacement == "legacy-reserve" {
            cache
                .incr_if_below(key, 0.01, 100.0, Some(Duration::from_secs(60)))
                .await
                .unwrap();
        }
        let before = cache.counter_value(key).await;
        assert!(
            !cache.incr_in_window(key, -0.8, &a).await.unwrap(),
            "{replacement}"
        );
        assert!(
            !cache.incr_in_window(key, 0.5, &a).await.unwrap(),
            "{replacement}"
        );
        assert_eq!(cache.counter_value(key).await, before);
        let (_, b) = allowed(
            cache
                .incr_if_below_in_window(key, 0.01, 100.0, Some(Duration::from_secs(60)), None)
                .await
                .unwrap(),
        );
        assert_ne!(a, b, "{replacement}");
        assert!(!cache.incr_in_window(key, -0.8, &a).await.unwrap());
    }
}

#[tokio::test]
async fn budget_window_atomic_stale_refunds_cannot_race_new_window_debits() {
    use tokio::sync::Barrier;
    let cache = Arc::new(InMemoryCache::default());
    let key = "spend:concurrent";
    let (_, a) = allowed(
        cache
            .incr_if_below_in_window(key, 0.8, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    cache.expire_counter_for_test(key).await;
    let (_, b) = allowed(
        cache
            .incr_if_below_in_window(key, 0.01, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    let barrier = Arc::new(Barrier::new(33));
    let mut tasks = Vec::new();
    for index in 0..32 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        let window = if index % 2 == 0 { a.clone() } else { b.clone() };
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
}

#[cfg(feature = "redis")]
#[tokio::test]
async fn budget_window_dual_uses_the_authoritative_receipt_not_a_local_mirrors_window() {
    let Ok(url) = std::env::var("OXLLM_REDIS_URL") else {
        eprintln!("SKIP disposable local Redis test: OXLLM_REDIS_URL is unset");
        return;
    };
    assert!(matches!(
        url::Url::parse(&url).unwrap().host_str(),
        Some("127.0.0.1" | "localhost" | "[::1]")
    ));
    let prefix = format!("budget-dual:{}", CounterWindow::fresh().as_str());
    let redis = RedisCache::new(&url, &prefix).await.unwrap();
    let memory = Arc::new(InMemoryCache::default());
    let key = "spend:authority";
    let (_, local) = allowed(
        memory
            .incr_if_below_in_window(key, 0.8, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    let dual = DualCache::new(memory, Some(redis.clone()));
    let (_, remote) = allowed(
        dual.incr_if_below_in_window(key, 0.1, 100.0, Some(Duration::from_secs(60)), None)
            .await
            .unwrap(),
    );
    assert_ne!(local, remote);
    assert!(!dual.incr_in_window(key, -0.8, &local).await.unwrap());
    assert_eq!(redis.counter_value(key).await, Some(0.1));
    assert!(dual.incr_in_window(key, -0.09, &remote).await.unwrap());
    assert!((dual.counter_value(key).await.unwrap() - 0.01).abs() < 1e-12);
    redis.delete(key).await;
}
