//! Blackhole tests: assert a per-operation timeout wraps every command path
//! so a partitioned / hung Redis cannot hang callers.
//!
//! These run without a real Redis server. They point [`RedisCache`] at a fake
//! TCP server that completes the redis-rs connection handshake (`CLIENT
//! SETINFO` x2 for RESP2 / no-auth) and then drops every subsequent command on
//! the floor — never replying. That is the exact partition shape under which
//! redis's `ConnectionManager` would otherwise retry the in-flight command
//! forever, hanging the caller. Each command that lacks a `REDIS_OP_TIMEOUT`
//! wrap would hang this test's outer bound; each command that has one returns
//! `Err` within ~500ms.

#![cfg(feature = "redis")]

use std::time::Duration;

use oxllm_cache_redis::RedisCache;
use oxllm_core::Cache as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Outer bound generous enough to absorb `REDIS_OP_TIMEOUT` (500ms) plus
/// connect / reconnect slack, while still catching a missing-timeout
/// regression (which would hang and trip this).
const OUTER: Duration = Duration::from_secs(6);

/// Connect a [`RedisCache`] to a blackhole fake Redis, completing the
/// handshake so `new()` returns successfully while leaving every subsequent
/// command unanswered.
async fn blackhole_cache(prefix: &str) -> RedisCache {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => return,
            };
            tokio::spawn(async move {
                let mut buf: Vec<u8> = Vec::new();
                let mut sent = 0usize;
                let mut chunk = [0u8; 1024];
                loop {
                    let n = match sock.read(&mut chunk).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    buf.extend_from_slice(&chunk[..n]);
                    // redis-rs sends two `CLIENT SETINFO` commands during the
                    // connection handshake (RESP2 / no-auth); reply with one
                    // `+OK` per marker so `get_connection_manager()` /
                    // `get_async_pubsub()` succeed. Markers are counted on the
                    // accumulated stream so a marker split across TCP segments
                    // is still detected. Everything else (GET, PUBLISH,
                    // SUBSCRIBE) is left unanswered -> the in-flight command
                    // hangs -> the per-op timeout must fire.
                    let count = count_marker(&buf, b"SETINFO");
                    if count > sent {
                        let mut out = Vec::new();
                        for _ in 0..(count - sent) {
                            out.extend_from_slice(b"+OK\r\n");
                        }
                        sent = count;
                        if sock.write_all(&out).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
    });
    let url = format!("redis://{addr}");
    tokio::time::timeout(Duration::from_secs(5), RedisCache::new(&url, prefix))
        .await
        .expect("outer timeout on RedisCache::new")
        .expect("handshake should complete against the blackhole fake server")
}

/// Count non-overlapping occurrences of `needle` in `hay`.
fn count_marker(hay: &[u8], needle: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + needle.len() <= hay.len() {
        if &hay[i..i + needle.len()] == needle {
            count += 1;
            i += needle.len();
        } else {
            i += 1;
        }
    }
    count
}

#[tokio::test]
async fn try_counter_value_times_out_under_partition() {
    let cache = blackhole_cache("blackhole-tcv").await;
    let res = tokio::time::timeout(OUTER, cache.try_counter_value("any-key"))
        .await
        .expect("try_counter_value hung (REDIS_OP_TIMEOUT missing?)");
    assert!(
        res.is_err(),
        "expected Err under Redis partition, got {res:?}"
    );
}

#[tokio::test]
async fn publish_times_out_under_partition() {
    let cache = blackhole_cache("blackhole-pub").await;
    let res = tokio::time::timeout(OUTER, cache.publish("any-chan", "x"))
        .await
        .expect("publish hung (REDIS_OP_TIMEOUT missing?)");
    assert!(
        res.is_err(),
        "expected Err under Redis partition, got {res:?}"
    );
}

#[tokio::test]
async fn subscribe_times_out_under_partition() {
    let cache = blackhole_cache("blackhole-sub").await;
    let res = tokio::time::timeout(OUTER, cache.subscribe("any-chan"))
        .await
        .expect("subscribe hung (REDIS_OP_TIMEOUT missing?)");
    assert!(res.is_err(), "expected Err under Redis partition, got Ok");
}

#[tokio::test]
async fn budget_window_reservation_times_out_under_partition() {
    let cache = blackhole_cache("blackhole-budget-reserve").await;
    let result = tokio::time::timeout(
        OUTER,
        cache.incr_if_below_in_window("spend:x", 0.5, 1.0, Some(Duration::from_secs(60)), None),
    )
    .await
    .expect("windowed reservation must have an operation timeout");
    assert!(result.is_err());
}

#[tokio::test]
async fn budget_window_adjustment_times_out_under_partition() {
    let cache = blackhole_cache("blackhole-budget-adjust").await;
    let window = oxllm_core::CounterWindow::fresh();
    let result = tokio::time::timeout(OUTER, cache.incr_in_window("spend:x", -0.5, &window))
        .await
        .expect("windowed adjustment must have an operation timeout");
    assert!(result.is_err());
}
