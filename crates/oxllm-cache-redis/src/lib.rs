//! Redis cache implementation for oxllm.
//!
//! Implements the [`oxllm_core::Cache`] trait backed by Redis. All data keys are
//! prefixed with `{prefix}:` to avoid collisions. Uses `ConnectionManager` for
//! automatic reconnection and connection pooling.

#![forbid(unsafe_code)]

#[cfg(feature = "redis")]
mod inner {
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use oxllm_core::Cache as CacheTrait;
    use oxllm_core::{Error, IncrResult};
    use redis::AsyncCommands;
    use redis::Script;
    use redis::aio::{ConnectionManager, PubSubStream};
    use redis::cmd;

    /// Upper bound on how long [`RedisCache::new`] will wait for the initial
    /// Redis handshake before failing. Hosts that want a tighter bound may still
    /// wrap the future in their own `tokio::time::timeout`.
    const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

    /// Per-operation timeout for Redis commands. When Redis is unreachable
    /// (network partition, Redis down), the `ConnectionManager` keeps
    /// retrying indefinitely, which hangs every request that touches the
    /// cache. This timeout bounds each command so the caller gets an error
    /// (and can fail-open) instead of hanging forever. 500ms is generous for
    /// any healthy Redis (typical op < 1ms) while keeping outage recovery
    /// snappy.
    const REDIS_OP_TIMEOUT: Duration = Duration::from_millis(500);

    /// Upper bound (seconds) for any single Redis TTL — 30 days, matching
    /// the in-memory and disk backends' `MAX_TTL_SECS` clamp. TTLs are
    /// sometimes derived from provider-issued input (e.g. a JWT `exp` claim
    /// clamped nowhere before reaching the cache), and an absurd TTL would
    /// otherwise convert to a millisecond count that overflows Redis's i64
    /// `PEXPIRE`/`PSETEX` argument ("value is out of range"), hard-failing
    /// `incr`/`incr_if_below` and silently dropping `set` entries instead of
    /// degrading to a long-but-valid window.
    const MAX_TTL_SECS: u64 = 30 * 24 * 60 * 60;

    /// Read the value and its remaining TTL atomically. Separate GET/PTTL
    /// commands can pair an old value with a replacement's longer lifetime.
    const GET_WITH_TTL_SCRIPT: &str = r#"
local value = redis.call('GET', KEYS[1])
if not value then return nil end
return {value, redis.call('PTTL', KEYS[1])}
"#;

    /// Lua: atomic `INCRBYFLOAT` + first-creation window TTL.
    ///
    /// `INCRBYFLOAT` preserves any existing expiry, so the window TTL is only
    /// stamped when the key has no TTL yet (`PTTL < 0`), keeping fixed-window
    /// (non-sliding) semantics. Performing the increment and the TTL check
    /// inside a single script eliminates the lost-`EXPIRE` window and the
    /// TOCTOU at the window edge that the previous multi-round-trip impl had.
    /// `INCRBYFLOAT` already returns a bulk string, so the f64 survives intact.
    const INCR_SCRIPT: &str = r#"
local new_val = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
if ARGV[2] ~= '' then
  local existing_ttl = redis.call('PTTL', KEYS[1])
  if existing_ttl < 0 then
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
  end
end
return new_val
"#;

    /// Lua: atomic compare-and-increment for budget / RPM gating.
    ///
    /// The whole read-check-write runs server-side, so concurrent clients across
    /// instances can never exceed `ceiling`. The `<= ceiling` gate is computed
    /// with full f64 precision. The new/current values are returned as bulk
    /// strings via `string.format("%.17g", ...)`: Redis converts a returned Lua
    /// *number* into an *integer* reply (dropping the decimals), so returning
    /// numbers would silently floor fractional counters (e.g. `0.25 -> 0`).
    /// Returning strings preserves the exact value and matches what `SET`
    /// stored, keeping the returned counter consistent with a later read.
    /// `%.17g` is the shortest IEEE-754 round-trippable representation, so the
    /// stored value keeps full f64 precision just like the `INCRBYFLOAT`-based
    /// [`INCR_SCRIPT`] path (Lua's default `tostring` formats with only `%.14g`,
    /// which would silently round to 14 significant figures and drift relative
    /// to `incr` on the same key).
    ///
    /// TTL handling mirrors `INCR_SCRIPT`: the existing remaining TTL is
    /// captured before `SET` (which clears expiry) and re-applied; the original
    /// window is stamped only on first creation (`PTTL < 0`).
    const INCR_IF_BELOW_SCRIPT: &str = r#"
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local new_val = current + tonumber(ARGV[1])
local ceiling = tonumber(ARGV[2])
if new_val <= ceiling then
  local existing_ttl = redis.call('PTTL', KEYS[1])
  redis.call('SET', KEYS[1], string.format('%.17g', new_val))
  if existing_ttl >= 0 then
    redis.call('PEXPIRE', KEYS[1], existing_ttl)
  elseif ARGV[3] ~= '' then
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[3]))
  end
  return {1, string.format('%.17g', new_val)}
else
  return {0, string.format('%.17g', current)}
end
"#;

    /// Lua: decrement only when the key exists, so budget reservation
    /// releases landing after the window TTL expired cannot mint a fresh
    /// negative, never-expiring counter (Redis `INCRBYFLOAT` on a missing
    /// key creates it at `0 + delta`). `EXISTS` is false for expired keys,
    /// and `INCRBYFLOAT` preserves any remaining TTL.
    const DECR_IF_EXISTS_SCRIPT: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 1 then
  return {1, redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])}
else
  return {0, '0'}
end
"#;

    /// Validate a `Cache::incr_if_below` (`delta`, `ceiling`) pair, rejecting
    /// any non-finite input. See [`INCR_IF_BELOW_SCRIPT`] and the override
    /// below for why this guard is required here even though the `incr` path
    /// is protected server-side by `INCRBYFLOAT`.
    fn validate_finite_incr_if_below(delta: f64, ceiling: f64) -> Result<(), Error> {
        if !delta.is_finite() || !ceiling.is_finite() {
            return Err(Error::Config(
                "incr_if_below rejected non-finite delta/ceiling".into(),
            ));
        }
        Ok(())
    }

    /// Reject an empty/whitespace-only prefix. An empty prefix yields
    /// `:`-prefixed keys (`:spend:...`) that collide with any other tenant on
    /// a shared Redis instance, silently corrupting counters and cross-reading
    /// cache entries.
    fn validate_prefix(prefix: &str) -> Result<(), redis::RedisError> {
        if prefix.trim().is_empty() {
            return Err(redis::RedisError::from((
                redis::ErrorKind::InvalidClientConfig,
                "RedisCache prefix must be non-empty",
                format!(
                    "prefix {prefix:?} is empty/whitespace-only; an empty \
                     prefix produces ':'-prefixed keys that collide with \
                     other tenants on a shared Redis instance"
                ),
            )));
        }
        Ok(())
    }

    /// Upper bound (milliseconds) on any single Redis TTL: [`MAX_TTL_SECS`]
    /// expressed in the unit Redis's `PEXPIRE`/`PSETEX` arguments take (an
    /// i64), so an absurd caller TTL degrades to a long-but-valid window
    /// instead of an out-of-range argument error.
    const MAX_TTL_MILLIS: u64 = MAX_TTL_SECS * 1000;

    /// Convert a positive TTL to milliseconds with a 1ms floor and a
    /// [`MAX_TTL_MILLIS`] ceiling. Uses Redis's millisecond-precision expiry
    /// variants so sub-second durations are not silently rounded up to a
    /// full second. Zero remains zero (immediate expiry, never persistence).
    fn ttl_to_millis(ttl: Duration) -> u64 {
        if ttl.is_zero() {
            return 0;
        }
        let ms = ttl.as_millis().min(u64::MAX as u128) as u64;
        ms.clamp(1, MAX_TTL_MILLIS)
    }

    /// Build a namespaced name (`{prefix}:{name}`) for a data key or a pub/sub
    /// channel. Centralized so data keys and channels namespace identically.
    fn prefixed(prefix: &str, name: &str) -> String {
        format!("{prefix}:{name}")
    }

    /// A Redis-backed cache implementing [`oxllm_core::Cache`].
    #[derive(Clone)]
    pub struct RedisCache {
        conn: ConnectionManager,
        client: redis::Client,
        prefix: String,
        get_with_ttl_script: Script,
        incr_script: Script,
        incr_if_below_script: Script,
        decr_if_exists_script: Script,
    }

    impl RedisCache {
        /// Create a new Redis cache, connecting to the given URL.
        ///
        /// All data keys will be prefixed with `{prefix}:`. The initial
        /// connection handshake is bounded by `DEFAULT_CONNECT_TIMEOUT`; a
        /// longer / shorter bound can be imposed by the caller by wrapping the
        /// returned future in its own timeout.
        ///
        /// Returns an error if `prefix` is empty or whitespace-only: an empty
        /// prefix yields `:`-prefixed keys (`:spend:...`) that collide with
        /// any other tenant on a shared Redis instance, silently corrupting
        /// counters and cross-reading cache entries.
        pub async fn new(url: &str, prefix: &str) -> Result<Self, redis::RedisError> {
            validate_prefix(prefix)?;
            let client = redis::Client::open(url)?;
            let conn = match tokio::time::timeout(
                DEFAULT_CONNECT_TIMEOUT,
                client.get_connection_manager(),
            )
            .await
            {
                Ok(res) => res?,
                Err(_) => {
                    return Err(redis::RedisError::from(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!(
                            "timed out connecting to redis after {}s",
                            DEFAULT_CONNECT_TIMEOUT.as_secs()
                        ),
                    )));
                }
            };
            Self::build(conn, client, prefix)
        }

        /// Create a Redis cache from an existing client + connection manager.
        /// Both must point at the same Redis instance. Useful as an injection
        /// point (e.g. a harness that supplies its own `ConnectionManager`).
        ///
        /// Returns an error if `prefix` is empty or whitespace-only (see
        /// [`RedisCache::new`]).
        pub fn from_connection(
            conn: ConnectionManager,
            client: redis::Client,
            prefix: &str,
        ) -> Result<Self, redis::RedisError> {
            Self::build(conn, client, prefix)
        }

        fn build(
            conn: ConnectionManager,
            client: redis::Client,
            prefix: &str,
        ) -> Result<Self, redis::RedisError> {
            validate_prefix(prefix)?;
            Ok(Self {
                conn,
                client,
                prefix: prefix.to_string(),
                get_with_ttl_script: Script::new(GET_WITH_TTL_SCRIPT),
                incr_script: Script::new(INCR_SCRIPT),
                incr_if_below_script: Script::new(INCR_IF_BELOW_SCRIPT),
                decr_if_exists_script: Script::new(DECR_IF_EXISTS_SCRIPT),
            })
        }

        fn key(&self, key: &str) -> String {
            prefixed(&self.prefix, key)
        }

        /// Get a value and the time it may still be cached. `None` in the TTL
        /// slot means the Redis key has no expiry. The round-trip duration is
        /// subtracted conservatively, so a local backfill cannot extend the
        /// authoritative lifetime. Already-expired/in-flight-expired reads are
        /// misses. This never writes a value back to Redis.
        pub async fn get_with_ttl(&self, key: &str) -> Option<(String, Option<Duration>)> {
            let mut conn = self.conn.clone();
            let started = Instant::now();
            let result: redis::RedisResult<Option<(String, i64)>> = match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                self.get_with_ttl_script
                    .key(self.key(key))
                    .invoke_async(&mut conn),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    tracing::warn!(key = %key, "redis get_with_ttl timed out");
                    return None;
                }
            };
            let (value, millis) = match result {
                Ok(Some(entry)) => entry,
                Ok(None) => return None,
                Err(error) => {
                    tracing::warn!(error = %error, key = %key, "redis get_with_ttl failed");
                    return None;
                }
            };
            let remaining = match millis {
                -1 => None,
                millis if millis > 0 => {
                    let remaining =
                        Duration::from_millis(millis as u64).checked_sub(started.elapsed())?;
                    if remaining.is_zero() {
                        return None;
                    }
                    Some(remaining)
                }
                _ => return None,
            };
            Some((value, remaining))
        }

        /// Build a pub/sub channel name namespaced by this cache's prefix, so
        /// fleets sharing a Redis instance with different prefixes do not cross
        /// invalidate. Callers should prefer this over passing a raw channel to
        /// [`RedisCache::publish`] / [`RedisCache::subscribe`] when the channel
        /// is meant to be tenant-scoped.
        pub fn pubsub_channel(&self, name: &str) -> String {
            prefixed(&self.prefix, name)
        }

        /// Publish a message on a Redis pub/sub channel. The channel is used
        /// verbatim — namespace it with [`RedisCache::pubsub_channel`] if it
        /// should be isolated by the configured prefix. Uses the shared
        /// `ConnectionManager` (commands are multiplexed over a single
        /// connection, so this is safe to call concurrently).
        pub async fn publish(&self, channel: &str, message: &str) -> Result<(), redis::RedisError> {
            let mut conn = self.conn.clone();
            match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                cmd("PUBLISH")
                    .arg(channel)
                    .arg(message)
                    .query_async::<i64>(&mut conn),
            )
            .await
            {
                Ok(res) => res.map(|_| ()),
                Err(_) => Err(redis::RedisError::from(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "redis publish timed out",
                ))),
            }
        }

        /// Subscribe to a Redis pub/sub channel. The channel is used verbatim —
        /// namespace it with [`RedisCache::pubsub_channel`] if it should be
        /// isolated by the configured prefix. Opens a dedicated pub/sub
        /// connection (separate from the command `ConnectionManager`) and
        /// returns a stream of payloads. The caller is responsible for driving
        /// the stream (typically in a `tokio::spawn`'d task).
        pub async fn subscribe(&self, channel: &str) -> Result<PubSubStream, redis::RedisError> {
            let mut pubsub = match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                self.client.get_async_pubsub(),
            )
            .await
            {
                Ok(res) => res?,
                Err(_) => {
                    return Err(redis::RedisError::from(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "redis connect (pubsub) timed out",
                    )));
                }
            };
            match tokio::time::timeout(REDIS_OP_TIMEOUT, pubsub.subscribe(channel)).await {
                Ok(res) => res?,
                Err(_) => {
                    return Err(redis::RedisError::from(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "redis subscribe timed out",
                    )));
                }
            }
            Ok(pubsub.into_on_message())
        }
    }

    #[async_trait]
    impl CacheTrait for RedisCache {
        async fn get(&self, key: &str) -> Option<String> {
            let redis_key = self.key(key);
            let mut conn = self.conn.clone();
            let result =
                tokio::time::timeout(REDIS_OP_TIMEOUT, conn.get::<_, Option<String>>(&redis_key))
                    .await;
            match result {
                Ok(Ok(val)) => val,
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, key = %key, "redis get failed");
                    None
                }
                Err(_) => {
                    tracing::warn!(key = %key, "redis get timed out");
                    None
                }
            }
        }

        async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
            // PSETEX rejects zero; replacing it with a 1ms lifetime would
            // briefly expose an entry the caller requested to expire at once.
            if ttl.is_some_and(|ttl| ttl.is_zero()) {
                self.delete(key).await;
                return;
            }
            let redis_key = self.key(key);
            let mut conn = self.conn.clone();
            let fut = if let Some(ttl) = ttl {
                let millis = ttl_to_millis(ttl);
                conn.pset_ex::<_, _, ()>(&redis_key, &value, millis)
            } else {
                conn.set::<_, _, ()>(&redis_key, &value)
            };
            match tokio::time::timeout(REDIS_OP_TIMEOUT, fut).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, key = %key, "redis set failed"),
                Err(_) => tracing::warn!(key = %key, "redis set timed out"),
            }
        }

        async fn delete(&self, key: &str) {
            let redis_key = self.key(key);
            let mut conn = self.conn.clone();
            match tokio::time::timeout(REDIS_OP_TIMEOUT, conn.del::<_, ()>(&redis_key)).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, key = %key, "redis delete failed"),
                Err(_) => tracing::warn!(key = %key, "redis delete timed out"),
            }
        }

        async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error> {
            // Reject non-finite deltas before they reach Redis: `INCRBYFLOAT`
            // with NaN/±inf writes a poisoned value into the shared fleet
            // counter (mirrors the `decr_if_exists` / `incr_if_below` guards).
            if !by.is_finite() {
                return Err(Error::Config("incr rejected non-finite delta".into()));
            }
            let redis_key = self.key(key);
            let ttl_arg = match ttl {
                Some(d) => ttl_to_millis(d).to_string(),
                None => String::new(),
            };
            let mut conn = self.conn.clone();
            let result: redis::RedisResult<f64> = match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                self.incr_script
                    .key(&redis_key)
                    .arg(by)
                    .arg(&ttl_arg)
                    .invoke_async(&mut conn),
            )
            .await
            {
                Ok(r) => r,
                Err(_) => {
                    tracing::warn!(key = %key, "redis incr timed out");
                    return Err(Error::Network("redis incr timed out".into()));
                }
            };
            match result {
                Ok(v) => Ok(v),
                Err(e) => {
                    tracing::warn!(error = %e, key = %key, "redis incr failed");
                    Err(Error::Network(format!("redis incr failed: {e}")))
                }
            }
        }

        /// Atomic decrement-only-when-exists via a Lua script evaluated
        /// server-side (see `DECR_IF_EXISTS_SCRIPT`). Returning the bulk
        /// string from `INCRBYFLOAT` keeps full f64 precision; the `0` flag
        /// carries whether the decrement was applied.
        async fn decr_if_exists(&self, key: &str, by: f64) -> Result<bool, Error> {
            if !by.is_finite() {
                return Err(Error::Config(
                    "decr_if_exists rejected non-finite delta".into(),
                ));
            }
            let redis_key = self.key(key);
            let mut conn = self.conn.clone();
            let result: redis::RedisResult<(i64, String)> = match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                self.decr_if_exists_script
                    .key(&redis_key)
                    .arg(-by)
                    .invoke_async(&mut conn),
            )
            .await
            {
                Ok(r) => r,
                Err(_) => {
                    tracing::warn!(key = %key, "redis decr_if_exists timed out");
                    return Err(Error::Network("redis decr_if_exists timed out".into()));
                }
            };
            match result {
                Ok((applied, _value)) => Ok(applied == 1),
                Err(e) => {
                    tracing::warn!(error = %e, key = %key, "redis decr_if_exists failed");
                    Err(Error::Network(format!("redis decr_if_exists failed: {e}")))
                }
            }
        }

        /// Atomic compare-and-increment via a Lua script evaluated server-side
        /// (see `INCR_IF_BELOW_SCRIPT`). The gate decision is computed with
        /// full f64 precision in Redis; the carried new/current values are
        /// returned as strings to preserve fractional precision (a bare Lua
        /// number would be integer-truncated by the Redis reply encoder).
        async fn incr_if_below(
            &self,
            key: &str,
            delta: f64,
            ceiling: f64,
            ttl: Option<Duration>,
        ) -> Result<IncrResult, Error> {
            // Defense-in-depth guard matching the trait default
            // (`oxllm_core::Cache::incr_if_below`): a non-finite `delta` or
            // `ceiling` is rejected up front. Unlike the `incr` path — where
            // Redis's own `INCRBYFLOAT` rejects NaN/±∞ results — this override
            // evaluates the gate in Lua (`tonumber` + `+`), which happily
            // produces ±∞/NaN. A `-∞` delta would `SET` the counter to `-∞`,
            // making every subsequent gate pass (`-∞ <= ceiling`) and silently
            // bypassing the budget for the whole window TTL; a `+∞`/`NaN`
            // ceiling disables/mis-enforces the gate. Because Redis is the
            // shared fleet store, a poisoned key is observed by every replica
            // and survives restarts.
            validate_finite_incr_if_below(delta, ceiling)?;
            let redis_key = self.key(key);
            let ttl_arg = match ttl {
                Some(d) => ttl_to_millis(d).to_string(),
                None => String::new(),
            };
            let mut conn = self.conn.clone();
            let result: redis::RedisResult<(i64, String)> = match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                self.incr_if_below_script
                    .key(&redis_key)
                    .arg(delta)
                    .arg(ceiling)
                    .arg(&ttl_arg)
                    .invoke_async(&mut conn),
            )
            .await
            {
                Ok(r) => r,
                Err(_) => {
                    tracing::warn!(key = %key, "redis incr_if_below timed out");
                    return Err(Error::Network("redis incr_if_below timed out".into()));
                }
            };
            match result {
                Ok((allowed, value_str)) => {
                    let value = value_str.parse::<f64>().map_err(|e| {
                        Error::Network(format!(
                            "redis incr_if_below returned non-numeric value: {e}"
                        ))
                    })?;
                    if allowed == 1 {
                        Ok(IncrResult::Allowed(value))
                    } else {
                        Ok(IncrResult::Rejected(value))
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, key = %key, "redis incr_if_below failed");
                    Err(Error::Network(format!("redis incr_if_below failed: {e}")))
                }
            }
        }

        async fn ping(&self) -> bool {
            let mut conn = self.conn.clone();
            match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                cmd("PING").query_async::<String>(&mut conn),
            )
            .await
            {
                Ok(Ok(_)) => true,
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "redis ping failed");
                    false
                }
                Err(_) => {
                    tracing::warn!("redis ping timed out");
                    false
                }
            }
        }

        /// Surface Redis backend errors as `Err` (unlike `Cache::get`, which
        /// masks them as `None`) so callers can distinguish a genuine key
        /// absence from a transient outage. A present-but-non-numeric value is
        /// still treated as absent (`Ok(None)`) with a warning, matching
        /// `Cache::counter_value`.
        async fn try_counter_value(&self, key: &str) -> Result<Option<f64>, Error> {
            let redis_key = self.key(key);
            let mut conn = self.conn.clone();
            let val: Option<String> = match tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                conn.get::<_, Option<String>>(&redis_key),
            )
            .await
            {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, key = %key, "redis get failed");
                    return Err(Error::Network(format!("redis get failed: {e}")));
                }
                Err(_) => {
                    tracing::warn!(key = %key, "redis try_counter_value timed out");
                    return Err(Error::Network("redis try_counter_value timed out".into()));
                }
            };
            match val {
                None => Ok(None),
                Some(v) => match v.parse::<f64>() {
                    Ok(n) => Ok(Some(n)),
                    Err(_) => {
                        tracing::warn!(
                            key = %key,
                            value = %v.chars().take(200).collect::<String>(),
                            "counter value failed to parse as f64; treating as absent"
                        );
                        Ok(None)
                    }
                },
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn ttl_to_millis_floors_sub_millisecond_to_one() {
            assert_eq!(ttl_to_millis(Duration::from_nanos(1)), 1);
            assert_eq!(ttl_to_millis(Duration::ZERO), 0);
        }

        #[test]
        fn ttl_to_millis_preserves_millisecond_precision() {
            assert_eq!(ttl_to_millis(Duration::from_millis(500)), 500);
            assert_eq!(ttl_to_millis(Duration::from_millis(1500)), 1500);
            assert_eq!(ttl_to_millis(Duration::from_secs(60)), 60_000);
        }

        /// An absurd TTL (e.g. `Duration::from_secs(u64::MAX)` derived from a
        /// huge JWT `exp` claim) must clamp to the 30-day ceiling, mirroring
        /// the in-memory/disk backends' `MAX_TTL_SECS` behavior. Without the
        /// clamp the millisecond count overflows Redis's i64
        /// `PEXPIRE`/`PSETEX` argument ("value is out of range"), which
        /// hard-fails `incr`/`incr_if_below` and silently drops `set` entries.
        #[test]
        fn ttl_to_millis_clamps_absurd_ttl_to_ceiling() {
            assert_eq!(ttl_to_millis(Duration::from_secs(u64::MAX)), MAX_TTL_MILLIS);
            assert_eq!(ttl_to_millis(Duration::MAX), MAX_TTL_MILLIS);
            // The ceiling itself round-trips (and stays representable as i64).
            assert_eq!(
                ttl_to_millis(Duration::from_secs(MAX_TTL_SECS)),
                MAX_TTL_MILLIS
            );
            assert!(MAX_TTL_MILLIS <= i64::MAX as u64);
        }

        #[test]
        fn prefixed_joins_with_single_colon() {
            assert_eq!(prefixed("oxllm", "k"), "oxllm:k");
            assert_eq!(prefixed("oxllm", "invalidate"), "oxllm:invalidate");
            assert_ne!(
                prefixed("oxllm", "invalidate"),
                prefixed("oxllm-other", "invalidate")
            );
        }

        #[test]
        fn prefixed_empty_prefix_collides_under_colon() {
            // Premise for the empty-prefix guard: an empty prefix produces
            // ':'-prefixed keys indistinguishable from any other empty-prefix
            // tenant on the same instance.
            assert_eq!(prefixed("", "spend"), ":spend");
            assert_eq!(prefixed("", "spend"), prefixed("", "spend"));
        }

        #[test]
        fn validate_finite_incr_if_below_accepts_finite_inputs() {
            assert!(validate_finite_incr_if_below(0.0, 0.0).is_ok());
            assert!(validate_finite_incr_if_below(-1.5, 10.0).is_ok());
            assert!(validate_finite_incr_if_below(f64::MIN_POSITIVE, f64::MAX).is_ok());
        }

        #[test]
        fn validate_finite_incr_if_below_rejects_non_finite_delta() {
            // A -∞ delta would SET the counter to -∞, making every subsequent
            // gate pass and silently bypassing the budget for the window TTL.
            assert!(validate_finite_incr_if_below(f64::NEG_INFINITY, 10.0).is_err());
            assert!(validate_finite_incr_if_below(f64::INFINITY, 10.0).is_err());
            assert!(validate_finite_incr_if_below(f64::NAN, 10.0).is_err());
        }

        #[test]
        fn validate_finite_incr_if_below_rejects_non_finite_ceiling() {
            // +∞ ceiling disables enforcement; NaN ceiling blocks all traffic.
            assert!(validate_finite_incr_if_below(1.0, f64::INFINITY).is_err());
            assert!(validate_finite_incr_if_below(1.0, f64::NEG_INFINITY).is_err());
            assert!(validate_finite_incr_if_below(1.0, f64::NAN).is_err());
        }

        #[tokio::test]
        async fn new_rejects_empty_prefix_without_touching_network() {
            // The prefix is validated before any connection attempt, so an
            // empty/whitespace prefix fails fast regardless of whether a Redis
            // is reachable. Use a bogus URL to prove the network is not tried.
            for bad in ["", "   ", "\t\n"] {
                match RedisCache::new("redis://127.0.0.1:1", bad).await {
                    Err(e) => assert_eq!(
                        e.kind(),
                        redis::ErrorKind::InvalidClientConfig,
                        "expected empty/whitespace prefix {bad:?} rejected as config error"
                    ),
                    Ok(_) => panic!("expected empty/whitespace prefix {bad:?} to be rejected"),
                }
            }
        }

        #[test]
        fn validate_prefix_rejects_empty_and_whitespace() {
            assert!(validate_prefix("").is_err());
            assert!(validate_prefix("   ").is_err());
            assert!(validate_prefix("\t\n").is_err());
            assert!(validate_prefix("oxllm").is_ok());
            assert!(validate_prefix(" oxllm ").is_ok());
        }
    }
}

#[cfg(feature = "redis")]
pub use inner::RedisCache;
