//! The Cache trait — abstract storage for responses, counters, and keys.

use async_trait::async_trait;
use std::time::Duration;

use crate::Error;

/// Outcome of an atomic [`Cache::incr_if_below`] reservation.
///
/// `Allowed` carries the new counter value; `Rejected` carries the value
/// after rollback (mathematically the pre-`delta` value). Note that the
/// default `incr_if_below` may still materialize a previously-absent key as
/// present-but-zero on the reject path — see
/// [`Cache::incr_if_below`] for details.
#[derive(Debug, Clone, PartialEq)]
pub enum IncrResult {
    /// The increment was applied; the counter now holds this value.
    Allowed(f64),
    /// The increment was refused because it would exceed the ceiling; the
    /// counter has been rolled back to this value.
    Rejected(f64),
}

/// Abstract cache trait for response cache, rate-limit counters, and key cache.
///
/// Implementations: `InMemoryCache` (moka, in `oxllm-server`), `RedisCache`
/// (in `oxllm-cache-redis`). Both implement this trait so the server is
/// agnostic to the backing store.
#[async_trait]
pub trait Cache: Send + Sync {
    /// Get a value by key, returning `None` on miss.
    async fn get(&self, key: &str) -> Option<String>;

    /// Set a value with an optional TTL.
    ///
    /// `None` stores without an expiry. `Some(Duration::ZERO)` invalidates the
    /// key immediately (equivalent to [`Cache::delete`]) rather than storing a
    /// minimal-lifetime entry.
    async fn set(&self, key: &str, value: String, ttl: Option<Duration>);

    /// Delete a value by key.
    async fn delete(&self, key: &str);

    /// Atomic increment for rate-limit counters. Returns the new value, or
    /// an error when the backing store fails. Concrete stores (Redis) surface
    /// failures so callers do not silently drop spend debits; infallible
    /// in-memory stores return `Ok`.
    ///
    /// `ttl` defines a fixed window stamped when the counter is first created
    /// (an absent/expired counter starts at zero): later increments do not
    /// extend the window, and the counter expires when the TTL elapses. `None`
    /// stores the counter without an expiry.
    async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error>;

    /// Atomically increment `key` by `delta` only if the result would be
    /// `<= ceiling`. Returns [`IncrResult::Allowed`] with the new value, or
    /// [`IncrResult::Rejected`] with the current value (rolled back to its
    /// pre-`delta` value on the reject path).
    ///
    /// When provided, `ttl` defines a fixed window stamped when the counter is
    /// first created on the allowed path; as with [`Cache::incr`], later
    /// increments do not extend it and the counter expires when the TTL
    /// elapses. Backing stores with compare-and-set primitives (Redis Lua,
    /// in-memory write locks) override this for true cross-call atomicity.
    ///
    /// The default implementation is **non-atomic**: it falls back to
    /// [`Cache::incr`] followed by a check and rollback. This is subject to
    /// transient overcommit under concurrency and exists for backward
    /// compatibility; concrete stores should override it.
    ///
    /// # Non-finite inputs
    ///
    /// The default impl rejects a non-finite `delta` or `ceiling` (NaN,
    /// `+∞`, `-∞`) up front with [`Error::Config`]. A `NaN` delta would
    /// otherwise write `NaN` into the counter and brick the budget/RPM scope
    /// for the rest of its TTL (every subsequent `NaN <= ceiling` is `false`
    /// → reject → roll back to `NaN`, with no error surfaced). Concrete
    /// overrides that drop the speculative-incr pattern should apply the same
    /// guard: a counter poisoned by non-finite math is unrecoverable until
    /// the window TTL expires.
    ///
    /// # Reject-path note for the default impl
    ///
    /// On the reject path the default calls `incr(key, -delta, None)` to
    /// restore the value. Because the speculative `incr(key, delta, ttl)`
    /// already ran first, **a previously-absent key may be materialized as a
    /// present, near-zero, TTL-bearing key** even when the call returns
    /// `Rejected`. For the shipped stores (in-memory moka, Redis) a
    /// near-zero budget/RPM counter behaves identically to an absent one, so
    /// this is functionally benign. A store that distinguishes "key absent"
    /// from "key present at 0" for TTL-renewal or eviction purposes should
    /// override `incr_if_below` atomically (Redis and moka already do).
    async fn incr_if_below(
        &self,
        key: &str,
        delta: f64,
        ceiling: f64,
        ttl: Option<Duration>,
    ) -> Result<IncrResult, Error> {
        if !delta.is_finite() || !ceiling.is_finite() {
            return Err(Error::Config(
                "incr_if_below rejected non-finite delta/ceiling".into(),
            ));
        }
        let new_val = self.incr(key, delta, ttl).await?;
        if new_val <= ceiling {
            Ok(IncrResult::Allowed(new_val))
        } else {
            // Rollback the speculative increment. A store failure here would
            // leave the counter overshot past the ceiling while the caller
            // believes it was rejected unchanged — corrupting budget/RPM
            // enforcement for any `Cache` impl that relies on this default.
            // Surface it as an error (rather than a false `Rejected`) so the
            // misalignment is visible.
            if let Err(rollback_err) = self.incr(key, -delta, None).await {
                tracing::error!(
                    key = %key,
                    error = %rollback_err,
                    "incr_if_below rollback failed; counter may be overshot past the ceiling"
                );
                return Err(rollback_err);
            }
            Ok(IncrResult::Rejected(new_val - delta))
        }
    }

    /// Decrement counter `key` by `by` **only when the counter currently
    /// exists** (and has not expired), returning whether the decrement was
    /// applied.
    ///
    /// Reservation release/settle paths use this instead of
    /// `incr(key, -by, None)`: on a store where `incr` creates missing
    /// keys, a release that lands after the budget window expired would
    /// otherwise mint a fresh **negative, never-expiring** counter that
    /// permanently discounts every subsequent window's spend. Skipping the
    /// decrement on an absent/expired counter accepts a small one-window
    /// over-debit in exchange for never creating negative residue.
    ///
    /// The default implementation is check-then-decrement (non-atomic);
    /// stores with native atomic primitives (in-memory write locks, Redis
    /// Lua) override it.
    async fn decr_if_exists(&self, key: &str, by: f64) -> Result<bool, Error> {
        if !by.is_finite() {
            return Err(Error::Config(
                "decr_if_exists rejected non-finite delta".into(),
            ));
        }
        if self.counter_value(key).await.is_none() {
            return Ok(false);
        }
        self.incr(key, -by, None).await.map(|_| true)
    }

    /// Read a counter directly as `f64`. Avoids the string serialization
    /// round-trip of `get().parse()` for stores that keep native counters.
    /// Returns `None` when the key is absent.
    ///
    /// If a value is present but fails to parse as `f64` (corruption, a
    /// hand-edited store value, or a non-numeric blob written under a counter
    /// key), the parse error is logged at `warn` level and `None` is returned.
    /// For a budget/RPM counter this makes the corruption observable instead
    /// of silently reading as zero (which would reset the caller's spend ceiling
    /// to "unlimited" until the window TTL expires). Concrete stores with native
    /// counters should override this.
    async fn counter_value(&self, key: &str) -> Option<f64> {
        self.try_counter_value(key).await.unwrap_or_default()
    }

    /// Like [`Cache::counter_value`] but distinguishes a genuine key absence
    /// (`Ok(None)`) from a backing-store failure (`Err`). The default
    /// implementation delegates to [`Cache::get`] and never errors, which is
    /// correct for in-memory stores; remote stores (Redis) override this to
    /// surface transient outages so callers that must not act on a missing
    /// reading (e.g. alerting budget-reset detection) can skip the evaluation
    /// instead of mistaking an outage for a reset.
    async fn try_counter_value(&self, key: &str) -> Result<Option<f64>, Error> {
        match self.get(key).await {
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

    /// Liveness probe for the backing store. Returns `true` when the store is
    /// reachable. The default implementation returns `true` (in-memory caches
    /// are always healthy); remote stores (Redis) override with a real ping.
    async fn ping(&self) -> bool {
        true
    }

    /// Fan out a cache-key invalidation to other replicas. The default
    /// implementation is a no-op (single-instance in-memory caches have no
    /// peers). Stores backed by a shared pub/sub layer (Redis) override this
    /// to broadcast the invalidated key so every replica can clear its local
    /// in-memory copy. The caller is still responsible for clearing the local
    /// cache entry via [`Cache::delete`].
    async fn publish_invalidation(&self, _key: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory `Cache` whose `incr` succeeds for non-negative
    /// deltas and fails for negative deltas (simulating a store that cannot
    /// roll back). Used to exercise the *default* `incr_if_below` rollback
    /// path — it deliberately does NOT override `incr_if_below`.
    struct FailingRollbackCache {
        value: std::sync::Mutex<f64>,
    }

    #[async_trait]
    impl Cache for FailingRollbackCache {
        async fn get(&self, _key: &str) -> Option<String> {
            Some(self.value.lock().unwrap().to_string())
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(&self, _key: &str, by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
            if by < 0.0 {
                return Err(Error::Network(
                    "rollback not supported by this store".to_string(),
                ));
            }
            let mut v = self.value.lock().unwrap();
            *v += by;
            Ok(*v)
        }
    }

    /// A store whose `get` returns a non-numeric string under a counter key,
    /// exercising the default `counter_value` parse-failure path.
    struct GarbageCounterCache;

    #[async_trait]
    impl Cache for GarbageCounterCache {
        async fn get(&self, _key: &str) -> Option<String> {
            Some("not-a-number".to_string())
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
            Ok(0.0)
        }
    }

    #[tokio::test]
    async fn incr_if_below_allows_when_under_ceiling() {
        let cache = FailingRollbackCache {
            value: std::sync::Mutex::new(0.0),
        };
        let res = cache.incr_if_below("k", 1.0, 10.0, None).await.unwrap();
        assert_eq!(res, IncrResult::Allowed(1.0));
    }

    #[tokio::test]
    async fn incr_if_below_rejects_when_rollback_succeeds() {
        // A store whose rollback DOES succeed (override) returns Rejected.
        struct OkCache {
            value: std::sync::Mutex<f64>,
        }
        #[async_trait]
        impl Cache for OkCache {
            async fn get(&self, _key: &str) -> Option<String> {
                Some(self.value.lock().unwrap().to_string())
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, Error> {
                let mut v = self.value.lock().unwrap();
                *v += by;
                Ok(*v)
            }
        }
        let cache = OkCache {
            value: std::sync::Mutex::new(9.0),
        };
        let res = cache.incr_if_below("k", 5.0, 10.0, None).await.unwrap();
        // 9 + 5 = 14 > 10 → rejected; rollback (−5) succeeds → counter back to 9.
        assert_eq!(res, IncrResult::Rejected(9.0));
    }

    #[tokio::test]
    async fn incr_if_below_surfaces_rollback_failure_as_error() {
        // The default previously did `let _ = incr(key, -delta)` and returned a
        // false `Rejected`, hiding a broken rollback. It must now surface the
        // store error so the counter/ceiling misalignment is visible.
        let cache = FailingRollbackCache {
            value: std::sync::Mutex::new(9.0),
        };
        let res = cache.incr_if_below("k", 5.0, 10.0, None).await;
        assert!(
            res.is_err(),
            "rollback failure must surface as Err, not a false Rejected"
        );
    }

    #[tokio::test]
    async fn counter_value_returns_none_on_non_numeric_store_value() {
        // A corrupted (non-numeric) counter value must not silently parse as
        // zero/unlimited; it logs a warning and returns None (absent).
        let cache = GarbageCounterCache;
        assert_eq!(cache.counter_value("spend").await, None);
    }

    /// LOW (C10): the corruption-path warn truncates the store value for
    /// logging. A value whose byte 200 falls inside a multi-byte UTF-8
    /// character used to panic the `&v[..200]` slice — precisely on the
    /// hand-edited / corrupted-store path this code exists to defend. The
    /// truncation must be char-boundary-safe: `Ok(None)` + warn, no panic.
    #[tokio::test]
    async fn counter_value_warn_truncation_is_char_boundary_safe() {
        struct MultiByteGarbageCache;
        #[async_trait]
        impl Cache for MultiByteGarbageCache {
            async fn get(&self, _key: &str) -> Option<String> {
                // 300 multi-byte characters = 900 UTF-8 bytes; byte 200 is
                // mid-character (each '米' is 3 bytes; 200 % 3 != 0).
                Some("米".repeat(300))
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                _by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, Error> {
                Ok(0.0)
            }
        }
        let cache = MultiByteGarbageCache;
        let res = cache.try_counter_value("spend").await;
        assert!(
            matches!(res, Ok(None)),
            "multi-byte corrupted value must parse as absent without panicking: {res:?}"
        );
        assert_eq!(cache.counter_value("spend").await, None);
    }

    #[tokio::test]
    async fn counter_value_returns_none_on_absent_key() {
        struct EmptyCache;
        #[async_trait]
        impl Cache for EmptyCache {
            async fn get(&self, _key: &str) -> Option<String> {
                None
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                _by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, Error> {
                Ok(0.0)
            }
        }
        let cache = EmptyCache;
        assert_eq!(cache.counter_value("missing").await, None);
    }

    #[tokio::test]
    async fn incr_if_below_reject_on_absent_key_leaves_zero_or_absent() {
        // The default impl's reject path rolls back the speculative `incr`,
        // which may materialize a previously-absent key as present-but-zero
        // (the doc on `incr_if_below` calls this out). For the shipped stores
        // (moka, Redis) a near-zero budget/RPM counter is functionally
        // identical to an absent one. This test locks in that benign
        // contract: after a reject on a previously-absent key, the key is
        // either still absent OR reads as zero — never overshot past the
        // ceiling and never carrying a non-zero residual.
        struct TrackingCache {
            value: std::sync::Mutex<Option<f64>>,
        }
        #[async_trait]
        impl Cache for TrackingCache {
            async fn get(&self, _key: &str) -> Option<String> {
                self.value.lock().unwrap().map(|v| v.to_string())
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, Error> {
                let mut v = self.value.lock().unwrap();
                let new = v.unwrap_or(0.0) + by;
                *v = Some(new);
                Ok(new)
            }
        }
        let cache = TrackingCache {
            value: std::sync::Mutex::new(None),
        };
        let res = cache.incr_if_below("k", 5.0, 1.0, None).await.unwrap();
        assert!(matches!(res, IncrResult::Rejected(_)));
        let after = cache.counter_value("k").await;
        match after {
            None => { /* still absent — fine */ }
            Some(v) => {
                assert!(
                    v.abs() < f64::EPSILON,
                    "rejected rollback must leave zero (or absent), got {v}"
                );
            }
        }
    }

    #[tokio::test]
    async fn incr_if_below_with_nan_delta_returns_err_and_does_not_poison() {
        // A NaN delta must be rejected up front rather than written into the
        // counter — otherwise the budget/RPM scope bricks for the whole TTL.
        struct TrackingCache {
            value: std::sync::Mutex<Option<f64>>,
        }
        #[async_trait]
        impl Cache for TrackingCache {
            async fn get(&self, _key: &str) -> Option<String> {
                self.value.lock().unwrap().map(|v| v.to_string())
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, Error> {
                let mut v = self.value.lock().unwrap();
                let new = v.unwrap_or(0.0) + by;
                *v = Some(new);
                Ok(new)
            }
        }
        let cache = TrackingCache {
            value: std::sync::Mutex::new(None),
        };
        let res = cache.incr_if_below("k", f64::NAN, 10.0, None).await;
        assert!(
            res.is_err(),
            "NaN delta must be rejected, not written: {res:?}"
        );
        // The guard runs before `incr`, so the counter stays unpoisoned.
        assert_eq!(
            cache.counter_value("k").await,
            None,
            "counter was poisoned by a rejected NaN delta"
        );
    }

    #[tokio::test]
    async fn incr_if_below_with_inf_ceiling_returns_err() {
        // A non-finite ceiling is meaningless as a limit and must be rejected
        // rather than silently allowing the increment.
        struct LinearCache {
            value: std::sync::Mutex<f64>,
        }
        #[async_trait]
        impl Cache for LinearCache {
            async fn get(&self, _key: &str) -> Option<String> {
                Some(self.value.lock().unwrap().to_string())
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, Error> {
                let mut v = self.value.lock().unwrap();
                *v += by;
                Ok(*v)
            }
        }
        let cache = LinearCache {
            value: std::sync::Mutex::new(0.0),
        };
        let res = cache.incr_if_below("k", 1.0, f64::INFINITY, None).await;
        assert!(res.is_err(), "infinite ceiling must be rejected: {res:?}");
    }

    #[tokio::test]
    async fn incr_if_below_with_inf_delta_returns_err() {
        // A +∞ delta would always reject and roll back to −∞, bricking the
        // counter. Reject it up front.
        struct LinearCache {
            value: std::sync::Mutex<f64>,
        }
        #[async_trait]
        impl Cache for LinearCache {
            async fn get(&self, _key: &str) -> Option<String> {
                Some(self.value.lock().unwrap().to_string())
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, Error> {
                let mut v = self.value.lock().unwrap();
                *v += by;
                Ok(*v)
            }
        }
        let cache = LinearCache {
            value: std::sync::Mutex::new(0.0),
        };
        let res = cache.incr_if_below("k", f64::INFINITY, 10.0, None).await;
        assert!(res.is_err(), "infinite delta must be rejected: {res:?}");
    }
}
