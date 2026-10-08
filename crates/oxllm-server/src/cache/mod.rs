//! Cache backends.
//!
//! - [`InMemoryCache`] / [`DualCache`] (this file): in-memory `Cache`
//!   implementation backed by `moka`, plus the `DualCache` wrapper that
//!   layers in-memory + Redis. Used for: response cache, rate-limit/TPM
//!   counters, and the API-key cache. Single-instance only —
//!   `oxllm_cache_redis::RedisCache` makes this fleet-correct.
//! - [`disk`]: on-disk persistent cache backend.
//! - [`semantic`]: embedding-based semantic cache.
//! - [`redis_semantic`]: Redis-backed semantic cache.
//! - [`qdrant`]: Qdrant-backed semantic cache.

pub mod disk;
pub mod qdrant;
pub mod redis_semantic;
pub mod semantic;

pub use self::{disk::*, qdrant::*, redis_semantic::*, semantic::*};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use moka::Expiry;
use moka::future::Cache;
use oxllm_core::{Cache as CacheTrait, Error, IncrResult};
use tokio::sync::RwLock;

#[cfg(feature = "redis")]
use oxllm_cache_redis::RedisCache;

/// An entry stored in the moka cache, optionally with a per-entry expiry.
#[derive(Clone, Debug)]
struct CachedEntry {
    value: String,
    expires_at: Option<Instant>,
}

/// Per-entry expiry implementation: moka consults this on create/update to
/// determine when an individual entry should expire.
struct EntryExpiry;

impl Expiry<String, CachedEntry> for EntryExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &CachedEntry,
        _created_at: Instant,
    ) -> Option<Duration> {
        value
            .expires_at
            .map(|at| at.saturating_duration_since(Instant::now()))
    }

    fn expire_after_update(
        &self,
        _key: &String,
        value: &CachedEntry,
        _updated_at: Instant,
        _current_duration: Option<Duration>,
    ) -> Option<Duration> {
        value
            .expires_at
            .map(|at| at.saturating_duration_since(Instant::now()))
    }
}

/// Upper bound (seconds) for any cache-entry TTL. `Instant + Duration`
/// panics on overflow, and TTLs are sometimes derived from provider-issued
/// input (e.g. a JWT `exp` claim clamped nowhere before reaching the cache);
/// clamping far below `Instant`'s range turns an absurd TTL into a
/// long-but-safe window instead of a panic.
const MAX_TTL_SECS: u64 = 30 * 24 * 60 * 60;

/// Upper bound on the byte size of a single cached value, mirroring the reference
/// `MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB` (1 MiB). [`InMemoryCache::set`]
/// skips (and logs) values larger than this so a single large response body
/// cannot consume disproportionate memory. Moka's `max_capacity` bounds entry
/// *count*, not total bytes, so without this cap a client targeting a model
/// with large `max_tokens` and cache enabled could store ~50 MiB entries (the
/// response body cap) per slot.
const MAX_VALUE_BYTES: usize = 1024 * 1024;

/// Clamp a TTL so `Instant + Duration` can never overflow.
fn clamp_ttl(ttl: Duration) -> Duration {
    ttl.min(Duration::from_secs(MAX_TTL_SECS))
}

/// Per-counter state: value plus a fixed expiry window (not sliding).
#[derive(Clone, Debug)]
struct CounterState {
    value: f64,
    window_start: Instant,
    expires_at: Option<Instant>,
}

impl CounterState {
    fn fresh(value: f64, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            window_start: now,
            expires_at: ttl.map(|d| now + clamp_ttl(d)),
        }
    }

    fn is_expired(&self, now: Instant) -> bool {
        self.expires_at.is_some_and(|at| at <= now)
    }
}

/// Base name of the cross-replica cache-invalidation pub/sub channel.
///
/// Both the publisher ([`DualCache::publish_invalidation`]) and the subscriber
/// ([`crate::app::invalidation_subscriber_loop`]) namespace this name with the
/// configured Redis key prefix via [`RedisCache::pubsub_channel`], so two oxllm
/// fleets sharing one Redis instance with different prefixes (`fleet-a:`,
/// `fleet-b:`) do not cross-deliver invalidation messages. Using a single
/// constant on both sides keeps the publish/subscribe channel in sync.
pub const INVALIDATION_CHANNEL_BASE: &str = "invalidate";

/// Number of counter-write calls between sweeps of expired `counters` entries.
/// The `counters` map is unbounded (unlike the capacity-bounded moka `store`),
/// so `incr`/`incr_if_below` sweep expired entries on every Nth call to bound
/// the amortized cost — the same pattern as `AuthBrake::sweep_locked`. Without
/// a sweep, every distinct `spend:*` / `rpm:*` / `jti:seen:*` key ever
/// incremented would accumulate forever (a slow memory-growth / DoS surface).
const COUNTER_SWEEP_INTERVAL: u64 = 100;

/// Soft cap on the `counters` map size. The expiry sweep reclaims TTL'd
/// entries; TTL-less counters (`expires_at == None`, e.g. `spend:*` rows) are
/// never reclaimed by expiry, so once the map crosses this cap the sweep also
/// evicts the oldest TTL-less entries. Set high enough that healthy traffic
/// (spend keyed by virtual-key hash, RPM/TPM/jti TTL'd) never approaches it —
/// it is a safety net against a future hook that increments unbounded distinct
/// TTL-less keys (e.g. a per-request counter).
const DEFAULT_COUNTER_SOFT_CAP: usize = 100_000;

/// An in-memory cache: moka for `get/set/delete`, a RwLock-protected
/// `HashMap` for `incr` counters (so we can track per-counter TTL windows).
pub struct InMemoryCache {
    store: Cache<String, CachedEntry>,
    counters: RwLock<HashMap<String, CounterState>>,
    sweep_calls: AtomicU64,
    /// Upper bound on the `counters` map size. Once exceeded, the sweep evicts
    /// the oldest TTL-less entries — see [`DEFAULT_COUNTER_SOFT_CAP`].
    counter_soft_cap: usize,
}

impl InMemoryCache {
    /// Create a new in-memory cache with the given max capacity.
    pub fn new(capacity: u64) -> Self {
        let store = Cache::builder()
            .max_capacity(capacity)
            .expire_after(EntryExpiry)
            .build();
        Self {
            store,
            counters: RwLock::new(HashMap::new()),
            sweep_calls: AtomicU64::new(0),
            counter_soft_cap: DEFAULT_COUNTER_SOFT_CAP,
        }
    }

    /// Create a new in-memory cache with a non-default counter soft cap.
    /// Exposed for tests that exercise the TTL-less eviction path without
    /// having to insert ~100k entries.
    #[cfg(test)]
    pub(crate) fn with_counter_soft_cap(capacity: u64, counter_soft_cap: usize) -> Self {
        let store = Cache::builder()
            .max_capacity(capacity)
            .expire_after(EntryExpiry)
            .build();
        Self {
            store,
            counters: RwLock::new(HashMap::new()),
            sweep_calls: AtomicU64::new(0),
            counter_soft_cap,
        }
    }

    /// Evict expired counter entries on an amortized schedule. Called under the
    /// counters write lock by [`Self::incr`] / [`Self::incr_if_below`] every
    /// [`COUNTER_SWEEP_INTERVAL`]th call. TTL makes an expired counter read as
    /// absent, but the entry stays in the map until swept — without this the
    /// map would grow without bound.
    ///
    /// Also evicts the oldest TTL-less entries once the map exceeds
    /// [`Self::counter_soft_cap`]: TTL-less counters are never reclaimed by
    /// the expiry pass, so without this second stage a future hook that
    /// increments unbounded distinct `ttl=None` keys (e.g. a per-request
    /// counter) would be a slow memory leak.
    fn maybe_sweep_locked(
        &self,
        guard: &mut tokio::sync::RwLockWriteGuard<'_, HashMap<String, CounterState>>,
    ) {
        let calls = self.sweep_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if calls % COUNTER_SWEEP_INTERVAL != 0 {
            return;
        }
        self.sweep_locked(guard);
    }

    /// The actual sweep body — expired-entry eviction followed by soft-cap
    /// eviction of the oldest TTL-less entries. Split out from
    /// [`Self::maybe_sweep_locked`] so [`Self::sweep_now`] can run the full
    /// sweep deterministically in tests.
    fn sweep_locked(
        &self,
        guard: &mut tokio::sync::RwLockWriteGuard<'_, HashMap<String, CounterState>>,
    ) {
        let now = Instant::now();
        let before = guard.len();
        guard.retain(|_, state| !state.is_expired(now));
        let expired_removed = before.saturating_sub(guard.len());
        // Second stage: bound TTL-less growth past the soft cap. Only evicts
        // when over the cap, and only TTL-less entries (TTL'd ones are bounded
        // by the expiry pass above). Oldest-first by `window_start`.
        let mut ttl_less_removed = 0usize;
        if guard.len() > self.counter_soft_cap {
            let excess = guard.len() - self.counter_soft_cap;
            let mut ttl_less: Vec<(Instant, String)> = guard
                .iter()
                .filter(|(_, s)| s.expires_at.is_none())
                .map(|(k, s)| (s.window_start, k.clone()))
                .collect();
            ttl_less.sort_by_key(|(t, _)| *t);
            let to_evict: HashMap<String, ()> = ttl_less
                .into_iter()
                .take(excess)
                .map(|(_, k)| (k, ()))
                .collect();
            let before_cap = guard.len();
            guard.retain(|k, _| !to_evict.contains_key(k));
            ttl_less_removed = before_cap.saturating_sub(guard.len());
        }
        let removed = expired_removed + ttl_less_removed;
        if removed > 0 {
            tracing::debug!(
                removed,
                expired_removed,
                ttl_less_removed,
                remaining = guard.len(),
                "counter sweep: evicted expired and oldest TTL-less counters"
            );
        }
    }

    /// Force a full sweep (expired + soft-cap TTL-less eviction) now. Exposed
    /// for tests so the cap can be exercised deterministically without
    /// depending on the call counter.
    #[cfg(test)]
    pub(crate) async fn sweep_now(&self) {
        let mut guard = self.counters.write().await;
        self.sweep_locked(&mut guard);
    }

    /// Current number of entries in the counters map (including expired ones
    /// not yet swept). Exposed for tests to assert the map is bounded.
    #[cfg(test)]
    pub(crate) async fn counters_len(&self) -> usize {
        self.counters.read().await.len()
    }

    /// Reset a counter to zero (used by the budget reset scheduler).
    pub async fn reset_counter(&self, key: &str) {
        {
            let mut guard = self.counters.write().await;
            guard.insert(key.to_string(), CounterState::fresh(0.0, None));
        }
        self.store.invalidate(key).await;
    }

    /// Returns all counter keys (excluding internal bookkeeping). Used by the
    /// scheduler to iterate over spend/RPM/TPM counters.
    pub async fn counter_keys(&self) -> Vec<String> {
        let now = Instant::now();
        let guard = self.counters.read().await;
        guard
            .iter()
            .filter(|(_, state)| !state.is_expired(now))
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Returns the instant at which the counter's window started, or `None`
    /// if the counter has never been incremented or has expired.
    pub async fn counter_window_start(&self, key: &str) -> Option<Instant> {
        let now = Instant::now();
        let guard = self.counters.read().await;
        guard.get(key).and_then(|state| {
            if state.is_expired(now) {
                None
            } else {
                Some(state.window_start)
            }
        })
    }

    /// Returns the instant at which the counter expires, or `None` if the
    /// counter has never been incremented, has expired, or has no TTL.
    /// Exposed for tests to assert a bounded TTL is applied.
    #[cfg(test)]
    pub(crate) async fn counter_expires_at(&self, key: &str) -> Option<Instant> {
        let now = Instant::now();
        let guard = self.counters.read().await;
        guard.get(key).and_then(|state| {
            if state.is_expired(now) {
                None
            } else {
                state.expires_at
            }
        })
    }

    /// Atomically check eligibility and zero a counter. Under one write lock,
    /// returns `true` and zeroes the counter iff ALL of:
    /// - the counter exists and is not expired,
    /// - its global `window_start` is older than `budget_duration`,
    /// - its own per-key TTL (if any) does not extend past `now + ttl_grace`
    ///   (i.e. the cache itself still considers it about to expire).
    ///
    /// Folding the eligibility check and the zero into a single critical
    /// section closes the TOCTOU where a concurrent `incr_if_below` landing
    /// between a separate check and [`Self::reset_counter`] would be silently
    /// zeroed. It also honors per-key TTLs: a counter whose own TTL outlives
    /// the global `budget_duration` is left untouched so longer per-key
    /// windows are never truncated.
    pub async fn reset_spend_counter_if_due(
        &self,
        key: &str,
        budget_duration: Duration,
        ttl_grace: Duration,
    ) -> bool {
        let now = Instant::now();
        let reset = {
            let mut guard = self.counters.write().await;
            let Some(state) = guard.get(key) else {
                return false;
            };
            let window_start = state.window_start;
            let expires_at = state.expires_at;
            if state.is_expired(now) {
                return false;
            }
            if window_start.elapsed() < budget_duration {
                return false;
            }
            if let Some(expires_at) = expires_at {
                if expires_at > now + clamp_ttl(ttl_grace) {
                    return false;
                }
            }
            // A reset TTL-less counter is semantically absent (every budget
            // caller reads `counter_value(...).unwrap_or(0.0)`), so drop the
            // entry rather than leaving a never-expiring zero row that the
            // sweep would never reclaim. This mirrors the Redis path's
            // `delete` semantics (see `InMemoryCache::delete`).
            guard.remove(key);
            true
        };
        if reset {
            self.store.invalidate(key).await;
        }
        reset
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[async_trait]
impl CacheTrait for InMemoryCache {
    async fn get(&self, key: &str) -> Option<String> {
        let entry = self.store.get(key).await?;
        if let Some(at) = entry.expires_at {
            if at <= Instant::now() {
                self.store.invalidate(key).await;
                return None;
            }
        }
        Some(entry.value)
    }

    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        if ttl.is_some_and(|ttl| ttl.is_zero()) {
            self.delete(key).await;
            return;
        }
        if value.len() > MAX_VALUE_BYTES {
            tracing::warn!(
                key = %key,
                size = value.len(),
                cap = MAX_VALUE_BYTES,
                "in-memory cache set skipped oversized value"
            );
            return;
        }
        let expires_at = ttl.map(|d| Instant::now() + clamp_ttl(d));
        let entry = CachedEntry { value, expires_at };
        self.store.insert(key.to_string(), entry).await;
    }

    async fn delete(&self, key: &str) {
        self.store.invalidate(key).await;
        // Counters live in a separate map (not the moka store); clear them too
        // so a subsequent `counter_value` returns None rather than a stale
        // residual after the cached entry was deleted.
        self.counters.write().await.remove(key);
    }

    /// Increment a counter by `by`, atomically.
    ///
    /// U7: unlike the reference in-memory cache whose
    /// `increment` was non-atomic and lost counts under concurrency, this
    /// implementation acquires `self.counters.write().await` for the entire
    /// read-modify-write critical section, so concurrent callers can never
    /// lose counts. The same write-lock discipline is used by
    /// [`incr_if_below`](InMemoryCache::incr_if_below).
    async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error> {
        // Defense-in-depth mirroring `Cache::incr_if_below`'s default guard: a
        // non-finite `by` (NaN, ±∞) written via the best-effort settlement
        // paths (`BudgetHook::adjust`/`release`) would poison the in-memory
        // counter (`state.value += by`), and every subsequent `incr_if_below`
        // would see `NaN <= ceiling == false` → `Rejected`, bricking that
        // budget/RPM scope for the rest of its window TTL. Redis's
        // `INCRBYFLOAT` already rejects non-finite values; only the in-memory
        // store is exposed.
        if !by.is_finite() {
            return Err(Error::Config("incr rejected non-finite delta".into()));
        }
        let (new_value, expires_at) = {
            let mut guard = self.counters.write().await;
            self.maybe_sweep_locked(&mut guard);
            let now = Instant::now();
            let state = guard.remove(key);
            let mut state = match state {
                Some(s) if !s.is_expired(now) => s,
                _ => CounterState::fresh(0.0, ttl),
            };
            state.value += by;
            // Fixed window: only set expiry when the counter is new / reset.
            if state.expires_at.is_none() {
                if let Some(d) = ttl {
                    state.expires_at = Some(now + clamp_ttl(d));
                }
            }
            let new_value = state.value;
            let expires_at = state.expires_at;
            guard.insert(key.to_string(), state);
            (new_value, expires_at)
        };
        let entry = CachedEntry {
            value: new_value.to_string(),
            expires_at,
        };
        self.store.insert(key.to_string(), entry).await;
        Ok(new_value)
    }

    /// Truly atomic compare-and-increment: the counters `RwLock` is held for
    /// write across the read-check-write, so concurrent callers within this
    /// instance can never exceed `ceiling`.
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
        let (result, applied_expires) = {
            let mut guard = self.counters.write().await;
            self.maybe_sweep_locked(&mut guard);
            let now = Instant::now();
            let existing = guard.remove(key);
            let was_present = matches!(existing.as_ref(), Some(s) if !s.is_expired(now));
            let mut state = match existing {
                Some(s) if !s.is_expired(now) => s,
                _ => CounterState::fresh(0.0, ttl),
            };
            let current = state.value;
            let new_val = current + delta;
            if new_val <= ceiling {
                state.value = new_val;
                if state.expires_at.is_none() {
                    if let Some(d) = ttl {
                        state.expires_at = Some(now + clamp_ttl(d));
                    }
                }
                let expires_at = state.expires_at;
                guard.insert(key.to_string(), state);
                (IncrResult::Allowed(new_val), Some((new_val, expires_at)))
            } else {
                // Re-insert only when the entry was already present: a
                // freshly-created zero counter on the reject path would start
                // a TTL window ticking for a key that was never committed
                // (and disagrees with `DiskCache`, which never writes on the
                // reject path). For a present counter, the unchanged state is
                // put back so its expiry/window are preserved.
                if was_present {
                    guard.insert(key.to_string(), state);
                }
                (IncrResult::Rejected(current), None)
            }
        };
        if let Some((new_val, expires_at)) = applied_expires {
            let entry = CachedEntry {
                value: new_val.to_string(),
                expires_at,
            };
            self.store.insert(key.to_string(), entry).await;
        }
        Ok(result)
    }

    async fn counter_value(&self, key: &str) -> Option<f64> {
        let now = Instant::now();
        let guard = self.counters.read().await;
        guard.get(key).and_then(|state| {
            if state.is_expired(now) {
                None
            } else {
                Some(state.value)
            }
        })
    }

    /// Atomically decrement only when the counter exists (see
    /// [`oxllm_core::Cache::decr_if_exists`]): the counters write lock is
    /// held across the existence check and the decrement, so a concurrent
    /// window rollover cannot interleave and mint negative residue.
    async fn decr_if_exists(&self, key: &str, by: f64) -> Result<bool, Error> {
        if !by.is_finite() {
            return Err(Error::Config(
                "decr_if_exists rejected non-finite delta".into(),
            ));
        }
        let mut applied: Option<(f64, Option<Instant>)> = None;
        {
            let mut guard = self.counters.write().await;
            self.maybe_sweep_locked(&mut guard);
            let now = Instant::now();
            if let Some(state) = guard.get_mut(key) {
                if !state.is_expired(now) {
                    state.value -= by;
                    applied = Some((state.value, state.expires_at));
                }
            }
        }
        match applied {
            Some((new_value, expires_at)) => {
                let entry = CachedEntry {
                    value: new_value.to_string(),
                    expires_at,
                };
                self.store.insert(key.to_string(), entry).await;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// In-memory counters never experience outages, so the authoritative
    /// reading is infallible: delegate to [`Self::counter_value`] (the native
    /// counters map) rather than the default `get`-backed path, which would
    /// read the mirrored moka string and could disagree with the counters map.
    async fn try_counter_value(&self, key: &str) -> Result<Option<f64>, Error> {
        Ok(self.counter_value(key).await)
    }
}

/// A dual-layer cache: in-memory first, Redis second.
///
/// - `get`: check memory first, miss → Redis, backfill memory.
/// - `set`: write to both.
/// - `incr`: increment both (Redis is the source of truth for cross-instance).
/// - `delete`: delete from both.
///
/// When Redis is `None`, behaves as a pure in-memory cache.
///
/// U16: the read path must NEVER re-publish a
/// memory value back to Redis. The reference bug was that after a key update
/// deleted the Redis blob, the read-path writer re-published the stale
/// in-memory blob to Redis, un-doing the invalidation across replicas. oxllm
/// is safe by design: `get` only backfills **Redis → memory** (one-way, with
/// a bounded 60s TTL so revoked entries cannot stick forever). The
/// authoritative write path is `set`, called only from the DB-load path
/// (`lookup_virtual_key`), never from a cache `get`. There is no code path in
/// `get` that calls `redis.set`.
#[cfg(feature = "redis")]
pub struct DualCache {
    memory: Arc<InMemoryCache>,
    redis: Option<RedisCache>,
}

#[cfg(feature = "redis")]
impl Default for DualCache {
    fn default() -> Self {
        Self {
            memory: Arc::new(InMemoryCache::default()),
            redis: None,
        }
    }
}

#[cfg(feature = "redis")]
impl DualCache {
    /// Create a DualCache with an optional Redis backend.
    pub fn new(memory: Arc<InMemoryCache>, redis: Option<RedisCache>) -> Self {
        Self { memory, redis }
    }

    /// Access the underlying in-memory cache.
    pub fn memory(&self) -> &InMemoryCache {
        &self.memory
    }

    /// Access the underlying Redis cache when configured. Used by the
    /// startup wiring to spawn the pub/sub subscriber task.
    pub fn redis_cache(&self) -> Option<&RedisCache> {
        self.redis.as_ref()
    }
}

#[cfg(feature = "redis")]
#[async_trait]
impl CacheTrait for DualCache {
    async fn get(&self, key: &str) -> Option<String> {
        if let Some(val) = self.memory.get(key).await {
            return Some(val);
        }
        if let Some(redis) = &self.redis {
            if let Some((val, remaining)) = redis.get_with_ttl(key).await {
                // Bound auth backfills to 60s, but never extend a shorter
                // authoritative Redis expiry. GET + PTTL are one atomic read,
                // and get_with_ttl already subtracts the network round trip.
                let ttl = remaining
                    .unwrap_or(Duration::from_secs(60))
                    .min(Duration::from_secs(60));
                self.memory.set(key, val.clone(), Some(ttl)).await;
                return Some(val);
            }
        }
        None
    }

    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        self.memory.set(key, value.clone(), ttl).await;
        if let Some(redis) = &self.redis {
            redis.set(key, value, ttl).await;
        }
    }

    async fn delete(&self, key: &str) {
        self.memory.delete(key).await;
        if let Some(redis) = &self.redis {
            redis.delete(key).await;
        }
    }

    async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error> {
        // When Redis is configured it is the fleet source of truth: apply
        // there first, then mirror to local memory so a Redis failure does
        // not leave a divergent local counter ahead of the shared store. If
        // Redis fails, do not advance memory — return the error so callers do
        // not treat the call as completed.
        if let Some(redis) = &self.redis {
            let redis_val = redis.incr(key, by, ttl).await?;
            // Redis is authoritative, so a local mirror failure does not
            // corrupt the fleet counter — but it does cause this instance's
            // warm reads to diverge from the shared store. Log it rather than
            // silently dropping so the divergence is observable.
            if let Err(e) = self.memory.incr(key, by, ttl).await {
                tracing::warn!(error = %e, key = %key, "memory incr failed to mirror redis");
            }
            Ok(redis_val)
        } else {
            self.memory.incr(key, by, ttl).await
        }
    }

    /// When Redis is configured, the authoritative atomic decision is made in
    /// Redis (cross-instance correct); the result is then mirrored to the
    /// local memory counter best-effort so subsequent reads stay warm. When
    /// Redis is absent, the in-memory atomic impl is used.
    async fn incr_if_below(
        &self,
        key: &str,
        delta: f64,
        ceiling: f64,
        ttl: Option<Duration>,
    ) -> Result<IncrResult, Error> {
        if let Some(redis) = &self.redis {
            let result = redis.incr_if_below(key, delta, ceiling, ttl).await?;
            if matches!(result, IncrResult::Allowed(_)) {
                // Best-effort local mirror of the authoritative Redis decision;
                // log on failure so divergence is observable (see `incr`).
                if let Err(e) = self.memory.incr(key, delta, ttl).await {
                    tracing::warn!(error = %e, key = %key, "memory incr failed to mirror redis incr_if_below");
                }
            }
            Ok(result)
        } else {
            self.memory.incr_if_below(key, delta, ceiling, ttl).await
        }
    }

    /// Atomically decrement only when the counter exists (see
    /// [`oxllm_core::Cache::decr_if_exists`]). This override is required even
    /// though both layers already implement it atomically: the trait default
    /// is a non-atomic `counter_value` check followed by `incr(key, -by)`,
    /// so a window rollover between the two calls would mint a negative,
    /// never-expiring counter — fleet-wide when Redis is the store —
    /// permanently inflating the budget/RPM ceiling. Exactly what
    /// `RedisCache::DECR_IF_EXISTS_SCRIPT` and [`InMemoryCache::decr_if_exists`]
    /// exist to prevent. Redis is authoritative when configured; the local
    /// mirror is applied best-effort only when Redis applied the decrement.
    async fn decr_if_exists(&self, key: &str, by: f64) -> Result<bool, Error> {
        if let Some(redis) = &self.redis {
            let applied = redis.decr_if_exists(key, by).await?;
            if applied {
                // Best-effort local mirror of the authoritative Redis
                // decision; log on failure so divergence is observable
                // (see `incr`).
                if let Err(e) = self.memory.decr_if_exists(key, by).await {
                    tracing::warn!(
                        error = %e,
                        key = %key,
                        "memory decr_if_exists failed to mirror redis"
                    );
                }
            }
            Ok(applied)
        } else {
            self.memory.decr_if_exists(key, by).await
        }
    }

    async fn counter_value(&self, key: &str) -> Option<f64> {
        // Redis is the fleet source of truth when configured. A miss must
        // stay a miss — falling through to the local counter resurrects
        // expired/deleted Redis keys after a window reset.
        if let Some(redis) = &self.redis {
            return redis.counter_value(key).await;
        }
        self.memory.counter_value(key).await
    }

    /// When Redis is configured it is the fleet source of truth: a Redis
    /// backend error must surface as `Err` (not be masked as `None`) so callers
    /// like the alerting budget-reset detector can skip evaluation during an
    /// outage instead of mistaking it for a reset and wiping dedup markers
    /// (which would cause an alert storm on recovery). A genuine Redis miss
    /// is a miss — do not fall through to a stale local counter.
    async fn try_counter_value(&self, key: &str) -> Result<Option<f64>, Error> {
        if let Some(redis) = &self.redis {
            return redis.try_counter_value(key).await;
        }
        Ok(self.memory.counter_value(key).await)
    }

    async fn ping(&self) -> bool {
        match &self.redis {
            Some(redis) => redis.ping().await,
            None => true,
        }
    }

    /// Broadcast the invalidated key on the Redis pub/sub channel so every
    /// replica's subscriber task can clear its local moka entry. Best-effort:
    /// a publish failure is logged but does not surface to the caller, since
    /// the local cache has already been cleared and entries carry a bounded
    /// positive TTL as a backstop.
    ///
    /// The channel is namespaced with the configured Redis key prefix (via
    /// [`RedisCache::pubsub_channel`]) so fleets sharing one Redis instance
    /// with different prefixes do not cross-invalidate each other.
    async fn publish_invalidation(&self, key: &str) {
        if let Some(redis) = &self.redis {
            let channel = redis.pubsub_channel(INVALIDATION_CHANNEL_BASE);
            if let Err(e) = redis.publish(&channel, key).await {
                tracing::warn!(
                    error = %e,
                    key = %key,
                    "failed to publish cache invalidation; other replicas may serve stale data until TTL expiry"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn set_and_get() {
        let cache = InMemoryCache::default();
        cache.set("k", "v".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("v"));
    }

    #[tokio::test]
    async fn zero_ttl_contract_matches_memory_and_disk() {
        let dir = tempfile::tempdir().unwrap();
        let caches: Vec<Box<dyn CacheTrait>> = vec![
            Box::new(InMemoryCache::default()),
            Box::new(DiskCache::new(dir.path()).unwrap()),
        ];
        for cache in caches {
            cache.set("value", "persistent".to_string(), None).await;
            assert_eq!(cache.get("value").await.as_deref(), Some("persistent"));
            cache
                .set("value", "expired".to_string(), Some(Duration::ZERO))
                .await;
            assert!(
                cache.get("value").await.is_none(),
                "zero TTL replaces the old value with absence"
            );

            assert_eq!(
                cache.incr("new", 2.0, Some(Duration::ZERO)).await.unwrap(),
                2.0
            );
            assert!(cache.counter_value("new").await.is_none());
            assert_eq!(
                cache
                    .incr_if_below("reserve", 2.0, 3.0, Some(Duration::ZERO))
                    .await
                    .unwrap(),
                IncrResult::Allowed(2.0)
            );
            assert!(cache.counter_value("reserve").await.is_none());

            cache
                .incr("fixed", 1.0, Some(Duration::from_secs(60)))
                .await
                .unwrap();
            cache
                .incr("fixed", 1.0, Some(Duration::ZERO))
                .await
                .unwrap();
            cache
                .incr_if_below("fixed", 1.0, 4.0, Some(Duration::ZERO))
                .await
                .unwrap();
            assert_eq!(
                cache.counter_value("fixed").await,
                Some(3.0),
                "an existing fixed window is preserved"
            );
        }
    }

    #[test]
    fn expired_moka_entry_does_not_become_eternal() {
        let now = Instant::now();
        let entry = CachedEntry {
            value: "stale".to_string(),
            expires_at: Some(now),
        };
        let expiry = EntryExpiry;
        assert_eq!(
            expiry.expire_after_create(&"k".to_string(), &entry, now),
            Some(Duration::ZERO)
        );
        assert_eq!(
            expiry.expire_after_update(&"k".to_string(), &entry, now, None),
            Some(Duration::ZERO)
        );
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_backfill_preserves_remote_expiry_and_never_republishes() {
        // Explicit local fixture only; no default endpoint or real account.
        let Ok(url) = std::env::var("OXLLM_REDIS_URL") else {
            return;
        };
        let parsed = url::Url::parse(&url).unwrap();
        if !matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "[::1]")) {
            return;
        }
        let prefix = format!("oxllm-cache-review:{}", uuid::Uuid::new_v4());
        let redis = RedisCache::new(&url, &prefix)
            .await
            .expect("configured local Redis fixture");
        let memory = Arc::new(InMemoryCache::default());
        let cache = DualCache::new(memory.clone(), Some(redis.clone()));
        redis
            .set(
                "vk:short",
                "auth-value".to_string(),
                Some(Duration::from_secs(1)),
            )
            .await;
        assert_eq!(cache.get("vk:short").await.as_deref(), Some("auth-value"));
        let entry = memory.store.get("vk:short").await.unwrap();
        // The backfill must track the remote's remaining lifetime, not the
        // 60s auth-cache default. Compare the memory entry's remaining
        // lifetime to the remote TTL (1s) with a small allowance for Redis's
        // millisecond clock granularity and the round trip — a hard
        // pre-round-trip deadline is sub-millisecond brittle.
        let lifetime = entry
            .expires_at
            .unwrap()
            .saturating_duration_since(Instant::now());
        assert!(
            lifetime <= Duration::from_millis(1100),
            "backfill must not grant a new 60s lifetime (remaining {lifetime:?})"
        );
        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert!(redis.get("vk:short").await.is_none());
        assert!(cache.get("vk:short").await.is_none());

        redis.set("vk:invalidate", "old".to_string(), None).await;
        assert!(cache.get("vk:invalidate").await.is_some());
        redis.delete("vk:invalidate").await;
        assert!(cache.get("vk:invalidate").await.is_some());
        assert!(
            redis.get("vk:invalidate").await.is_none(),
            "memory reads must never republish a revoked blob"
        );
        memory.delete("vk:invalidate").await;
        assert!(cache.get("vk:invalidate").await.is_none());
    }

    /// MEDIUM (C11): a decrement on an absent or expired counter must be a
    /// no-op — minting a negative, never-expiring counter would permanently
    /// discount every subsequent budget window's spend.
    #[tokio::test]
    async fn decr_if_exists_never_mints_negative_residue() {
        let cache = InMemoryCache::default();
        // Absent key: no-op, and no counter is created.
        assert!(!cache.decr_if_exists("spend:x", 0.5).await.expect("ok"));
        assert!(cache.counter_value("spend:x").await.is_none());

        // Live counter: decrement applied, value tracked.
        let _ = cache
            .incr("spend:x", 1.0, Some(Duration::from_millis(500)))
            .await;
        assert!(cache.decr_if_exists("spend:x", 0.25).await.expect("ok"));
        assert_eq!(cache.counter_value("spend:x").await, Some(0.75));

        // Expired counter: no-op, no residue minted.
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert!(
            !cache.decr_if_exists("spend:x", 0.75).await.expect("ok"),
            "expired counter must not be decremented"
        );
        assert!(
            cache.counter_value("spend:x").await.is_none(),
            "no negative residue may be minted after window expiry"
        );

        // Non-finite delta is rejected without poisoning.
        assert!(cache.decr_if_exists("spend:x", f64::NAN).await.is_err());
        assert!(cache.counter_value("spend:x").await.is_none());
    }

    #[tokio::test]
    async fn miss_returns_none() {
        let cache = InMemoryCache::default();
        assert!(cache.get("missing").await.is_none());
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let cache = InMemoryCache::default();
        cache.set("k", "v".to_string(), None).await;
        cache.delete("k").await;
        assert!(cache.get("k").await.is_none());
    }

    /// M9: `delete` must also clear the counters map, not just the moka store
    /// entry. A residual counter would make `counter_value` return a stale
    /// value after the cached entry was deleted.
    #[tokio::test]
    async fn delete_clears_counter_state() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("rpm:k", 5.0, None).await;
        assert_eq!(cache.counter_value("rpm:k").await, Some(5.0));
        cache.delete("rpm:k").await;
        assert_eq!(
            cache.counter_value("rpm:k").await,
            None,
            "delete must clear the counters map so no stale counter remains"
        );
    }

    #[tokio::test]
    async fn incr_returns_new_value() {
        let cache = InMemoryCache::default();
        assert!((cache.incr("c", 1.0, None).await.unwrap() - 1.0).abs() < 1e-9);
        assert!((cache.incr("c", 2.0, None).await.unwrap() - 3.0).abs() < 1e-9);
        assert!((cache.incr("c", 0.5, None).await.unwrap() - 3.5).abs() < 1e-9);
    }

    /// LOW (C31): `InMemoryCache::incr` must reject a non-finite delta up front
    /// — otherwise the counter is poisoned (`state.value += by` writes NaN/∞),
    /// and every subsequent `incr_if_below` sees `NaN <= ceiling == false` and
    /// bricks the budget/RPM scope for the window TTL. Mirrors the existing
    /// `incr_if_below` non-finite guard. Redis's `INCRBYFLOAT` already rejects
    /// non-finite values, so only the in-memory store is exposed.
    #[tokio::test]
    async fn incr_rejects_nan_delta_in_memory() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("spend:k", 1.0, None).await.unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let res = cache.incr("spend:k", bad, None).await;
            assert!(
                res.is_err(),
                "incr must reject non-finite delta ({bad:?}), got {res:?}"
            );
        }
        // Counter must be unchanged (not poisoned) after the rejected writes.
        assert_eq!(
            cache.counter_value("spend:k").await,
            Some(1.0),
            "rejected non-finite delta must not mutate the counter"
        );
        // A legitimate reservation on the same key must still succeed — the
        // scope is not bricked.
        let res = cache.incr_if_below("spend:k", 1.0, 10.0, None).await;
        assert!(
            matches!(res, Ok(IncrResult::Allowed(2.0))),
            "scope must remain usable after a rejected non-finite incr, got {res:?}"
        );
    }

    #[tokio::test]
    async fn counter_keys_returns_all_counters() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("rpm:foo", 1.0, None).await;
        let _ = cache.incr("tpm:foo", 100.0, None).await;
        let mut keys = cache.counter_keys().await;
        keys.sort();
        assert_eq!(keys, vec!["rpm:foo".to_string(), "tpm:foo".to_string()]);
    }

    #[tokio::test]
    async fn reset_counter_zeros_value() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("c", 5.0, None).await;
        cache.reset_counter("c").await;
        assert_eq!(cache.counter_value("c").await, Some(0.0));
    }

    #[tokio::test]
    async fn ttl_expires_entry() {
        let cache = InMemoryCache::default();
        cache
            .set("k", "v".to_string(), Some(Duration::from_millis(50)))
            .await;
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(cache.get("k").await.is_none(), "ttl should expire entry");
    }

    #[tokio::test]
    async fn counter_window_start_set_on_first_incr() {
        let cache = InMemoryCache::default();
        assert!(cache.counter_window_start("c").await.is_none());
        let _ = cache.incr("c", 1.0, None).await;
        assert!(cache.counter_window_start("c").await.is_some());
    }

    /// Happy path: a TTL-less counter whose global window has elapsed is
    /// removed atomically (a reset TTL-less counter is semantically absent —
    /// every budget caller reads `counter_value(...).unwrap_or(0.0)`).
    #[tokio::test]
    async fn reset_spend_counter_if_due_removes_elapsed_ttl_less_counter() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("spend:k", 5.0, None).await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        let reset = cache
            .reset_spend_counter_if_due("spend:k", Duration::from_millis(1), Duration::ZERO)
            .await;
        assert!(reset, "window elapsed -> should reset");
        assert_eq!(cache.counter_value("spend:k").await, None);
        assert_eq!(
            cache.counters_len().await,
            0,
            "reset must remove the entry, not leave a never-expiring zero row"
        );
    }

    /// A reset TTL-less counter must be removed from the counters map — not
    /// left as a zero row the sweep can never reclaim (asymmetric with the
    /// Redis `delete` path).
    #[tokio::test]
    async fn reset_spend_counter_if_due_removes_entry_not_zero_row() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("spend:ttlless", 7.0, None).await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        let reset = cache
            .reset_spend_counter_if_due("spend:ttlless", Duration::from_millis(1), Duration::ZERO)
            .await;
        assert!(reset);
        assert_eq!(cache.counter_value("spend:ttlless").await, None);
        assert_eq!(cache.counters_len().await, 0);
    }

    /// Per-key TTL outlives the global window: the counter must NOT be reset
    /// (the CRITICAL scheduler regression, pinned at the cache layer).
    #[tokio::test]
    async fn reset_spend_counter_if_due_respects_longer_per_key_ttl() {
        let cache = InMemoryCache::default();
        let _ = cache
            .incr("spend:long", 5.0, Some(Duration::from_secs(3600)))
            .await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        // Global window (1ms) has elapsed, but the counter's own TTL (3600s)
        // extends well past ttl_grace (60s) -> must be left alone.
        let reset = cache
            .reset_spend_counter_if_due(
                "spend:long",
                Duration::from_millis(1),
                Duration::from_secs(60),
            )
            .await;
        assert!(!reset, "counter with live per-key TTL must not be reset");
        assert_eq!(
            cache.counter_value("spend:long").await,
            Some(5.0),
            "longer per-key window must not be truncated"
        );
    }

    /// A counter whose global window has not elapsed is never reset.
    #[tokio::test]
    async fn reset_spend_counter_if_due_skips_unelapsed_window() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("spend:k", 5.0, None).await;
        let reset = cache
            .reset_spend_counter_if_due("spend:k", Duration::from_secs(3600), Duration::ZERO)
            .await;
        assert!(!reset);
        assert_eq!(cache.counter_value("spend:k").await, Some(5.0));
    }

    /// A missing counter is a no-op.
    #[tokio::test]
    async fn reset_spend_counter_if_due_handles_missing_key() {
        let cache = InMemoryCache::default();
        let reset = cache
            .reset_spend_counter_if_due("spend:ghost", Duration::from_millis(1), Duration::ZERO)
            .await;
        assert!(!reset);
    }

    /// An absurd `ttl_grace` (e.g. `Duration::from_secs(u64::MAX)`) must not
    /// panic `Instant + Duration`. The grace is clamped to `MAX_TTL_SECS`
    /// like every other caller-derived TTL on this path. The counter has a
    /// short per-key TTL (well inside the clamped grace), so the reset still
    /// proceeds.
    #[tokio::test]
    async fn reset_spend_counter_if_due_survives_absurd_ttl_grace() {
        let cache = InMemoryCache::default();
        let _ = cache
            .incr("spend:k", 5.0, Some(Duration::from_secs(3600)))
            .await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        let reset = cache
            .reset_spend_counter_if_due(
                "spend:k",
                Duration::from_millis(1),
                Duration::from_secs(u64::MAX),
            )
            .await;
        assert!(
            reset,
            "absurd ttl_grace must be clamped, not panic; window elapsed -> reset"
        );
        assert_eq!(cache.counter_value("spend:k").await, None);
    }

    /// Atomicity: a reservation that lands strictly after the reset returns
    /// must survive (the TOCTOU the atomic API closes).
    #[tokio::test]
    async fn reset_spend_counter_if_due_preserves_post_reset_reservation() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("spend:k", 5.0, None).await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        let reset = cache
            .reset_spend_counter_if_due("spend:k", Duration::from_millis(1), Duration::ZERO)
            .await;
        assert!(reset);
        // Strictly after the reset — must not be zeroed.
        let _ = cache.incr("spend:k", 1.0, None).await;
        let _ = cache.incr("spend:k", 2.0, None).await;
        assert_eq!(
            cache.counter_value("spend:k").await,
            Some(3.0),
            "post-reset reservations must survive"
        );
    }

    #[tokio::test]
    async fn incr_if_below_allowed_when_under_ceiling() {
        let cache = InMemoryCache::default();
        let res = cache
            .incr_if_below("rpm:k", 1.0, 3.0, Some(Duration::from_secs(60)))
            .await
            .expect("ok");
        assert_eq!(res, IncrResult::Allowed(1.0));
        assert_eq!(cache.counter_value("rpm:k").await, Some(1.0));
    }

    #[tokio::test]
    async fn incr_if_below_rejected_when_exceeds_ceiling() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("rpm:k", 3.0, None).await;
        let res = cache
            .incr_if_below("rpm:k", 1.0, 3.0, Some(Duration::from_secs(60)))
            .await
            .expect("ok");
        assert_eq!(res, IncrResult::Rejected(3.0));
        assert_eq!(
            cache.counter_value("rpm:k").await,
            Some(3.0),
            "rejected reservation must not mutate the counter"
        );
    }

    #[tokio::test]
    async fn incr_if_below_at_boundary_allows_exact_ceiling() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("rpm:k", 2.0, None).await;
        // 2 + 1 == 3 == ceiling -> allowed
        let res = cache
            .incr_if_below("rpm:k", 1.0, 3.0, None)
            .await
            .expect("ok");
        assert_eq!(res, IncrResult::Allowed(3.0));
    }

    #[tokio::test]
    async fn counter_ttl_expires_fixed_window() {
        let cache = InMemoryCache::default();
        let _ = cache
            .incr("rpm:k", 5.0, Some(Duration::from_millis(40)))
            .await;
        assert_eq!(cache.counter_value("rpm:k").await, Some(5.0));
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(
            cache.counter_value("rpm:k").await,
            None,
            "counter should expire after the fixed window"
        );
        // A new increment after expiry starts a fresh window at the delta.
        assert!(
            (cache
                .incr("rpm:k", 1.0, Some(Duration::from_secs(60)))
                .await
                .unwrap()
                - 1.0)
                .abs()
                < 1e-9
        );
    }

    #[tokio::test]
    async fn incr_if_below_keys_are_independent() {
        let cache = InMemoryCache::default();
        let a = cache
            .incr_if_below("rpm:a", 1.0, 1.0, None)
            .await
            .expect("ok");
        let b = cache
            .incr_if_below("rpm:b", 1.0, 1.0, None)
            .await
            .expect("ok");
        assert_eq!(a, IncrResult::Allowed(1.0));
        assert_eq!(b, IncrResult::Allowed(1.0));
    }

    #[tokio::test]
    async fn incr_if_below_sets_ttl_on_store_entry() {
        let cache = InMemoryCache::default();
        cache
            .incr_if_below("rpm:k", 1.0, 3.0, Some(Duration::from_millis(50)))
            .await
            .expect("ok");
        assert_eq!(cache.counter_value("rpm:k").await, Some(1.0));
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(
            cache.get("rpm:k").await.is_none(),
            "store entry should expire per ttl"
        );
    }

    /// Concurrency: 8 callers racing for a ceiling of 3 must admit exactly 3
    /// and reject the rest, with no overcommit. The atomic write-lock makes
    /// this invariant hold where the non-atomic default could transiently
    /// overcommit.
    #[tokio::test]
    async fn incr_if_below_concurrent_never_overcommits() {
        let cache = Arc::new(InMemoryCache::default());
        let ceiling = 3.0f64;
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .incr_if_below("rpm:race", 1.0, ceiling, Some(Duration::from_secs(60)))
                    .await
                    .expect("ok")
            }));
        }
        let mut allowed = 0u32;
        for h in handles {
            if matches!(h.await.expect("join"), IncrResult::Allowed(_)) {
                allowed += 1;
            }
        }
        assert_eq!(
            allowed, 3,
            "exactly `ceiling` callers admitted; got {allowed}"
        );
        assert_eq!(cache.counter_value("rpm:race").await, Some(3.0));
    }

    /// A caller-supplied TTL far beyond `Instant`'s representable range (e.g.
    /// `Duration::from_secs(u64::MAX)` derived from a huge JWT `exp` claim)
    /// must not panic `Instant + Duration`; the window is clamped to
    /// [`MAX_TTL_SECS`] instead.
    #[tokio::test]
    async fn incr_if_below_huge_ttl_does_not_panic_and_clamps() {
        let cache = InMemoryCache::default();
        let res = cache
            .incr_if_below("jti:seen:k", 1.0, 1.0, Some(Duration::from_secs(u64::MAX)))
            .await
            .expect("must not panic");
        assert_eq!(res, IncrResult::Allowed(1.0));
        assert_eq!(cache.counter_value("jti:seen:k").await, Some(1.0));
        let expires_at = cache
            .counter_expires_at("jti:seen:k")
            .await
            .expect("marker present");
        let ttl = expires_at.saturating_duration_since(Instant::now());
        assert!(
            ttl <= Duration::from_secs(MAX_TTL_SECS),
            "TTL must be clamped to MAX_TTL_SECS, got {ttl:?}"
        );
    }

    /// A reject on a previously-absent key must not materialize a zero
    /// counter: it stays absent, matching `DiskCache::incr_if_below` (which
    /// never writes on the reject path). Previously the in-memory impl always
    /// re-inserted the freshly-created `CounterState`, starting a TTL window
    /// ticking for a never-committed key.
    #[tokio::test]
    async fn incr_if_below_reject_on_absent_key_leaves_key_absent() {
        let cache = InMemoryCache::default();
        let res = cache
            .incr_if_below("rpm:ghost", 5.0, 1.0, Some(Duration::from_secs(60)))
            .await
            .expect("ok");
        assert!(matches!(res, IncrResult::Rejected(_)));
        assert_eq!(
            cache.counter_value("rpm:ghost").await,
            None,
            "reject on absent key must not materialize a zero counter"
        );
        assert!(
            cache.counter_window_start("rpm:ghost").await.is_none(),
            "reject on absent key must not start a TTL window"
        );
        assert_eq!(
            cache.counter_keys().await,
            Vec::<String>::new(),
            "reject-only keys must not appear in counter_keys()"
        );
    }

    /// A reject on a previously-present key still preserves its state
    /// (value, window, expiry) — the reject-on-absent fix must not drop live
    /// counters.
    #[tokio::test]
    async fn incr_if_below_reject_on_present_key_preserves_state() {
        let cache = InMemoryCache::default();
        let _ = cache
            .incr("rpm:k", 3.0, Some(Duration::from_secs(60)))
            .await;
        let window_before = cache.counter_window_start("rpm:k").await;
        let res = cache
            .incr_if_below("rpm:k", 10.0, 1.0, Some(Duration::from_secs(60)))
            .await
            .expect("ok");
        assert!(matches!(res, IncrResult::Rejected(_)));
        assert_eq!(
            cache.counter_value("rpm:k").await,
            Some(3.0),
            "present counter must retain its value after a reject"
        );
        assert_eq!(
            cache.counter_window_start("rpm:k").await,
            window_before,
            "present counter must retain its window start after a reject"
        );
    }
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_memory_only_set_and_get() {
        let cache = DualCache::default();
        cache.set("k", "v".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("v"));
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_memory_only_delete() {
        let cache = DualCache::default();
        cache.set("k", "v".to_string(), None).await;
        cache.delete("k").await;
        assert!(cache.get("k").await.is_none());
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_memory_only_incr() {
        let cache = DualCache::default();
        let v1 = cache.incr("c", 1.0, None).await.unwrap();
        assert!((v1 - 1.0).abs() < 1e-9);
        let v2 = cache.incr("c", 2.0, None).await.unwrap();
        assert!((v2 - 3.0).abs() < 1e-9);
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_memory_only_miss() {
        let cache = DualCache::default();
        assert!(cache.get("missing").await.is_none());
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_write_through_to_memory() {
        let cache = DualCache::default();
        cache.set("k", "v".to_string(), None).await;
        assert_eq!(cache.memory().get("k").await.as_deref(), Some("v"));
    }

    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_counter_value_reads_memory_layer() {
        let cache = DualCache::default();
        let _ = cache.incr("rpm:test", 3.0, None).await;
        assert_eq!(cache.counter_value("rpm:test").await, Some(3.0));
    }

    /// `decr_if_exists` must be delegated to the in-memory layer's atomic
    /// implementation, not the trait default's non-atomic
    /// `counter_value`-then-`incr` sequence: a decrement on an absent counter
    /// is a no-op (`Ok(false)`) rather than minting a negative, TTL-less
    /// residue, and a non-finite delta surfaces as `Err` (matching
    /// [`InMemoryCache::decr_if_exists`]). With Redis configured the same
    /// delegation runs against `RedisCache::decr_if_exists`'s atomic Lua
    /// script.
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_decr_if_exists_delegates_atomically() {
        let cache = DualCache::default();
        // Absent counter → no-op, never a negative residue.
        assert!(!cache.decr_if_exists("spend:x", 1.0).await.unwrap());
        assert_eq!(cache.counter_value("spend:x").await, None);
        // Present counter → decremented through the memory layer.
        let _ = cache.incr("spend:x", 1.0, None).await.unwrap();
        assert!(cache.decr_if_exists("spend:x", 0.25).await.unwrap());
        assert_eq!(cache.counter_value("spend:x").await, Some(0.75));
        // Non-finite delta → Err, counter unchanged.
        assert!(cache.decr_if_exists("spend:x", f64::NAN).await.is_err());
        assert_eq!(cache.counter_value("spend:x").await, Some(0.75));
    }

    /// After set, get should hit memory without Redis.
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_get_prefers_memory_after_set() {
        let cache = DualCache::default();
        cache
            .set("litellm:vcr:key", "cached".to_string(), None)
            .await;
        assert_eq!(
            cache.memory().get("litellm:vcr:key").await.as_deref(),
            Some("cached")
        );
        assert_eq!(
            cache.get("litellm:vcr:key").await.as_deref(),
            Some("cached")
        );
    }

    /// Without Redis configured, `publish_invalidation` is a no-op: the local
    /// entry must still be cleared explicitly by the caller. This test pins
    /// the contract that callers cannot rely on `publish_invalidation` to
    /// clear the local cache — only `delete` does that.
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_publish_invalidation_noop_without_redis() {
        let cache = DualCache::default();
        cache.set("vk:k", "v".to_string(), None).await;
        cache.publish_invalidation("vk:k").await;
        assert_eq!(
            cache.get("vk:k").await.as_deref(),
            Some("v"),
            "publish_invalidation must not clear local cache; only delete does"
        );
        cache.delete("vk:k").await;
        assert!(cache.get("vk:k").await.is_none());
    }

    /// `redis_cache()` returns `Some` only when Redis is configured.
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn dual_cache_redis_cache_accessor_none_by_default() {
        let cache = DualCache::default();
        assert!(cache.redis_cache().is_none());
    }

    // ---- U16: read path must never re-publish memory → Redis ----------------
    //
    // After a key update deleted the Redis blob,
    // the read-path writer re-published the stale in-memory blob back to
    // Redis, un-doing the cross-replica invalidation. oxllm's `DualCache::get`
    // only backfills **Redis → memory** (never the reverse), so a stale memory
    // entry can never be re-published to Redis. These tests pin that contract.

    /// After `delete` (the invalidation path), a subsequent `get` must NOT
    /// re-materialize the value — confirming the read path does not write.
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn u16_get_after_delete_does_not_rematerialize() {
        let cache = DualCache::default();
        cache.set("vk:stale", "old-payload".to_string(), None).await;
        // Simulate the invalidation a key update triggers.
        cache.delete("vk:stale").await;
        // The read path must return None — it must never re-publish or
        // re-materialize the deleted value from any layer.
        assert_eq!(
            cache.get("vk:stale").await,
            None,
            "U16: read path must not re-materialize a deleted entry"
        );
    }

    /// `get` with no Redis configured must not perform any write — verifying
    /// the read path is side-effect-free with respect to the shared store.
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn u16_read_path_is_write_free_without_redis() {
        let cache = DualCache::default();
        // A value only in Redis would exist if get re-published; with no Redis
        // a miss must stay a miss (no write-back from a non-existent layer).
        assert!(cache.get("vk:never-set").await.is_none());
        // Confirm no memory entry was created by the read.
        assert!(
            cache.memory().get("vk:never-set").await.is_none(),
            "U16: a cache miss must not create a memory entry"
        );
    }

    /// The counters map must not grow without bound: expired entries are
    /// evicted by the periodic sweep on `incr`/`incr_if_below`. Regression for
    /// the unbounded-growth finding — TTL alone only made expired counters read
    /// as absent, it never removed the entry from the map.
    #[tokio::test]
    async fn counter_sweep_evicts_expired_entries() {
        let cache = InMemoryCache::default();
        // Insert more than the sweep interval's worth of counters with a short
        // TTL so the map is populated past a sweep boundary.
        for i in 0..(COUNTER_SWEEP_INTERVAL + 20) {
            let _ = cache
                .incr(&format!("c:{i}"), 1.0, Some(Duration::from_millis(20)))
                .await;
        }
        assert_eq!(
            cache.counters_len().await,
            COUNTER_SWEEP_INTERVAL as usize + 20
        );
        // Let the short TTLs elapse.
        tokio::time::sleep(Duration::from_millis(80)).await;
        // More writes cross the next sweep boundary; the expired entries must
        // be evicted rather than accumulate alongside the new ones.
        for i in 0..(COUNTER_SWEEP_INTERVAL + 1) {
            let _ = cache
                .incr(&format!("new:{i}"), 1.0, Some(Duration::from_secs(60)))
                .await;
        }
        let len = cache.counters_len().await;
        assert!(
            len <= COUNTER_SWEEP_INTERVAL as usize + 2,
            "expired counters must be swept, not accumulated (map has {len} entries)"
        );
    }

    /// `sweep_now` deterministically evicts expired entries, pinning the exact
    /// eviction behaviour without depending on the call counter.
    #[tokio::test]
    async fn counter_sweep_now_removes_expired_entries() {
        let cache = InMemoryCache::default();
        let _ = cache
            .incr("expired", 1.0, Some(Duration::from_millis(20)))
            .await;
        let _ = cache.incr("persistent", 1.0, None).await;
        tokio::time::sleep(Duration::from_millis(80)).await;
        cache.sweep_now().await;
        assert_eq!(
            cache.counters_len().await,
            1,
            "only the live counter should remain"
        );
        assert_eq!(cache.counter_value("persistent").await, Some(1.0));
    }

    /// TTL-less counters are never reclaimed by the expiry sweep, so the map
    /// could grow without bound on a future hook that increments unbounded
    /// distinct `ttl=None` keys. The soft-cap second stage evicts the oldest
    /// TTL-less entries once the map crosses [`Self::counter_soft_cap`].
    #[tokio::test]
    async fn counter_sweep_evicts_oldest_ttl_less_past_soft_cap() {
        // Use a tiny soft cap so the test does not need to insert ~100k
        // entries.
        let soft_cap = 10usize;
        let cache = InMemoryCache::with_counter_soft_cap(10_000, soft_cap);
        // Insert well past the cap, all TTL-less.
        for i in 0..(soft_cap * 4) {
            let _ = cache.incr(&format!("spend:{i}"), 1.0, None).await;
        }
        // Force a deterministic sweep so the post-sweep bound is exact (between
        // sweeps the map may transiently exceed the cap by up to
        // COUNTER_SWEEP_INTERVAL entries).
        cache.sweep_now().await;
        let len = cache.counters_len().await;
        assert!(
            len <= soft_cap,
            "TTL-less counters must be bounded by the soft cap; \
             cap={soft_cap}, actual={len}"
        );
    }

    /// MEDIUM: `set` must skip (and not store) values larger than the
    /// per-item cap, mirroring the reference `check_value_size`
    /// (`MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB`, default 1 MiB). Moka's
    /// `max_capacity` bounds entry *count*, not total bytes, so without this
    /// cap a single cached response body (up to 50 MiB on the response side)
    /// could consume disproportionate memory.
    #[tokio::test]
    async fn set_skips_oversized_value() {
        let cache = InMemoryCache::default();
        // A value exactly at the cap is accepted; one byte over is skipped.
        let at_cap = "x".repeat(MAX_VALUE_BYTES);
        cache.set("at_cap", at_cap.clone(), None).await;
        assert_eq!(
            cache.get("at_cap").await.as_deref(),
            Some(at_cap.as_str()),
            "a value at the cap must be stored"
        );
        let over_cap = "y".repeat(MAX_VALUE_BYTES + 1);
        cache.set("over_cap", over_cap, None).await;
        assert!(
            cache.get("over_cap").await.is_none(),
            "a value over the cap must be skipped"
        );
    }

    /// LOW: `incr_if_below` must reject non-finite delta/ceiling with `Err`
    /// (matching the trait default), not return `Ok(Rejected(...))`. A NaN
    /// delta would compute `NaN <= ceiling == false` → a silent `Rejected`; a
    /// NaN ceiling would reject all traffic. Callers that check `is_err()` to
    /// detect invalid input must see the error.
    #[tokio::test]
    async fn incr_if_below_rejects_non_finite_delta_returns_err() {
        let cache = InMemoryCache::default();
        let _ = cache.incr("rpm:k", 1.0, None).await.unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let res = cache.incr_if_below("rpm:k", bad, 3.0, None).await;
            assert!(
                res.is_err(),
                "in-memory incr_if_below must reject non-finite delta ({bad:?}), got {res:?}"
            );
        }
        let res = cache.incr_if_below("rpm:k", 1.0, f64::NAN, None).await;
        assert!(
            res.is_err(),
            "in-memory incr_if_below must reject non-finite ceiling, got {res:?}"
        );
        // The counter must be unchanged after the rejected writes.
        assert_eq!(
            cache.counter_value("rpm:k").await,
            Some(1.0),
            "rejected non-finite inputs must not mutate the counter"
        );
    }
}
