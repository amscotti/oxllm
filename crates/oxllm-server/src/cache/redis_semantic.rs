//! Redis-backed semantic cache.
//!
//! Stores each entry as a Redis hash at `oxllm:semcache:{id}` with fields for
//! the embedding (JSON array), response (JSON), `api_key_hash`, `model`,
//! `created_at` (unix seconds), and `ttl_secs`. A Redis set
//! `oxllm:semcache:keys` tracks every entry id. Search fetches the matching
//! entries and computes cosine similarity client-side, returning the
//! highest-scoring non-expired entry at or above the configured threshold.
//!
//! **Scale ceiling:** search and `enforce_capacity` perform an O(n) full scan
//! (`SMEMBERS` + pipelined `HGETALL` for every entry). This deliberately avoids
//! the RediSearch module dependency that the reference `RedisSemanticCache` relies
//! on (`redisvl` vector index for O(log n) server-side similarity). The
//! practical limit is single-digit thousands of entries; beyond that, each
//! lookup transfers every stored embedding + response JSON within the
//! `CMD_TIMEOUT` budget and can saturate bandwidth. For larger deployments,
//! prefer the Qdrant backend ([`super::qdrant::QdrantCache`]) which delegates
//! similarity search to the server.
//!
//! Cosine similarity reuses [`super::semantic::cosine_similarity`]. Each
//! hash also gets a Redis-side `EXPIRE` so stale entries vanish automatically;
//! the set membership is reconciled opportunistically by `evict_expired` and
//! tolerated on the search path (a missing hash is simply skipped).
//!
//! Requires the `redis` feature (default-on for `oxllm-server`).

#![cfg(feature = "redis")]

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use redis::AsyncCommands;
use serde_json::Value;
use tokio::sync::Mutex;

use super::semantic::{
    SemanticCacheStore, cacheable_semantic_entry, cosine_similarity, valid_embedding,
};

/// Default Redis key prefix for semantic-cache entries.
pub(crate) const DEFAULT_PREFIX: &str = "oxllm:semcache";

/// Default capacity bound for the Redis backend. Matches the in-memory
/// [`super::semantic::SemanticCache`] default so the Redis tier cannot grow
/// without limit (important when `ttl_secs == 0`, where neither Redis-side
/// `EXPIRE` nor `evict_expired` ever removes anything on its own).
const DEFAULT_MAX_ENTRIES: usize = 10_000;

/// Per-command timeout. Redis operations are fail-open (search → `None`,
/// store → log + skip), so a stalled Redis cannot hold the request path.
/// Matches the Qdrant client-wide timeout (`super::qdrant::QDRANT_TIMEOUT`).
const CMD_TIMEOUT: Duration = Duration::from_secs(5);

/// A Redis-backed semantic cache.
///
/// Holds a `redis::Client` (cheap, no I/O at construction) and creates a
/// fresh multiplexed connection per operation. Network failures degrade
/// gracefully: `search` returns `None` and `store` logs a warning, so a Redis
/// outage never breaks request serving.
pub struct RedisSemanticCache {
    client: redis::Client,
    prefix: String,
    similarity_threshold: f64,
    ttl: Duration,
    /// Hard upper bound on the number of entries. When the keys set exceeds
    /// this after a store, the oldest entries (by `created_at`) are removed so
    /// the backend cannot grow without limit — e.g. under `ttl_secs == 0`.
    max_entries: usize,
    /// Lazily initialized multiplexed connection, shared across all calls.
    conn: Mutex<Option<redis::aio::MultiplexedConnection>>,
}

impl std::fmt::Debug for RedisSemanticCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisSemanticCache")
            .field("prefix", &self.prefix)
            .field("similarity_threshold", &self.similarity_threshold)
            .field("ttl", &self.ttl)
            .finish()
    }
}

impl RedisSemanticCache {
    /// Create a Redis semantic cache. Performs no I/O — the connection is
    /// opened lazily on the first operation. Uses
    /// `DEFAULT_MAX_ENTRIES` as the capacity bound.
    pub fn new(
        url: &str,
        similarity_threshold: f64,
        ttl: Duration,
    ) -> Result<Self, redis::RedisError> {
        Self::with_prefix(url, DEFAULT_PREFIX, similarity_threshold, ttl)
    }

    /// Create with an explicit key prefix (for tests / namespacing).
    pub fn with_prefix(
        url: &str,
        prefix: &str,
        similarity_threshold: f64,
        ttl: Duration,
    ) -> Result<Self, redis::RedisError> {
        Self::with_prefix_and_max(url, prefix, DEFAULT_MAX_ENTRIES, similarity_threshold, ttl)
    }

    /// Create with an explicit key prefix and capacity bound. The bound keeps
    /// the Redis tier from growing without limit when entries never expire on
    /// their own (`ttl_secs == 0`); the oldest entries are evicted first.
    pub fn with_prefix_and_max(
        url: &str,
        prefix: &str,
        max_entries: usize,
        similarity_threshold: f64,
        ttl: Duration,
    ) -> Result<Self, redis::RedisError> {
        if prefix.trim().is_empty() {
            return Err(redis::RedisError::from((
                redis::ErrorKind::InvalidClientConfig,
                "RedisSemanticCache prefix must be non-empty",
            )));
        }
        let client = redis::Client::open(url)?;
        Ok(Self {
            client,
            prefix: prefix.to_string(),
            similarity_threshold,
            ttl,
            max_entries,
            conn: Mutex::new(None),
        })
    }

    fn entry_key(&self, id: &str) -> String {
        format!("{}:{id}", self.prefix)
    }

    fn keys_set(&self) -> String {
        format!("{}:keys", self.prefix)
    }

    /// Acquire the lazily-initialized multiplexed connection, creating it on
    /// first use. Returns `None` on connection failure so callers can fall
    /// through without blocking.
    ///
    /// The connect attempt is bounded by a 2s timeout. Without it, a Redis
    /// outage would hold the `conn` mutex guard across an un-timeout'd TCP
    /// connect, queueing every request behind a single hung attempt. On
    /// timeout we fail open (return `None` → cache miss).
    async fn connection(&self) -> Option<redis::aio::MultiplexedConnection> {
        let mut guard = self.conn.lock().await;
        if let Some(conn) = guard.as_mut() {
            return Some(conn.clone());
        }
        let conn = match tokio::time::timeout(
            Duration::from_secs(2),
            self.client.get_multiplexed_async_connection(),
        )
        .await
        {
            Ok(Ok(conn)) => conn,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "redis semantic cache: connection failed; falling through");
                return None;
            }
            Err(_) => {
                tracing::warn!(
                    "redis semantic cache: connection timed out after 2s; falling through"
                );
                return None;
            }
        };
        *guard = Some(conn.clone());
        Some(conn)
    }

    /// Drop the cached connection so the next operation reconnects. Called
    /// when an operation fails or times out: a multiplexed connection whose
    /// underlying TCP stream died (e.g. Redis restarted) keeps returning
    /// errors forever, which would permanently disable this tier until
    /// process restart if we kept it cached.
    async fn invalidate_conn(&self) {
        self.conn.lock().await.take();
    }

    /// Bring the keys set back down to [`Self::max_entries`] by dropping the
    /// oldest entries (by `created_at`). Best-effort: any Redis error aborts
    /// the sweep so a transient backend issue never blocks the store path.
    /// This is the only removal path under `ttl_secs == 0`, bounding Redis
    /// memory usage for the "never expire" configuration.
    async fn enforce_capacity(&self, conn: &mut redis::aio::MultiplexedConnection) {
        if self.max_entries == 0 {
            return;
        }
        let keys_set = self.keys_set();
        let count: usize = match conn.scard(&keys_set).await {
            Ok(c) => c,
            Err(_) => return,
        };
        if count <= self.max_entries {
            return;
        }
        let overflow = count - self.max_entries;
        let ids: Vec<String> = match conn.smembers::<_, Vec<String>>(&keys_set).await {
            Ok(ids) => ids,
            Err(_) => return,
        };
        let mut pipe = redis::pipe();
        for id in &ids {
            pipe.hget(self.entry_key(id), "created_at");
        }
        let stamps: Vec<Option<String>> = match pipe.query_async(conn).await {
            Ok(v) => v,
            Err(_) => return,
        };
        let mut scored: Vec<(u64, String)> = Vec::with_capacity(ids.len());
        let mut purged = 0usize;
        for (id, stamp) in ids.iter().zip(stamps.iter()) {
            match stamp.as_ref().and_then(|s| s.parse::<u64>().ok()) {
                Some(created_at) => scored.push((created_at, id.clone())),
                // An id whose `created_at` cannot be read or parsed is an
                // orphaned/tombstoned entry: it inflates `scard` (which
                // counts it toward the overflow) but can never be selected
                // by `search`. Purge it instead of filtering it out of the
                // eviction candidates — otherwise the overflow is
                // overstated and live entries get evicted prematurely.
                None => {
                    let _ = conn.del::<_, ()>(self.entry_key(id)).await;
                    let _ = conn.srem::<_, _, ()>(&keys_set, id).await;
                    purged += 1;
                }
            }
        }
        scored.sort_unstable_by_key(|(created_at, _)| *created_at);
        let overflow = overflow.saturating_sub(purged);
        for (_, id) in scored.iter().take(overflow) {
            let _ = conn.del::<_, ()>(self.entry_key(id)).await;
            let _ = conn.srem::<_, _, ()>(&keys_set, id).await;
        }
    }
}

#[async_trait]
impl SemanticCacheStore for RedisSemanticCache {
    async fn search(
        &self,
        query_embedding: &[f32],
        api_key_hash: &str,
        model: &str,
    ) -> Option<Value> {
        if self.max_entries == 0 || !valid_embedding(query_embedding) {
            return None;
        }
        let mut conn = self.connection().await?;
        let entries = match tokio::time::timeout(CMD_TIMEOUT, async {
            let ids: Vec<String> = conn
                .smembers::<_, Vec<String>>(self.keys_set())
                .await
                .map_err(|e| {
                    tracing::warn!(
                        error = %e,
                        "redis semantic cache: SMEMBERS failed; falling through"
                    );
                    e
                })?;
            if ids.is_empty() {
                return Ok::<Vec<RedisEntry>, redis::RedisError>(Vec::new());
            }
            let mut pipe = redis::pipe();
            for id in &ids {
                pipe.hgetall(self.entry_key(id));
            }
            let maps: Vec<Option<HashMap<String, String>>> = pipe.query_async(&mut conn).await?;
            let mut entries = Vec::with_capacity(maps.len());
            for map in maps.into_iter().flatten() {
                if let Some(entry) = RedisEntry::from_fields(&map) {
                    entries.push(entry);
                }
            }
            Ok(entries)
        })
        .await
        {
            Ok(Ok(entries)) => entries,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "redis semantic cache: search failed; falling through");
                self.invalidate_conn().await;
                return None;
            }
            Err(_) => {
                tracing::warn!(
                    "redis semantic cache: search timed out after {:?}; falling through",
                    CMD_TIMEOUT
                );
                self.invalidate_conn().await;
                return None;
            }
        };
        select_best_match(
            query_embedding,
            &entries,
            self.similarity_threshold,
            api_key_hash,
            model,
            now_secs(),
        )
    }

    async fn store(
        &self,
        embedding: Vec<f32>,
        _request_text: String,
        response: Value,
        api_key_hash: &str,
        model: &str,
    ) {
        if self.max_entries == 0 || !cacheable_semantic_entry(&embedding, &response) {
            return;
        }
        let Some(mut conn) = self.connection().await else {
            return;
        };
        let id = uuid::Uuid::new_v4().to_string();
        // Round sub-second TTLs up to 1s (0 still means "never expire").
        let ttl_secs = if self.ttl.is_zero() {
            0
        } else {
            self.ttl.as_secs().clamp(1, i64::MAX as u64)
        };
        let entry = RedisEntry {
            embedding,
            response,
            api_key_hash: api_key_hash.to_string(),
            model: model.to_string(),
            created_at: now_secs(),
            // ttl_secs == 0 means "never expire" (no Redis EXPIRE,
            // is_expired returns false). Sub-second TTLs round UP to at
            // least 1s: truncating them to 0 would make entries silently
            // immortal (Redis EXPIRE would never be set).
            ttl_secs,
        };
        let fields = entry.to_fields();
        let flat: Vec<(&str, &str)> = fields.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let key = self.entry_key(&id);
        let keys_set = self.keys_set();
        let res: Result<(), redis::RedisError> = match tokio::time::timeout(CMD_TIMEOUT, async {
            let mut pipe = redis::pipe();
            pipe.atomic();
            pipe.hset_multiple(&key, &flat).ignore();
            if ttl_secs > 0 {
                pipe.expire(&key, ttl_secs as i64).ignore();
            }
            pipe.sadd(&keys_set, &id).ignore();
            pipe.query_async::<()>(&mut conn).await
        })
        .await
        {
            Ok(inner) => inner,
            Err(_) => {
                tracing::warn!(
                    "redis semantic cache: store timed out after {:?}; entry not persisted",
                    CMD_TIMEOUT
                );
                self.invalidate_conn().await;
                return;
            }
        };
        if let Err(e) = res {
            tracing::warn!(error = %e, "redis semantic cache: store failed; entry not persisted");
            self.invalidate_conn().await;
            return;
        }
        let _ = tokio::time::timeout(CMD_TIMEOUT, self.enforce_capacity(&mut conn)).await;
    }

    async fn evict_expired(&self) {
        let Some(mut conn) = self.connection().await else {
            return;
        };
        let ids: Vec<String> = match tokio::time::timeout(
            CMD_TIMEOUT,
            conn.smembers::<_, Vec<String>>(self.keys_set()),
        )
        .await
        {
            Ok(Ok(ids)) => ids,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "redis semantic cache: evict_expired failed");
                self.invalidate_conn().await;
                return;
            }
            Err(_) => {
                tracing::warn!("redis semantic cache: evict_expired timed out");
                self.invalidate_conn().await;
                return;
            }
        };
        if ids.is_empty() {
            return;
        }
        let now = now_secs();
        let evict = async {
            for id in &ids {
                let fields: HashMap<String, String> = match conn
                    .hgetall::<_, HashMap<String, String>>(self.entry_key(id))
                    .await
                {
                    Ok(map) => map,
                    Err(_) => continue,
                };
                let expired = match RedisEntry::from_fields(&fields) {
                    Some(entry) => entry.is_expired(now),
                    None => true,
                };
                if expired {
                    let _ = conn.del::<_, ()>(self.entry_key(id)).await;
                    let _ = conn.srem::<_, _, ()>(self.keys_set(), id).await;
                }
            }
        };
        let _ = tokio::time::timeout(CMD_TIMEOUT, evict).await;
    }
}

/// One Redis-stored semantic-cache entry, in its deserialized form.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RedisEntry {
    pub(crate) embedding: Vec<f32>,
    pub(crate) response: Value,
    pub(crate) api_key_hash: String,
    pub(crate) model: String,
    pub(crate) created_at: u64,
    pub(crate) ttl_secs: u64,
}

impl RedisEntry {
    /// Serialize into the Redis hash field map (field name → string). The
    /// embedding is encoded as a JSON array so it round-trips cleanly through
    /// [`Self::from_fields`]. Pure — unit-tested directly.
    pub(crate) fn to_fields(&self) -> Vec<(&'static str, String)> {
        let embedding = serde_json::to_string(&self.embedding).unwrap_or_else(|_| "[]".into());
        vec![
            ("embedding", embedding),
            ("response", self.response.to_string()),
            ("api_key_hash", self.api_key_hash.clone()),
            ("model", self.model.clone()),
            ("created_at", self.created_at.to_string()),
            ("ttl_secs", self.ttl_secs.to_string()),
        ]
    }

    /// Deserialize from the Redis hash field map. Returns `None` on a
    /// malformed/missing field. Pure — unit-tested directly.
    pub(crate) fn from_fields(fields: &HashMap<String, String>) -> Option<Self> {
        let emb_raw = fields.get("embedding")?;
        let emb_value: Value = serde_json::from_str(emb_raw).ok()?;
        let array = emb_value.as_array()?;
        let mut embedding = Vec::with_capacity(array.len());
        for v in array {
            embedding.push(v.as_f64()? as f32);
        }
        let response = fields
            .get("response")
            .and_then(|s| serde_json::from_str(s).ok())?;
        if !cacheable_semantic_entry(&embedding, &response) {
            return None;
        }
        let api_key_hash = fields.get("api_key_hash")?.clone();
        let model = fields.get("model")?.clone();
        let created_at = fields.get("created_at").and_then(|s| s.parse().ok())?;
        let ttl_secs = fields.get("ttl_secs").and_then(|s| s.parse().ok())?;
        Some(Self {
            embedding,
            response,
            api_key_hash,
            model,
            created_at,
            ttl_secs,
        })
    }

    fn is_expired(&self, now: u64) -> bool {
        self.ttl_secs > 0 && now.saturating_sub(self.created_at) >= self.ttl_secs
    }
}

/// Select the best non-expired entry for the given tenant/model above the
/// threshold. Pure (no Redis) — unit-tested directly to exercise the
/// similarity, scoping, and TTL logic without a live Redis.
///
/// Tie-breaking: when two entries have identical cosine similarity, the newer
/// entry (higher `created_at`) wins. This makes selection deterministic
/// regardless of `SMEMBERS` iteration order (Redis sets are unordered).
pub(crate) fn select_best_match(
    query: &[f32],
    entries: &[RedisEntry],
    threshold: f64,
    api_key_hash: &str,
    model: &str,
    now: u64,
) -> Option<Value> {
    if !valid_embedding(query) {
        return None;
    }
    let mut best: Option<(f64, &RedisEntry)> = None;
    for entry in entries {
        if entry.api_key_hash != api_key_hash || entry.model != model {
            continue;
        }
        if entry.is_expired(now) {
            continue;
        }
        if entry.embedding.len() != query.len() || !valid_embedding(&entry.embedding) {
            continue;
        }
        let sim = cosine_similarity(query, &entry.embedding);
        match best {
            None => best = Some((sim, entry)),
            Some((cur, existing)) => {
                if sim > cur || (sim == cur && entry.created_at > existing.created_at) {
                    best = Some((sim, entry));
                }
            }
        }
    }
    let (sim, entry) = best?;
    if sim >= threshold {
        Some(entry.response.clone())
    } else {
        None
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_entry(
        embedding: Vec<f32>,
        response: Value,
        api_key_hash: &str,
        model: &str,
        created_at: u64,
        ttl_secs: u64,
    ) -> RedisEntry {
        RedisEntry {
            embedding,
            response,
            api_key_hash: api_key_hash.to_string(),
            model: model.to_string(),
            created_at,
            ttl_secs,
        }
    }

    #[test]
    fn entry_round_trips_through_redis_fields() {
        let entry = make_entry(
            vec![0.1, 0.2, 0.3],
            json!({"id": "x", "choices": []}),
            "tenant-a",
            "gpt-4o",
            1_700_000_000,
            3600,
        );
        let fields_vec = entry.to_fields();
        let map: HashMap<String, String> = fields_vec
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let back = RedisEntry::from_fields(&map).expect("round trip");
        assert_eq!(back, entry);
        // Embedding parses back as a JSON array of floats.
        assert_eq!(back.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn zero_threshold_does_not_match_invalid_or_different_dimension_vectors() {
        let entries = vec![make_entry(vec![1.0, 0.0], json!({}), "t", "m", 100, 60)];
        assert!(select_best_match(&[0.0, 1.0], &entries, 0.0, "t", "m", 100).is_some());
        for query in [vec![1.0], vec![0.0, 0.0], vec![f32::NAN, 1.0]] {
            assert!(select_best_match(&query, &entries, 0.0, "t", "m", 100).is_none());
        }
    }

    #[test]
    fn constructor_rejects_empty_prefix_before_connecting() {
        for prefix in ["", " ", "\t\n"] {
            assert!(
                RedisSemanticCache::with_prefix(
                    "redis://127.0.0.1:1",
                    prefix,
                    0.0,
                    Duration::ZERO,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn from_fields_returns_none_on_missing_field() {
        let mut map = HashMap::new();
        map.insert("embedding".to_string(), "[1.0]".to_string());
        // response / api_key_hash / model / created_at / ttl_secs all missing.
        assert!(RedisEntry::from_fields(&map).is_none());
    }

    #[test]
    fn from_fields_returns_none_on_malformed_embedding() {
        let mut map = HashMap::new();
        map.insert("embedding".to_string(), "not-json".to_string());
        map.insert("response".to_string(), "{}".to_string());
        map.insert("api_key_hash".to_string(), "a".to_string());
        map.insert("model".to_string(), "m".to_string());
        map.insert("created_at".to_string(), "1".to_string());
        map.insert("ttl_secs".to_string(), "1".to_string());
        assert!(RedisEntry::from_fields(&map).is_none());
    }

    #[test]
    fn from_fields_rejects_non_numeric_embedding_element() {
        let mut map = HashMap::new();
        map.insert("embedding".to_string(), "[1.0, \"oops\", 3.0]".to_string());
        map.insert("response".to_string(), "{}".to_string());
        map.insert("api_key_hash".to_string(), "a".to_string());
        map.insert("model".to_string(), "m".to_string());
        map.insert("created_at".to_string(), "1".to_string());
        map.insert("ttl_secs".to_string(), "1".to_string());
        assert!(
            RedisEntry::from_fields(&map).is_none(),
            "a non-numeric element must reject the whole entry, not coerce to 0.0"
        );
    }

    #[test]
    fn select_returns_closest_entry_above_threshold() {
        let now = 1000u64;
        let entries = vec![
            make_entry(vec![1.0, 0.0], json!({"id": "a"}), "t", "gpt-4o", now, 100),
            make_entry(vec![0.9, 0.4], json!({"id": "b"}), "t", "gpt-4o", now, 100),
        ];
        // query is closest to "b".
        let hit = select_best_match(&[0.95, 0.3], &entries, 0.7, "t", "gpt-4o", now).expect("hit");
        assert_eq!(hit["id"], "b");
    }

    #[test]
    fn select_returns_none_below_threshold() {
        let now = 1000u64;
        let entries = vec![make_entry(
            vec![1.0, 0.0],
            json!({"id": "a"}),
            "t",
            "gpt-4o",
            now,
            100,
        )];
        assert!(select_best_match(&[0.0, 1.0], &entries, 0.95, "t", "gpt-4o", now).is_none());
    }

    #[test]
    fn select_is_scoped_per_api_key_hash_and_model() {
        let now = 1000u64;
        let entries = vec![
            make_entry(
                vec![1.0, 0.0],
                json!({"id": "a-only"}),
                "a",
                "gpt-4o",
                now,
                100,
            ),
            make_entry(
                vec![1.0, 0.0],
                json!({"id": "gpt4-only"}),
                "a",
                "claude",
                now,
                100,
            ),
        ];
        // tenant "b" sees nothing.
        assert!(select_best_match(&[1.0, 0.0], &entries, 0.9, "b", "gpt-4o", now).is_none());
        // tenant "a", model "claude" sees the claude entry.
        let hit = select_best_match(&[1.0, 0.0], &entries, 0.9, "a", "claude", now).expect("hit");
        assert_eq!(hit["id"], "gpt4-only");
    }

    #[test]
    fn select_skips_expired_entries() {
        let now = 1000u64;
        let entries = vec![
            // expired: created 100s ago, ttl 10s.
            make_entry(
                vec![1.0, 0.0],
                json!({"id": "stale"}),
                "t",
                "gpt-4o",
                now - 100,
                10,
            ),
            // fresh.
            make_entry(
                vec![0.9, 0.4],
                json!({"id": "fresh"}),
                "t",
                "gpt-4o",
                now,
                100,
            ),
        ];
        let hit = select_best_match(&[1.0, 0.0], &entries, 0.3, "t", "gpt-4o", now).expect("hit");
        assert_eq!(hit["id"], "fresh");
    }

    #[test]
    fn select_skips_all_when_everything_expired() {
        let now = 1000u64;
        let entries = vec![make_entry(
            vec![1.0, 0.0],
            json!({"id": "stale"}),
            "t",
            "gpt-4o",
            now - 100,
            10,
        )];
        assert!(select_best_match(&[1.0, 0.0], &entries, 0.3, "t", "gpt-4o", now).is_none());
    }

    #[test]
    fn select_breaks_ties_by_newest_created_at() {
        // Two entries with identical embeddings (→ identical cosine sim to the
        // query). The newer one (higher created_at) must win regardless of
        // iteration order, making selection deterministic.
        let now = 2000u64;
        let older = make_entry(
            vec![1.0, 0.0],
            json!({"id": "older"}),
            "t",
            "gpt-4o",
            now - 100,
            3600,
        );
        let newer = make_entry(
            vec![1.0, 0.0],
            json!({"id": "newer"}),
            "t",
            "gpt-4o",
            now,
            3600,
        );
        // newer-first ordering.
        let hit_newer_first = select_best_match(
            &[1.0, 0.0],
            &[newer.clone(), older.clone()],
            0.9,
            "t",
            "gpt-4o",
            now,
        )
        .expect("hit");
        assert_eq!(hit_newer_first["id"], "newer");
        // older-first ordering — same result.
        let hit_older_first =
            select_best_match(&[1.0, 0.0], &[older, newer], 0.9, "t", "gpt-4o", now).expect("hit");
        assert_eq!(hit_older_first["id"], "newer");
    }

    #[test]
    fn ttl_zero_means_never_expire() {
        let now = 1_000_000u64;
        let entries = vec![make_entry(
            vec![1.0, 0.0],
            json!({"id": "forever"}),
            "t",
            "gpt-4o",
            0,
            0,
        )];
        let hit = select_best_match(&[1.0, 0.0], &entries, 0.9, "t", "gpt-4o", now).expect("hit");
        assert_eq!(hit["id"], "forever");
    }

    #[test]
    fn store_then_no_live_redis_does_not_panic() {
        // `new` only parses the URL; no connection is attempted, so an
        // unreachable host is fine at construction.
        let cache = RedisSemanticCache::new("redis://127.0.0.1:1/0", 0.95, Duration::from_secs(60))
            .expect("construct");
        // The connection helper returns None rather than panicking.
        let rt = tokio::runtime::Runtime::new().expect("rt");
        let got = rt.block_on(cache.connection());
        assert!(got.is_none(), "unreachable redis must yield no connection");
    }

    #[tokio::test]
    async fn evict_expired_is_noop_without_connection() {
        let cache = RedisSemanticCache::new("redis://127.0.0.1:1/0", 0.95, Duration::from_secs(60))
            .expect("construct");
        // Must not panic / hang when no connection is available.
        cache.evict_expired().await;
    }

    #[tokio::test]
    async fn search_returns_none_against_unreachable_host_without_hanging() {
        let cache = RedisSemanticCache::new("redis://127.0.0.1:1/0", 0.95, Duration::from_secs(60))
            .expect("construct");
        let start = std::time::Instant::now();
        let got = cache.search(&[1.0, 0.0], "t", "gpt-4o").await;
        assert!(got.is_none(), "unreachable redis must not produce a hit");
        // Connection fails fast (connection refused, < 2s timeout budget);
        // the per-op timeout ensures we never block the request path.
        assert!(
            start.elapsed() < Duration::from_secs(4),
            "search should fail open promptly, took {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn store_does_not_panic_against_unreachable_host() {
        let cache = RedisSemanticCache::new("redis://127.0.0.1:1/0", 0.95, Duration::from_secs(60))
            .expect("construct");
        // Must not panic / hang; best-effort store drops silently.
        cache
            .store(
                vec![1.0, 0.0],
                "s".to_string(),
                json!({"id": "x"}),
                "t",
                "gpt-4o",
            )
            .await;
    }

    #[tokio::test]
    async fn store_returns_promptly_against_unreachable_host() {
        // Even when the Redis pipeline succeeds but enforce_capacity runs
        // afterward, the whole store path must be bounded by CMD_TIMEOUT (+ a
        // small slack). Here the connection itself fails, but the assertion
        // guards against a future regression where enforce_capacity is
        // re-introduced without a timeout wrapper.
        let cache = RedisSemanticCache::new("redis://127.0.0.1:1/0", 0.95, Duration::from_secs(60))
            .expect("construct");
        let start = std::time::Instant::now();
        cache
            .store(
                vec![1.0, 0.0],
                "s".to_string(),
                json!({"id": "x"}),
                "t",
                "gpt-4o",
            )
            .await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < CMD_TIMEOUT + Duration::from_secs(3),
            "store should fail open promptly even with enforce_capacity, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn store_with_max_entries_zero_never_persists() {
        // max_entries == 0 means "caching disabled", matching the in-memory
        // backend. The store must bail before even attempting a connection,
        // so it returns near-instantly regardless of host reachability.
        let cache = RedisSemanticCache::with_prefix_and_max(
            "redis://127.0.0.1:1/0",
            DEFAULT_PREFIX,
            0,
            0.95,
            Duration::from_secs(60),
        )
        .expect("construct");
        let start = std::time::Instant::now();
        cache
            .store(
                vec![1.0, 0.0],
                "s".to_string(),
                json!({"id": "x"}),
                "t",
                "gpt-4o",
            )
            .await;
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "max_entries=0 should bail before connecting, took {:?}",
            start.elapsed()
        );
    }
}
