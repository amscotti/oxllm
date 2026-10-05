//! Qdrant-backed semantic cache.
//!
//! Uses Qdrant's REST API (`/collections/{name}/points/search` for similarity
//! search and `/collections/{name}/points` for upserts) as a persistent,
//! shared vector store. Cosine similarity is computed by Qdrant itself
//! (collection configured with `Cosine` distance); the client only issues the
//! HTTP requests and applies the TTL/expiry check on the returned payload.
//!
//! The collection is created lazily on the first store when it does not yet
//! exist (idempotent — a 409 "already exists" is treated as success). All
//! network failures degrade gracefully: `search` returns `None` (fall through
//! to the upstream call) and `store` logs a warning and skips, so a Qdrant
//! outage never breaks request serving.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::semantic::{SemanticCacheStore, cacheable_semantic_entry, valid_embedding};

/// HTTP timeout for Qdrant requests. Failures MUST NOT block the request
/// path — they log a warning and fall through.
const QDRANT_TIMEOUT: Duration = Duration::from_secs(5);

/// A Qdrant-backed semantic cache using the REST API.
pub struct QdrantCache {
    client: reqwest::Client,
    url: String,
    api_key: Option<String>,
    collection: String,
    dimension: usize,
    similarity_threshold: f64,
    ttl: Duration,
    /// Whether a collection-creation attempt has been made. Guards the lazy
    /// `PUT /collections/{name}` so it runs at most once per instance.
    collection_ensured: AtomicBool,
}

impl std::fmt::Debug for QdrantCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantCache")
            .field("url", &self.url)
            .field("collection", &self.collection)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl QdrantCache {
    /// Construct a Qdrant cache. No I/O is performed — the collection is
    /// created lazily on the first store.
    pub fn new(
        url: impl Into<String>,
        api_key: Option<String>,
        collection: impl Into<String>,
        dimension: usize,
        similarity_threshold: f64,
        ttl: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(QDRANT_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            url: url.into().trim_end_matches('/').to_string(),
            api_key,
            collection: collection.into(),
            dimension,
            similarity_threshold,
            ttl,
            collection_ensured: AtomicBool::new(false),
        }
    }

    fn points_url(&self, path: &str) -> String {
        format!("{}/collections/{}/{}", self.url, self.collection, path)
    }

    fn collection_url(&self) -> String {
        format!("{}/collections/{}", self.url, self.collection)
    }

    fn auth_header(&self) -> Option<(&str, String)> {
        self.api_key
            .as_deref()
            .map(|key| ("api-key", key.to_string()))
    }

    /// Create the collection with a cosine-distance vector config if it does
    /// not already exist. Idempotent: a 409 (already exists) is treated as
    /// success. Best-effort — failures are logged and swallowed.
    async fn ensure_collection(&self) {
        if self.collection_ensured.load(Ordering::Relaxed) {
            return;
        }
        let body = json!({
            "vectors": {
                "size": self.dimension,
                "distance": "Cosine"
            }
        });
        let mut req = self.client.put(self.collection_url()).json(&body);
        if let Some((name, value)) = self.auth_header() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&value) {
                req = req.header(name, v);
            }
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                // 200 (created) and 409 (already exists) are both fine.
                if status.is_success() || status.as_u16() == 409 {
                    self.ensure_payload_indexes().await;
                    self.collection_ensured.store(true, Ordering::Relaxed);
                } else {
                    tracing::warn!(
                        status = status.as_u16(),
                        collection = %self.collection,
                        "qdrant collection creation returned non-success; points will be upserted best-effort"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "qdrant collection creation failed; will retry on next store"
                );
            }
        }
    }

    /// Create `keyword` payload indexes on `api_key_hash` and `model`. Every
    /// search filters on both fields; without an index Qdrant evaluates the
    /// `must` clause by scanning every point, degrading to O(n). Idempotent —
    /// a 409 (already indexed) is treated as success. Best-effort.
    async fn ensure_payload_indexes(&self) {
        for field in ["api_key_hash", "model"] {
            self.create_payload_index(field).await;
        }
    }

    async fn create_payload_index(&self, field: &str) {
        let body = json!({ "field_name": field, "field_schema": "keyword" });
        let url = format!("{}/index", self.collection_url());
        let mut req = self.client.put(&url).json(&body);
        if let Some((name, value)) = self.auth_header() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&value) {
                req = req.header(name, v);
            }
        }
        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if !(status.is_success() || status.as_u16() == 409) {
                    tracing::debug!(
                        status = status.as_u16(),
                        field = field,
                        "qdrant payload index creation returned non-success (best-effort)"
                    );
                }
            }
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    field = field,
                    "qdrant payload index creation failed (best-effort)"
                );
            }
        }
    }

    /// Build a Qdrant filter matching both `api_key_hash` and `model`.
    fn scope_filter(api_key_hash: &str, model: &str) -> Value {
        json!({
            "must": [
                {"key": "api_key_hash", "match": {"value": api_key_hash}},
                {"key": "model", "match": {"value": model}}
            ]
        })
    }
}

#[async_trait]
impl SemanticCacheStore for QdrantCache {
    async fn search(
        &self,
        query_embedding: &[f32],
        api_key_hash: &str,
        model: &str,
    ) -> Option<Value> {
        if !valid_embedding(query_embedding) {
            return None;
        }
        let mut filter = Self::scope_filter(api_key_hash, model);
        // Exclude already-expired entries server-side: search returns the
        // top-N by score, so if the top-10 are all stale, fresher in-scope
        // entries would never be considered and the lookup would degrade to
        // a false cache miss. Entries without `expires_at` (never-expiring
        // or legacy) do not match the range and stay in scope.
        filter["must_not"] = json!([{
            "key": "expires_at",
            "range": { "lte": now_secs() }
        }]);
        let body = json!({
            "vector": query_embedding,
            "filter": filter,
            "limit": 10,
            "with_payload": true,
            "score_threshold": self.similarity_threshold,
        });
        let mut req = self
            .client
            .post(self.points_url("points/search"))
            .json(&body);
        if let Some((name, value)) = self.auth_header() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&value) {
                req = req.header(name, v);
            }
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "qdrant search request failed; falling through");
                return None;
            }
        };
        if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 404 {
                tracing::debug!(
                    status = status.as_u16(),
                    "qdrant search: collection not found (not yet created); falling through"
                );
            } else {
                tracing::warn!(
                    status = status.as_u16(),
                    "qdrant search non-2xx; falling through"
                );
            }
            return None;
        }
        let parsed: SearchResponse = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "qdrant search body parse failed; falling through");
                return None;
            }
        };
        select_search_result(&parsed, self.ttl)
    }

    async fn store(
        &self,
        embedding: Vec<f32>,
        _request_text: String,
        response: Value,
        api_key_hash: &str,
        model: &str,
    ) {
        if !cacheable_semantic_entry(&embedding, &response) {
            return;
        }
        let vector_dim = embedding.len();
        self.ensure_collection().await;
        let id = uuid_str();
        let created_at = now_secs();
        // Materialized per-entry deadline (absent when the entry never
        // expires). `evict_expired` filters on this rather than on
        // `created_at` + the *current* config TTL, so a config change to a
        // shorter TTL cannot mass-delete still-fresh entries written under a
        // longer one.
        // Round sub-second TTLs UP to at least 1s: a 0 < ttl < 1s config
        // must not truncate to 0 (which both the reader here and
        // `evict_expired` treat as "never expire" — silently immortal).
        let ttl_secs = if self.ttl.is_zero() {
            0
        } else {
            self.ttl.as_secs().max(1)
        };
        let mut payload = json!({
            "response": response,
            "api_key_hash": api_key_hash,
            "model": model,
            "created_at": created_at,
            // ttl_secs == 0 means "never expire" (select_search_result skips
            // the expiry check).
            "ttl_secs": ttl_secs,
        });
        if ttl_secs > 0 {
            payload["expires_at"] = json!(created_at.saturating_add(ttl_secs));
        }
        let body = json!({
            "points": [{
                "id": id,
                "vector": embedding,
                "payload": payload,
            }]
        });
        let mut req = self.client.put(self.points_url("points")).json(&body);
        if let Some((name, value)) = self.auth_header() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&value) {
                req = req.header(name, v);
            }
        }
        match req.send().await {
            Ok(resp) if !resp.status().is_success() => {
                let status = resp.status();
                let body_text = resp.text().await.unwrap_or_default();
                if body_text.to_ascii_lowercase().contains("dimension") {
                    tracing::warn!(
                        status = status.as_u16(),
                        collection = %self.collection,
                        collection_dimension = self.dimension,
                        vector_dimension = vector_dim,
                        "qdrant dimension mismatch: collection {:?} expects {} dimensions but the \
                         embedding model produced {}; cache will not populate until the \
                         `qdrant_dimension` config matches the model",
                        self.collection,
                        self.dimension,
                        vector_dim
                    );
                } else {
                    tracing::warn!(
                        status = status.as_u16(),
                        "qdrant upsert non-2xx; entry not stored"
                    );
                }
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "qdrant upsert request failed; entry not stored");
            }
        }
    }

    async fn evict_expired(&self) {
        // Expiry is per entry: a point is expired when its own stored
        // `created_at + ttl_secs` (materialized as `expires_at` at write
        // time) is in the past — NOT when `created_at` falls below a cutoff
        // derived from the *current* config TTL, which would mass-delete
        // still-fresh entries written under a longer TTL after a config
        // change. Missing deadlines are deliberately not guessed from the
        // current config: that would delete explicitly never-expiring entries
        // along with legacy points. A new zero-TTL config must still clean up
        // expired deadlines written under a previous finite-TTL config.
        let now = now_secs();
        let body = json!({
            "filter": {
                "must": [{ "key": "expires_at", "range": { "lte": now } }]
            }
        });
        let mut req = self
            .client
            .post(self.points_url("points/delete"))
            .json(&body);
        if let Some((name, value)) = self.auth_header() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&value) {
                req = req.header(name, v);
            }
        }
        match req.send().await {
            Ok(resp) if !resp.status().is_success() => {
                tracing::debug!(
                    status = resp.status().as_u16(),
                    "qdrant expired-point cleanup returned non-2xx (best-effort)"
                );
            }
            Ok(_) => {}
            Err(e) => {
                tracing::debug!(error = %e, "qdrant expired-point cleanup failed (best-effort)");
            }
        }
    }
}

/// Parsed top-level Qdrant search response: `{"result": [ {score, payload}, ... ]}`.
#[derive(Debug, Deserialize)]
struct SearchResponse {
    result: Vec<SearchHit>,
}

#[derive(Debug, Deserialize)]
struct SearchHit {
    #[allow(dead_code)]
    score: f64,
    payload: Option<Payload>,
}

#[derive(Debug, Deserialize)]
struct Payload {
    response: Value,
    #[serde(default)]
    expires_at: Option<u64>,
    #[serde(default)]
    created_at: Option<u64>,
    #[serde(default)]
    ttl_secs: Option<u64>,
}

/// Pick the first non-expired hit from a Qdrant search response. Qdrant
/// returns results sorted by descending similarity (and applies
/// `score_threshold` server-side), so the first payload that has not exceeded
/// its TTL wins. Pure function — unit-tested directly.
fn select_search_result(resp: &SearchResponse, fallback_ttl: Duration) -> Option<Value> {
    let now = now_secs();
    for hit in &resp.result {
        let Some(payload) = &hit.payload else {
            continue;
        };
        if let Some(deadline) = payload.expires_at {
            if now >= deadline {
                continue;
            }
            return Some(payload.response.clone());
        }
        let created_at = payload.created_at.unwrap_or(now);
        let ttl_secs = payload.ttl_secs.unwrap_or(fallback_ttl.as_secs());
        if ttl_secs > 0 && now.saturating_sub(created_at) >= ttl_secs {
            continue;
        }
        return Some(payload.response.clone());
    }
    None
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn uuid_str() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::MockServer;
    use serde_json::json;

    fn search_response(hits: Vec<Value>) -> Value {
        json!({ "result": hits })
    }

    fn hit(score: f64, response: Value, created_at: Option<u64>, ttl_secs: Option<u64>) -> Value {
        let mut payload = json!({ "response": response });
        if let Some(c) = created_at {
            payload["created_at"] = json!(c);
        }
        if let Some(t) = ttl_secs {
            payload["ttl_secs"] = json!(t);
        }
        json!({ "score": score, "payload": payload })
    }

    #[test]
    fn select_returns_first_non_expired_hit() {
        let now = now_secs();
        let resp: SearchResponse = serde_json::from_value(search_response(vec![
            hit(0.99, json!({"id": "expired"}), Some(now - 10), Some(5)),
            hit(0.97, json!({"id": "fresh"}), Some(now), Some(100)),
        ]))
        .expect("parse");
        let chosen = select_search_result(&resp, Duration::from_secs(60)).expect("a hit");
        assert_eq!(chosen["id"], "fresh");
    }

    #[test]
    fn select_skips_all_when_everything_expired() {
        let now = now_secs();
        let resp: SearchResponse = serde_json::from_value(search_response(vec![
            hit(0.99, json!({"id": "a"}), Some(now - 100), Some(5)),
            hit(0.98, json!({"id": "b"}), Some(now - 100), Some(5)),
        ]))
        .expect("parse");
        assert!(select_search_result(&resp, Duration::from_secs(60)).is_none());
    }

    #[test]
    fn select_honors_materialized_deadlines_across_ttl_config_changes() {
        let now = now_secs();
        let resp: SearchResponse = serde_json::from_value(search_response(vec![
            json!({"score": 1.0, "payload": {
                "response": {"id": "expired"}, "created_at": now,
                "ttl_secs": 0, "expires_at": now
            }}),
            json!({"score": 0.99, "payload": {
                "response": {"id": "still-fresh"}, "created_at": now - 100,
                "ttl_secs": 3600, "expires_at": now + 3500
            }}),
        ]))
        .unwrap();
        assert_eq!(
            select_search_result(&resp, Duration::from_secs(1)).unwrap()["id"],
            "still-fresh"
        );
        assert_eq!(
            select_search_result(&resp, Duration::ZERO).unwrap()["id"],
            "still-fresh"
        );
    }

    #[test]
    fn select_uses_fallback_ttl_when_payload_missing_ttl() {
        let now = now_secs();
        // No ttl_secs in payload → fallback_ttl (60s) governs.
        let resp: SearchResponse = serde_json::from_value(search_response(vec![hit(
            0.95,
            json!({"id": "ok"}),
            Some(now - 10),
            None,
        )]))
        .expect("parse");
        let chosen = select_search_result(&resp, Duration::from_secs(60)).expect("hit");
        assert_eq!(chosen["id"], "ok");

        // Same created_at but a tiny fallback ttl → expired.
        let resp2: SearchResponse = serde_json::from_value(search_response(vec![hit(
            0.95,
            json!({"id": "ok"}),
            Some(now - 10),
            None,
        )]))
        .expect("parse");
        assert!(select_search_result(&resp2, Duration::from_secs(5)).is_none());
    }

    #[test]
    fn scope_filter_matches_both_key_and_model() {
        let f = QdrantCache::scope_filter("tenant-a", "gpt-4o");
        let must = f
            .get("must")
            .and_then(|v| v.as_array())
            .expect("must array");
        assert_eq!(must.len(), 2);
        let keys: Vec<&str> = must
            .iter()
            .filter_map(|c| c.get("key").and_then(|v| v.as_str()))
            .collect();
        assert!(keys.contains(&"api_key_hash"));
        assert!(keys.contains(&"model"));
    }

    #[tokio::test]
    async fn store_upserts_point_and_creates_collection() {
        let server = MockServer::start();
        let collection = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/collections/oxllm_cache");
            then.status(200).json_body(json!({"result": true}));
        });
        // `ensure_collection` now also creates keyword payload indexes on the
        // two filter fields so scoped searches do not full-scan the collection.
        mock_payload_indexes(&server);
        let upsert = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/collections/oxllm_cache/points");
            then.status(200)
                .json_body(json!({"result": {"operation_id": 0}}));
        });

        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(60),
        );
        cache
            .store(
                vec![0.1, 0.2, 0.3],
                "ignored".to_string(),
                json!({"id": "x"}),
                "abc",
                "gpt-4o",
            )
            .await;

        collection.assert_hits(1);
        upsert.assert_hits(1);
    }

    #[tokio::test]
    async fn ensure_collection_creates_payload_indexes_once() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/collections/oxllm_cache");
            then.status(200).json_body(json!({"result": true}));
        });
        let idx_key = mock_payload_index_hits(&server, "api_key_hash");
        let idx_model = mock_payload_index_hits(&server, "model");

        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(60),
        );
        // Two stores both call ensure_collection, but only the first should
        // emit the index-creation PUTs (collection_ensured guards the rest).
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/collections/oxllm_cache/points");
            then.status(200)
                .json_body(json!({"result": {"operation_id": 0}}));
        });
        cache
            .store(
                vec![0.1, 0.2, 0.3],
                "ignored".to_string(),
                json!({"id": "a"}),
                "abc",
                "gpt-4o",
            )
            .await;
        cache
            .store(
                vec![0.3, 0.2, 0.1],
                "ignored".to_string(),
                json!({"id": "b"}),
                "abc",
                "gpt-4o",
            )
            .await;

        idx_key.assert_hits(1);
        idx_model.assert_hits(1);
    }

    /// Mock the keyword payload-index PUT for `field` and return the handle so
    /// the caller can assert hit count.
    fn mock_payload_index_hits<'a>(server: &'a MockServer, field: &str) -> httpmock::Mock<'a> {
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/collections/oxllm_cache/index")
                .json_body_partial(format!(
                    r#"{{"field_name":"{field}","field_schema":"keyword"}}"#
                ));
            then.status(200).json_body(json!({"result": true}));
        })
    }

    fn mock_payload_indexes(server: &MockServer) {
        mock_payload_index_hits(server, "api_key_hash");
        mock_payload_index_hits(server, "model");
    }

    #[tokio::test]
    async fn store_reports_dimension_mismatch_on_failed_upsert() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/collections/oxllm_cache");
            then.status(200).json_body(json!({"result": true}));
        });
        mock_payload_indexes(&server);
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/collections/oxllm_cache/points");
            then.status(400).json_body(json!({
                "status": "error",
                "message": "No vector found with name '', expected: 3 dimensions, found: 2"
            }));
        });

        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(60),
        );
        // Best-effort: the store must not panic; the dedicated diagnostic is
        // emitted at warn level (asserted here only by absence of panic and a
        // completed call — the log line itself is not observable in-process).
        cache
            .store(
                vec![0.1, 0.2],
                "ignored".to_string(),
                json!({"id": "x"}),
                "abc",
                "gpt-4o",
            )
            .await;
    }

    #[tokio::test]
    async fn search_returns_cached_response_on_hit() {
        let server = MockServer::start();
        let now = now_secs();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/collections/oxllm_cache/points/search");
            then.status(200).json_body(search_response(vec![hit(
                0.97,
                json!({"id": "cached"}),
                Some(now),
                Some(3600),
            )]));
        });

        let cache = QdrantCache::new(
            server.base_url(),
            Some("s3cr3t".to_string()),
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(60),
        );
        let hit = cache
            .search(&[0.1, 0.2, 0.3], "abc", "gpt-4o")
            .await
            .expect("hit");
        assert_eq!(hit["id"], "cached");
    }

    #[tokio::test]
    async fn search_returns_none_on_non_2xx() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/collections/oxllm_cache/points/search");
            then.status(500);
        });
        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(60),
        );
        assert!(cache.search(&[0.1], "abc", "gpt-4o").await.is_none());
    }

    #[tokio::test]
    async fn search_returns_none_on_404_collection_not_found() {
        // Before the first store creates the collection, Qdrant returns 404.
        // The search must fall through gracefully without panicking.
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/collections/oxllm_cache/points/search");
            then.status(404).json_body(json!({
                "status": "error",
                "message": "Not found: collection `oxllm_cache` doesn't exist"
            }));
        });
        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(60),
        );
        assert!(
            cache.search(&[0.1], "abc", "gpt-4o").await.is_none(),
            "404 must fall through without panic"
        );
    }

    #[tokio::test]
    async fn search_returns_none_when_all_hits_expired() {
        let server = MockServer::start();
        let now = now_secs();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/collections/oxllm_cache/points/search");
            then.status(200).json_body(search_response(vec![hit(
                0.97,
                json!({"id": "stale"}),
                Some(now - 1000),
                Some(5),
            )]));
        });
        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(60),
        );
        assert!(cache.search(&[0.1], "abc", "gpt-4o").await.is_none());
    }

    #[tokio::test]
    async fn evict_expired_issues_delete_by_expires_at_filter() {
        let server = MockServer::start();
        let ttl = 10i64;
        let delete = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/collections/oxllm_cache/points/delete")
                .json_body_partial(r#"{"filter":{"must":[{"key":"expires_at"}]}}"#);
            then.status(200)
                .json_body(json!({"result": {"operation_id": 0}}));
        });
        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::from_secs(ttl as u64),
        );
        cache.evict_expired().await;
        delete.assert_hits(1);
    }

    #[tokio::test]
    async fn evict_expired_after_zero_ttl_reload_only_deletes_materialized_deadlines() {
        let server = MockServer::start();
        let unsafe_legacy_delete = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/collections/oxllm_cache/points/delete")
                .body_contains("created_at");
            then.status(400);
        });
        let delete = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/collections/oxllm_cache/points/delete")
                .json_body_partial(r#"{"filter":{"must":[{"key":"expires_at"}]}}"#)
                .body_contains("\"lte\"");
            then.status(200);
        });
        let cache = QdrantCache::new(
            server.base_url(),
            None,
            "oxllm_cache",
            3,
            0.9,
            Duration::ZERO,
        );
        cache.evict_expired().await;
        unsafe_legacy_delete.assert_hits(0);
        delete.assert_hits(1);
    }
}
