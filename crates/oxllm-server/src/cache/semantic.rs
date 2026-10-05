//! In-memory semantic cache: embedding-based similarity matching for
//! semantically similar past requests.
//!
//! Entries are stored as `(embedding, response, api_key_hash,
//! created_at, ttl)`. Search computes cosine similarity between the query
//! embedding and every stored embedding for the same `api_key_hash`,
//! returning the highest-scoring non-expired entry whose similarity is at or
//! above the configured threshold.
//!
//! Single-instance only (mirrors `InMemoryCache`). Suits workloads up to
//! ~10,000 entries; the linear scan is `O(n*dim)` but Rust autovectorizes the
//! dot-product inner loop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

/// One stored semantic-cache entry.
struct SemanticCacheEntry {
    embedding: Vec<f32>,
    response: Value,
    api_key_hash: String,
    model: String,
    created_at: Instant,
    ttl: Duration,
}

/// Backend trait for semantic-cache stores. The in-memory [`SemanticCache`],
/// the Redis backend ([`super::redis_semantic::RedisSemanticCache`]),
/// and the Qdrant backend ([`super::qdrant::QdrantCache`]) all implement
/// it; [`DualSemanticCache`] layers in-memory first with one of the external
/// backends as a fallback.
///
/// All methods are async because external backends perform I/O; the in-memory
/// implementation is lock-only.
///
/// `model` is an opaque exact-match scope. The response-cache hook supplies
/// [`semantic_scope_for`], which includes the requested model, embedding-model
/// identity, and exact request compatibility. Tenant isolation is an additional,
/// independent constraint; an embedding score must never replace either check.
#[async_trait]
pub trait SemanticCacheStore: Send + Sync {
    /// Search for a semantically similar entry. Returns a cloned response when
    /// the best match has cosine similarity `>= threshold`, shares the same
    /// `api_key_hash` and `model`, and has not exceeded its TTL. Returns
    /// `None` otherwise.
    async fn search(
        &self,
        query_embedding: &[f32],
        api_key_hash: &str,
        model: &str,
    ) -> Option<Value>;

    /// Store a new entry. `request_text` is accepted for API symmetry but
    /// ignored by all backends (retained in the trait signature to avoid
    /// breaking call sites that compute it for the embedding fetch).
    async fn store(
        &self,
        embedding: Vec<f32>,
        request_text: String,
        response: Value,
        api_key_hash: &str,
        model: &str,
    );

    /// Remove expired entries. Called opportunistically by the hook on the
    /// store path so the scan cost is paid by writers, not readers.
    async fn evict_expired(&self);

    /// Current entry count (for diagnostics / tests). External backends may
    /// return `0` when a count is not cheaply available.
    async fn len(&self) -> usize {
        0
    }

    /// Whether the store currently holds zero entries. Defaults to
    /// `self.len() == 0`; backends may override for a cheaper check.
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

/// In-memory vector store for semantic caching.
///
/// Thread-safe via a single `RwLock<Vec<_>>`: searches take a read lock and
/// iterate; stores take a write lock and append (with FIFO eviction when at
/// capacity). Searches are linear — fine for the documented ~10k entry ceiling.
pub struct SemanticCache {
    entries: RwLock<Vec<SemanticCacheEntry>>,
    max_entries: usize,
    similarity_threshold: f64,
    ttl: Duration,
}

impl SemanticCache {
    /// Create a new cache with the given capacity, similarity cutoff, and
    /// default TTL applied to every entry.
    pub fn new(max_entries: usize, similarity_threshold: f64, ttl: Duration) -> Self {
        Self {
            entries: RwLock::new(Vec::with_capacity(max_entries.min(1024))),
            max_entries,
            similarity_threshold,
            ttl,
        }
    }
}

#[async_trait]
impl SemanticCacheStore for SemanticCache {
    /// Search for a semantically similar entry. Returns a cloned response when
    /// the best match has cosine similarity `>= threshold`, shares the same
    /// `api_key_hash` and `model`, and has not exceeded its TTL. Returns
    /// `None` otherwise.
    async fn search(
        &self,
        query_embedding: &[f32],
        api_key_hash: &str,
        model: &str,
    ) -> Option<Value> {
        if !valid_embedding(query_embedding) {
            return None;
        }
        let guard = self.entries.read().await;
        let now = Instant::now();
        let mut best: Option<(f64, &SemanticCacheEntry)> = None;
        for entry in guard.iter() {
            if entry.api_key_hash != api_key_hash || entry.model != model {
                continue;
            }
            if now.duration_since(entry.created_at) >= entry.ttl && !entry.ttl.is_zero() {
                continue;
            }
            if entry.embedding.len() != query_embedding.len() {
                continue;
            }
            let sim = cosine_similarity(query_embedding, &entry.embedding);
            match best {
                None => best = Some((sim, entry)),
                Some((cur, _)) if sim > cur => best = Some((sim, entry)),
                _ => {}
            }
        }
        let (sim, entry) = best?;
        if sim >= self.similarity_threshold
            && (entry.ttl.is_zero() || entry.created_at.elapsed() < entry.ttl)
        {
            Some(entry.response.clone())
        } else {
            None
        }
    }

    /// Store a new entry. Evicts oldest entries (FIFO) when at capacity.
    /// A capacity of `0` disables caching (the entry is dropped).
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
        let entry = SemanticCacheEntry {
            embedding,
            response,
            api_key_hash: api_key_hash.to_string(),
            model: model.to_string(),
            created_at: Instant::now(),
            ttl: self.ttl,
        };
        let mut guard = self.entries.write().await;
        if guard.len() >= self.max_entries {
            let overflow = guard.len() + 1 - self.max_entries;
            let take = overflow.min(guard.len());
            // FIFO: drop the `take` oldest entries (lowest indices, since we
            // only ever append).
            guard.drain(0..take);
        }
        guard.push(entry);
    }

    /// Remove expired entries. Called opportunistically by the hook on the
    /// store path so the scan cost is paid by writers, not readers.
    async fn evict_expired(&self) {
        let mut guard = self.entries.write().await;
        let now = Instant::now();
        guard.retain(|e| e.ttl.is_zero() || now.duration_since(e.created_at) < e.ttl);
    }

    /// Current entry count (for diagnostics / tests).
    async fn len(&self) -> usize {
        self.entries.read().await.len()
    }
}

/// A two-tier semantic cache: an in-memory [`SemanticCache`] as the primary
/// (hot, zero-latency) tier and an optional external backend
/// (Redis or Qdrant) as a fallback for entries that miss in memory.
///
/// `search` consults the primary first and, on a miss, the fallback — so the
/// in-memory tier absorbs the hot path while the external tier provides
/// persistence and cross-replica sharing. `store` writes to both so the
/// fallback stays warm. This mirrors the `DualCache` pattern used for the
/// exact-match response cache.
pub struct DualSemanticCache {
    primary: Arc<SemanticCache>,
    fallback: Option<Arc<dyn SemanticCacheStore>>,
}

impl DualSemanticCache {
    /// Compose an in-memory primary with an optional external fallback.
    pub fn new(primary: Arc<SemanticCache>, fallback: Option<Arc<dyn SemanticCacheStore>>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl SemanticCacheStore for DualSemanticCache {
    async fn search(
        &self,
        query_embedding: &[f32],
        api_key_hash: &str,
        model: &str,
    ) -> Option<Value> {
        if let Some(hit) = self
            .primary
            .search(query_embedding, api_key_hash, model)
            .await
        {
            return Some(hit);
        }
        match &self.fallback {
            Some(fb) => fb.search(query_embedding, api_key_hash, model).await,
            None => None,
        }
    }

    async fn store(
        &self,
        embedding: Vec<f32>,
        request_text: String,
        response: Value,
        api_key_hash: &str,
        model: &str,
    ) {
        if !cacheable_semantic_entry(&embedding, &response) {
            return;
        }
        match &self.fallback {
            Some(fb) => {
                self.primary
                    .store(
                        embedding.clone(),
                        String::new(),
                        response.clone(),
                        api_key_hash,
                        model,
                    )
                    .await;
                fb.store(embedding, request_text, response, api_key_hash, model)
                    .await;
            }
            None => {
                self.primary
                    .store(embedding, request_text, response, api_key_hash, model)
                    .await;
            }
        }
    }

    async fn evict_expired(&self) {
        self.primary.evict_expired().await;
        if let Some(fb) = &self.fallback {
            fb.evict_expired().await;
        }
    }

    /// Total entry count: primary plus fallback, so diagnostics don't
    /// understate occupancy when an external backend is active.
    async fn len(&self) -> usize {
        match &self.fallback {
            Some(fb) => self.primary.len().await.saturating_add(fb.len().await),
            None => self.primary.len().await,
        }
    }
}

/// Cosine similarity between two vectors, returned as `f64` for stable
/// threshold comparisons. Returns `0.0` when either vector is zero-norm
/// (avoids NaN).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Build the semantic-cache runtime from `oxllm_settings.cache.semantic_cache`.
///
/// Returns `None` when semantic caching is disabled or the embedding model is
/// missing. The embedding API base is resolved in this order:
/// 1. `semantic_cache.api_base` (explicit override)
/// 2. The `api_base` of the first deployment whose `model_name` matches the
///    configured `embedding_model`
/// 3. `None` → semantic caching is disabled with a warning (no upstream URL)
///
/// The embedding API key follows the same precedence. When neither is set,
/// the deployment's `oxllm_params.api_key` is used.
pub(crate) fn build_semantic_runtime(
    config: &crate::config::ProxyConfig,
    http_client: &reqwest::Client,
) -> Option<std::sync::Arc<crate::hooks::cache_hook::SemanticCacheRuntime>> {
    let semantic = config
        .oxllm_settings
        .cache
        .as_ref()
        .and_then(|c| c.semantic_cache.as_ref())?;
    if !semantic.enabled {
        return None;
    }
    if semantic.embedding_model.trim().is_empty() {
        tracing::warn!(
            "semantic cache enabled but embedding_model is empty; disabling semantic cache"
        );
        return None;
    }

    let threshold = semantic.similarity_threshold.unwrap_or(0.95);
    let max_entries = semantic.max_entries.unwrap_or(10_000);
    if max_entries == 0 {
        return None;
    }
    let ttl = Duration::from_secs(semantic.ttl_secs.unwrap_or(3600));

    let (api_base, api_key) = resolve_embedding_endpoint(config, semantic);

    let Some(api_base) = api_base else {
        tracing::warn!(
            model = %semantic.embedding_model,
            "semantic cache enabled but no embedding api_base resolved; disabling semantic cache"
        );
        return None;
    };

    let store: Arc<dyn SemanticCacheStore> =
        build_semantic_store(semantic, threshold, max_entries, ttl);
    Some(std::sync::Arc::new(
        crate::hooks::cache_hook::SemanticCacheRuntime::new(
            store,
            semantic.embedding_model.clone(),
            api_base,
            api_key,
            http_client.clone(),
        ),
    ))
}

/// Build the semantic-cache store from the `semantic_cache` config section.
///
/// An in-memory [`SemanticCache`] is always present (the hot tier). When
/// `redis_url` is configured it is layered with [`RedisSemanticCache`] as a
/// fallback via [`DualSemanticCache`]; otherwise `qdrant_url` selects
/// [`QdrantCache`]. When neither external backend is set, the in-memory store
/// is used on its own. Redis is preferred over Qdrant when both are set.
pub(crate) fn build_semantic_store(
    semantic: &crate::config::types::SemanticCacheConfig,
    threshold: f64,
    max_entries: usize,
    ttl: Duration,
) -> Arc<dyn SemanticCacheStore> {
    let primary = Arc::new(SemanticCache::new(max_entries, threshold, ttl));
    // A disabled hot tier must not still read/write an external tier (Qdrant
    // has no max_entries setting of its own).
    if max_entries == 0 {
        return primary;
    }
    let external = build_external_store(semantic, threshold, max_entries, ttl);
    match external {
        Some(fb) => Arc::new(DualSemanticCache::new(primary, Some(fb))),
        None => primary,
    }
}

/// Construct the optional external backend from config. Returns `None` when
/// no external backend is configured, or when construction fails (the
/// in-memory store then runs standalone with a warning).
fn build_external_store(
    semantic: &crate::config::types::SemanticCacheConfig,
    threshold: f64,
    max_entries: usize,
    ttl: Duration,
) -> Option<Arc<dyn SemanticCacheStore>> {
    if let Some(url) = semantic.redis_url.as_deref().filter(|s| !s.is_empty()) {
        #[cfg(feature = "redis")]
        {
            match super::redis_semantic::RedisSemanticCache::with_prefix_and_max(
                url,
                super::redis_semantic::DEFAULT_PREFIX,
                max_entries,
                threshold,
                ttl,
            ) {
                Ok(store) => return Some(Arc::new(store)),
                Err(e) => tracing::warn!(
                    error = %e,
                    redis_url = %crate::security::sanitize_url(url),
                    "redis semantic cache init failed; falling back to in-memory only"
                ),
            }
        }
        #[cfg(not(feature = "redis"))]
        {
            let _ = (url, threshold, max_entries, ttl);
            tracing::warn!(
                "redis_url is set for the semantic cache but the 'redis' feature is disabled; \
                 using in-memory only"
            );
        }
    }

    if let Some(url) = semantic.qdrant_url.as_deref().filter(|s| !s.is_empty()) {
        let collection = semantic
            .qdrant_collection
            .clone()
            .unwrap_or_else(|| DEFAULT_QDRANT_COLLECTION.to_string());
        let dimension = semantic
            .qdrant_dimension
            .unwrap_or(DEFAULT_QDRANT_DIMENSION);
        return Some(Arc::new(super::qdrant::QdrantCache::new(
            url,
            semantic.qdrant_api_key.clone(),
            collection,
            dimension,
            threshold,
            ttl,
        )));
    }

    None
}

/// Default Qdrant collection name when `qdrant_collection` is unset.
pub(crate) const DEFAULT_QDRANT_COLLECTION: &str = "oxllm_cache";
/// Default vector dimension for the Qdrant collection (OpenAI
/// `text-embedding-3-small`) when `qdrant_dimension` is unset.
pub(crate) const DEFAULT_QDRANT_DIMENSION: usize = 1536;

/// Resolve the embedding endpoint (api_base + api_key) by consulting the
/// explicit config first, then falling back to a deployment match.
pub(crate) fn resolve_embedding_endpoint(
    config: &crate::config::ProxyConfig,
    semantic: &crate::config::types::SemanticCacheConfig,
) -> (Option<String>, Option<String>) {
    let api_base = semantic.api_base.clone().or_else(|| {
        config.model_list.iter().find_map(|dep| {
            if dep.model_name == semantic.embedding_model {
                dep.oxllm_params.api_base.clone()
            } else {
                None
            }
        })
    });

    let api_key = semantic.api_key.clone().or_else(|| {
        config.model_list.iter().find_map(|dep| {
            if dep.model_name == semantic.embedding_model {
                dep.oxllm_params.api_key.clone()
            } else {
                None
            }
        })
    });

    (api_base, api_key)
}

/// Maximum number of bytes of extracted text fed to the embedding endpoint.
/// Generous (well beyond any realistic single request, so semantic similarity
/// is unaffected) while bounding resident allocation and the outbound POST
/// body: a client that opts into caching cannot amplify an arbitrarily large
/// request through the embedding fetch beyond this ceiling.
pub const MAX_EMBEDDING_TEXT_BYTES: usize = 256 * 1024;

/// Bound each vector before retaining it in the pending stash or either cache
/// tier. This accommodates common embedding models while bounding a malformed
/// provider's dimension count independently of its JSON response size.
pub(crate) const MAX_EMBEDDING_DIMENSIONS: usize = 16_384;

pub(crate) fn valid_embedding(embedding: &[f32]) -> bool {
    !embedding.is_empty()
        && embedding.len() <= MAX_EMBEDDING_DIMENSIONS
        && embedding.iter().all(|v| v.is_finite())
        && embedding.iter().any(|v| *v != 0.0)
}

/// Count serialized bytes without allocating another response-sized buffer.
/// The writer stops at the cap, so rejected responses do not need a full scan.
pub(crate) fn json_fits(value: &Value, max_bytes: usize) -> bool {
    struct LimitedWriter(usize);
    impl std::io::Write for LimitedWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.0 {
                return Err(std::io::Error::other("cache JSON size limit exceeded"));
            }
            self.0 -= bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(LimitedWriter(max_bytes), value).is_ok()
}

pub(crate) fn cacheable_semantic_entry(embedding: &[f32], response: &Value) -> bool {
    valid_embedding(embedding) && json_fits(response, super::MAX_VALUE_BYTES)
}

/// Sort every object, including nested tool schemas, independently of whether
/// serde_json was built with `preserve_order`. Arrays retain their identity and
/// order. This is shared with the exact cache key.
pub(crate) fn canonicalize_json(value: &mut Value) {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<_> = std::mem::take(object).into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (key, mut value) in entries {
                canonicalize_json(&mut value);
                object.insert(key, value);
            }
        }
        Value::Array(array) => {
            for value in array {
                canonicalize_json(value);
            }
        }
        _ => {}
    }
}

/// Exact compatibility namespace for semantic matching. Only ordinary user
/// text, string `input` items, and a string `prompt` may vary semantically.
/// Roles, message/part boundaries, system/developer instructions, assistant
/// history, tool results/calls, multimedia identity, generation parameters,
/// tool/schema configuration, and every unknown field remain exact constraints.
/// `cache` controls affect lookup/store policy rather than model output.
///
/// The embedding model and endpoint are included even when two models produce
/// the same number of dimensions. The versioned scope also makes legacy
/// key+model-only persistent entries unreachable without deleting shared data.
///
/// Oversized requests bypass semantic caching rather than matching on a
/// truncated prompt. Checking serialized size is conservative and bounds both
/// compatibility-key work and the text sent to the embedding provider.
pub fn semantic_scope_for(
    body: &Value,
    model: &str,
    embedding_model: &str,
    api_base: &str,
) -> Option<String> {
    if !json_fits(body, MAX_EMBEDDING_TEXT_BYTES) {
        return None;
    }
    let mut compatible = body.clone();
    let object = compatible.as_object_mut()?;
    object.remove("cache");
    if let Some(Value::Array(messages)) = object.get_mut("messages") {
        for message in messages {
            if message.get("role").and_then(Value::as_str) == Some("user") {
                if let Some(content) = message.get_mut("content") {
                    mask_user_text(content);
                }
            }
        }
    }
    if let Some(input) = object.get_mut("input") {
        match input {
            Value::String(text) => text.clear(),
            Value::Array(items) => {
                for item in items {
                    if let Value::String(text) = item {
                        text.clear();
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(Value::String(prompt)) = object.get_mut("prompt") {
        prompt.clear();
    }
    let mut scope = serde_json::json!([
        "oxllm-semantic-v2",
        model,
        embedding_model,
        api_base.trim_end_matches('/'),
        compatible,
    ]);
    canonicalize_json(&mut scope);
    let serialized = serde_json::to_vec(&scope).ok()?;
    Some(format!("sem:v2:{:x}", Sha256::digest(serialized)))
}

fn mask_user_text(content: &mut Value) {
    match content {
        Value::String(text) => text.clear(),
        Value::Array(parts) => {
            for part in parts {
                if let Value::String(text) = part {
                    text.clear();
                } else if is_text_part(part) {
                    if let Some(Value::String(text)) = part.get_mut("text") {
                        text.clear();
                    }
                }
            }
        }
        _ => {}
    }
}

fn is_text_part(part: &Value) -> bool {
    matches!(
        part.get("type").and_then(Value::as_str),
        Some("text" | "input_text")
    )
}

/// Extract text from a chat-shaped request body for embedding. Concatenates
/// every message's `content` field with newlines, handling both string and
/// array-shaped content (OpenAI multimodal format). Also pulls the top-level
/// `input` field used by the embeddings endpoint, so the same extractor covers
/// both chat and embeddings requests.
///
/// Tools and response-shaping configuration are exact constraints in
/// [`semantic_scope_for`], never approximate embedding input. Distinct tool
/// serializations can still have arbitrarily similar embeddings.
///
/// Output is capped at [`MAX_EMBEDDING_TEXT_BYTES`] (truncated on a UTF-8 char
/// boundary) so a cache-enabled request with huge `messages`/`input` arrays
/// cannot amplify its body into an unbounded embedding POST and drive gateway
/// OOM.
pub fn extract_request_text(body: &Value) -> String {
    let mut out = String::new();

    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in messages {
            if let Some(content) = msg.get("content") {
                push_content(content, &mut out);
            }
            if out.len() >= MAX_EMBEDDING_TEXT_BYTES {
                return finalize_capped(out);
            }
        }
    }

    if let Some(input) = body.get("input") {
        match input {
            Value::String(s) => append_part(&mut out, s),
            Value::Array(arr) => {
                for item in arr {
                    if let Value::String(s) = item {
                        append_part(&mut out, s);
                    }
                    if out.len() >= MAX_EMBEDDING_TEXT_BYTES {
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(prompt) = body.get("prompt").and_then(|v| v.as_str()) {
        append_part(&mut out, prompt);
    }

    finalize_capped(out)
}

/// Append `part` to `out`, preceded by a `'\n'` separator when `out` is
/// non-empty (mirrors `Vec::join("\n")`). Stops appending once `out` reaches
/// [`MAX_EMBEDDING_TEXT_BYTES`], truncating the final segment on a UTF-8 char
/// boundary. Each part is read by reference, so a single oversized string is
/// never cloned in full.
fn append_part(out: &mut String, part: &str) {
    let cap = MAX_EMBEDDING_TEXT_BYTES;
    if out.len() >= cap {
        return;
    }
    if !out.is_empty() {
        if out.len() + 1 > cap {
            return;
        }
        out.push('\n');
    }
    let room = cap - out.len();
    if part.len() <= room {
        out.push_str(part);
        return;
    }
    let mut end = room;
    while end > 0 && !part.is_char_boundary(end) {
        end -= 1;
    }
    out.push_str(&part[..end]);
}

/// Enforce the cap on an already-accumulated string (defence in depth for the
/// paths that grow `out` without a per-segment check), truncating on a char
/// boundary.
fn finalize_capped(out: String) -> String {
    if out.len() <= MAX_EMBEDDING_TEXT_BYTES {
        return out;
    }
    let mut end = MAX_EMBEDDING_TEXT_BYTES;
    while end > 0 && !out.is_char_boundary(end) {
        end -= 1;
    }
    let mut s = out;
    s.truncate(end);
    s
}

fn push_content(content: &Value, out: &mut String) {
    match content {
        Value::String(s) => append_part(out, s),
        Value::Array(arr) => {
            for item in arr {
                if is_text_part(item) {
                    if let Some(text) = item.get("text").and_then(Value::as_str) {
                        append_part(out, text);
                    }
                } else if let Some(s) = item.as_str() {
                    append_part(out, s);
                }
                if out.len() >= MAX_EMBEDDING_TEXT_BYTES {
                    return;
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use tokio::time::sleep;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9, "got {sim}");
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9, "got {sim}");
    }

    #[test]
    fn cosine_opposite_vectors_is_minus_one() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![-1.0_f32, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-9, "got {sim}");
    }

    #[test]
    fn cosine_partial_overlap_between_zero_and_one() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.0 && sim < 1.0, "got {sim}");
        assert!(
            (sim - (std::f64::consts::FRAC_1_SQRT_2)).abs() < 1e-6,
            "got {sim}"
        );
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 2.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn cosine_mismatched_lengths_returns_zero() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn extract_text_from_string_content() {
        let body = json!({
        "messages": [
        {"role": "user", "content": "What is the capital of France?"}
        ]
        });
        assert_eq!(
            extract_request_text(&body),
            "What is the capital of France?"
        );
    }

    #[test]
    fn extract_text_from_array_content() {
        let body = json!({
        "messages": [
        {
        "role": "user",
        "content": [
        {"type": "text", "text": "describe"},
        {"type": "text", "text": "this image"}
        ]
        }
        ]
        });
        assert_eq!(extract_request_text(&body), "describe\nthis image");
    }

    #[test]
    fn extract_text_concatenates_system_and_user() {
        let body = json!({
        "messages": [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hi"}
        ]
        });
        assert_eq!(extract_request_text(&body), "be concise\nhi");
    }

    #[test]
    fn extract_text_handles_missing_content() {
        let body = json!({"messages": [{"role": "user"}]});
        assert_eq!(extract_request_text(&body), "");
    }

    #[test]
    fn extract_text_from_embeddings_input_field() {
        let body = json!({"model": "text-embedding-3-small", "input": "hello world"});
        assert_eq!(extract_request_text(&body), "hello world");
    }

    #[test]
    fn extract_text_from_embeddings_input_array() {
        let body = json!({"input": ["alpha", "beta"]});
        assert_eq!(extract_request_text(&body), "alpha\nbeta");
    }

    #[test]
    fn extract_text_from_legacy_completions_prompt() {
        let body = json!({"prompt": "once upon a time"});
        assert_eq!(extract_request_text(&body), "once upon a time");
    }

    #[test]
    fn extract_text_caps_message_content_length() {
        // A single huge message must not be cloned in full; output is capped.
        let big = "a".repeat(MAX_EMBEDDING_TEXT_BYTES * 4);
        let body = json!({"messages": [{"role": "user", "content": big}]});
        let text = extract_request_text(&body);
        assert!(
            text.len() <= MAX_EMBEDDING_TEXT_BYTES,
            "capped to {}, got {}",
            MAX_EMBEDDING_TEXT_BYTES,
            text.len()
        );
        assert_eq!(text.len(), MAX_EMBEDDING_TEXT_BYTES);
    }

    #[test]
    fn extract_text_caps_huge_input_array() {
        // An array of huge strings must be bounded; output never exceeds cap.
        let chunk = "b".repeat(MAX_EMBEDDING_TEXT_BYTES / 4 + 1);
        let body = json!({"input": [chunk.clone(), chunk.clone(), chunk.clone(), chunk.clone(), chunk.clone(), chunk.clone()]});
        let text = extract_request_text(&body);
        assert!(text.len() <= MAX_EMBEDDING_TEXT_BYTES, "got {}", text.len());
    }

    #[test]
    fn extract_text_truncates_on_char_boundary() {
        // Fill just past the cap with multibyte chars; the cut point must land
        // on a UTF-8 boundary (String::is_char_boundary).
        let unit = "é"; // 2 bytes
        let bytes_needed = MAX_EMBEDDING_TEXT_BYTES + 3;
        let count = bytes_needed / unit.len() + 1;
        let big = unit.repeat(count);
        let body = json!({"messages": [{"role": "user", "content": big}]});
        let text = extract_request_text(&body);
        assert!(text.len() <= MAX_EMBEDDING_TEXT_BYTES);
        assert!(text.is_char_boundary(text.len()));
    }

    #[test]
    fn extract_text_normal_request_unaffected_by_cap() {
        // A realistic request is well under the cap and unchanged.
        let body = json!({
        "messages": [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "What is the capital of France?"}
        ]
        });
        assert_eq!(
            extract_request_text(&body),
            "be concise\nWhat is the capital of France?"
        );
    }

    #[test]
    fn sanitize_url_redacts_redis_password() {
        let redacted = crate::security::sanitize_url("redis://:s3cr3t@h:6379");
        assert!(
            !redacted.contains("s3cr3t"),
            "password must not appear in redacted url, got {redacted}"
        );
        assert!(
            redacted.contains("h:6379"),
            "host/port preserved: {redacted}"
        );
    }

    #[test]
    fn tools_are_exact_constraints_not_embedding_input() {
        let base = json!({"messages": [{"role": "user", "content": "get weather"}]});
        let with_tools = json!({
            "messages": [{"role": "user", "content": "get weather"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}]
        });
        let text_base = extract_request_text(&base);
        let text_tools = extract_request_text(&with_tools);
        assert_eq!(
            text_base, text_tools,
            "tool schemas do not belong in the approximate text input"
        );
        assert_ne!(scope(&base), scope(&with_tools));
    }

    fn scope(body: &Value) -> String {
        semantic_scope_for(body, "gpt-4o", "embed-v1", "https://embed.example/v1").unwrap()
    }

    #[test]
    fn compatibility_allows_paraphrases_but_preserves_all_other_fields() {
        let base = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "cache": {"insert": true}
        });
        let paraphrase = json!({
            "messages": [{"role": "user", "content": "hi there"}],
            "cache": {"use_cache": true, "no-store": true}
        });
        assert_eq!(scope(&base), scope(&paraphrase));
        for (field, value) in [
            ("temperature", json!(0.2)),
            ("top_p", json!(0.7)),
            ("max_tokens", json!(100)),
            ("seed", json!(42)),
            ("stop", json!(["END"])),
            ("n", json!(2)),
            ("tools", json!([])),
            ("functions", json!([{"name": "weather"}])),
            ("tool_choice", json!("required")),
            ("response_format", json!({"type": "json_object"})),
            ("vendor_future_option", json!({"enabled": true})),
        ] {
            let mut different = base.clone();
            different[field] = value;
            assert_ne!(scope(&base), scope(&different), "{field} must remain exact");
        }
    }

    #[test]
    fn compatibility_preserves_roles_instructions_and_content_part_identity() {
        let base = json!({"messages": [
            {"role": "system", "content": "Only answer in French"},
            {"role": "user", "content": [
                {"type": "text", "text": "describe", "id": "part-1"},
                {"type": "image_url", "image_url": {"url": "https://images.example/a"}}
            ]}
        ]});
        let mut paraphrase = base.clone();
        paraphrase["messages"][1]["content"][0]["text"] = json!("tell me about this");
        assert_eq!(scope(&base), scope(&paraphrase));
        for (pointer, replacement) in [
            ("/messages/0/content", json!("Only answer in English")),
            ("/messages/1/role", json!("assistant")),
            ("/messages/1/content/0/id", json!("part-2")),
            ("/messages/1/content/0/type", json!("input_text")),
            (
                "/messages/1/content/1/image_url/url",
                json!("https://images.example/b"),
            ),
        ] {
            let mut different = base.clone();
            *different.pointer_mut(pointer).unwrap() = replacement;
            assert_ne!(
                scope(&base),
                scope(&different),
                "{pointer} must remain exact"
            );
        }
        let text = json!({"messages": [{"role": "user", "content": "a\nb"}]});
        let parts = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}
        ]}]});
        assert_eq!(extract_request_text(&text), extract_request_text(&parts));
        assert_ne!(scope(&text), scope(&parts));
    }

    #[test]
    fn compatibility_is_canonical_and_namespaces_embedding_identity() {
        let a: Value = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"a"}],"tools":[{"function":{"name":"f","parameters":{"b":2,"a":1}}}]}"#,
        ).unwrap();
        let b: Value = serde_json::from_str(
            r#"{"tools":[{"function":{"parameters":{"a":1,"b":2},"name":"f"}}],"messages":[{"content":"b","role":"user"}]}"#,
        ).unwrap();
        assert_eq!(scope(&a), scope(&b));
        for (model, embedding, endpoint) in [
            ("other-chat", "embed-v1", "https://embed.example/v1"),
            ("gpt-4o", "embed-v2", "https://embed.example/v1"),
            ("gpt-4o", "embed-v1", "https://other-embed.example/v1"),
        ] {
            assert_ne!(
                Some(scope(&a)),
                semantic_scope_for(&a, model, embedding, endpoint)
            );
        }
    }

    #[tokio::test]
    async fn zero_threshold_accepts_orthogonal_vectors_but_not_invalid_vectors() {
        let cache = SemanticCache::new(10, 0.0, Duration::from_secs(60));
        cache
            .store(
                vec![1.0, 0.0],
                String::new(),
                json!({"id": "valid"}),
                "t",
                "m",
            )
            .await;
        assert!(cache.search(&[0.0, 1.0], "t", "m").await.is_some());
        for invalid in [
            vec![1.0],
            vec![0.0, 0.0],
            vec![f32::NAN, 1.0],
            vec![f32::INFINITY, 1.0],
        ] {
            assert!(cache.search(&invalid, "t", "m").await.is_none());
        }
        cache
            .store(vec![f32::NAN, 1.0], String::new(), json!({}), "t", "m")
            .await;
        cache
            .store(
                vec![1.0; MAX_EMBEDDING_DIMENSIONS + 1],
                String::new(),
                json!({}),
                "t",
                "m",
            )
            .await;
        cache
            .store(
                vec![1.0, 0.0],
                String::new(),
                json!("x".repeat(super::super::MAX_VALUE_BYTES)),
                "t",
                "m",
            )
            .await;
        assert_eq!(
            cache.len().await,
            1,
            "invalid/oversized entries must not consume capacity"
        );
    }

    #[test]
    fn extract_text_ignores_null_shaping_fields() {
        let a = json!({"messages": [{"role": "user", "content": "hi"}], "tools": null});
        let b = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(extract_request_text(&a), extract_request_text(&b));
    }

    #[test]
    fn extract_text_object_content_yields_nothing() {
        let body = json!({"messages": [{"role": "user", "content": {"weird": "shape"}}]});
        assert_eq!(extract_request_text(&body), "");
    }

    #[tokio::test]
    async fn ttl_zero_means_never_expire_in_memory() {
        let cache = SemanticCache::new(10, 0.95, Duration::ZERO);
        let emb = vec![1.0_f32, 0.0];
        cache
            .store(
                emb.clone(),
                "s".to_string(),
                json!({"id": "x"}),
                "t",
                "gpt-4o",
            )
            .await;
        sleep(Duration::from_millis(20)).await;
        assert!(
            cache.search(&emb, "t", "gpt-4o").await.is_some(),
            "ttl=0 means never expire"
        );
        cache.evict_expired().await;
        assert_eq!(cache.len().await, 1, "ttl=0 entries survive eviction");
    }

    #[tokio::test]
    async fn store_then_search_by_identical_embedding() {
        let cache = SemanticCache::new(100, 0.95, Duration::from_secs(60));
        let emb = vec![0.1_f32, 0.2, 0.3];
        cache
            .store(
                emb.clone(),
                "hello".to_string(),
                json!({"id": "x"}),
                "tenant-a",
                "gpt-4o",
            )
            .await;
        let hit = cache.search(&emb, "tenant-a", "gpt-4o").await.expect("hit");
        assert_eq!(hit["id"], "x");
    }

    #[tokio::test]
    async fn search_miss_below_threshold_returns_none() {
        let cache = SemanticCache::new(100, 0.95, Duration::from_secs(60));
        let stored = vec![1.0_f32, 0.0, 0.0];
        cache
            .store(
                stored.clone(),
                "hello".to_string(),
                json!({"id": "x"}),
                "tenant-a",
                "gpt-4o",
            )
            .await;
        let query = vec![0.0_f32, 1.0, 0.0];
        assert!(cache.search(&query, "tenant-a", "gpt-4o").await.is_none());
    }

    #[tokio::test]
    async fn search_returns_highest_similarity_above_threshold() {
        let cache = SemanticCache::new(100, 0.7, Duration::from_secs(60));
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.9_f32, 0.4, 0.0];
        cache
            .store(
                a.clone(),
                "a".to_string(),
                json!({"id": "a"}),
                "t",
                "gpt-4o",
            )
            .await;
        cache
            .store(
                b.clone(),
                "b".to_string(),
                json!({"id": "b"}),
                "t",
                "gpt-4o",
            )
            .await;
        let query = vec![0.95_f32, 0.3, 0.0];
        let hit = cache.search(&query, "t", "gpt-4o").await.expect("hit");
        assert_eq!(hit["id"], "b", "should pick the closest entry");
    }

    #[tokio::test]
    async fn search_is_scoped_per_api_key_hash() {
        let cache = SemanticCache::new(100, 0.95, Duration::from_secs(60));
        let emb = vec![1.0_f32, 0.0];
        cache
            .store(
                emb.clone(),
                "s".to_string(),
                json!({"id": "tenant-a-only"}),
                "tenant-a",
                "gpt-4o",
            )
            .await;
        assert!(
            cache.search(&emb, "tenant-b", "gpt-4o").await.is_none(),
            "tenant-b must not see tenant-a's entry"
        );
        assert!(cache.search(&emb, "tenant-a", "gpt-4o").await.is_some());
    }

    #[tokio::test]
    async fn search_is_scoped_per_model() {
        let cache = SemanticCache::new(100, 0.95, Duration::from_secs(60));
        let emb = vec![1.0_f32, 0.0];
        cache
            .store(
                emb.clone(),
                "s".to_string(),
                json!({"id": "gpt-4o-only"}),
                "tenant-a",
                "gpt-4o",
            )
            .await;
        assert!(
            cache.search(&emb, "tenant-a", "claude-3").await.is_none(),
            "claude-3 query must not match a gpt-4o entry"
        );
        assert!(cache.search(&emb, "tenant-a", "gpt-4o").await.is_some());
    }

    #[tokio::test]
    async fn expired_entries_are_skipped_by_search() {
        let cache = SemanticCache::new(100, 0.95, Duration::from_millis(30));
        let emb = vec![1.0_f32, 0.0];
        cache
            .store(
                emb.clone(),
                "s".to_string(),
                json!({"id": "x"}),
                "t",
                "gpt-4o",
            )
            .await;
        sleep(Duration::from_millis(60)).await;
        assert!(cache.search(&emb, "t", "gpt-4o").await.is_none());
    }

    #[tokio::test]
    async fn search_rechecks_expiry_after_waiting_for_the_lock() {
        let cache = Arc::new(SemanticCache::new(10, 0.95, Duration::from_millis(30)));
        cache
            .store(vec![1.0, 0.0], String::new(), json!({}), "t", "m")
            .await;
        let guard = cache.entries.write().await;
        let waiting_cache = cache.clone();
        let lookup = tokio::spawn(async move { waiting_cache.search(&[1.0, 0.0], "t", "m").await });
        tokio::task::yield_now().await;
        sleep(Duration::from_millis(60)).await;
        drop(guard);
        assert!(
            lookup.await.unwrap().is_none(),
            "time spent waiting must count toward expiry"
        );
    }

    #[tokio::test]
    async fn evict_expired_drops_stale_entries() {
        let cache = SemanticCache::new(100, 0.95, Duration::from_millis(30));
        cache
            .store(vec![1.0], "s".to_string(), json!({}), "t", "gpt-4o")
            .await;
        assert_eq!(cache.len().await, 1);
        sleep(Duration::from_millis(60)).await;
        cache.evict_expired().await;
        assert_eq!(cache.len().await, 0);
    }

    #[tokio::test]
    async fn max_entries_zero_disables_caching() {
        let cache = SemanticCache::new(0, 0.95, Duration::from_secs(60));
        let emb = vec![1.0_f32, 0.0];
        cache
            .store(
                emb.clone(),
                "s".to_string(),
                json!({"id": "x"}),
                "t",
                "gpt-4o",
            )
            .await;
        assert_eq!(cache.len().await, 0, "capacity 0 must not retain entries");
        assert!(
            cache.search(&emb, "t", "gpt-4o").await.is_none(),
            "capacity 0 must not serve hits"
        );
    }

    #[tokio::test]
    async fn max_entries_zero_disables_external_tier_too() {
        let server = httpmock::MockServer::start();
        let external = server.mock(|_, then| {
            then.status(200);
        });
        let config = crate::config::types::SemanticCacheConfig {
            enabled: true,
            embedding_model: "test-embed".to_string(),
            qdrant_url: Some(server.base_url()),
            ..Default::default()
        };
        let cache = build_semantic_store(&config, 0.0, 0, Duration::ZERO);
        cache
            .store(vec![1.0], String::new(), json!({}), "t", "m")
            .await;
        assert!(cache.search(&[1.0], "t", "m").await.is_none());
        assert_eq!(cache.len().await, 0);
        external.assert_hits(0);
    }

    #[tokio::test]
    async fn store_evicts_oldest_at_capacity() {
        // Use 2D unit vectors so each entry is distinguishable by direction
        // (cosine similarity of length-1 vectors is always ±1.0, which would
        // make every query match every entry).
        let cache = SemanticCache::new(2, 0.95, Duration::from_secs(60));
        cache
            .store(
                vec![1.0, 0.0],
                "a".to_string(),
                json!({"id": "a"}),
                "t",
                "gpt-4o",
            )
            .await;
        cache
            .store(
                vec![0.0, 1.0],
                "b".to_string(),
                json!({"id": "b"}),
                "t",
                "gpt-4o",
            )
            .await;
        assert_eq!(cache.len().await, 2);
        // 'c' is the opposite direction of 'a', so after FIFO evicts 'a',
        // a query for 'a' finds no match.
        cache
            .store(
                vec![-1.0, 0.0],
                "c".to_string(),
                json!({"id": "c"}),
                "t",
                "gpt-4o",
            )
            .await;
        assert_eq!(cache.len().await, 2, "capacity respected");
        let hit = cache.search(&[1.0, 0.0], "t", "gpt-4o").await;
        assert!(
            hit.is_none(),
            "FIFO should have evicted entry 'a' first (only orthogonal/opposite entries remain)"
        );
        assert!(cache.search(&[-1.0, 0.0], "t", "gpt-4o").await.is_some());
    }

    #[tokio::test]
    async fn empty_query_embedding_returns_none() {
        let cache: Arc<SemanticCache> =
            Arc::new(SemanticCache::new(10, 0.95, Duration::from_secs(60)));
        cache
            .store(
                vec![1.0],
                "a".to_string(),
                json!({"id": "a"}),
                "t",
                "gpt-4o",
            )
            .await;
        assert!(cache.search(&[], "t", "gpt-4o").await.is_none());
    }

    // -----------------------------------------------------------------
    // DualSemanticCache: in-memory primary + external fallback wiring.
    // -----------------------------------------------------------------

    /// A test-double external store that records whether `search`/`store`
    /// were consulted, so the dual-cache delegation can be asserted without a
    /// live Redis/Qdrant.
    struct RecordingFallback {
        search_called: std::sync::Mutex<bool>,
        store_called: std::sync::Mutex<bool>,
        response: Option<Value>,
    }

    #[async_trait]
    impl SemanticCacheStore for RecordingFallback {
        async fn search(&self, _q: &[f32], _ak: &str, _m: &str) -> Option<Value> {
            *self.search_called.lock().unwrap() = true;
            self.response.clone()
        }
        async fn store(&self, _emb: Vec<f32>, _req: String, _resp: Value, _ak: &str, _m: &str) {
            *self.store_called.lock().unwrap() = true;
        }
        async fn evict_expired(&self) {}
    }

    #[tokio::test]
    async fn dual_primary_hit_does_not_consult_fallback() {
        let primary = Arc::new(SemanticCache::new(10, 0.95, Duration::from_secs(60)));
        primary
            .store(
                vec![1.0, 0.0],
                "a".to_string(),
                json!({"id": "mem"}),
                "t",
                "gpt-4o",
            )
            .await;
        let fb = Arc::new(RecordingFallback {
            search_called: std::sync::Mutex::new(false),
            store_called: std::sync::Mutex::new(false),
            response: Some(json!({"id": "ext"})),
        });
        let dual = DualSemanticCache::new(primary, Some(fb.clone()));
        let hit = dual.search(&[1.0, 0.0], "t", "gpt-4o").await.expect("hit");
        assert_eq!(hit["id"], "mem", "primary hit must win");
        assert!(
            !*fb.search_called.lock().unwrap(),
            "fallback must not be consulted on a primary hit"
        );
    }

    #[tokio::test]
    async fn dual_primary_miss_falls_back_to_external() {
        let primary = Arc::new(SemanticCache::new(10, 0.95, Duration::from_secs(60)));
        let fb = Arc::new(RecordingFallback {
            search_called: std::sync::Mutex::new(false),
            store_called: std::sync::Mutex::new(false),
            response: Some(json!({"id": "ext"})),
        });
        let dual = DualSemanticCache::new(primary, Some(fb.clone()));
        let hit = dual.search(&[1.0, 0.0], "t", "gpt-4o").await.expect("hit");
        assert_eq!(hit["id"], "ext", "miss in memory must consult the fallback");
        assert!(*fb.search_called.lock().unwrap());
    }

    #[tokio::test]
    async fn dual_store_writes_to_both_tiers() {
        let primary = Arc::new(SemanticCache::new(10, 0.95, Duration::from_secs(60)));
        let fb = Arc::new(RecordingFallback {
            search_called: std::sync::Mutex::new(false),
            store_called: std::sync::Mutex::new(false),
            response: None,
        });
        let dual = DualSemanticCache::new(primary.clone(), Some(fb.clone()));
        dual.store(
            vec![1.0, 0.0],
            "req".to_string(),
            json!({"id": "both"}),
            "t",
            "gpt-4o",
        )
        .await;
        assert!(
            *fb.store_called.lock().unwrap(),
            "fallback must receive the store"
        );
        assert_eq!(primary.len().await, 1, "primary must also store");
    }

    #[tokio::test]
    async fn dual_without_fallback_uses_only_memory() {
        let primary = Arc::new(SemanticCache::new(10, 0.95, Duration::from_secs(60)));
        let dual = DualSemanticCache::new(primary, None);
        // Nothing stored yet → no hit, no panic.
        assert!(dual.search(&[1.0, 0.0], "t", "gpt-4o").await.is_none());
        dual.store(
            vec![1.0, 0.0],
            "req".to_string(),
            json!({"id": "x"}),
            "t",
            "gpt-4o",
        )
        .await;
        assert_eq!(
            dual.search(&[1.0, 0.0], "t", "gpt-4o").await.expect("hit")["id"],
            "x"
        );
    }

    #[tokio::test]
    async fn build_semantic_store_uses_in_memory_when_no_external_configured() {
        let cfg = crate::config::types::SemanticCacheConfig {
            enabled: true,
            embedding_model: "text-embedding-3-small".to_string(),
            similarity_threshold: Some(0.9),
            ttl_secs: Some(60),
            ..Default::default()
        };
        let store = build_semantic_store(&cfg, 0.9, 100, Duration::from_secs(60));
        store
            .store(
                vec![1.0, 0.0],
                "a".to_string(),
                json!({"id": "a"}),
                "t",
                "gpt-4o",
            )
            .await;
        assert_eq!(store.len().await, 1);
        assert_eq!(
            store.search(&[1.0, 0.0], "t", "gpt-4o").await.expect("hit")["id"],
            "a"
        );
    }

    #[tokio::test]
    async fn build_semantic_store_picks_qdrant_when_url_set() {
        let cfg = crate::config::types::SemanticCacheConfig {
            enabled: true,
            embedding_model: "text-embedding-3-small".to_string(),
            similarity_threshold: Some(0.9),
            ttl_secs: Some(60),
            qdrant_url: Some("http://127.0.0.1:1".to_string()),
            qdrant_collection: Some("oxllm_cache".to_string()),
            qdrant_dimension: Some(4),
            ..Default::default()
        };
        let store = build_semantic_store(&cfg, 0.9, 100, Duration::from_secs(60));
        // A Qdrant-backed dual store: in-memory primary still absorbs stores.
        store
            .store(
                vec![1.0, 0.0, 0.0, 0.0],
                "a".to_string(),
                json!({"id": "a"}),
                "t",
                "gpt-4o",
            )
            .await;
        // Primary (in-memory) reflects the entry even though the Qdrant
        // upsert to an unreachable host failed best-effort.
        assert_eq!(store.len().await, 1);
        assert_eq!(
            store
                .search(&[1.0, 0.0, 0.0, 0.0], "t", "gpt-4o")
                .await
                .expect("hit")["id"],
            "a"
        );
    }
}
