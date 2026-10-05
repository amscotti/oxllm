//! `CacheHook` — opt-in response caching keyed by a hash of the request.
//!
//! Pre-call: if the request carries `cache: {"insert": true}` (or any object
//! implying "use cache"), look up the cache key. On a hit, short-circuit with
//! the cached response. Post-success: store the response under the cache key
//! with a TTL.
//!
//! When an optional [`SemanticCacheRuntime`] is attached, an exact-match miss
//! falls back to embedding-based similarity search: the request text is
//! embedded, scanned against the in-memory vector store, and the highest-
//! similarity entry above the threshold short-circuits the upstream call. On
//! the post-success path, the stashed embedding + new response are stored so
//! subsequent similar requests hit.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::{Cache, CallContext, Hook, HookAction, ResponseSummary};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::cache::semantic::{
    MAX_EMBEDDING_DIMENSIONS, SemanticCacheStore, cacheable_semantic_entry, canonicalize_json,
    semantic_scope_for, valid_embedding,
};

/// Timeout for the embedding HTTP fetch. Failures (timeout, non-200, network
/// error) MUST NOT block the request — they log a warning and fall through to
/// the upstream call.
const EMBEDDING_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Upper bound on the embedding response body read into memory. The embedding
/// endpoint is config-trusted, but a compromised/misconfigured provider could
/// otherwise stream a multi-GB body that `reqwest` would buffer in full before
/// the `data[0].embedding` extraction. Realistic embedding payloads are a few
/// hundred KB even for large models, so 8 MiB is a generous ceiling.
const MAX_EMBEDDING_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// A hook that implements opt-in response caching. Requests opt in by setting
/// `cache: {"insert": true}` (LiteLLM-style) on the top-level request body.
/// The cache key is derived from a hash of the model + messages + relevant
/// params.
pub struct CacheHook {
    /// Backing cache (in-memory or Redis).
    pub cache: Arc<dyn Cache>,
    /// TTL for cached responses.
    pub ttl: Duration,
    /// Optional semantic-cache runtime. When `Some`, an exact-match miss
    /// falls back to embedding-based similarity search.
    pub semantic: Option<Arc<SemanticCacheRuntime>>,
}

/// Runtime state needed to perform embedding-based semantic lookup + store.
///
/// Kept separate from the vector store so the embedding-fetch client/config
/// can be swapped or mocked independently of the vector store.
pub struct SemanticCacheRuntime {
    /// The vector store (in-memory, Redis-backed `DualSemanticCache`, or
    /// Qdrant-backed — see `crate::cache::semantic::build_semantic_store`).
    pub cache: Arc<dyn SemanticCacheStore>,
    /// Model name used for embedding (e.g. `"text-embedding-3-small"`).
    pub embedding_model: String,
    /// Fully-qualified base URL for the embeddings endpoint, e.g.
    /// `https://api.openai.com/v1`. The runtime POSTs to `{api_base}/embeddings`.
    pub api_base: String,
    /// Embedding API key. Resolved from config — never from the inbound
    /// request's auth — so semantic cache tenants cannot impersonate each
    /// other's embedding quotas.
    pub api_key: Option<String>,
    /// HTTP client used for the embedding fetch.
    pub http_client: reqwest::Client,
    /// Per-request stash of embeddings computed in `pre_call` and consumed in
    /// `post_success`/`post_failure`. Cleaned up on both post paths so it
    /// cannot grow unboundedly under normal flow. Entries additionally carry
    /// a TTL and the map has a size cap: if the handler future is dropped
    /// between `pre_call` and the post hooks (client disconnect,
    /// cancellation), the stashed embedding cannot leak forever — it is
    /// evicted on the next access once expired, or in bulk when the cap is
    /// exceeded.
    pending: Mutex<HashMap<String, PendingEmbedding>>,
    /// Count of semantic stores performed through this runtime, used to drive
    /// opportunistic (every Nth store) rather than per-store eviction.
    store_count: AtomicU64,
}

/// A stashed embedding plus the instant it was stashed (for TTL eviction).
/// The `guard` is the weak side of an `Arc<()>` shared with the in-flight
/// request's [`CallContext`](oxllm_core::CallContext): while the handler
/// future is alive the guard can be upgraded (the entry is "in-flight"); once
/// the handler is dropped without reaching the post hooks (cancellation), the
/// guard can no longer be upgraded (the entry is "abandoned") and cap-pressure
/// eviction may reclaim it.
struct PendingEmbedding {
    embedding: Vec<f32>,
    scope: String,
    request_key: String,
    stashed_at: std::time::Instant,
    guard: std::sync::Weak<()>,
}

/// Maximum number of stashed embeddings to retain. When exceeded, abandoned
/// entries (whose handler future was dropped) are dropped first; only when no
/// abandoned entries remain does the oldest half of in-flight entries get
/// dropped, bounding the leak while preferring to keep live consumers' data.
const PENDING_STASH_CAP: usize = 1000;

/// Maximum age of a stashed embedding before it is considered abandoned.
/// Chosen well above the default request timeout (600s) so a legitimately
/// in-flight upstream call is never evicted; only requests whose handler
/// future was dropped between `pre_call` and the post hooks linger this long.
const PENDING_STASH_TTL: Duration = Duration::from_secs(1200);

/// Run `evict_expired()` on the vector store once every N semantic stores
/// instead of on every store. The store's own capacity-based eviction still
/// fires at insert time, so paying the O(n) TTL sweep on every cache-miss
/// store would dominate the request path under a sustained miss workload.
const EVICT_EVERY_N_STORES: u64 = 64;

/// Response-cache short-circuits by requested model (exact-match and
/// semantic hits share the series — both are cache hits).
pub const PROMPT_CACHE_HITS_TOTAL: &str = "oxllm_prompt_cache_hits_total";

impl SemanticCacheRuntime {
    /// Construct the runtime.
    pub fn new(
        cache: Arc<dyn SemanticCacheStore>,
        embedding_model: String,
        api_base: String,
        api_key: Option<String>,
        http_client: reqwest::Client,
    ) -> Self {
        Self {
            cache,
            embedding_model,
            api_base,
            api_key,
            http_client,
            pending: Mutex::new(HashMap::new()),
            store_count: AtomicU64::new(0),
        }
    }

    /// Stash `embedding` under `request_id` for later retrieval in
    /// `post_success`. Returns a strong guard `Arc<()>` that the caller must
    /// hold for the duration of the in-flight request (typically on the
    /// [`CallContext`]); while the strong guard is alive the entry is
    /// considered in-flight and cap-pressure eviction will prefer reclaiming
    /// abandoned entries (whose strong side has been dropped) over this one.
    async fn stash_embedding(
        &self,
        request_id: &str,
        embedding: Vec<f32>,
        scope: String,
        request_key: String,
    ) -> std::sync::Arc<()> {
        let mut guard = self.pending.lock().await;
        let now = std::time::Instant::now();
        // Evict expired entries (handler future dropped between pre_call and
        // the post hooks) so the map cannot leak stale embeddings.
        guard.retain(|_, entry| now.duration_since(entry.stashed_at) < PENDING_STASH_TTL);
        if guard.len() >= PENDING_STASH_CAP {
            // Prefer reclaiming abandoned entries (whose strong guard has been
            // dropped) so cap pressure never evicts a live in-flight embedding
            // when abandoned ones exist to reclaim. Only when every entry is
            // in-flight do we fall back to dropping the oldest half by stash
            // time (bounding memory under sustained concurrency beyond cap).
            let abandoned: Vec<String> = guard
                .iter()
                .filter(|(_, e)| e.guard.upgrade().is_none())
                .map(|(k, _)| k.clone())
                .collect();
            for key in &abandoned {
                guard.remove(key);
            }
            if guard.len() >= PENDING_STASH_CAP {
                let mut by_age: Vec<(String, std::time::Instant)> = guard
                    .iter()
                    .map(|(k, v)| (k.clone(), v.stashed_at))
                    .collect();
                by_age.sort_by_key(|(_, stashed_at)| *stashed_at);
                let drop_count = guard.len() / 2;
                for (key, _) in by_age.into_iter().take(drop_count) {
                    guard.remove(&key);
                }
            }
        }
        let strong = std::sync::Arc::new(());
        guard.insert(
            request_id.to_string(),
            PendingEmbedding {
                embedding,
                scope,
                request_key,
                stashed_at: now,
                guard: std::sync::Arc::downgrade(&strong),
            },
        );
        strong
    }

    async fn pop_embedding(&self, request_id: &str) -> Option<PendingEmbedding> {
        let mut guard = self.pending.lock().await;
        guard.remove(request_id)
    }

    fn scope_for(&self, request: &Value, model: &str) -> Option<String> {
        semantic_scope_for(request, model, &self.embedding_model, &self.api_base)
    }

    /// Fetch the embedding for `text` from the configured provider. Returns
    /// `None` on any failure (timeout, network, non-200, malformed body) —
    /// callers fall through to the upstream call rather than blocking.
    async fn fetch_embedding(&self, text: &str) -> Option<Vec<f32>> {
        let url = format!("{}/embeddings", self.api_base.trim_end_matches('/'));
        let mut req = self
            .http_client
            .post(&url)
            .timeout(EMBEDDING_FETCH_TIMEOUT)
            .header("content-type", "application/json")
            .json(&serde_json::json!({
            "model": self.embedding_model,
            "input": text,
            }));
        if let Some(key) = &self.api_key {
            // OpenAI-style Bearer auth. Tolerate callers that already include
            // the scheme so we don't end up with `Bearer Bearer sk-…`.
            let header_value = if key.starts_with("Bearer ") {
                key.clone()
            } else {
                format!("Bearer {key}")
            };
            if let Ok(value) = reqwest::header::HeaderValue::from_str(&header_value) {
                req = req.header("authorization", value);
            }
        }
        let response = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                error = %e,
                model = %self.embedding_model,
                "semantic cache: embedding fetch failed; falling through to upstream"
                );
                return None;
            }
        };
        if !response.status().is_success() {
            tracing::warn!(
                status = response.status().as_u16(),
                "semantic cache: embedding fetch non-2xx; falling through"
            );
            return None;
        }
        let body: Value = match read_capped_json(response).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                error = %e,
                "semantic cache: embedding body parse failed; falling through"
                );
                return None;
            }
        };
        let embedding = body
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|first| first.get("embedding"))
            .and_then(|e| e.as_array())
            .filter(|arr| arr.len() <= MAX_EMBEDDING_DIMENSIONS)
            .and_then(|arr| {
                arr.iter()
                    .map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Option<Vec<f32>>>()
            })
            .filter(|embedding| valid_embedding(embedding));
        if embedding.is_none() {
            tracing::warn!(
                "semantic cache: embedding response had no valid vector; falling through"
            );
        }
        embedding
    }
}

/// Read an embedding response body as JSON, refusing payloads larger than
/// [`MAX_EMBEDDING_RESPONSE_BYTES`]. The body is streamed chunk-by-chunk and
/// reading is aborted as soon as the cap is exceeded, so a misconfigured
/// provider cannot drive gateway OOM by returning an oversized body.
async fn read_capped_json(response: reqwest::Response) -> Result<Value, String> {
    use futures_util::StreamExt;
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        if buf.len() + chunk.len() > MAX_EMBEDDING_RESPONSE_BYTES {
            return Err(format!(
                "embedding response exceeds {} byte cap",
                MAX_EMBEDDING_RESPONSE_BYTES
            ));
        }
        buf.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&buf).map_err(|e| e.to_string())
}

impl CacheHook {
    /// Create with the given cache and TTL (no semantic cache).
    pub fn new(cache: Arc<dyn Cache>, ttl: Duration) -> Self {
        Self {
            cache,
            ttl,
            semantic: None,
        }
    }

    /// Attach a semantic-cache runtime, enabling embedding-based fallback on
    /// exact-match miss.
    pub fn with_semantic(mut self, runtime: Arc<SemanticCacheRuntime>) -> Self {
        self.semantic = Some(runtime);
        self
    }
}

/// Returns true if the request opts into caching. LiteLLM uses the convention
/// `cache: {"insert": true}` (or `cache: {"use_cache": true}`). Each key is
/// evaluated independently and the results OR-ed, so a request carrying
/// `cache: {"insert": false, "use_cache": true}` (a reasonable "don't force a
/// write, but do consult the cache" combination) still opts in.
pub fn request_wants_cache(request: &Value) -> bool {
    let Some(Value::Object(map)) = request.get("cache") else {
        return false;
    };
    let insert = map.get("insert").and_then(|v| v.as_bool()).unwrap_or(false);
    let use_cache = map
        .get("use_cache")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    insert || use_cache
}

/// Read a boolean flag from the request's `cache:` object, defaulting to
/// `false` when the flag or the `cache:` object is absent.
fn cache_flag(request: &Value, key: &str) -> bool {
    request
        .get("cache")
        .and_then(|c| c.get(key))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Returns true when the cache **read** path (exact-match / semantic lookup)
/// may run for this request: the caller opted into caching AND did not send
/// the per-request `cache: {"no-cache": true}` directive that skips the
/// lookup (LiteLLM `caching_handler.py:184`).
pub fn request_allows_cache_lookup(request: &Value) -> bool {
    request_wants_cache(request) && !cache_flag(request, "no-cache")
}

/// Returns true when the cache **write** path (`post_success` store) may run
/// for this request: the caller opted into caching AND did not send the
/// per-request `cache: {"no-store": true}` directive that skips the store
/// (LiteLLM `caching_handler.py:1045`).
pub fn request_allows_cache_store(request: &Value) -> bool {
    request_wants_cache(request) && !cache_flag(request, "no-store")
}

/// Enforce the per-key `allowed_cache_controls` allowlist against every key
/// in the request's `cache:` object, mirroring LiteLLM's
/// `_PROXY_CacheControlCheck`. A non-empty `Some` allowlist rejects any
/// `cache:` key not on it; `None` means unrestricted, an empty list allows no
/// explicit controls. Master
/// keys bypass the check entirely. Returns `Ok(())` when the request is
/// allowed (or no allowlist is configured), `Err(BadRequest)` otherwise.
fn check_allowed_cache_controls(ctx: &CallContext) -> Result<(), oxllm_core::Error> {
    if ctx.is_master {
        return Ok(());
    }
    let Some(allowed) = ctx.allowed_cache_controls.as_ref() else {
        return Ok(());
    };
    let Some(Value::Object(cache)) = ctx.request_body.get("cache") else {
        return Ok(());
    };
    for key in cache.keys() {
        if !allowed.iter().any(|a| a == key) {
            return Err(oxllm_core::Error::BadRequest(format!(
                "cache control '{key}' is not allowed for this API key"
            )));
        }
    }
    Ok(())
}

/// Compute a stable cache key from the complete request body, excluding only
/// cache-control fields that do not affect model output. The `stream` field is
/// intentionally included: streaming and non-streaming variants of the same
/// request produce different response shapes (chunk vs. full object), so they
/// must not share a cache entry.
///
/// The api key hash scopes entries per principal so multi-tenant deployments
/// cannot share or steal each other's cached responses.
pub fn cache_key_for(request: &Value, api_key_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(api_key_hash.as_bytes());
    hasher.update(b"\x00");
    let mut cache_input = request.clone();
    if let Some(object) = cache_input.as_object_mut() {
        object.remove("cache");
    }
    canonicalize_json(&mut cache_input);
    if let Ok(serialized) = serde_json::to_vec(&cache_input) {
        hasher.update(serialized);
    }
    let digest = hasher.finalize();
    format!("resp:{:x}", digest)
}

fn context_cache_key(ctx: &CallContext) -> String {
    let key = cache_key_for(&ctx.request_body, &ctx.api_key_hash);
    match ctx.cache_namespace.as_deref() {
        Some(namespace) => format!(
            "{key}:{}",
            hex::encode(Sha256::digest(namespace.as_bytes()))
        ),
        None => key,
    }
}

fn cache_entry(body: &Value, cost: Option<f64>) -> Value {
    serde_json::json!({"_oxllm_cache_version":1,"response":body,"cost":cost})
}

fn decode_cache_entry(mut value: Value) -> (Value, Option<f64>) {
    if value.get("_oxllm_cache_version").and_then(Value::as_u64) == Some(1)
        && value.get("response").is_some()
    {
        let cost = value
            .get("cost")
            .and_then(Value::as_f64)
            .filter(|cost| cost.is_finite() && *cost >= 0.0);
        (value["response"].take(), cost)
    } else {
        // Existing cache entries contain the response directly.
        (value, None)
    }
}

fn context_semantic_scope(semantic: &SemanticCacheRuntime, ctx: &CallContext) -> Option<String> {
    let scope = semantic.scope_for(&ctx.request_body, &ctx.model)?;
    Some(match ctx.cache_namespace.as_deref() {
        Some(namespace) => format!(
            "{scope}:{}",
            hex::encode(Sha256::digest(namespace.as_bytes()))
        ),
        None => scope,
    })
}

#[async_trait]
impl Hook for CacheHook {
    async fn pre_call(&self, ctx: &mut CallContext) -> oxllm_core::Result<HookAction> {
        if let Err(err) = check_allowed_cache_controls(ctx) {
            return Ok(HookAction::Reject(err));
        }
        if !request_allows_cache_lookup(&ctx.request_body) {
            return Ok(HookAction::Continue);
        }
        let key = context_cache_key(ctx);
        if let Some(raw) = self.cache.get(&key).await {
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                tracing::info!(
                request_id = %ctx.request_id,
                "response cache hit"
                );
                ::metrics::counter!(
                    PROMPT_CACHE_HITS_TOTAL,
                    "model" => crate::metrics::model_label(&ctx.model)
                )
                .increment(1);
                let (body, cost) = decode_cache_entry(value);
                ctx.cached_response_cost = cost;
                return Ok(HookAction::ShortCircuit(body));
            }
        }

        if let Some(semantic) = self.semantic.as_ref() {
            let is_streaming = ctx
                .request_body
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !is_streaming {
                let Some(scope) = context_semantic_scope(semantic, ctx) else {
                    return Ok(HookAction::Continue);
                };
                let text = crate::cache::semantic::extract_request_text(&ctx.request_body);
                if !text.trim().is_empty() {
                    if let Some(embedding) = semantic.fetch_embedding(&text).await {
                        if let Some(cached) = semantic
                            .cache
                            .search(&embedding, &ctx.api_key_hash, &scope)
                            .await
                        {
                            tracing::info!(
                                request_id = %ctx.request_id,
                                "semantic cache hit"
                            );
                            ::metrics::counter!(
                                PROMPT_CACHE_HITS_TOTAL,
                                "model" => crate::metrics::model_label(&ctx.model)
                            )
                            .increment(1);
                            let (body, cost) = decode_cache_entry(cached);
                            ctx.cached_response_cost = cost;
                            return Ok(HookAction::ShortCircuit(body));
                        }
                        if request_allows_cache_store(&ctx.request_body) {
                            ctx.pending_cache_guard = Some(
                                semantic
                                    .stash_embedding(&ctx.request_id, embedding, scope, key)
                                    .await,
                            );
                        }
                    }
                }
            }
        }

        Ok(HookAction::Continue)
    }

    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        // Consume the stash before every early return and before awaiting any
        // exact-cache write. no-store, hook mutations, missing bodies, cache
        // hits, and cancelled writes must not leave request embeddings behind.
        let pending = match self.semantic.as_ref() {
            Some(semantic) => semantic.pop_embedding(&ctx.request_id).await,
            None => None,
        };
        if ctx.cache_hit {
            return;
        }
        let Some(body) = resp.body.as_ref() else {
            return;
        };
        if !request_allows_cache_store(&ctx.request_body) {
            return;
        }
        let key = context_cache_key(ctx);
        let entry = cache_entry(body, resp.cost);
        let Ok(serialized) = serde_json::to_string(&entry) else {
            return;
        };
        self.cache.set(&key, serialized, Some(self.ttl)).await;

        if let Some(semantic) = self.semantic.as_ref() {
            if let Some(pending) = pending {
                // A later hook may have changed the request after it was
                // embedded. Do not pair that old vector with a new response.
                if pending.request_key != key
                    || context_semantic_scope(semantic, ctx).as_deref()
                        != Some(pending.scope.as_str())
                    || !cacheable_semantic_entry(&pending.embedding, body)
                {
                    return;
                }
                semantic
                    .cache
                    .store(
                        pending.embedding,
                        String::new(), // Store backends deliberately do not retain request text.
                        entry,
                        &ctx.api_key_hash,
                        &pending.scope,
                    )
                    .await;
                // Opportunistic eviction: amortise the O(n) sweep across many
                // stores instead of paying it on every cache-miss request. The
                // store's own capacity eviction still bounds the entry count
                // at insert time.
                let count = semantic.store_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count % EVICT_EVERY_N_STORES == 0 {
                    semantic.cache.evict_expired().await;
                }
            }
        }
    }

    async fn post_failure(&self, ctx: &CallContext, _err: &oxllm_core::Error) {
        if let Some(semantic) = self.semantic.as_ref() {
            let _ = semantic.pop_embedding(&ctx.request_id).await;
        }
    }
}

/// Look up a cached response for the given request. Returns `Some(cached_json)`
/// on a hit.
pub async fn lookup(cache: &dyn Cache, request: &Value, api_key_hash: &str) -> Option<Value> {
    if !request_allows_cache_lookup(request) {
        return None;
    }
    let key = cache_key_for(request, api_key_hash);
    let raw = cache.get(&key).await?;
    serde_json::from_str(&raw)
        .ok()
        .map(decode_cache_entry)
        .map(|(body, _)| body)
}

/// Store a response in the cache (only when caching is requested).
pub async fn store(
    cache: &dyn Cache,
    request: &Value,
    response: &Value,
    ttl: Duration,
    api_key_hash: &str,
) {
    if !request_allows_cache_store(request) {
        return;
    }
    let key = cache_key_for(request, api_key_hash);
    let Ok(serialized) = serde_json::to_string(response) else {
        return;
    };
    cache.set(&key, serialized, Some(ttl)).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use crate::cache::semantic::SemanticCache;
    use crate::config::types::SemanticCacheConfig;
    use httpmock::MockServer;
    use serde_json::json;

    fn cached_request(messages: &str) -> Value {
        json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": messages}],
        "cache": {"insert": true}
        })
    }

    fn uncached_request(messages: &str) -> Value {
        json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": messages}]
        })
    }

    #[tokio::test]
    async fn cache_preserves_original_cost_and_isolates_response_representations() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let mut ctx = CallContext::for_test("public-alias");
        ctx.request_body = Arc::new(cached_request("hello"));
        let body = json!({"id":"original","model":"unpriced-alias","choices":[]});
        hook.post_success(
            &ctx,
            &ResponseSummary {
                model: "unpriced-alias".into(),
                cost: Some(0.25),
                usage: None,
                body: Some(body.clone()),
                provider: "openai".into(),
            },
        )
        .await;
        match hook.pre_call(&mut ctx).await.unwrap() {
            HookAction::ShortCircuit(cached) => assert_eq!(cached, body),
            _ => panic!("cache hit expected"),
        }
        assert_eq!(ctx.cached_response_cost, Some(0.25));
        ctx.cache_namespace = Some("native-messages".into());
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Continue
        ));
    }

    fn http_client() -> reqwest::Client {
        reqwest::Client::builder()
            .timeout(EMBEDDING_FETCH_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("client")
    }

    fn semantic_runtime_with_server(
        server_url: String,
        threshold: f64,
    ) -> Arc<SemanticCacheRuntime> {
        let _ = SemanticCacheConfig::default();
        let cache = Arc::new(SemanticCache::new(100, threshold, Duration::from_secs(60)));
        Arc::new(SemanticCacheRuntime::new(
            cache,
            "text-embedding-3-small".to_string(),
            server_url,
            Some("sk-embed".to_string()),
            http_client(),
        ))
    }

    #[tokio::test]
    async fn lookup_returns_none_for_uncached_request() {
        let cache = InMemoryCache::default();
        let req = uncached_request("hi");
        assert!(lookup(&cache, &req, "abc").await.is_none());
    }

    #[tokio::test]
    async fn store_and_lookup_hit() {
        let cache = InMemoryCache::default();
        let req = cached_request("hello");
        let resp = json!({"id": "x", "choices": []});
        store(&cache, &req, &resp, Duration::from_secs(60), "abc").await;
        let hit = lookup(&cache, &req, "abc").await.expect("cache hit");
        assert_eq!(hit["id"], "x");
    }

    #[tokio::test]
    async fn miss_returns_none() {
        let cache = InMemoryCache::default();
        let req = cached_request("never stored");
        assert!(lookup(&cache, &req, "abc").await.is_none());
    }

    #[tokio::test]
    async fn different_messages_have_different_keys() {
        let a = cached_request("alpha");
        let b = cached_request("beta");
        assert_ne!(cache_key_for(&a, "abc"), cache_key_for(&b, "abc"));
    }

    #[test]
    fn different_embedding_inputs_have_different_keys() {
        let a = json!({"model": "text-embedding-3-small", "input": "alpha"});
        let b = json!({"model": "text-embedding-3-small", "input": "beta"});
        assert_ne!(cache_key_for(&a, "abc"), cache_key_for(&b, "abc"));
    }

    #[test]
    fn cache_controls_do_not_change_key() {
        let a = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "cache": {"insert": true},
            "stream": true
        });
        let b = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "cache": {"use_cache": true},
            "stream": true
        });
        assert_eq!(cache_key_for(&a, "abc"), cache_key_for(&b, "abc"));
    }

    #[test]
    fn stream_format_changes_key() {
        let a = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "cache": {"insert": true},
            "stream": true
        });
        let b = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "cache": {"insert": true},
            "stream": false
        });
        assert_ne!(cache_key_for(&a, "abc"), cache_key_for(&b, "abc"));
    }

    #[tokio::test]
    async fn different_tenants_have_different_keys() {
        let req = cached_request("same");
        assert_ne!(
            cache_key_for(&req, "tenant-a"),
            cache_key_for(&req, "tenant-b")
        );
    }

    #[tokio::test]
    async fn store_no_op_when_request_does_not_want_cache() {
        let cache = InMemoryCache::default();
        let req = uncached_request("hi");
        let resp = json!({"id": "x"});
        store(&cache, &req, &resp, Duration::from_secs(60), "abc").await;
        assert!(lookup(&cache, &req, "abc").await.is_none());
    }

    #[test]
    fn request_wants_cache_detects_insert() {
        assert!(request_wants_cache(&json!({"cache": {"insert": true}})));
        assert!(request_wants_cache(&json!({"cache": {"use_cache": true}})));
        assert!(!request_wants_cache(&json!({"cache": {"insert": false}})));
        assert!(!request_wants_cache(&json!({})));
    }

    /// LOW: only canonical JSON booleans opt into caching. Truthy non-boolean
    /// values (e.g. `"yes"`, `1`) are treated as not opting in — a
    /// conservative posture that prevents loosely-typed clients from
    /// triggering unexpected cache behaviour, diverging from LiteLLM's proxy
    /// layer (which coerces truthy values) only for non-canonical input.
    #[test]
    fn request_wants_cache_rejects_non_boolean_insert() {
        assert!(
            !request_wants_cache(&json!({"cache": {"insert": "true"}})),
            "string \"true\" must not opt in"
        );
        assert!(
            !request_wants_cache(&json!({"cache": {"insert": 1}})),
            "number 1 must not opt in"
        );
        assert!(
            !request_wants_cache(&json!({"cache": {"use_cache": "yes"}})),
            "string \"yes\" must not opt in"
        );
        // Sanity: a real boolean still opts in.
        assert!(request_wants_cache(&json!({"cache": {"insert": true}})));
    }

    #[test]
    fn request_wants_cache_ors_insert_and_use_cache() {
        assert!(
            request_wants_cache(&json!({"cache": {"insert": false, "use_cache": true}})),
            "insert:false must not mask a truthy use_cache"
        );
        assert!(
            request_wants_cache(&json!({"cache": {"insert": true, "use_cache": false}})),
            "use_cache:false must not mask a truthy insert"
        );
        assert!(
            !request_wants_cache(&json!({"cache": {"insert": false, "use_cache": false}})),
            "both false must not opt in"
        );
    }

    #[test]
    fn request_allows_cache_lookup_and_store_respect_directives() {
        assert!(request_allows_cache_lookup(
            &json!({"cache": {"insert": true}})
        ));
        assert!(request_allows_cache_store(
            &json!({"cache": {"insert": true}})
        ));

        assert!(
            !request_allows_cache_lookup(&json!({"cache": {"insert": true, "no-cache": true}})),
            "no-cache must skip the lookup path"
        );
        assert!(
            request_allows_cache_store(&json!({"cache": {"insert": true, "no-cache": true}})),
            "no-cache must NOT skip the store path"
        );

        assert!(
            request_allows_cache_lookup(&json!({"cache": {"insert": true, "no-store": true}})),
            "no-store must NOT skip the lookup path"
        );
        assert!(
            !request_allows_cache_store(&json!({"cache": {"insert": true, "no-store": true}})),
            "no-store must skip the store path"
        );

        assert!(
            !request_allows_cache_lookup(&json!({"cache": {"no-cache": true}})),
            "without opt-in, lookup is off even with no-cache"
        );
    }

    fn ctx_for(request: Value) -> CallContext {
        CallContext {
            request_body: std::sync::Arc::new(request),
            ..CallContext::for_test("gpt-4o")
        }
    }

    #[tokio::test]
    async fn pre_call_continues_when_cache_not_requested() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let mut ctx = ctx_for(uncached_request("hi"));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn pre_call_continues_on_miss() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let mut ctx = ctx_for(cached_request("miss"));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn pre_call_short_circuits_on_hit() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = cached_request("hello");
        let resp = json!({"id": "hit", "choices": []});
        store(cache.as_ref(), &req, &resp, Duration::from_secs(60), "abc").await;

        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let mut ctx = ctx_for(req);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        match action {
            HookAction::ShortCircuit(value) => assert_eq!(value["id"], "hit"),
            other => panic!("expected ShortCircuit, got {other:?}"),
        }
    }

    /// A key with `allowed_cache_controls: Some(["insert"])` must reject a
    /// request that sends a control not on the allowlist (e.g. `no-cache`),
    /// and allow one that only uses allowed controls.
    #[tokio::test]
    async fn pre_call_rejects_disallowed_cache_control() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60));

        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x"}],
        "cache": {"insert": true, "no-cache": true}
        });
        let mut ctx = ctx_for(req);
        ctx.allowed_cache_controls = Some(vec!["insert".to_string()]);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        match action {
            HookAction::Reject(oxllm_core::Error::BadRequest(msg)) => {
                assert!(
                    msg.contains("no-cache"),
                    "rejection must name the offending key"
                )
            }
            other => panic!("expected Reject(BadRequest), got {other:?}"),
        }
    }

    /// `allowed_cache_controls = None` (the default) means unrestricted: the
    /// request flows through normally.
    #[tokio::test]
    async fn pre_call_allows_all_when_no_allowlist_configured() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x"}],
        "cache": {"insert": true, "no-cache": true}
        });
        let mut ctx = ctx_for(req);
        ctx.allowed_cache_controls = None;
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "no allowlist configured -> request must not be rejected"
        );
    }

    /// Master keys bypass the allowlist even when it is configured.
    #[tokio::test]
    async fn pre_call_master_key_bypasses_allowlist() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x"}],
        "cache": {"insert": true, "no-cache": true}
        });
        let mut ctx = ctx_for(req);
        ctx.is_master = true;
        ctx.allowed_cache_controls = Some(vec!["insert".to_string()]);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "master key must bypass the cache-control allowlist"
        );
    }

    /// A short-circuiting exact hit increments
    /// `oxllm_prompt_cache_hits_total{model}` (unique model label keeps the
    /// global-registry assertion exact).
    #[tokio::test]
    #[serial_test::serial]
    async fn cache_hit_increments_prompt_cache_hits_metric() {
        // Install the Prometheus recorder BEFORE writing — macro writes go
        // to a no-op recorder until the first `render()`/`install()`.
        let _ = crate::metrics::render();
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = json!({
        "model": "cache-metric-model",
        "messages": [{"role": "user", "content": "metric-hit"}],
        "cache": {"insert": true}
        });
        store(
            cache.as_ref(),
            &req,
            &json!({"id": "hit", "choices": []}),
            Duration::from_secs(60),
            "abc",
        )
        .await;

        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let mut ctx = CallContext {
            request_body: std::sync::Arc::new(req),
            ..CallContext::for_test("cache-metric-model")
        };
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::ShortCircuit(_)));
        let rendered = crate::metrics::render();
        assert!(
            rendered.contains("oxllm_prompt_cache_hits_total{model=\"cache-metric-model\"} 1"),
            "the exact hit must increment the counter, got:\n{rendered}"
        );
    }

    #[tokio::test]
    async fn post_success_stores_response_for_next_hit() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = cached_request("fresh");
        let hook = CacheHook::new(cache.clone(), Duration::from_secs(60));
        let ctx = ctx_for(req.clone());
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: Some(json!({"id": "stored"})),
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;

        let mut ctx2 = ctx_for(req);
        let action = hook.pre_call(&mut ctx2).await.expect("ok");
        match action {
            HookAction::ShortCircuit(value) => assert_eq!(value["id"], "stored"),
            other => panic!("expected ShortCircuit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn post_success_skips_when_body_absent() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache.clone(), Duration::from_secs(60));
        let ctx = ctx_for(cached_request("no-body"));
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;
        let mut ctx2 = ctx_for(uncached_request("no-body"));
        let action = hook.pre_call(&mut ctx2).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn post_success_does_not_rewrite_cache_hits() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = cached_request("hit");
        store(
            cache.as_ref(),
            &req,
            &json!({"id": "original"}),
            Duration::from_secs(60),
            "abc",
        )
        .await;
        let hook = CacheHook::new(cache.clone(), Duration::from_secs(60));
        let mut ctx = ctx_for(req.clone());
        ctx.cache_hit = true;
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: Some(json!({"id": "replacement"})),
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;

        let cached = lookup(cache.as_ref(), &req, "abc")
            .await
            .expect("cache hit");
        assert_eq!(cached["id"], "original");
    }

    /// `cache: {"insert": true, "no-cache": true}` opts into caching (so the
    /// store path still runs) but must NOT short-circuit the lookup on
    /// `pre_call` even when the key is seeded — LiteLLM `caching_handler.py:184`.
    #[tokio::test]
    async fn no_cache_directive_skips_lookup_but_keeps_store() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "warm"}],
        "cache": {"insert": true, "no-cache": true}
        });
        store(
            cache.as_ref(),
            &req,
            &json!({"id": "seeded"}),
            Duration::from_secs(60),
            "abc",
        )
        .await;

        let hook = CacheHook::new(cache.clone(), Duration::from_secs(60));
        let mut ctx = ctx_for(req.clone());
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "no-cache must skip the lookup even when the key is seeded"
        );

        // post_success must still store (no-store is not set).
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: Some(json!({"id": "fresh"})),
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;

        // A subsequent request WITHOUT no-cache now hits on the fresh entry.
        let req2 = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "warm"}],
        "cache": {"insert": true}
        });
        let mut ctx2 = ctx_for(req2);
        let action = hook.pre_call(&mut ctx2).await.expect("ok");
        match action {
            HookAction::ShortCircuit(value) => assert_eq!(value["id"], "fresh"),
            other => panic!("expected ShortCircuit from stored entry, got {other:?}"),
        }
    }

    /// `cache: {"insert": true, "no-store": true}` must consult the cache on
    /// `pre_call` but must NOT write a new entry on `post_success` — LiteLLM
    /// `caching_handler.py:1045`.
    #[tokio::test]
    async fn no_store_directive_skips_store_but_keeps_lookup() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "consult-only"}],
        "cache": {"insert": true, "no-store": true}
        });
        let hook = CacheHook::new(cache.clone(), Duration::from_secs(60));

        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: Some(json!({"id": "not-stored"})),
            provider: "openai".to_string(),
        };
        let ctx = ctx_for(req.clone());
        hook.post_success(&ctx, &resp).await;

        assert!(
            lookup(cache.as_ref(), &req, "abc").await.is_none(),
            "no-store must not write a cache entry"
        );

        // pre_call on the same request still runs the lookup (which misses).
        let mut ctx2 = ctx_for(req);
        let action = hook.pre_call(&mut ctx2).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    // -----------------------------------------------------------------
    // Semantic cache hook integration tests (httpmock-backed).
    // -----------------------------------------------------------------

    fn semantic_response() -> ResponseSummary {
        ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: Some(json!({"id": "semantic-original"})),
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    async fn semantic_identical_vectors_cannot_cross_request_compatibility() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .json_body(json!({"data": [{"embedding": [1.0, 0.0]}]}));
        });
        let runtime = semantic_runtime_with_server(server.base_url(), 0.0);
        let hook = CacheHook::new(Arc::new(InMemoryCache::default()), Duration::from_secs(60))
            .with_semantic(runtime.clone());
        let mut initial = ctx_for(cached_request("initial question"));
        assert!(matches!(
            hook.pre_call(&mut initial).await.unwrap(),
            HookAction::Continue
        ));
        hook.post_success(&initial, &semantic_response()).await;

        let mut paraphrase = cached_request("similar question");
        paraphrase["cache"]["no-store"] = json!(true);
        let mut compatible = ctx_for(paraphrase.clone());
        assert!(matches!(
            hook.pre_call(&mut compatible).await.unwrap(),
            HookAction::ShortCircuit(_)
        ));

        let mut incompatible = Vec::new();
        for (field, value) in [
            ("temperature", json!(0.1)),
            ("max_tokens", json!(8)),
            ("seed", json!(99)),
            (
                "tools",
                json!([{"type": "function", "function": {"name": "f"}}]),
            ),
            ("response_format", json!({"type": "json_object"})),
            ("future_option", json!({"mode": "different"})),
        ] {
            let mut request = paraphrase.clone();
            request[field] = value;
            incompatible.push(request);
        }
        let mut role = paraphrase.clone();
        role["messages"][0]["role"] = json!("system");
        incompatible.push(role);
        let mut part = paraphrase;
        part["messages"][0]["content"] = json!([{"type": "text", "text": "similar question"}]);
        incompatible.push(part);
        for request in incompatible {
            let mut ctx = ctx_for(request);
            assert!(
                matches!(hook.pre_call(&mut ctx).await.unwrap(), HookAction::Continue),
                "identical embeddings cannot override exact semantics: {}",
                ctx.request_body
            );
        }
        assert!(
            runtime.pending.lock().await.is_empty(),
            "lookup-only misses must not stash"
        );
        assert_eq!(runtime.cache.len().await, 1);
    }

    #[tokio::test]
    async fn semantic_reload_namespaces_shared_entries_by_embedding_model() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .json_body(json!({"data": [{"embedding": [1.0, 0.0]}]}));
        });
        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let original = CacheHook::new(Arc::new(InMemoryCache::default()), Duration::from_secs(60))
            .with_semantic(runtime.clone());
        let mut ctx = ctx_for(cached_request("initial"));
        original.pre_call(&mut ctx).await.unwrap();
        original.post_success(&ctx, &semantic_response()).await;

        // Reuse the store as a persistent tier would across runtime reloads.
        // Both embedding models deliberately return the same vector/dimension.
        for (model, should_hit) in [
            ("text-embedding-3-small", true),
            ("new-embedding-model", false),
        ] {
            let reloaded = Arc::new(SemanticCacheRuntime::new(
                runtime.cache.clone(),
                model.to_string(),
                server.base_url(),
                None,
                http_client(),
            ));
            let hook = CacheHook::new(Arc::new(InMemoryCache::default()), Duration::from_secs(60))
                .with_semantic(reloaded);
            let mut request = cached_request("a paraphrase after reload");
            request["cache"]["no-store"] = json!(true);
            let mut ctx = ctx_for(request);
            assert_eq!(
                matches!(
                    hook.pre_call(&mut ctx).await.unwrap(),
                    HookAction::ShortCircuit(_)
                ),
                should_hit
            );
        }
    }

    #[tokio::test]
    async fn semantic_post_success_cleans_stash_when_store_becomes_disallowed() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .json_body(json!({"data": [{"embedding": [1.0, 0.0]}]}));
        });
        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook =
            CacheHook::new(cache.clone(), Duration::from_secs(60)).with_semantic(runtime.clone());
        let mut ctx = ctx_for(cached_request("no store"));
        hook.pre_call(&mut ctx).await.unwrap();
        assert_eq!(runtime.pending.lock().await.len(), 1);
        Arc::make_mut(&mut ctx.request_body)["cache"]["no-store"] = json!(true);
        hook.post_success(&ctx, &semantic_response()).await;
        assert!(runtime.pending.lock().await.is_empty());
        assert!(runtime.cache.is_empty().await);
        assert!(
            lookup(cache.as_ref(), &ctx.request_body, &ctx.api_key_hash)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn semantic_post_success_does_not_pair_changed_request_with_old_vector() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .json_body(json!({"data": [{"embedding": [1.0, 0.0]}]}));
        });
        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let hook = CacheHook::new(Arc::new(InMemoryCache::default()), Duration::from_secs(60))
            .with_semantic(runtime.clone());
        let mut ctx = ctx_for(cached_request("before another hook"));
        hook.pre_call(&mut ctx).await.unwrap();
        Arc::make_mut(&mut ctx.request_body)["messages"][0]["content"] =
            json!("changed by another hook");
        hook.post_success(&ctx, &semantic_response()).await;
        assert!(runtime.pending.lock().await.is_empty());
        assert!(runtime.cache.is_empty().await);
    }

    #[tokio::test]
    async fn fetch_embedding_rejects_malformed_or_unbounded_vectors() {
        for embedding in [
            json!([1.0, "not a number", 0.0]),
            json!([1e100, 1.0]),
            json!([0.0, 0.0]),
            json!(vec![1.0; MAX_EMBEDDING_DIMENSIONS + 1]),
        ] {
            let server = MockServer::start();
            server.mock(|when, then| {
                when.method(httpmock::Method::POST).path("/embeddings");
                then.status(200)
                    .json_body(json!({"data": [{"embedding": embedding}]}));
            });
            let runtime = semantic_runtime_with_server(server.base_url(), 0.0);
            assert!(runtime.fetch_embedding("probe").await.is_none());
        }
    }

    #[tokio::test]
    async fn semantic_pre_call_short_circuits_on_similarity_hit() {
        let server = MockServer::start();
        // The hook fetches the same embedding on every call (deterministic
        // mock), so similarity will be 1.0 against the stored entry.
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2}
                }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        // Seed the semantic cache directly with the same vector.
        runtime
            .cache
            .store(
                vec![0.1, 0.2, 0.3],
                "seed".to_string(),
                json!({"id": "cached", "choices": []}),
                "abc",
                &runtime
                    .scope_for(&cached_request("seed"), "gpt-4o")
                    .unwrap(),
            )
            .await;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime);
        let mut ctx = ctx_for(cached_request("anything similar"));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        match action {
            HookAction::ShortCircuit(value) => assert_eq!(value["id"], "cached"),
            other => panic!("expected ShortCircuit, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn semantic_pre_call_stashes_embedding_on_miss() {
        let server = MockServer::start();
        let emb_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.4, 0.5, 0.6]}]
            }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.99);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime.clone());
        let mut ctx = ctx_for(cached_request("fresh question"));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        emb_mock.assert_hits(1);
        let stash = runtime.pending.lock().await;
        assert!(
            stash.contains_key("req-1"),
            "embedding should be stashed for post_success"
        );
    }

    #[tokio::test]
    async fn semantic_post_success_stores_embedding_and_response() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [1.0, 0.0, 0.0]}]
            }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime.clone());

        let req = cached_request("first request");
        let mut ctx = ctx_for(req.clone());
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));

        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: Some(json!({"id": "stored-sem"})),
            provider: "openai".to_string(),
        };
        hook.post_success(&ctx, &resp).await;

        assert_eq!(runtime.cache.len().await, 1);
        assert!(
            runtime.pending.lock().await.is_empty(),
            "stash must be drained on post_success"
        );
    }

    #[tokio::test]
    async fn semantic_post_failure_clears_stash() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [1.0, 0.0]}]
            }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime.clone());

        let mut ctx = ctx_for(cached_request("doomed"));
        let _ = hook.pre_call(&mut ctx).await;
        assert!(!runtime.pending.lock().await.is_empty());

        hook.post_failure(&ctx, &oxllm_core::Error::Network("boom".to_string()))
            .await;
        assert!(
            runtime.pending.lock().await.is_empty(),
            "post_failure must clear the stash"
        );
    }

    #[tokio::test]
    async fn semantic_fetch_failure_falls_through_silently() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(500);
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime);
        let mut ctx = ctx_for(cached_request("embedding fetch will fail"));
        let started = std::time::Instant::now();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert!(
            started.elapsed() < Duration::from_secs(6),
            "must not block for long on embedding failure"
        );
    }

    #[tokio::test]
    async fn semantic_cache_tenants_do_not_cross_pollinate() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.7, 0.7]}]
            }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        // Seed for tenant-a only.
        runtime
            .cache
            .store(
                vec![0.7, 0.7],
                "a".to_string(),
                json!({"id": "tenant-a"}),
                "tenant-a",
                &runtime.scope_for(&cached_request("a"), "gpt-4o").unwrap(),
            )
            .await;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime);
        let mut ctx = ctx_for(cached_request("anything"));
        ctx.api_key_hash = "tenant-b".to_string();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "tenant-b must not see tenant-a's semantic entry"
        );
    }

    #[tokio::test]
    async fn semantic_post_success_skipped_on_cache_hit_clears_stash() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.5, 0.5]}]
            }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = cached_request("hit me");
        store(
            cache.as_ref(),
            &req,
            &json!({"id": "exact"}),
            Duration::from_secs(60),
            "abc",
        )
        .await;

        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime.clone());
        let mut ctx = ctx_for(req);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::ShortCircuit(_)));
        // pre_call short-circuited on exact hit — embedding was never stashed.
        assert!(runtime.pending.lock().await.is_empty());
    }

    #[tokio::test]
    async fn pending_stash_is_capped_at_limit() {
        let server = MockServer::start();
        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        // Hold the guards so entries stay "in-flight" (matching production,
        // where the guard lives on the CallContext for the request duration).
        let mut held = Vec::new();
        for i in 0..(PENDING_STASH_CAP + 10) {
            let g = runtime
                .stash_embedding(&format!("req-{i}"), vec![1.0], String::new(), String::new())
                .await;
            held.push(g);
        }
        let len = runtime.pending.lock().await.len();
        assert!(
            len <= PENDING_STASH_CAP,
            "pending stash must be capped at {PENDING_STASH_CAP}, got {len}"
        );
    }

    /// Under cap pressure, abandoned entries (handler future dropped, guard
    /// released) are reclaimed first so a live in-flight consumer's embedding
    /// survives. This pins the LOW fix: cap pressure no longer evicts a live
    /// in-flight embedding when abandoned entries exist to reclaim instead.
    #[tokio::test]
    async fn pending_stash_prefers_abandoned_entries_under_cap_pressure() {
        let server = MockServer::start();
        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);

        // Stash PENDING_STASH_CAP abandoned entries (drop the guards).
        for i in 0..PENDING_STASH_CAP {
            let _ = runtime
                .stash_embedding(
                    &format!("abandoned-{i}"),
                    vec![1.0],
                    String::new(),
                    String::new(),
                )
                .await;
        }
        // Now stash one more entry whose guard IS held — simulating an
        // in-flight request.
        let in_flight_guard = runtime
            .stash_embedding("in-flight", vec![2.0], String::new(), String::new())
            .await;
        let _ = in_flight_guard; // hold it alive

        let guard = runtime.pending.lock().await;
        assert!(
            guard.contains_key("in-flight"),
            "in-flight entry must survive cap pressure while abandoned entries exist to reclaim"
        );
        assert!(
            !guard.contains_key("abandoned-0"),
            "abandoned entries must be reclaimed before in-flight ones"
        );
    }

    #[tokio::test]
    async fn pending_stash_evicts_expired_entries_on_access() {
        let server = MockServer::start();
        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        {
            let mut guard = runtime.pending.lock().await;
            guard.insert(
                "stale".to_string(),
                PendingEmbedding {
                    embedding: vec![1.0],
                    scope: String::new(),
                    request_key: String::new(),
                    stashed_at: std::time::Instant::now()
                        .checked_sub(PENDING_STASH_TTL + Duration::from_secs(1))
                        .expect("no underflow"),
                    guard: std::sync::Arc::downgrade(&std::sync::Arc::new(())),
                },
            );
        }
        // A new stash triggers TTL eviction of the abandoned entry.
        let _ = runtime
            .stash_embedding("fresh", vec![2.0], String::new(), String::new())
            .await;
        let guard = runtime.pending.lock().await;
        assert!(
            !guard.contains_key("stale"),
            "expired entries (dropped handler futures) must be evicted on access"
        );
        assert!(guard.contains_key("fresh"));
    }

    #[tokio::test]
    async fn semantic_pre_call_skips_streaming_requests() {
        let server = MockServer::start();
        let emb_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        // Seed a semantically-identical entry under the same tenant/model.
        runtime
            .cache
            .store(
                vec![0.1, 0.2, 0.3],
                "seed".to_string(),
                json!({"id": "cached-non-stream", "choices": []}),
                "abc",
                &runtime
                    .scope_for(&cached_request("seed"), "gpt-4o")
                    .unwrap(),
            )
            .await;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime.clone());
        let mut req = cached_request("anything similar");
        req["stream"] = json!(true);
        let mut ctx = ctx_for(req);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "streaming requests must not short-circuit on a non-streaming semantic entry"
        );
        emb_mock.assert_hits(0);
        assert!(
            runtime.pending.lock().await.is_empty(),
            "streaming requests must not stash embeddings"
        );
    }

    /// A request beyond the semantic input cap must bypass embedding entirely.
    /// Matching a truncated input would alias requests with different tails.
    #[tokio::test]
    async fn semantic_embedding_post_caps_huge_input() {
        use crate::cache::semantic::MAX_EMBEDDING_TEXT_BYTES;

        let server = MockServer::start();
        // A tail marker placed beyond the cap must NOT reach the embedding
        // endpoint. httpmock 0.7 has no negation matcher, so we register a
        // strict mock (matches only when the tail is present) plus a
        // path-only fallback: a capped POST matches only the fallback.
        let tail_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .body_contains("TAILmarker-beyond-cap");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }));
        });
        let fallback_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }));
        });

        let runtime = semantic_runtime_with_server(server.base_url(), 0.99);
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime);

        // 5 chunks already exceed the cap; the 6th (the tail) is truncated.
        let chunk = "z".repeat(MAX_EMBEDDING_TEXT_BYTES / 4 + 100);
        let req = json!({
        "model": "gpt-4o",
        "input": [chunk.clone(), chunk.clone(), chunk.clone(), chunk.clone(), chunk.clone(), "TAILmarker-beyond-cap"],
        "cache": {"insert": true}
        });
        let mut ctx = ctx_for(req);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        tail_mock.assert_hits(0);
        fallback_mock.assert_hits(0);
    }

    /// A misconfigured embedding provider returning an oversized body must not
    /// be buffered in full — `fetch_embedding` aborts streaming past the cap
    /// and falls through to the upstream call.
    #[tokio::test]
    async fn fetch_embedding_caps_oversized_response_body() {
        let oversized = vec![b' '; MAX_EMBEDDING_RESPONSE_BYTES + 1];
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200)
                .header("content-type", "application/json")
                .body(oversized);
        });

        // Call fetch_embedding directly so the assertion does not depend on
        // the exact request shape; any non-empty text triggers the fetch.
        let runtime = semantic_runtime_with_server(server.base_url(), 0.95);
        let result = runtime.fetch_embedding("probe").await;
        assert!(
            result.is_none(),
            "oversized embedding response must fall through (return None), not OOM"
        );
    }

    /// `evict_expired` runs at most once per [`EVICT_EVERY_N_STORES`] stores,
    /// not on every store — bounding the amortised cost of the O(n) sweep on
    /// the cache-miss path.
    #[tokio::test]
    async fn evict_expired_runs_at_most_every_nth_store() {
        use crate::cache::semantic::SemanticCacheStore;

        struct CountingStore {
            inner: SemanticCache,
            evicts: Arc<std::sync::atomic::AtomicU32>,
        }

        #[async_trait]
        impl SemanticCacheStore for CountingStore {
            async fn search(&self, q: &[f32], k: &str, m: &str) -> Option<Value> {
                self.inner.search(q, k, m).await
            }
            async fn store(&self, e: Vec<f32>, t: String, r: Value, k: &str, m: &str) {
                self.inner.store(e, t, r, k, m).await;
            }
            async fn evict_expired(&self) {
                self.evicts
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.inner.evict_expired().await;
            }
            async fn len(&self) -> usize {
                self.inner.len().await
            }
        }

        let evicts = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let store: Arc<dyn SemanticCacheStore> = Arc::new(CountingStore {
            inner: SemanticCache::new(10000, 0.95, Duration::from_secs(60)),
            evicts: evicts.clone(),
        });
        let runtime = Arc::new(SemanticCacheRuntime::new(
            store,
            "text-embedding-3-small".to_string(),
            "unused".to_string(),
            None,
            http_client(),
        ));

        // Seed the stash directly so post_success has an embedding to store.
        // Each stash key must match the per-request `ctx.request_id` that
        // `pop_embedding` will look up.
        for i in 0..(EVICT_EVERY_N_STORES * 3) {
            let request = cached_request(&format!("store {i}"));
            runtime
                .stash_embedding(
                    &format!("req-{i}"),
                    vec![1.0, i as f32, 0.0],
                    runtime.scope_for(&request, "gpt-4o").unwrap(),
                    cache_key_for(&request, "abc"),
                )
                .await;
        }

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = CacheHook::new(cache, Duration::from_secs(60)).with_semantic(runtime);
        for i in 0..(EVICT_EVERY_N_STORES * 3) {
            let req = cached_request(&format!("store {i}"));
            let mut ctx = ctx_for(req);
            ctx.request_id = format!("req-{i}");
            let resp = ResponseSummary {
                model: "gpt-4o".to_string(),
                cost: None,
                usage: None,
                body: Some(json!({"id": format!("s{i}")})),
                provider: "openai".to_string(),
            };
            hook.post_success(&ctx, &resp).await;
        }

        let total = evicts.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            total <= 3,
            "evict_expired must run at most once per {EVICT_EVERY_N_STORES} stores, got {total}"
        );
        assert!(
            total >= 1,
            "evict_expired must eventually run after {EVICT_EVERY_N_STORES}+ stores"
        );
    }

    /// A cache payload that fails to deserialize must be treated as a miss and
    /// fall through to the semantic/upstream path, never error.
    #[tokio::test]
    async fn corrupted_cache_payload_is_treated_as_a_miss() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let req = cached_request("hello");
        // Poison the exact-match slot with invalid JSON under the real key.
        let key = cache_key_for(&req, "abc");
        cache
            .set(
                &key,
                "{not valid json".to_string(),
                Some(Duration::from_secs(60)),
            )
            .await;

        let hook = CacheHook::new(cache, Duration::from_secs(60));
        let mut ctx = ctx_for(req);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "corrupted payload must fall through as a miss, not error"
        );
    }

    /// When no embedding API key is configured, the embedding POST must omit
    /// the `authorization` header (public/anonymous embeddings).
    #[tokio::test]
    async fn fetch_embedding_omits_authorization_when_api_key_none() {
        let server = MockServer::start();
        // httpmock 0.7 has no negation matcher, so register a strict mock
        // (matches only when authorization IS present) plus a path-only
        // fallback: a headerless POST matches only the fallback.
        let authed_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .header_exists("authorization");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.1, 0.2]}]
            }));
        });
        let fallback_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({
            "data": [{"embedding": [0.1, 0.2]}]
            }));
        });

        // Build a runtime with api_key = None.
        let cache = Arc::new(SemanticCache::new(100, 0.95, Duration::from_secs(60)));
        let runtime = Arc::new(SemanticCacheRuntime::new(
            cache,
            "text-embedding-3-small".to_string(),
            server.base_url(),
            None,
            http_client(),
        ));
        let embedding = runtime.fetch_embedding("probe").await;
        authed_mock.assert_hits(0);
        fallback_mock.assert_hits(1);
        assert!(embedding.is_some(), "embedding should be fetched");
    }
}
