//! Response-cache and semantic-cache configuration.

use serde::{Deserialize, Serialize};

/// Cache configuration within `oxllm_settings`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache backend type: `"memory"` (default), `"redis"`, `"s3"`, `"gcs"`,
    /// or `"azure"`. The blob-store backends (`s3`/`gcs`/`azure`) serve cached
    /// LLM responses from object storage; the server layers an in-memory
    /// counter store on top so rate limits and budgets are still enforced per
    /// replica (the raw backends reject counter writes). See the
    /// per-backend crate docs for details.
    #[serde(default, rename = "type")]
    pub cache_type: Option<String>,
    /// Redis host (used when `type == "redis"`).
    #[serde(default)]
    pub host: Option<String>,
    /// Redis port (used when `type == "redis"`).
    #[serde(default)]
    pub port: Option<u16>,
    /// Full Redis connection URL (e.g. `redis://user:pass@host:6379/0` or
    /// `rediss://host:6379` for TLS). When set, takes precedence over
    /// `host`/`port` and is the only way to express AUTH credentials, TLS, or a
    /// database index — `host`+`port` alone produce an unauthenticated
    /// plaintext `redis://host:port` URL. Mirrors the `redis_url` field already
    /// used by `semantic_cache`.
    #[serde(default)]
    pub redis_url: Option<String>,
    /// Redis key/pub-sub namespace, shared by every replica in a fleet.
    /// Defaults to `oxllm`; independent fleets should choose different values.
    #[serde(default, deserialize_with = "nonempty_cache_prefix")]
    pub prefix: Option<String>,
    /// S3-compatible backend config (used when `type == "s3"`).
    #[serde(default)]
    pub s3: Option<S3CacheConfig>,
    /// Google Cloud Storage backend config (used when `type == "gcs"`).
    #[serde(default)]
    pub gcs: Option<GcsCacheConfig>,
    /// Azure Blob Storage backend config (used when `type == "azure"`).
    #[serde(default)]
    pub azure: Option<AzureBlobCacheConfig>,
    /// Optional semantic-cache configuration. When `enabled` is true the cache
    /// hook falls back from exact-match to embedding-based similarity search.
    #[serde(default)]
    pub semantic_cache: Option<SemanticCacheConfig>,
}

/// S3-compatible object-storage cache backend (AWS S3, MinIO, R2, …).
///
/// The raw backend rejects counter writes; the server wiring layers an
/// in-memory counter store so rate limits and budgets are enforced per replica.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S3CacheConfig {
    /// Bucket name.
    pub bucket: String,
    /// AWS region (e.g. `"us-east-1"`).
    #[serde(default)]
    pub region: Option<String>,
    /// Key prefix joined to every cache key as `{prefix}:{key}`. Defaults to
    /// `"oxllm"` when unset at construction time. An explicit prefix must contain
    /// a non-whitespace character; use distinct prefixes for independent fleets.
    #[serde(default, deserialize_with = "nonempty_cache_prefix")]
    pub prefix: Option<String>,
    /// AWS access key id. When omitted, `AWS_ACCESS_KEY_ID` is read at startup.
    #[serde(default)]
    pub access_key_id: Option<String>,
    /// AWS secret access key. When omitted, `AWS_SECRET_ACCESS_KEY` is read.
    #[serde(default)]
    pub secret_access_key: Option<String>,
    /// Optional AWS session token (temporary credentials).
    #[serde(default)]
    pub session_token: Option<String>,
    /// Custom endpoint base (e.g. `http://localhost:9000` for MinIO / R2).
    /// When set, path-style URLs are used.
    #[serde(default)]
    pub endpoint: Option<String>,
}

/// Google Cloud Storage cache backend.
///
/// Response-cache-only: counter writes are unsupported. Per-object TTL is enforced
/// client-side via an `expires` custom-metadata deadline written atomically
/// with the response using a JSON API multipart upload. Reads use generation
/// preconditions and treat expired or legacy unversioned objects as misses;
/// objects are not auto-deleted, so bucket lifecycle rules should still be
/// configured to reclaim space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GcsCacheConfig {
    /// Bucket name.
    pub bucket: String,
    /// Key prefix joined to every cache key as `{prefix}:{key}`. Defaults to
    /// `"oxllm"` when unset. An explicit prefix must contain a non-whitespace
    /// character; use distinct prefixes for independent fleets.
    #[serde(default, deserialize_with = "nonempty_cache_prefix")]
    pub prefix: Option<String>,
    /// Raw JSON of a GCP service-account key file. When omitted, the Compute
    /// Metadata server is used for auth (valid only on GCP).
    #[serde(default)]
    pub service_account_json: Option<String>,
}

/// Azure Blob Storage cache backend.
///
/// Response-cache-only: counter writes are unsupported. Per-object TTL is enforced
/// client-side via the `x-ms-meta-expires` metadata deadline (reads past
/// the deadline are misses); blobs are not auto-deleted, so container
/// lifecycle rules should still be configured to reclaim space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AzureBlobCacheConfig {
    /// Storage account name.
    pub account: String,
    /// Container name.
    pub container: String,
    /// Key prefix joined to every cache key as `{prefix}:{key}`. Defaults to
    /// `"oxllm"` when unset. An explicit prefix must contain a non-whitespace
    /// character; use distinct prefixes for independent fleets.
    #[serde(default, deserialize_with = "nonempty_cache_prefix")]
    pub prefix: Option<String>,
    /// Azure Storage connection string (parsed for `AccountKey=`). Enables
    /// SharedKey auth. When omitted, managed-identity auth is used (valid only
    /// on Azure).
    #[serde(default)]
    pub connection_string: Option<String>,
}

/// Semantic-cache configuration, opt-in via `oxllm_settings.cache.semantic_cache`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticCacheConfig {
    /// Master switch. When `false` (or absent) semantic caching is disabled.
    #[serde(default)]
    pub enabled: bool,
    /// Model used to embed request text (e.g. `"text-embedding-3-small"`).
    pub embedding_model: String,
    /// Cosine-similarity cutoff in `[0.0, 1.0]`. Defaults to `0.95`.
    #[serde(default)]
    pub similarity_threshold: Option<f64>,
    /// Global vector capacity of the in-memory tier (also used by the Redis
    /// tier), not a per-tenant quota. Defaults to `10_000`. Zero disables both
    /// tiers. Qdrant storage is governed by entry expiry and operator retention.
    #[serde(default)]
    pub max_entries: Option<usize>,
    /// Entry TTL in seconds. Defaults to `3600`; zero means never expire for
    /// semantic entries. This config sentinel is separate from `Cache::set`'s
    /// optional TTL, where `None` means no expiry and `Some(0)` expires at once.
    #[serde(default)]
    pub ttl_secs: Option<u64>,
    /// Embedding API base URL (e.g. `https://api.openai.com/v1`). When
    /// omitted the embedding model's deployment is resolved from
    /// `model_list`.
    #[serde(default)]
    pub api_base: Option<String>,
    /// Embedding API key. When omitted the deployment's configured key is
    /// used. Resolved from config (never from the inbound request's auth).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Redis URL (e.g. `redis://127.0.0.1:6379`). When set, semantic-cache
    /// entries are mirrored to Redis in addition to in-memory storage, so
    /// they survive restarts and are shared across replicas. Cosine
    /// similarity is computed client-side (fetch + scan).
    #[serde(default)]
    pub redis_url: Option<String>,
    /// Qdrant endpoint (e.g. `http://127.0.0.1:6333`). When set, semantic
    /// cache uses Qdrant as a vector store via its REST API. Takes
    /// precedence over neither; when both this and `redis_url` are set,
    /// Redis is preferred.
    #[serde(default)]
    pub qdrant_url: Option<String>,
    /// Optional Qdrant API key (for Qdrant Cloud / authenticated clusters).
    #[serde(default)]
    pub qdrant_api_key: Option<String>,
    /// Qdrant collection name. Defaults to `"oxllm_cache"`.
    #[serde(default)]
    pub qdrant_collection: Option<String>,
    /// Vector dimension for the Qdrant collection (only used at collection
    /// creation). Defaults to `1536` (OpenAI `text-embedding-3-small`).
    #[serde(default)]
    pub qdrant_dimension: Option<usize>,
}

fn nonempty_cache_prefix<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let prefix = Option::<String>::deserialize(deserializer)?;
    if prefix
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(serde::de::Error::custom(
            "cache prefix must not be empty or whitespace-only",
        ));
    }
    Ok(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn explicit_empty_blob_prefixes_are_rejected_during_config_deserialization() {
        for prefix in ["", " ", "\t\n"] {
            let bucket = json!({"bucket": "bucket", "prefix": prefix});
            let container = json!({"account": "acct", "container": "data", "prefix": prefix});
            assert!(serde_json::from_value::<S3CacheConfig>(bucket.clone()).is_err());
            assert!(serde_json::from_value::<GcsCacheConfig>(bucket).is_err());
            assert!(serde_json::from_value::<AzureBlobCacheConfig>(container).is_err());
        }
    }

    #[test]
    fn omitted_blob_prefix_still_uses_the_backend_default() {
        let s3: S3CacheConfig = serde_json::from_value(json!({"bucket": "bucket"})).unwrap();
        let gcs: GcsCacheConfig =
            serde_json::from_value(json!({"bucket": "bucket", "prefix": null})).unwrap();
        let azure: AzureBlobCacheConfig = serde_json::from_value(json!({
            "account": "acct", "container": "data", "prefix": "fleet-a"
        }))
        .unwrap();
        assert!(s3.prefix.is_none());
        assert!(gcs.prefix.is_none());
        assert_eq!(azure.prefix.as_deref(), Some("fleet-a"));
    }
}
