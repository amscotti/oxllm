//! Google Cloud Storage response cache backend for oxllm.
//!
//! Implements [`oxllm_core::Cache`] over GCS via the JSON REST API
//! (`POST`/`GET`/`DELETE` on objects). Object name = `{prefix}:{original_key}`;
//! values are stored verbatim as object bodies (the caller-supplied JSON
//! strings).
//!
//! Auth uses the [service-account OAuth2 JWT
//! flow](https://developers.google.com/identity/protocols/oauth2/service-account#jwt-auth)
//! when `service_account_json` is provided: an RS256 JWT is signed with the
//! service-account private key and exchanged for an access token. When no
//! service account is configured, the backend falls back to the Compute
//! Metadata server token (works on GCE / Cloud Run / Cloud Functions). The
//! resulting token is cached for ~90 % of its advertised lifetime so a token is
//! exchanged at most once per hour, not once per cache operation. A
//! `token_uri` embedded in the service-account JSON must use `https:` with a
//! host name (never an IP literal or plaintext endpoint).
//!
//! ## Response-cache-only (no atomic counters)
//!
//! Blob stores have no atomic increment primitive. This backend models only
//! the response-blob operations (`get`/`set`/`delete`/`ping`); the counter
//! write operations surface an explicit error so a misconfigured direct use
//! fails loudly instead of silently disabling budget / rate-limit / JWT-replay
//! enforcement:
//!
//! - [`oxllm_core::Cache::incr`] → `Err` (counters unsupported)
//! - [`oxllm_core::Cache::incr_if_below`] → `Err` (counters unsupported)
//! - [`oxllm_core::Cache::counter_value`] / [`oxllm_core::Cache::try_counter_value`]
//!   → `None` / `Ok(None)` (a miss is the correct fail-open for an unsupported
//!   read)
//!
//! Because silently returning zero would disable budget / rate-limit / JWT
//! replay enforcement, the server wiring (`build_gcs_cache`) does **not**
//! expose this raw backend as the application cache: it wraps it with an
//! in-memory counter layer so response blobs flow to GCS while atomic counters
//! stay accurate per replica. Direct users of `GcsCache` must supply their own
//! counter layer if enforcement is required. Per-object TTL is enforced
//! client-side via the `expires` custom metadata field (a Unix-seconds deadline).
//! JSON API writes use a multipart upload so data and metadata are stored
//! atomically. Reads fetch metadata, then download with generation and
//! metageneration preconditions. GCS does not auto-delete on this deadline.
//! Legacy data-only uploads have no reliable expiry metadata and are misses.
//! `None` means no expiry; `Some(Duration::ZERO)` invalidates a prior value.
//! Public cache operations have a 500ms total optimization budget including
//! token acquisition, retries, backoff, and response-body reads.

#![forbid(unsafe_code)]

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use jsonwebtoken::Algorithm;
use jsonwebtoken::EncodingKey;
use jsonwebtoken::Header;
use serde::Deserialize;
use serde::Serialize;

use oxllm_core::Cache as CacheTrait;
use oxllm_core::{Error, IncrResult};

/// OAuth2 scope granting read/write object access.
const STORAGE_SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";
/// Default token endpoint.
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
/// Default GCS JSON API base.
const DEFAULT_API_BASE: &str = "https://storage.googleapis.com";
/// Transport ceiling; public cache operations use the tighter total budget below.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CACHE_OPERATION_TIMEOUT: Duration = Duration::from_millis(500);
/// Total attempts per object operation for transient failures.
const MAX_SEND_ATTEMPTS: usize = 3;
/// Base backoff between transient retries; doubled per attempt.
const RETRY_BASE_DELAY: Duration = Duration::from_millis(100);

/// True for statuses worth retrying: 429 (throttle) and 5xx. 401 has its own
/// dedicated token-refresh retry in [`GcsCache::send_with_retry`]; every
/// other 4xx is terminal.
fn is_retryable_status(code: u16) -> bool {
    code == 429 || (500..=599).contains(&code)
}
/// Default cap on a single cached response body read or written (16 MiB).
/// Guards against an oversized or hostile object body OOM-killing the proxy on
/// a cache read; checked against `Content-Length` and enforced while streaming.
const DEFAULT_MAX_OBJECT_SIZE: usize = 16 * 1024 * 1024;
/// Refresh a cached token at 90 % of its lifetime to avoid using a token that
/// expires mid-flight.
const TOKEN_REFRESH_FRACTION: u32 = 9;
const TOKEN_LIFETIME_DENOMINATOR: u32 = 10;
/// Backoff applied after a failed token fetch: concurrent and subsequent
/// callers within this window get an immediate error instead of re-issuing a
/// network call serialized behind the lock, preventing cascading stalls on the
/// request hot path during a GCS auth outage.
const TOKEN_FAILURE_BACKOFF: Duration = Duration::from_secs(5);
/// Maximum bytes read from a token/metadata error response body before
/// truncation. Bounds log volume and CPU if an endpoint streams a large body;
/// the body is informational only.
const MAX_ERROR_BODY: usize = 4 * 1024;
const MAX_METADATA_BODY: usize = 64 * 1024;
const CACHE_FORMAT_VERSION: &str = "2";

async fn cache_operation<T>(
    name: &str,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    match tokio::time::timeout(CACHE_OPERATION_TIMEOUT, operation).await {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::warn!(
                operation = name,
                "gcs cache operation exceeded optimization deadline"
            );
            None
        }
    }
}

/// A GCS-backed cache implementing [`oxllm_core::Cache`].
pub struct GcsCache {
    client: reqwest::Client,
    bucket: String,
    prefix: String,
    /// Object API base (`https://storage.googleapis.com` in production);
    /// injectable so the object round-trip can be driven against a local mock.
    api_base: String,
    auth: GcsAuth,
    /// Upper bound (bytes) on either a `get` or `set` body.
    max_object_size: usize,
    /// Short-lived bearer token cache so a token is exchanged at most once per
    /// lifetime (minus a skew margin) instead of once per cache operation. A
    /// failed fetch is also cached for [`TOKEN_FAILURE_BACKOFF`] so a dead
    /// endpoint degrades to instant cache-misses instead of serialized stalls.
    token_cache: tokio::sync::Mutex<Option<TokenEntry>>,
}

/// Cached token state. `Valid` holds a usable token and its refresh deadline;
/// `Failed` records a short backoff window during which [`GcsCache::token`]
/// returns immediately without re-issuing the network call, preventing
/// serialized cascading stalls during a GCS auth outage.
enum TokenEntry {
    Valid { token: String, refresh_at: Instant },
    Failed { retry_after: Instant },
}

enum GcsAuth {
    /// Explicit service account → JWT-exchange token flow.
    ServiceAccount(ServiceAccountCredentials),
    /// No key configured → fetch tokens from the Compute Metadata server.
    Metadata,
}

#[derive(Clone)]
struct ServiceAccountCredentials {
    client_email: String,
    private_key_id: Option<String>,
    token_uri: String,
    encoding_key: EncodingKey,
}

/// Subset of a GCP service-account JSON key file.
#[derive(Deserialize)]
struct ServiceAccountJson {
    client_email: String,
    private_key: String,
    #[serde(default)]
    private_key_id: Option<String>,
    #[serde(default)]
    token_uri: Option<String>,
}

/// Token endpoint response.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<i64>,
}

#[derive(Deserialize)]
struct ObjectMetadata {
    generation: String,
    metageneration: String,
    size: String,
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
}

/// Service-account JWT claims.
#[derive(Serialize)]
struct Claims {
    iss: String,
    scope: String,
    aud: String,
    iat: i64,
    exp: i64,
}

impl GcsCache {
    /// Create a new GCS cache.
    ///
    /// When `service_account_json` is `Some`, it must be the raw JSON of a
    /// service-account key file. When `None`, the backend falls back to the
    /// Compute Metadata server for tokens (valid only when running on GCP).
    ///
    /// A `token_uri` embedded in the service-account JSON must use `https:`
    /// and a host name (not an IP literal) — the RS256-signed JWT (which
    /// carries the service-account `client_email`) must never be sent over a
    /// plaintext or host-less endpoint.
    pub fn new(
        bucket: &str,
        prefix: &str,
        service_account_json: Option<&str>,
    ) -> Result<Self, Error> {
        Self::build(bucket, prefix, DEFAULT_API_BASE, service_account_json, true)
    }

    /// Shared constructor. `validate_token_https` gates the `token_uri` scheme
    /// check (skipped only by the test constructor, which points `token_uri`
    /// at a local plaintext mock).
    fn build(
        bucket: &str,
        prefix: &str,
        api_base: &str,
        service_account_json: Option<&str>,
        validate_token_https: bool,
    ) -> Result<Self, Error> {
        validate_bucket_name(bucket)?;
        if prefix.trim().is_empty() {
            return Err(Error::Config("gcs cache prefix must not be empty".into()));
        }
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Config(format!("gcs http client build failed: {e}")))?;
        let auth = match service_account_json {
            Some(json) => {
                let parsed: ServiceAccountJson = serde_json::from_str(json).map_err(|e| {
                    Error::Config(format!("gcs service_account_json parse failed: {e}"))
                })?;
                let encoding_key = EncodingKey::from_rsa_pem(parsed.private_key.as_bytes())
                    .map_err(|e| Error::Config(format!("gcs private_key parse failed: {e}")))?;
                let token_uri = parsed
                    .token_uri
                    .unwrap_or_else(|| DEFAULT_TOKEN_URI.to_string());
                if validate_token_https {
                    validate_token_uri(&token_uri)?;
                }
                GcsAuth::ServiceAccount(ServiceAccountCredentials {
                    client_email: parsed.client_email,
                    private_key_id: parsed.private_key_id,
                    token_uri,
                    encoding_key,
                })
            }
            None => GcsAuth::Metadata,
        };
        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            api_base: api_base.to_string(),
            auth,
            max_object_size: DEFAULT_MAX_OBJECT_SIZE,
            token_cache: tokio::sync::Mutex::new(None),
        })
    }

    fn object_name(&self, key: &str) -> String {
        format!("{}:{key}", self.prefix)
    }

    /// Build the object resource path (`/storage/v1/b/{bucket}/o/{enc}`).
    fn object_path(&self, object_name: &str) -> String {
        format!(
            "/storage/v1/b/{}/o/{}",
            self.bucket,
            encode_component(object_name)
        )
    }

    /// Obtain a valid bearer token. The token is cached for 90 % of its
    /// lifetime (typically ~1h) and the refresh is coalesced through the lock,
    /// so concurrent operations share one token exchange instead of issuing a
    /// JWT-bearer POST per cache op. A failed fetch is negative-cached for
    /// [`TOKEN_FAILURE_BACKOFF`] so a dead token endpoint degrades to instant
    /// cache-misses instead of serialized cascading stalls.
    async fn token(&self) -> Result<String, Error> {
        let mut cache = self.token_cache.lock().await;
        let now = Instant::now();
        if let Some(entry) = cache.as_ref() {
            match entry {
                TokenEntry::Valid { token, refresh_at } if *refresh_at > now => {
                    return Ok(token.clone());
                }
                TokenEntry::Failed { retry_after } if *retry_after > now => {
                    return Err(Error::Network(
                        "gcs token fetch recently failed; backing off".to_string(),
                    ));
                }
                _ => {}
            }
        }
        // Leave a bounded backoff even if the enclosing cache operation is
        // cancelled while awaiting the token endpoint. Otherwise each waiter
        // would start another stalled fetch after the previous lock is dropped.
        *cache = Some(TokenEntry::Failed {
            retry_after: now + TOKEN_FAILURE_BACKOFF,
        });
        match self.fetch_token().await {
            Ok((token, expires_in)) => {
                let lifetime =
                    Duration::from_secs(expires_in.unwrap_or(3600).clamp(60, 86_400) as u64);
                *cache = Some(TokenEntry::Valid {
                    token: token.clone(),
                    refresh_at: now
                        + lifetime / TOKEN_LIFETIME_DENOMINATOR * TOKEN_REFRESH_FRACTION,
                });
                Ok(token)
            }
            Err(e) => {
                *cache = Some(TokenEntry::Failed {
                    retry_after: now + TOKEN_FAILURE_BACKOFF,
                });
                Err(e)
            }
        }
    }

    /// Exchange a JWT (service account) or hit the metadata server for a fresh
    /// access token, returning the token and its advertised lifetime.
    async fn fetch_token(&self) -> Result<(String, Option<i64>), Error> {
        match &self.auth {
            GcsAuth::ServiceAccount(creds) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                let mut header = Header::new(Algorithm::RS256);
                header.kid = creds.private_key_id.clone();
                let claims = Claims {
                    iss: creds.client_email.clone(),
                    scope: STORAGE_SCOPE.to_string(),
                    aud: creds.token_uri.clone(),
                    iat: now,
                    exp: now + 3600,
                };
                let jwt = jsonwebtoken::encode(&header, &claims, &creds.encoding_key)
                    .map_err(|e| Error::Network(format!("gcs jwt encode failed: {e}")))?;

                let resp = self
                    .client
                    .post(&creds.token_uri)
                    .form(&[
                        ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                        ("assertion", jwt.as_str()),
                    ])
                    .send()
                    .await
                    .map_err(|e| Error::Network(format!("gcs token request failed: {e}")))?;
                let status = resp.status();
                if !status.is_success() {
                    let body = read_error_body(resp).await;
                    return Err(Error::Network(format!(
                        "gcs token endpoint returned {status}: {body}"
                    )));
                }
                let token_resp: TokenResponse = resp
                    .json()
                    .await
                    .map_err(|e| Error::Network(format!("gcs token decode failed: {e}")))?;
                Ok((token_resp.access_token, token_resp.expires_in))
            }
            GcsAuth::Metadata => {
                let resp = self
                    .client
                    .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
                    .header("Metadata-Flavor", "Google")
                    .send()
                    .await
                    .map_err(|e| Error::Network(format!("gcs metadata token failed: {e}")))?;
                let status = resp.status();
                if !status.is_success() {
                    let body = read_error_body(resp).await;
                    return Err(Error::Network(format!(
                        "gcs metadata token returned {status}: {body}"
                    )));
                }
                let token_resp: TokenResponse = resp.json().await.map_err(|e| {
                    Error::Network(format!("gcs metadata token decode failed: {e}"))
                })?;
                Ok((token_resp.access_token, token_resp.expires_in))
            }
        }
    }

    /// Send an object request, retrying once with a freshly-fetched token when
    /// GCS returns 401. A cached token can become invalid mid-life (e.g. the
    /// service-account key was revoked or IAM tightened); without recovery the
    /// stale token would be re-used until its `refresh_at`, leaving the cache
    /// effectively dead for up to ~54 min. On a 401 the token cache is cleared
    /// and the operation retried exactly once — no further retries, so a
    /// genuinely revoked key degrades to a miss instead of a retry storm.
    ///
    /// Transient failures (transport errors, 429, 5xx) are additionally
    /// retried with exponential backoff up to [`MAX_SEND_ATTEMPTS`] total
    /// attempts; the 401 refresh does not consume the transient budget.
    async fn send_with_retry<F>(&self, build: F) -> Result<reqwest::Response, Error>
    where
        F: Fn(&str) -> reqwest::RequestBuilder,
    {
        let mut refreshed = false;
        let mut attempt = 0usize;
        loop {
            let token = self.token().await?;
            let resp = match build(token.as_str()).send().await {
                Ok(resp) => resp,
                Err(e) => {
                    if attempt + 1 < MAX_SEND_ATTEMPTS {
                        attempt += 1;
                        tracing::debug!(error = %e, attempt, "gcs request failed, retrying");
                        tokio::time::sleep(RETRY_BASE_DELAY * (1 << (attempt - 1).min(4))).await;
                        continue;
                    }
                    return Err(Error::Network(format!("gcs request failed: {e}")));
                }
            };
            let code = resp.status().as_u16();
            if code == 401 && !refreshed {
                tracing::warn!("gcs 401 with cached token; invalidating token and retrying once");
                *self.token_cache.lock().await = None;
                refreshed = true;
                continue;
            }
            if is_retryable_status(code) && attempt + 1 < MAX_SEND_ATTEMPTS {
                attempt += 1;
                tracing::debug!(status = code, attempt, "gcs transient status, retrying");
                tokio::time::sleep(RETRY_BASE_DELAY * (1 << (attempt - 1).min(4))).await;
                continue;
            }
            return Ok(resp);
        }
    }
}

/// Validate a service-account `token_uri`: must be `https:` with a non-empty
/// host that is not an IP literal. Prevents the RS256-signed JWT (carrying the
/// service-account email) from being sent to a plaintext or host-less endpoint
/// embedded in a tampered key file.
fn validate_token_uri(token_uri: &str) -> Result<(), Error> {
    let parsed = url::Url::parse(token_uri)
        .map_err(|e| Error::Config(format!("gcs token_uri is not a valid url: {e}")))?;
    if parsed.scheme() != "https" {
        return Err(Error::Config(format!(
            "gcs token_uri must use https: {token_uri}"
        )));
    }
    let host = parsed.host_str().unwrap_or("");
    if host.is_empty() || host.parse::<std::net::IpAddr>().is_ok() {
        return Err(Error::Config(format!(
            "gcs token_uri must use a host name, not an IP literal: {token_uri}"
        )));
    }
    Ok(())
}

/// Validate a GCS bucket name: 3-63 characters of `[a-z0-9._-]`, starting and
/// ending with a lowercase letter or digit, with no consecutive dots. Fails
/// fast at construction rather than producing malformed request URLs at runtime.
fn validate_bucket_name(bucket: &str) -> Result<(), Error> {
    let bytes = bucket.as_bytes();
    if !(3..=63).contains(&bytes.len()) {
        return Err(Error::Config(format!(
            "gcs bucket name must be 3-63 characters: got {}",
            bytes.len()
        )));
    }
    if bucket.contains("..") {
        return Err(Error::Config(
            "gcs bucket name must not contain consecutive dots".to_string(),
        ));
    }
    let is_edge = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit();
    if !is_edge(bytes[0]) || !is_edge(bytes[bytes.len() - 1]) {
        return Err(Error::Config(
            "gcs bucket name must start and end with a lowercase letter or digit".to_string(),
        ));
    }
    for &b in bytes {
        let valid = b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.');
        if !valid {
            return Err(Error::Config(format!(
                "gcs bucket name contains invalid character {:?}",
                b as char
            )));
        }
    }
    Ok(())
}

/// Read at most [`MAX_ERROR_BODY`] bytes from an error response body, streaming
/// so a multi-MiB body is never fully materialized. The body is informational
/// only (interpolated into an [`Error::Network`] message that may be logged).
async fn read_error_body(resp: reqwest::Response) -> String {
    let mut resp = resp;
    let mut buf = Vec::new();
    while let Ok(Some(chunk)) = resp.chunk().await {
        let remaining = MAX_ERROR_BODY.saturating_sub(buf.len());
        if remaining == 0 {
            break;
        }
        let take = chunk.len().min(remaining);
        buf.extend_from_slice(&chunk[..take]);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

impl GcsCache {
    async fn get_inner(&self, key: &str) -> Option<String> {
        let name = self.object_name(key);
        let url = format!("{}{}", self.api_base, self.object_path(&name));
        let mut metadata_response = match self
            .send_with_retry(|token| {
                self.client.get(&url).bearer_auth(token).query(&[
                    ("alt", "json"),
                    ("fields", "generation,metageneration,size,metadata"),
                ])
            })
            .await
        {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(error = %error, key = %key, "gcs metadata get failed");
                return None;
            }
        };
        let status = metadata_response.status();
        if status.as_u16() == 404 {
            return None;
        }
        if !status.is_success() {
            tracing::warn!(status = %status, key = %key, "gcs metadata get non-success");
            return None;
        }
        let mut metadata_bytes = Vec::new();
        while let Some(chunk) = metadata_response.chunk().await.ok()? {
            if metadata_bytes.len() + chunk.len() > MAX_METADATA_BODY {
                return None;
            }
            metadata_bytes.extend_from_slice(&chunk);
        }
        let metadata: ObjectMetadata = serde_json::from_slice(&metadata_bytes).ok()?;
        if metadata
            .metadata
            .get("oxllm_cache_version")
            .map(String::as_str)
            != Some(CACHE_FORMAT_VERSION)
            || metadata.size.parse::<u64>().ok()? > self.max_object_size as u64
        {
            return None;
        }
        let deadline = match metadata.metadata.get("expires") {
            Some(expires) => Some(expires.parse::<u64>().ok()?),
            None => None,
        };
        if deadline_expired(deadline) {
            return None;
        }
        let generation = metadata.generation.parse::<u64>().ok()?.to_string();
        let metageneration = metadata.metageneration.parse::<u64>().ok()?.to_string();
        let mut resp = match self
            .send_with_retry(|token| {
                self.client.get(&url).bearer_auth(token).query(&[
                    ("alt", "media"),
                    // Do not request a historical generation: after a concurrent
                    // replacement/deletion, the current-object precondition must
                    // fail rather than resurrecting an old version.
                    ("ifGenerationMatch", generation.as_str()),
                    ("ifMetagenerationMatch", metageneration.as_str()),
                ])
            })
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "gcs get failed");
                return None;
            }
        };
        let status = resp.status();
        if status.as_u16() == 404 {
            return None;
        }
        if !status.is_success() {
            tracing::warn!(status = %status, key = %key, "gcs get non-success");
            return None;
        }
        if let Some(len) = resp.content_length() {
            if len > self.max_object_size as u64 {
                tracing::warn!(
                    key = %key,
                    len,
                    cap = self.max_object_size,
                    "gcs get object exceeds max size"
                );
                return None;
            }
        }
        if deadline_expired(deadline) {
            tracing::debug!(
                key = %key,
                "gcs get object past client-side expires deadline; treating as miss"
            );
            return None;
        }
        let mut buf = Vec::new();
        loop {
            match resp.chunk().await {
                Ok(Some(chunk)) => {
                    if buf.len() + chunk.len() > self.max_object_size {
                        tracing::warn!(
                            key = %key,
                            cap = self.max_object_size,
                            "gcs get object exceeded max size while streaming"
                        );
                        return None;
                    }
                    buf.extend_from_slice(&chunk);
                }
                Ok(None) => break,
                Err(e) => {
                    tracing::warn!(error = %e, key = %key, "gcs get body read failed");
                    return None;
                }
            }
        }
        if deadline_expired(deadline) {
            return None;
        }
        match String::from_utf8(buf) {
            Ok(text) => Some(text),
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "gcs get body not utf-8");
                None
            }
        }
    }

    async fn set_inner(&self, key: &str, value: String, ttl: Option<Duration>) {
        if ttl.is_some_and(|ttl| ttl.is_zero()) {
            self.delete_inner(key).await;
            return;
        }
        if value.len() > self.max_object_size {
            tracing::warn!(
                key = %key,
                size = value.len(),
                cap = self.max_object_size,
                "gcs set skipped oversized value"
            );
            return;
        }
        let deadline = ttl.map(|t| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            // Round sub-second TTLs up to 1s so they never truncate to
            // "never expire".
            now.saturating_add(t.as_secs().max(1))
        });
        let name = self.object_name(key);
        let Some((content_type, body)) = multipart_upload(&name, &value, deadline) else {
            tracing::warn!(key = %key, "gcs set skipped multipart boundary collision");
            return;
        };
        let url = format!(
            "{}/upload/storage/v1/b/{}/o?uploadType=multipart&name={}",
            self.api_base,
            self.bucket,
            encode_component(&name)
        );
        let resp = match self
            .send_with_retry(|token| {
                self.client
                    .post(&url)
                    .bearer_auth(token)
                    .header(reqwest::header::CONTENT_TYPE, &content_type)
                    .body(body.clone())
            })
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "gcs set failed");
                return;
            }
        };
        if !resp.status().is_success() {
            tracing::warn!(
                status = %resp.status(),
                key = %key,
                "gcs set non-success"
            );
        }
    }

    async fn delete_inner(&self, key: &str) {
        let name = self.object_name(key);
        let url = format!("{}{}", self.api_base, self.object_path(&name));
        let resp = match self
            .send_with_retry(|token| self.client.delete(&url).bearer_auth(token))
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "gcs delete failed");
                return;
            }
        };
        let status = resp.status().as_u16();
        // 204 success; 404 (absent) is not an error.
        if !(status == 204 || status == 200 || status == 404) {
            tracing::warn!(status = %status, key = %key, "gcs delete non-success");
        }
    }
}

#[async_trait]
impl CacheTrait for GcsCache {
    async fn get(&self, key: &str) -> Option<String> {
        cache_operation("get", self.get_inner(key)).await.flatten()
    }

    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        let _ = cache_operation("set", self.set_inner(key, value, ttl)).await;
    }

    async fn delete(&self, key: &str) {
        let _ = cache_operation("delete", self.delete_inner(key)).await;
    }

    /// Unsupported: blob stores have no atomic increment. Returning an error
    /// (rather than a silent `Ok`) ensures a direct use cannot silently drop
    /// spend debits / rate-limit / JWT-replay enforcement. Layer a counter
    /// store (e.g. the server's `BlobCounterCache`) for metered paths.
    /// See module docs.
    async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
        Err(Error::Config(
            "gcs cache backend does not support atomic counters; layer a counter store".into(),
        ))
    }

    /// Unsupported: see [`CacheTrait::incr`].
    async fn incr_if_below(
        &self,
        _key: &str,
        _delta: f64,
        _ceiling: f64,
        _ttl: Option<Duration>,
    ) -> Result<IncrResult, Error> {
        Err(Error::Config(
            "gcs cache backend does not support atomic counters; layer a counter store".into(),
        ))
    }

    /// No-op: counters are not modeled. A miss is the correct fail-open for an
    /// unsupported read. See module docs.
    async fn counter_value(&self, _key: &str) -> Option<f64> {
        None
    }

    /// Like [`CacheTrait::counter_value`]: counters are not modeled, so a read
    /// is an absence (`Ok(None)`) rather than a routing through the blob store
    /// (which would issue a GCS GET per read and mis-parse a non-counter body).
    async fn try_counter_value(&self, _key: &str) -> Result<Option<f64>, Error> {
        Ok(None)
    }

    /// Liveness: a metadata GET on the bucket must succeed (2xx) for the cache
    /// to be considered reachable. A transport `Ok` with a 403/404 (wrong
    /// service account / missing bucket) or a 5xx (GCS outage) is `false` —
    /// reporting healthy on those would mask a deployment that fails every
    /// actual `get`/`set`.
    async fn ping(&self) -> bool {
        cache_operation("ping", async {
            let url = format!("{}/storage/v1/b/{}", self.api_base, self.bucket);
            match self
                .send_with_retry(|token| self.client.get(&url).bearer_auth(token))
                .await
            {
                Ok(resp) => {
                    if resp.status().is_success() {
                        true
                    } else {
                        tracing::warn!(
                            status = %resp.status(),
                            "gcs ping non-success; cache unreachable"
                        );
                        false
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "gcs ping request failed");
                    false
                }
            }
        })
        .await
        .unwrap_or(false)
    }
}

fn deadline_expired(deadline: Option<u64>) -> bool {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    deadline.is_some_and(|deadline| now >= deadline)
}

/// The JSON API's data-only (`uploadType=media`) endpoint does not accept
/// custom metadata headers. Use multipart/related to atomically store metadata
/// with the verbatim response. Boundary collisions skip the optimization; the
/// check, not unpredictability of the boundary, guarantees correct MIME framing.
fn multipart_upload(name: &str, value: &str, deadline: Option<u64>) -> Option<(String, Vec<u8>)> {
    let mut metadata = serde_json::json!({
        "name": name,
        "metadata": {"oxllm_cache_version": CACHE_FORMAT_VERSION},
    });
    if let Some(deadline) = deadline {
        metadata["metadata"]["expires"] = serde_json::json!(deadline.to_string());
    }
    let metadata = metadata.to_string();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let boundary = format!("oxllm-cache-{nonce:x}");
    if value.contains(&boundary) || metadata.contains(&boundary) {
        return None;
    }
    let mut body = format!(
        "--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{metadata}\r\n\
         --{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(value.as_bytes());
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    Some((format!("multipart/related; boundary={boundary}"), body))
}

/// Percent-encode a single path segment (RFC 3986 component encoding), encoding
/// `/` too — GCS object names are a single path segment after `/o/`.
fn encode_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &byte in input.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only constructor: injects the object-API `base_url` (so object
    /// round-trips can hit a local mock) and skips the `https` check on
    /// `token_uri` (so the JWT exchange can hit a plaintext mock). Production
    /// code uses [`GcsCache::new`].
    fn new_for_test(
        bucket: &str,
        prefix: &str,
        api_base: &str,
        service_account_json: Option<&str>,
    ) -> Result<GcsCache, Error> {
        GcsCache::build(bucket, prefix, api_base, service_account_json, false)
    }

    /// Build a service-account JSON string carrying a real RSA PEM and the
    /// given `token_uri`.
    fn sa_json(token_uri: &str) -> String {
        let dummy_key = include_str!("../tests/data/dummy_rsa_key.pem");
        serde_json::json!({
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "private_key": dummy_key,
            "private_key_id": "kid123",
            "token_uri": token_uri,
        })
        .to_string()
    }

    /// A token-endpoint mock returning a fixed short-lived access token.
    fn mock_token(server: &httpmock::MockServer) -> httpmock::Mock<'_> {
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/token");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({
                    "access_token": "ya29.test",
                    "expires_in": 3600,
                }));
        })
    }

    fn gcs_mock(server: &httpmock::MockServer) -> GcsCache {
        new_for_test(
            "bucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&format!("{}/token", server.base_url()))),
        )
        .unwrap()
    }

    fn mock_object_metadata<'a>(
        server: &'a httpmock::MockServer,
        bucket: &str,
        key: &str,
        size: usize,
        deadline: Option<u64>,
    ) -> httpmock::Mock<'a> {
        let path = format!(
            "/storage/v1/b/{bucket}/o/{}",
            encode_component(&format!("oxllm:{key}"))
        );
        let mut metadata = serde_json::json!({
            "generation": "123", "metageneration": "4", "size": size.to_string(),
            "metadata": {"oxllm_cache_version": CACHE_FORMAT_VERSION},
        });
        if let Some(deadline) = deadline {
            metadata["metadata"]["expires"] = serde_json::json!(deadline.to_string());
        }
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path(path)
                .query_param("alt", "json")
                .header("authorization", "Bearer ya29.test");
            then.status(200).json_body(metadata);
        })
    }

    #[test]
    fn object_name_applies_prefix() {
        let cache = GcsCache::new("bucket", "oxllm", None).unwrap();
        assert_eq!(
            cache.object_name("litellm:vcr:abc"),
            "oxllm:litellm:vcr:abc"
        );
    }

    #[test]
    fn constructor_rejects_empty_prefix() {
        for prefix in ["", " ", "\t\n"] {
            assert!(GcsCache::new("bucket", prefix, None).is_err());
        }
    }

    #[test]
    fn multipart_upload_keeps_metadata_and_verbatim_value_in_separate_parts() {
        let value = "{\"content\":\"café\"}\r\n";
        let (content_type, body) = multipart_upload("oxllm:k", value, Some(123)).unwrap();
        let boundary = content_type
            .strip_prefix("multipart/related; boundary=")
            .unwrap();
        let body = String::from_utf8(body).unwrap();
        let delimiter = format!("--{boundary}");
        let parts: Vec<_> = body.split(&delimiter).collect();
        assert_eq!(parts.len(), 4);
        let metadata: serde_json::Value = serde_json::from_str(
            parts[1]
                .split_once("\r\n\r\n")
                .unwrap()
                .1
                .trim_end_matches("\r\n"),
        )
        .unwrap();
        assert_eq!(metadata["name"], "oxllm:k");
        assert_eq!(metadata["metadata"]["expires"], "123");
        assert_eq!(
            metadata["metadata"]["oxllm_cache_version"],
            CACHE_FORMAT_VERSION
        );
        assert_eq!(
            parts[2]
                .split_once("\r\n\r\n")
                .unwrap()
                .1
                .strip_suffix("\r\n")
                .unwrap(),
            value
        );
        assert_eq!(parts[3], "--\r\n");
    }

    #[tokio::test]
    async fn get_rejects_a_generation_change_between_metadata_and_data() {
        let server = httpmock::MockServer::start();
        let cache = gcs_mock(&server);
        mock_token(&server);
        mock_object_metadata(&server, "bucket", "k", 5, None);
        let changed = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/bucket/o/oxllm%3Ak")
                .query_param("alt", "media")
                .query_param("ifGenerationMatch", "123")
                .query_param("ifMetagenerationMatch", "4");
            then.status(412);
        });
        let unguarded = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/bucket/o/oxllm%3Ak");
            then.status(200).body("stale");
        });
        assert!(cache.get("k").await.is_none());
        changed.assert_hits(1);
        unguarded.assert_hits(0);
    }

    #[tokio::test]
    async fn get_rejects_legacy_or_malformed_expiry_metadata() {
        for metadata in [
            serde_json::json!({}),
            serde_json::json!({"oxllm_cache_version": CACHE_FORMAT_VERSION, "expires": "invalid"}),
        ] {
            let server = httpmock::MockServer::start();
            let cache = gcs_mock(&server);
            mock_token(&server);
            server.mock(|when, then| {
                when.method(httpmock::Method::GET)
                    .query_param("alt", "json");
                then.status(200).json_body(serde_json::json!({
                    "generation": "123", "metageneration": "4", "size": "5", "metadata": metadata,
                }));
            });
            let data = server.mock(|when, then| {
                when.method(httpmock::Method::GET)
                    .query_param("alt", "media");
                then.status(200).body("stale");
            });
            assert!(cache.get("k").await.is_none());
            data.assert_hits(0);
        }
    }

    #[tokio::test]
    async fn write_cap_matches_read_cap_in_bytes() {
        let server = httpmock::MockServer::start();
        let mut cache = gcs_mock(&server);
        cache.max_object_size = 4;
        let token = mock_token(&server);
        mock_object_metadata(&server, "bucket", "k", 4, None);
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/upload/storage/v1/b/bucket/o");
            then.status(200);
        });
        let get = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/bucket/o/oxllm%3Ak");
            then.status(200).body("éé");
        });
        cache.set("k", "ééx".to_string(), None).await;
        token.assert_hits(0);
        put.assert_hits(0);
        cache.set("k", "éé".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("éé"));
        put.assert_hits(1);
        get.assert_hits(1);
    }

    #[tokio::test]
    async fn zero_ttl_invalidates_instead_of_uploading() {
        let server = httpmock::MockServer::start();
        let cache = gcs_mock(&server);
        mock_token(&server);
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/upload/storage/v1/b/bucket/o");
            then.status(200);
        });
        let delete = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path("/storage/v1/b/bucket/o/oxllm%3Ak");
            then.status(204);
        });
        cache.set("k", "old value".to_string(), None).await;
        cache
            .set("k", "new value".to_string(), Some(Duration::ZERO))
            .await;
        put.assert_hits(1);
        delete.assert_hits(1);
    }

    #[tokio::test]
    async fn all_object_operations_share_a_short_total_deadline() {
        let server = httpmock::MockServer::start();
        let cache = gcs_mock(&server);
        mock_token(&server);
        server.mock(|_, then| {
            then.status(503).delay(Duration::from_secs(2));
        });
        let (get, (), (), ping) = tokio::time::timeout(
            CACHE_OPERATION_TIMEOUT + Duration::from_millis(500),
            async {
                tokio::join!(
                    cache.get("k"),
                    cache.set("k", "v".to_string(), None),
                    cache.delete("k"),
                    cache.ping(),
                )
            },
        )
        .await
        .expect("cache operations must not wait through the transport retry ladder");
        assert!(get.is_none());
        assert!(!ping);
    }

    #[tokio::test]
    async fn timeout_negative_caches_cancelled_token_refresh() {
        let server = httpmock::MockServer::start();
        let cache = gcs_mock(&server);
        let token = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/token");
            then.status(503).delay(Duration::from_secs(2));
        });
        assert!(cache.get("k").await.is_none());
        assert!(
            tokio::time::timeout(Duration::from_millis(200), cache.get("k"))
                .await
                .expect("cancelled refresh must leave failure backoff")
                .is_none()
        );
        token.assert_hits(1);
    }

    #[test]
    fn encode_component_encodes_colon_and_slash() {
        assert_eq!(encode_component("oxllm:k/x"), "oxllm%3Ak%2Fx");
    }

    #[test]
    fn encode_component_encodes_non_ascii() {
        // 'é' is two UTF-8 bytes (0xC3 0xA9); both must be percent-encoded.
        assert_eq!(encode_component("café"), "caf%C3%A9");
    }

    #[test]
    fn object_path_builds_canonical_resource() {
        let cache = GcsCache::new("mybucket", "oxllm", None).unwrap();
        let path = cache.object_path("oxllm:k1");
        assert_eq!(path, "/storage/v1/b/mybucket/o/oxllm%3Ak1");
    }

    #[test]
    fn new_rejects_malformed_service_account_json() {
        let res = GcsCache::new("bucket", "oxllm", Some("not json"));
        assert!(res.is_err());
    }

    #[test]
    fn new_rejects_malformed_private_key_pem() {
        let sa = serde_json::json!({
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "private_key": "not-a-valid-pem",
        })
        .to_string();
        let res = GcsCache::new("bucket", "oxllm", Some(&sa));
        assert!(res.is_err(), "garbage private_key PEM must be rejected");
    }

    #[test]
    fn new_rejects_http_token_uri() {
        let res = GcsCache::new(
            "bucket",
            "oxllm",
            Some(&sa_json("http://evil.example.com/token")),
        );
        assert!(res.is_err(), "plaintext token_uri must be rejected");
    }

    #[test]
    fn new_rejects_ip_literal_token_uri() {
        let res = GcsCache::new(
            "bucket",
            "oxllm",
            Some(&sa_json("https://169.254.169.254/token")),
        );
        assert!(res.is_err(), "IP-literal token_uri must be rejected");
    }

    #[test]
    fn new_accepts_default_token_uri_when_omitted() {
        let dummy_key = include_str!("../tests/data/dummy_rsa_key.pem");
        let sa_json = serde_json::json!({
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "private_key": dummy_key,
            "private_key_id": "kid123",
        })
        .to_string();
        let cache = GcsCache::new("mybucket", "oxllm", Some(&sa_json));
        assert!(cache.is_ok(), "valid service-account key should construct");
    }

    #[tokio::test]
    async fn incr_returns_err_not_silent_success() {
        let cache = GcsCache::new("bucket", "oxllm", None).unwrap();
        assert!(cache.incr("c", 5.0, None).await.is_err());
    }

    #[tokio::test]
    async fn incr_if_below_returns_err() {
        let cache = GcsCache::new("bucket", "oxllm", None).unwrap();
        assert!(cache.incr_if_below("c", 5.0, 1.0, None).await.is_err());
    }

    #[tokio::test]
    async fn counter_value_is_none() {
        let cache = GcsCache::new("bucket", "oxllm", None).unwrap();
        assert!(cache.counter_value("c").await.is_none());
    }

    /// `try_counter_value` must short-circuit to `Ok(None)` rather than routing
    /// through the trait default (a full GCS GET + parse). Asserts both the
    /// result and that **zero** object GETs are issued.
    #[tokio::test]
    async fn try_counter_value_returns_ok_none_without_network() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ac");
            then.status(200).body("1.0");
        });

        assert_eq!(cache.try_counter_value("c").await.unwrap(), None);
        get_mock.assert_hits(0);
    }

    /// `get` must refuse to read an object whose body exceeds the size cap,
    /// returning `None` without allocating the full body.
    #[tokio::test]
    async fn get_returns_none_when_object_exceeds_max_size() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let mut cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();
        // Tight cap so the oversized body trips it.
        cache.max_object_size = 4;

        mock_token(&server);
        let metadata = mock_object_metadata(&server, "mybucket", "big", 100, None);
        let oversized = "x".repeat(100);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Abig");
            then.status(200).body(oversized);
        });

        assert!(cache.get("big").await.is_none());
        metadata.assert_hits(1);
        get_mock.assert_hits(0);
    }

    /// A cache miss (404) degrades to `None`.
    #[tokio::test]
    async fn get_returns_none_on_404() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        mock_token(&server);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Amissing");
            then.status(404);
        });

        assert!(cache.get("missing").await.is_none());
        get_mock.assert_hits(1);
    }

    /// Two sequential reads within the token lifetime must exchange the token
    /// exactly once (the second read reuses the cached token).
    #[tokio::test]
    async fn token_is_cached_across_operations() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        let token_mock = mock_token(&server);
        let metadata = mock_object_metadata(&server, "mybucket", "k", "cached-body".len(), None);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ak");
            then.status(200).body("cached-body");
        });

        assert_eq!(cache.get("k").await.as_deref(), Some("cached-body"));
        assert_eq!(cache.get("k").await.as_deref(), Some("cached-body"));

        // Object fetched twice, but the token exchanged only once.
        get_mock.assert_hits(2);
        metadata.assert_hits(2);
        token_mock.assert_hits(1);
    }

    /// A 401 (revoked/rotated token) must invalidate the cached token and retry
    /// the operation exactly once. This exercises the retry guard: against a
    /// persistent 401 there are precisely two object GETs and two token POSTs
    /// (proving both the single retry bound and the token-cache invalidation),
    /// and the op degrades to `None` instead of looping. httpmock 0.7 has no
    /// sequence support, so the persistent-401 form is used rather than a
    /// 401-then-200 sequence; it gives stronger coverage of the loop bound.
    #[tokio::test]
    async fn get_retries_once_on_persistent_401_then_gives_up() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        let token_mock = mock_token(&server);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ak");
            then.status(401);
        });

        assert!(cache.get("k").await.is_none());

        // Exactly one retry (two GETs total) — never an unbounded retry storm.
        get_mock.assert_hits(2);
        // Token cache was invalidated on the 401, so a fresh token was fetched.
        token_mock.assert_hits(2);
    }

    #[tokio::test]
    async fn ping_returns_true_on_200() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        mock_token(&server);
        let meta_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket");
            then.status(200);
        });

        assert!(cache.ping().await);
        meta_mock.assert_hits(1);
    }

    #[tokio::test]
    async fn ping_returns_false_on_403() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        mock_token(&server);
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket");
            then.status(403);
        });

        assert!(
            !cache.ping().await,
            "403 (bad credentials) must be unhealthy"
        );
    }

    #[tokio::test]
    async fn ping_returns_false_on_404() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        mock_token(&server);
        server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket");
            then.status(404);
        });

        assert!(
            !cache.ping().await,
            "404 (missing bucket) must be unhealthy"
        );
    }

    #[test]
    fn new_rejects_invalid_bucket_name() {
        for bad in ["/slash", "UPPER", "_under", "a..b", "ab", "no/slash/here"] {
            assert!(
                GcsCache::new(bad, "oxllm", None).is_err(),
                "bucket {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn new_accepts_valid_bucket_name() {
        for good in [
            "bkt",
            "my-bucket",
            "a.b.c",
            "bucket_1",
            "x".repeat(63).as_str(),
        ] {
            assert!(
                GcsCache::new(good, "oxllm", None).is_ok(),
                "bucket {good:?} must be accepted"
            );
        }
    }

    /// A failed token fetch is negative-cached: the second `get` must not
    /// re-issue the network call (which would serialize behind the lock).
    #[tokio::test]
    async fn token_failure_is_negative_cached() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        let token_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/token");
            then.status(500).body("internal error");
        });

        assert!(cache.get("k").await.is_none());
        assert!(cache.get("k").await.is_none());

        token_mock.assert_hits(1);
    }

    /// N concurrent `get`s against a dead token endpoint must coalesce into a
    /// single fetch, not N serialized stalls. Without negative caching each
    /// caller would wait `N × timeout` before degrading to a miss.
    #[tokio::test]
    async fn token_failure_coalesces_concurrent_calls() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = std::sync::Arc::new(
            new_for_test(
                "mybucket",
                "oxllm",
                &server.base_url(),
                Some(&sa_json(&token_uri)),
            )
            .unwrap(),
        );

        let token_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/token");
            then.status(500);
        });

        let mut set = tokio::task::JoinSet::new();
        for _ in 0..10 {
            let c = cache.clone();
            set.spawn(async move { c.get("k").await });
        }
        while set.join_next().await.is_some() {}

        token_mock.assert_hits(1);
    }

    /// The error-body helper must stop reading after [`MAX_ERROR_BODY`] bytes
    /// even when the response is much larger.
    #[tokio::test]
    async fn read_error_body_truncates_at_cap() {
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/err");
            then.status(500).body("x".repeat(MAX_ERROR_BODY * 4));
        });
        let resp = reqwest::get(format!("{}/err", server.base_url()))
            .await
            .unwrap();
        let body = read_error_body(resp).await;
        assert_eq!(body.len(), MAX_ERROR_BODY);
    }

    /// A successful `set` → `get` → `delete` round-trip against a local mock,
    /// exercising the upload URL, the object resource path, and idempotent
    /// delete (204) — none of which were covered before.
    #[tokio::test]
    async fn get_set_delete_round_trip_via_mock() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        mock_token(&server);
        mock_object_metadata(&server, "mybucket", "k", "round-trip-value".len(), None);
        let set_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/upload/storage/v1/b/mybucket/o")
                .query_param("uploadType", "multipart")
                .query_param("name", "oxllm:k");
            then.status(200);
        });
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ak")
                .query_param("alt", "media")
                .query_param("ifGenerationMatch", "123")
                .query_param("ifMetagenerationMatch", "4");
            then.status(200).body("round-trip-value");
        });
        let del_mock = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ak");
            then.status(204);
        });

        cache.set("k", "round-trip-value".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("round-trip-value"));
        cache.delete("k").await;

        set_mock.assert_hits(1);
        get_mock.assert_hits(1);
        del_mock.assert_hits(1);
    }

    /// TTL is persisted in multipart metadata, not a header ignored by the
    /// data-only upload API. An expired metadata response skips downloading.
    #[tokio::test]
    async fn set_ttl_stamps_expires_metadata_and_get_honors_it() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        mock_token(&server);
        let set_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/upload/storage/v1/b/mybucket/o")
                .query_param("name", "oxllm:k")
                .query_param("uploadType", "multipart")
                .body_contains("\"oxllm_cache_version\":\"2\"")
                .body_contains("\"expires\":");
            then.status(200);
        });
        cache
            .set("k", "v".to_string(), Some(Duration::from_secs(60)))
            .await;
        set_mock.assert();

        let metadata = mock_object_metadata(&server, "mybucket", "k", 5, Some(0));
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ak")
                .query_param("alt", "media");
            then.status(200)
                .header("x-goog-meta-expires", "0")
                .body("stale");
        });
        assert_eq!(cache.get("k").await, None);
        metadata.assert_hits(1);
        get_mock.assert_hits(0);
    }

    /// Every object operation must carry the bearer token — the regression
    /// guard that would have caught the C40/S3 CRITICAL (computed-then-dropped
    /// credential).
    #[tokio::test]
    async fn bearer_token_attached_to_all_operations() {
        let server = httpmock::MockServer::start();
        let token_uri = format!("{}/token", server.base_url());
        let cache = new_for_test(
            "mybucket",
            "oxllm",
            &server.base_url(),
            Some(&sa_json(&token_uri)),
        )
        .unwrap();

        mock_token(&server);
        mock_object_metadata(&server, "mybucket", "k", 1, None);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ak")
                .header_exists("authorization");
            then.status(200).body("v");
        });
        let set_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/upload/storage/v1/b/mybucket/o")
                .header_exists("authorization");
            then.status(200);
        });
        let del_mock = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path("/storage/v1/b/mybucket/o/oxllm%3Ak")
                .header_exists("authorization");
            then.status(204);
        });

        cache.get("k").await;
        cache.set("k", "v".to_string(), None).await;
        cache.delete("k").await;

        get_mock.assert();
        set_mock.assert();
        del_mock.assert();
    }
}
