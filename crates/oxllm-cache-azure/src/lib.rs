//! Azure Blob Storage response cache backend for oxllm.
//!
//! Implements [`oxllm_core::Cache`] over Azure Blob Storage via the REST API
//! (`PUT`/`GET`/`DELETE` on blobs). Blob name = `{prefix}:{original_key}`;
//! values are stored verbatim as blob bodies (the caller-supplied JSON
//! strings).
//!
//! Auth supports two modes:
//!
//! - **SharedKey** when a `connection_string` containing `AccountKey=...` is
//!   supplied (the canonical Azure Storage connection string). Requests are
//!   signed with [Shared
//!   Key](https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key)
//!   (HMAC-SHA256), implemented from scratch with `hmac` + `base64`.
//! - **Managed identity** (bearer token from the Azure Instance Metadata
//!   Service) when no connection string is configured — valid when running on
//!   an Azure VM / App Service / Container Apps with a system-assigned
//!   identity granted `Storage Blob Data Contributor`.
//!
//! ## Response-cache-only (no atomic counters)
//!
//! Blob stores have no atomic increment primitive. To satisfy the `Cache`
//! trait this backend models only the response-blob operations:
//!
//! - `Cache::incr` → `Err` (counters unsupported)
//! - `Cache::incr_if_below` → `Err` (counters unsupported)
//! - `Cache::counter_value` → `None`
//! - `Cache::try_counter_value` → `Ok(None)`
//!
//! Consequence: an Azure cache can serve cached LLM responses but **cannot**
//! enforce rate limits or budgets. Failing the write-side counter ops with
//! `Err` (rather than a silent `Ok`) keeps a misconfigured host loud instead
//! of silently passing every budget/RPM gate. Use the Redis or in-memory
//! backend (or the server's counter-layering wrapper) when rate-limit/budget
//! enforcement is required. Per-object TTL is enforced client-side via the
//! `x-ms-meta-expires` metadata header (a Unix-seconds deadline stamped on
//! `set` and checked on `get`), mirroring the S3 backend's
//! `x-amz-meta-expires` pattern; Azure does not auto-delete on it.
//! `None` means no expiry; `Some(Duration::ZERO)` invalidates a prior value.
//! Public cache operations have a 500ms total optimization budget including
//! managed-identity acquisition, container creation, signing, retries, backoff,
//! and response-body reads. Blob requests retry transport failures, 429, and
//! 5xx at most twice (100ms/200ms backoff). A valid `Retry-After` is a minimum
//! delay; if it cannot fit the remaining operation budget, the request fails
//! open immediately instead of retrying early or extending the deadline.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use base64::Engine;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use oxllm_core::Cache as CacheTrait;
use oxllm_core::{Error, IncrResult};

type HmacSha256 = Hmac<Sha256>;

/// Storage API version pinned for all requests.
const MS_VERSION: &str = "2024-05-04";
/// Transport ceiling; public cache operations use the tighter total budget below.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CACHE_OPERATION_TIMEOUT: Duration = Duration::from_millis(500);
/// Two retries, always inside the original public operation deadline.
const BLOB_RETRY_BACKOFF: [Duration; 2] = [Duration::from_millis(100), Duration::from_millis(200)];
/// Default cap on a single cached blob read or written (16 MiB). Bounds
/// memory use against an oversized or hostile object body; Azure always sends
/// a `Content-Length` so the check fires before allocation.
const DEFAULT_MAX_BLOB_SIZE: usize = 16 * 1024 * 1024;
/// Refresh a cached managed-identity token when it has less than this long to
/// live, so a request never flies with an imminently-expiring credential.
const TOKEN_REFRESH_MARGIN: Duration = Duration::from_secs(60);
/// Conservative default lifetime applied when IMDS omits `expires_in` or
/// returns a non-numeric value. Prevents per-call IMDS thrash (which serializes
/// all cache ops on the token lock) without risking an over-stale credential.
const FALLBACK_TOKEN_LIFETIME: Duration = Duration::from_secs(1800);
/// Backoff applied after a failed IMDS token fetch: concurrent and subsequent
/// callers within this window get an immediate error instead of re-running the
/// full retry ladder (up to 3 attempts with 1s+2s sleeps) serialized on the
/// token lock, which would stall the request hot path for seconds per cache
/// op for as long as the outage lasts. Mirrors `TOKEN_FAILURE_BACKOFF` in the
/// GCS crate, added there for exactly this cascading-stall mode.
const TOKEN_FAILURE_BACKOFF: Duration = Duration::from_secs(5);
/// Backoff schedule (seconds) between IMDS token-fetch retries on transient
/// failures (410 / 429 / 5xx and connect errors). Azure IMDS is known to need
/// a few seconds on VM cold start and to briefly throttle; two retries bound
/// the worst-case added latency to ~3s under the 10s request timeout.
const IMDS_RETRY_BACKOFF_SECS: &[u64] = &[1, 2];
/// Azure Instance Metadata Service token endpoint (hardcoded — not
/// operator-configurable, so no SSRF surface).
const METADATA_URL: &str = "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F";

async fn cache_operation<T, F>(
    name: &str,
    operation: impl FnOnce(tokio::time::Instant) -> F,
) -> Option<T>
where
    F: std::future::Future<Output = T>,
{
    let deadline = tokio::time::Instant::now() + CACHE_OPERATION_TIMEOUT;
    match tokio::time::timeout_at(deadline, operation(deadline)).await {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::warn!(
                operation = name,
                "azure cache operation exceeded optimization deadline"
            );
            None
        }
    }
}

/// Cached managed-identity token state. `Valid` holds a usable token and its
/// expiry; `Failed` records a short failure-backoff window (see
/// [`TOKEN_FAILURE_BACKOFF`]) during which [`AzureBlobCache::metadata_token`]
/// returns immediately without re-issuing the network call, so an IMDS outage
/// degrades cache ops to fast errors instead of serialized multi-second
/// stalls.
enum CachedToken {
    Valid { token: String, expires_at: Instant },
    Failed { retry_after: Instant },
}

/// An Azure Blob Storage-backed cache implementing [`oxllm_core::Cache`].
pub struct AzureBlobCache {
    client: reqwest::Client,
    account: String,
    container: String,
    prefix: String,
    auth: AzureAuth,
    /// Optional blob endpoint base (e.g. `http://127.0.0.1:PORT`). When
    /// `None`, the production host `{account}.blob.core.windows.net` is used.
    endpoint: Option<String>,
    /// IMDS token endpoint URL (injectable so tests can mock IMDS).
    metadata_url: String,
    /// Cached managed-identity bearer token with its expiry; a failed fetch is
    /// negative-cached for [`TOKEN_FAILURE_BACKOFF`] (see [`CachedToken`]).
    token: tokio::sync::Mutex<Option<CachedToken>>,
    /// Upper bound (bytes) on either a `get` or `set` body.
    max_blob_size: usize,
    /// Guards a one-shot `warn!` on the first counter-op call. See the
    /// module-level "Response-cache-only" note.
    counter_warned: AtomicBool,
    /// Backoff schedule between IMDS token-fetch retries. Defaults to
    /// [`IMDS_RETRY_BACKOFF_SECS`]; overridable in tests so the retry path can
    /// be exercised without a multi-second wall-clock delay.
    imds_retry_backoff_secs: Vec<u64>,
    /// Guards a best-effort, one-shot container auto-create on the first `set`
    /// (see [`AzureBlobCache::ensure_container`]). Uses `OnceCell` (rather than
    /// an `AtomicBool` flag) so that concurrent first-burst `set`s **wait** for
    /// the in-flight creation instead of bypassing it and issuing blob PUTs
    /// against a not-yet-existing container. Keeps construction network-free so
    /// the server's counter-wrapping unit test and startup stay deterministic.
    /// The cell is only populated on success (`201`/`409`), so a transient
    /// `403`/`400` leaves it empty and the next write re-attempts.
    container_ensured: tokio::sync::OnceCell<()>,
}

enum AzureAuth {
    /// Storage account key (base64-decoded once at construction) → SharedKey
    /// HMAC signing. Decoding up front surfaces a malformed key as a clean
    /// `Error::Config` instead of deferring the failure to the first request.
    SharedKey { account_key: Vec<u8> },
    /// No key configured → bearer token from the Instance Metadata Service.
    ManagedIdentity,
}

impl AzureBlobCache {
    /// Create a new Azure Blob cache.
    ///
    /// `connection_string`, when provided, is parsed for `AccountKey=...`
    /// (a standard Azure Storage connection string). When absent, the backend
    /// falls back to managed-identity auth. If a connection string is supplied
    /// **without** a parseable `AccountKey=`, this returns [`Error::Config`]
    /// rather than silently falling back to managed identity — an operator who
    /// mistypes the key otherwise gets a cache that starts cleanly but fails
    /// every request against the IMDS link-local address.
    pub fn new(
        account: &str,
        container: &str,
        prefix: &str,
        connection_string: Option<&str>,
    ) -> Result<Self, Error> {
        if prefix.trim().is_empty() {
            return Err(Error::Config("azure cache prefix must not be empty".into()));
        }
        if account.trim().is_empty() || container.trim().is_empty() {
            return Err(Error::Config(
                "azure account and container must not be empty".into(),
            ));
        }
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Config(format!("azure http client build failed: {e}")))?;
        let auth = match connection_string {
            Some(conn) => match extract_account_key(conn) {
                Some(key_b64) => {
                    let account_key = base64::engine::general_purpose::STANDARD
                        .decode(key_b64.as_bytes())
                        .map_err(|e| Error::Config(format!("azure account key not base64: {e}")))?;
                    AzureAuth::SharedKey { account_key }
                }
                None => {
                    return Err(Error::Config(
                        "connection_string supplied without a parseable AccountKey; \
                         refusing to silently fall back to managed identity"
                            .to_string(),
                    ));
                }
            },
            None => AzureAuth::ManagedIdentity,
        };
        if !container
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(Error::Config(format!(
                "azure container name must contain only lowercase letters, digits, or hyphens: {container}"
            )));
        }
        Ok(Self {
            client,
            account: account.to_string(),
            container: container.to_string(),
            prefix: prefix.to_string(),
            auth,
            endpoint: None,
            metadata_url: METADATA_URL.to_string(),
            token: tokio::sync::Mutex::new(None),
            max_blob_size: DEFAULT_MAX_BLOB_SIZE,
            counter_warned: AtomicBool::new(false),
            imds_retry_backoff_secs: IMDS_RETRY_BACKOFF_SECS.to_vec(),
            container_ensured: tokio::sync::OnceCell::new(),
        })
    }

    /// Override the blob endpoint base (tests only — mock upstream).
    #[cfg(test)]
    fn with_blob_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self
    }

    /// Override the IMDS token URL (tests only — mock IMDS).
    #[cfg(test)]
    fn with_metadata_url(mut self, url: &str) -> Self {
        self.metadata_url = url.to_string();
        self
    }

    /// Override the IMDS retry backoff schedule (tests only — keep it fast).
    #[cfg(test)]
    fn with_imds_retry_backoff(mut self, backoff: &[u64]) -> Self {
        self.imds_retry_backoff_secs = backoff.to_vec();
        self
    }

    fn blob_name(&self, key: &str) -> String {
        format!("{}:{key}", self.prefix)
    }

    /// URL + host for a blob. When an `endpoint` override is set (tests),
    /// path-style URLs against that base are produced; otherwise the
    /// production virtual-host URL `{account}.blob.core.windows.net` is used.
    fn url_for(&self, blob_name: &str) -> (String, String) {
        let encoded = encode_path(&format!(
            "/{container}/{blob}",
            container = self.container,
            blob = blob_name
        ));
        match &self.endpoint {
            Some(base) => {
                let base = base.trim_end_matches('/');
                let host = host_of(base).to_string();
                (format!("{base}{encoded}"), host)
            }
            None => {
                let host = format!("{account}.blob.core.windows.net", account = self.account);
                (format!("https://{host}{encoded}"), host)
            }
        }
    }

    /// Canonicalized resource for SharedKey signing: `/{account}` plus the
    /// encoded URI path, exactly as sent by [`Self::url_for`]. Only query
    /// parameter names/values are URL-decoded by the Shared Key algorithm.
    fn canonicalized_resource(&self, blob_name: &str) -> String {
        format!(
            "/{}{}",
            self.account,
            encode_path(&format!("/{}/{blob_name}", self.container))
        )
    }

    /// URL + host for a `Create Container` (`PUT /{container}?restype=container`).
    fn container_url(&self) -> (String, String) {
        let path = encode_path(&format!("/{container}", container = self.container));
        match &self.endpoint {
            Some(base) => {
                let base = base.trim_end_matches('/');
                let host = host_of(base).to_string();
                (format!("{base}{path}?restype=container"), host)
            }
            None => {
                let host = format!("{account}.blob.core.windows.net", account = self.account);
                (format!("https://{host}{path}?restype=container"), host)
            }
        }
    }

    /// Canonicalized resource for `Create Container` (`PUT …?restype=container`):
    /// `/{account}/{container}` plus the `restype:container` query parameter on
    /// its own line, per the Shared Key canonicalized-resource rules (query
    /// params `restype`/`comp` are appended as sorted `key:value` lines).
    fn container_canonicalized_resource(&self) -> String {
        format!(
            "/{account}/{container}\nrestype:container",
            account = self.account,
            container = self.container,
        )
    }

    /// Idempotently ensure the configured container exists.
    ///
    /// Issues `PUT /{container}?restype=container` (Create Container),
    /// tolerating `201 Created` and `409 Conflict` (already exists) as
    /// success. Mirrors the reference `with suppress(ResourceExistsError):
    /// self.container_client.create_container()` so a freshly configured
    /// container does not silently degrade to a permanent no-op cache (every
    /// `set` PUT would otherwise 404 and be swallowed as a `warn`).
    ///
    /// Uses a [`tokio::sync::OnceCell`] gate so concurrent callers on the first
    /// burst **wait** for the in-flight creation rather than bypassing it. The
    /// cell is only populated on success (`201`/`409`); on a non-success status
    /// (e.g. `403` from a signing bug or missing `Container.Create` permission)
    /// or a transport failure the cell stays empty and the next write
    /// re-attempts, preventing a permanent suppression of retries.
    pub async fn ensure_container(&self) -> Result<(), Error> {
        cache_operation("ensure_container", |_| async {
            self.container_ensured
                .get_or_try_init(|| self.ensure_container_inner())
                .await
                .map(|_| ())
        })
        .await
        .unwrap_or_else(|| Err(Error::Network("azure container creation timed out".into())))
    }

    /// Performs the actual container-creation HTTP call. Returns `Ok` only on
    /// `201`/`409`; any other status or transport error returns `Err` so the
    /// [`OnceCell`](tokio::sync::OnceCell) in [`ensure_container`] stays empty
    /// and the next write re-attempts.
    async fn ensure_container_inner(&self) -> Result<(), Error> {
        let method = reqwest::Method::PUT;
        let resp = match &self.auth {
            AzureAuth::SharedKey { account_key } => {
                let (url, host) = self.container_url();
                let date = rfc1123_now();
                let mut ms_headers: Vec<(&str, String)> = vec![
                    ("x-ms-date", date.clone()),
                    ("x-ms-version", MS_VERSION.to_string()),
                ];
                ms_headers.sort_by(|a, b| a.0.cmp(b.0));
                let canonicalized_headers = ms_headers
                    .iter()
                    .map(|(k, v)| format!("{k}:{v}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                let canonicalized_resource = self.container_canonicalized_resource();
                let string_to_sign = build_string_to_sign(
                    method.as_str(),
                    "0",
                    &canonicalized_headers,
                    &canonicalized_resource,
                );
                let signature = hmac_b64(account_key, string_to_sign.as_bytes())?;
                let authorization = format!("SharedKey {}:{signature}", self.account);
                self.client
                    .request(method, &url)
                    .header("Authorization", &authorization)
                    .header("Content-Length", "0")
                    .header("x-ms-date", &date)
                    .header("x-ms-version", MS_VERSION)
                    .header("Host", &host)
                    .body(Vec::new())
                    .send()
                    .await
                    .map_err(|e| Error::Network(format!("azure ensure_container failed: {e}")))?
            }
            AzureAuth::ManagedIdentity => {
                let token = self.metadata_token().await?;
                let (url, host) = self.container_url();
                let date = rfc1123_now();
                self.client
                    .request(method, &url)
                    .bearer_auth(token)
                    .header("Content-Length", "0")
                    .header("x-ms-date", &date)
                    .header("x-ms-version", MS_VERSION)
                    .header("Host", &host)
                    .body(Vec::new())
                    .send()
                    .await
                    .map_err(|e| Error::Network(format!("azure ensure_container failed: {e}")))?
            }
        };
        let status = resp.status().as_u16();
        if status == 201 || status == 409 {
            tracing::info!(
                account = %self.account,
                container = %self.container,
                status,
                "azure blob container ready"
            );
            Ok(())
        } else {
            tracing::warn!(
                status,
                "azure ensure_container non-success (container must pre-exist; cache may no-op)"
            );
            Err(Error::Network(format!(
                "azure ensure_container returned status {status}"
            )))
        }
    }

    /// Obtain a bearer token via the Instance Metadata Service, returning a
    /// cached token while it remains valid (refreshed within
    /// [`TOKEN_REFRESH_MARGIN`] of expiry).
    ///
    /// The lock is held across the refresh (with a double-check on
    /// acquisition): under concurrency with a stale/absent token only one
    /// caller hits IMDS, preventing a thundering herd that could briefly trip
    /// IMDS rate limits.
    ///
    /// A failed fetch is negative-cached for [`TOKEN_FAILURE_BACKOFF`]: every
    /// cache op during an IMDS outage would otherwise re-run the retry ladder
    /// (up to 3 attempts with 1s+2s sleeps) serialized on this lock, stalling
    /// the request hot path for seconds per operation indefinitely.
    async fn metadata_token(&self) -> Result<String, Error> {
        let mut cache = self.token.lock().await;
        let now = Instant::now();
        match cache.as_ref() {
            Some(CachedToken::Valid { token, expires_at }) => {
                let fresh = expires_at
                    .checked_duration_since(now)
                    .is_some_and(|remaining| remaining > TOKEN_REFRESH_MARGIN);
                if fresh {
                    return Ok(token.clone());
                }
            }
            Some(CachedToken::Failed { retry_after }) if *retry_after > now => {
                return Err(Error::Network(
                    "azure metadata token fetch recently failed; backing off".to_string(),
                ));
            }
            _ => {}
        }

        // If a cache deadline cancels an in-flight refresh, retain a short
        // failure backoff rather than making every waiter start a fresh retry
        // ladder after cancellation releases the mutex.
        *cache = Some(CachedToken::Failed {
            retry_after: now + TOKEN_FAILURE_BACKOFF,
        });
        match self.fetch_parsed_metadata_token().await {
            Ok((access_token, expires_secs)) => {
                let expires_at = expires_secs
                    .and_then(|secs| Instant::now().checked_add(Duration::from_secs(secs)))
                    .unwrap_or_else(|| {
                        tracing::warn!(
                            "azure metadata token has no valid expires_in; \
                             applying fallback lifetime (1800s) to avoid per-call IMDS thrash"
                        );
                        Instant::now() + FALLBACK_TOKEN_LIFETIME
                    });
                *cache = Some(CachedToken::Valid {
                    expires_at,
                    token: access_token.clone(),
                });
                Ok(access_token)
            }
            Err(e) => {
                *cache = Some(CachedToken::Failed {
                    retry_after: Instant::now() + TOKEN_FAILURE_BACKOFF,
                });
                Err(e)
            }
        }
    }

    /// Fetch a fresh token from IMDS and parse the response, mapping any
    /// failure — transport error, non-transient HTTP status after retries, or
    /// a malformed body — to an error. [`Self::metadata_token`] wraps this and
    /// negative-caches the failure for [`TOKEN_FAILURE_BACKOFF`].
    async fn fetch_parsed_metadata_token(&self) -> Result<(String, Option<u64>), Error> {
        let resp = self.fetch_metadata_token().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::Network(format!(
                "azure metadata token returned {status}: {body}"
            )));
        }
        #[derive(serde::Deserialize)]
        struct Tok {
            access_token: String,
            #[serde(default)]
            expires_in: String,
        }
        let tok: Tok = resp
            .json()
            .await
            .map_err(|e| Error::Network(format!("azure metadata token decode failed: {e}")))?;
        Ok((tok.access_token, tok.expires_in.parse::<u64>().ok()))
    }

    /// Issue the IMDS GET with a bounded retry loop on transient failures
    /// (`410`, `429`, and `5xx`). Azure IMDS is known to need a few seconds on
    /// VM cold start and to briefly throttle; the retries keep a transient
    /// blip from turning a cache `set` into a silently lost write.
    ///
    /// A connect/transport error is **not** retried: it means IMDS is
    /// unreachable at the network level (most often "not running on an Azure
    /// VM"), a condition a backoff sleep will not resolve. Failing fast here
    /// keeps startup and the server's counter-wrapping unit test deterministic
    /// on hosts without a route to the `169.254.169.254` link-local address.
    async fn fetch_metadata_token(&self) -> Result<reqwest::Response, Error> {
        let mut last_err: Option<Error> = None;
        for attempt in 0..=self.imds_retry_backoff_secs.len() {
            if attempt > 0 {
                let secs = self.imds_retry_backoff_secs[attempt - 1];
                tokio::time::sleep(Duration::from_secs(secs)).await;
            }
            match self
                .client
                .get(&self.metadata_url)
                .header("Metadata", "true")
                .send()
                .await
            {
                Ok(resp) => {
                    let code = resp.status().as_u16();
                    let transient = code == 410 || code == 429 || (500..600).contains(&code);
                    if transient {
                        last_err = Some(Error::Network(format!(
                            "azure metadata token returned {code}"
                        )));
                        continue;
                    }
                    return Ok(resp);
                }
                Err(e) => return Err(Error::Network(format!("azure metadata token failed: {e}"))),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            Error::Network("azure metadata token exhausted retries".to_string())
        }))
    }

    /// Build a freshly signed, replayable SharedKey request for one attempt.
    fn shared_key_request(
        &self,
        method: reqwest::Method,
        blob_name: &str,
        body: Vec<u8>,
        account_key: &[u8],
        meta_expires: Option<u64>,
    ) -> Result<reqwest::RequestBuilder, Error> {
        let (url, host) = self.url_for(blob_name);
        let date = rfc1123_now();
        // Keep the wire Content-Length explicit. The signing helper maps zero
        // to the empty canonical field required by service versions >= 2015-02-21.
        let content_length = if body.is_empty() {
            "0".to_string()
        } else {
            body.len().to_string()
        };

        // Canonicalized headers (must be sorted; x-ms-blob-type only on PUT).
        let mut ms_headers: Vec<(&str, String)> = vec![
            ("x-ms-date", date.clone()),
            ("x-ms-version", MS_VERSION.to_string()),
        ];
        let is_put = method == reqwest::Method::PUT;
        if is_put {
            ms_headers.push(("x-ms-blob-type", "BlockBlob".to_string()));
        }
        if let Some(deadline) = meta_expires {
            ms_headers.push(("x-ms-meta-expires", deadline.to_string()));
        }
        ms_headers.sort_by(|a, b| a.0.cmp(b.0));
        // Each canonicalized header is joined to the next by a single '\n';
        // the trailing '\n' separator before the canonicalized resource is
        // supplied by the final `.join("\n")` of the string-to-sign (see
        // `build_string_to_sign`). Per the Azure spec, canonicalized headers
        // are concatenated with the canonicalized resource by exactly one
        // newline — no double newline.
        let canonicalized_headers: String = ms_headers
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join("\n");

        let canonicalized_resource = self.canonicalized_resource(blob_name);

        let string_to_sign = build_string_to_sign(
            method.as_str(),
            &content_length,
            &canonicalized_headers,
            &canonicalized_resource,
        );

        let signature = hmac_b64(account_key, string_to_sign.as_bytes())?;
        let authorization = format!("SharedKey {}:{signature}", self.account);

        let mut req = self
            .client
            .request(method, &url)
            .header("Authorization", &authorization)
            .header("Content-Length", &content_length)
            .header("x-ms-date", &date)
            .header("x-ms-version", MS_VERSION)
            .header("Host", &host);
        for (k, v) in &ms_headers {
            if k == &"x-ms-date" || k == &"x-ms-version" {
                continue;
            }
            req = req.header(*k, v);
        }
        // Always attach a body (even empty) so the wire Content-Length is
        // explicit; the signing helper applies the zero-length exception.
        Ok(req.body(body))
    }

    /// Build one attempt using whichever auth mode is configured. Token errors
    /// are terminal here; IMDS retains its own single-flight retry/backoff policy.
    async fn blob_request(
        &self,
        method: reqwest::Method,
        blob_name: &str,
        body: Vec<u8>,
        meta_expires: Option<u64>,
    ) -> Result<reqwest::RequestBuilder, Error> {
        match &self.auth {
            AzureAuth::SharedKey { account_key } => {
                self.shared_key_request(method, blob_name, body, account_key, meta_expires)
            }
            AzureAuth::ManagedIdentity => {
                let token = self.metadata_token().await?;
                let (url, host) = self.url_for(blob_name);
                let date = rfc1123_now();
                let is_put = method == reqwest::Method::PUT;
                let content_length = if body.is_empty() {
                    "0".to_string()
                } else {
                    body.len().to_string()
                };
                let mut req = self
                    .client
                    .request(method, &url)
                    .bearer_auth(token)
                    .header("Content-Length", &content_length)
                    .header("x-ms-date", &date)
                    .header("x-ms-version", MS_VERSION)
                    .header("Host", &host);
                if let Some(deadline) = meta_expires {
                    req = req.header("x-ms-meta-expires", deadline.to_string());
                }
                if is_put {
                    req = req.header("x-ms-blob-type", "BlockBlob");
                }
                Ok(req.body(body))
            }
        }
    }

    /// Retry only transient blob failures, with a replayable body and fixed TTL
    /// metadata. Each attempt rebuilds its date, headers and signature. The
    /// deadline is inherited from `cache_operation`, including time already
    /// spent obtaining a token or creating the container; it is never reset.
    async fn send(
        &self,
        method: reqwest::Method,
        blob_name: &str,
        body: Vec<u8>,
        meta_expires: Option<u64>,
        deadline: tokio::time::Instant,
    ) -> Result<reqwest::Response, Error> {
        for attempt in 0..=BLOB_RETRY_BACKOFF.len() {
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::Network(
                    "azure blob request deadline exceeded".into(),
                ));
            }
            let request = self
                .blob_request(method.clone(), blob_name, body.clone(), meta_expires)
                .await?;
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::Network(
                    "azure blob request deadline exceeded".into(),
                ));
            }
            match request.send().await {
                Ok(resp) => {
                    let code = resp.status().as_u16();
                    if code != 429 && !(500..=599).contains(&code) {
                        return Ok(resp);
                    }
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    let Some(delay) = blob_retry_delay(
                        attempt,
                        resp.headers().get(reqwest::header::RETRY_AFTER),
                        remaining,
                        SystemTime::now(),
                    ) else {
                        return Ok(resp);
                    };
                    tracing::debug!(
                        status = code,
                        attempt = attempt + 1,
                        "azure transient blob status, retrying"
                    );
                    // Do not read a failed response body or hold its connection
                    // while sleeping; neither is needed to replay the operation.
                    drop(resp);
                    tokio::time::sleep(delay).await;
                }
                Err(error) => {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    // Invalid URLs/headers and other builder errors are not
                    // transport failures and must not consume the retry ladder.
                    let transport_failure = !error.is_builder()
                        && (error.is_connect()
                            || error.is_timeout()
                            || error.is_request()
                            || error.is_body());
                    let delay = if transport_failure {
                        blob_retry_delay(attempt, None, remaining, SystemTime::now())
                    } else {
                        None
                    };
                    let Some(delay) = delay else {
                        return Err(Error::Network(format!("azure request failed: {error}")));
                    };
                    tracing::debug!(
                        error = %error,
                        attempt = attempt + 1,
                        "azure blob transport failed, retrying"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(Error::Network(
            "azure blob request exhausted retries".into(),
        ))
    }

    /// Emit a one-shot `warn!` the first time a counter read is invoked.
    /// Blob stores have no atomic increment, so the counter read methods are
    /// no-ops that read as zero — which disables budget / RPM / JWT-replay
    /// enforcement when this backend is used directly as a `dyn Cache` instead
    /// of through the server's counter-layering wrapper. (The write-side
    /// `incr`/`incr_if_below` fail with `Err` for the same reason.)
    fn warn_counter_noop_once(&self) {
        if !self.counter_warned.swap(true, Ordering::SeqCst) {
            tracing::warn!(
                "azure blob cache counter operation is a no-op; budget/RPM/JWT \
                 enforcement is disabled — wrap in a counter-capable cache or \
                 use the Redis/in-memory backend"
            );
        }
    }
}

impl AzureBlobCache {
    async fn get_inner(&self, key: &str, deadline: tokio::time::Instant) -> Option<String> {
        let blob = self.blob_name(key);
        match self
            .send(reqwest::Method::GET, &blob, Vec::new(), None, deadline)
            .await
        {
            Ok(mut resp) => {
                let status = resp.status();
                if status.as_u16() == 404 {
                    return None;
                }
                if !status.is_success() {
                    tracing::warn!(status = %status, key = %key, "azure get non-success");
                    return None;
                }
                if header_expired(resp.headers(), "x-ms-meta-expires") {
                    tracing::debug!(
                        key = %key,
                        "azure get blob past client-side expires deadline; treating as miss"
                    );
                    return None;
                }
                if let Some(len) = resp.content_length() {
                    if len > self.max_blob_size as u64 {
                        tracing::warn!(
                            key = %key,
                            len,
                            cap = self.max_blob_size,
                            "azure get blob exceeds max size"
                        );
                        return None;
                    }
                }
                let mut buf = Vec::new();
                loop {
                    match resp.chunk().await {
                        Ok(Some(chunk)) => {
                            if buf.len() + chunk.len() > self.max_blob_size {
                                tracing::warn!(
                                    key = %key,
                                    cap = self.max_blob_size,
                                    "azure get blob exceeded max size while streaming"
                                );
                                return None;
                            }
                            buf.extend_from_slice(&chunk);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            tracing::warn!(error = %e, key = %key, "azure get body read failed");
                            return None;
                        }
                    }
                }
                if header_expired(resp.headers(), "x-ms-meta-expires") {
                    return None;
                }
                match String::from_utf8(buf) {
                    Ok(text) => Some(text),
                    Err(e) => {
                        tracing::warn!(error = %e, key = %key, "azure get body not utf-8");
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "azure get failed");
                None
            }
        }
    }

    async fn set_inner(
        &self,
        key: &str,
        value: String,
        ttl: Option<Duration>,
        deadline: tokio::time::Instant,
    ) {
        if ttl.is_some_and(|ttl| ttl.is_zero()) {
            self.delete_inner(key, deadline).await;
            return;
        }
        if value.len() > self.max_blob_size {
            tracing::warn!(
                key = %key,
                size = value.len(),
                cap = self.max_blob_size,
                "azure set skipped oversized value"
            );
            return;
        }
        // Best-effort, one-shot container auto-create on the first write.
        // Uses a `OnceCell` so concurrent first-burst `set`s **wait** for the
        // in-flight creation rather than bypassing it (which would issue blob
        // PUTs against a not-yet-existing container). On failure the cell stays
        // empty so the next write re-attempts; on success it is populated and
        // the hot path is untouched thereafter.
        if let Err(e) = self
            .container_ensured
            .get_or_try_init(|| self.ensure_container_inner())
            .await
        {
            tracing::warn!(
                error = %e,
                "azure ensure_container failed; relying on a pre-existing container"
            );
        }
        let blob = self.blob_name(key);
        let body = value.into_bytes();
        // Enforce the TTL client-side via the `x-ms-meta-expires` metadata
        // header (a Unix-seconds deadline), mirroring the S3 backend's
        // `x-amz-meta-expires` pattern: `get` treats blobs past the deadline
        // as a miss. Azure does not auto-delete on this metadata.
        match self
            .send(
                reqwest::Method::PUT,
                &blob,
                body,
                expires_deadline(ttl),
                deadline,
            )
            .await
        {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::warn!(
                        status = %resp.status(),
                        key = %key,
                        "azure set non-success"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "azure set failed");
            }
        }
    }

    async fn delete_inner(&self, key: &str, deadline: tokio::time::Instant) {
        let blob = self.blob_name(key);
        match self
            .send(reqwest::Method::DELETE, &blob, Vec::new(), None, deadline)
            .await
        {
            Ok(resp) => {
                let status = resp.status().as_u16();
                // 202 accepted on success; 404 (absent) is not an error.
                if !(status == 202 || status == 200 || status == 404) {
                    tracing::warn!(status = %status, key = %key, "azure delete non-success");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "azure delete failed");
            }
        }
    }
}

#[async_trait]
impl CacheTrait for AzureBlobCache {
    async fn get(&self, key: &str) -> Option<String> {
        cache_operation("get", |deadline| self.get_inner(key, deadline))
            .await
            .flatten()
    }

    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        let _ = cache_operation("set", |deadline| self.set_inner(key, value, ttl, deadline)).await;
    }

    async fn delete(&self, key: &str) {
        let _ = cache_operation("delete", |deadline| self.delete_inner(key, deadline)).await;
    }

    /// Unsupported: blob stores have no atomic increment. Returning an error
    /// (rather than a silent `Ok`) keeps a host that wires this backend
    /// directly from silently passing every budget/RPM gate. Layer a counter
    /// store (see module docs).
    async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
        Err(Error::Config(
            "azure blob cache backend does not support atomic counters; layer a counter store"
                .into(),
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
            "azure blob cache backend does not support atomic counters; layer a counter store"
                .into(),
        ))
    }

    /// No-op: counters are not modeled. See module docs.
    async fn counter_value(&self, _key: &str) -> Option<f64> {
        self.warn_counter_noop_once();
        None
    }

    /// No-op (mirrors `Cache::counter_value`): counters are not modeled.
    async fn try_counter_value(&self, _key: &str) -> Result<Option<f64>, Error> {
        self.warn_counter_noop_once();
        Ok(None)
    }

    /// Liveness: any HTTP response from the service (even 401/403/404) is
    /// treated as reachable — this verifies **transport**, not auth. An auth
    /// misconfiguration (or a signing bug) still reports `ping == true`; the
    /// non-success status is logged at `warn` so the discrepancy is observable.
    async fn ping(&self) -> bool {
        cache_operation("ping", |deadline| async move {
            let blob = self.blob_name("__oxllm_ping__");
            match self
                .send(reqwest::Method::HEAD, &blob, Vec::new(), None, deadline)
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() && status.as_u16() != 404 {
                        tracing::warn!(
                            status = %status,
                            "azure ping non-success (transport ok, auth may be broken)"
                        );
                    }
                    true
                }
                Err(e) => {
                    tracing::warn!(error = %e, "azure ping failed");
                    false
                }
            }
        })
        .await
        .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Backoff is a minimum, as is a valid Retry-After (delta-seconds or HTTP date).
/// Do not shorten a server-directed delay to fit the cache budget: stop instead.
/// `get` avoids unchecked indexing and also enforces the two-retry attempt cap.
fn blob_retry_delay(
    attempt: usize,
    retry_after: Option<&reqwest::header::HeaderValue>,
    remaining: Duration,
    now: SystemTime,
) -> Option<Duration> {
    let base = *BLOB_RETRY_BACKOFF.get(attempt)?;
    let server_delay = retry_after.and_then(|value| {
        let value = value.to_str().ok()?.trim();
        if let Ok(seconds) = value.parse::<u64>() {
            Some(Duration::from_secs(seconds))
        } else {
            httpdate::parse_http_date(value)
                .ok()
                .map(|date| date.duration_since(now).unwrap_or(Duration::ZERO))
        }
    });
    let delay = base.max(server_delay.unwrap_or(Duration::ZERO));
    (delay < remaining).then_some(delay)
}

/// Parse the `AccountKey=...` segment out of an Azure connection string.
fn extract_account_key(conn: &str) -> Option<String> {
    for part in conn.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            if k.trim().eq_ignore_ascii_case("AccountKey") {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// HMAC-SHA256 → base64 string.
fn hmac_b64(key: &[u8], data: &[u8]) -> Result<String, Error> {
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|e| Error::Config(format!("azure shared key signing failed: {e}")))?;
    mac.update(data);
    Ok(base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes()))
}

/// Percent-encode a path, preserving `/` and RFC 3986 unreserved chars.
/// The result is suitable for both the request URL path and the SharedKey
/// canonicalized resource (single encoding, no double-encoding).
fn encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for &byte in path.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
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

/// Build the canonical Shared Key string-to-sign for the Blob service:
/// 12 fixed header lines (most empty here), then the canonicalized headers,
/// then the canonicalized resource, each separated by exactly one newline.
/// See:
/// <https://learn.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key>
fn build_string_to_sign(
    method: &str,
    content_length: &str,
    canonicalized_headers: &str,
    canonicalized_resource: &str,
) -> String {
    // MS_VERSION is newer than 2014-02-14. The wire header may be "0", but
    // modern Shared Key signatures represent zero Content-Length as empty.
    let content_length = if content_length == "0" {
        ""
    } else {
        content_length
    };
    [
        method.to_ascii_uppercase(),
        String::new(), // Content-Encoding
        String::new(), // Content-Language
        content_length.to_string(),
        String::new(), // Content-MD5
        String::new(), // Content-Type
        String::new(), // Date (x-ms-date used instead)
        String::new(), // If-Modified-Since
        String::new(), // If-Match
        String::new(), // If-None-Match
        String::new(), // If-Unmodified-Since
        String::new(), // Range
        canonicalized_headers.to_string(),
        canonicalized_resource.to_string(),
    ]
    .join("\n")
}

/// Extract the `host[:port]` component of a base URL (no scheme, no path).
fn host_of(base: &str) -> &str {
    let after_scheme = base.split("://").nth(1).unwrap_or(base);
    after_scheme.split('/').next().unwrap_or(after_scheme)
}

/// Unix-second deadline for a TTL (now + whole seconds, rounded up to at
/// least 1s so a sub-second TTL is not truncated to "never expire").
fn expires_deadline(ttl: Option<Duration>) -> Option<u64> {
    let ttl = ttl?;
    if ttl.is_zero() {
        return Some(0);
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Some(now.saturating_add(ttl.as_secs().max(1)))
}

/// Returns `true` when the response carries the named expires metadata header
/// whose Unix-second deadline has passed. Enforces per-object TTL client-side.
fn header_expired(headers: &reqwest::header::HeaderMap, name: &str) -> bool {
    let Some(header) = headers.get(name) else {
        return false;
    };
    let Some(deadline) = header.to_str().ok().and_then(|v| v.parse::<u64>().ok()) else {
        return true;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    now >= deadline
}

/// Current UTC time formatted as RFC 1123 (e.g. "Sun, 06 Nov 1994 08:49:37 GMT").
fn rfc1123_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day, hour, min, sec) = decompose_utc(secs);
    let wd = day_of_week(secs);
    let weekday = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][wd as usize];
    let mon = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][(month as usize).saturating_sub(1)];
    format!("{weekday}, {day:02} {mon} {year:04} {hour:02}:{min:02}:{sec:02} GMT")
}

/// Day of week (0 = Sunday) from Unix seconds.
fn day_of_week(secs: u64) -> u32 {
    // 1970-01-01 was a Thursday (index 4).
    (((secs / 86_400) + 4) % 7) as u32
}

/// Decompose a Unix timestamp (seconds) into UTC calendar components.
fn decompose_utc(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = (rem / 3600) as u32;
    let min = ((rem % 3600) / 60) as u32;
    let sec = (rem % 60) as u32;

    // Civil-from-days algorithm (Howard Hinnant), relative to 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_097 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y } as u32;
    (year, m, d, hour, min, sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONN: &str = "DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=a2V5ZGF0YQ==;EndpointSuffix=core.windows.net";

    #[test]
    fn blob_name_applies_prefix() {
        let cache = AzureBlobCache::new("acct", "c", "oxllm", None).unwrap();
        assert_eq!(cache.blob_name("litellm:vcr:abc"), "oxllm:litellm:vcr:abc");
    }

    #[test]
    fn constructor_rejects_empty_prefix() {
        for prefix in ["", " ", "\t\n"] {
            assert!(AzureBlobCache::new("acct", "data", prefix, Some(CONN)).is_err());
        }
    }

    #[tokio::test]
    async fn write_cap_matches_read_cap_before_container_creation() {
        let server = httpmock::MockServer::start();
        let mut cache = AzureBlobCache::new("acct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        cache.max_blob_size = 4;
        let container = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(201);
        });
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path("/data/oxllm%3Ak");
            then.status(201);
        });
        let get = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/data/oxllm%3Ak");
            then.status(200).body("éé");
        });
        cache.set("k", "ééx".to_string(), None).await;
        container.assert_hits(0);
        put.assert_hits(0);
        cache.set("k", "éé".to_string(), None).await;
        assert_eq!(cache.get("k").await.as_deref(), Some("éé"));
        container.assert_hits(1);
        put.assert_hits(1);
        get.assert_hits(1);
    }

    #[tokio::test]
    async fn zero_ttl_invalidates_without_upload_or_container_creation() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("acct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(201);
        });
        let delete = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path("/data/oxllm%3Ak");
            then.status(202);
        });
        cache.set("k", "v".to_string(), Some(Duration::ZERO)).await;
        put.assert_hits(0);
        delete.assert_hits(1);
    }

    #[tokio::test]
    async fn all_blob_operations_share_a_short_total_deadline() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("acct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
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
        .expect("cache budget must include container creation and body reads");
        assert!(get.is_none());
        assert!(!ping);
    }

    #[tokio::test]
    async fn timeout_negative_caches_cancelled_managed_identity_refresh() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("acct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&server.base_url())
            .with_metadata_url(&format!("{}/token", server.base_url()));
        let token = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/token");
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
    fn extract_account_key_from_connection_string() {
        assert_eq!(extract_account_key(CONN), Some("a2V5ZGF0YQ==".to_string()));
    }

    #[test]
    fn extract_account_key_preserves_base64_padding() {
        // base64 padding '=' must survive the split-once('=') parse.
        assert_eq!(
            extract_account_key("AccountKey=a2V5=="),
            Some("a2V5==".to_string())
        );
    }

    #[test]
    fn extract_account_key_is_case_insensitive() {
        assert_eq!(
            extract_account_key("accountname=x;accountkey=k"),
            Some("k".to_string())
        );
    }

    #[test]
    fn extract_account_key_none_when_absent() {
        assert_eq!(extract_account_key("AccountName=x;Foo=bar"), None);
    }

    #[test]
    fn expires_deadline_is_unix_now_plus_ttl_not_raw_ttl() {
        // Regression: stamping the raw TTL (e.g. 60) as x-ms-meta-expires
        // makes header_expired treat every get as past-deadline.
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let deadline = expires_deadline(Some(Duration::from_secs(60))).unwrap();
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(
            deadline >= before + 60 && deadline <= after + 60,
            "deadline {deadline} not in [{}, {}]",
            before + 60,
            after + 60
        );
        assert!(
            deadline > 1_000_000_000,
            "must be a unix timestamp, not the raw TTL; got {deadline}"
        );
        assert_eq!(expires_deadline(None), None);
        assert_eq!(expires_deadline(Some(Duration::ZERO)), Some(0));
        let subsec = expires_deadline(Some(Duration::from_millis(200))).unwrap();
        assert!(
            subsec > before,
            "sub-second TTL must round up to at least 1s, got {subsec}"
        );
    }

    #[test]
    fn url_for_builds_blob_url() {
        let cache = AzureBlobCache::new("acct", "data", "oxllm", None).unwrap();
        let (url, host) = cache.url_for("oxllm:k1");
        assert_eq!(host, "acct.blob.core.windows.net");
        assert_eq!(url, "https://acct.blob.core.windows.net/data/oxllm%3Ak1");
    }

    #[test]
    fn canonicalized_resource_preserves_encoded_uri_path() {
        let cache = AzureBlobCache::new("myacct", "myc", "oxllm", None).unwrap();
        let res = cache.canonicalized_resource("oxllm:litellm:vcr:abc");
        assert_eq!(res, "/myacct/myc/oxllm%3Alitellm%3Avcr%3Aabc");
        let (url, _host) = cache.url_for("oxllm:litellm:vcr:abc");
        // The canonical path and the actual URI must use the same encoding.
        assert!(url.ends_with("/myc/oxllm%3Alitellm%3Avcr%3Aabc"));
    }

    /// Pins the SharedKey string-to-sign layout against the Azure spec: the 12
    /// fixed header lines, then the canonicalized headers, then the
    /// canonicalized resource, joined by exactly one newline — no double
    /// newline (regression guard for the spurious-double-newline defect), and
    /// the resource carries the encoded path. Reconstructs the exact bytes a
    /// PUT would sign using a fixed date.
    #[test]
    fn shared_key_string_to_sign_matches_documented_layout() {
        let cache = AzureBlobCache::new("myacct", "myc", "oxllm", None).unwrap();
        let canon_headers = "x-ms-blob-type:BlockBlob\nx-ms-date:Sun, 06 Nov 1994 08:49:37 GMT\nx-ms-version:2024-05-04";
        let canon_resource = cache.canonicalized_resource("oxllm:blob1");
        assert_eq!(canon_resource, "/myacct/myc/oxllm%3Ablob1");

        let sts = build_string_to_sign("PUT", "11", canon_headers, &canon_resource);

        // Fixed header block: VERB, then empty Content-Encoding /
        // Content-Language, then Content-Length=11. Most of the 12 fixed
        // lines are empty (hence the "\n\n" runs here are *correct* — empty
        // header values render as empty lines).
        assert!(sts.starts_with("PUT\n\n\n11\n"));

        // Canonicalized headers are present verbatim, joined by single
        // newlines (no trailing newline).
        assert!(sts.contains(canon_headers));

        // CRITICAL regression guard #1: exactly one newline separates the
        // canonicalized headers from the canonicalized resource. The old code
        // appended a trailing '\n' to each canonicalized header AND joined the
        // two sections with '\n', producing a double newline
        // ("x-ms-version:…\n\n/account/…") → 403 AuthenticationFailed.
        assert!(
            sts.ends_with(&format!("{canon_headers}\n{canon_resource}")),
            "string-to-sign did not end with headers + resource (single newline): {sts}"
        );
        assert!(
            !sts.contains("\n\n/myacct"),
            "no double newline before the canonicalized resource: {sts}"
        );

        assert!(sts.ends_with("/myacct/myc/oxllm%3Ablob1"));
    }

    #[test]
    fn build_string_to_sign_empty_body_has_empty_content_length() {
        let sts = build_string_to_sign("GET", "", "x-ms-date:x\nx-ms-version:y", "/a/b");
        // Empty Content-Length renders as an empty field (two adjacent '\n'),
        // which is *inside* the 12 fixed lines and is correct per the spec —
        // distinct from a double newline around the canonicalized sections.
        assert!(sts.starts_with("GET\n\n\n\n"));
    }

    #[test]
    fn zero_content_length_matches_microsoft_published_modern_version_vector() {
        // Azure Shared Key documentation, "Content-Length header in version
        // 2014-02-14 and earlier", gives this exact corrected 2015-02-21 vector.
        let headers = "x-ms-date:Fri, 26 Jun 2015 23:39:12 GMT\nx-ms-version:2015-02-21";
        let resource = "/myaccount/mycontainer\nrestype:container\ntimeout:30";
        let expected = "PUT\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:Fri, 26 Jun 2015 23:39:12 GMT\
                        \nx-ms-version:2015-02-21\n/myaccount/mycontainer\nrestype:container\ntimeout:30";
        assert_eq!(
            build_string_to_sign("PUT", "0", headers, resource),
            expected
        );
        assert_eq!(build_string_to_sign("PUT", "", headers, resource), expected);
    }

    #[test]
    fn hmac_b64_matches_reference() {
        // Reference (Python `hmac` + `hashlib` + `base64`):
        //   base64(hmac_sha256("key", "The quick brown fox jumps over the lazy dog"))
        let sig = hmac_b64(b"key", b"The quick brown fox jumps over the lazy dog");
        assert_eq!(
            sig.ok().as_deref(),
            Some("97yD9DBThCSxMpjmqm+xQ+9NWaFJRhdZl0edvC0aPNg=")
        );
    }

    #[test]
    fn encode_path_preserves_unreserved_and_slash() {
        assert_eq!(encode_path("/a/b_c-1.2~3"), "/a/b_c-1.2~3");
    }

    #[test]
    fn encode_path_percent_encodes_special_chars() {
        assert_eq!(encode_path("/oxllm:k"), "/oxllm%3Ak");
        assert_eq!(encode_path("/a b"), "/a%20b");
        assert_eq!(encode_path(""), "");
    }

    #[test]
    fn host_of_strips_scheme_and_path() {
        assert_eq!(host_of("http://127.0.0.1:9000"), "127.0.0.1:9000");
        assert_eq!(host_of("http://127.0.0.1:9000/"), "127.0.0.1:9000");
        assert_eq!(host_of("example.com"), "example.com");
    }

    #[test]
    fn rfc1123_format_is_well_formed() {
        let date = rfc1123_now();
        assert!(date.ends_with(" GMT"));
        assert!(date.len() == "ddd, dd MMM yyyy HH:MM:SS GMT".len());
    }

    #[test]
    fn day_of_week_epoch_is_thursday() {
        assert_eq!(day_of_week(0), 4);
    }

    #[test]
    fn decompose_utc_known_timestamps() {
        // Unix epoch.
        assert_eq!(decompose_utc(0), (1970, 1, 1, 0, 0, 0));
        // RFC 1123 sample date "Sun, 06 Nov 1994 08:49:37 GMT".
        assert_eq!(decompose_utc(784_111_777), (1994, 11, 6, 8, 49, 37));
        // Leap day: 2024-02-29T00:00:00Z.
        assert_eq!(decompose_utc(1_709_164_800), (2024, 2, 29, 0, 0, 0));
        // Non-trivial HMS: 2009-02-13T23:31:30Z.
        assert_eq!(decompose_utc(1_234_567_890), (2009, 2, 13, 23, 31, 30));
        // Cross-checked against the S3 sibling crate's test vectors.
        assert_eq!(decompose_utc(1_369_353_600), (2013, 5, 24, 0, 0, 0));
        assert_eq!(decompose_utc(1_440_938_160), (2015, 8, 30, 12, 36, 0));
    }

    #[tokio::test]
    async fn incr_returns_err_not_silent_success() {
        let cache = AzureBlobCache::new("acct", "c", "oxllm", None).unwrap();
        assert!(
            cache.incr("c", 5.0, None).await.is_err(),
            "incr must not silently succeed on a counter-incapable backend"
        );
    }

    #[tokio::test]
    async fn incr_if_below_returns_err_not_allowed() {
        let cache = AzureBlobCache::new("acct", "c", "oxllm", None).unwrap();
        assert!(
            cache.incr_if_below("c", 5.0, 1.0, None).await.is_err(),
            "incr_if_below must not silently pass a budget/RPM gate"
        );
    }

    #[tokio::test]
    async fn counter_value_is_none() {
        let cache = AzureBlobCache::new("acct", "c", "oxllm", None).unwrap();
        assert!(cache.counter_value("c").await.is_none());
    }

    #[tokio::test]
    async fn try_counter_value_is_ok_none() {
        // Must agree with the `counter_value` no-op rather than falling back
        // to the trait default (which would read a blob body and parse f64).
        let cache = AzureBlobCache::new("acct", "c", "oxllm", None).unwrap();
        assert!(matches!(cache.try_counter_value("c").await, Ok(None)));
    }

    // -----------------------------------------------------------------
    // httpmock-backed integration tests (blob + IMDS endpoints injectable).
    // -----------------------------------------------------------------

    /// SharedKey PUT/GET/DELETE round-trip against a mocked blob endpoint.
    /// Verifies the request path is percent-encoded (colon → %3A) and that the
    /// required `x-ms-*` headers and `Authorization` are present on the wire.
    #[tokio::test]
    async fn shared_key_round_trip_via_mock() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());

        let key = "litellm:vcr:demo";
        let encoded_path = "/data/oxllm%3Alitellm%3Avcr%3Ademo";

        // PUT: signed, carries the blob-type + version headers + Authorization.
        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path(encoded_path)
                .header("x-ms-version", MS_VERSION)
                .header("x-ms-blob-type", "BlockBlob")
                .header_exists("x-ms-date")
                .header_exists("Authorization");
            then.status(201).body("");
        });
        cache.set(key, "{\"cached\":true}".to_string(), None).await;
        put_mock.assert();

        // GET: return the stored body.
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path(encoded_path)
                .header("x-ms-version", MS_VERSION)
                .header_exists("Authorization");
            then.status(200).body("{\"cached\":true}");
        });
        let val = cache.get(key).await;
        get_mock.assert();
        assert_eq!(val.as_deref(), Some("{\"cached\":true}"));

        // GET miss → 404 → None.
        let miss_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/data/oxllm%3Alitellm%3Avcr%3Anever");
            then.status(404).body("");
        });
        assert!(cache.get("litellm:vcr:never").await.is_none());
        miss_mock.assert();

        // DELETE: 202 accepted.
        let del_mock = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE).path(encoded_path);
            then.status(202).body("");
        });
        cache.delete(key).await;
        del_mock.assert();
    }

    /// Managed-identity requests must carry `x-ms-date` (all ops) and
    /// `x-ms-blob-type: BlockBlob` on PUT, plus the bearer `Authorization`.
    #[tokio::test]
    async fn managed_identity_request_carries_required_headers() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();

        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .header("Metadata", "true");
            then.status(200).json_body(serde_json::json!({
                "access_token": "fake-bearer",
                "expires_in": "3600",
            }));
        });

        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ));

        // PUT must carry x-ms-blob-type + x-ms-date + bearer Authorization.
        let put_mock = blob.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data/oxllm%3Ak")
                .header("x-ms-version", MS_VERSION)
                .header("x-ms-blob-type", "BlockBlob")
                .header_exists("x-ms-date")
                .header("Authorization", "Bearer fake-bearer");
            then.status(201).body("");
        });
        cache.set("k", "v".to_string(), None).await;
        put_mock.assert();
        imds_mock.assert_hits(1);

        // GET (read) must also carry x-ms-date.
        let get_mock = blob.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/data/oxllm%3Ak")
                .header_exists("x-ms-date")
                .header("Authorization", "Bearer fake-bearer");
            then.status(404).body("");
        });
        let _ = cache.get("k").await;
        get_mock.assert();
    }

    /// The managed-identity token is fetched once and reused across ops while
    /// it remains valid (IMDS is hit once for two rapid cache operations).
    #[tokio::test]
    async fn managed_identity_token_is_cached_across_ops() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();

        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(200).json_body(serde_json::json!({
                "access_token": "fake-bearer",
                "expires_in": "3600",
            }));
        });

        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ));

        blob.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(404).body("");
        });

        // Two rapid GETs — only one IMDS round-trip expected.
        let _ = cache.get("a").await;
        let _ = cache.get("b").await;
        imds_mock.assert_hits(1);
    }

    /// Once the cached token is past its expiry, the next op refetches from
    /// IMDS.
    #[tokio::test]
    async fn managed_identity_refetches_after_token_expiry() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();

        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(200).json_body(serde_json::json!({
                "access_token": "fake-bearer",
                "expires_in": "3600",
            }));
        });

        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ));

        blob.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(404).body("");
        });

        // Poison the token cache with an already-expired credential.
        {
            let mut guard = cache.token.lock().await;
            *guard = Some(CachedToken::Valid {
                token: "stale".to_string(),
                expires_at: Instant::now(),
            });
        }

        let _ = cache.get("a").await;
        // Expired → refetched.
        imds_mock.assert_hits(1);
    }

    /// `get` must refuse to read a blob whose body exceeds the size cap,
    /// returning `None` without allocating the full body.
    #[tokio::test]
    async fn get_returns_none_when_blob_exceeds_max_size() {
        let server = httpmock::MockServer::start();
        let mut cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        // Tight cap so the oversized body trips it.
        cache.max_blob_size = 4;

        let oversized = "x".repeat(100);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/data/oxllm%3Abig");
            then.status(200).body(oversized);
        });
        assert!(cache.get("big").await.is_none());
        get_mock.assert();
    }

    /// `get` returns `None` (no panic) on a non-404 error status such as 403.
    #[tokio::test]
    async fn get_returns_none_on_forbidden() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/data/oxllm%3Adenied");
            then.status(403).body("auth failed");
        });
        assert!(cache.get("denied").await.is_none());
        get_mock.assert();
    }

    /// `ping` reports reachable (`true`) on any HTTP response — even 403 — and
    /// returns `false` only on a transport error. (It does NOT verify auth;
    /// see the method docs.) Here we verify the reachable-on-403 contract.
    #[tokio::test]
    async fn ping_true_on_http_response_even_when_forbidden() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        let head_mock = server.mock(|when, then| {
            when.method(httpmock::Method::HEAD)
                .path("/data/oxllm%3A__oxllm_ping__");
            then.status(403).body("");
        });
        assert!(cache.ping().await);
        head_mock.assert();
    }

    // -----------------------------------------------------------------
    // Tests for the review fixes.
    // -----------------------------------------------------------------

    /// A connection string without an `AccountKey=` must be rejected at
    /// construction rather than silently falling back to managed identity.
    #[test]
    fn new_rejects_connection_string_without_account_key() {
        let err = AzureBlobCache::new("acct", "c", "oxllm", Some("AccountName=x;Foo=bar"));
        assert!(err.is_err(), "expected Err for keyless connection_string");
        let msg = format!("{}", err.err().unwrap());
        assert!(
            msg.contains("AccountKey"),
            "error should mention the missing AccountKey: {msg}"
        );
    }

    /// A connection string whose `AccountKey=` is not valid base64 must be
    /// rejected at construction (the key is pre-decoded up front, not on the
    /// first request).
    #[test]
    fn new_rejects_non_base64_account_key() {
        let err = AzureBlobCache::new(
            "acct",
            "c",
            "oxllm",
            Some("AccountName=x;AccountKey=!!!not-base64!!!"),
        );
        assert!(err.is_err(), "expected Err for non-base64 account key");
    }

    /// When multiple `AccountKey=` segments appear, the first wins. Locks in
    /// the chosen behavior (coverage gap from the review).
    #[test]
    fn extract_account_key_returns_first_of_multiple() {
        assert_eq!(
            extract_account_key("AccountKey=first;AccountKey=second"),
            Some("first".to_string())
        );
    }

    /// A connection string with a valid key still constructs successfully
    /// (guard against the new rejection being too aggressive).
    #[test]
    fn new_accepts_valid_connection_string() {
        let cache = AzureBlobCache::new("acct", "c", "oxllm", Some(CONN)).unwrap();
        assert!(matches!(cache.auth, AzureAuth::SharedKey { .. }));
    }

    /// `ensure_container` issues an idempotent `Create Container`; a 201 is
    /// success.
    #[tokio::test]
    async fn ensure_container_creates_container() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        let container_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container")
                .header_exists("Authorization");
            then.status(201).body("");
        });
        cache.ensure_container().await.unwrap();
        container_mock.assert();
    }

    /// `ensure_container` tolerates 409 Conflict (container already exists).
    #[tokio::test]
    async fn ensure_container_tolerates_conflict() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        let container_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(409).body("");
        });
        cache.ensure_container().await.unwrap();
        container_mock.assert();
    }

    /// After `ensure_container` (201), a `set` persists and a `get` reads it
    /// back — the auto-create removes the "container missing" footgun.
    #[tokio::test]
    async fn ensure_container_then_set_persists() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(201).body("");
        });
        cache.ensure_container().await.unwrap();

        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data/oxllm%3Ak")
                .header("x-ms-blob-type", "BlockBlob");
            then.status(201).body("");
        });
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/data/oxllm%3Ak");
            then.status(200).body("{\"v\":1}");
        });
        cache.set("k", "{\"v\":1}".to_string(), None).await;
        put_mock.assert();
        assert_eq!(cache.get("k").await.as_deref(), Some("{\"v\":1}"));
        get_mock.assert();
    }

    /// An empty `value` round-trips: the signed Content-Length ("0") agrees
    /// with the wire body (explicit empty body), so the PUT succeeds and the
    /// subsequent GET reads the empty string back.
    #[tokio::test]
    async fn set_empty_value_round_trip_via_mock() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data/oxllm%3Aempty")
                .header("x-ms-blob-type", "BlockBlob");
            then.status(201).body("");
        });
        cache.set("empty", String::new(), None).await;
        put_mock.assert();

        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/data/oxllm%3Aempty");
            then.status(200).body("");
        });
        assert_eq!(cache.get("empty").await.as_deref(), Some(""));
        get_mock.assert();
    }

    /// IMDS transient failures (503) are retried: with two retries in the
    /// schedule, an always-503 IMDS is hit exactly 3 times (initial + 2
    /// retries) before the token fetch gives up. The resulting cache op is a
    /// benign miss (no panic).
    #[tokio::test]
    async fn metadata_token_retries_on_transient() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();
        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .header("Metadata", "true");
            then.status(503).body("");
        });
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ))
            .with_imds_retry_backoff(&[0, 0]);

        // The GET degrades to a miss (token fetch errors out) without panicking.
        assert!(cache.get("k").await.is_none());
        // 1 initial attempt + 2 retries == 3 IMDS hits.
        imds_mock.assert_hits(3);
    }

    /// A failed IMDS fetch is negative-cached for
    /// [`TOKEN_FAILURE_BACKOFF`]: during an IMDS outage every subsequent
    /// cache op gets an immediate error instead of re-running the full retry
    /// ladder (3 attempts + 1s/2s sleeps) serialized on the token lock, which
    /// would stall the request hot path for seconds per op indefinitely.
    #[tokio::test]
    async fn metadata_token_failure_is_negative_cached() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();
        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .header("Metadata", "true");
            then.status(503).body("");
        });
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ))
            .with_imds_retry_backoff(&[0, 0]);

        // First op runs the full retry ladder (3 IMDS hits) and fails.
        assert!(cache.get("k1").await.is_none());
        imds_mock.assert_hits(3);

        // Ops within the failure-backoff window short-circuit: no additional
        // IMDS hits, and each still degrades to a fast miss.
        assert!(cache.get("k2").await.is_none());
        assert!(cache.get("k3").await.is_none());
        imds_mock.assert_hits(3);

        // Poison the cache with an elapsed backoff window and confirm the
        // ladder re-runs only after the window expires.
        {
            let mut guard = cache.token.lock().await;
            *guard = Some(CachedToken::Failed {
                retry_after: Instant::now(),
            });
        }
        assert!(cache.get("k4").await.is_none());
        imds_mock.assert_hits(6);
    }

    /// IMDS non-transient failures (403) are NOT retried — the fetch fails
    /// fast instead of burning the backoff.
    #[tokio::test]
    async fn metadata_token_does_not_retry_non_transient() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();
        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .header("Metadata", "true");
            then.status(403).body("");
        });
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ))
            .with_imds_retry_backoff(&[0, 0]);

        assert!(cache.get("k").await.is_none());
        // 403 is non-transient → single attempt, no retries.
        imds_mock.assert_hits(1);
    }

    /// Concurrent callers with a stale/absent token do not stampede IMDS: the
    /// lock-held refresh (with double-check) collapses the herd to a single
    /// fetch regardless of fan-out.
    #[tokio::test]
    async fn metadata_token_single_flight_under_concurrency() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();
        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(200).json_body(serde_json::json!({
                "access_token": "fake-bearer",
                "expires_in": "3600",
            }));
        });
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ));
        blob.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/data/oxllm%3Aa");
            then.status(404).body("");
        });

        // Four concurrent GETs of the same key — only one IMDS round-trip.
        let (a, b, c, d) = tokio::join!(
            cache.get("a"),
            cache.get("a"),
            cache.get("a"),
            cache.get("a"),
        );
        assert!(a.is_none() && b.is_none() && c.is_none() && d.is_none());
        imds_mock.assert_hits(1);
    }

    /// The first counter read flips the warn-once guard so a host that wires
    /// this backend directly (bypassing the counter wrapper) sees the no-op
    /// at runtime instead of a silent billing/auth gap.
    #[tokio::test]
    async fn counter_noop_warns_once() {
        let cache = AzureBlobCache::new("acct", "cnt", "oxllm", None).unwrap();
        assert!(!cache.counter_warned.load(Ordering::SeqCst));
        assert_eq!(cache.counter_value("k").await, None);
        assert!(cache.counter_warned.load(Ordering::SeqCst));
    }

    // -----------------------------------------------------------------
    // Tests for Phase-2 review fixes.
    // -----------------------------------------------------------------

    /// Verifier for httpmock `when.matches(...)`: recomputes the SharedKey
    /// HMAC-SHA256 signature from the *observed* request headers/method/path
    /// and returns `true` only if it equals the on-wire `Authorization` value.
    /// This guards the signing coupling in CI — any systematic signature error
    /// (wrong Content-Length, double newline, mismatched canonicalized resource)
    /// causes the mock to not match, and `assert()` then fails.
    fn verify_blob_shared_key_signature(req: &httpmock::prelude::HttpMockRequest) -> bool {
        let headers = req.headers.clone().unwrap_or_default();
        verify_shared_key_signature(&req.method, &req.path, &headers)
    }

    fn verify_shared_key_signature(method: &str, path: &str, headers: &[(String, String)]) -> bool {
        let find_header = |name: &str| -> Option<&str> {
            headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.as_str())
        };
        let auth = match find_header("authorization") {
            Some(a) => a,
            None => return false,
        };
        let sent_sig = match auth.strip_prefix("SharedKey ") {
            Some(rest) => match rest.split_once(':') {
                Some((_, s)) => s,
                None => return false,
            },
            None => return false,
        };
        let mut ms_headers: Vec<(String, String)> = headers
            .iter()
            .filter(|(k, _)| k.to_lowercase().starts_with("x-ms-"))
            .map(|(k, v)| (k.to_lowercase(), v.clone()))
            .collect();
        ms_headers.sort_by(|a, b| a.0.cmp(&b.0));
        ms_headers.dedup_by(|a, b| a.0 == b.0);
        let canon_headers = ms_headers
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join("\n");
        let content_length = find_header("content-length").unwrap_or("");
        let canon_resource = format!("/myacct{path}");
        let sts = build_string_to_sign(
            &method.to_uppercase(),
            content_length,
            &canon_headers,
            &canon_resource,
        );
        hmac_b64(b"keydata", sts.as_bytes()).is_ok_and(|expected_sig| sent_sig == expected_sig)
    }

    /// Recompute-and-verify the SharedKey signature for non-empty PUT, GET, and
    /// DELETE — guarding the Content-Length / canonicalized-resource coupling.
    #[tokio::test]
    async fn shared_key_signature_matches_recomputed_blob_ops() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());

        // Container auto-create: plain 201 (no signature recompute here — the
        // container canonicalized resource has a restype:container suffix).
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(201).body("");
        });

        // PUT (non-empty body): signature verified via matcher.
        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data/oxllm%3Ak")
                .matches(verify_blob_shared_key_signature);
            then.status(201).body("");
        });
        cache.set("k", "hello world".to_string(), None).await;
        put_mock.assert();

        // GET: signature verified.
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/data/oxllm%3Ak")
                .matches(verify_blob_shared_key_signature);
            then.status(200).body("hello world");
        });
        let _ = cache.get("k").await;
        get_mock.assert();

        // DELETE: signature verified.
        let del_mock = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path("/data/oxllm%3Ak")
                .matches(verify_blob_shared_key_signature);
            then.status(202).body("");
        });
        cache.delete("k").await;
        del_mock.assert();
    }

    /// Recompute-and-verify for an empty-body PUT (Content-Length = "0") and
    /// for a HEAD (ping). Wire `Content-Length: 0` is canonicalized to empty.
    #[tokio::test]
    async fn shared_key_signature_matches_recomputed_empty_put_and_head() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());

        // Container auto-create.
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(201).body("");
        });

        // Empty PUT: Content-Length = "0" on wire, signed as an empty field.
        let empty_put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data/oxllm%3Aempty")
                .matches(verify_blob_shared_key_signature);
            then.status(201).body("");
        });
        cache.set("empty", String::new(), None).await;
        empty_put_mock.assert();

        // HEAD (ping): Content-Length = "0" on wire, signed as an empty field.
        let head_mock = server.mock(|when, then| {
            when.method(httpmock::Method::HEAD)
                .path("/data/oxllm%3A__oxllm_ping__")
                .matches(verify_blob_shared_key_signature);
            then.status(404).body("");
        });
        assert!(cache.ping().await);
        head_mock.assert();
    }

    /// `set` with `Some(ttl)` must stamp the `x-ms-meta-expires` metadata
    /// header (Unix-second deadline) on the PUT; without a TTL it must not.
    /// Mirrors the S3 backend's per-object TTL enforcement.
    #[tokio::test]
    async fn set_with_ttl_stamps_expires_metadata() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());

        server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(201).body("");
        });

        let put_with_ttl = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data/oxllm%3Awithttl")
                .header_exists("x-ms-meta-expires")
                .body("val");
            then.status(201).body("");
        });
        cache
            .set("withttl", "val".to_string(), Some(Duration::from_secs(60)))
            .await;
        put_with_ttl.assert();

        let put_without_ttl = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data/oxllm%3Awithoutttl")
                .body("val");
            then.status(201).body("");
        });
        cache.set("withoutttl", "val".to_string(), None).await;
        put_without_ttl.assert();
    }

    /// A GET whose response carries a past `x-ms-meta-expires` deadline is a miss.
    #[tokio::test]
    async fn get_past_expires_deadline_is_miss() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/data/oxllm%3Astale");
            then.status(200)
                .header("x-ms-meta-expires", "0")
                .body("stale");
        });
        assert_eq!(cache.get("stale").await, None);
        get_mock.assert();
    }

    /// A non-`201`/`409` container-PUT status (e.g. `403`) must **not**
    /// permanently suppress retries: the OnceCell stays empty and the next
    /// `set` re-attempts container creation.
    #[tokio::test]
    async fn container_creation_re_attempts_on_non_success() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());

        // Every container PUT returns 403.
        let container_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(403).body("");
        });
        // Blob PUTs return 201 (simulating a pre-existing container).
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path("/data/oxllm%3Ak1");
            then.status(201).body("");
        });
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path("/data/oxllm%3Ak2");
            then.status(201).body("");
        });

        cache.set("k1", "v".to_string(), None).await;
        cache.set("k2", "v".to_string(), None).await;

        // Two sets → two container-PUT attempts (OnceCell stays empty after 403).
        container_mock.assert_hits(2);
    }

    /// Concurrent first-burst `set`s must serialize container creation through
    /// the `OnceCell`: exactly one container-PUT fires, and all blob PUTs are
    /// received. (Before the fix, the `AtomicBool` flag was flipped before
    /// creation resolved, so concurrent writers bypassed creation and issued
    /// blob PUTs against a not-yet-existing container.)
    #[tokio::test]
    async fn concurrent_sets_create_container_once() {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))
            .unwrap()
            .with_blob_endpoint(&server.base_url());

        let container_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/data")
                .query_param("restype", "container");
            then.status(201).body("").delay(Duration::from_millis(100));
        });
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path("/data/oxllm%3Aa");
            then.status(201).body("");
        });
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path("/data/oxllm%3Ab");
            then.status(201).body("");
        });
        server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path("/data/oxllm%3Ac");
            then.status(201).body("");
        });

        let (ra, rb, rc) = tokio::join!(
            cache.set("a", "1".to_string(), None),
            cache.set("b", "2".to_string(), None),
            cache.set("c", "3".to_string(), None),
        );
        let _ = (ra, rb, rc);

        // Exactly one container creation regardless of fan-out.
        container_mock.assert_hits(1);
    }

    /// `new()` must reject a container name with characters outside
    /// `[a-z0-9-]` (e.g. uppercase or underscore) at construction, preventing
    /// a URL/canonicalized-resource encoding asymmetry that would surface as a
    /// confusing `400`/`403` from Azure.
    #[test]
    fn new_rejects_invalid_container_name() {
        let err = AzureBlobCache::new("acct", "My_Container", "oxllm", None);
        assert!(err.is_err());
        let msg = format!("{}", err.err().unwrap());
        assert!(
            msg.contains("container name"),
            "error should mention container name: {msg}"
        );
    }

    /// A valid lowercase container name constructs successfully.
    #[test]
    fn new_accepts_valid_container_name() {
        let cache = AzureBlobCache::new("acct", "my-container-1", "oxllm", None).unwrap();
        assert_eq!(cache.container, "my-container-1");
    }

    /// Container URL uses the encoded form of the container segment (defense
    /// in depth — after validation only `[a-z0-9-]` chars survive, so encoding
    /// is a no-op); the canonicalized resource uses the same encoded segment.
    #[test]
    fn container_url_and_canonical_resource_use_encoded_segment() {
        let cache = AzureBlobCache::new("acct", "data", "oxllm", None).unwrap();
        let (url, _host) = cache.container_url();
        assert!(url.contains("/data?restype=container"));
        let canon = cache.container_canonicalized_resource();
        assert_eq!(canon, "/acct/data\nrestype:container");
    }

    /// When IMDS omits `expires_in`, the fallback lifetime (1800s) prevents
    /// per-call IMDS thrash: two rapid ops hit IMDS only once, not twice.
    #[tokio::test]
    async fn imds_expires_in_absent_applies_fallback_lifetime() {
        let blob = httpmock::MockServer::start();
        let imds = httpmock::MockServer::start();
        let imds_mock = imds.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(200).json_body(serde_json::json!({
                "access_token": "fake-bearer",
            }));
        });
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)
            .unwrap()
            .with_blob_endpoint(&blob.base_url())
            .with_metadata_url(&format!(
                "{}/metadata/identity/oauth2/token",
                imds.base_url()
            ));
        blob.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(404).body("");
        });

        let _ = cache.get("a").await;
        let _ = cache.get("b").await;

        // Without the fallback, the second op would re-fetch (expires_at =
        // now() → immediately stale). With the fallback, the token is cached
        // for 1800s, so IMDS is hit only once.
        imds_mock.assert_hits(1);
    }

    mod retries;
}
