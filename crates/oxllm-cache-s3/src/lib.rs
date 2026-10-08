//! S3-compatible object-storage cache backend for oxllm.
//!
//! Implements [`oxllm_core::Cache`] over any S3-compatible blob store (AWS S3,
//! MinIO, R2, etc.) using PUT/GET/DELETE on objects whose key is
//! `{prefix}:{original_key}`. Values are stored verbatim as object bodies (the
//! caller-supplied JSON strings).
//!
//! Auth is [AWS Signature Version 4](https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-authenticating-requests.html),
//! implemented from scratch with `sha2` + `hmac` so the crate stays
//! `#![forbid(unsafe_code)]` and dependency-light (no `aws-sdk-*`).
//!
//! ## Response-cache-only (no atomic counters)
//!
//! Blob stores have no atomic increment primitive. This backend models only
//! the response-blob operations (`get`/`set`/`delete`/`ping`); the counter
//! write operations surface an explicit error so a misconfigured direct use
//! fails loudly instead of silently disabling budget / rate-limit / JWT-replay
//! enforcement:
//!
//! - `Cache::incr` → `Err` (counters unsupported)
//! - `Cache::incr_if_below` → `Err` (counters unsupported)
//! - `Cache::counter_value` / `Cache::try_counter_value` → `None` /
//!   `Ok(None)` (a miss is the correct fail-open for an unsupported read)
//!
//! Because silently returning zero would disable budget / rate-limit / JWT
//! replay enforcement, the server wiring (`build_s3_cache`) does **not** expose
//! this raw backend as the application cache: it wraps it with an in-memory
//! counter layer so response blobs flow to S3 while atomic counters stay
//! accurate per replica. Direct users of `S3Cache` must supply their own
//! counter layer if enforcement is required.
//!
//! ## Per-object TTL
//!
//! Per-object TTL is enforced client-side by stamping an
//! `x-amz-meta-expires` header (Unix-seconds deadline) on PUT and checking it
//! on GET, mirroring the reference read-side `Expires` enforcement. S3 does not
//! auto-delete on `Expires`, so a bucket lifecycle rule is still recommended
//! to bound storage growth.
//! `None` means no expiry; `Some(Duration::ZERO)` invalidates a prior value.
//! Each public cache operation has a 500ms total optimization budget, including
//! signing, retries, backoff, and body reads; failure is a miss or skipped write.
//!
//! ## Readiness permissions
//!
//! `ping` uses `HeadObject` on the reserved `{prefix}:__oxllm_ping__` key. If
//! that returns 403/404, it initializes a small persistent marker with
//! `PutObject`, then requires a successful `HeadObject`. An absent bucket cannot
//! accept that write, so its 404 is never treated as a healthy cache miss.
//! This also handles AWS's 403 for a missing object without `s3:ListBucket`.
//! See AWS's [HeadObject permissions and generic errors](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html).
//!
//! Grant `s3:GetObject` on the marker (the permission used by `HeadObject`) and
//! `s3:PutObject` to initialize/recreate it, normally via the same
//! `arn:aws:s3:::<bucket>/<prefix>:*` resource as cache entries. Bucket listing,
//! `HeadBucket`, ACL changes and marker deletion are not required. A read-only
//! role can probe a pre-existing readable marker, but cannot initialize an
//! absent one. Existing bucket-policy/encryption conditions on object writes
//! still apply. Reserve this key for the probe; it has no cache TTL and remains
//! until a bucket lifecycle rule or an operator removes it. Only initialization
//! writes a marker version; a successful probe is a single HEAD. All steps and
//! their bounded retries share the same 500ms operation deadline.

#![forbid(unsafe_code)]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use oxllm_core::Cache as CacheTrait;
use oxllm_core::{Error, IncrResult};

type HmacSha256 = Hmac<Sha256>;

/// The SigV4 algorithm identifier (constant for every signed request).
const ALGORITHM: &str = "AWS4-HMAC-SHA256";
/// The S3 service name used in the SigV4 credential scope.
const SERVICE: &str = "s3";
/// Transport ceiling; the complete cache operation has the tighter budget below.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// Total budget, not a fresh deadline per retry or body chunk.
const CACHE_OPERATION_TIMEOUT: Duration = Duration::from_millis(500);
/// Total attempts per object operation (1 initial + 2 retries).
const MAX_SEND_ATTEMPTS: usize = 3;
/// Base backoff between retries; doubled per attempt (100ms, 200ms).
const RETRY_BASE_DELAY: Duration = Duration::from_millis(100);

/// True for statuses worth retrying: 429 (throttle) and 5xx. Every other
/// status — including 403 (auth, surfaced distinctly by callers) — is
/// terminal for the attempt.
fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    let code = status.as_u16();
    code == 429 || (500..=599).contains(&code)
}
/// Default cap on a single cached object read or written (16 MiB). Bounds
/// memory use against an oversized or hostile object body; S3 always sends a
/// `Content-Length` so the check fires before allocation.
const DEFAULT_MAX_OBJECT_SIZE: usize = 16 * 1024 * 1024;
const PING_KEY: &str = "__oxllm_ping__";
const PING_BODY: &[u8] = b"oxllm-cache-readiness\n";

async fn cache_operation<T>(
    name: &str,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    match tokio::time::timeout(CACHE_OPERATION_TIMEOUT, operation).await {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::warn!(
                operation = name,
                "s3 cache operation exceeded optimization deadline"
            );
            None
        }
    }
}

/// An S3-compatible object-storage cache.
///
/// Construct with [`S3Cache::new`]. The same struct works against AWS S3 and
/// S3-API-compatible stores (MinIO, Cloudflare R2, …) via the `endpoint`
/// override, which switches the URL scheme to path-style.
#[derive(Clone)]
pub struct S3Cache {
    client: reqwest::Client,
    bucket: String,
    region: String,
    prefix: String,
    /// Custom endpoint base (e.g. `http://localhost:9000`, optionally with a
    /// path prefix such as `http://host:9000/gateway`). When set, URLs use
    /// path-style (`{endpoint}/{bucket}/{key}`) suitable for MinIO/R2; any
    /// path prefix is included in both the request URL and the SigV4
    /// canonical URI so the signature covers what is sent on the wire.
    endpoint: Option<String>,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
    /// Upper bound (bytes) on either a `get` or `set` body.
    max_object_size: usize,
}

impl S3Cache {
    /// Create a new S3 cache.
    ///
    /// `endpoint`, when set, is used verbatim as the scheme+host (plus any
    /// path prefix, e.g. a gateway at `http://host:9000/gateway`) with
    /// path-style URLs; otherwise AWS virtual-host style
    /// `https://{bucket}.s3.{region}.amazonaws.com` is used. Credentials may
    /// be empty when the endpoint is unauthenticated
    /// (some MinIO setups); signing then proceeds with blank keys.
    pub fn new(
        bucket: &str,
        region: &str,
        prefix: &str,
        access_key: &str,
        secret_key: &str,
        session_token: Option<String>,
        endpoint: Option<String>,
    ) -> Result<Self, Error> {
        validate_bucket_name(bucket)?;
        if prefix.trim().is_empty() {
            return Err(Error::Config("s3 cache prefix must not be empty".into()));
        }
        if region.trim().is_empty() {
            return Err(Error::Config("s3 region must not be empty".into()));
        }
        if let Some(ep) = endpoint.as_deref() {
            validate_endpoint(ep)?;
        }
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| Error::Config(format!("s3 http client build failed: {e}")))?;
        Ok(Self {
            client,
            bucket: bucket.to_string(),
            region: region.to_string(),
            prefix: prefix.to_string(),
            endpoint,
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            session_token,
            max_object_size: DEFAULT_MAX_OBJECT_SIZE,
        })
    }

    /// Build the prefixed object key for a logical cache key.
    fn object_key(&self, key: &str) -> String {
        format!("{}:{key}", self.prefix)
    }

    /// Resolve the request URL (host + already-encoded path) for an object.
    /// Returns `(full_url, host, encoded_path)`.
    fn url_for(&self, object_key: &str) -> (String, String, String) {
        let encoded_path = encode_path(&format!("/{}", object_key));
        match &self.endpoint {
            Some(base) => {
                let base = base.trim_end_matches('/');
                // SigV4 requires the canonical `host` value to be lowercase,
                // and reqwest normalizes URL hosts to lowercase per RFC 3986 —
                // so the value we sign must match the wire header exactly.
                let host = host_of(base).to_ascii_lowercase();
                // A custom endpoint may carry a path prefix (e.g. a gateway
                // at `http://host:9000/gateway`). SigV4 signs the canonical
                // URI exactly as requested on the wire, so the prefix must
                // appear in both the request URL and the signed path —
                // omitting it from the signature makes every request 403
                // (SignatureDoesNotMatch), silently disabling the cache
                // (`get` returns None, `set` only warns). Encode the prefix
                // once and use it verbatim in both places (single encoding,
                // no double-encoding).
                let base_path = endpoint_path_of(base);
                let encoded_base_path = encode_path(base_path);
                let authority = &base[..base.len() - base_path.len()];
                let canonical = format!(
                    "{encoded_base_path}/{bucket}{encoded_path}",
                    bucket = self.bucket
                );
                let full = format!("{authority}{canonical}");
                (full, host, canonical)
            }
            None => {
                // AWS S3 does not support virtual-host-style HTTPS URLs for
                // dotted bucket names (the wildcard TLS cert does not match),
                // so switch to path-style automatically in that case.
                if self.bucket.contains('.') {
                    let host = format!("s3.{}.amazonaws.com", self.region);
                    let full = format!(
                        "https://{host}/{bucket}{encoded_path}",
                        bucket = self.bucket
                    );
                    let canonical = format!("/{bucket}{encoded_path}", bucket = self.bucket);
                    (full, host, canonical)
                } else {
                    let host = format!("{}.s3.{}.amazonaws.com", self.bucket, self.region);
                    let full = format!("https://{host}{encoded_path}");
                    (full, host, encoded_path)
                }
            }
        }
    }

    /// Sign and send a request, returning the raw response.
    ///
    /// `metadata` headers (e.g. `x-amz-meta-expires`) are upserted into the
    /// signed-header set before signing so they are covered by the signature.
    ///
    /// Transient failures (transport errors, 429, 5xx) are retried with
    /// exponential backoff (up to [`MAX_SEND_ATTEMPTS`] total attempts): S3
    /// throttles aggressively under burst load and a single 503 previously
    /// turned the op into a miss/warn. Bodies are replayable (`Vec<u8>` held
    /// across attempts) and every attempt is freshly signed, so retries are
    /// safe for GET/PUT/DELETE/HEAD alike. Non-retryable statuses (4xx other
    /// than 429) return immediately.
    async fn send(
        &self,
        method: reqwest::Method,
        object_key: &str,
        body: Vec<u8>,
        metadata: Vec<(String, String)>,
    ) -> Result<reqwest::Response, Error> {
        let (full, host, canonical_path) = self.url_for(object_key);
        let payload_hash = hex::encode(sha256(&body));

        let mut last_err: Option<Error> = None;
        for attempt in 0..MAX_SEND_ATTEMPTS {
            if attempt > 0 {
                // 100ms, 200ms, 400ms … — small enough to keep cache ops fast.
                let backoff = RETRY_BASE_DELAY * (1 << (attempt - 1).min(4));
                tokio::time::sleep(backoff).await;
            }
            // Fresh timestamp per attempt: SigV4 rejects skewed `x-amz-date`
            // (>15min), so a reused timestamp would fail the retry anyway.
            let amz_date = amz_date_now();

            let mut headers: Vec<(String, String)> = Vec::new();
            upsert(&mut headers, "host", &host);
            upsert(&mut headers, "x-amz-date", &amz_date);
            upsert(&mut headers, "x-amz-content-sha256", &payload_hash);
            if let Some(token) = &self.session_token {
                upsert(&mut headers, "x-amz-security-token", token);
            }
            for (k, v) in &metadata {
                upsert(&mut headers, k, v);
            }

            let authorization = sign(
                method.as_str(),
                &canonical_path,
                "", // no query string
                &headers,
                &payload_hash,
                &amz_date,
                &self.access_key,
                &self.secret_key,
                &self.region,
            )?;
            upsert(&mut headers, "authorization", &authorization);

            let mut req = self.client.request(method.clone(), &full);
            for (k, v) in &headers {
                // `host` is set by reqwest from the URL; `authorization` and all
                // other signed headers must be attached or the request goes unsigned.
                if k == "host" {
                    continue;
                }
                req = req.header(k, v);
            }
            if !body.is_empty() {
                req = req.body(body.clone());
            }
            match req.send().await {
                Ok(resp) => {
                    if is_retryable_status(resp.status()) && attempt + 1 < MAX_SEND_ATTEMPTS {
                        tracing::debug!(
                            status = %resp.status(),
                            attempt = attempt + 1,
                            "s3 transient status, retrying"
                        );
                        continue;
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    // Transport errors (connect/timeout/reset) are retryable;
                    // builder-level errors surface identically, so cap the
                    // damage with the same attempt bound.
                    tracing::debug!(error = %e, attempt = attempt + 1, "s3 request failed, retrying");
                    last_err = Some(Error::Network(format!("s3 request failed: {e}")));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Network("s3 request failed".into())))
    }
}

impl S3Cache {
    async fn ping_inner(&self) -> bool {
        let object_key = self.object_key(PING_KEY);
        let response = match self
            .send(reqwest::Method::HEAD, &object_key, Vec::new(), Vec::new())
            .await
        {
            Ok(response) => response,
            Err(_) => return false,
        };
        match response.status().as_u16() {
            200 => return true,
            // HeadObject intentionally omits the specific S3 error: 404 can
            // mean NoSuchKey OR NoSuchBucket, and without ListBucket an absent
            // key can be 403. Never infer readiness from these statuses alone.
            403 | 404 => {}
            _ => return false,
        }
        drop(response);

        let response = match self
            .send(
                reqwest::Method::PUT,
                &object_key,
                PING_BODY.to_vec(),
                Vec::new(),
            )
            .await
        {
            Ok(response) => response,
            Err(_) => return false,
        };
        if !response.status().is_success() {
            return false;
        }
        drop(response);

        // A successful write alone is insufficient: a role might allow writes
        // but forbid reads. Keeping the marker also avoids a write on every
        // subsequent readiness check (and excess versions in versioned buckets).
        match self
            .send(reqwest::Method::HEAD, &object_key, Vec::new(), Vec::new())
            .await
        {
            Ok(response) => response.status() == reqwest::StatusCode::OK,
            Err(_) => false,
        }
    }

    async fn get_inner(&self, key: &str) -> Option<String> {
        let object_key = self.object_key(key);
        match self
            .send(reqwest::Method::GET, &object_key, Vec::new(), Vec::new())
            .await
        {
            Ok(mut resp) => {
                let status = resp.status();
                if status.as_u16() == 404 {
                    return None;
                }
                if status.as_u16() == 403 {
                    // Auth / signature misconfiguration, NOT a miss: without
                    // a distinct signal every 403 (bad keys, wrong region,
                    // unsigned prefix) is indistinguishable from "not cached"
                    // and the backend silently stops caching. Keep the
                    // fail-open read (return None) but log at warn with an
                    // actionable message so operators notice.
                    tracing::warn!(
                        status = %status,
                        key = %key,
                        "s3 get forbidden: check access_key/secret_key, region, bucket and endpoint; treating as miss"
                    );
                    return None;
                }
                if !status.is_success() {
                    tracing::warn!(status = %status, key = %key, "s3 get non-success");
                    return None;
                }
                if is_expired(&resp) {
                    tracing::debug!(key = %key, "s3 get object past client-side expiry");
                    return None;
                }
                if let Some(len) = resp.content_length() {
                    if len > self.max_object_size as u64 {
                        tracing::warn!(
                            key = %key,
                            len,
                            cap = self.max_object_size,
                            "s3 get object exceeds max size"
                        );
                        return None;
                    }
                }
                let mut buf = Vec::new();
                loop {
                    match resp.chunk().await {
                        Ok(Some(chunk)) => {
                            if buf.len() + chunk.len() > self.max_object_size {
                                tracing::warn!(
                                    key = %key,
                                    cap = self.max_object_size,
                                    "s3 get object exceeded max size while streaming"
                                );
                                return None;
                            }
                            buf.extend_from_slice(&chunk);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            tracing::warn!(error = %e, key = %key, "s3 get body read failed");
                            return None;
                        }
                    }
                }
                if is_expired(&resp) {
                    return None;
                }
                match String::from_utf8(buf) {
                    Ok(text) => Some(text),
                    Err(e) => {
                        tracing::warn!(error = %e, key = %key, "s3 get body not utf-8");
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "s3 get failed");
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
                "s3 set skipped oversized value"
            );
            return;
        }
        let object_key = self.object_key(key);
        let body = value.into_bytes();
        let metadata = expires_metadata(ttl);
        match self
            .send(reqwest::Method::PUT, &object_key, body, metadata)
            .await
        {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::warn!(status = %resp.status(), key = %key, "s3 set non-success");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "s3 set failed");
            }
        }
    }

    async fn delete_inner(&self, key: &str) {
        let object_key = self.object_key(key);
        match self
            .send(reqwest::Method::DELETE, &object_key, Vec::new(), Vec::new())
            .await
        {
            Ok(resp) => {
                // 204 on success; 404 (key absent) is not an error.
                let status = resp.status().as_u16();
                if !(status == 204 || status == 200 || status == 404) {
                    tracing::warn!(status = %status, key = %key, "s3 delete non-success");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, key = %key, "s3 delete failed");
            }
        }
    }
}

#[async_trait]
impl CacheTrait for S3Cache {
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
            "s3 cache backend does not support atomic counters; layer a counter store".into(),
        ))
    }

    /// Unsupported: see `Cache::incr`.
    async fn incr_if_below(
        &self,
        _key: &str,
        _delta: f64,
        _ceiling: f64,
        _ttl: Option<Duration>,
    ) -> Result<IncrResult, Error> {
        Err(Error::Config(
            "s3 cache backend does not support atomic counters; layer a counter store".into(),
        ))
    }

    /// No-op: counters are not modeled. A miss is the correct fail-open for an
    /// unsupported read. See module docs.
    async fn counter_value(&self, _key: &str) -> Option<f64> {
        None
    }

    /// Like `Cache::counter_value`: counters are not modeled, so a read is
    /// an absence (`Ok(None)`) rather than a routing through the blob store.
    async fn try_counter_value(&self, _key: &str) -> Result<Option<f64>, Error> {
        Ok(None)
    }

    /// Readiness via a readable reserved marker, initialized on 403/404 using
    /// object-write permissions. Missing buckets, auth errors and outages are
    /// unhealthy. No bucket-listing permission is needed; see module docs.
    async fn ping(&self) -> bool {
        cache_operation("ping", self.ping_inner())
            .await
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// SigV4 signing (self-contained; avoids aws-sdk-* and unsafe code).
// ---------------------------------------------------------------------------

/// Build the `Authorization` header value for a request.
///
/// `canonical_path` must already be percent-encoded with [`encode_path`] so it
/// can be used verbatim in both the request URL and the canonical request
/// (no double encoding).
#[allow(clippy::too_many_arguments)]
fn sign(
    method: &str,
    canonical_path: &str,
    canonical_query: &str,
    headers: &[(String, String)],
    payload_hash: &str,
    amz_date: &str,
    access_key: &str,
    secret_key: &str,
    region: &str,
) -> Result<String, Error> {
    let date_stamp = amz_date
        .get(..8)
        .ok_or_else(|| Error::Config("s3 signing timestamp must contain a date stamp".into()))?;

    // SigV4 mandates lowercase header names in the canonical request. All
    // callers pass lowercase names today, but normalizing here keeps the
    // signature correct if a mixed-case metadata name ever arrives (a raw
    // byte-wise sort would otherwise sign `X-Amz-Meta-Foo` while the wire
    // form differs, yielding SignatureDoesNotMatch). `upsert` is already
    // case-insensitive, so lowering cannot create colliding pairs.
    let lowered: Vec<(String, &str)> = headers
        .iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v.as_str()))
        .collect();
    let mut sorted: Vec<(&str, &str)> = lowered.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let canonical_headers: String = sorted
        .iter()
        .map(|(k, v)| format!("{k}:{}\n", v.trim()))
        .collect();
    let signed_headers: String = sorted.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(";");

    let canonical_request = format!(
        "{method}\n{path}\n{query}\n{hdrs}\n{signed}\n{hash}",
        method = method.to_ascii_uppercase(),
        path = canonical_path,
        query = canonical_query,
        hdrs = canonical_headers,
        signed = signed_headers,
        hash = payload_hash,
    );

    let credential_scope = format!("{date_stamp}/{region}/{SERVICE}/aws4_request");
    let hashed_canonical = hex::encode(sha256(canonical_request.as_bytes()));
    let string_to_sign =
        format!("{ALGORITHM}\n{amz_date}\n{credential_scope}\n{hashed_canonical}",);

    let signing_key = derive_signing_key(secret_key, date_stamp, region)?;
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes())?);
    Ok(format!(
        "{ALGORITHM} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
    ))
}

/// Derive the SigV4 signing key per the AWS spec.
fn derive_signing_key(secret_key: &str, date_stamp: &str, region: &str) -> Result<Vec<u8>, Error> {
    let k_date = hmac_sha256(
        format!("AWS4{secret_key}").as_bytes(),
        date_stamp.as_bytes(),
    )?;
    let k_region = hmac_sha256(&k_date, region.as_bytes())?;
    let k_service = hmac_sha256(&k_region, SERVICE.as_bytes())?;
    hmac_sha256(&k_service, b"aws4_request")
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|e| Error::Config(format!("s3 request signing failed: {e}")))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

/// Insert or replace `name` in the header list (case-insensitive on name).
fn upsert(headers: &mut Vec<(String, String)>, name: &str, value: &str) {
    if let Some(slot) = headers
        .iter_mut()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
    {
        slot.1 = value.to_string();
    } else {
        headers.push((name.to_string(), value.to_string()));
    }
}

/// Percent-encode a path, preserving `/` and RFC 3986 unreserved chars.
/// The result is suitable for both the request URL path and the SigV4
/// canonical URI (single encoding, no double-encoding).
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

/// Validate an S3 bucket name at construction: it must be non-empty.
/// Deliberately lenient (no AWS 3-63-char / charset rules): this backend
/// targets S3-compatible stores (MinIO/R2) whose naming rules differ, and
/// existing callers use short fixture-style names. An empty bucket is the
/// real misconfiguration signal (forgotten env var) — it otherwise builds
/// malformed URLs that 403 on every request.
fn validate_bucket_name(bucket: &str) -> Result<(), Error> {
    if bucket.trim().is_empty() {
        Err(Error::Config("s3 bucket must not be empty".into()))
    } else {
        Ok(())
    }
}

/// Validate a custom endpoint override: must parse as `http(s)://host[:port]`
/// with a non-empty host. A typo here otherwise surfaces as a connection
/// error on every cache op with no hint at the misconfiguration.
fn validate_endpoint(endpoint: &str) -> Result<(), Error> {
    let after_scheme = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or("");
    let (scheme, host) = (
        endpoint.split_once("://").map(|(s, _)| s).unwrap_or(""),
        after_scheme.split('/').next().unwrap_or(""),
    );
    if (scheme == "http" || scheme == "https") && !host.is_empty() {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "s3 endpoint {endpoint:?} is invalid: expected http(s)://host[:port][/prefix]"
        )))
    }
}

/// Extract the `host[:port]` component of a base URL (no scheme, no path).
fn host_of(base: &str) -> &str {
    let after_scheme = base.split("://").nth(1).unwrap_or(base);
    after_scheme.split('/').next().unwrap_or(after_scheme)
}

/// Extract the path component (starting with `/`) of a base URL, or `""`
/// when the base is scheme+authority only. Used by [`S3Cache::url_for`] so an
/// endpoint with a path prefix is included in the SigV4 canonical URI exactly
/// as it appears in the request URL.
fn endpoint_path_of(base: &str) -> &str {
    let after_scheme = match base.split_once("://") {
        Some((_, rest)) => rest,
        None => base,
    };
    match after_scheme.find('/') {
        Some(i) => &after_scheme[i..],
        None => "",
    }
}

/// Build the `x-amz-meta-expires` metadata header pair for a PUT when a TTL is
/// requested. The value is the Unix-second deadline at which the object should
/// be treated as stale on read (S3 does not auto-delete on `Expires`). Returns
/// an empty vector when no TTL is set.
fn expires_metadata(ttl: Option<Duration>) -> Vec<(String, String)> {
    let ttl = match ttl {
        Some(t) => t,
        None => return Vec::new(),
    };
    if ttl.is_zero() {
        return vec![("x-amz-meta-expires".to_string(), "0".to_string())];
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    vec![(
        "x-amz-meta-expires".to_string(),
        now.saturating_add(ttl.as_secs().max(1)).to_string(),
    )]
}

/// Returns `true` when the response carries an `x-amz-meta-expires` header
/// whose Unix-second deadline has passed. Enforces per-object TTL client-side.
fn is_expired(resp: &reqwest::Response) -> bool {
    let Some(header) = resp.headers().get("x-amz-meta-expires") else {
        return false;
    };
    let Some(deadline) = header.to_str().ok().and_then(|v| v.parse::<u64>().ok()) else {
        return true; // Malformed expiry must not turn a TTL-bearing object immortal.
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    now >= deadline
}

/// Current UTC time formatted as AWS's `yyyyMMddTHHmmssZ` "amz date".
fn amz_date_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day, hour, min, sec) = decompose_utc(secs);
    format!("{year:04}{month:02}{day:02}T{hour:02}{min:02}{sec:02}Z")
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
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
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

    #[test]
    fn object_key_applies_prefix() {
        let cache = S3Cache::new("b", "us-east-1", "oxllm", "ak", "sk", None, None).unwrap();
        assert_eq!(cache.object_key("litellm:vcr:abc"), "oxllm:litellm:vcr:abc");
    }

    #[test]
    fn constructor_rejects_empty_prefix() {
        for prefix in ["", " ", "\t\n"] {
            assert!(S3Cache::new("b", "us-east-1", prefix, "ak", "sk", None, None).is_err());
        }
    }

    #[tokio::test]
    async fn write_cap_matches_read_cap_in_bytes() {
        let server = httpmock::MockServer::start();
        let mut cache = s3_mock(&server);
        cache.max_object_size = 4;
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/mybucket/oxllm%3Ak");
            then.status(200);
        });
        let get = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/mybucket/oxllm%3Ak");
            then.status(200).body("éé");
        });
        cache.set("k", "éé".to_string(), None).await;
        cache.set("k", "ééx".to_string(), None).await;
        put.assert_hits(1);
        assert_eq!(cache.get("k").await.as_deref(), Some("éé"));
        get.assert_hits(1);
    }

    #[tokio::test]
    async fn zero_ttl_invalidates_instead_of_uploading() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let delete = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path("/mybucket/oxllm%3Ak");
            then.status(204);
        });
        cache.set("k", "old value".to_string(), None).await;
        cache
            .set("k", "new value".to_string(), Some(Duration::ZERO))
            .await;
        put.assert_hits(1);
        delete.assert_hits(1);
        assert_eq!(expires_metadata(Some(Duration::ZERO))[0].1, "0");
    }

    #[tokio::test]
    async fn all_object_operations_share_a_short_total_deadline() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
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
    async fn get_deadline_includes_a_stalled_response_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (body_sent, mut body_received) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let read = socket.read(&mut request).await.unwrap();
            assert!(read > 0, "client request should arrive");
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nx")
                .await
                .unwrap();
            let _ = body_sent.send(());
            std::future::pending::<()>().await;
        });
        let cache = S3Cache::new(
            "bucket",
            "us-east-1",
            "oxllm",
            "ak",
            "sk",
            None,
            Some(endpoint),
        )
        .unwrap();
        let result = tokio::time::timeout(
            CACHE_OPERATION_TIMEOUT + Duration::from_millis(500),
            cache.get("k"),
        )
        .await;
        server.abort();
        assert!(
            body_received.try_recv().is_ok(),
            "the partial response must be exercised"
        );
        assert!(
            matches!(result, Ok(None)),
            "a partially received body must not escape the cache deadline"
        );
    }

    #[tokio::test]
    async fn malformed_expiry_is_a_miss() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        server.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(200)
                .header("x-amz-meta-expires", "invalid")
                .body("stale");
        });
        assert!(cache.get("k").await.is_none());
    }

    #[test]
    fn constructor_rejects_empty_bucket_region_and_bad_endpoint() {
        assert!(S3Cache::new("", "us-east-1", "p", "ak", "sk", None, None).is_err());
        assert!(S3Cache::new("b", "", "p", "ak", "sk", None, None).is_err());
        assert!(
            S3Cache::new(
                "b",
                "us-east-1",
                "p",
                "ak",
                "sk",
                None,
                Some("nope".to_string())
            )
            .is_err()
        );
        // Short MinIO-style names and mock endpoints stay accepted.
        assert!(S3Cache::new("b", "us-east-1", "p", "ak", "sk", None, None).is_ok());
        assert!(
            S3Cache::new(
                "b",
                "us-east-1",
                "p",
                "",
                "",
                None,
                Some("http://localhost:9000".to_string()),
            )
            .is_ok()
        );
    }

    #[test]
    fn url_for_aws_uses_virtual_host_style() {
        let cache = S3Cache::new("mybucket", "us-west-2", "oxllm", "ak", "sk", None, None).unwrap();
        let (full, host, path) = cache.url_for("oxllm:k1");
        assert_eq!(host, "mybucket.s3.us-west-2.amazonaws.com");
        assert!(full.starts_with("https://mybucket.s3.us-west-2.amazonaws.com/"));
        assert_eq!(path, "/oxllm%3Ak1");
    }

    #[test]
    fn url_for_custom_endpoint_uses_path_style() {
        let cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "ak",
            "sk",
            None,
            Some("http://localhost:9000".to_string()),
        )
        .unwrap();
        let (full, host, path) = cache.url_for("oxllm:k1");
        assert_eq!(host, "localhost:9000");
        assert_eq!(full, "http://localhost:9000/mybucket/oxllm%3Ak1");
        assert_eq!(path, "/mybucket/oxllm%3Ak1");
    }

    /// A custom endpoint with a path prefix (e.g. a gateway at
    /// `http://host:9000/gateway`) must include the prefix in the SigV4
    /// canonical URI as well: the signature covers the path exactly as sent
    /// on the wire, so signing only `/{bucket}/{key}` while requesting
    /// `{base}/{bucket}/{key}` 403s every request (SignatureDoesNotMatch) —
    /// `get` silently returns None and `set` only warns, disabling the cache.
    #[test]
    fn url_for_custom_endpoint_with_path_prefix_signs_prefix() {
        for endpoint in [
            "http://localhost:9000/gateway",
            // Trailing slash on the prefix must behave identically.
            "http://localhost:9000/gateway/",
        ] {
            let cache = S3Cache::new(
                "mybucket",
                "us-east-1",
                "oxllm",
                "ak",
                "sk",
                None,
                Some(endpoint.to_string()),
            )
            .unwrap();
            let (full, host, path) = cache.url_for("oxllm:k1");
            assert_eq!(host, "localhost:9000", "host for {endpoint}");
            assert_eq!(
                full, "http://localhost:9000/gateway/mybucket/oxllm%3Ak1",
                "full URL for {endpoint}"
            );
            assert_eq!(
                path, "/gateway/mybucket/oxllm%3Ak1",
                "canonical path for {endpoint}"
            );
        }
    }

    #[test]
    fn endpoint_path_of_extracts_path_component() {
        assert_eq!(endpoint_path_of("http://h:9000"), "");
        assert_eq!(endpoint_path_of("http://h:9000/gateway"), "/gateway");
        assert_eq!(endpoint_path_of("http://h:9000/a/b"), "/a/b");
        assert_eq!(endpoint_path_of("http://h/gateway/"), "/gateway/");
    }

    #[test]
    fn encode_path_preserves_unreserved_and_slash() {
        assert_eq!(encode_path("/a/b_c-1.2~3"), "/a/b_c-1.2~3");
    }

    #[test]
    fn encode_path_percent_encodes_special_chars() {
        // ':' is not unreserved → encoded.
        assert_eq!(encode_path("/oxllm:k"), "/oxllm%3Ak");
        assert_eq!(encode_path("/a b"), "/a%20b");
    }

    #[test]
    fn sign_produces_well_formed_authorization() -> Result<(), Error> {
        let mut headers: Vec<(String, String)> = Vec::new();
        upsert(&mut headers, "host", "example.amazonaws.com");
        upsert(&mut headers, "x-amz-date", "20150830T123600Z");
        upsert(&mut headers, "x-amz-content-sha256", "e3b0c442");
        let auth = sign(
            "GET",
            "/",
            "",
            &headers,
            "e3b0c442",
            "20150830T123600Z",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "us-east-1",
        )?;
        assert!(auth.starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/s3/aws4_request"
        ));
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
        assert!(auth.contains("Signature="));
        Ok(())
    }

    /// Verified against the documented AWS SigV4 "Get Object" example
    /// (https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html)
    /// and re-derived independently with a Python `hmac`+`hashlib` reference.
    #[test]
    fn sign_matches_aws_s3_get_object_test_vector() -> Result<(), Error> {
        let payload_hash = hex::encode(sha256(&[])); // empty-body hash
        let mut headers: Vec<(String, String)> = Vec::new();
        upsert(&mut headers, "host", "examplebucket.s3.amazonaws.com");
        upsert(&mut headers, "x-amz-content-sha256", &payload_hash);
        upsert(&mut headers, "x-amz-date", "20130524T000000Z");
        let auth = sign(
            "GET",
            "/test.txt",
            "",
            &headers,
            &payload_hash,
            "20130524T000000Z",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "us-east-1",
        )?;
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;x-amz-content-sha256;x-amz-date, \
             Signature=df548e2ce037944d03f3e68682813b093763996d597cf890ca3d9037fd231eb4"
        );
        Ok(())
    }

    #[tokio::test]
    async fn incr_returns_err_not_silent_success() {
        let cache = S3Cache::new("b", "us-east-1", "oxllm", "ak", "sk", None, None).unwrap();
        assert!(cache.incr("c", 5.0, None).await.is_err());
    }

    #[tokio::test]
    async fn incr_if_below_returns_err() {
        let cache = S3Cache::new("b", "us-east-1", "oxllm", "ak", "sk", None, None).unwrap();
        assert!(cache.incr_if_below("c", 5.0, 1.0, None).await.is_err());
    }

    #[tokio::test]
    async fn counter_value_is_none() {
        let cache = S3Cache::new("b", "us-east-1", "oxllm", "ak", "sk", None, None).unwrap();
        assert!(cache.counter_value("c").await.is_none());
    }

    #[tokio::test]
    async fn try_counter_value_returns_ok_none() {
        let cache = S3Cache::new("b", "us-east-1", "oxllm", "ak", "sk", None, None).unwrap();
        assert_eq!(cache.try_counter_value("c").await.unwrap(), None);
    }

    /// End-to-end get/set/delete against a mocked S3 endpoint. Verifies that
    /// the signed requests are accepted (200) and that values round-trip.
    #[tokio::test]
    async fn get_set_delete_round_trip_via_mock() {
        let server = httpmock::MockServer::start();
        let cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            Some(server.base_url()),
        )
        .unwrap();

        let key = "litellm:vcr:demo";

        // PUT: accept any signed PUT on the object path.
        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/mybucket/oxllm%3Alitellm%3Avcr%3Ademo");
            then.status(200).body("");
        });
        cache.set(key, "{\"cached\":true}".to_string(), None).await;
        put_mock.assert();

        // GET: return the stored body.
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/mybucket/oxllm%3Alitellm%3Avcr%3Ademo");
            then.status(200).body("{\"cached\":true}");
        });
        let val = cache.get(key).await;
        get_mock.assert();
        assert_eq!(val.as_deref(), Some("{\"cached\":true}"));

        // DELETE: succeed.
        let del_mock = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path("/mybucket/oxllm%3Alitellm%3Avcr%3Ademo");
            then.status(204).body("");
        });
        cache.delete(key).await;
        del_mock.assert();

        // GET miss → 404 → None. Use a distinct, never-written key so the
        // earlier 200 GET mock above doesn't shadow the 404 mock (httpmock
        // returns the first-created matching mock).
        let miss_key = "litellm:vcr:never";
        let miss_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/mybucket/oxllm%3Alitellm%3Avcr%3Anever");
            then.status(404).body("");
        });
        assert!(cache.get(miss_key).await.is_none());
        miss_mock.assert();
    }

    /// End-to-end get/set against a mocked S3 endpoint **with a path-prefix
    /// gateway** (`{base}/gateway`): requests must go to
    /// `/gateway/{bucket}/{key}` and the canonical (signed) URI must equal the
    /// wire path — the mismatch this guards against made every request 403
    /// with SignatureDoesNotMatch against real MinIO-style gateways.
    #[tokio::test]
    async fn get_set_round_trip_via_mock_with_path_prefix_endpoint() {
        let server = httpmock::MockServer::start();
        let cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            Some(format!("{}/gateway", server.base_url())),
        )
        .unwrap();

        let key = "litellm:vcr:demo";
        let object_path = "/gateway/mybucket/oxllm%3Alitellm%3Avcr%3Ademo";

        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path(object_path);
            then.status(200).body("");
        });
        cache.set(key, "{\"cached\":true}".to_string(), None).await;
        put_mock.assert();

        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path(object_path);
            then.status(200).body("{\"cached\":true}");
        });
        let val = cache.get(key).await;
        get_mock.assert();
        assert_eq!(val.as_deref(), Some("{\"cached\":true}"));
    }

    #[test]
    fn decompose_utc_known_timestamps() {
        // Unix epoch.
        assert_eq!(decompose_utc(0), (1970, 1, 1, 0, 0, 0));
        // AWS SigV4 "Get Object" test-vector date: 2013-05-24T00:00:00Z.
        assert_eq!(decompose_utc(1_369_353_600), (2013, 5, 24, 0, 0, 0));
        // AWS SigV4 example date with a nonzero time: 2015-08-30T12:36:00Z.
        assert_eq!(decompose_utc(1_440_938_160), (2015, 8, 30, 12, 36, 0));
        // Leap day: 2024-02-29T00:00:00Z.
        assert_eq!(decompose_utc(1_709_164_800), (2024, 2, 29, 0, 0, 0));
        // Non-trivial HMS: 2009-02-13T23:31:30Z.
        assert_eq!(decompose_utc(1_234_567_890), (2009, 2, 13, 23, 31, 30));
    }

    #[test]
    fn url_for_dotted_bucket_uses_path_style() {
        // Dotted bucket names cannot use virtual-host style over HTTPS (TLS
        // cert mismatch), so the AWS arm must switch to path-style.
        let cache =
            S3Cache::new("my.bucket", "us-east-1", "oxllm", "ak", "sk", None, None).unwrap();
        let (full, host, path) = cache.url_for("oxllm:k1");
        assert_eq!(host, "s3.us-east-1.amazonaws.com");
        assert_eq!(
            full,
            "https://s3.us-east-1.amazonaws.com/my.bucket/oxllm%3Ak1"
        );
        assert_eq!(path, "/my.bucket/oxllm%3Ak1");
    }

    /// `get` must refuse to read an object whose body exceeds the configured
    /// size cap, returning `None` without allocating the full body.
    #[tokio::test]
    async fn get_returns_none_when_object_exceeds_max_size() {
        let server = httpmock::MockServer::start();
        let mut cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            Some(server.base_url()),
        )
        .unwrap();
        // Tight cap so the oversized body trips it.
        cache.max_object_size = 4;

        let oversized = "x".repeat(100);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/mybucket/oxllm%3Abig");
            then.status(200).body(oversized);
        });
        assert!(cache.get("big").await.is_none());
        get_mock.assert();
    }

    /// After dropping the redundant explicit header, the PUT must still carry
    /// exactly one `x-amz-content-sha256` matching the body's hash. Matching
    /// the header value proves the loop is the (sole) source of truth.
    #[tokio::test]
    async fn put_sends_x_amz_content_sha256_once() {
        let server = httpmock::MockServer::start();
        let cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            Some(server.base_url()),
        )
        .unwrap();

        let body = "{\"cached\":true}";
        let hash = hex::encode(sha256(body.as_bytes()));
        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/mybucket/oxllm%3Ak")
                .header("x-amz-content-sha256", hash.as_str());
            then.status(200).body("");
        });
        cache.set("k", body.to_string(), None).await;
        put_mock.assert();
    }

    /// Regression guard for the CRITICAL bug where the computed `Authorization`
    /// header was dropped before sending. PUT/GET/DELETE must each carry a
    /// well-formed SigV4 `Authorization` header. Without this assertion the mock
    /// fakes (which don't validate signatures) hide the omission entirely.
    #[tokio::test]
    async fn send_attaches_authorization_header_on_all_methods() {
        let server = httpmock::MockServer::start();
        let cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            Some(server.base_url()),
        )
        .unwrap();

        let path = "/mybucket/oxllm%3Ak";

        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path(path)
                .header_exists("authorization");
            then.status(200).body("");
        });
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path(path)
                .header_exists("authorization");
            then.status(200).body("{}");
        });
        let del_mock = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE)
                .path(path)
                .header_exists("authorization");
            then.status(204).body("");
        });

        cache.set("k", "{}".to_string(), None).await;
        let _ = cache.get("k").await;
        cache.delete("k").await;

        put_mock.assert();
        get_mock.assert();
        del_mock.assert();
    }

    #[test]
    fn url_for_custom_endpoint_normalizes_uppercase_host() {
        let cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "ak",
            "sk",
            None,
            Some("http://MyMinIO.LOCAL:9000".to_string()),
        )
        .unwrap();
        let (_full, host, _path) = cache.url_for("oxllm:k1");
        assert_eq!(host, "myminio.local:9000");
    }

    #[tokio::test]
    async fn ping_reports_healthy_on_200() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let head = server.mock(|when, then| {
            when.method(httpmock::Method::HEAD)
                .path("/mybucket/oxllm%3A__oxllm_ping__")
                .header_exists("authorization");
            then.status(200).body("");
        });
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(403);
        });
        assert!(cache.ping().await);
        head.assert_hits(1);
        put.assert_hits(0);
    }

    #[tokio::test]
    async fn ping_reports_unhealthy_on_missing_bucket() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let head = server.mock(|when, then| {
            when.method(httpmock::Method::HEAD)
                .path("/mybucket/oxllm%3A__oxllm_ping__");
            then.status(404).body("");
        });
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/mybucket/oxllm%3A__oxllm_ping__");
            then.status(404)
                .body("<Error><Code>NoSuchBucket</Code></Error>");
        });
        assert!(!cache.ping().await);
        head.assert_hits(1);
        put.assert_hits(1);
    }

    #[tokio::test]
    async fn ping_reports_unhealthy_on_auth_failure() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let head = server.mock(|when, then| {
            when.method(httpmock::Method::HEAD);
            then.status(403).body("");
        });
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(403)
                .body("<Error><Code>AccessDenied</Code></Error>");
        });
        assert!(!cache.ping().await);
        head.assert_hits(1);
        put.assert_hits(1);
    }

    #[tokio::test]
    async fn ping_reports_unhealthy_on_5xx() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        server.mock(|when, then| {
            when.method(httpmock::Method::HEAD);
            then.status(503).body("");
        });
        assert!(!cache.ping().await);
    }

    #[tokio::test]
    async fn ping_reports_unhealthy_on_network_error() {
        // Point at a port that is not listening → connection refused.
        let cache = S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            Some("http://127.0.0.1:1".to_string()),
        )
        .unwrap();
        assert!(!cache.ping().await);
    }

    #[tokio::test]
    async fn set_stores_expires_metadata_when_ttl_given() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/mybucket/oxllm%3Ak")
                .header_exists("x-amz-meta-expires");
            then.status(200).body("");
        });
        cache
            .set("k", "{}".to_string(), Some(Duration::from_secs(300)))
            .await;
        put_mock.assert();
    }

    #[tokio::test]
    async fn set_without_ttl_omits_expires_metadata() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .path("/mybucket/oxllm%3Ak")
                .header_exists("x-amz-meta-expires");
            then.status(200).body("");
        });
        cache.set("k", "{}".to_string(), None).await;
        put_mock.assert_hits(0);
    }

    #[tokio::test]
    async fn get_returns_none_when_object_past_expiry() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/mybucket/oxllm%3Ak");
            then.status(200)
                .header("x-amz-meta-expires", "0")
                .body("{\"cached\":true}");
        });
        assert!(cache.get("k").await.is_none());
        get_mock.assert();
    }

    #[tokio::test]
    async fn get_returns_value_when_not_past_expiry() {
        let server = httpmock::MockServer::start();
        let cache = s3_mock(&server);
        let far_future: String = (u64::MAX).to_string();
        let get_mock = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/mybucket/oxllm%3Ak");
            then.status(200)
                .header("x-amz-meta-expires", far_future.as_str())
                .body("{\"cached\":true}");
        });
        assert_eq!(cache.get("k").await.as_deref(), Some("{\"cached\":true}"));
        get_mock.assert();
    }

    /// Construct an `S3Cache` against a mocked S3 endpoint (helper).
    fn s3_mock(server: &httpmock::MockServer) -> S3Cache {
        S3Cache::new(
            "mybucket",
            "us-east-1",
            "oxllm",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            Some(server.base_url()),
        )
        .unwrap()
    }

    mod readiness;
}
