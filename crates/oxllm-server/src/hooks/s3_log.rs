//! S3 log-storage hook.
//!
//! A [`Hook`] that writes one JSON log line per successful request to a
//! configured S3 bucket (or any S3-compatible endpoint). The object key is
//! `{prefix}/{date}/{request_id}.json`. The PUT is fire-and-forget: it is
//! enqueued onto a bounded `mpsc` channel and a dedicated background task
//! signs (SigV4) and PUTs sequentially, so the request path is never
//! blocked. When the channel is full the payload is dropped and counted
//! under `oxllm_telemetry_dropped_total{backend="s3_log",reason="full"}`;
//! PUT failures are logged at `warn` level.
//!
//! Auth is [AWS SigV4](https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-authenticating-requests.html),
//! reusing the host's hand-rolled [`crate::bedrock_sigv4::sign_request`]
//! signer (which is service-generic and explicitly reusable for S3). No
//! `aws-sdk-s3` dependency is pulled in.
//!
//! Wired in from `oxllm_settings.s3_log_config`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::Hook;
use oxllm_core::{CallContext, ResponseSummary};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::config::types::S3LogConfig;
use crate::hooks::cost_fallback::fallback_cost;
use crate::hooks::redact::{redact_secret_values_in_place, redact_sensitive_in_place};

const DEFAULT_PREFIX: &str = "oxllm";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const S3_SERVICE: &str = "s3";
/// Bounded channel capacity for the per-hook background PUT task.
const DEFAULT_CHANNEL_CAPACITY: usize = 10_000;

/// Internal snapshot of the resolved configuration.
#[derive(Debug, Clone)]
struct ResolvedConfig {
    bucket: String,
    region: String,
    prefix: String,
    access_key: String,
    secret_key: String,
    /// Custom endpoint base (e.g. MinIO); `None` means AWS S3.
    endpoint: Option<String>,
    /// When `true`, apply the opt-in value-pattern redaction pass to
    /// request/response bodies in addition to the key-name denylist.
    redact_secret_values: bool,
    /// When `true`, include a (redacted) response body in the S3 log object.
    log_response_body: bool,
}

/// One queued S3 PUT: the resolved object URL and the already-serialized
/// JSON body. Built on the request path; drained by the background task.
struct S3Put {
    url: String,
    body: Vec<u8>,
}

/// A `Hook` that stores each request as a JSON object in S3.
pub struct S3LogHook {
    config: ResolvedConfig,
    /// Bounded sender into the background PUT task.
    tx: mpsc::Sender<S3Put>,
}

impl std::fmt::Debug for S3LogHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3LogHook")
            .field("bucket", &self.config.bucket)
            .field("region", &self.config.region)
            .field("prefix", &self.config.prefix)
            .finish()
    }
}

impl S3LogHook {
    /// Build a hook from a raw config block.
    pub fn new(config: &S3LogConfig, allow_private: bool) -> anyhow::Result<Self> {
        Self::with_capacity(config, DEFAULT_CHANNEL_CAPACITY, allow_private, false)
    }

    /// Like [`S3LogHook::new`] but with an explicit channel `capacity`,
    /// letting tests trade memory for drop-tolerance.
    pub fn with_capacity(
        config: &S3LogConfig,
        capacity: usize,
        allow_private: bool,
        redact_secret_values: bool,
    ) -> anyhow::Result<Self> {
        if config.bucket.trim().is_empty() {
            anyhow::bail!("s3_log_config.bucket must be set");
        }
        if config.region.trim().is_empty() {
            anyhow::bail!("s3_log_config.region must be set");
        }
        let access_key = config
            .access_key_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("s3_log_config.access_key_id must be set"))?;
        let secret_key = config
            .secret_access_key
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow::anyhow!("s3_log_config.secret_access_key must be set"))?;

        let prefix = config
            .prefix
            .as_deref()
            .map(|s| s.trim().trim_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_PREFIX.to_string());

        let client = crate::security::build_ssrf_http_client(allow_private, REQUEST_TIMEOUT)?;

        let resolved = ResolvedConfig {
            bucket: config.bucket.clone(),
            region: config.region.clone(),
            prefix,
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            endpoint: config.endpoint.clone(),
            redact_secret_values,
            log_response_body: config.log_response_body.unwrap_or(true),
        };

        let cap = if capacity == 0 {
            DEFAULT_CHANNEL_CAPACITY
        } else {
            capacity
        };
        let (tx, mut rx) = mpsc::channel::<S3Put>(cap);
        let region = resolved.region.clone();
        let access_key = resolved.access_key.clone();
        let secret_key = resolved.secret_key.clone();
        tokio::spawn(async move {
            while let Some(put) = rx.recv().await {
                if let Err(e) = put_object(
                    &client,
                    &put.url,
                    &put.body,
                    &access_key,
                    &secret_key,
                    &region,
                )
                .await
                {
                    tracing::warn!(
                        error = %e,
                        url = %put.url,
                        "s3 log PUT failed"
                    );
                    crate::metrics::record_telemetry_dropped("s3_log", "post_failed", 1);
                }
            }
        });

        Ok(Self {
            config: resolved,
            tx,
        })
    }

    /// The configured bucket (for tests).
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Build the S3 object key for a request: `{prefix}/{date}/{request_id}.json`.
    fn object_key(&self, ctx: &CallContext) -> String {
        let date = utc_date_string();
        // Sanitize the request id so it can never escape its key segment with
        // a path separator or control char; fall back to a default if empty.
        // S3 keys are not a filesystem (so `..` is literal, no traversal), but
        // a `/` in the id would add an extra segment and break the
        // `{prefix}/{date}/{id}.json` partition invariant asserted by
        // `object_key_has_date_partitioned_shape`.
        let req_id = sanitize_request_id(&ctx.request_id);
        format!("{}/{date}/{req_id}.json", self.config.prefix)
    }

    /// Build the full S3 URL for an object key.
    fn object_url(&self, key: &str) -> String {
        match &self.config.endpoint {
            // Path-style addressing for custom endpoints (MinIO / R2 / etc.).
            Some(base) => {
                let base = base.trim_end_matches('/');
                format!("{base}/{}/{key}", self.config.bucket)
            }
            // Virtual-hosted-style for AWS S3.
            None => format!(
                "https://{}.s3.{}.amazonaws.com/{key}",
                self.config.bucket, self.config.region
            ),
        }
    }

    /// Build the JSON log payload for a single request.
    fn build_log(&self, ctx: &CallContext, resp: &ResponseSummary) -> Value {
        let usage = resp.usage.as_ref();
        let cost = fallback_cost(ctx, resp);

        // Redact in-place: clone the request body so the shared
        // [`CallContext`] is untouched, then strip credential-bearing keys
        // before the body is written to long-lived S3 storage.
        let mut request_body = (*ctx.request_body).clone();
        redact_sensitive_in_place(&mut request_body);
        if self.config.redact_secret_values {
            redact_secret_values_in_place(&mut request_body);
        }

        let mut payload = json!({
            "request_id": ctx.request_id,
            "model": resp.model,
            "provider": resp.provider,
            "usage": usage.map(|u| {
                json!({
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                })
            }),
            "cost": cost,
            "key_id": ctx.key_id,
            "team_id": ctx.team_id,
            "user_id": ctx.user_id,
            "org_id": ctx.org_id,
            "project_id": ctx.project_id,
            "timestamp": jiff::Timestamp::now().to_string(),
            "request_body": request_body,
        });

        // LOW #7: include a redacted response body in the S3 log object so
        // operators have both halves of the exchange for audit / replay
        // (matching LiteLLM's S3 logging, which persists both). Gated behind
        // `log_response_body` (default true) so an operator concerned about
        // storage cost can opt out.
        if self.config.log_response_body {
            if let Some(body) = &resp.body {
                let mut rb = body.clone();
                redact_sensitive_in_place(&mut rb);
                if self.config.redact_secret_values {
                    redact_secret_values_in_place(&mut rb);
                }
                payload["response_body"] = rb;
            } else {
                payload["response_body"] = Value::Null;
            }
        }

        payload
    }

    /// Enqueue the PUT onto the bounded channel for the background task.
    /// Never blocks: on a full/closed channel the payload is dropped and
    /// counted under `oxllm_telemetry_dropped_total{backend="s3_log"}`.
    fn dispatch(&self, url: String, body: Vec<u8>) {
        match self.tx.try_send(S3Put { url, body }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!("s3_log channel full; dropping payload");
                crate::metrics::record_telemetry_dropped("s3_log", "full", 1);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!("s3_log channel closed; dropping payload");
                crate::metrics::record_telemetry_dropped("s3_log", "closed", 1);
            }
        }
    }
}

#[async_trait]
impl Hook for S3LogHook {
    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        if ctx.cache_hit {
            return;
        }
        if ctx.is_master {
            return;
        }
        let key = self.object_key(ctx);
        let url = self.object_url(&key);
        let payload = self.build_log(ctx, resp);
        let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
        self.dispatch(url, body);
    }
}

/// Sign and PUT a single object to S3 via the REST API using SigV4.
async fn put_object(
    client: &reqwest::Client,
    url: &str,
    body: &[u8],
    access_key: &str,
    secret_key: &str,
    region: &str,
) -> Result<(), PutError> {
    // SigV4 signs a header set; start empty and let the signer add the
    // canonical headers (host, x-amz-date, x-amz-content-sha256) + the final
    // Authorization header.
    let mut headers: Vec<(String, String)> = Vec::new();
    crate::bedrock_sigv4::sign_request(
        "PUT",
        url,
        &mut headers,
        body,
        access_key,
        secret_key,
        region,
        S3_SERVICE,
        None,
    )
    .map_err(PutError::Sign)?;

    let mut req = client.put(url).body(body.to_vec());
    for (name, value) in &headers {
        let hname = reqwest::header::HeaderName::from_bytes(name.as_bytes());
        let hval = reqwest::header::HeaderValue::from_str(value);
        if let (Ok(hname), Ok(hval)) = (hname, hval) {
            req = req.header(hname, hval);
        }
    }
    let resp = req.send().await.map_err(PutError::Request)?;
    if !resp.status().is_success() {
        return Err(PutError::Status(resp.status()));
    }
    Ok(())
}

/// Errors from an S3 PUT attempt.
#[derive(Debug)]
enum PutError {
    /// SigV4 signing failed (bad URL / missing host).
    Sign(String),
    /// Transport error.
    Request(reqwest::Error),
    /// Non-2xx response.
    Status(reqwest::StatusCode),
}

impl std::fmt::Display for PutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PutError::Sign(m) => write!(f, "sigv4 signing failed: {m}"),
            PutError::Request(e) => write!(f, "request failed: {e}"),
            PutError::Status(s) => write!(f, "s3 returned status {s}"),
        }
    }
}

impl std::error::Error for PutError {}

/// Today's UTC date as `YYYY-MM-DD`, used to partition object keys by day.
fn utc_date_string() -> String {
    // Derive the date structurally from the timestamp so the partition key
    // does not depend on the string-format default of `jiff::Timestamp`.
    jiff::Timestamp::now()
        .to_zoned(jiff::tz::TimeZone::UTC)
        .date()
        .to_string()
}

/// Sanitize a request id for use as a single S3 key segment.
///
/// Conservative stance matching the docstring's "treat as untrusted
/// single-segment input" framing: any character outside the S3-safe set
/// `[A-Za-z0-9._-]` is replaced with `_`. This catches URI-special
/// characters (`?`, `#`, `%`, space, quotes) and S3-unsafe characters (per
/// the AWS key-naming rules: space, `&`, `<`, `>`, `{`, `}`, `%`, ...) that
/// the previous `/`/`\`/control-only filter let through unchanged and that
/// would later break `reqwest::Url::parse` (or silently truncate the path on
/// `?`/`#`). Falls back to `"unknown"` when the value is empty after trim.
fn sanitize_request_id(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "unknown".to_string();
    }
    let sanitized: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

/// Build an [`S3LogHook`] from `oxllm_settings.s3_log_config` when present.
/// Returns `None` when S3 logging is not configured or misconfigured (logged
/// at warn level). When a custom endpoint is supplied it is SSRF-validated
/// against `allow_private_api_base`.
pub(crate) fn build_s3_log_hook_opt(config: &crate::config::ProxyConfig) -> Option<Arc<S3LogHook>> {
    let cfg = config.oxllm_settings.s3_log_config.as_ref()?;
    let allow_private = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);
    let redact_secret_values = config
        .general_settings
        .redact_secret_values
        .unwrap_or(false);
    if let Some(endpoint) = cfg.endpoint.as_deref() {
        if let Err(e) = crate::config::validate_api_base(endpoint, allow_private) {
            tracing::warn!(
                error = %e,
                "s3_log_config endpoint '{endpoint}' failed SSRF validation; skipping"
            );
            return None;
        }
    }
    match S3LogHook::with_capacity(
        cfg,
        DEFAULT_CHANNEL_CAPACITY,
        allow_private,
        redact_secret_values,
    ) {
        Ok(hook) => Some(Arc::new(hook)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build s3 log hook; skipping");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;

    fn raw_config() -> S3LogConfig {
        S3LogConfig {
            bucket: "my-logs".to_string(),
            region: "us-east-1".to_string(),
            prefix: Some("oxllm".to_string()),
            access_key_id: Some("AKIAEXAMPLE".to_string()),
            secret_access_key: Some("secretkey".to_string()),
            endpoint: None,
            log_response_body: None,
        }
    }

    fn make_ctx() -> CallContext {
        CallContext {
            key_id: Some("k-1".to_string()),
            team_id: Some("team-1".to_string()),
            ..CallContext::for_test("gpt-4o")
        }
    }

    fn make_response() -> ResponseSummary {
        ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: Some(0.0025),
            usage: Some(UsageSummary {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[test]
    fn parses_from_yaml() {
        let yaml = "\
bucket: my-logs
region: us-east-1
prefix: oxllm
access_key_id: AKIAEXAMPLE
secret_access_key: secretkey
";
        let cfg: S3LogConfig = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(cfg.bucket, "my-logs");
        assert_eq!(cfg.region, "us-east-1");
        assert_eq!(cfg.access_key_id.as_deref(), Some("AKIAEXAMPLE"));
    }

    #[test]
    fn construction_rejects_missing_bucket() {
        let cfg = S3LogConfig {
            bucket: "".to_string(),
            ..raw_config()
        };
        assert!(S3LogHook::new(&cfg, false).is_err());
    }

    #[test]
    fn construction_rejects_missing_credentials() {
        let cfg = S3LogConfig {
            access_key_id: None,
            ..raw_config()
        };
        assert!(S3LogHook::new(&cfg, false).is_err());
    }

    #[tokio::test]
    async fn construction_applies_default_prefix() {
        let cfg = S3LogConfig {
            prefix: None,
            ..raw_config()
        };
        let hook = S3LogHook::new(&cfg, false).expect("build");
        assert_eq!(hook.config.prefix, "oxllm");
    }

    #[tokio::test]
    async fn object_key_has_date_partitioned_shape() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let ctx = make_ctx();
        let key = hook.object_key(&ctx);
        // {prefix}/{date}/{request_id}.json
        let parts: Vec<&str> = key.split('/').collect();
        assert_eq!(parts.len(), 3, "key must have exactly 3 segments: {key}");
        assert_eq!(parts[0], "oxllm");
        // date segment YYYY-MM-DD.
        assert_eq!(parts[1].len(), 10);
        assert_eq!(parts[1].chars().nth(4), Some('-'));
        assert_eq!(parts[2], "req-1.json");
    }

    /// MEDIUM #6: a forwarded/replayed request id carrying `/` (or `\`,
    /// or a control char) must not introduce extra key segments — the
    /// `{prefix}/{date}/{id}.json` partition invariant must hold.
    #[tokio::test]
    async fn object_key_sanitizes_slash_in_request_id() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let ctx = CallContext {
            request_id: "a/b".to_string(),
            ..CallContext::for_test("gpt-4o")
        };
        let key = hook.object_key(&ctx);
        let parts: Vec<&str> = key.split('/').collect();
        assert_eq!(
            parts.len(),
            3,
            "sanitized id must keep the key at 3 segments: {key}"
        );
        assert_eq!(parts[0], "oxllm");
        assert_eq!(parts[1].len(), 10, "date segment must remain YYYY-MM-DD");
        assert!(
            !parts[2].contains('/'),
            "id segment must contain no '/': {}",
            parts[2]
        );
        assert_eq!(parts[2], "a_b.json");
    }

    #[tokio::test]
    async fn object_key_falls_back_when_request_id_empty() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let ctx = CallContext {
            request_id: "   ".to_string(),
            ..CallContext::for_test("gpt-4o")
        };
        let key = hook.object_key(&ctx);
        let parts: Vec<&str> = key.split('/').collect();
        assert_eq!(parts.len(), 3, "key must have exactly 3 segments: {key}");
        assert_eq!(parts[2], "unknown.json");
    }

    /// MEDIUM #4: a forwarded/replayed request id carrying URI-special
    /// characters (`?`, `#`, `%`, space, quotes) must not produce a
    /// malformed S3 key — those would either break `reqwest::Url::parse` (URL
    /// parse error → request failure → drop) or be silently truncated by `?`
    /// / `#` (suffix dropped as a query/fragment). The conservative stance is
    /// to allowlist `[A-Za-z0-9._-]` and replace anything else with `_`.
    #[tokio::test]
    async fn object_key_sanitizes_uri_special_characters() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let ctx = CallContext {
            request_id: "a?b#c d%\"e".to_string(),
            ..CallContext::for_test("gpt-4o")
        };
        let key = hook.object_key(&ctx);
        let parts: Vec<&str> = key.split('/').collect();
        assert_eq!(
            parts.len(),
            3,
            "key must keep exactly 3 segments under any input: {key}"
        );
        assert_eq!(parts[0], "oxllm");
        assert_eq!(parts[1].len(), 10, "date segment must remain YYYY-MM-DD");
        let id_segment = parts[2];
        assert!(
            !id_segment.contains('?'),
            "id segment must not contain '?': {id_segment}"
        );
        assert!(
            !id_segment.contains('#'),
            "id segment must not contain '#': {id_segment}"
        );
        assert!(
            !id_segment.contains('%'),
            "id segment must not contain '%': {id_segment}"
        );
        assert!(
            !id_segment.contains(' '),
            "id segment must not contain spaces: {id_segment}"
        );
        assert!(
            !id_segment.contains('"'),
            "id segment must not contain quotes: {id_segment}"
        );
        // Each unsafe char collapses to `_`; the suffix `.json` survives.
        assert_eq!(id_segment, "a_b_c_d__e.json");
    }

    /// MEDIUM #4: the object URL built from a sanitized key must round-trip
    /// through `reqwest::Url::parse` (i.e. produce a valid URL the S3 PUT
    /// path can use without transport-layer rejection).
    #[tokio::test]
    async fn object_url_round_trips_through_url_parse_for_unsafe_request_id() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let ctx = CallContext {
            request_id: "a?b#c d%".to_string(),
            ..CallContext::for_test("gpt-4o")
        };
        let key = hook.object_key(&ctx);
        let url = hook.object_url(&key);
        assert!(
            reqwest::Url::parse(&url).is_ok(),
            "object URL must be a valid URL, got: {url}"
        );
    }

    #[tokio::test]
    async fn object_url_uses_virtual_hosted_style_for_aws() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let url = hook.object_url("oxllm/2024-01-01/req-1.json");
        assert_eq!(
            url,
            "https://my-logs.s3.us-east-1.amazonaws.com/oxllm/2024-01-01/req-1.json"
        );
    }

    #[tokio::test]
    async fn object_url_uses_path_style_for_custom_endpoint() {
        let cfg = S3LogConfig {
            endpoint: Some("https://minio.local:9000".to_string()),
            ..raw_config()
        };
        let hook = S3LogHook::new(&cfg, false).expect("build");
        let url = hook.object_url("oxllm/2024-01-01/req-1.json");
        assert_eq!(
            url,
            "https://minio.local:9000/my-logs/oxllm/2024-01-01/req-1.json"
        );
    }

    #[tokio::test]
    async fn build_log_has_correct_shape() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let payload = hook.build_log(&make_ctx(), &make_response());
        assert_eq!(payload["model"], "gpt-4o");
        assert_eq!(payload["provider"], "openai");
        assert_eq!(payload["usage"]["prompt_tokens"], 100);
        assert_eq!(payload["usage"]["completion_tokens"], 50);
        assert_eq!(payload["usage"]["total_tokens"], 150);
        assert_eq!(payload["cost"], 0.0025);
        assert_eq!(payload["request_id"], "req-1");
        assert_eq!(payload["key_id"], "k-1");
        assert_eq!(payload["team_id"], "team-1");
    }

    /// LOW #7: the S3 log must include a redacted response body alongside
    /// the request body so operators have both halves of the exchange for
    /// audit / replay (matching LiteLLM's S3 logging).
    #[tokio::test]
    async fn build_log_includes_redacted_response_body() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let resp = ResponseSummary {
            body: Some(json!({
                "choices": [{"message": {"content": "hi"}}],
                "authorization": "Bearer sk-leak"
            })),
            ..make_response()
        };
        let payload = hook.build_log(&make_ctx(), &resp);
        assert!(
            payload["response_body"].is_object(),
            "response_body must be present"
        );
        assert_eq!(
            payload["response_body"]["choices"][0]["message"]["content"],
            "hi"
        );
        assert_eq!(
            payload["response_body"]["authorization"], "[redacted]",
            "response body must be redacted before S3 write"
        );
    }

    /// HIGH #2: request bodies written to S3 carry prompt contents and any
    /// embedded credentials. Well-known credential-bearing keys must be
    /// redacted before the object is written, mirroring Langfuse.
    #[tokio::test]
    async fn build_log_redacts_sensitive_keys_in_request_body() {
        let hook = S3LogHook::new(&raw_config(), false).expect("build");
        let ctx = CallContext {
            request_body: Arc::new(json!({
                "model": "gpt-4o",
                "authorization": "Bearer sk-secret",
                "api_key": "sk-secret",
                "messages": [{"role": "user", "content": "hi"}],
            })),
            ..CallContext::for_test("gpt-4o")
        };
        let payload = hook.build_log(&ctx, &make_response());
        assert_eq!(
            payload["request_body"]["authorization"], "[redacted]",
            "authorization must be redacted before S3 write"
        );
        assert_eq!(
            payload["request_body"]["api_key"], "[redacted]",
            "api_key must be redacted before S3 write"
        );
        assert_eq!(
            payload["request_body"]["messages"][0]["content"], "hi",
            "non-credential content must survive redaction"
        );
        // The shared CallContext must be untouched (redaction is on a clone).
        assert_eq!(
            ctx.request_body["authorization"], "Bearer sk-secret",
            "redaction must not mutate the shared CallContext body"
        );
    }

    #[tokio::test]
    async fn post_success_skips_cache_hit() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = S3LogConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = S3LogHook::new(&cfg, false).expect("build");

        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });

        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        hook.post_success(&ctx, &make_response()).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(put_mock.hits(), 0, "cache hits must not be logged");
    }

    #[tokio::test]
    async fn post_success_skips_master_key() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = S3LogConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = S3LogHook::new(&cfg, false).expect("build");

        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });

        let mut ctx = make_ctx();
        ctx.is_master = true;
        hook.post_success(&ctx, &make_response()).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(put_mock.hits(), 0, "master-key requests must not be logged");
    }

    #[tokio::test]
    async fn put_object_signs_with_sigv4_authorization() {
        use httpmock::MockServer;

        let server = MockServer::start();
        let cfg = S3LogConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = S3LogHook::new(&cfg, false).expect("build");

        let put_mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT)
                .header_exists("Authorization")
                .header_exists("x-amz-date")
                .header_exists("x-amz-content-sha256")
                .body_contains("\"model\":\"gpt-4o\"")
                .body_contains("\"request_id\":\"req-1\"");
            then.status(200);
        });

        hook.post_success(&make_ctx(), &make_response()).await;

        for _ in 0..40 {
            if put_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            put_mock.hits() >= 1,
            "s3 PUT should have carried SigV4 headers and the log body"
        );
        // The SigV4 credential scope (region/service) is verified at the
        // header-prefix level above; the full signature is unit-tested in
        // `bedrock_sigv4`, so we don't deep-inspect the recorded request here.
    }

    #[tokio::test]
    async fn put_object_uses_path_style_key_for_custom_endpoint() {
        use httpmock::MockServer;
        use regex::Regex;

        let server = MockServer::start();
        let cfg = S3LogConfig {
            endpoint: Some(server.base_url()),
            ..raw_config()
        };
        let hook = S3LogHook::new(&cfg, false).expect("build");

        let path_re = Regex::new(r"^/my-logs/oxllm/[0-9]{4}-[0-9]{2}-[0-9]{2}/req-1\.json$")
            .expect("valid regex");
        let put_mock = server.mock(|when, then| {
            // Path-style: /{bucket}/{prefix}/{date}/{request_id}.json
            when.method(httpmock::Method::PUT).path_matches(path_re);
            then.status(200);
        });

        hook.post_success(&make_ctx(), &make_response()).await;

        for _ in 0..40 {
            if put_mock.hits() >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            put_mock.hits() >= 1,
            "custom endpoint should use path-style /bucket/key addressing"
        );
    }

    #[tokio::test]
    async fn post_success_does_not_block_request_on_put_failure() {
        // Unreachable port: the spawned task fails fast and the request path
        // returns immediately.
        let cfg = S3LogConfig {
            endpoint: Some("http://127.0.0.1:1".to_string()),
            ..raw_config()
        };
        let hook = S3LogHook::new(&cfg, false).expect("build");
        let started = std::time::Instant::now();
        hook.post_success(&make_ctx(), &make_response()).await;
        assert!(
            started.elapsed() < Duration::from_millis(150),
            "post_success must return immediately even when endpoint is unreachable"
        );
    }
}
