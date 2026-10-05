//! HTTP client for the oxllm proxy management API.
//!
//! Every call carries `Authorization: Bearer {master_key}`. The browser never
//! sees the master key — it lives only in the admin server process.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// Connect deadline for proxy calls. A proxy that accepts the TCP connection
/// but never responds (GC pause, deadlock, …) would otherwise hang every admin
/// request indefinitely.
pub(crate) const PROXY_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Overall deadline for a single proxy call.
pub(crate) const PROXY_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// A management list is a bounded view, not necessarily the full collection.
/// Keep the server's truncation sentinel alongside the rows through rendering.
#[derive(Debug, Default)]
pub struct ListResponse {
    pub items: Vec<Value>,
    pub truncated: bool,
    /// Used only for optional dropdown feeds that failed to load.
    pub unavailable: bool,
}

impl ListResponse {
    pub(crate) fn unavailable() -> Self {
        Self {
            unavailable: true,
            ..Self::default()
        }
    }
}

/// Thin HTTP client over the proxy's master-key-only management endpoints.
pub struct ProxyClient {
    base: String,
    master_key: String,
    http: reqwest::Client,
}

impl ProxyClient {
    pub fn new(base: &str, master_key: &str) -> Self {
        let base = base.trim_end_matches('/').to_string();
        let http = reqwest::Client::builder()
            .connect_timeout(PROXY_CONNECT_TIMEOUT)
            .timeout(PROXY_REQUEST_TIMEOUT)
            .build()
            .expect("reqwest client with connect/request timeouts");
        Self {
            base,
            master_key: master_key.to_string(),
            http,
        }
    }

    /// Test-only constructor with an explicit request timeout, used to verify
    /// hung upstreams surface as errors within a bounded window.
    #[cfg(test)]
    pub(crate) fn with_request_timeout(base: &str, master_key: &str, timeout: Duration) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(PROXY_CONNECT_TIMEOUT)
            .timeout(timeout)
            .build()
            .expect("reqwest client with connect/request timeouts");
        Self {
            base: base.trim_end_matches('/').to_string(),
            master_key: master_key.to_string(),
            http,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    async fn get_json(&self, path: &str) -> Result<Value> {
        let resp = self
            .http
            .get(self.url(path))
            .bearer_auth(&self.master_key)
            .send()
            .await
            .context("proxy GET request failed")?;
        decode_json(resp).await
    }

    async fn get_json_with_query(&self, path: &str, query: &str) -> Result<Value> {
        let url = if query.is_empty() {
            self.url(path)
        } else {
            format!("{}?{}", self.url(path), query.trim_start_matches('?'))
        };
        let resp = self
            .http
            .get(url)
            .bearer_auth(&self.master_key)
            .send()
            .await
            .context("proxy GET request failed")?;
        decode_json(resp).await
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let resp = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.master_key)
            .json(body)
            .send()
            .await
            .context("proxy POST request failed")?;
        decode_json(resp).await
    }

    async fn post_json_discard_body(&self, path: &str, body: &Value) -> Result<()> {
        let resp = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.master_key)
            .json(body)
            .send()
            .await
            .context("proxy POST request failed")?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = read_capped_error_body(resp).await;
            bail!(ProxyStatusError {
                status,
                body: format!("{path}: {body}"),
            });
        }
        Ok(())
    }

    /// Proxy liveness probe (`GET /health`). Always 200 if the process is up.
    pub async fn health(&self) -> Result<Value> {
        let resp = self
            .http
            .get(self.url("/health"))
            .send()
            .await
            .context("proxy health request failed")?;
        decode_json(resp).await
    }

    /// Proxy readiness probe (`GET /health/readiness`). May be 503 if Redis/DB
    /// is down; returns body with per-check status when possible.
    ///
    /// A non-2xx status (e.g. a `503` from a broken backend, or a fronting
    /// proxy's HTML error page) is surfaced as `Err` so the dashboard can
    /// render a degraded state rather than masking it as healthy.
    pub async fn readiness(&self) -> Result<Value> {
        let resp = self
            .http
            .get(self.url("/health/readiness"))
            .send()
            .await
            .context("proxy readiness request failed")?;
        let status = resp.status();
        if !status.is_success() {
            bail!("proxy readiness returned status {}", status.as_u16());
        }
        resp.json::<Value>()
            .await
            .context("proxy readiness returned non-JSON body")
    }

    /// `GET /key/list`.
    pub async fn list_keys(&self) -> Result<ListResponse> {
        let value = self.get_json("/key/list").await?;
        Ok(extract_list(&value, "keys"))
    }

    /// `POST /key/generate`.
    pub async fn generate_key(&self, body: Value) -> Result<Value> {
        self.post_json("/key/generate", &body).await
    }

    /// `GET /key/info?key_id=...`.
    pub async fn key_info(&self, key_id: &str) -> Result<Value> {
        self.get_json_with_query(
            "/key/info",
            &format!("key_id={}", encode_query_value(key_id)),
        )
        .await
    }

    /// `POST /key/delete` (soft-delete).
    pub async fn delete_key(&self, key_id: &str) -> Result<()> {
        let body = serde_json::json!({ "key_id": key_id });
        self.post_json_discard_body("/key/delete", &body).await
    }

    /// `POST /key/purge` (hard-delete).
    pub async fn purge_key(&self, key_id: &str) -> Result<()> {
        let body = serde_json::json!({ "key_id": key_id });
        self.post_json_discard_body("/key/purge", &body).await
    }

    /// `POST /key/update`.
    pub async fn update_key(&self, body: Value) -> Result<Value> {
        self.post_json("/key/update", &body).await
    }

    /// Re-enable a soft-deleted key via `POST /key/update`.
    pub async fn enable_key(&self, key_id: &str) -> Result<Value> {
        let body = serde_json::json!({ "key_id": key_id, "is_active": true });
        self.post_json("/key/update", &body).await
    }

    /// `POST /key/rotate` — new secret, returned once.
    pub async fn rotate_key(&self, key_id: &str) -> Result<Value> {
        let body = serde_json::json!({ "key_id": key_id });
        self.post_json("/key/rotate", &body).await
    }

    /// `GET /team/list`.
    pub async fn list_teams(&self) -> Result<ListResponse> {
        let value = self.get_json("/team/list").await?;
        Ok(extract_list(&value, "teams"))
    }

    /// `POST /team/add`.
    pub async fn create_team(&self, body: Value) -> Result<Value> {
        self.post_json("/team/add", &body).await
    }

    /// `POST /team/update`.
    pub async fn update_team(&self, body: Value) -> Result<Value> {
        self.post_json("/team/update", &body).await
    }

    /// `POST /team/delete`.
    pub async fn delete_team(&self, team_id: &str) -> Result<()> {
        let body = serde_json::json!({ "team_id": team_id });
        self.post_json_discard_body("/team/delete", &body).await
    }

    /// `GET /user/list`.
    pub async fn list_users(&self) -> Result<ListResponse> {
        let value = self.get_json("/user/list").await?;
        Ok(extract_list(&value, "users"))
    }

    /// `POST /user/add`.
    pub async fn create_user(&self, body: Value) -> Result<Value> {
        self.post_json("/user/add", &body).await
    }

    /// `POST /user/update`.
    pub async fn update_user(&self, body: Value) -> Result<Value> {
        self.post_json("/user/update", &body).await
    }

    /// `POST /user/delete`.
    pub async fn delete_user(&self, user_id: &str) -> Result<()> {
        let body = serde_json::json!({ "user_id": user_id });
        self.post_json_discard_body("/user/delete", &body).await
    }

    /// `GET /organization/list`.
    pub async fn list_organizations(&self) -> Result<ListResponse> {
        let value = self.get_json("/organization/list").await?;
        Ok(extract_list(&value, "organizations"))
    }

    /// `POST /organization/add`.
    pub async fn create_organization(&self, body: Value) -> Result<Value> {
        self.post_json("/organization/add", &body).await
    }

    /// `POST /organization/update`.
    pub async fn update_organization(&self, body: Value) -> Result<Value> {
        self.post_json("/organization/update", &body).await
    }

    /// `POST /organization/delete`.
    pub async fn delete_organization(&self, org_id: &str) -> Result<()> {
        let body = serde_json::json!({ "org_id": org_id });
        self.post_json_discard_body("/organization/delete", &body)
            .await
    }

    /// `GET /project/list`.
    pub async fn list_projects(&self) -> Result<ListResponse> {
        let value = self.get_json("/project/list").await?;
        Ok(extract_list(&value, "projects"))
    }

    /// `POST /project/add`.
    pub async fn create_project(&self, body: Value) -> Result<Value> {
        self.post_json("/project/add", &body).await
    }

    /// `POST /project/update`.
    pub async fn update_project(&self, body: Value) -> Result<Value> {
        self.post_json("/project/update", &body).await
    }

    /// `POST /project/delete`.
    pub async fn delete_project(&self, project_id: &str) -> Result<()> {
        let body = serde_json::json!({ "project_id": project_id });
        self.post_json_discard_body("/project/delete", &body).await
    }

    /// `GET /spend/logs/v2?<params>` — raw query string.
    pub async fn spend_logs(&self, params: &str) -> Result<Value> {
        self.get_json_with_query("/spend/logs/v2", params).await
    }

    /// `POST /spend/calculate`.
    pub async fn spend_calculate(&self, body: Value) -> Result<Value> {
        self.post_json("/spend/calculate", &body).await
    }

    /// `GET /spend/total`.
    pub async fn spend_total(&self, params: &str) -> Result<Value> {
        self.get_json_with_query("/spend/total", params).await
    }

    /// `GET /global/spend`.
    pub async fn global_spend(&self) -> Result<Value> {
        self.get_json("/global/spend").await
    }

    /// `GET /global/activity?days=<days>`.
    pub async fn global_activity(&self, days: u32) -> Result<Value> {
        self.get_json_with_query("/global/activity", &format!("days={days}"))
            .await
    }

    /// `GET /model/list`.
    pub async fn list_models(&self) -> Result<ListResponse> {
        let value = self.get_json("/model/list").await?;
        let data = extract_list(&value, "data");
        if data.items.is_empty() {
            Ok(extract_list(&value, "models"))
        } else {
            Ok(data)
        }
    }

    /// `GET /model/info?model=...`.
    pub async fn model_info(&self, model: &str) -> Result<Value> {
        self.get_json_with_query(
            "/model/info",
            &format!("model={}", encode_query_value(model)),
        )
        .await
    }

    /// `POST /model/new` — register a new deployment in the model list.
    /// `body` should contain `model_name` and `oxllm_params.model` at a
    /// minimum; other `oxllm_params.*` fields (e.g. `api_key`, `api_base`)
    /// are passed through verbatim.
    pub async fn create_model(&self, body: Value) -> Result<Value> {
        self.post_json("/model/new", &body).await
    }

    /// `POST /model/delete` — remove a DB-stored deployment by its `model_id`
    /// (the `models` table primary key surfaced by `/model/list` for
    /// DB-backed rows, NOT the client-facing `model_name`; the proxy's
    /// `DeleteModelRequest` rejects anything else with 422/404). The proxy
    /// soft-stops routing to the deleted deployment; the change is persistent
    /// when a database is configured.
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        let body = serde_json::json!({ "model_id": model_id });
        self.post_json_discard_body("/model/delete", &body).await
    }
}

/// Maximum bytes of an upstream error body to read into memory for the
/// diagnostic message. Keeps a misbehaving reverse proxy from buffering an
/// unbounded HTML error page into admin-server memory on every failing call.
const ERROR_BODY_CAP: u64 = 8 * 1024;

/// Maximum bytes of an upstream *success* body to buffer before deserialising.
/// The error path is bounded by `ERROR_BODY_CAP`; the success path previously
/// trusted the proxy to send a bounded payload. A misbehaving or compromised
/// proxy (or a `/key/list` over millions of keys) could drive unbounded
/// allocation in the admin process — the 15 s request timeout bounds the
/// transfer window, not the byte count. 64 MiB is well above any legitimate
/// paginated response (`limit=200`) while still rejecting runaway bodies.
const SUCCESS_BODY_CAP: u64 = 64 * 1024 * 1024;

/// Typed marker for a non-2xx proxy response, carried inside the `anyhow`
/// chain so callers (e.g. `app::is_proxy_unauthorized`) can match on the
/// numeric status code instead of grepping the error string — the wording of
/// `bail!` messages is fragile and a 5xx body that happens to contain the
/// substring `"status 401"` would otherwise trigger a spurious login redirect.
#[derive(Debug, thiserror::Error)]
#[error("proxy returned status {status}: {body}")]
pub(crate) struct ProxyStatusError {
    pub(crate) status: u16,
    #[allow(dead_code)]
    pub(crate) body: String,
}

async fn decode_json(resp: reqwest::Response) -> Result<Value> {
    let status = resp.status();
    if !status.is_success() {
        let body = read_capped_error_body(resp).await;
        bail!(ProxyStatusError {
            status: status.as_u16(),
            body,
        });
    }
    let bytes = read_capped_success_body(resp).await?;
    serde_json::from_slice(&bytes).context("proxy returned non-JSON body")
}

/// Read at most `SUCCESS_BODY_CAP` bytes of a 2xx response body, then stop and
/// abort the stream — so a runaway success body (e.g. a `/key/list` over
/// millions of keys, or a compromised proxy) surfaces as an error rather than
/// an unbounded allocation. Mirrors the bounded-read pattern of
/// [`read_capped_error_body`] but returns the raw bytes for JSON deserialisation
/// instead of a truncated diagnostic string.
async fn read_capped_success_body(resp: reqwest::Response) -> Result<Vec<u8>> {
    use futures_util::StreamExt;
    let cap = SUCCESS_BODY_CAP as usize;
    let limit = cap + 1;
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while buf.len() < limit {
        match stream.next().await {
            None => break,
            Some(Ok(bytes)) => {
                let remaining = limit - buf.len();
                if bytes.len() <= remaining {
                    buf.extend_from_slice(&bytes);
                } else {
                    buf.extend_from_slice(&bytes[..remaining]);
                    break;
                }
            }
            Some(Err(e)) => {
                bail!("proxy response body stream error: {e}");
            }
        }
    }
    if buf.len() > cap {
        bail!("proxy success response body exceeds {SUCCESS_BODY_CAP} bytes");
    }
    Ok(buf)
}

/// Read at most `ERROR_BODY_CAP` bytes of an upstream error body, then stop
/// and abort the stream — so a misbehaving reverse proxy serving an unbounded
/// HTML error page on every failing admin call cannot exhaust admin-server
/// memory. `reqwest::Response::bytes()` would materialise the whole body
/// before we could slice it; streaming with `bytes_stream()` lets us drop the
/// connection (and the response future) the moment we cross the cap.
async fn read_capped_error_body(resp: reqwest::Response) -> String {
    use futures_util::StreamExt;
    let cap = ERROR_BODY_CAP as usize;
    // Read up to `cap + 1` bytes: the extra byte is a sentinel so we can
    // distinguish "stream ended at exactly cap" (fits, no truncation) from
    // "stream had more than cap" (must truncate). Without it, a body whose
    // chunks align exactly with `cap` would slip through unmarked.
    let limit = cap + 1;
    let mut buf: Vec<u8> = Vec::with_capacity(cap.min(8 * 1024));
    let mut stream = resp.bytes_stream();
    let mut truncated = false;
    while buf.len() < limit {
        match stream.next().await {
            None => break,
            Some(Ok(bytes)) => {
                let remaining = limit - buf.len();
                if bytes.len() <= remaining {
                    buf.extend_from_slice(&bytes);
                } else {
                    buf.extend_from_slice(&bytes[..remaining]);
                    break;
                }
            }
            Some(Err(_)) => {
                // Best-effort: surface whatever we managed to read before the
                // stream broke. Treat it as truncated since more may exist.
                truncated = !buf.is_empty();
                break;
            }
        }
    }
    if buf.len() > cap {
        buf.truncate(cap);
        truncated = true;
    }
    let mut body = String::from_utf8_lossy(&buf).into_owned();
    if truncated {
        body.push_str("…[truncated]");
    }
    body
}

/// Accept legacy bare arrays as complete lists. In a wrapped response only the
/// explicit flag tells us that more records exist: exactly 1000 rows may still
/// be complete, and a shorter response may have a different upstream cap.
fn extract_list(value: &Value, key: &str) -> ListResponse {
    let items = match value {
        Value::Array(arr) => arr.clone(),
        Value::Object(map) => map
            .get(key)
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    ListResponse {
        items,
        truncated: value
            .get("truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        unavailable: false,
    }
}

/// Minimal percent-encoding for a query parameter value. Escapes everything
/// outside the RFC 3986 unreserved set (`A-Za-z0-9-_.~`) so key/model ids
/// containing `/`, `+`, `=`, or spaces reach the proxy unmodified.
///
/// Thin wrapper over [`crate::urlencode::urlencoding_minimal`] — both the
/// spend handlers and the proxy client share one implementation so a future
/// fix (handling `%` itself, multi-byte UTF-8, etc.) only has to land once.
fn encode_query_value(s: &str) -> String {
    crate::urlencode::urlencoding_minimal(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::MockServer;
    use serde_json::json;

    fn client(server: &MockServer) -> ProxyClient {
        ProxyClient::new(&server.base_url(), "sk-test-master")
    }

    #[test]
    fn list_metadata_is_explicit_not_inferred_from_the_row_count() {
        let rows = vec![json!({"key_id": "key"}); 1000];
        for value in [json!(rows), json!({"keys": rows, "truncated": false})] {
            let list = extract_list(&value, "keys");
            assert_eq!(list.items.len(), 1000);
            assert!(!list.truncated, "exactly 1000 rows can be complete");
            assert!(!list.unavailable);
        }
        let list = extract_list(
            &json!({"keys": [{"key_id": "key"}], "truncated": true}),
            "keys",
        );
        assert_eq!(list.items.len(), 1);
        assert!(list.truncated, "a proxy may apply a smaller cap");
        assert!(!list.unavailable);
    }

    #[tokio::test]
    async fn list_keys_hits_correct_path_with_bearer() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/key/list")
                .header("authorization", "Bearer sk-test-master");
            then.status(200).body(r#"{"keys": [{"key_id": "key-1"}]}"#);
        });
        let keys = client(&server).list_keys().await.expect("ok");
        m.assert();
        assert_eq!(keys.items.len(), 1);
        assert_eq!(keys.items[0]["key_id"], json!("key-1"));
    }

    #[tokio::test]
    async fn generate_key_posts_body() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/key/generate")
                .header("authorization", "Bearer sk-test-master")
                .json_body_partial(r#"{"key_alias":"a","max_budget":5.0}"#);
            then.status(200)
                .body(r#"{"key":"sk-ox-1","key_id":"key-1"}"#);
        });
        let resp = client(&server)
            .generate_key(json!({"key_alias": "a", "max_budget": 5.0}))
            .await
            .expect("ok");
        m.assert();
        assert_eq!(resp["key_id"], json!("key-1"));
    }

    #[tokio::test]
    async fn key_info_uses_query_param() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/key/info")
                .query_param("key_id", "key-42");
            then.status(200).body(r#"{"key_id":"key-42"}"#);
        });
        let resp = client(&server).key_info("key-42").await.expect("ok");
        m.assert();
        assert_eq!(resp["key_id"], json!("key-42"));
    }

    #[tokio::test]
    async fn delete_key_posts_key_id_body() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/key/delete")
                .json_body_partial(r#"{"key_id":"key-9"}"#);
            then.status(200).body(r#"{"status":"deleted"}"#);
        });
        client(&server).delete_key("key-9").await.expect("ok");
        m.assert();
    }

    #[tokio::test]
    async fn error_status_is_propagated() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/key/list");
            then.status(401).body(r#"{"error":"unauthorized"}"#);
        });
        let err = client(&server).list_keys().await.expect_err("should err");
        assert!(err.to_string().contains("401"));
    }

    #[tokio::test]
    async fn list_models_extracts_data_array() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/model/list");
            then.status(200).body(r#"{"data":[{"id":"gpt-4o"}]}"#);
        });
        let models = client(&server).list_models().await.expect("ok");
        m.assert();
        assert_eq!(models.items.len(), 1);
        assert_eq!(models.items[0]["id"], json!("gpt-4o"));
    }

    #[tokio::test]
    async fn key_info_percent_encodes_id() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/key/info")
                .query_param("key_id", "sk/space+test");
            then.status(200).body(r#"{"key_id":"sk/space+test"}"#);
        });
        let resp = client(&server).key_info("sk/space+test").await.expect("ok");
        m.assert();
        assert_eq!(resp["key_id"], json!("sk/space+test"));
    }

    #[test]
    fn encode_query_value_leaves_unreserved_alone() {
        assert_eq!(encode_query_value("key-123_abc.4o"), "key-123_abc.4o");
    }

    #[test]
    fn encode_query_value_escapes_special_bytes() {
        // `/` -> %2F, ` ` -> %20, `+` -> %2B, `=` -> %3D.
        assert_eq!(encode_query_value("a/b c+d=e"), "a%2Fb%20c%2Bd%3De");
    }

    #[tokio::test]
    async fn list_organizations_hits_correct_path() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/organization/list")
                .header("authorization", "Bearer sk-test-master");
            then.status(200)
                .body(r#"{"organizations":[{"org_id":"org-1"}]}"#);
        });
        let orgs = client(&server).list_organizations().await.expect("ok");
        m.assert();
        assert_eq!(orgs.items.len(), 1);
        assert_eq!(orgs.items[0]["org_id"], json!("org-1"));
    }

    #[tokio::test]
    async fn create_organization_posts_body() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/organization/add")
                .header("authorization", "Bearer sk-test-master")
                .json_body_partial(r#"{"org_name":"Acme"}"#);
            then.status(200).body(r#"{"org_id":"org-1"}"#);
        });
        let resp = client(&server)
            .create_organization(json!({"org_name": "Acme"}))
            .await
            .expect("ok");
        m.assert();
        assert_eq!(resp["org_id"], json!("org-1"));
    }

    #[tokio::test]
    async fn list_projects_hits_correct_path() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::GET)
                .path("/project/list")
                .header("authorization", "Bearer sk-test-master");
            then.status(200)
                .body(r#"{"projects":[{"project_id":"project-1"}]}"#);
        });
        let projects = client(&server).list_projects().await.expect("ok");
        m.assert();
        assert_eq!(projects.items.len(), 1);
        assert_eq!(projects.items[0]["project_id"], json!("project-1"));
    }

    #[tokio::test]
    async fn delete_model_posts_model_id_body() {
        // The proxy's DeleteModelRequest requires `model_id` (the DB primary
        // key); the historical `{"id": ...}` body failed with 422 on every
        // model delete from the UI.
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/model/delete")
                .json_body_partial(r#"{"model_id":"model-9"}"#);
            then.status(200)
                .body(r#"{"model_id":"model-9","status":"deleted"}"#);
        });
        client(&server).delete_model("model-9").await.expect("ok");
        m.assert();
    }

    #[tokio::test]
    async fn delete_project_posts_body() {
        let server = MockServer::start();
        let m = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/project/delete")
                .json_body_partial(r#"{"project_id":"project-9"}"#);
            then.status(200).body(r#"{"status":"deleted"}"#);
        });
        client(&server)
            .delete_project("project-9")
            .await
            .expect("ok");
        m.assert();
    }

    #[tokio::test]
    async fn hung_upstream_surfaces_as_error_within_timeout() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/key/list");
            then.status(200)
                .delay(Duration::from_secs(5))
                .body(r#"{"keys":[]}"#);
        });
        let c = ProxyClient::with_request_timeout(
            &server.base_url(),
            "sk-test",
            Duration::from_millis(200),
        );
        let start = std::time::Instant::now();
        let res = c.list_keys().await;
        let elapsed = start.elapsed();
        assert!(res.is_err(), "expected a timeout error, got {res:?}");
        assert!(
            elapsed < Duration::from_secs(2),
            "client did not bound the hung request: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn readiness_non_success_status_is_an_error() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/health/readiness");
            then.status(503).body(r#"{"status":"unhealthy"}"#);
        });
        let res = client(&server).readiness().await;
        let err = res.expect_err("503 readiness should be an error");
        assert!(err.to_string().contains("503"), "{}", err);
    }

    #[tokio::test]
    async fn error_body_is_truncated_when_above_cap() {
        // A misbehaving upstream returning many KiB of HTML must not be fully
        // buffered; the surfaced error should be truncated.
        let server = MockServer::start();
        let big_body = "X".repeat((ERROR_BODY_CAP as usize) * 4);
        server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/key/list");
            then.status(500).body(big_body);
        });
        let err = client(&server).list_keys().await.expect_err("500 -> Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("500"), "missing status in {msg}");
        assert!(
            msg.contains("[truncated]"),
            "missing truncation marker in error message (len {}): {msg}",
            msg.len()
        );
        assert!(
            !msg.contains(&"X".repeat((ERROR_BODY_CAP as usize) * 2)),
            "error message retained an unbounded slice of the body"
        );
    }

    #[tokio::test]
    async fn error_body_small_enough_passes_through_verbatim() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/key/list");
            then.status(401).body(r#"{"error":"unauthorized"}"#);
        });
        let err = client(&server).list_keys().await.expect_err("401 -> Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("unauthorized"));
        assert!(!msg.contains("[truncated]"));
    }

    /// Verifies the bounded-read fix in `read_capped_error_body`: a 1 MiB body
    /// trickled at ~20 ms / 8 KiB would take ~2,600 s to drain fully. The
    /// streaming reader must abort at `ERROR_BODY_CAP` and surface a truncated
    /// error in well under a second, with the upstream connection dropped so
    /// the remaining bytes are never materialised.
    #[tokio::test]
    async fn decode_json_does_not_buffer_unbounded_error_body() {
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body_size: usize = 1024 * 1024;
        let chunk = vec![b'X'; 8192];

        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut req_buf = [0u8; 4096];
            loop {
                let n = sock.read(&mut req_buf).await.unwrap();
                if n == 0 {
                    return;
                }
                if req_buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let header = format!(
                "HTTP/1.1 500 Internal Server Error\r\nContent-Length: {body_size}\r\nConnection: close\r\n\r\n"
            );
            sock.write_all(header.as_bytes()).await.unwrap();
            for _ in 0..(body_size / chunk.len()) {
                if sock.write_all(&chunk).await.is_err() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });

        let url = format!("http://{addr}/");
        let c = ProxyClient::with_request_timeout(&url, "sk-test", Duration::from_secs(30));
        let start = std::time::Instant::now();
        let err = c.list_keys().await.expect_err("500 -> Err");
        let elapsed = start.elapsed();
        let msg = format!("{err:#}");
        assert!(msg.contains("500"), "missing status in {msg}");
        assert!(
            msg.contains("[truncated]"),
            "expected truncation marker, got: {msg}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "decode_json took {elapsed:?} — looks like the body was buffered unbounded"
        );
    }

    /// A 2xx success body that exceeds `SUCCESS_BODY_CAP` must be rejected as
    /// an error instead of being buffered into memory unbounded. Uses a raw
    /// TCP listener so we can stream a body larger than the cap without
    /// materialising it all at once on the server side.
    #[tokio::test]
    async fn decode_json_rejects_oversized_success_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let chunk = vec![b'X'; 8192];
        // Body must exceed the cap and be an exact multiple of the chunk size
        // so Content-Length matches the bytes actually sent (a mismatch makes
        // reqwest report a decode error instead of our "exceeds" message).
        let body_size: usize = (SUCCESS_BODY_CAP as usize) + chunk.len();

        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut req_buf = [0u8; 4096];
            loop {
                let n = sock.read(&mut req_buf).await.unwrap();
                if n == 0 {
                    return;
                }
                if req_buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {body_size}\r\nConnection: close\r\n\r\n"
            );
            sock.write_all(header.as_bytes()).await.unwrap();
            for _ in 0..(body_size / chunk.len()) {
                if sock.write_all(&chunk).await.is_err() {
                    return;
                }
            }
        });

        let url = format!("http://{addr}/");
        let c = ProxyClient::with_request_timeout(&url, "sk-test", Duration::from_secs(60));
        let err = c.list_keys().await.expect_err("oversized 2xx -> Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("exceeds"),
            "expected 'exceeds' in error message, got: {msg}"
        );
    }
}
