use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug)]
struct BlobRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl BlobRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn is_signed(&self) -> bool {
        verify_shared_key_signature(&self.method, &self.path, &self.headers)
    }
}

/// Read the complete request, including replayed PUT bodies. The fixture has
/// a small cap and rejects incomplete requests rather than guessing at a match.
async fn read_request(socket: &mut tokio::net::TcpStream) -> std::io::Result<BlobRequest> {
    const MAX_REQUEST: usize = 64 * 1024;
    let invalid = || std::io::Error::other("invalid or oversized mock request");
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break end + 4;
        }
        let read = socket.read(&mut chunk).await?;
        if read == 0 || bytes.len() + read > MAX_REQUEST {
            return Err(invalid());
        }
        bytes.extend_from_slice(&chunk[..read]);
    };
    let head = std::str::from_utf8(&bytes[..header_end]).map_err(|_| invalid())?;
    let mut lines = head.split("\r\n");
    let mut first = lines.next().ok_or_else(invalid)?.split_whitespace();
    let method = first.next().ok_or_else(invalid)?.to_string();
    let path = first.next().ok_or_else(invalid)?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    let body_len = headers
        .iter()
        .find(|(key, _)| key == "content-length")
        .map(|(_, value)| value.parse::<usize>())
        .transpose()
        .map_err(|_| invalid())?
        .unwrap_or(0);
    if body_len > MAX_REQUEST.saturating_sub(header_end) {
        return Err(invalid());
    }
    while bytes.len() < header_end + body_len {
        let remaining = (header_end + body_len - bytes.len()).min(chunk.len());
        let read = socket.read(&mut chunk[..remaining]).await?;
        if read == 0 {
            return Err(invalid());
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    Ok(BlobRequest {
        method,
        path,
        headers,
        body: bytes[header_end..header_end + body_len].to_vec(),
    })
}

fn response(status: u16, body: &str, retry_after: Option<&str>) -> String {
    let retry_after = retry_after
        .map(|value| format!("Retry-After: {value}\r\n"))
        .unwrap_or_default();
    format!(
        "HTTP/1.1 {status} Mock\r\nContent-Length: {}\r\nConnection: close\r\n{retry_after}\r\n{body}",
        body.len()
    )
}

type MockServer = tokio::task::JoinHandle<std::io::Result<Vec<BlobRequest>>>;

/// Unlike a static mock, this exercises an actual failure followed by recovery.
/// A None reply disconnects after accepting the complete request.
async fn sequence(replies: Vec<Option<String>>) -> std::io::Result<(String, MockServer)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}", listener.local_addr()?);
    let server = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(2), async move {
            let mut requests = Vec::new();
            for reply in replies {
                let (mut socket, _) = listener.accept().await?;
                requests.push(read_request(&mut socket).await?);
                if let Some(reply) = reply {
                    socket.write_all(reply.as_bytes()).await?;
                    socket.shutdown().await?;
                }
            }
            Ok(requests)
        })
        .await
        .map_err(std::io::Error::other)?
    });
    Ok((endpoint, server))
}

#[tokio::test]
async fn get_recovers_from_503_and_429_with_signed_attempts() -> TestResult {
    let (endpoint, server) = sequence(vec![
        Some(response(503, "", None)),
        Some(response(429, "", Some("0"))),
        Some(response(200, "cached", None)),
    ])
    .await?;
    let cache =
        AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?.with_blob_endpoint(&endpoint);
    assert_eq!(cache.get("retry/é").await.as_deref(), Some("cached"));
    let requests = server.await??;
    assert_eq!(requests.len(), 3);
    for request in requests {
        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/data/oxllm%3Aretry/%C3%A9");
        assert_eq!(request.header("content-length"), Some("0"));
        assert!(request.body.is_empty());
        assert!(request.is_signed());
    }
    Ok(())
}

#[tokio::test]
async fn set_recovers_from_500_replaying_body_metadata_and_signature() -> TestResult {
    let (endpoint, server) = sequence(vec![
        Some(response(201, "", None)), // Container creation shares this operation.
        Some(response(500, "", None)),
        Some(response(201, "", None)),
    ])
    .await?;
    let cache =
        AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?.with_blob_endpoint(&endpoint);
    let value = "{\"value\":\"café\"}";
    cache
        .set("k", value.to_string(), Some(Duration::from_secs(60)))
        .await;
    let requests = server.await??;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].path, "/data?restype=container");
    let expiry = requests[1].header("x-ms-meta-expires");
    assert!(expiry.and_then(|value| value.parse::<u64>().ok()).is_some());
    let content_length = value.len().to_string();
    for request in &requests[1..] {
        assert_eq!(request.method, "PUT");
        assert_eq!(request.path, "/data/oxllm%3Ak");
        assert_eq!(request.body.as_slice(), value.as_bytes());
        assert_eq!(
            request.header("content-length"),
            Some(content_length.as_str())
        );
        assert_eq!(request.header("x-ms-blob-type"), Some("BlockBlob"));
        assert_eq!(request.header("x-ms-meta-expires"), expiry);
        assert!(request.is_signed());
    }
    Ok(())
}

#[tokio::test]
async fn delete_recovers_from_502_with_signed_attempts() -> TestResult {
    let (endpoint, server) = sequence(vec![
        Some(response(502, "", None)),
        Some(response(202, "", None)),
    ])
    .await?;
    let cache =
        AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?.with_blob_endpoint(&endpoint);
    cache.delete("k").await;
    let requests = server.await??;
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request.method, "DELETE");
        assert_eq!(request.path, "/data/oxllm%3Ak");
        assert_eq!(request.header("content-length"), Some("0"));
        assert!(request.is_signed());
    }
    Ok(())
}

#[tokio::test]
async fn transport_disconnect_is_retried() -> TestResult {
    let (endpoint, server) = sequence(vec![None, Some(response(200, "recovered", None))]).await?;
    let cache =
        AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?.with_blob_endpoint(&endpoint);
    assert_eq!(cache.get("k").await.as_deref(), Some("recovered"));
    let requests = server.await??;
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(BlobRequest::is_signed));
    Ok(())
}

#[tokio::test]
async fn repeated_transport_disconnects_are_bounded() -> TestResult {
    let (endpoint, server) = sequence(vec![None, None, None]).await?;
    let cache =
        AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?.with_blob_endpoint(&endpoint);
    assert!(cache.get("k").await.is_none());
    let requests = server.await??;
    assert_eq!(requests.len(), 3);
    assert!(requests.iter().all(BlobRequest::is_signed));
    Ok(())
}

#[tokio::test]
async fn request_builder_failure_does_not_consume_retry_backoff() -> TestResult {
    let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?
        .with_blob_endpoint("http://[invalid");
    let value = tokio::time::timeout(Duration::from_millis(200), cache.get("k")).await?;
    assert!(value.is_none());
    Ok(())
}

#[tokio::test]
async fn managed_identity_retry_keeps_bearer_headers_and_replayable_body() -> TestResult {
    let imds = httpmock::MockServer::start();
    let token = imds.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .header("Metadata", "true");
        then.status(200).json_body(serde_json::json!({
            "access_token": "retry-bearer", "expires_in": "3600",
        }));
    });
    let (endpoint, server) = sequence(vec![
        Some(response(201, "", None)),
        Some(response(503, "", None)),
        Some(response(201, "", None)),
    ])
    .await?;
    let cache = AzureBlobCache::new("myacct", "data", "oxllm", None)?
        .with_blob_endpoint(&endpoint)
        .with_metadata_url(&imds.url("/token"));
    cache
        .set("k", "body".to_string(), Some(Duration::from_secs(60)))
        .await;
    let requests = server.await??;
    assert_eq!(requests.len(), 3);
    token.assert_hits(1);
    for request in &requests {
        assert_eq!(request.header("authorization"), Some("Bearer retry-bearer"));
        assert_eq!(request.header("x-ms-version"), Some(MS_VERSION));
        assert!(request.header("x-ms-date").is_some());
    }
    let expiry = requests[1].header("x-ms-meta-expires");
    assert!(expiry.is_some());
    for request in &requests[1..] {
        assert_eq!(request.path, "/data/oxllm%3Ak");
        assert_eq!(request.body.as_slice(), b"body");
        assert_eq!(request.header("content-length"), Some("4"));
        assert_eq!(request.header("x-ms-blob-type"), Some("BlockBlob"));
        assert_eq!(request.header("x-ms-meta-expires"), expiry);
    }
    Ok(())
}

#[tokio::test]
async fn persistent_transient_blob_failures_have_only_three_attempts() -> TestResult {
    let server = httpmock::MockServer::start();
    let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?
        .with_blob_endpoint(&server.base_url());
    server.mock(|when, then| {
        when.method(httpmock::Method::PUT).path("/data");
        then.status(201);
    });
    let get = server.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/data/oxllm%3Ak");
        then.status(504);
    });
    let put = server.mock(|when, then| {
        when.method(httpmock::Method::PUT).path("/data/oxllm%3Ak");
        then.status(503);
    });
    let delete = server.mock(|when, then| {
        when.method(httpmock::Method::DELETE)
            .path("/data/oxllm%3Ak");
        then.status(429);
    });
    let (value, (), ()) = tokio::join!(
        cache.get("k"),
        cache.set("k", "v".to_string(), None),
        cache.delete("k"),
    );
    assert!(value.is_none());
    get.assert_hits(3);
    put.assert_hits(3);
    delete.assert_hits(3);
    Ok(())
}

#[tokio::test]
async fn deterministic_blob_4xx_do_not_retry_any_method() -> TestResult {
    let server = httpmock::MockServer::start();
    let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?
        .with_blob_endpoint(&server.base_url());
    server.mock(|when, then| {
        when.method(httpmock::Method::PUT).path("/data");
        then.status(201);
    });
    for status in [400, 401, 403, 404, 409, 412] {
        let key = status.to_string();
        let path = format!("/data/oxllm%3A{key}");
        let get = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path(&path);
            then.status(status).header("Retry-After", "0");
        });
        let put = server.mock(|when, then| {
            when.method(httpmock::Method::PUT).path(&path);
            then.status(status).header("Retry-After", "0");
        });
        let delete = server.mock(|when, then| {
            when.method(httpmock::Method::DELETE).path(&path);
            then.status(status).header("Retry-After", "0");
        });
        let (value, (), ()) = tokio::join!(
            cache.get(&key),
            cache.set(&key, "v".to_string(), None),
            cache.delete(&key),
        );
        assert!(value.is_none());
        get.assert_hits(1);
        put.assert_hits(1);
        delete.assert_hits(1);
    }
    Ok(())
}

#[tokio::test]
async fn long_retry_after_returns_a_fast_miss_instead_of_retrying_early() -> TestResult {
    for retry_after in [
        "1".to_string(),
        httpdate::fmt_http_date(SystemTime::now() + Duration::from_secs(60)),
    ] {
        let server = httpmock::MockServer::start();
        let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?
            .with_blob_endpoint(&server.base_url());
        let get = server.mock(|when, then| {
            when.method(httpmock::Method::GET);
            then.status(503).header("Retry-After", &retry_after);
        });
        let value = tokio::time::timeout(Duration::from_millis(200), cache.get("k")).await?;
        assert!(value.is_none());
        get.assert_hits(1);
    }
    Ok(())
}

#[test]
fn retry_after_http_date_is_a_minimum_within_the_remaining_deadline() -> TestResult {
    let second = UNIX_EPOCH + Duration::from_secs(1_000_000_000);
    let now = second + Duration::from_millis(750);
    let date = reqwest::header::HeaderValue::from_str(&httpdate::fmt_http_date(
        second + Duration::from_secs(1),
    ))?;
    assert_eq!(
        blob_retry_delay(0, Some(&date), Duration::from_millis(400), now),
        Some(Duration::from_millis(250))
    );
    assert_eq!(
        blob_retry_delay(0, Some(&date), Duration::from_millis(200), now),
        None
    );
    let invalid = reqwest::header::HeaderValue::from_static("not-a-date");
    assert_eq!(
        blob_retry_delay(0, Some(&invalid), CACHE_OPERATION_TIMEOUT, now),
        Some(Duration::from_millis(100))
    );
    Ok(())
}

#[tokio::test]
async fn set_retries_share_the_deadline_with_container_creation() -> TestResult {
    let server = httpmock::MockServer::start();
    let cache = AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?
        .with_blob_endpoint(&server.base_url());
    let container = server.mock(|when, then| {
        when.method(httpmock::Method::PUT).path("/data");
        then.status(201).delay(Duration::from_millis(300));
    });
    let put = server.mock(|when, then| {
        when.method(httpmock::Method::PUT).path("/data/oxllm%3Ak");
        then.status(503);
    });
    tokio::time::timeout(
        CACHE_OPERATION_TIMEOUT + Duration::from_millis(200),
        cache.set("k", "v".to_string(), None),
    )
    .await?;
    container.assert_hits(1);
    // 300ms container creation + 100ms first backoff leave less than the next
    // 200ms backoff. Skip that retry rather than starting a fresh send budget.
    put.assert_hits(2);
    Ok(())
}

#[tokio::test]
async fn get_deadline_includes_retry_backoff_and_a_stalled_body() -> TestResult {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}", listener.local_addr()?);
    let (body_sent, mut body_received) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await?;
        let _ = read_request(&mut first).await?;
        tokio::time::sleep(Duration::from_millis(200)).await;
        first.write_all(response(503, "", None).as_bytes()).await?;
        drop(first);
        let (mut second, _) = listener.accept().await?;
        let _ = read_request(&mut second).await?;
        second
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nx")
            .await?;
        let _ = body_sent.send(());
        std::future::pending::<std::io::Result<()>>().await
    });
    let cache =
        AzureBlobCache::new("myacct", "data", "oxllm", Some(CONN))?.with_blob_endpoint(&endpoint);
    let value = tokio::time::timeout(
        CACHE_OPERATION_TIMEOUT + Duration::from_millis(200),
        cache.get("k"),
    )
    .await;
    server.abort();
    assert!(
        body_received.try_recv().is_ok(),
        "the retry must reach the partial body"
    );
    assert!(
        matches!(value, Ok(None)),
        "body reads must retain the original deadline"
    );
    Ok(())
}
