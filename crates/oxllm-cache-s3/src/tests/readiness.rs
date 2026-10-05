use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const ACCESS_KEY: &str = "AKIDEXAMPLE";
const SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
const PROBE_PATH: &str = "/gateway/mybucket/oxllm%3A__oxllm_ping__";

#[derive(Debug)]
struct ProbeRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl ProbeRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn is_signed(&self) -> bool {
        let Some(date) = self.header("x-amz-date") else {
            return false;
        };
        let Some(payload_hash) = self.header("x-amz-content-sha256") else {
            return false;
        };
        if payload_hash != hex::encode(sha256(&self.body)) {
            return false;
        }
        let names = [
            "host",
            "x-amz-date",
            "x-amz-content-sha256",
            "x-amz-security-token",
        ];
        let Some(headers) = names
            .iter()
            .map(|name| {
                self.header(name)
                    .map(|value| (name.to_string(), value.to_string()))
            })
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        sign(
            &self.method,
            &self.path,
            "",
            &headers,
            payload_hash,
            date,
            ACCESS_KEY,
            SECRET_KEY,
            "us-east-1",
        )
        .is_ok_and(|expected| self.header("authorization") == Some(expected.as_str()))
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> std::io::Result<ProbeRequest> {
    const MAX_REQUEST: usize = 16 * 1024;
    let invalid = || std::io::Error::other("invalid or oversized probe request");
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
    Ok(ProbeRequest {
        method,
        path,
        headers,
        body: bytes[header_end..header_end + body_len].to_vec(),
    })
}

enum Reply {
    Status(u16),
    Delayed(u16, Duration),
}

use Reply::{Delayed, Status};

type MockServer = tokio::task::JoinHandle<std::io::Result<Vec<ProbeRequest>>>;

/// A stateful server models the marker becoming readable only after its PUT.
/// Every request is recorded so bucket-level probes or extra writes fail tests.
async fn sequence(replies: Vec<Reply>) -> std::io::Result<(String, MockServer)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let endpoint = format!("http://{}", listener.local_addr()?);
    let server = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(2), async move {
            let mut requests = Vec::new();
            for reply in replies {
                let (mut socket, _) = listener.accept().await?;
                requests.push(read_request(&mut socket).await?);
                let (status, delay) = match reply {
                    Status(status) => (status, Duration::ZERO),
                    Delayed(status, delay) => (status, delay),
                };
                tokio::time::sleep(delay).await;
                let response = format!(
                    "HTTP/1.1 {status} Mock\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                // Deadline regressions intentionally close a connection while
                // the server is delayed; retain the observed request either way.
                if socket.write_all(response.as_bytes()).await.is_err() {
                    break;
                }
            }
            Ok(requests)
        })
        .await
        .map_err(std::io::Error::other)?
    });
    Ok((endpoint, server))
}

fn cache(endpoint: &str) -> Result<S3Cache, Error> {
    S3Cache::new(
        "mybucket",
        "us-east-1",
        "oxllm",
        ACCESS_KEY,
        SECRET_KEY,
        Some("probe-session".to_string()),
        Some(format!("{endpoint}/gateway")),
    )
}

fn verify_requests(requests: &[ProbeRequest], methods: &[&str]) {
    assert_eq!(requests.len(), methods.len());
    for (request, method) in requests.iter().zip(methods) {
        assert_eq!(request.method, *method);
        assert_eq!(request.path, PROBE_PATH);
        assert_eq!(
            request.header("x-amz-security-token"),
            Some("probe-session")
        );
        assert!(
            request.is_signed(),
            "each probe step must carry a valid SigV4 signature"
        );
        if *method == "PUT" {
            assert_eq!(request.body.as_slice(), PING_BODY);
        } else {
            assert!(request.body.is_empty());
        }
    }
}

#[tokio::test]
async fn missing_object_becomes_healthy_without_rewriting_each_ping() -> TestResult {
    let (endpoint, server) =
        sequence(vec![Status(404), Status(200), Status(200), Status(200)]).await?;
    let cache = cache(&endpoint)?;
    assert!(
        cache.ping().await,
        "a missing probe object can be initialized"
    );
    assert!(
        cache.ping().await,
        "the existing marker must remain readable"
    );
    let requests = server.await??;
    verify_requests(&requests, &["HEAD", "PUT", "HEAD", "HEAD"]);
    Ok(())
}

#[tokio::test]
async fn object_only_credentials_work_when_missing_key_head_returns_403() -> TestResult {
    // AWS returns this 403 even with GetObject/PutObject when ListBucket is
    // absent. The only permission proof is to create and then read the marker.
    let (endpoint, server) = sequence(vec![Status(403), Status(200), Status(200)]).await?;
    let cache = cache(&endpoint)?;
    assert!(cache.ping().await);
    let requests = server.await??;
    verify_requests(&requests, &["HEAD", "PUT", "HEAD"]);
    Ok(())
}

#[tokio::test]
async fn readable_preexisting_marker_needs_no_put_or_bucket_permission() -> TestResult {
    let (endpoint, server) = sequence(vec![Status(200)]).await?;
    let cache = cache(&endpoint)?;
    assert!(cache.ping().await);
    let requests = server.await??;
    verify_requests(&requests, &["HEAD"]);
    Ok(())
}

#[tokio::test]
async fn write_only_credentials_are_unhealthy_after_successful_marker_put() -> TestResult {
    let (endpoint, server) = sequence(vec![Status(403), Status(200), Status(403)]).await?;
    let cache = cache(&endpoint)?;
    assert!(
        !cache.ping().await,
        "a successful write must not hide denied reads"
    );
    let requests = server.await??;
    verify_requests(&requests, &["HEAD", "PUT", "HEAD"]);
    Ok(())
}

#[tokio::test]
async fn marker_still_missing_after_put_is_unhealthy() -> TestResult {
    let (endpoint, server) = sequence(vec![Status(404), Status(200), Status(404)]).await?;
    let cache = cache(&endpoint)?;
    assert!(
        !cache.ping().await,
        "the verification HEAD cannot accept another 404"
    );
    let requests = server.await??;
    verify_requests(&requests, &["HEAD", "PUT", "HEAD"]);
    Ok(())
}

#[tokio::test]
async fn read_only_credentials_cannot_initialize_an_absent_marker() -> TestResult {
    let (endpoint, server) = sequence(vec![Status(404), Status(403)]).await?;
    let cache = cache(&endpoint)?;
    assert!(!cache.ping().await);
    let requests = server.await??;
    verify_requests(&requests, &["HEAD", "PUT"]);
    Ok(())
}

#[tokio::test]
async fn initialization_retries_and_verification_share_one_total_deadline() -> TestResult {
    let (endpoint, server) = sequence(vec![
        Status(404),
        Delayed(503, Duration::from_millis(150)),
        Status(200),
        Delayed(200, Duration::from_millis(300)),
    ])
    .await?;
    let cache = cache(&endpoint)?;
    let value = tokio::time::timeout(
        CACHE_OPERATION_TIMEOUT + Duration::from_millis(200),
        cache.ping(),
    )
    .await;
    assert!(
        matches!(value, Ok(false)),
        "initialization must not reset the 500ms budget"
    );
    let requests = server.await??;
    verify_requests(&requests, &["HEAD", "PUT", "PUT", "HEAD"]);
    Ok(())
}
