//! Connect using the addresses that passed SSRF policy, without a second DNS
//! lookup. The original request URI is retained for HTTP Host and TLS/SNI.

use std::net::SocketAddr;

use oxllm_core::Error;
use tokio_tungstenite::tungstenite::handshake::client::Request;

use super::UpstreamStream;

/// Preserve HTTP handshake statuses and the local timeout's 504 policy bucket.
/// The core error still distinguishes a provider response from a transport
/// failure for key-scope RPM settlement.
#[derive(Debug)]
pub(super) struct ConnectError {
    pub(super) error: Error,
    pub(super) failure_status: Option<u16>,
}

impl From<Error> for ConnectError {
    fn from(error: Error) -> Self {
        let failure_status = match &error {
            Error::Network(_) => Some(0),
            Error::Upstream { status, .. } => Some(*status),
            _ => None,
        };
        Self {
            error,
            failure_status,
        }
    }
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, f)
    }
}

pub(super) async fn connect(
    request: Request,
    allow_private: bool,
) -> Result<UpstreamStream, ConnectError> {
    match tokio::time::timeout(
        super::UPSTREAM_CONNECT_TIMEOUT,
        connect_inner(request, allow_private),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(ConnectError {
            error: Error::Network("realtime upstream WebSocket connect timed out".into()),
            failure_status: Some(504),
        }),
    }
}

async fn connect_inner(
    request: Request,
    allow_private: bool,
) -> Result<UpstreamStream, ConnectError> {
    let url = url::Url::parse(&request.uri().to_string())
        .map_err(|_| Error::Config("invalid realtime upstream URL".into()))?;
    if !matches!(url.scheme(), "ws" | "wss") {
        return Err(Error::Config("realtime upstream requires ws or wss".into()).into());
    }
    let host = url
        .host_str()
        .ok_or_else(|| Error::Config("realtime upstream URL has no host".into()))?
        .trim_start_matches('[')
        .trim_end_matches(']');
    let port = url
        .port_or_known_default()
        .ok_or_else(|| Error::Config("realtime upstream URL has no port".into()))?;
    let addresses: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|_| Error::Network("realtime upstream DNS lookup failed".into()))?
        .collect();
    connect_addresses(request, addresses, allow_private).await
}

async fn connect_addresses(
    request: Request,
    addresses: Vec<SocketAddr>,
    allow_private: bool,
) -> Result<UpstreamStream, ConnectError> {
    let addresses: Vec<_> = addresses
        .into_iter()
        .filter(|address| allow_private || crate::security::check_ip_allowed(address.ip()))
        .collect();
    if addresses.is_empty() {
        return Err(Error::Network(
            "realtime upstream has no allowed addresses (SSRF policy)".into(),
        )
        .into());
    }
    // TcpStream receives SocketAddrs, never the hostname: DNS cannot change
    // the destination between validation and dialing. Tungstenite derives the
    // TLS server name from request.uri(), not from this socket's IP address.
    let stream = tokio::net::TcpStream::connect(addresses.as_slice())
        .await
        .map_err(|_| Error::Network("realtime upstream TCP connect failed".into()))?;
    tokio_tungstenite::client_async_tls_with_config(
        request,
        stream,
        Some(super::upstream_ws_config()),
        None,
    )
    .await
    .map(|(socket, _)| socket)
    .map_err(|error| match error {
        tokio_tungstenite::tungstenite::Error::Http(response) => Error::Upstream {
            status: response.status().as_u16(),
            message: "realtime upstream rejected the WebSocket handshake".into(),
        }
        .into(),
        _ => Error::Network("realtime upstream WebSocket handshake failed".into()).into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    #[tokio::test]
    async fn rejects_private_resolution_before_dialing() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let request = "ws://public-name.invalid/realtime"
            .into_client_request()
            .unwrap();
        let error = connect_addresses(request, vec![listener.local_addr().unwrap()], false)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("SSRF"));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), listener.accept())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    #[allow(clippy::result_large_err)] // callback signature fixed by tungstenite
    async fn connects_to_pinned_address_and_preserves_original_host() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            tokio_tungstenite::accept_hdr_async(stream, |request: &Request, response| {
                assert_eq!(request.headers()["host"], "unresolvable.invalid");
                Ok(response)
            })
            .await
            .unwrap()
        });
        let request = "ws://unresolvable.invalid/realtime"
            .into_client_request()
            .unwrap();
        let connected = connect_addresses(request, vec![address], true).await;
        assert!(
            connected.is_ok(),
            "must use the supplied IP instead of resolving the URI again"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn literal_loopback_requires_the_operator_opt_in() {
        let request = "ws://[::ffff:127.0.0.1]:1/realtime"
            .into_client_request()
            .unwrap();
        assert!(
            connect(request, false)
                .await
                .unwrap_err()
                .to_string()
                .contains("SSRF")
        );
    }

    #[tokio::test]
    async fn a_stalled_handshake_timeout_keeps_the_504_policy_bucket_and_transport_class() {
        use tokio::io::AsyncReadExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let request = format!("ws://{}/realtime", listener.local_addr().unwrap())
            .into_client_request()
            .unwrap();
        let task = tokio::spawn(connect(request, true));
        let (mut peer, _) = listener.accept().await.unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(3),
            peer.read_exact(&mut [0u8; 1]),
        )
        .await
        .unwrap()
        .unwrap();
        // Pause only after real socket I/O proves the handshake timer is armed;
        // automatic paused-time advancement must not race TCP setup.
        tokio::time::pause();
        tokio::time::advance(super::super::UPSTREAM_CONNECT_TIMEOUT).await;
        let error = task.await.unwrap().unwrap_err();
        assert_eq!(error.failure_status, Some(504));
        assert!(
            matches!(error.error, Error::Network(_)),
            "a local timeout must not pretend that an upstream HTTP response arrived"
        );
    }
}
