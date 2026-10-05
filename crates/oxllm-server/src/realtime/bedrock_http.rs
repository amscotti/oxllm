//! Native Bedrock InvokeModelWithBidirectionalStream: signed POST over HTTP/2.
//!
//! The official SDK owns the REST operation, modeled event envelopes, CRCs,
//! initial SigV4 signature and chained event signatures. A Smithy connector
//! carries that signed, streaming body through our SSRF-protected HTTP stack.
//! Nova/OpenAI event translation remains in the pure provider crate.

use std::future::Future;
use std::io;
use std::net::IpAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};

use aws_sdk_bedrockruntime::config::{Credentials, Region, StalledStreamProtectionConfig};
use aws_sdk_bedrockruntime::operation::invoke_model_with_bidirectional_stream::{
    InvokeModelWithBidirectionalStreamError, InvokeModelWithBidirectionalStreamOutput,
};
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::error::{
    InvokeModelWithBidirectionalStreamInputError, InvokeModelWithBidirectionalStreamOutputError,
};
use aws_sdk_bedrockruntime::types::{
    BidirectionalInputPayloadPart, InvokeModelWithBidirectionalStreamInput,
    InvokeModelWithBidirectionalStreamOutput as OutputEvent,
};
use aws_smithy_runtime_api::client::auth::AuthSchemeId;
use aws_smithy_runtime_api::client::http::{
    HttpConnector, HttpConnectorFuture, SharedHttpConnector, http_client_fn,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::result::{ConnectorError, SdkError};
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::error::metadata::ProvideErrorMetadata;
use aws_smithy_types::event_stream::RawMessage;
use aws_smithy_types::retry::RetryConfig;
use aws_smithy_types::timeout::TimeoutConfig;
use axum::extract::ws::{Message as AxumMessage, WebSocket};
use futures_util::{Stream, StreamExt};
use http_body::Frame;
use http_body_util::{BodyExt, StreamBody};
use oxllm_core::{Error, RealtimeProvider};
use oxllm_providers::bedrock::BedrockRealtimeTransform;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use super::bedrock::{BedrockRealtimeAdapter, deliver_upstream_texts};
use super::connect::ConnectError;
use super::session::{SessionEnd, SessionMeter, SessionOutcome};

/// Bounded independently of TCP/HTTP2 flow control. A stalled peer cannot make
/// client audio accumulate in an unbounded producer task.
const INPUT_QUEUE_CAPACITY: usize = 4;
const EVENTSTREAM_FRAME_CAP: usize = 16 * 1024 * 1024;

type InputSender = mpsc::Sender<InvokeModelWithBidirectionalStreamInput>;
type SetupError = SdkError<InvokeModelWithBidirectionalStreamError, HttpResponse>;
type StreamError = SdkError<InvokeModelWithBidirectionalStreamOutputError, RawMessage>;
type PendingInput = Pin<Box<dyn Future<Output = oxllm_core::Result<()>> + Send>>;

/// The connector reports real headers before the SDK's error-body parser can
/// wait on an oversized or indefinitely slow provider error response.
#[derive(Clone, Debug)]
struct DuplexConnector {
    client: reqwest::Client,
    allow_private: bool,
    headers: Arc<Mutex<Option<oneshot::Sender<u16>>>>,
    input_finished: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl HttpConnector for DuplexConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let connector = self.clone();
        HttpConnectorFuture::new(async move {
            let request = request
                .try_into_http1x()
                .map_err(|error| ConnectorError::user(Box::new(error)))?;
            let (parts, body) = request.into_parts();
            let url = reqwest::Url::parse(&parts.uri.to_string())
                .map_err(|error| ConnectorError::user(Box::new(error)))?;
            check_literal_host(&url, connector.allow_private)
                .map_err(|error| ConnectorError::io(Box::new(error)))?;
            let input_finished = connector
                .input_finished
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take();
            // The SDK has already signed the exact URI and headers. HTTP2 is
            // mandatory; no redirects, proxy discovery or HTTP1 fallback.
            let response = connector
                .client
                .request(parts.method, url)
                .headers(parts.headers)
                .version(reqwest::Version::HTTP_2)
                .body(reqwest::Body::wrap_stream(InputBody::new(
                    body.into_data_stream(),
                    input_finished,
                )))
                .send()
                .await
                .map_err(|error| {
                    if error.is_timeout() {
                        ConnectorError::timeout(Box::new(error))
                    } else {
                        ConnectorError::io(Box::new(error))
                    }
                })?;
            if response.version() != reqwest::Version::HTTP_2 {
                return Err(ConnectorError::io(Box::new(io::Error::other(
                    "Bedrock bidirectional streaming requires an HTTP/2 response",
                ))));
            }
            let status = response.status();
            if let Some(sender) = connector
                .headers
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
            {
                let _ = sender.send(status.as_u16());
            }
            let headers = response.headers().clone();
            let body = bounded_body(response, status.as_u16() == 200);
            let mut response = axum::http::Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = headers;
            *response.version_mut() = axum::http::Version::HTTP_2;
            HttpResponse::try_from(response)
                .map_err(|error| ConnectorError::other(Box::new(error), None))
        })
    }
}

fn check_literal_host(url: &reqwest::Url, allow_private: bool) -> io::Result<()> {
    let host = url.host_str().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Bedrock endpoint has no host")
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "native Bedrock transport requires HTTP(S)",
        ));
    }
    // Reqwest bypasses its resolver for IP literals. DNS names are checked by
    // SsrfResolver at the one resolution actually used to dial the socket.
    if !allow_private
        && let Ok(ip) = host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<IpAddr>()
        && !crate::security::check_ip_allowed(ip)
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "Bedrock endpoint blocked by the private-address policy",
        ));
    }
    Ok(())
}

fn http2_client(allow_private: bool) -> oxllm_core::Result<reqwest::Client> {
    static PUBLIC: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    static PRIVATE: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    let client = if allow_private { &PRIVATE } else { &PUBLIC };
    client
        .get_or_init(|| {
            let mut builder = reqwest::Client::builder()
                .http2_prior_knowledge()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(super::UPSTREAM_CONNECT_TIMEOUT);
            if !allow_private {
                builder = builder.dns_resolver(Arc::new(crate::security::SsrfResolver::new()));
            }
            // Do not install a total/read timeout on a healthy duplex body.
            // Setup, writes, draining and session lifetime are bounded below.
            builder.build().map_err(|error| error.to_string())
        })
        .clone()
        .map_err(|_| Error::Config("could not build the Bedrock HTTP/2 transport".into()))
}

/// Request/response streams have one consumer. This adapter makes that ownership
/// explicit while satisfying SdkBody's Sync bound without requiring a Sync
/// network stream or holding a lock across an await.
struct SyncStream<S> {
    inner: Mutex<Pin<Box<S>>>,
}

/// Notify after the HTTP stack has consumed the SDK's signed terminal frame
/// and end-of-input. A final usage event must not make us discard queued Nova
/// shutdown events before the producer has actually consumed them.
struct InputBody<S> {
    inner: Pin<Box<S>>,
    finished: Option<oneshot::Sender<()>>,
}

impl<S> InputBody<S> {
    fn new(stream: S, finished: Option<oneshot::Sender<()>>) -> Self {
        Self {
            inner: Box::pin(stream),
            finished,
        }
    }
}

impl<S: Stream> Stream for InputBody<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let result = this.inner.as_mut().poll_next(cx);
        if matches!(&result, Poll::Ready(None))
            && let Some(finished) = this.finished.take()
        {
            let _ = finished.send(());
        }
        result
    }
}

impl<S> SyncStream<S> {
    fn new(stream: S) -> Self {
        Self {
            inner: Mutex::new(Box::pin(stream)),
        }
    }
}

impl<S: Stream> Stream for SyncStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_mut()
            .poll_next(cx)
    }
}

/// Inspect each declared frame length before the SDK can buffer its payload.
/// Only a four-byte prefix is retained here; CRC/headers/unmarshalling belong
/// to the official SDK. Multiple/fragmented frames are handled identically.
#[derive(Default)]
struct FrameBounds {
    prefix: [u8; 4],
    filled: usize,
    remaining: usize,
}

impl FrameBounds {
    fn observe(&mut self, mut bytes: &[u8]) -> io::Result<()> {
        while !bytes.is_empty() {
            if self.remaining > 0 {
                let count = self.remaining.min(bytes.len());
                self.remaining -= count;
                bytes = &bytes[count..];
                continue;
            }
            let count = (4 - self.filled).min(bytes.len());
            self.prefix[self.filled..self.filled + count].copy_from_slice(&bytes[..count]);
            self.filled += count;
            bytes = &bytes[count..];
            if self.filled == 4 {
                let length = u32::from_be_bytes(self.prefix) as usize;
                if !(16..=EVENTSTREAM_FRAME_CAP).contains(&length) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Bedrock eventstream frame length exceeded the allowed bounds",
                    ));
                }
                self.filled = 0;
                self.remaining = length - 4;
            }
        }
        Ok(())
    }

    fn complete(&self) -> bool {
        self.filled == 0 && self.remaining == 0
    }
}

fn bounded_body(response: reqwest::Response, events: bool) -> SdkBody {
    let stream = futures_util::stream::unfold(
        (
            Box::pin(response.bytes_stream()),
            FrameBounds::default(),
            0usize,
            false,
        ),
        move |(mut stream, mut bounds, mut total, ended)| async move {
            if ended {
                return None;
            }
            let (item, ended) = match stream.next().await {
                Some(Ok(bytes)) => {
                    let valid = if events {
                        bounds.observe(&bytes)
                    } else {
                        total = total.saturating_add(bytes.len());
                        if total > crate::security::MAX_ERROR_BODY_BYTES {
                            Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "Bedrock error body exceeded the limit",
                            ))
                        } else {
                            Ok(())
                        }
                    };
                    match valid {
                        Ok(()) => (Ok(Frame::data(bytes)), false),
                        Err(error) => (Err(error), true),
                    }
                }
                Some(Err(_)) => (
                    Err(io::Error::other("Bedrock HTTP/2 response body failed")),
                    true,
                ),
                None if events && !bounds.complete() => (
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "incomplete Bedrock eventstream frame",
                    )),
                    true,
                ),
                None => return None,
            };
            Some((item, (stream, bounds, total, ended)))
        },
    )
    // HTTP body consumers may poll again after EOF. Unfold itself panics on
    // that legal body usage; fuse the shipped stream at the SDK boundary.
    .fuse();
    SdkBody::from_body_1_x(StreamBody::new(SyncStream::new(stream)))
}

/// Return the endpoint without a query, plus the original query for a pre-signing
/// SDK request interceptor. Smithy's endpoint path merger otherwise drops it.
fn native_endpoint(
    adapter: &BedrockRealtimeAdapter,
) -> oxllm_core::Result<(String, Option<String>)> {
    let default = format!("https://bedrock-runtime.{}.amazonaws.com", adapter.region);
    let mut endpoint = url::Url::parse(adapter.api_base.as_deref().unwrap_or(&default))
        .map_err(|_| Error::Config("invalid native Bedrock endpoint".into()))?;
    if !matches!(endpoint.scheme(), "http" | "https")
        || endpoint.host_str().is_none()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(Error::Config(
            "native Bedrock endpoint must be HTTP(S), without userinfo or a fragment".into(),
        ));
    }
    let query = endpoint
        .query()
        .filter(|query| !query.is_empty())
        .map(str::to_string);
    endpoint.set_query(None);
    Ok((endpoint.to_string(), query))
}

fn sdk_client(
    adapter: &BedrockRealtimeAdapter,
    connector: DuplexConnector,
    endpoint: String,
) -> oxllm_core::Result<aws_sdk_bedrockruntime::Client> {
    if adapter.access_key.is_empty() || adapter.secret_key.is_empty() {
        return Err(Error::Config(
            "missing Bedrock realtime signing credentials".into(),
        ));
    }
    let credentials = Credentials::new(
        adapter.access_key.clone(),
        adapter.secret_key.clone(),
        adapter.session_token.clone(),
        None,
        "oxllm-realtime-deployment",
    );
    let connector = SharedHttpConnector::new(connector);
    let config = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version_latest()
        .region(Region::new(adapter.region.clone()))
        .credentials_provider(credentials)
        .auth_scheme_preference([AuthSchemeId::from("sigv4")])
        .endpoint_url(endpoint)
        .http_client(http_client_fn(move |_, _| connector.clone()))
        // A duplex input is not replayable. Retrying can duplicate speech and
        // hide attempts from the shared router's failure policy.
        .retry_config(RetryConfig::disabled())
        .timeout_config(TimeoutConfig::disabled())
        // Quiet audio sessions are healthy. The host bounds active writes and
        // total session time; a generic upload/download rate floor would kill
        // an idle microphone or a model awaiting the next user turn.
        .stalled_stream_protection(StalledStreamProtectionConfig::disabled())
        .build();
    Ok(aws_sdk_bedrockruntime::Client::from_conf(config))
}

fn append_endpoint_query(
    mut request: HttpRequest,
    query: &Option<String>,
) -> io::Result<HttpRequest> {
    if let Some(query) = query {
        let separator = if request.uri().contains('?') {
            '&'
        } else {
            '?'
        };
        let uri = format!("{}{separator}{query}", request.uri());
        request
            .set_uri(uri)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    }
    Ok(request)
}

fn status_error(status: u16) -> ConnectError {
    Error::Upstream {
        status,
        message: "Bedrock rejected the bidirectional HTTP/2 request".into(),
    }
    .into()
}

fn setup_error(error: &SetupError) -> ConnectError {
    if let Some(response) = error
        .raw_response()
        .filter(|response| response.status().as_u16() != 200)
    {
        return status_error(response.status().as_u16());
    }
    match error {
        SdkError::ConstructionFailure(_) => {
            Error::Config("could not construct the Bedrock duplex request".into()).into()
        }
        SdkError::TimeoutError(_) => connect_timeout(),
        SdkError::DispatchFailure(error) if error.is_timeout() => connect_timeout(),
        _ => Error::Network("Bedrock HTTP/2 connection failed".into()).into(),
    }
}

fn connect_timeout() -> ConnectError {
    ConnectError {
        error: Error::Network("Bedrock bidirectional HTTP/2 setup timed out".into()),
        failure_status: Some(504),
    }
}

fn stream_error(error: &StreamError) -> Error {
    if let Some(error) = error.as_service_error() {
        if let InvokeModelWithBidirectionalStreamOutputError::ModelStreamErrorException(detail) =
            error
        {
            let status = detail
                .original_status_code()
                .and_then(|status| u16::try_from(status).ok())
                .filter(|status| (400..=599).contains(status))
                .unwrap_or(424);
            return Error::Upstream {
                status,
                message: "Bedrock model stream failed".into(),
            };
        }
        // The SDK selects the modeled variant from :exception-type. Its
        // generic metadata is parsed separately from the JSON payload and
        // often has no code at all, so metadata alone loses 400/408/429/503.
        let status = match error {
            InvokeModelWithBidirectionalStreamOutputError::ValidationException(_) => 400,
            InvokeModelWithBidirectionalStreamOutputError::ThrottlingException(_) => 429,
            InvokeModelWithBidirectionalStreamOutputError::ModelTimeoutException(_) => 408,
            InvokeModelWithBidirectionalStreamOutputError::ServiceUnavailableException(_) => 503,
            InvokeModelWithBidirectionalStreamOutputError::InternalServerException(_) => 500,
            _ => match error.code() {
                Some("ValidationException") => 400,
                Some("ThrottlingException") => 429,
                Some("ModelTimeoutException") => 408,
                Some("ModelStreamErrorException") => 424,
                Some("ServiceUnavailableException") => 503,
                _ => 500,
            },
        };
        return Error::Upstream {
            status,
            message: "Bedrock bidirectional stream failed".into(),
        };
    }
    Error::Network("Bedrock HTTP/2 eventstream read failed".into())
}

fn output_text(event: OutputEvent) -> oxllm_core::Result<String> {
    let OutputEvent::Chunk(chunk) = event else {
        return Err(Error::Network(
            "unknown Bedrock bidirectional stream event".into(),
        ));
    };
    let bytes = chunk
        .bytes()
        .ok_or_else(|| Error::Network("Bedrock chunk has no payload".into()))?;
    if bytes.as_ref().len() > super::REALTIME_MAX_MESSAGE_SIZE {
        return Err(Error::Network(
            "Bedrock realtime event exceeded the message limit".into(),
        ));
    }
    std::str::from_utf8(bytes.as_ref())
        .map(str::to_string)
        .map_err(|_| Error::Serialization("invalid Bedrock realtime UTF-8 payload".into()))
}

async fn send_events(sender: &InputSender, events: Vec<String>) -> oxllm_core::Result<()> {
    if events
        .iter()
        .any(|event| event.len() > super::REALTIME_MAX_MESSAGE_SIZE)
    {
        return Err(Error::BadRequest(
            "realtime event exceeded the message limit".into(),
        ));
    }
    tokio::time::timeout(super::WS_SEND_TIMEOUT, async {
        for event in events {
            let chunk = BidirectionalInputPayloadPart::builder()
                .bytes(Blob::new(event.into_bytes()))
                .build();
            sender
                .send(InvokeModelWithBidirectionalStreamInput::Chunk(chunk))
                .await
                .map_err(|_| Error::Network("Bedrock duplex input closed".into()))?;
        }
        Ok(())
    })
    .await
    .unwrap_or_else(|_| Err(Error::Network("Bedrock duplex input stalled".into())))
}

async fn client_event(
    transform: &BedrockRealtimeTransform,
    sender: &InputSender,
    client: &mut WebSocket,
    message: AxumMessage,
) -> Result<(), SessionEnd> {
    let event = match message {
        AxumMessage::Text(text) => text,
        AxumMessage::Binary(_) => {
            let error =
                Error::BadRequest("Nova Sonic audio must use input_audio_buffer.append".into());
            return if super::send_client(client, super::client_event_error("{}", &error)).await {
                Ok(())
            } else {
                Err(SessionEnd::ClientDisconnected)
            };
        }
        AxumMessage::Ping(_) | AxumMessage::Pong(_) => return Ok(()),
        AxumMessage::Close(_) => return Err(SessionEnd::ClientDisconnected),
    };
    let result = match transform.transform_client_event(event.as_str()) {
        Ok(events) => send_events(sender, events).await,
        Err(error) => Err(error),
    };
    match result {
        Ok(()) => Ok(()),
        Err(error @ Error::Network(_)) => Err(SessionEnd::UpstreamError(error)),
        Err(error) => {
            if super::send_client(client, super::client_event_error(event.as_str(), &error)).await {
                Ok(())
            } else {
                Err(SessionEnd::ClientDisconnected)
            }
        }
    }
}

/// Opens the actual native API and proxies the accepted client socket. Client
/// events can flow during setup; servers may wait for the first input before
/// returning HTTP headers. No default session.update/sessionStart is injected.
pub(super) async fn connect_and_proxy(
    adapter: BedrockRealtimeAdapter,
    mut client: WebSocket,
    allow_private: bool,
    mut meter: SessionMeter,
    on_connect: impl FnOnce(Result<(), &ConnectError>) + Send,
) -> oxllm_core::Result<SessionOutcome> {
    let (endpoint, query) = match native_endpoint(&adapter) {
        Ok(endpoint) => endpoint,
        Err(error) => {
            super::send_client(
                &mut client,
                super::upstream_failure_close("invalid Bedrock endpoint"),
            )
            .await;
            return Err(error);
        }
    };
    let (headers_tx, mut headers_rx) = oneshot::channel();
    let (input_finished_tx, mut input_finished_rx) = oneshot::channel();
    let http = match http2_client(allow_private) {
        Ok(http) => http,
        Err(error) => {
            super::send_client(
                &mut client,
                super::upstream_failure_close("Bedrock HTTP setup failed"),
            )
            .await;
            return Err(error);
        }
    };
    let connector = DuplexConnector {
        client: http,
        allow_private,
        headers: Arc::new(Mutex::new(Some(headers_tx))),
        input_finished: Arc::new(Mutex::new(Some(input_finished_tx))),
    };
    let sdk = match sdk_client(&adapter, connector, endpoint) {
        Ok(sdk) => sdk,
        Err(error) => {
            super::send_client(
                &mut client,
                super::upstream_failure_close("Bedrock signing setup failed"),
            )
            .await;
            return Err(error);
        }
    };
    let (sender, receiver) = mpsc::channel(INPUT_QUEUE_CAPACITY);
    let mut sender = Some(sender);
    let body = ReceiverStream::new(receiver).map(
        Ok::<InvokeModelWithBidirectionalStreamInput, InvokeModelWithBidirectionalStreamInputError>,
    );
    let request = sdk
        .invoke_model_with_bidirectional_stream()
        .model_id(adapter.model_id.clone())
        .body(body.into())
        .customize()
        .map_request(move |request| append_endpoint_query(request, &query))
        .send();
    tokio::pin!(request);
    let mut on_connect = Some(on_connect);
    let mut received_status = None;
    let mut headers_finished = false;
    let mut early_end = None;
    let mut pending_input: Option<PendingInput> = None;
    let mut pending_text: Option<String> = None;
    let mut deadline = tokio::time::Instant::now() + super::UPSTREAM_CONNECT_TIMEOUT;
    let mut output = loop {
        tokio::select! {
            biased;
            status = &mut headers_rx, if !headers_finished => {
                headers_finished = true;
                if let Ok(status) = status {
                    received_status = Some(status);
                    if status != 200 {
                        let error = status_error(status);
                        if let Some(callback) = on_connect.take() { callback(Err(&error)); }
                        super::send_client(&mut client, super::upstream_failure_close("Bedrock HTTP request failed")).await;
                        return Err(error.error);
                    }
                    if let Some(callback) = on_connect.take() { callback(Ok(())); }
                }
            }
            result = &mut request => {
                // The SDK can finish in the same poll that made the status
                // channel ready. Consume that real status before notifying.
                if received_status.is_none() {
                    received_status = headers_rx.try_recv().ok();
                }
                if received_status == Some(200) {
                    if let Some(callback) = on_connect.take() { callback(Ok(())); }
                }
                match result {
                    Ok(output) if received_status == Some(200) => {
                        // A response can arrive midway through a client config
                        // batch. Preserve the unsent tail; never drop prompt or
                        // tool configuration when changing from setup to pump.
                        if let Some(input) = pending_input.take() {
                            if let Err(error) = input.await {
                                if matches!(error, Error::Network(_)) {
                                    meter.ending(&SessionEnd::UpstreamError(error.clone()));
                                    super::send_client(&mut client, super::upstream_failure_close("Bedrock duplex input failed")).await;
                                    return Err(error);
                                }
                                if !super::send_client(&mut client, super::client_event_error(
                                    pending_text.take().as_deref().unwrap_or("{}"), &error,
                                )).await {
                                    meter.ending(&SessionEnd::ClientDisconnected);
                                    early_end = Some(SessionEnd::ClientDisconnected);
                                }
                            }
                        }
                        break output;
                    }
                    Ok(_) => {
                        let error = received_status.map(status_error).unwrap_or_else(|| {
                            Error::Network("Bedrock response arrived without HTTP status metadata".into()).into()
                        });
                        if let Some(callback) = on_connect.take() { callback(Err(&error)); }
                        super::send_client(&mut client, super::upstream_failure_close("invalid Bedrock HTTP response")).await;
                        return Err(error.error);
                    }
                    Err(error) => {
                        let error = received_status.filter(|status| *status != 200)
                            .map(status_error).unwrap_or_else(|| setup_error(&error));
                        if let Some(callback) = on_connect.take() { callback(Err(&error)); }
                        meter.ending(&SessionEnd::UpstreamError(error.error.clone()));
                        super::send_client(&mut client, super::upstream_failure_close("Bedrock HTTP setup failed")).await;
                        return Err(error.error);
                    }
                }
            }
            result = async {
                match pending_input.as_mut() {
                    Some(input) => input.await,
                    None => futures_util::future::pending().await,
                }
            }, if pending_input.is_some() => {
                pending_input.take();
                let text = pending_text.take().unwrap_or_default();
                if let Err(error) = result {
                    if matches!(error, Error::Network(_)) {
                        // Header delivery and input cancellation can become
                        // ready in one SDK poll (notably on an HTTP rejection).
                        // Keep that real status instead of flattening it into
                        // an input-channel/network failure.
                        if received_status.is_none() {
                            if let Ok(status) = headers_rx.try_recv() {
                                received_status = Some(status);
                            }
                        }
                        if received_status == Some(200) {
                            if let Some(callback) = on_connect.take() { callback(Ok(())); }
                        }
                        let connect_error = received_status.filter(|status| *status != 200)
                            .map(status_error).unwrap_or_else(|| ConnectError::from(error));
                        if let Some(callback) = on_connect.take() { callback(Err(&connect_error)); }
                        meter.ending(&SessionEnd::UpstreamError(connect_error.error.clone()));
                        super::send_client(&mut client, super::upstream_failure_close("Bedrock duplex input failed")).await;
                        return Err(connect_error.error);
                    }
                    if !super::send_client(&mut client, super::client_event_error(&text, &error)).await {
                        meter.ending(&SessionEnd::ClientDisconnected);
                        early_end = Some(SessionEnd::ClientDisconnected);
                        deadline = tokio::time::Instant::now() + super::USAGE_DRAIN_TIMEOUT;
                    }
                }
            }
            message = client.next(), if early_end.is_none() && pending_input.is_none() => {
                // Poll the SDK request concurrently with bounded input writes.
                // Awaiting a whole batch here before polling request would
                // deadlock a first session.update larger than the queue.
                if let Some(Ok(AxumMessage::Text(text))) = &message {
                    match adapter.transform.transform_client_event(text.as_str()) {
                        Ok(events) => {
                            if let Some(input) = sender.as_ref().cloned() {
                                pending_text = Some(text.to_string());
                                pending_input = Some(Box::pin(async move { send_events(&input, events).await }));
                                continue;
                            }
                        }
                        Err(error) => {
                            if super::send_client(&mut client, super::client_event_error(text.as_str(), &error)).await {
                                continue;
                            }
                            meter.ending(&SessionEnd::ClientDisconnected);
                            early_end = Some(SessionEnd::ClientDisconnected);
                            deadline = tokio::time::Instant::now() + super::USAGE_DRAIN_TIMEOUT;
                            continue;
                        }
                    }
                }
                let end = match (message, sender.as_ref()) {
                    (Some(Ok(message)), Some(sender)) => client_event(&adapter.transform, sender, &mut client, message).await.err(),
                    _ => Some(SessionEnd::ClientDisconnected),
                };
                if let Some(end) = end {
                    meter.ending(&end);
                    if end.should_drain() {
                        if let Some(input) = sender.take() {
                            let _ = tokio::time::timeout(super::USAGE_DRAIN_TIMEOUT,
                                send_events(&input, adapter.transform.shutdown_events())).await;
                            drop(input); // SDK sends its signed terminal event.
                        }
                        deadline = tokio::time::Instant::now() + super::USAGE_DRAIN_TIMEOUT;
                        early_end = Some(end);
                    } else if let SessionEnd::UpstreamError(error) = end {
                        return Err(error);
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                // A caller disconnect is not a failure of deployment health.
                if early_end.is_some() && received_status.is_none() {
                    return Err(Error::BadRequest("realtime client disconnected before Bedrock accepted the request".into()));
                }
                let error = connect_timeout();
                if let Some(callback) = on_connect.take() { callback(Err(&error)); }
                meter.ending(&SessionEnd::UpstreamError(error.error.clone()));
                super::send_client(&mut client, super::upstream_failure_close("Bedrock HTTP setup timeout")).await;
                return Err(error.error);
            }
        }
    };

    let end = if let Some(end) = early_end {
        end
    } else {
        // HTTP 200 accepts the proxy session. Emit its OpenAI control event
        // without sending a default Nova sessionStart/promptStart upstream or
        // mutating the client's inference/voice/tool configuration.
        let created = vec![r#"{"event":{"sessionStart":{}}}"#.to_string()];
        match deliver_upstream_texts(&adapter.transform, &mut meter, &mut client, &created).await {
            Ok(()) => {
                pump(
                    &adapter.transform,
                    &mut client,
                    &mut output,
                    sender.as_ref(),
                    &mut meter,
                )
                .await
            }
            Err(end) => end,
        }
    };
    meter.ending(&end);
    if end.should_drain() {
        let _ = tokio::time::timeout(super::USAGE_DRAIN_TIMEOUT, async {
            if let Some(input) = sender.take() {
                let _ = send_events(&input, adapter.transform.shutdown_events()).await;
                drop(input);
            }
            while let Ok(Some(event)) = output.body.recv().await {
                let Ok(text) = output_text(event) else {
                    break;
                };
                if meter.observe(&adapter.transform, &text) {
                    break;
                }
            }
            let _ = (&mut input_finished_rx).await;
        })
        .await;
    }
    // Dropping both halves cancels just this HTTP2 stream, including on hard
    // task abort. There is no detached producer that can keep audio flowing.
    drop(sender);
    drop(output);
    let close = if matches!(end, SessionEnd::UpstreamError(_)) {
        super::upstream_failure_close("Bedrock realtime session failed")
    } else {
        AxumMessage::Close(None)
    };
    let _ = tokio::time::timeout(super::USAGE_DRAIN_TIMEOUT, client.send(close)).await;
    Ok(meter.finish(end))
}

async fn pump(
    transform: &BedrockRealtimeTransform,
    client: &mut WebSocket,
    output: &mut InvokeModelWithBidirectionalStreamOutput,
    sender: Option<&InputSender>,
    meter: &mut SessionMeter,
) -> SessionEnd {
    let deadline = tokio::time::Instant::now() + super::REALTIME_MAX_SESSION_DURATION;
    loop {
        tokio::select! {
            message = client.next() => {
                let end = match (message, sender) {
                    (Some(Ok(message)), Some(sender)) => client_event(transform, sender, client, message).await.err(),
                    _ => Some(SessionEnd::ClientDisconnected),
                };
                if let Some(end) = end { return end; }
            }
            event = output.body.recv() => match event {
                Ok(Some(event)) => {
                    let text = match output_text(event) {
                        Ok(text) => text,
                        Err(error) => return SessionEnd::UpstreamError(error),
                    };
                    if let Err(end) = deliver_upstream_texts(transform, meter, client, &[text]).await {
                        return end;
                    }
                }
                Ok(None) => return SessionEnd::UpstreamClosed,
                Err(error) => return SessionEnd::UpstreamError(stream_error(&error)),
            },
            _ = tokio::time::sleep_until(deadline) => return SessionEnd::TimedOut,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use aws_sdk_bedrockruntime::primitives::DateTimeFormat;
    use aws_sdk_bedrockruntime::primitives::event_stream::{Header, HeaderValue, Message};
    use aws_smithy_eventstream::frame::{read_message_from, write_message_to};
    use axum::body::Bytes;
    use base64::Engine;
    use futures_util::SinkExt;
    use hmac::{Hmac, Mac};
    use http_body_util::combinators::BoxBody;
    use hyper::body::Incoming;
    use hyper::service::service_fn;
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use tokio::sync::Notify;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

    type MockBody = BoxBody<Bytes, io::Error>;

    struct Peer {
        base: String,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for Peer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn http2_peer<F, Fut>(handler: F) -> Peer
    where
        F: Fn(axum::http::Request<Incoming>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = axum::http::Response<MockBody>> + Send + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let service = service_fn(move |request| {
                let response = handler(request);
                async move { Ok::<_, Infallible>(response.await) }
            });
            let _ = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
        Peer { base, task }
    }

    fn empty_body() -> MockBody {
        BodyExt::boxed(http_body_util::Empty::<Bytes>::new().map_err(|never| match never {}))
    }

    fn response_body(receiver: mpsc::Receiver<io::Result<Frame<Bytes>>>) -> MockBody {
        BodyExt::boxed(StreamBody::new(ReceiverStream::new(receiver)))
    }

    fn drain_request_body(mut body: Incoming) {
        // A native caller still owns the upload half when response headers
        // arrive. Keep the mock's receive half alive rather than accidentally
        // resetting the HTTP2 stream by dropping an unread request body.
        tokio::spawn(async move { while let Some(Ok(_)) = body.frame().await {} });
    }

    fn adapter(base: &str, model: &str) -> BedrockRealtimeAdapter {
        BedrockRealtimeAdapter::from_deployment(&oxllm_core::Deployment {
            model_name: "sonic".into(),
            oxllm_params: oxllm_core::OxllmParams {
                model: format!("bedrock/{model}"),
                api_base: Some(base.into()),
                aws_access_key_id: Some("AKIATEST".into()),
                aws_secret_access_key: Some("secret".into()),
                aws_session_token: Some("test-token+/=".into()),
                extra: serde_json::Map::from_iter([("aws_region_name".into(), "eu-west-1".into())]),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap()
    }

    fn encoded_message(event_type: &str, payload: Value) -> Bytes {
        let message = Message::new(payload.to_string().into_bytes())
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            ))
            .add_header(Header::new(
                ":event-type",
                HeaderValue::String(event_type.to_owned().into()),
            ))
            .add_header(Header::new(
                ":content-type",
                HeaderValue::String("application/json".into()),
            ));
        let mut bytes = Vec::new();
        write_message_to(&message, &mut bytes).unwrap();
        bytes.into()
    }

    fn nova_frame(event: Value) -> Bytes {
        encoded_message(
            "chunk",
            json!({
                "bytes":base64::engine::general_purpose::STANDARD.encode(event.to_string())
            }),
        )
    }

    fn exception_frame(kind: &str, payload: Value) -> Bytes {
        let message = Message::new(payload.to_string().into_bytes())
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("exception".into()),
            ))
            .add_header(Header::new(
                ":exception-type",
                HeaderValue::String(kind.to_owned().into()),
            ))
            .add_header(Header::new(
                ":content-type",
                HeaderValue::String("application/json".into()),
            ));
        let mut bytes = Vec::new();
        write_message_to(&message, &mut bytes).unwrap();
        bytes.into()
    }

    fn hmac(key: &[u8], text: &[u8]) -> Vec<u8> {
        let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
        mac.update(text);
        mac.finalize().into_bytes().to_vec()
    }

    fn signing_key(date: &str, region: &str) -> Vec<u8> {
        let date_key = hmac(b"AWS4secret", date.as_bytes());
        let region_key = hmac(&date_key, region.as_bytes());
        let service_key = hmac(&region_key, b"bedrock");
        hmac(&service_key, b"aws4_request")
    }

    fn hash(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    /// Independent authentication oracle: no SDK signer is used to calculate
    /// the expected HTTP signature or chained frame signatures. The nested
    /// event encoding is parsed by the official CRC-checked frame reader.
    struct SignedInput {
        body: Incoming,
        buffer: Vec<u8>,
        previous_signature: String,
        region: String,
        signed_end: bool,
    }

    impl SignedInput {
        fn new(request: axum::http::Request<Incoming>) -> Self {
            let (parts, body) = request.into_parts();
            assert_eq!(parts.method, axum::http::Method::POST);
            assert_eq!(parts.version, axum::http::Version::HTTP_2);
            assert_eq!(
                parts.headers["content-type"],
                "application/vnd.amazon.eventstream"
            );
            assert_eq!(parts.headers["x-amz-security-token"], "test-token+/=");
            assert!(!parts.headers.contains_key("upgrade"));
            let auth = parts.headers["authorization"].to_str().unwrap();
            assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIATEST/"));
            let credential = auth
                .split("Credential=")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap();
            let scope = credential.split_once('/').unwrap().1;
            let scope_parts: Vec<_> = scope.split('/').collect();
            assert_eq!(scope_parts[1], "eu-west-1");
            assert_eq!(scope_parts[2], "bedrock");
            let signed_headers = auth
                .split("SignedHeaders=")
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap();
            let signature = auth.split("Signature=").nth(1).unwrap().to_string();
            let canonical_headers: String = signed_headers
                .split(';')
                .map(|name| {
                    let value = if name == "host" {
                        parts.uri.authority().unwrap().as_str()
                    } else {
                        parts.headers.get(name).unwrap().to_str().unwrap()
                    };
                    format!(
                        "{name}:{}\n",
                        value.split_whitespace().collect::<Vec<_>>().join(" ")
                    )
                })
                .collect();
            let canonical = format!(
                "POST\n{}\n{}\n{canonical_headers}\n{signed_headers}\n{}",
                parts.uri.path().replace('%', "%25"),
                parts.uri.query().unwrap_or(""),
                hash(b"")
            );
            let date = parts.headers["x-amz-date"].to_str().unwrap();
            let to_sign = format!(
                "AWS4-HMAC-SHA256\n{date}\n{scope}\n{}",
                hash(canonical.as_bytes())
            );
            assert_eq!(
                signature,
                hex::encode(hmac(
                    &signing_key(scope_parts[0], scope_parts[1]),
                    to_sign.as_bytes()
                ))
            );
            Self {
                body,
                buffer: Vec::new(),
                previous_signature: signature,
                region: scope_parts[1].to_string(),
                signed_end: false,
            }
        }

        fn verify_frame(&mut self, frame: &Message) {
            let timestamp = frame
                .headers()
                .iter()
                .find(|header| header.name().as_str() == ":date")
                .unwrap()
                .value()
                .as_timestamp()
                .unwrap();
            let iso = timestamp.fmt(DateTimeFormat::DateTime).unwrap();
            let date_time = iso.replace(['-', ':'], "");
            let date = &date_time[..8];
            let scope = format!("{date}/{}/bedrock/aws4_request", self.region);
            // Header type 8 is a big-endian millisecond timestamp. Encode it
            // independently rather than reusing the signer's header writer.
            let mut date_header = b"\x05:date\x08".to_vec();
            date_header.extend_from_slice(&timestamp.to_millis().unwrap().to_be_bytes());
            let to_sign = format!(
                "AWS4-HMAC-SHA256-PAYLOAD\n{date_time}\n{scope}\n{}\n{}\n{}",
                self.previous_signature,
                hash(&date_header),
                hash(frame.payload())
            );
            let actual = frame
                .headers()
                .iter()
                .find(|header| header.name().as_str() == ":chunk-signature")
                .unwrap()
                .value()
                .as_byte_array()
                .unwrap();
            let expected = hmac(&signing_key(date, &self.region), to_sign.as_bytes());
            assert_eq!(
                actual.as_ref(),
                expected.as_slice(),
                "each frame must chain from the prior signature"
            );
            self.previous_signature = hex::encode(actual);
        }

        async fn next(&mut self) -> Option<Value> {
            loop {
                if self.buffer.len() >= 4 {
                    let length = u32::from_be_bytes(self.buffer[..4].try_into().unwrap()) as usize;
                    assert!((16..=EVENTSTREAM_FRAME_CAP).contains(&length));
                    if self.buffer.len() >= length {
                        let frame = read_message_from(&self.buffer[..length]).unwrap();
                        self.buffer.drain(..length);
                        self.verify_frame(&frame);
                        if frame.payload().is_empty() {
                            self.signed_end = true;
                            return None;
                        }
                        let event = read_message_from(&frame.payload()[..]).unwrap();
                        assert_eq!(
                            event
                                .headers()
                                .iter()
                                .find(|header| header.name().as_str() == ":event-type")
                                .unwrap()
                                .value()
                                .as_string()
                                .unwrap()
                                .as_str(),
                            "chunk"
                        );
                        let envelope: Value = serde_json::from_slice(event.payload()).unwrap();
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(envelope["bytes"].as_str().unwrap())
                            .unwrap();
                        return Some(serde_json::from_slice(&bytes).unwrap());
                    }
                }
                let frame = match self.body.frame().await {
                    Some(Ok(frame)) => frame,
                    _ => return None,
                };
                if let Ok(bytes) = frame.into_data() {
                    self.buffer.extend_from_slice(&bytes);
                }
            }
        }
    }

    async fn receive_type(client: &mut super::super::UpstreamStream, kind: &str) -> Value {
        tokio::time::timeout(Duration::from_secs(3), async {
            while let Some(Ok(message)) = client.next().await {
                if let Ok(text) = message.to_text() {
                    let event: Value = serde_json::from_str(text).unwrap();
                    if event["type"] == kind {
                        return event;
                    }
                }
            }
            panic!("session ended before {kind}");
        })
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn native_signed_post_duplex_keeps_client_config_and_drains_reported_usage() {
        let release = Arc::new(Notify::new());
        let first_input = Arc::new(Notify::new());
        let completed = Arc::new(Mutex::new(None));
        let recorded = completed.clone();
        let release_peer = release.clone();
        let input_peer = first_input.clone();
        let peer = http2_peer(move |request| {
            let release = release_peer.clone();
            let first_input = input_peer.clone();
            let recorded = recorded.clone();
            async move {
                assert_eq!(request.uri().path(), "/tenant/aws/model/amazon.nova-sonic-v1%3A0/invoke-with-bidirectional-stream");
                assert_eq!(request.uri().query(), Some("route=sonic"));
                let mut input = SignedInput::new(request);
                let first = input.next().await.unwrap();
                assert_eq!(first["event"]["sessionStart"]["inferenceConfiguration"]["maxTokens"], 333);
                first_input.notify_one();
                let (tx, rx) = mpsc::channel(4);
                tokio::spawn(async move {
                    let mut kinds = vec!["sessionStart".to_string()];
                    let mut prompts = 0;
                    while let Some(event) = input.next().await {
                        let kind = event["event"].as_object().unwrap().keys().next().unwrap().clone();
                        kinds.push(kind.clone());
                        match kind.as_str() {
                            "sessionStart" => {
                                assert!(kinds.iter().any(|kind| kind == "contentEnd"),
                                    "the initial config's tail must survive the HTTP response transition");
                                assert_eq!(event["event"]["sessionStart"]["inferenceConfiguration"]["maxTokens"], 444);
                            }
                            "promptStart" => {
                                prompts += 1;
                                let voice = &event["event"]["promptStart"]["audioOutputConfiguration"]["voiceId"];
                                assert_eq!(voice, if prompts == 1 { "tiffany" } else { "matthew" });
                                if prompts == 2 {
                                    for event in [
                                        json!({"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT","contentId":"assistant-1"}}}),
                                        json!({"event":{"textOutput":{"content":"hello","contentId":"assistant-1"}}}),
                                    ] {
                                        let frame = nova_frame(event);
                                        // The real SDK receiver must handle a prelude and
                                        // payload split across arbitrary HTTP2 DATA frames.
                                        for bytes in [frame.slice(..7), frame.slice(7..12), frame.slice(12..)] {
                                            tx.send(Ok(Frame::data(bytes))).await.unwrap();
                                        }
                                    }
                                }
                            }
                            "textInput" => assert_eq!(event["event"]["textInput"]["content"], "custom system instruction"),
                            "audioInput" => assert_eq!(event["event"]["audioInput"]["content"], "AAAA"),
                            "sessionEnd" => {
                                tx.send(Ok(Frame::data(nova_frame(json!({"event":{"usageEvent":{"details":{"delta":{
                                    "input":{"speechTokens":2,"textTokens":3},
                                    "output":{"speechTokens":4,"textTokens":1}
                                }}}}}))))).await.unwrap();
                                assert!(input.next().await.is_none());
                                assert!(input.signed_end, "shutdown must consume the SDK's signed terminal event");
                                *recorded.lock().unwrap() = Some(kinds);
                                return;
                            }
                            _ => {}
                        }
                    }
                    panic!("input ended before Nova shutdown");
                });
                release.notified().await;
                axum::http::Response::builder().status(200)
                    .header("content-type", "application/vnd.amazon.eventstream")
                    .body(response_body(rx)).unwrap()
            }
        }).await;
        let (socket, mut client, client_server) = super::super::tests::session_socket_pair().await;
        let notifications = Arc::new(AtomicUsize::new(0));
        let notify = notifications.clone();
        let session = tokio::spawn(
            adapter(
                &format!("{}/tenant/aws/?route=sonic", peer.base),
                "amazon.nova-sonic-v1:0:18k[200k]",
            )
            .connect_and_proxy(socket, true, SessionMeter::default(), move |result| {
                assert!(result.is_ok());
                notify.fetch_add(1, Ordering::SeqCst);
            }),
        );
        client.send(ClientMessage::Text(json!({"type":"session.update","session":{
            "max_response_output_tokens":333,"voice":"tiffany","instructions":"custom system instruction"
        }}).to_string().into())).await.unwrap();
        tokio::time::timeout(Duration::from_secs(3), first_input.notified())
            .await
            .unwrap();
        assert_eq!(
            notifications.load(Ordering::SeqCst),
            0,
            "no success before real HTTP headers"
        );
        release.notify_one();
        receive_type(&mut client, "session.created").await;
        assert_eq!(notifications.load(Ordering::SeqCst), 1);
        assert!(!session.is_finished());
        client
            .send(ClientMessage::Text(
                json!({"type":"session.update","session":{
                    "max_response_output_tokens":444,"voice":"matthew"
                }})
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        receive_type(&mut client, "response.text.delta").await;
        client
            .send(ClientMessage::Text(
                json!({"type":"input_audio_buffer.append","audio":"AAAA"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        // Receiving audio at the peer is ordered before our shutdown input;
        // no timing sleeps are needed to test cancellation/drained metering.
        client.close(None).await.unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(5), session)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(outcome.end, SessionEnd::ClientDisconnected));
        assert!(!outcome.unmetered_output);
        let usage = outcome.usage.unwrap();
        assert_eq!(usage.total_tokens, 10);
        assert_eq!(usage.audio_input_tokens, Some(2));
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if completed.lock().unwrap().is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let kinds = completed.lock().unwrap().clone().unwrap();
        assert_eq!(
            kinds
                .iter()
                .filter(|kind| kind.as_str() == "sessionStart")
                .count(),
            2,
            "only the two actual client updates may initialize inference config"
        );
        assert!(kinds.iter().any(|kind| kind == "audioInput"));
        assert_eq!(
            &kinds[kinds.len() - 3..],
            ["contentEnd", "promptEnd", "sessionEnd"]
        );
        client_server.abort();
    }

    #[tokio::test]
    async fn native_setup_reports_http_errors_promptly_without_waiting_for_error_body() {
        for status in [400, 403, 429, 503, 307] {
            let peer = http2_peer(move |request| async move {
                assert_eq!(request.method(), axum::http::Method::POST);
                drain_request_body(request.into_body());
                // Keep the body pending indefinitely. The response status
                // callback must beat the SDK's error deserializer and timeout.
                let body = BodyExt::boxed(StreamBody::new(futures_util::stream::pending::<
                    io::Result<Frame<Bytes>>,
                >()));
                axum::http::Response::builder()
                    .status(status)
                    .header("location", "https://must-not-follow.invalid/")
                    .body(body)
                    .unwrap()
            })
            .await;
            let (socket, mut client, server) = super::super::tests::session_socket_pair().await;
            let results = Arc::new(Mutex::new(Vec::new()));
            let recorded = results.clone();
            let result = tokio::time::timeout(
                Duration::from_secs(3),
                adapter(&peer.base, "amazon.nova-sonic-v1:0").connect_and_proxy(
                    socket,
                    true,
                    SessionMeter::default(),
                    move |result| {
                        recorded
                            .lock()
                            .unwrap()
                            .push(result.err().unwrap().failure_status);
                    },
                ),
            )
            .await
            .unwrap()
            .unwrap_err();
            assert!(matches!(result, Error::Upstream { status: actual, .. } if actual == status));
            assert_eq!(*results.lock().unwrap(), vec![Some(status)]);
            assert!(
                matches!(client.next().await.unwrap().unwrap(), ClientMessage::Close(Some(close)) if u16::from(close.code) == 1011)
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn native_arn_keeps_one_encoded_wire_segment_and_signed_base_prefix() {
        let observed = Arc::new(Mutex::new(None));
        let capture = observed.clone();
        let peer = http2_peer(move |request| {
            let capture = capture.clone();
            async move {
                let path = request.uri().path().to_string();
                // Also authenticate the actual request's canonical double
                // encoding, rather than only checking a URL substring.
                let input = SignedInput::new(request);
                drain_request_body(input.body);
                *capture.lock().unwrap() = Some(path);
                axum::http::Response::builder()
                    .status(200)
                    .header("content-type", "application/vnd.amazon.eventstream")
                    .body(empty_body())
                    .unwrap()
            }
        })
        .await;
        let (socket, _client, server) = super::super::tests::session_socket_pair().await;
        let arn = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/sonic";
        let result = tokio::time::timeout(
            Duration::from_secs(3),
            adapter(&format!("{}/prefix/", peer.base), arn).connect_and_proxy(
                socket,
                true,
                SessionMeter::default(),
                |result| assert!(result.is_ok()),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(matches!(result.end, SessionEnd::UpstreamClosed));
        assert_eq!(
            observed.lock().unwrap().as_deref(),
            Some(
                "/prefix/model/arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Aapplication-inference-profile%2Fsonic/invoke-with-bidirectional-stream"
            )
        );
        // Exercise the actual HTTP/2 response body through the SDK beyond its
        // first EOF. The ARN and independent request-signature oracle above
        // remains active for this request too; this is not a synthetic stream.
        let configuration = adapter(&format!("{}/prefix/", peer.base), arn);
        let (headers, _headers_rx) = oneshot::channel();
        let sdk = sdk_client(
            &configuration,
            DuplexConnector {
                client: http2_client(true).unwrap(),
                allow_private: true,
                headers: Arc::new(Mutex::new(Some(headers))),
                input_finished: Arc::new(Mutex::new(None)),
            },
            native_endpoint(&configuration).unwrap().0,
        )
        .unwrap();
        let input = futures_util::stream::empty::<
            Result<
                InvokeModelWithBidirectionalStreamInput,
                InvokeModelWithBidirectionalStreamInputError,
            >,
        >();
        let mut output = sdk
            .invoke_model_with_bidirectional_stream()
            .model_id(configuration.model_id)
            .body(input.into())
            .send()
            .await
            .unwrap();
        for _ in 0..3 {
            assert!(
                output.body.recv().await.unwrap().is_none(),
                "EOF must remain terminal"
            );
        }
        server.abort();
    }

    #[tokio::test]
    async fn native_private_literal_is_rejected_before_dialing_and_never_notifies_success() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (socket, _client, server) = super::super::tests::session_socket_pair().await;
        let error = adapter(&base, "amazon.nova-sonic-v1:0")
            .connect_and_proxy(socket, false, SessionMeter::default(), |result| {
                assert!(result.is_err())
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Network(_)));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_err()
        );
        server.abort();
    }

    #[tokio::test]
    async fn native_modeled_stream_errors_retain_their_status_after_a_real_http_success() {
        for (kind, status, original) in [
            ("throttlingException", 429, None),
            ("validationException", 400, None),
            ("modelTimeoutException", 408, None),
            ("serviceUnavailableException", 503, None),
            ("internalServerException", 500, None),
            ("modelStreamErrorException", 503, Some(503)),
            ("modelStreamErrorException", 424, None),
            ("modelStreamErrorException", 424, Some(-1)),
        ] {
            let peer = http2_peer(move |request| async move {
                drain_request_body(request.into_body());
                let bytes = exception_frame(
                    kind,
                    json!({"message":"provider detail","originalStatusCode":original}),
                );
                let body = BodyExt::boxed(
                    http_body_util::Full::new(bytes).map_err(|never| match never {}),
                );
                axum::http::Response::builder()
                    .status(200)
                    .header("content-type", "application/vnd.amazon.eventstream")
                    .body(body)
                    .unwrap()
            })
            .await;
            let (socket, _client, server) = super::super::tests::session_socket_pair().await;
            let notifications = Arc::new(AtomicUsize::new(0));
            let count = notifications.clone();
            let outcome = tokio::time::timeout(
                Duration::from_secs(3),
                adapter(&peer.base, "amazon.nova-sonic-v1:0").connect_and_proxy(
                    socket,
                    true,
                    SessionMeter::default(),
                    move |result| {
                        assert!(result.is_ok());
                        count.fetch_add(1, Ordering::SeqCst);
                    },
                ),
            )
            .await
            .unwrap()
            .unwrap();
            assert!(
                matches!(outcome.end, SessionEnd::UpstreamError(Error::Upstream { status: actual, .. }) if actual == status),
                "{kind}: expected status {status}, got {:?}",
                outcome.end
            );
            assert_eq!(notifications.load(Ordering::SeqCst), 1);
            server.abort();
        }
    }

    #[tokio::test]
    async fn native_task_abort_cancels_stream_and_retains_meter_checkpoint() {
        let stopped = Arc::new(Notify::new());
        let peer_stopped = stopped.clone();
        let confirmed = Arc::new(Notify::new());
        let peer_confirmed = confirmed.clone();
        let peer = http2_peer(move |request| {
            let stopped = peer_stopped.clone();
            let confirmed = peer_confirmed.clone();
            async move {
                let (tx, rx) = mpsc::channel(4);
                tokio::spawn(async move {
                    for event in [
                        json!({"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT","contentId":"a"}}}),
                        json!({"event":{"textOutput":{"content":"hello","contentId":"a"}}}),
                    ] {
                        tx.send(Ok(Frame::data(nova_frame(event)))).await.unwrap();
                    }
                    let mut input = SignedInput::new(request);
                    while let Some(event) = input.next().await {
                        if event["event"].get("sessionStart").is_some() { confirmed.notify_one(); }
                    }
                    stopped.notify_one();
                    drop(tx);
                });
                axum::http::Response::builder().status(200)
                    .header("content-type", "application/vnd.amazon.eventstream")
                    .body(response_body(rx)).unwrap()
            }
        }).await;
        let (socket, mut client, server) = super::super::tests::session_socket_pair().await;
        let checkpoint = SessionMeter::default();
        let task = tokio::spawn(
            adapter(&peer.base, "amazon.nova-sonic-v1:0").connect_and_proxy(
                socket,
                true,
                checkpoint.clone(),
                |result| assert!(result.is_ok()),
            ),
        );
        receive_type(&mut client, "response.text.delta").await;
        // The peer's acknowledgement of this client event proves the pump
        // completed its delivered-output checkpoint before the hard abort.
        client
            .send(ClientMessage::Text(
                r#"{"type":"session.update","session":{}}"#.into(),
            ))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), confirmed.notified())
            .await
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(3), stopped.notified())
            .await
            .unwrap();
        assert!(checkpoint.aborted_outcome().unmetered_output);
        server.abort();
    }

    #[tokio::test]
    async fn native_response_header_timeout_notifies_the_transport_504_bucket() {
        let accepted = Arc::new(Notify::new());
        let peer_accepted = accepted.clone();
        let peer = http2_peer(move |request| {
            let accepted = peer_accepted.clone();
            async move {
                let _request = request;
                accepted.notify_one();
                futures_util::future::pending::<axum::http::Response<MockBody>>().await
            }
        })
        .await;
        let (socket, _client, server) = super::super::tests::session_socket_pair().await;
        let results = Arc::new(Mutex::new(Vec::new()));
        let received = results.clone();
        let task = tokio::spawn(
            adapter(&peer.base, "amazon.nova-sonic-v1:0").connect_and_proxy(
                socket,
                true,
                SessionMeter::default(),
                move |result| {
                    received
                        .lock()
                        .unwrap()
                        .push(result.err().unwrap().failure_status);
                },
            ),
        );
        tokio::time::timeout(Duration::from_secs(3), accepted.notified())
            .await
            .unwrap();
        // The real HTTP2 request has arrived; virtual time cannot race setup.
        tokio::time::pause();
        tokio::time::advance(super::super::UPSTREAM_CONNECT_TIMEOUT).await;
        assert!(matches!(
            task.await.unwrap().unwrap_err(),
            Error::Network(_)
        ));
        assert_eq!(*results.lock().unwrap(), vec![Some(504)]);
        server.abort();
    }

    #[test]
    fn native_defaults_are_https_and_frame_bounds_reject_huge_or_truncated_events() {
        let mut adapter = adapter(
            "https://bedrock-runtime-fips.eu-west-1.amazonaws.com/prefix/",
            "amazon.nova-sonic-v1:0",
        );
        let (endpoint, _) = native_endpoint(&adapter).unwrap();
        assert_eq!(
            endpoint,
            "https://bedrock-runtime-fips.eu-west-1.amazonaws.com/prefix/"
        );
        adapter.api_base = None;
        assert_eq!(
            native_endpoint(&adapter).unwrap().0,
            "https://bedrock-runtime.eu-west-1.amazonaws.com/"
        );
        let mut bounds = FrameBounds::default();
        let oversized = u32::try_from(EVENTSTREAM_FRAME_CAP + 1)
            .unwrap()
            .to_be_bytes();
        bounds.observe(&oversized[..3]).unwrap();
        assert!(bounds.observe(&oversized[3..]).is_err());
        let valid = nova_frame(json!({"event":{"usageEvent":{}}}));
        let mut bounds = FrameBounds::default();
        bounds.observe(&valid[..11]).unwrap();
        assert!(!bounds.complete());
        bounds.observe(&valid[11..]).unwrap();
        assert!(bounds.complete());
    }
}
