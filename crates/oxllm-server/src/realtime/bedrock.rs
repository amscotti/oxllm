//! Bedrock Nova Sonic session adapter.
//!
//! The pure event transforms live in
//! [`oxllm_providers::bedrock::BedrockRealtimeTransform`]. That transform owns
//! the OpenAI ⇄ Nova Sonic event mapping but cannot, on its own, talk to
//! Bedrock realtime, because a live session needs two host-level facilities
//! the [`RealtimeProvider`] trait cannot express:
//!
//! 1. **Signed HTTP/2 duplex I/O** for the native
//!    `POST /model/{modelId}/invoke-with-bidirectional-stream` operation.
//! 2. **AWS eventstream framing and chained signatures** for the streaming
//!    request body, plus typed eventstream response decoding.
//!
//! [`BedrockRealtimeAdapter`] closes both gaps. It wraps a
//! [`BedrockRealtimeTransform`] with the AWS credential material and owns the
//! full connection lifecycle. Default and HTTP(S) endpoints use the official
//! AWS SDK through `super::bedrock_http`. Only an explicitly configured
//! `ws://` / `wss://` endpoint selects the compatible WebSocket bridge below;
//! the native AWS runtime has no WebSocket GET upgrade endpoint.
//!
//! # Wire shape
//!
//! Client WebSocket text is translated into Nova events. The SDK envelopes and
//! signs these events on native HTTP/2; the explicit bridge uses binary WS
//! eventstream frames. Both transports feed decoded Nova JSON through the same
//! pure transform and shared session meter. Raw client binary audio is rejected;
//! OpenAI-compatible clients use `input_audio_buffer.append` text events.

use axum::extract::ws::{Message as AxumMessage, WebSocket};
use futures_util::{SinkExt, StreamExt};
use oxllm_core::Deployment;
use oxllm_core::RealtimeProvider;
use oxllm_providers::bedrock::{
    BedrockRealtimeTransform, decode_eventstream_frame, encode_eventstream_frame,
};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message as TungsteniteMessage;

use super::session::{SessionEnd, SessionMeter, SessionOutcome};
use crate::bedrock_sigv4::{
    presign_bedrock_ws_url, presign_bedrock_ws_url_for_base, validate_aws_region,
};
use crate::credentials::bedrock::{EnvLookup, resolve_aws_material};
use crate::error::ApiError;

/// Maximum bytes accumulated in the Bedrock upstream→client deframing buffer.
///
/// Each WS message is individually capped by tungstenite's default, but a
/// frame that advertises a large `total_len` and never delivers the body
/// would otherwise grow `buf` without bound. Tear down the session when the
/// cap is exceeded (16 MiB comfortably exceeds any legitimate eventstream
/// frame while bounding worker memory).
const UPSTREAM_BUFFER_CAP: usize = 16 * 1024 * 1024;

/// Whether the deframing buffer has grown past the safety cap. Extracted as a
/// pure helper so the tear-down threshold is unit-testable without live
/// sockets (the bedrock pump needs SigV4 + eventstream plumbing to drive
/// end-to-end). Returns `true` when the session must be torn down to avoid
/// unbounded memory growth from a malicious / buggy upstream.
pub(super) fn upstream_buffer_exceeds_cap(len: usize) -> bool {
    len > UPSTREAM_BUFFER_CAP
}

/// Send a `1011` (Internal Error) close frame to the client on a Bedrock
/// upstream setup failure (presign / request / connect / timeout). This lets
/// the client distinguish an upstream failure from a normal session end (code
/// 1000) instead of being misled by an abrupt socket drop. See
/// [`super::upstream_failure_close`].
async fn close_on_failure(socket: &mut WebSocket, reason: &'static str) {
    super::send_client(socket, super::upstream_failure_close(reason)).await;
}

/// Bedrock realtime session adapter.
///
/// Wraps a [`BedrockRealtimeTransform`] with the AWS credential material
/// required to sign the native HTTP/2 request (or an explicit WebSocket bridge),
/// plus the model id / region the request is scoped to. Construct one per session via
/// [`BedrockRealtimeAdapter::from_deployment`], then drive it with
/// `BedrockRealtimeAdapter::connect_and_proxy`.
///
/// Implements [`RealtimeProvider`] by delegating the pure event transforms to
/// the inner [`BedrockRealtimeTransform`]; this keeps it usable anywhere a
/// `dyn RealtimeProvider` is expected (e.g. the resolver's unit tests) while
/// the live framing/signing layer lives in `connect_and_proxy`.
pub struct BedrockRealtimeAdapter {
    pub(super) transform: BedrockRealtimeTransform,
    pub(super) access_key: String,
    pub(super) secret_key: String,
    pub(super) session_token: Option<String>,
    pub(super) region: String,
    pub(super) model_id: String,
    pub(super) api_base: Option<String>,
}

impl BedrockRealtimeAdapter {
    /// Construct an adapter from a deployment, extracting AWS credentials.
    ///
    /// Credential resolution is delegated to
    /// `crate::credentials::bedrock::resolve_aws_material` so the
    /// typed → extra → env credential precedence and region validation are
    /// shared with [`crate::credentials::BedrockCredentialResolver`]. Explicit
    /// `aws_region_name` / `aws_region` wins; recognizable runtime/FIPS/VPC
    /// endpoints supply the region before env/default fallback. Custom proxy
    /// hosts retain the resolver's configured/env region. Model ids have their
    /// provider prefix and throughput/context suffixes stripped.
    ///
    /// Returns [`ApiError::bad_request`] (HTTP 400) when the required access /
    /// secret keys are missing or the region is invalid — both are admin
    /// configuration errors, not internal server failures. Realtime cannot
    /// fall back to an unsigned URL.
    pub fn from_deployment(deployment: &Deployment) -> Result<Self, ApiError> {
        // Production reads the ambient AWS environment; unit tests inject a
        // lookup so they never inspect or mutate the process credentials.
        let env_lookup: EnvLookup = std::sync::Arc::new(|name: &'static str| {
            std::env::var(name).ok().filter(|s| !s.is_empty())
        });
        Self::from_deployment_with_env(deployment, &env_lookup)
    }

    fn from_deployment_with_env(
        deployment: &Deployment,
        env_lookup: &EnvLookup,
    ) -> Result<Self, ApiError> {
        let mut effective = deployment.clone();
        let explicit_region = ["aws_region_name", "aws_region"].iter().any(|key| {
            deployment
                .oxllm_params
                .extra
                .get(*key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| !value.is_empty())
        });
        if !explicit_region {
            if let Some(region) = deployment
                .oxllm_params
                .api_base
                .as_deref()
                .and_then(runtime_endpoint_region)
            {
                effective
                    .oxllm_params
                    .extra
                    .insert("aws_region_name".into(), serde_json::Value::String(region));
            }
        }
        let material = resolve_aws_material(&effective, env_lookup)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;

        let configured = &deployment.oxllm_params.model;
        let bare = configured
            .strip_prefix("bedrock/")
            .or_else(|| configured.strip_prefix("bedrock_mantle/"))
            .unwrap_or(configured);
        let model_id = oxllm_providers::bedrock::strip_model_suffixes(
            &oxllm_providers::bedrock::strip_model_suffixes(bare),
        );
        if model_id.trim().is_empty()
            || model_id.contains('*')
            || model_id.chars().any(char::is_control)
            || model_id
                .split(['/', '\\'])
                .any(|segment| segment == "." || segment == "..")
        {
            return Err(ApiError::bad_request(
                "Bedrock realtime requires a concrete model id without traversal segments",
            ));
        }

        Ok(Self {
            transform: BedrockRealtimeTransform::new(&model_id),
            access_key: material.access_key,
            secret_key: material.secret_key,
            session_token: material.session_token,
            region: material.region,
            model_id,
            api_base: deployment.oxllm_params.api_base.clone(),
        })
    }

    fn presigned_url(&self) -> Result<String, String> {
        if self.api_base.is_none() {
            return presign_bedrock_ws_url(
                &self.access_key,
                &self.secret_key,
                self.session_token.as_deref(),
                &self.region,
                &self.model_id,
            );
        }
        presign_bedrock_ws_url_for_base(
            &self.access_key,
            &self.secret_key,
            self.session_token.as_deref(),
            &self.region,
            &self.model_id,
            self.api_base.as_deref(),
        )
    }

    /// Open a signed native HTTP/2 duplex request, or an explicit WS(S) bridge,
    /// and run the eventstream pump until either side closes.
    ///
    /// Setup errors are returned separately from a connected, control-only
    /// session so the caller runs failure hooks, rather than recording success.
    /// `on_connect` runs at the actual handshake result, before pumping frames;
    /// this keeps reachability and handshake latency independent of session age.
    pub(super) async fn connect_and_proxy(
        self,
        mut client: WebSocket,
        allow_private: bool,
        meter: SessionMeter,
        on_connect: impl FnOnce(Result<(), &super::connect::ConnectError>) + Send,
    ) -> oxllm_core::Result<SessionOutcome> {
        let websocket_bridge = self
            .api_base
            .as_deref()
            .and_then(|base| url::Url::parse(base).ok())
            .is_some_and(|url| matches!(url.scheme(), "ws" | "wss"));
        if !websocket_bridge {
            return super::bedrock_http::connect_and_proxy(
                self,
                client,
                allow_private,
                meter,
                on_connect,
            )
            .await;
        }
        let url = match self.presigned_url() {
            Ok(u) => u,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    model = %self.model_id,
                    "bedrock realtime: presign failed"
                );
                close_on_failure(&mut client, "bedrock presign failed").await;
                return Err(oxllm_core::Error::Config(
                    "bedrock realtime presign failed".into(),
                ));
            }
        };

        self.connect_url(client, &url, allow_private, meter, on_connect)
            .await
    }

    async fn connect_url(
        self,
        mut client: WebSocket,
        url: &str,
        allow_private: bool,
        meter: SessionMeter,
        on_connect: impl FnOnce(Result<(), &super::connect::ConnectError>) + Send,
    ) -> oxllm_core::Result<SessionOutcome> {
        let request = match url.into_client_request() {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = %e, "bedrock realtime: request build failed");
                close_on_failure(&mut client, "bedrock request build failed").await;
                return Err(oxllm_core::Error::Config(
                    "bedrock realtime request build failed".into(),
                ));
            }
        };

        tracing::info!(
            url = %crate::security::sanitize_signed_url(url),
            model = %self.model_id,
            "bedrock realtime: opening upstream"
        );

        let upstream = match super::connect::connect(request, allow_private).await {
            Ok(ws) => {
                on_connect(Ok(()));
                ws
            }
            Err(e) => {
                on_connect(Err(&e));
                tracing::warn!(error = %e, "bedrock realtime: upstream connect failed");
                close_on_failure(&mut client, "bedrock upstream connect failed").await;
                return Err(e.error);
            }
        };

        Ok(self.pump(client, upstream, meter).await)
    }

    /// Bidirectional pump with eventstream framing on the Bedrock side.
    ///
    /// On disconnect send Nova's documented contentEnd → promptEnd → sessionEnd
    /// sequence and drain for usage within the shared deadline. Manual
    /// response.cancel is explicitly unsupported; closing a session is distinct.
    async fn pump(
        self,
        mut client: WebSocket,
        mut upstream: super::UpstreamStream,
        mut meter: SessionMeter,
    ) -> SessionOutcome {
        let transform = self.transform;
        let mut decoder = UpstreamDecoder::default();
        let deadline = tokio::time::Instant::now() + super::REALTIME_MAX_SESSION_DURATION;
        let end = 'pump: loop {
            tokio::select! {
                msg = client.next() => match msg {
                    Some(Ok(AxumMessage::Text(text))) => {
                        match transform.transform_client_event(text.as_str()) {
                            Ok(events) => for event in events {
                                let frame = encode_eventstream_frame(event.as_bytes());
                                if !super::send_upstream(&mut upstream, TungsteniteMessage::Binary(frame.into())).await {
                                    break 'pump super::upstream_send_error();
                                }
                            },
                            Err(error) => {
                                if !super::send_client(&mut client, super::client_event_error(text.as_str(), &error)).await {
                                    break SessionEnd::ClientDisconnected;
                                }
                            }
                        }
                    }
                    Some(Ok(AxumMessage::Binary(_))) => {
                        let error = oxllm_core::Error::BadRequest("Nova Sonic audio must use input_audio_buffer.append".into());
                        if !super::send_client(&mut client, super::client_event_error("{}", &error)).await {
                            break SessionEnd::ClientDisconnected;
                        }
                    }
                    Some(Ok(AxumMessage::Ping(_) | AxumMessage::Pong(_))) => {}
                    _ => break SessionEnd::ClientDisconnected,
                },
                msg = upstream.next() => match msg {
                    Some(Ok(TungsteniteMessage::Close(close))) => {
                        let end = super::upstream_close_end(close.as_ref());
                        break if matches!(end, SessionEnd::UpstreamError(_)) { end } else { decoder.end() };
                    }
                    None => break decoder.end(),
                    Some(Err(_)) => break SessionEnd::UpstreamError(oxllm_core::Error::Network("bedrock realtime upstream read failed".into())),
                    Some(Ok(message)) => {
                        let texts = match decoder.decode(message) {
                            Ok(texts) => texts,
                            Err(error) => break SessionEnd::UpstreamError(error),
                        };
                        if let Err(end) = deliver_upstream_texts(&transform, &mut meter, &mut client, &texts).await {
                            break 'pump end;
                        }
                    }
                },
                _ = tokio::time::sleep_until(deadline) => break SessionEnd::TimedOut,
            }
        };
        meter.ending(&end);
        if end.should_drain() {
            let _ = tokio::time::timeout(super::USAGE_DRAIN_TIMEOUT, async {
                for event in transform.shutdown_events() {
                    let frame = encode_eventstream_frame(event.as_bytes());
                    if upstream
                        .send(TungsteniteMessage::Binary(frame.into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                while let Some(Ok(message)) = upstream.next().await {
                    if matches!(message, TungsteniteMessage::Close(_)) {
                        break;
                    }
                    let Ok(texts) = decoder.decode(message) else {
                        break;
                    };
                    let mut recovered = false;
                    for text in texts {
                        recovered |= meter.observe(&transform, &text);
                    }
                    if recovered {
                        break;
                    }
                }
            })
            .await;
        }
        super::close_sockets(&mut client, &mut upstream, &end).await;
        meter.finish(end)
    }
}

/// Infer only from known AWS runtime endpoint shapes. An arbitrary proxy host
/// containing a region-looking label is not an AWS signing-region authority.
fn runtime_endpoint_region(base: &str) -> Option<String> {
    let parsed = url::Url::parse(base).ok()?;
    let labels: Vec<_> = parsed
        .host_str()?
        .trim_end_matches('.')
        .split('.')
        .collect();
    let is_runtime = |label: &str| matches!(label, "bedrock-runtime" | "bedrock-runtime-fips");
    let service = match labels.as_slice() {
        [first, ..] if is_runtime(first) => 0,
        [vpc, service, ..] if vpc.starts_with("vpce-") && is_runtime(service) => 1,
        _ => return None,
    };
    let region = *labels.get(service + 1)?;
    let suffix = labels.get(service + 2..)?.join(".");
    let recognized = if service == 1 {
        matches!(
            suffix.as_str(),
            "vpce.amazonaws.com" | "vpce.amazonaws.com.cn"
        )
    } else {
        matches!(
            suffix.as_str(),
            "amazonaws.com" | "amazonaws.com.cn" | "api.aws"
        )
    };
    if !recognized || validate_aws_region(region).is_err() {
        return None;
    }
    Some(region.to_string())
}

pub(super) async fn deliver_upstream_texts(
    transform: &BedrockRealtimeTransform,
    meter: &mut SessionMeter,
    client: &mut WebSocket,
    texts: &[String],
) -> Result<(), SessionEnd> {
    for (index, text) in texts.iter().enumerate() {
        meter.observe(transform, text);
        let events = transform
            .transform_upstream_event(text)
            .map_err(SessionEnd::UpstreamError)?;
        for event in events {
            if !super::send_client(client, AxumMessage::Text(event.clone().into())).await {
                // These frames were already read from the socket; retain
                // trailing usage in the same WS message when delivery fails.
                for remaining in &texts[index + 1..] {
                    meter.observe(transform, remaining);
                }
                return Err(SessionEnd::ClientDisconnected);
            }
            meter.delivered_text(&event);
        }
    }
    Ok(())
}

#[derive(Default)]
struct UpstreamDecoder {
    buffer: Vec<u8>,
}

impl UpstreamDecoder {
    fn end(&self) -> SessionEnd {
        if self.buffer.is_empty() {
            SessionEnd::UpstreamClosed
        } else {
            SessionEnd::UpstreamError(oxllm_core::Error::Network(
                "bedrock realtime ended with an incomplete eventstream frame".into(),
            ))
        }
    }

    fn decode(&mut self, message: TungsteniteMessage) -> oxllm_core::Result<Vec<String>> {
        let bytes = match message {
            TungsteniteMessage::Text(text) => return Ok(vec![text.to_string()]),
            TungsteniteMessage::Binary(bytes) => bytes,
            _ => return Ok(Vec::new()),
        };
        if upstream_buffer_exceeds_cap(self.buffer.len().saturating_add(bytes.len())) {
            return Err(oxllm_core::Error::Network(
                "bedrock realtime frame buffer exceeded cap".into(),
            ));
        }
        self.buffer.extend_from_slice(&bytes);
        let mut texts = Vec::new();
        while let Some((consumed, payload)) = decode_eventstream_frame(&self.buffer) {
            let text = std::str::from_utf8(payload)
                .map_err(|_| {
                    oxllm_core::Error::Serialization(
                        "invalid Bedrock realtime UTF-8 payload".into(),
                    )
                })?
                .to_string();
            self.buffer.drain(..consumed);
            texts.push(text);
        }
        Ok(texts)
    }
}

impl RealtimeProvider for BedrockRealtimeAdapter {
    fn transform_client_event(&self, event: &str) -> oxllm_core::error::Result<Vec<String>> {
        self.transform.transform_client_event(event)
    }

    fn transform_upstream_event(&self, event: &str) -> oxllm_core::error::Result<Vec<String>> {
        self.transform.transform_upstream_event(event)
    }

    fn extract_usage(&self, upstream_event: &str) -> Option<oxllm_core::UsageSummary> {
        self.transform.extract_usage(upstream_event)
    }

    fn ws_url(&self, deployment: &Deployment) -> String {
        self.transform.ws_url(deployment)
    }

    fn ws_auth_headers(&self, deployment: &Deployment) -> Vec<(String, String)> {
        self.transform.ws_auth_headers(deployment)
    }
}

/// Manual `Debug` — the wrapped [`BedrockRealtimeTransform`] holds a `Mutex`
/// over a non-`Debug` session state and so cannot derive `Debug`. We surface
/// the model id / region (useful in logs) and redact the credential fields.
impl std::fmt::Debug for BedrockRealtimeAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockRealtimeAdapter")
            .field("model_id", &self.model_id)
            .field("region", &self.region)
            .field("api_base", &self.api_base.as_ref().map(|_| "<redacted>"))
            .field("access_key", &"<redacted>")
            .field("secret_key", &"<redacted>")
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;

    fn deployment(model: &str) -> Deployment {
        Deployment {
            model_name: model.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn deployment_with_creds(model: &str) -> Deployment {
        let mut dep = deployment(model);
        dep.oxllm_params.aws_access_key_id = Some("AKIATEST".to_string());
        dep.oxllm_params.aws_secret_access_key = Some("secret".to_string());
        dep.oxllm_params.aws_session_token = Some("tok".to_string());
        dep.oxllm_params
            .extra
            .insert("aws_region".to_string(), serde_json::json!("eu-west-1"));
        dep
    }

    #[test]
    fn from_deployment_extracts_creds_and_strips_model_prefix() {
        let dep = deployment_with_creds("bedrock/amazon.nova-sonic-v1:0");
        let adapter = BedrockRealtimeAdapter::from_deployment(&dep).expect("creds present");
        assert_eq!(adapter.access_key, "AKIATEST");
        assert_eq!(adapter.secret_key, "secret");
        assert_eq!(adapter.session_token.as_deref(), Some("tok"));
        assert_eq!(adapter.region, "eu-west-1");
        assert_eq!(adapter.model_id, "amazon.nova-sonic-v1:0");
    }

    #[test]
    fn from_deployment_strips_bedrock_mantle_prefix() {
        let dep = deployment_with_creds("bedrock_mantle/amazon.nova-sonic-v1:0");
        let adapter = BedrockRealtimeAdapter::from_deployment(&dep).expect("creds present");
        assert_eq!(adapter.model_id, "amazon.nova-sonic-v1:0");
    }

    #[test]
    fn presigning_retains_custom_host_port_prefix_and_proxy_query() {
        let env: EnvLookup = std::sync::Arc::new(|_| None);
        let mut dep = deployment_with_creds("bedrock/amazon.nova-sonic-v1:0:51k[1m]");
        dep.oxllm_params.api_base =
            Some("https://proxy.example:9443/tenant/bedrock/?route=sonic".into());
        let adapter = BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env).unwrap();
        let url = url::Url::parse(&adapter.presigned_url().unwrap()).unwrap();
        assert_eq!(url.scheme(), "wss");
        assert_eq!(url.host_str(), Some("proxy.example"));
        assert_eq!(url.port(), Some(9443));
        assert_eq!(
            url.path(),
            "/tenant/bedrock/model/amazon.nova-sonic-v1%3A0/invoke-with-bidirectional-stream"
        );
        let query: std::collections::HashMap<_, _> = url.query_pairs().collect();
        assert_eq!(
            query.get("route").map(|value| value.as_ref()),
            Some("sonic")
        );
        assert_eq!(
            query
                .get("X-Amz-Security-Token")
                .map(|value| value.as_ref()),
            Some("tok")
        );
        assert!(query["X-Amz-Credential"].contains("/eu-west-1/bedrock/aws4_request"));
        assert_eq!(query["X-Amz-Signature"].len(), 64);
    }

    #[test]
    fn runtime_fips_and_vpc_hosts_infer_region_ahead_of_environment() {
        let env: EnvLookup =
            std::sync::Arc::new(|name| (name == "AWS_REGION_NAME").then(|| "ap-south-1".into()));
        for (base, expected_region) in [
            (
                "https://bedrock-runtime.eu-west-2.amazonaws.com",
                "eu-west-2",
            ),
            (
                "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com:9443/prefix",
                "us-gov-west-1",
            ),
            (
                "https://vpce-123-abc.bedrock-runtime.ap-northeast-1.vpce.amazonaws.com/gateway/",
                "ap-northeast-1",
            ),
            (
                "https://bedrock-runtime.cn-north-1.amazonaws.com.cn",
                "cn-north-1",
            ),
            ("https://bedrock-runtime.us-east-2.api.aws", "us-east-2"),
        ] {
            let mut dep = deployment_with_creds("bedrock/amazon.nova-sonic-v1:0");
            dep.oxllm_params.extra.remove("aws_region");
            dep.oxllm_params.api_base = Some(base.into());
            let adapter = BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env).unwrap();
            assert_eq!(adapter.region, expected_region, "{base}");
            let url = url::Url::parse(&adapter.presigned_url().unwrap()).unwrap();
            let base = url::Url::parse(base).unwrap();
            assert_eq!(url.host_str(), base.host_str());
            assert_eq!(url.port(), base.port());
            assert!(url.path().starts_with(base.path().trim_end_matches('/')));
            assert!(url.query_pairs().any(|(name, value)| {
                name == "X-Amz-Credential"
                    && value.contains(&format!("/{expected_region}/bedrock/aws4_request"))
            }));
        }
    }

    #[test]
    fn explicit_region_precedence_and_credential_precedence_survive_endpoint_inference() {
        let env: EnvLookup = std::sync::Arc::new(|name| match name {
            "AWS_ACCESS_KEY_ID" => Some("env-access".into()),
            "AWS_SECRET_ACCESS_KEY" => Some("env-secret".into()),
            "AWS_SESSION_TOKEN" => Some("env-token".into()),
            "AWS_REGION" => Some("ap-south-1".into()),
            _ => None,
        });
        let mut dep = deployment_with_creds("bedrock_mantle/amazon.nova-sonic-v1:0");
        dep.oxllm_params.api_base =
            Some("https://bedrock-runtime-fips.us-east-1.amazonaws.com".into());
        dep.oxllm_params
            .extra
            .insert("aws_region_name".into(), "eu-central-1".into());
        dep.oxllm_params
            .extra
            .insert("aws_access_key_id".into(), "extra-access".into());
        dep.oxllm_params
            .extra
            .insert("aws_secret_access_key".into(), "extra-secret".into());
        dep.oxllm_params
            .extra
            .insert("aws_session_token".into(), "extra-token".into());
        let adapter = BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env).unwrap();
        assert_eq!(adapter.region, "eu-central-1");
        assert_eq!(adapter.access_key, "AKIATEST");
        assert_eq!(adapter.secret_key, "secret");
        assert_eq!(adapter.session_token.as_deref(), Some("tok"));

        dep.oxllm_params.extra.remove("aws_region_name");
        dep.oxllm_params.aws_access_key_id = None;
        dep.oxllm_params.aws_secret_access_key = None;
        dep.oxllm_params.aws_session_token = None;
        let adapter = BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env).unwrap();
        assert_eq!(
            adapter.region, "eu-west-1",
            "aws_region overrides a FIPS hostname"
        );
        assert_eq!(adapter.access_key, "extra-access");
        assert_eq!(adapter.secret_key, "extra-secret");
        assert_eq!(adapter.session_token.as_deref(), Some("extra-token"));

        dep.oxllm_params
            .extra
            .insert("aws_region_name".into(), "us-east-1 ".into());
        assert!(BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env).is_err());
    }

    #[test]
    fn arbitrary_proxy_hosts_keep_the_configured_or_environment_region() {
        let env: EnvLookup =
            std::sync::Arc::new(|name| (name == "AWS_REGION").then(|| "ap-south-1".into()));
        for base in [
            "https://proxy.example/eu-west-2",
            "https://bedrock-runtime.eu-west-2.amazonaws.com.proxy.example",
            "https://proxy.example/bedrock-runtime.eu-west-2.amazonaws.com",
        ] {
            let mut dep = deployment_with_creds("bedrock/amazon.nova-sonic-v1:0");
            dep.oxllm_params.api_base = Some(base.into());
            assert_eq!(
                BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env)
                    .unwrap()
                    .region,
                "eu-west-1"
            );
            dep.oxllm_params.extra.remove("aws_region");
            assert_eq!(
                BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env)
                    .unwrap()
                    .region,
                "ap-south-1"
            );
        }
    }

    #[test]
    fn presigning_encodes_an_arn_as_one_wire_segment_after_the_base_prefix() {
        let env: EnvLookup = std::sync::Arc::new(|_| None);
        let arn = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/sonic";
        for model in [
            arn.to_string(),
            format!("bedrock/{arn}"),
            format!("bedrock_mantle/{arn}"),
        ] {
            let mut dep = deployment_with_creds(&model);
            dep.oxllm_params.api_base = Some("https://proxy.example:8443/aws/".into());
            let adapter = BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env).unwrap();
            assert_eq!(adapter.model_id, arn);
            let url = url::Url::parse(&adapter.presigned_url().unwrap()).unwrap();
            assert_eq!(
                url.path(),
                "/aws/model/arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Aapplication-inference-profile%2Fsonic/invoke-with-bidirectional-stream"
            );
            assert_eq!(url.path_segments().unwrap().count(), 4);
        }
    }

    #[test]
    fn unresolved_or_traversing_models_are_rejected_before_presigning() {
        let env: EnvLookup = std::sync::Arc::new(|_| None);
        for model in [
            "bedrock/*",
            "bedrock/",
            "bedrock/..",
            "bedrock/a/../b",
            "bedrock/a\\..\\b",
            "bedrock/a\u{0}b",
        ] {
            let error = BedrockRealtimeAdapter::from_deployment_with_env(
                &deployment_with_creds(model),
                &env,
            )
            .unwrap_err();
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn from_deployment_errors_when_creds_missing() {
        let dep = deployment("bedrock/amazon.nova-sonic-v1:0");
        let env: EnvLookup = std::sync::Arc::new(|_| None);
        let err = BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env)
            .expect_err("missing creds");
        assert!(err.message.contains("AWS_ACCESS_KEY_ID"));
        // MEDIUM #8: a missing credential is a config error (4xx), not an
        // internal server error (5xx).
        assert_eq!(
            err.status,
            axum::http::StatusCode::BAD_REQUEST,
            "missing AWS creds must be 4xx, not 5xx"
        );
    }

    #[test]
    fn adapter_delegates_transforms() {
        // The adapter must satisfy `resolve_realtime_provider`'s contract:
        // session.update expands into multiple Nova Sonic events.
        let dep = deployment_with_creds("bedrock/amazon.nova-sonic-v1:0");
        let adapter = BedrockRealtimeAdapter::from_deployment(&dep).expect("creds");
        let out = adapter
            .transform_client_event(r#"{"type":"session.update","session":{}}"#)
            .expect("adapter delegates");
        assert!(
            out.len() > 1,
            "adapter must delegate to the translating transform (expand)"
        );
    }

    #[test]
    fn adapter_framing_round_trips_through_encode_decode() {
        // A Nova Sonic event produced by the transform, framed, then
        // deframed, must yield the original JSON payload.
        let event = r#"{"event":{"sessionStart":{"inferenceConfiguration":{"maxTokens":1024}}}}"#;
        let frame = encode_eventstream_frame(event.as_bytes());
        let (consumed, payload) = decode_eventstream_frame(&frame).expect("decode");
        assert_eq!(consumed, frame.len());
        assert_eq!(payload, event.as_bytes());
    }

    #[test]
    fn upstream_decoder_admits_exactly_the_cap_and_tears_down_past_it() {
        // Build a real eventstream frame whose declared total_len is one byte
        // over UPSTREAM_BUFFER_CAP, then deliver it in pieces. The declaration
        // is untrusted (it is just a length field), so the decoder buffers
        // until the accumulated bytes cross the cap rather than rejecting the
        // frame up front.
        let oversized = encode_eventstream_frame(&vec![0u8; UPSTREAM_BUFFER_CAP + 1 - 16]);
        assert_eq!(
            u32::from_be_bytes(oversized[..4].try_into().unwrap()) as usize,
            UPSTREAM_BUFFER_CAP + 1,
            "fixture must declare a frame past the cap"
        );

        let mut decoder = UpstreamDecoder::default();
        // Only the 12-byte prelude arrived: still buffered, nothing decoded.
        assert!(
            decoder
                .decode(TungsteniteMessage::Binary(oversized[..12].to_vec().into()))
                .unwrap()
                .is_empty()
        );
        // Filling to exactly the cap is admitted — the tear-down guard is a
        // strict `>` so a single maximum-size frame can complete.
        let fill = UPSTREAM_BUFFER_CAP - 12;
        assert!(
            decoder
                .decode(TungsteniteMessage::Binary(vec![0u8; fill].into()))
                .unwrap()
                .is_empty()
        );
        assert_eq!(decoder.buffer.len(), UPSTREAM_BUFFER_CAP);
        // One byte past the cap tears the session down with the cap error...
        let err = decoder
            .decode(TungsteniteMessage::Binary(vec![0u8; 1].into()))
            .expect_err("a buffer past the cap must be rejected");
        assert!(
            matches!(&err, oxllm_core::Error::Network(message) if message.contains("exceeded cap")),
            "unexpected cap error: {err:?}"
        );
        // ...and the already-buffered bytes are neither truncated nor decoded.
        assert_eq!(decoder.buffer.len(), UPSTREAM_BUFFER_CAP);
        assert!(matches!(decoder.end(), SessionEnd::UpstreamError(_)));
    }

    #[test]
    fn upstream_buffer_exceeds_cap_flags_oversized_buffer() {
        // The tear-down threshold: at/below the cap is fine, above tears down.
        assert!(!upstream_buffer_exceeds_cap(0));
        assert!(!upstream_buffer_exceeds_cap(UPSTREAM_BUFFER_CAP));
        assert!(upstream_buffer_exceeds_cap(UPSTREAM_BUFFER_CAP + 1));
        // A 1 GiB declared-but-undelivered frame would force the buffer past
        // the cap → the pump tears down instead of OOMing.
        assert!(upstream_buffer_exceeds_cap(1024 * 1024 * 1024));
    }

    #[test]
    fn declared_oversized_incomplete_frame_is_buffered_not_accepted() {
        // A frame whose first 4 bytes claim a 1 GiB total_len but whose body
        // is undelivered: the decoder must report an incomplete frame (None)
        // rather than accepting it. The actual oversized-frame guard is the
        // accumulated-byte cap exercised above; this test pins that an
        // oversized *declaration* alone neither decodes nor gets dropped —
        // the bytes stay buffered and end() flags the truncation.
        // Valid 12-byte AWS prelude; CRC32 independently computed with zlib.
        let buf = hex::decode("40000000000000004cbf1d84").unwrap();
        assert!(decode_eventstream_frame(&buf).is_none());
        let mut decoder = UpstreamDecoder::default();
        assert!(
            decoder
                .decode(TungsteniteMessage::Binary(buf.clone().into()))
                .unwrap()
                .is_empty()
        );
        assert_eq!(decoder.buffer, buf, "incomplete frame stays buffered");
        assert!(matches!(decoder.end(), SessionEnd::UpstreamError(_)));
    }

    #[test]
    fn corrupted_prelude_crc_is_rejected_and_never_advanced() {
        // A valid frame decodes; flipping the stored prelude CRC must make the
        // decoder refuse the (now untrusted) length fields. Without prelude
        // CRC validation the valid total_len would still decode the payload,
        // so this assertion fails on a regression that skips the check.
        let valid = encode_eventstream_frame(br#"{"event":{"usageEvent":{}}}"#);
        assert!(decode_eventstream_frame(&valid).is_some());
        let mut frame = valid.clone();
        frame[11] ^= 0xFF;
        assert!(
            decode_eventstream_frame(&frame).is_none(),
            "a corrupt prelude CRC must never yield a decoded frame"
        );
        let mut decoder = UpstreamDecoder::default();
        assert!(
            decoder
                .decode(TungsteniteMessage::Binary(frame.clone().into()))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            decoder.buffer, frame,
            "the untrusted frame must stay buffered, not be advanced past"
        );
        assert!(matches!(decoder.end(), SessionEnd::UpstreamError(_)));
    }

    #[test]
    fn corrupted_message_crc_is_dropped_without_forwarding_its_payload() {
        // A complete frame whose trailing CRC32 does not match was corrupted
        // in transit: the boundary is known, so it is advanced past with an
        // empty payload instead of being buffered forever.
        let mut frame = encode_eventstream_frame(br#"{"event":{"usageEvent":{}}}"#);
        let last = frame.len() - 1;
        frame[last] ^= 0xFF;
        let (consumed, payload) =
            decode_eventstream_frame(&frame).expect("complete corrupt frame advances");
        assert_eq!(consumed, frame.len());
        assert!(payload.is_empty(), "corrupt payload must be dropped");

        let mut decoder = UpstreamDecoder::default();
        let texts = decoder
            .decode(TungsteniteMessage::Binary(frame.into()))
            .unwrap();
        assert_eq!(texts, vec![String::new()]);
        assert!(
            decoder.buffer.is_empty(),
            "corrupt frame must not re-buffer"
        );
        // An empty payload is not valid Nova JSON, so the corruption surfaces
        // as a session error instead of being forwarded to the client.
        assert!(
            BedrockRealtimeTransform::default()
                .transform_upstream_event("")
                .is_err()
        );
    }

    #[test]
    fn native_twelve_byte_prelude_survives_fragmented_websocket_messages() {
        // Independent Python struct/zlib fixture, not encoded by the decoder's
        // companion helper. Payload starts after the 12-byte prelude.
        let frame = hex::decode(concat!(
            "00000095000000007cc477097b226576656e74223a7b2275736167654576656e74223a7b",
            "2264657461696c73223a7b2264656c7461223a7b22696e707574223a7b227370656563",
            "68546f6b656e73223a322c2274657874546f6b656e73223a337d2c226f757470757422",
            "3a7b22737065656368546f6b656e73223a342c2274657874546f6b656e73223a317d7d",
            "7d7d7d7d4d230ce5"
        ))
        .unwrap();
        let mut decoder = UpstreamDecoder::default();
        assert!(
            decoder
                .decode(TungsteniteMessage::Binary(frame[..8].to_vec().into()))
                .unwrap()
                .is_empty()
        );
        assert!(matches!(decoder.end(), SessionEnd::UpstreamError(_)));
        assert!(
            decoder
                .decode(TungsteniteMessage::Binary(frame[8..12].to_vec().into()))
                .unwrap()
                .is_empty()
        );
        let texts = decoder
            .decode(TungsteniteMessage::Binary(frame[12..].to_vec().into()))
            .unwrap();
        assert_eq!(texts.len(), 1);
        let usage = BedrockRealtimeTransform::default()
            .extract_usage(&texts[0])
            .unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.audio_input_tokens, Some(2));
        assert!(matches!(decoder.end(), SessionEnd::UpstreamClosed));
    }

    fn test_adapter() -> BedrockRealtimeAdapter {
        let env: EnvLookup = std::sync::Arc::new(|_| None);
        BedrockRealtimeAdapter::from_deployment_with_env(
            &deployment_with_creds("bedrock/amazon.nova-sonic-v1:0"),
            &env,
        )
        .unwrap()
    }

    async fn client_socket_pair() -> (
        WebSocket,
        super::super::UpstreamStream,
        tokio::task::JoinHandle<()>,
    ) {
        use axum::extract::ws::WebSocketUpgrade;
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move |upgrade: WebSocketUpgrade| {
                let tx = tx.clone();
                async move {
                    upgrade.on_upgrade(move |socket| async move {
                        let sender = tx.lock().unwrap().take().unwrap();
                        let _ = sender.send(socket);
                    })
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let (client, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        (rx.await.unwrap(), client, server)
    }

    #[tokio::test]
    #[allow(clippy::result_large_err)] // tungstenite fixes the callback error type
    async fn configured_endpoint_and_real_handshake_notification_are_used_by_the_adapter() {
        use std::sync::Arc;
        use tokio::sync::{Notify, oneshot};
        use tokio_tungstenite::tungstenite::handshake::client::Request;

        let (socket, _client, server) = client_socket_pair().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let release = Arc::new(Notify::new());
        let peer_release = release.clone();
        let (uri_tx, uri_rx) = oneshot::channel();
        let peer = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut upstream =
                tokio_tungstenite::accept_hdr_async(stream, move |request: &Request, response| {
                    assert_eq!(
                        request.headers()["host"].to_str().unwrap(),
                        address.to_string()
                    );
                    let _ = uri_tx.send(request.uri().to_string());
                    Ok(response)
                })
                .await
                .unwrap();
            peer_release.notified().await;
            upstream.close(None).await.unwrap();
        });
        let mut dep = deployment_with_creds("bedrock/amazon.nova-sonic-v1:0:18k[200k]");
        dep.oxllm_params.api_base = Some(format!("ws://{address}/proxy/base/"));
        let env: EnvLookup = Arc::new(|_| None);
        let adapter = BedrockRealtimeAdapter::from_deployment_with_env(&dep, &env).unwrap();
        let (connected_tx, connected_rx) = oneshot::channel();
        let session = tokio::spawn(adapter.connect_and_proxy(
            socket,
            true,
            SessionMeter::default(),
            move |result| {
                let _ = connected_tx.send(result.is_ok());
            },
        ));
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(3), connected_rx)
                .await
                .unwrap()
                .unwrap()
        );
        assert!(
            !session.is_finished(),
            "notification must precede the healthy session's end"
        );
        let uri = uri_rx.await.unwrap();
        assert!(uri.starts_with(
            "/proxy/base/model/amazon.nova-sonic-v1%3A0/invoke-with-bidirectional-stream?"
        ));
        assert!(uri.contains("X-Amz-Signature="));
        release.notify_one();
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(3), session)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(outcome.end, SessionEnd::UpstreamClosed));
        peer.await.unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn setup_failures_return_errors_and_close_with_1011() {
        for case in ["presign", "request", "connect"] {
            let (socket, mut client, server) = client_socket_pair().await;
            let mut adapter = test_adapter();
            let notifications = std::sync::Mutex::new(Vec::new());
            let on_connect = |result: Result<(), &super::super::connect::ConnectError>| {
                notifications.lock().unwrap().push(result.is_ok());
            };
            let result = match case {
                "presign" => {
                    adapter.secret_key.clear();
                    adapter.api_base = Some("wss://bridge.example/".into());
                    adapter
                        .connect_and_proxy(socket, true, SessionMeter::default(), on_connect)
                        .await
                }
                "request" => {
                    adapter
                        .connect_url(socket, "ws://[", true, SessionMeter::default(), on_connect)
                        .await
                }
                _ => {
                    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                    let url = format!("ws://{}/", listener.local_addr().unwrap());
                    let peer = tokio::spawn(async move {
                        drop(listener.accept().await.unwrap());
                    });
                    let outcome = adapter
                        .connect_url(socket, &url, true, SessionMeter::default(), on_connect)
                        .await;
                    peer.await.unwrap();
                    outcome
                }
            };
            assert!(
                result.is_err(),
                "{case} failure must not be an unmetered success"
            );
            assert_eq!(
                *notifications.lock().unwrap(),
                if case == "connect" {
                    vec![false]
                } else {
                    vec![]
                },
                "only a real connection attempt can emit a handshake notification"
            );
            let frame = tokio::time::timeout(std::time::Duration::from_secs(2), client.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(
                matches!(frame, TungsteniteMessage::Close(Some(close)) if u16::from(close.code) == 1011)
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn unsupported_controls_are_returned_to_client_not_forwarded_upstream() {
        let (socket, mut client, server) = client_socket_pair().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/", listener.local_addr().unwrap());
        let peer = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let mut kinds = Vec::new();
            while let Some(Ok(TungsteniteMessage::Binary(frame))) = ws.next().await {
                let (size, payload) = decode_eventstream_frame(&frame).unwrap();
                assert_eq!(size, frame.len());
                let value: serde_json::Value = serde_json::from_slice(payload).unwrap();
                let kind = value["event"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .next()
                    .unwrap()
                    .clone();
                let ended = kind == "sessionEnd";
                kinds.push(kind);
                if ended {
                    ws.close(None).await.unwrap();
                    break;
                }
            }
            kinds
        });
        let (upstream, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        let pump = tokio::spawn(test_adapter().pump(socket, upstream, SessionMeter::default()));
        client
            .send(TungsteniteMessage::Text(
                r#"{"type":"session.update","session":{}}"#.into(),
            ))
            .await
            .unwrap();
        for (event, id) in [
            ("response.cancel", "cancel"),
            ("response.create", "text-only"),
        ] {
            client
                .send(TungsteniteMessage::Text(
                    serde_json::json!({"type":event,"event_id":id})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
            let message = tokio::time::timeout(std::time::Duration::from_secs(2), client.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let value: serde_json::Value =
                serde_json::from_str(message.to_text().unwrap()).unwrap();
            assert_eq!(value["type"], "error");
            assert_eq!(value["error"]["code"], "unsupported_event");
            assert_eq!(value["error"]["event_id"], id);
        }
        client.close(None).await.unwrap();
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), pump)
            .await
            .unwrap()
            .unwrap();
        assert!(!outcome.unmetered_output);
        assert!(outcome.usage.is_none());
        assert_eq!(
            peer.await.unwrap(),
            vec!["sessionStart", "promptStart", "promptEnd", "sessionEnd"]
        );
        server.abort();
    }

    #[tokio::test]
    async fn failed_client_delivery_retains_usage_already_decoded_from_the_same_message() {
        let (mut socket, _client, server) = client_socket_pair().await;
        // Force deterministic send-after-close, without a timing-sensitive TCP
        // race. The batch models two eventstream frames in one WS message.
        socket.send(AxumMessage::Close(None)).await.unwrap();
        let texts = vec![
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#.to_string(),
            r#"{"event":{"usageEvent":{"details":{"delta":{
                "input":{"speechTokens":2,"textTokens":3},
                "output":{"speechTokens":4,"textTokens":1}
            }}}}}"#
                .to_string(),
        ];
        let mut meter = SessionMeter::default();
        let end = deliver_upstream_texts(
            &BedrockRealtimeTransform::default(),
            &mut meter,
            &mut socket,
            &texts,
        )
        .await
        .unwrap_err();
        assert!(matches!(end, SessionEnd::ClientDisconnected));
        assert_eq!(meter.finish(end).usage.unwrap().total_tokens, 10);
        server.abort();
    }
}
