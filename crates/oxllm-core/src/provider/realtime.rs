//! Realtime WebSocket provider trait.
//!
//! Realtime APIs (OpenAI Realtime, Bedrock Nova Sonic, …) are
//! **bidirectional** WebSocket sessions: the client streams audio/text in,
//! the provider streams audio/text + tool calls back. Unlike the
//! request/response provider traits in the parent module, a realtime
//! provider does not own any I/O — the host (oxllm-server) owns the two
//! WebSocket connections (client ↔ oxllm ↔ upstream) and calls into the
//! transform methods on every frame.
//!
//! # Wire format
//!
//! oxllm always exposes the **OpenAI Realtime API** event format to
//! clients. OpenAI-compatible providers use the identity implementation
//! (passthrough); other providers (Bedrock Nova Sonic) translate between
//! the OpenAI event format and their native event format in both
//! directions.
//!
//! Events are exchanged as **raw JSON strings** (one OpenAI realtime event
//! per WebSocket text frame). Realtime audio may also travel as binary
//! WebSocket frames; those are forwarded verbatim by the proxy (the trait
//! only transforms text/JSON frames — binary audio is provider-native PCM
//! and is not re-encoded here).
//!
//! # Purity
//!
//! All trait methods are **synchronous** — no network, no async, no I/O.
//! Providers may keep per-session transform state (one provider instance is
//! constructed per realtime session), e.g. the xAI normalizer's content-part
//! back-fill cache. This keeps the trait fully unit-testable in `oxllm-core`
//! without a live WebSocket server.

use crate::deployment::Deployment;
use crate::error::Result;
use crate::hook::UsageSummary;

/// A provider that can transform WebSocket realtime events.
///
/// OpenAI-compatible providers use the identity implementation (return the
/// input event unchanged). Providers with a different native wire format
/// (e.g. Bedrock Nova Sonic) translate events in both directions.
///
/// The host resolves a [`Deployment`] to a concrete `RealtimeProvider`,
/// opens the upstream WebSocket described by [`RealtimeProvider::ws_url`]
/// / [`RealtimeProvider::ws_auth_headers`], and pumps frames through the
/// transform pipeline.
///
/// # Why `Vec<String>`?
///
/// A single inbound event may map to **zero, one, or many** outbound
/// events:
/// - **zero**: the provider wants to drop/suppress an event (e.g. an
///   OpenAI-only control event the upstream does not understand);
/// - **one**: the common case (1:1 mapping or identity passthrough);
/// - **many**: the provider expands one OpenAI event into several native
///   events (e.g. splitting a multi-tool `response.create`).
///
/// # Metering
///
/// Realtime billing is genuinely hard (long-lived bidirectional sessions,
/// streamed audio), so realtime endpoints bypass the router budget hierarchy
/// by design. Metering is **provider-optional**: a provider that can extract
/// token usage from an upstream event (e.g. OpenAI Realtime's `response.done`
/// carries `usage`, Bedrock Nova Sonic emits a session summary) overrides
/// [`RealtimeProvider::extract_usage`] so a host hook *can* attribute spend
/// when the provider reports it. The default returns `None` (un-metered).
pub trait RealtimeProvider: Send + Sync {
    /// Transform a client→upstream event.
    ///
    /// `event` is the raw JSON string of an OpenAI realtime event received
    /// from the client WebSocket (e.g. `session.update`,
    /// `input_audio_buffer.append`). Returns the list of event JSON strings
    /// to forward to the upstream WebSocket — usually one (identity), but
    /// may be zero (drop) or many (expand).
    ///
    /// Errors are surfaced to the host (logged/metered) rather than silently
    /// dropped; the host decides whether to tear down the session. Identity
    /// providers never parse and thus never error — they pass every frame
    /// through, including malformed JSON.
    fn transform_client_event(&self, event: &str) -> Result<Vec<String>>;

    /// Transform an upstream→client event.
    ///
    /// `event` is the raw string received from the upstream WebSocket.
    /// Returns the list of OpenAI-format event JSON strings to forward back
    /// to the client WebSocket.
    ///
    /// Errors are surfaced to the host (logged/metered) rather than silently
    /// dropped. A translating provider that fails to parse an upstream frame
    /// returns `Error::Serialization` (or `Error::Transform`); an
    /// identity provider passes every frame through unchanged.
    fn transform_upstream_event(&self, event: &str) -> Result<Vec<String>>;

    /// Optionally extract token usage from an upstream event.
    ///
    /// Returns `Some(UsageSummary)` when the upstream event carries a usage
    /// report (e.g. OpenAI Realtime `response.done` with a `usage` object),
    /// enabling a host hook to meter/cost a realtime session. The default
    /// returns `None` — realtime metering is provider-optional (see the trait
    /// doc on [metering](RealtimeProvider#metering)).
    fn extract_usage(&self, _upstream_event: &str) -> Option<UsageSummary> {
        None
    }

    /// The WebSocket URL to connect to for this provider/deployment.
    ///
    /// The host connects to this URL with `tokio_tungstenite::connect_async`.
    /// Implementations should derive the URL purely from the deployment's
    /// `api_base` / `model` fields.
    fn ws_url(&self, deployment: &Deployment) -> String;

    /// Auth headers for the upstream WebSocket connection.
    ///
    /// Returns `(header_name, header_value)` pairs (e.g.
    /// `("Authorization", "Bearer sk-…")`). The host attaches these to the
    /// upstream WS upgrade request.
    ///
    /// # Credential-resolver caveat
    ///
    /// Unlike [`ChatProvider::apply_auth`](crate::provider::ChatProvider::apply_auth)
    /// / [`EmbeddingProvider::apply_auth`](crate::provider::EmbeddingProvider::apply_auth),
    /// this accessor receives only `&Deployment` — **not** host-resolved
    /// [`ResolvedCredentials`](crate::provider::ResolvedCredentials) — and so
    /// reads `deployment.oxllm_params.api_key` directly. This works for the
    /// OpenAI/Azure/xAI set whose key is statically resolved into the config
    /// field, but a provider needing request-time credential exchange (AWS
    /// STS, OAuth token refresh, SigV4 *query-string* URL signing) cannot be
    /// expressed through this return type. The host therefore short-circuits
    /// to a dedicated adapter before the resolver runs for such providers
    /// (e.g. the Bedrock realtime adapter), and a deployment whose `api_key`
    /// is resolver-only (empty in config) must fail closed rather than
    /// connect unauthenticated. A future realtime provider with dynamic
    /// credentials will need a `ws_auth(&self, deployment, creds)` (or a
    /// `ws_upgrade_sign` that can also mutate the URL) signature instead.
    fn ws_auth_headers(&self, deployment: &Deployment) -> Vec<(String, String)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    /// Minimal identity provider used to exercise the trait contract without
    /// depending on `oxllm-server`. The real identity implementation lives in
    /// `oxllm-server::realtime::openai::OpenAIRealtimeProvider`.
    struct IdentityProvider;

    impl RealtimeProvider for IdentityProvider {
        fn transform_client_event(&self, event: &str) -> Result<Vec<String>> {
            Ok(vec![event.to_string()])
        }
        fn transform_upstream_event(&self, event: &str) -> Result<Vec<String>> {
            Ok(vec![event.to_string()])
        }
        fn ws_url(&self, deployment: &Deployment) -> String {
            format!(
                "wss://example.com/realtime?model={}",
                deployment.oxllm_params.model
            )
        }
        fn ws_auth_headers(&self, deployment: &Deployment) -> Vec<(String, String)> {
            deployment
                .oxllm_params
                .api_key
                .as_deref()
                .map(|k| ("Authorization".to_string(), format!("Bearer {k}")))
                .into_iter()
                .collect()
        }
    }

    fn dep(model: &str, key: Option<&str>) -> Deployment {
        Deployment {
            model_name: model.to_string(),
            oxllm_params: crate::deployment::OxllmParams {
                model: model.to_string(),
                api_key: key.map(String::from),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn identity_client_event_passes_through_unchanged() {
        let p = IdentityProvider;
        let event = r#"{"type":"session.update","session":{"voice":"alloy"}}"#;
        let out = p
            .transform_client_event(event)
            .expect("identity never errors");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], event);
    }

    #[test]
    fn identity_upstream_event_passes_through_unchanged() {
        let p = IdentityProvider;
        let event = r#"{"type":"response.audio.delta","delta":"base64"}"#;
        let out = p
            .transform_upstream_event(event)
            .expect("identity never errors");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], event);
    }

    #[test]
    fn identity_passes_through_empty_and_binary_looking_strings() {
        // The identity transform is a pure string passthrough; it never parses
        // and thus never errors — malformed JSON passes through unchanged.
        let p = IdentityProvider;
        assert_eq!(
            p.transform_client_event("").expect("identity never errors"),
            vec!["".to_string()]
        );
        assert_eq!(
            p.transform_upstream_event("not-valid-json{{{")
                .expect("identity never errors"),
            vec!["not-valid-json{{{".to_string()]
        );
    }

    #[test]
    fn ws_url_and_auth_headers_derive_from_deployment() {
        let p = IdentityProvider;
        let d = dep("gpt-4o-realtime", Some("sk-test"));
        assert_eq!(
            p.ws_url(&d),
            "wss://example.com/realtime?model=gpt-4o-realtime"
        );
        let headers = p.ws_auth_headers(&d);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "Authorization");
        assert!(headers[0].1.contains("sk-test"));
    }

    #[test]
    fn ws_auth_headers_empty_when_no_api_key() {
        let p = IdentityProvider;
        let d = dep("gpt-4o-realtime", None);
        assert!(p.ws_auth_headers(&d).is_empty());
    }

    /// A transform provider that *expands* one event into many and *drops*
    /// others — exercises the `Vec<String>` return contract that the identity
    /// provider never stresses.
    #[test]
    fn transform_can_expand_and_drop_events() {
        struct Expander;
        impl RealtimeProvider for Expander {
            fn transform_client_event(&self, event: &str) -> Result<Vec<String>> {
                if event.contains("multi") {
                    Ok(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                } else if event.contains("drop") {
                    Ok(vec![])
                } else {
                    Ok(vec![event.to_string()])
                }
            }
            fn transform_upstream_event(&self, e: &str) -> Result<Vec<String>> {
                Ok(vec![e.to_string()])
            }
            fn ws_url(&self, _: &Deployment) -> String {
                "wss://x".to_string()
            }
            fn ws_auth_headers(&self, _: &Deployment) -> Vec<(String, String)> {
                vec![]
            }
        }
        let p = Expander;
        assert_eq!(
            p.transform_client_event(r#"{"type":"multi"}"#)
                .expect("expand never errors")
                .len(),
            3
        );
        assert!(
            p.transform_client_event(r#"{"type":"drop"}"#)
                .expect("expand never errors")
                .is_empty()
        );
        assert_eq!(
            p.transform_client_event("plain")
                .expect("expand never errors"),
            vec!["plain".to_string()]
        );
    }

    /// A *validating* provider surfaces a parse failure through the error
    /// channel instead of silently dropping/passthrough — the contract that
    /// the identity provider (which never parses) cannot demonstrate.
    #[test]
    fn validating_provider_surfaces_malformed_json_as_error() {
        struct Validator;
        impl RealtimeProvider for Validator {
            fn transform_client_event(&self, event: &str) -> Result<Vec<String>> {
                serde_json::from_str::<serde_json::Value>(event)
                    .map(|_| vec![event.to_string()])
                    .map_err(|e| Error::Serialization(format!("invalid realtime frame: {e}")))
            }
            fn transform_upstream_event(&self, event: &str) -> Result<Vec<String>> {
                serde_json::from_str::<serde_json::Value>(event)
                    .map(|_| vec![event.to_string()])
                    .map_err(|e| Error::Serialization(format!("invalid realtime frame: {e}")))
            }
            fn ws_url(&self, _: &Deployment) -> String {
                "wss://x".to_string()
            }
            fn ws_auth_headers(&self, _: &Deployment) -> Vec<(String, String)> {
                vec![]
            }
        }
        let p = Validator;
        // Valid JSON passes through.
        assert_eq!(
            p.transform_upstream_event(r#"{"type":"response.done"}"#)
                .expect("valid frame"),
            vec![r#"{"type":"response.done"}"#.to_string()]
        );
        // Malformed JSON is surfaced as an error, not silently passed through.
        let err = p
            .transform_upstream_event("not-valid-json{{{")
            .expect_err("malformed frame must error");
        assert!(
            matches!(err, Error::Serialization(_)),
            "expected Serialization error, got {err:?}"
        );
    }

    /// The default `extract_usage` is `None` (realtime metering is
    /// provider-optional); a provider can override it to surface usage.
    #[test]
    fn extract_usage_defaults_to_none_and_can_be_overridden() {
        // Default: identity provider reports no usage.
        let identity = IdentityProvider;
        assert!(
            identity
                .extract_usage(r#"{"type":"response.done"}"#)
                .is_none()
        );

        // A provider that parses `response.done` usage can surface it.
        struct UsageExtractor;
        impl RealtimeProvider for UsageExtractor {
            fn transform_client_event(&self, e: &str) -> Result<Vec<String>> {
                Ok(vec![e.to_string()])
            }
            fn transform_upstream_event(&self, e: &str) -> Result<Vec<String>> {
                Ok(vec![e.to_string()])
            }
            fn extract_usage(&self, upstream_event: &str) -> Option<UsageSummary> {
                let v: serde_json::Value = serde_json::from_str(upstream_event).ok()?;
                let usage = v.get("usage")?;
                Some(UsageSummary {
                    prompt_tokens: usage
                        .get("prompt_tokens")
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0),
                    completion_tokens: usage
                        .get("completion_tokens")
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0),
                    total_tokens: usage
                        .get("total_tokens")
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0),
                    ..Default::default()
                })
            }
            fn ws_url(&self, _: &Deployment) -> String {
                "wss://x".to_string()
            }
            fn ws_auth_headers(&self, _: &Deployment) -> Vec<(String, String)> {
                vec![]
            }
        }
        let p = UsageExtractor;
        // No usage object → None.
        assert!(
            p.extract_usage(r#"{"type":"response.audio.delta"}"#)
                .is_none()
        );
        // Usage object → parsed.
        let u = p
            .extract_usage(r#"{"type":"response.done","usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#)
            .expect("usage present");
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 7);
        assert_eq!(u.total_tokens, 12);
    }
}
