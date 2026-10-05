//! Realtime observations use the original deployment's stable router identity,
//! even when its wildcard model is resolved to a concrete upstream selector.

use std::sync::Arc;
use std::time::Duration;

use oxllm_core::{Deployment, Error};

use super::connect::ConnectError;
use super::session::{SessionEnd, SessionOutcome};
use crate::orchestration::chat::{HEALTH_CHECK_UNHEALTHY_TOTAL, allowed_fails_limit_reached};
use crate::state::AppState;

/// A synchronous owner for one least-busy slot. Drop works even after the
/// runtime is gone; asynchronous budget settlement never owns this gauge.
pub(super) struct RealtimeRoutingGuard {
    state: Arc<AppState>,
    pub(super) cooldown_key: String,
    pub(super) dep_key: String,
    model: String,
    connected: bool,
    failure_recorded: bool,
}

impl RealtimeRoutingGuard {
    pub(super) fn new(state: Arc<AppState>, deployment: &Deployment, model: &str) -> Self {
        let cooldown_key = state.router.cooldown_key(deployment);
        state.router.record_in_flight(&cooldown_key, 1);
        Self {
            state,
            cooldown_key,
            dep_key: oxllm_core::router::deployment_key(deployment),
            model: model.to_string(),
            connected: false,
            failure_recorded: false,
        }
    }

    pub(super) fn connected(&mut self, handshake_latency: Duration) {
        if self.connected {
            return;
        }
        self.connected = true;
        self.state
            .router
            .record_connection_success(&self.cooldown_key);
        // A healthy socket can stay open for minutes. Latency routing observes
        // only its handshake, while billing retains the full session duration.
        self.state.router.record_latency(
            &self.cooldown_key,
            u64::try_from(handshake_latency.as_millis()).unwrap_or(u64::MAX),
        );
    }

    pub(super) fn is_connected(&self) -> bool {
        self.connected
    }

    pub(super) fn connect_failed(&mut self, error: &ConnectError) {
        // An HTTP rejection still proves reachability, exactly like chat.
        if matches!(&error.error, Error::Upstream { .. }) {
            self.state
                .router
                .record_connection_success(&self.cooldown_key);
        }
        self.record_failure(&error.error, error.failure_status);
        self.failure_recorded = true;
    }

    pub(super) fn finish(&mut self, outcome: &oxllm_core::Result<SessionOutcome>) {
        if self.failure_recorded {
            return;
        }
        match outcome {
            Ok(SessionOutcome {
                end: SessionEnd::TaskAborted,
                ..
            }) => {}
            Ok(SessionOutcome {
                end: SessionEnd::UpstreamError(error),
                ..
            })
            | Err(error) => {
                let status = match error {
                    Error::Network(_) => Some(0),
                    Error::Upstream { status, .. } => Some(*status),
                    _ => None,
                };
                self.record_failure(error, status);
            }
            Ok(_) if self.connected => {
                // Success in the percentage window means a healthy terminal
                // session, not just 101 headers before a transport failure.
                self.state.router.clear_failure_counts(&self.cooldown_key);
            }
            Ok(_) => {}
        }
    }

    fn record_failure(&self, error: &Error, status: Option<u16>) {
        if let Error::Network(message) = error {
            if self
                .state
                .router
                .record_connection_failure(&self.cooldown_key, message)
            {
                ::metrics::counter!(
                    HEALTH_CHECK_UNHEALTHY_TOTAL,
                    "deployment" => self.dep_key.clone()
                )
                .increment(1);
            }
        }
        // Malformed client events and deterministic transform/config failures
        // say nothing about deployment availability. An explicit 400 policy
        // must not turn those errors into a way to cool every backend.
        let Some(status) = status.filter(|status| {
            *status == 0 || *status >= 500 || matches!(*status, 401 | 403 | 404 | 408 | 424 | 429)
        }) else {
            return;
        };
        if allowed_fails_limit_reached(&self.state, &self.cooldown_key, status, &self.model) {
            self.state
                .router
                .mark_cooldown_for(&self.cooldown_key, &self.model);
        }
    }
}

impl Drop for RealtimeRoutingGuard {
    fn drop(&mut self) {
        self.state.router.record_in_flight(&self.cooldown_key, -1);
    }
}

/// Resolve wildcards without changing the selected deployment's router key.
/// Provider prefixes are routing syntax; endpoint selectors receive bare ids.
pub(super) fn concrete_deployment(
    selected: &Deployment,
    requested: &str,
) -> Result<Deployment, crate::error::ApiError> {
    let mut concrete = selected.clone();
    let configured = selected.oxllm_params.model.as_str();
    if !configured.contains('*') {
        return Ok(concrete);
    }
    let (provider, model) = if configured == "*" {
        if requested.starts_with("arn:") {
            ("bedrock", requested)
        } else {
            requested.split_once('/').unwrap_or(("openai", requested))
        }
    } else {
        let provider = selected.provider_slug();
        let model = requested
            .strip_prefix(provider)
            .and_then(|rest| rest.strip_prefix('/'))
            .unwrap_or(requested);
        if model == requested && requested.contains('/') && !requested.starts_with("arn:") {
            return Err(crate::error::ApiError::bad_request(
                "realtime model provider does not match the selected deployment",
            ));
        }
        (provider, model)
    };
    if provider.is_empty()
        || model.trim().is_empty()
        || model.contains('*')
        || model.chars().any(char::is_control)
        || model
            .split(['/', '\\'])
            .any(|segment| segment == "." || segment == "..")
    {
        return Err(crate::error::ApiError::bad_request(
            "realtime requires a concrete model id without traversal segments",
        ));
    }
    concrete.oxllm_params.model = format!("{provider}/{model}");
    Ok(concrete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig};

    fn selected(model: &str) -> Deployment {
        Deployment {
            model_name: "*".into(),
            oxllm_params: oxllm_core::OxllmParams {
                model: model.into(),
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn wildcard_models_are_concrete_and_keep_provider_syntax_out_of_selectors() {
        let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/sonic";
        for (pattern, requested, provider, selector) in [
            (
                "openai/*",
                "openai/gpt-4o-realtime",
                "openai",
                "gpt-4o-realtime",
            ),
            (
                "openai/gpt-*",
                "gpt-4o-realtime",
                "openai",
                "gpt-4o-realtime",
            ),
            ("azure/*", "azure/deployment", "azure", "deployment"),
            ("xai/*", "xai/grok-realtime", "xai", "grok-realtime"),
            ("bedrock/*", arn, "bedrock", arn),
            ("*", "xai/grok-realtime", "xai", "grok-realtime"),
            ("*", arn, "bedrock", arn),
        ] {
            let selected = selected(pattern);
            let concrete = concrete_deployment(&selected, requested).unwrap();
            assert_eq!(concrete.provider_slug(), provider);
            assert_eq!(super::super::openai::bare_model(&concrete), selector);
            assert_eq!(concrete.model_name, selected.model_name);
            assert_eq!(
                selected.oxllm_params.model, pattern,
                "the selected deployment's router identity must not be mutated"
            );
        }
        let selected = selected("bedrock/*");
        let concrete = concrete_deployment(&selected, &format!("bedrock/{arn}")).unwrap();
        assert_eq!(super::super::openai::bare_model(&concrete), arn);
    }

    #[test]
    fn wildcard_selector_rejects_literal_patterns_cross_provider_prefixes_and_traversal() {
        for requested in [
            "*",
            "openai/*",
            "openai/",
            "anthropic/claude",
            "openai/../model",
            "openai/a\\..\\b",
            "openai/\u{0}",
        ] {
            let error = concrete_deployment(&selected("openai/*"), requested).unwrap_err();
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        }
        assert!(concrete_deployment(&selected("openai/*"), "openai/模型&trace=true").is_ok());
    }

    #[test]
    fn deterministic_bad_request_does_not_feed_health_or_explicit_failure_policies() {
        let mut state = AppState::new(ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .unwrap();
        state.router = Arc::new(oxllm_core::router::Router::new(
            oxllm_core::router::RouterConfig {
                allowed_fails: Some(0),
                allowed_fails_policy: Some(std::collections::HashMap::from([("400".into(), 0)])),
                enable_health_check_routing: true,
                health_check_failure_threshold: 1,
                ..Default::default()
            },
        ));
        let state = Arc::new(state);
        let selected = selected("openai/gpt-4o");
        let mut guard = RealtimeRoutingGuard::new(state.clone(), &selected, "gpt-4o");
        let key = guard.cooldown_key.clone();
        guard.connected(Duration::from_millis(7));
        guard.finish(&Ok(SessionOutcome {
            usage: None,
            unmetered_output: false,
            end: SessionEnd::UpstreamError(Error::BadRequest("unsupported event".into())),
        }));
        drop(guard);
        assert_eq!(state.router.usage_store().in_flight(&key), 0);
        assert_eq!(state.router.cooldown_store().outcome_counts(&key), (0, 0));
        assert_eq!(state.router.cooldown_store().failure_count(&key, 400), 0);
        assert!(!state.router.cooldown_store().is_cooling(&key));
        assert!(state.router.health_store().reason(&key).is_none());
        assert_eq!(state.router.latency_store().p50(&key), Some(7));
    }
}
