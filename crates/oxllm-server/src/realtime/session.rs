//! Realtime accounting separates control-only sessions from output awaiting
//! a provider usage report. Missing usage is never converted to invented tokens.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use oxllm_core::{Error, RealtimeProvider, UsageSummary};
use serde_json::Value;

#[derive(Clone, Debug)]
pub(super) enum SessionEnd {
    UpstreamClosed,
    ClientDisconnected,
    UpstreamError(Error),
    TimedOut,
    /// Infrastructure cancelled the owning task, rather than either peer.
    TaskAborted,
}

impl SessionEnd {
    pub(super) fn is_failure(&self) -> bool {
        matches!(self, Self::UpstreamError(_) | Self::TaskAborted)
    }

    pub(super) fn should_drain(&self) -> bool {
        matches!(self, Self::ClientDisconnected | Self::TimedOut)
    }
}

#[derive(Debug)]
pub(super) struct SessionOutcome {
    pub usage: Option<UsageSummary>,
    pub unmetered_output: bool,
    pub end: SessionEnd,
}

/// The pump and its settlement guard share the latest accounting checkpoint.
/// No lock is held across I/O, so dropping an aborted pump can recover partial
/// usage without guessing whether output was delivered.
#[derive(Clone, Default)]
pub(super) struct SessionMeter {
    inner: Arc<Mutex<SessionUsage>>,
}

impl SessionMeter {
    pub(super) fn observe(&mut self, provider: &dyn RealtimeProvider, text: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .observe(provider, text)
    }

    pub(super) fn delivered_text(&mut self, text: &str) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .delivered_text(text);
    }

    pub(super) fn delivered_binary(&mut self) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .delivered_binary();
    }

    pub(super) fn cancel_events(&self) -> Vec<String> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cancel_events()
    }

    pub(super) fn snapshot(&self, end: SessionEnd) -> SessionOutcome {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finish(end)
    }

    /// Preserve a peer's terminal cause before cancellable drain/close I/O.
    /// A hard abort during client-disconnect draining must still retain the
    /// client's unmetered-output estimate instead of refunding it as a fault.
    pub(super) fn ending(&self, end: &SessionEnd) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).end = Some(end.clone());
    }

    pub(super) fn aborted_outcome(&self) -> SessionOutcome {
        let usage = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        usage.finish(usage.end.clone().unwrap_or(SessionEnd::TaskAborted))
    }

    pub(super) fn finish(self, end: SessionEnd) -> SessionOutcome {
        self.snapshot(end)
    }
}

#[derive(Default)]
struct SessionUsage {
    total: Option<UsageSummary>,
    pending: HashSet<String>,
    active: HashSet<String>,
    tracking_saturated: bool,
    end: Option<SessionEnd>,
}

impl SessionUsage {
    /// Returns true when this frame reports usage for all previously delivered
    /// output. OpenAI reports per-response totals; Nova reports session deltas.
    pub(super) fn observe(&mut self, provider: &dyn RealtimeProvider, text: &str) -> bool {
        let event = serde_json::from_str::<Value>(text).ok();
        if let Some(event) = &event {
            if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                match event.get("type").and_then(Value::as_str) {
                    Some("response.created") if self.active.len() < 128 && id.len() <= 1024 => {
                        self.active.insert(id.to_string());
                    }
                    Some("response.done") => {
                        self.active.remove(id);
                    }
                    _ => {}
                }
            }
        }
        let Some(usage) = provider.extract_usage(text) else {
            return false;
        };
        match &mut self.total {
            Some(total) => total.merge(&usage),
            None => self.total = Some(usage),
        }
        if let Some(event) = event {
            if event.get("event").is_some() {
                // Nova usageEvent.details.delta covers the session's generated
                // output since the preceding usage event, across content blocks.
                self.pending.clear();
            } else {
                let id = event
                    .pointer("/response/id")
                    .or_else(|| event.get("response_id"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.pending.remove(id);
            }
        }
        self.pending.is_empty() && self.active.is_empty() && !self.tracking_saturated
    }

    pub(super) fn delivered_text(&mut self, text: &str) {
        let Ok(event) = serde_json::from_str::<Value>(text) else {
            return;
        };
        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
        let is_delta = matches!(
            kind,
            "response.text.delta"
                | "response.output_text.delta"
                | "response.audio.delta"
                | "response.output_audio.delta"
                | "response.audio_transcript.delta"
                | "response.output_audio_transcript.delta"
                | "response.function_call_arguments.delta"
        ) && event
            .get("delta")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty());
        let is_tool_start = kind == "response.output_item.added"
            && event.pointer("/item/type").and_then(Value::as_str) == Some("function_call")
            && event
                .pointer("/item/name")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty());
        if is_delta || is_tool_start {
            let id = event
                .get("response_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            self.mark_pending(id);
        }
    }

    pub(super) fn delivered_binary(&mut self) {
        self.mark_pending("<unattributed-binary>");
    }

    pub(super) fn cancel_events(&self) -> Vec<String> {
        let mut events = Vec::new();
        // A bare cancel targets only the default conversation. Also cancel
        // known out-of-band responses rather than leaving them generating.
        for id in self.pending.union(&self.active) {
            if !id.is_empty() && id != "<unattributed-binary>" {
                events.push(
                    serde_json::json!({"type":"response.cancel","response_id":id}).to_string(),
                );
            }
        }
        if events.is_empty()
            || self.pending.contains("")
            || self.pending.contains("<unattributed-binary>")
        {
            events.push(serde_json::json!({"type":"response.cancel"}).to_string());
        }
        events
    }

    fn mark_pending(&mut self, id: &str) {
        // Bound tracking memory for malformed compatible upstreams. Saturation
        // only makes accounting conservative; it does not reject any responses.
        if self.pending.contains(id) {
            return;
        }
        if self.pending.len() < 128 && id.len() <= 1024 {
            self.pending.insert(id.to_string());
        } else {
            self.tracking_saturated = true;
        }
    }

    fn finish(&self, end: SessionEnd) -> SessionOutcome {
        SessionOutcome {
            usage: self.total.clone(),
            unmetered_output: !self.pending.is_empty() || self.tracking_saturated,
            end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::OpenAIRealtimeProvider;

    #[test]
    fn control_events_do_not_make_an_idle_session_billable() {
        let mut meter = SessionMeter::default();
        meter.delivered_text(r#"{"type":"session.created","session":{"id":"s"}}"#);
        meter.delivered_text(r#"{"type":"response.created","response":{"id":"r"}}"#);
        let outcome = meter.finish(SessionEnd::ClientDisconnected);
        assert!(!outcome.unmetered_output);
        assert!(outcome.usage.is_none());
    }

    #[test]
    fn a_completed_response_does_not_hide_another_partial_response() {
        let mut meter = SessionMeter::default();
        for id in ["first", "second"] {
            meter.delivered_text(
                &serde_json::json!({
                    "type":"response.audio.delta","response_id":id,"delta":"AAAA"
                })
                .to_string(),
            );
        }
        assert!(!meter.observe(&OpenAIRealtimeProvider,
            r#"{"type":"response.done","response":{"id":"first","usage":{"input_tokens":2,"output_tokens":3}}}"#));
        let outcome = meter.finish(SessionEnd::ClientDisconnected);
        assert!(outcome.unmetered_output);
        assert_eq!(outcome.usage.unwrap().total_tokens, 5);
    }

    #[test]
    fn final_usage_after_disconnect_resolves_delivered_output() {
        let mut meter = SessionMeter::default();
        meter.delivered_text(r#"{"type":"response.text.delta","response_id":"r","delta":"hello"}"#);
        assert!(meter.observe(&OpenAIRealtimeProvider,
            r#"{"type":"response.done","response":{"id":"r","usage":{"input_tokens":2,"output_tokens":3}}}"#));
        let outcome = meter.finish(SessionEnd::ClientDisconnected);
        assert!(!outcome.unmetered_output);
        assert_eq!(outcome.usage.unwrap().total_tokens, 5);
    }

    #[test]
    fn tool_names_and_audio_transcripts_are_generated_output_too() {
        for event in [
            r#"{"type":"response.output_item.added","response_id":"r","item":{"type":"function_call","name":"lookup"}}"#,
            r#"{"type":"response.output_audio_transcript.delta","response_id":"r","delta":"hello"}"#,
        ] {
            let mut meter = SessionMeter::default();
            meter.delivered_text(event);
            assert!(
                meter
                    .finish(SessionEnd::ClientDisconnected)
                    .unmetered_output
            );
        }
    }
}
