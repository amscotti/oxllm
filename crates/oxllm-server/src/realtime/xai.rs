//! xAI (Grok Voice) realtime provider — normalizes wire-format quirks.
//!
//! xAI's Grok Voice realtime API is OpenAI-compatible on the wire but emits a
//! handful of frames that crash generally-available OpenAI realtime clients.
//! [`XaiRealtimeProvider`] wraps the identity [`OpenAIRealtimeProvider`] (so
//! connection setup — URL + bearer auth — is identical) and rewrites frames
//! in both directions to patch the quirks:
//!
//! # Upstream → client (Grok → GA client)
//!
//! 1. **Drop `ping` events**: Grok emits `{"type":"ping"}` keep-alives that
//!    GA clients do not expect. These are dropped (returned as no frames).
//! 2. **Back-fill missing `part`**: Grok omits the `part` object on
//!    `response.content_part.added` / `response.content_part.done`, crashing
//!    strict GA clients (Pipecat) that unconditionally index into it. A cache
//!    keyed by `(response_id, item_id, content_index)` is seeded from
//!    `content_part.added`, updated from `output_text.done` /
//!    `output_audio_transcript.done`, and used to back-fill a missing `part`
//!    on `.added`/`.done` (defaulting to `{"type":"audio","transcript":""}`
//!    when no cached entry exists). This is the reference pass 1, ported from
//!    `llms/xai/realtime/transformation.py`.
//! 3. **Inject missing indices**: Grok omits `output_index` /
//!    `content_index` on streaming response events; GA clients index off
//!    these fields and panic when absent. Missing fields are defaulted to
//!    `0` based on the exact event-type sets defined by the reference
//!    `_EVENTS_NEEDING_OUTPUT_INDEX` / `_EVENTS_NEEDING_CONTENT_INDEX`.
//! 4. **Normalize usage objects**: Grok sends an empty usage object where
//!    the OpenAI format carries `null` or a populated object, and partial
//!    usage objects that lack the `*_token_details` subobjects crash strict
//!    GA clients. Empty usage is rewritten to `null` (on `response.created`)
//!    or the default GA shape (on `response.done`); non-empty usage is
//!    default-filled with `input_token_details` / `output_token_details`.
//! 5. **Fix `role: "tool"`**: Grok marks `conversation.item.added` items with
//!    `role:"tool"` where the OpenAI contract is `role:"assistant"`. The role
//!    is rewritten so tool-call flow clients route the item correctly.
//!
//! # Client → upstream (GA client → Grok)
//!
//! 6. **Default `create_response: true`**: Grok requires the
//!    `turn_detection.create_response` field to be present on `server_vad`
//!    session config; when a GA client omits it, Grok rejects the session
//!    update. The field is injected (defaulting to `true`) whenever a
//!    `server_vad` turn-detection block lacks it. (Applied symmetrically to
//!    upstream `session.created`/`session.updated` echoes as well, so the GA
//!    client always observes a complete config.)
//!
//! All transforms are best-effort: malformed JSON passes through unchanged so
//! a single bad frame never tears down the session.

use std::collections::HashMap;

use oxllm_core::RealtimeProvider;
use oxllm_core::deployment::Deployment;
use oxllm_core::error::Result;
use oxllm_core::hook::UsageSummary;
use serde_json::{Map, Value};

use super::openai::OpenAIRealtimeProvider;

/// Event types whose GA contract requires an `output_index` field. xAI omits
/// it on all of these, crashing strict GA clients (e.g. pipecat). Ported
/// verbatim from the reference `_EVENTS_NEEDING_OUTPUT_INDEX` frozenset
/// (`llms/xai/realtime/transformation.py`).
const EVENTS_NEEDING_OUTPUT_INDEX: &[&str] = &[
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.output_audio.delta",
    "response.output_audio.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
];

/// Event types whose GA contract requires a `content_index` field. A subset
/// of [`EVENTS_NEEDING_OUTPUT_INDEX`] — the output-item-level events
/// (`response.output_item.*`, `response.function_call_arguments.*`) carry
/// only `output_index`. Ported from the reference
/// `_EVENTS_NEEDING_CONTENT_INDEX`.
const EVENTS_NEEDING_CONTENT_INDEX: &[&str] = &[
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.output_audio.delta",
    "response.output_audio.done",
];

/// xAI Grok Voice realtime provider — identity connection setup plus
/// wire-format normalization for the quirks enumerated in the module docs.
///
/// Construct via [`XaiRealtimeProvider::new`]; selected for the `xai` provider
/// slug by `super::resolve_realtime_provider`.
pub struct XaiRealtimeProvider {
    /// Identity provider handling URL/auth derivation.
    inner: OpenAIRealtimeProvider,
    /// Cache of `part` objects keyed by `(response_id, item_id,
    /// content_index)`, used to back-fill a missing `part` on
    /// `response.content_part.added`/`.done` events. See
    /// [`XaiRealtimeProvider::normalize_content_part_events`]. One provider
    /// instance is created per realtime session, so the cache is naturally
    /// scoped to a single session; growth is capped at
    /// [`CONTENT_PART_CACHE_CAP`] entries (parts are read back at most once,
    /// so past the cap back-fill degrades to the default part rather than
    /// accumulating state for a talkative long session).
    content_parts: std::sync::Mutex<HashMap<ContentPartKey, Value>>,
}

/// Maximum number of content parts retained for back-fill. Sessions are
/// capped at 600 s, but a chatty session can still produce thousands of
/// distinct `(response_id, item_id, content_index)` triples; entries past
/// the cap are simply not retained (their `.done` events fall back to the
/// default part).
const CONTENT_PART_CACHE_CAP: usize = 1024;

/// Cache key for the content-part back-fill pass. Mirrors the reference
/// `_content_part_key` tuple: `(response_id, item_id, content_index)`.
type ContentPartKey = (Option<String>, Option<String>, u64);

impl XaiRealtimeProvider {
    /// Construct the xAI normalizing realtime provider.
    pub fn new() -> Self {
        Self {
            inner: OpenAIRealtimeProvider::new(),
            content_parts: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl Default for XaiRealtimeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl RealtimeProvider for XaiRealtimeProvider {
    fn transform_client_event(&self, event: &str) -> Result<Vec<String>> {
        let mut v: Value = match serde_json::from_str(event) {
            Ok(v) => v,
            // Malformed frame: pass through unchanged rather than tearing
            // down the session on one bad event.
            Err(_) => return Ok(vec![event.to_string()]),
        };
        let mut dirty = false;
        if let Some(obj) = v.as_object_mut() {
            dirty = inject_create_response_default(obj);
        }
        if dirty {
            Ok(vec![v.to_string()])
        } else {
            Ok(vec![event.to_string()])
        }
    }

    fn transform_upstream_event(&self, event: &str) -> Result<Vec<String>> {
        let mut v: Value = match serde_json::from_str(event) {
            Ok(v) => v,
            Err(_) => return Ok(vec![event.to_string()]),
        };

        // 1. Drop Grok ping keep-alives — GA clients do not expect them.
        if v.get("type").and_then(|t| t.as_str()) == Some("ping") {
            return Ok(Vec::new());
        }

        let mut dirty = false;
        if let Some(obj) = v.as_object_mut() {
            // 1. Back-fill missing `part` on content_part.* events (reference
            //    pass 1). Runs before index injection so the cache key uses
            //    the raw event's content_index (defaulting to 0), matching
            //    content-part cache-key behavior.
            if self.normalize_content_part_events(obj) {
                dirty = true;
            }

            // 2. Inject missing indices on streaming response events. Use
            //    exact event-type sets (matching the reference frozensets) rather
            //    than a substring heuristic: the heuristic missed non-delta
            //    events that still require indices (output_item.added/done,
            //    content_part.added/done, *.done) and would over-match any
            //    future event whose type happens to contain "delta".
            let event_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let needs_output = EVENTS_NEEDING_OUTPUT_INDEX.contains(&event_type);
            let needs_content = EVENTS_NEEDING_CONTENT_INDEX.contains(&event_type);
            if needs_output && !obj.contains_key("output_index") {
                obj.insert("output_index".to_string(), Value::from(0u64));
                dirty = true;
            }
            if needs_content && !obj.contains_key("content_index") {
                obj.insert("content_index".to_string(), Value::from(0u64));
                dirty = true;
            }

            // 3. Normalize empty `usage: {}` → `null` (top-level edge case).
            if let Some(usage) = obj.get_mut("usage") {
                if usage.as_object().map(|m| m.is_empty()).unwrap_or(false) {
                    *usage = Value::Null;
                    dirty = true;
                }
            }

            // 3b. Normalize `response.usage` on response.created/done into
            //     the full GA shape (default-fill *_token_details).
            if normalize_response_usage(obj) {
                dirty = true;
            }

            // 4. Fix `role:"tool"` → `"assistant"` on conversation.item.added.
            if obj.get("type").and_then(|t| t.as_str()) == Some("conversation.item.added") {
                if let Some(item) = obj.get_mut("item").and_then(|i| i.as_object_mut()) {
                    if item.get("role").and_then(|r| r.as_str()) == Some("tool") {
                        item.insert("role".to_string(), Value::from("assistant"));
                        dirty = true;
                    }
                }
            }

            // 5. Default create_response on server_vad (echo path).
            if inject_create_response_default(obj) {
                dirty = true;
            }
        }

        if dirty {
            Ok(vec![v.to_string()])
        } else {
            Ok(vec![event.to_string()])
        }
    }

    fn ws_url(&self, deployment: &Deployment) -> String {
        // xAI has its own realtime host. Without an explicit `api_base` the
        // OpenAI identity default would send xAI credentials to OpenAI's
        // endpoint, so fall back to xAI's documented realtime URL instead.
        if deployment.oxllm_params.api_base.is_none() {
            let base = super::scheme_rewrite_for_ws("https://api.x.ai/v1");
            return super::build_realtime_ws_url(
                &base,
                "/realtime",
                super::openai::bare_model(deployment),
            );
        }
        self.inner.ws_url(deployment)
    }

    fn ws_auth_headers(&self, deployment: &Deployment) -> Vec<(String, String)> {
        self.inner.ws_auth_headers(deployment)
    }

    /// Delegate usage extraction to the identity OpenAI provider. xAI's Grok
    /// Voice realtime API is OpenAI-wire-compatible and reports usage in the
    /// same `response.done` shape, so without this override every xAI
    /// realtime session would inherit the trait default (`None`) and settle
    /// as zero-cost.
    fn extract_usage(&self, upstream_event: &str) -> Option<UsageSummary> {
        self.inner.extract_usage(upstream_event)
    }
}

impl XaiRealtimeProvider {
    /// Reference pass 1: cache and back-fill `part` on `response.content_part.*`
    /// events. xAI omits `part` on `content_part.added`/`content_part.done`,
    /// crashing strict GA clients (Pipecat) that index into it unconditionally.
    ///
    /// The cache is keyed by `(response_id, item_id, content_index)` so a
    /// later `.done` can be back-filled from an earlier `.added`, or from the
    /// text/transcript captured on `output_text.done` /
    /// `output_audio_transcript.done`. Returns `true` when the event was
    /// mutated (a `part` was back-filled).
    fn normalize_content_part_events(&self, obj: &mut Map<String, Value>) -> bool {
        let event_type = obj
            .get("type")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());
        let key = content_part_key(obj);
        match event_type.as_deref() {
            Some("response.content_part.added") | Some("response.content_part.done") => {
                // Remember the part first (before any back-fill) so a present
                // part on `.added` seeds the cache for a later `.done`.
                remember_content_part(&self.content_parts, &key, obj);
                let needs_part = !matches!(obj.get("part"), Some(Value::Object(_)));
                if needs_part {
                    let part = resolve_content_part(&self.content_parts, &key, obj);
                    obj.insert("part".to_string(), part);
                    return true;
                }
                false
            }
            Some("response.output_text.done") => {
                let text = obj.get("text").cloned();
                update_content_part_field(&self.content_parts, &key, "text", "text", text);
                // The event itself is unchanged (only the cache is updated).
                false
            }
            Some("response.output_audio_transcript.done") => {
                let transcript = obj.get("transcript").cloned();
                update_content_part_field(
                    &self.content_parts,
                    &key,
                    "audio",
                    "transcript",
                    transcript,
                );
                false
            }
            _ => false,
        }
    }
}

/// Compute the content-part cache key for an event: `(response_id, item_id,
/// content_index)`. Mirrors the reference content-part cache key; `content_index`
/// defaults to 0 when absent (xAI omits it — index injection runs *after*
/// this pass).
fn content_part_key(obj: &Map<String, Value>) -> ContentPartKey {
    let response_id = obj
        .get("response_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let item_id = obj
        .get("item_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let content_index = obj
        .get("content_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    (response_id, item_id, content_index)
}

/// Seed the cache with a present `part` object (the reference
/// `_remember_content_part`). No-op when `part` is absent or not an object.
/// New keys are not retained once the cache is at
/// [`CONTENT_PART_CACHE_CAP`].
fn remember_content_part(
    cache: &std::sync::Mutex<HashMap<ContentPartKey, Value>>,
    key: &ContentPartKey,
    obj: &Map<String, Value>,
) {
    if matches!(obj.get("part"), Some(Value::Object(_))) {
        if let Ok(mut c) = cache.lock() {
            if let Some(part) = obj.get("part") {
                if !c.contains_key(key) && c.len() >= CONTENT_PART_CACHE_CAP {
                    return;
                }
                c.insert(key.clone(), part.clone());
            }
        }
    }
}

/// Resolve the `part` for an event: a present object part, else the cached
/// entry, else the `{"type":"audio","transcript":""}` default (the reference
/// `_resolve_content_part`).
fn resolve_content_part(
    cache: &std::sync::Mutex<HashMap<ContentPartKey, Value>>,
    key: &ContentPartKey,
    obj: &Map<String, Value>,
) -> Value {
    if matches!(obj.get("part"), Some(Value::Object(_))) {
        if let Some(part) = obj.get("part") {
            return part.clone();
        }
    }
    if let Ok(c) = cache.lock() {
        if let Some(cached) = c.get(key) {
            if cached.is_object() {
                return cached.clone();
            }
        }
    }
    serde_json::json!({"type": "audio", "transcript": ""})
}

/// Update a single field on the cached content-part (the reference
/// `_update_content_part_field`). Used by `output_text.done` /
/// `output_audio_transcript.done` to keep the cache fresh so a later
/// `content_part.done` missing `part` back-fills the latest value. If the
/// field value is `None`, the cache is left untouched. A missing cache entry
/// is seeded with `{"type": part_type, field: value}` — unless the cache is
/// at [`CONTENT_PART_CACHE_CAP`], in which case new keys are not retained.
fn update_content_part_field(
    cache: &std::sync::Mutex<HashMap<ContentPartKey, Value>>,
    key: &ContentPartKey,
    part_type: &str,
    field: &str,
    value: Option<Value>,
) {
    let Some(value) = value else {
        return;
    };
    let mut c = match cache.lock() {
        Ok(c) => c,
        Err(_) => return,
    };
    if !c.contains_key(key) && c.len() >= CONTENT_PART_CACHE_CAP {
        return;
    }
    let existing = c.get(key).cloned();
    let updated = match existing {
        Some(existing) if existing.is_object() => {
            let mut m = existing;
            if let Some(obj) = m.as_object_mut() {
                if !obj.contains_key("type") {
                    obj.insert("type".to_string(), Value::from(part_type));
                }
                obj.insert(field.to_string(), value);
            }
            m
        }
        _ => serde_json::json!({ "type": part_type, field: value }),
    };
    c.insert(key.clone(), updated);
}

/// Inject `create_response: true` into `server_vad` turn-detection blocks
/// that lack the field. Walks both the top-level `session.turn_detection`
/// nesting and the GA-nested `session.audio.input.turn_detection` form —
/// the latter is where GA-schema clients (per the reference
/// `_remap_beta_session_to_ga`) place the turn-detection block.
///
/// Returns `true` when any value was mutated, so callers can avoid
/// reserializing unchanged frames (notably high-frequency audio deltas).
fn inject_create_response_default(obj: &mut Map<String, Value>) -> bool {
    let Some(session) = obj.get_mut("session").and_then(|s| s.as_object_mut()) else {
        return false;
    };
    let mut dirty = false;
    if let Some(td) = session
        .get_mut("turn_detection")
        .and_then(|t| t.as_object_mut())
    {
        dirty |= ensure_server_vad_create_response(td);
    }
    if let Some(nested_td) = session
        .get_mut("audio")
        .and_then(|a| a.as_object_mut())
        .and_then(|a| a.get_mut("input"))
        .and_then(|i| i.as_object_mut())
        .and_then(|i| i.get_mut("turn_detection"))
        .and_then(|t| t.as_object_mut())
    {
        dirty |= ensure_server_vad_create_response(nested_td);
    }
    dirty
}

/// Ensure a `server_vad` turn-detection object carries `create_response: true`
/// when the field is absent. Returns `true` when the field was injected.
fn ensure_server_vad_create_response(td: &mut Map<String, Value>) -> bool {
    if td.get("type").and_then(|t| t.as_str()) == Some("server_vad")
        && !td.contains_key("create_response")
    {
        td.insert("create_response".to_string(), Value::Bool(true));
        return true;
    }
    false
}

/// Build the default GA-shaped token-details subobject.
fn default_token_details() -> Value {
    serde_json::json!({
        "cached_tokens": 0,
        "text_tokens": 0,
        "audio_tokens": 0,
    })
}

/// Coerce a usage object into the full OpenAI GA shape with default-filled
/// `input_token_details` / `output_token_details`. Mirrors the reference
/// `_normalize_usage`: existing detail subobjects are merged over the
/// defaults so partial objects (e.g. only `cached_tokens`) still carry the
/// full key set strict GA clients index into.
fn normalize_usage_shape(usage: &Map<String, Value>) -> Value {
    let mut normalized = serde_json::json!({
        "total_tokens": usage.get("total_tokens").and_then(|t| t.as_u64()).unwrap_or(0),
        "input_tokens": usage.get("input_tokens").and_then(|t| t.as_u64()).unwrap_or(0),
        "output_tokens": usage.get("output_tokens").and_then(|t| t.as_u64()).unwrap_or(0),
    });
    let obj = normalized.as_object_mut().expect("just constructed");
    for key in ["input_token_details", "output_token_details"] {
        let merged = match usage.get(key).and_then(|d| d.as_object()) {
            Some(existing) => {
                let mut m = default_token_details();
                if let Some(m_obj) = m.as_object_mut() {
                    for (k, v) in existing {
                        m_obj.insert(k.clone(), v.clone());
                    }
                }
                m
            }
            None => default_token_details(),
        };
        obj.insert(key.to_string(), merged);
    }
    normalized
}

/// Normalize `response.usage` on `response.created` / `response.done` events.
///
/// Mirrors the reference response-usage normalization: on `response.created`
/// an empty `{}` usage becomes `null`; on `response.done` an empty `{}`
/// becomes the default GA shape; any non-empty usage is default-filled with
/// `*_token_details`. Returns `true` when the event was mutated.
fn normalize_response_usage(obj: &mut Map<String, Value>) -> bool {
    let empty_as_null = match obj.get("type").and_then(|t| t.as_str()) {
        Some("response.created") => true,
        Some("response.done") => false,
        _ => return false,
    };
    let Some(response) = obj.get_mut("response").and_then(|r| r.as_object_mut()) else {
        return false;
    };
    let Some(usage_val) = response.get("usage").cloned() else {
        return false;
    };
    let Some(usage) = usage_val.as_object() else {
        return false;
    };
    let normalized = if usage.is_empty() {
        if empty_as_null {
            Value::Null
        } else {
            normalize_usage_shape(&serde_json::Map::new())
        }
    } else {
        normalize_usage_shape(usage)
    };
    response.insert("usage".to_string(), normalized);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;
    use serde_json::Map;

    fn dep(slug_model: &str) -> Deployment {
        Deployment {
            model_name: slug_model.to_string(),
            oxllm_params: OxllmParams {
                model: slug_model.to_string(),
                api_key: Some("sk-test".to_string()),
                extra: Map::new(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn parse_out(out: &[String]) -> Value {
        assert_eq!(out.len(), 1, "expected exactly one frame, got {out:?}");
        serde_json::from_str(&out[0]).expect("output is valid JSON")
    }

    fn upstream(p: &XaiRealtimeProvider, event: &str) -> Vec<String> {
        p.transform_upstream_event(event)
            .expect("xai transform never errors (malformed passes through)")
    }

    fn client(p: &XaiRealtimeProvider, event: &str) -> Vec<String> {
        p.transform_client_event(event)
            .expect("xai transform never errors (malformed passes through)")
    }

    // ---- upstream → client: drop ping -------------------------------------

    #[test]
    fn drops_ping_events() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(&p, r#"{"type":"ping"}"#);
        assert!(out.is_empty(), "ping must be dropped, got {out:?}");
    }

    #[test]
    fn keeps_non_ping_events() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(&p, r#"{"type":"response.output_audio.delta","delta":"aa"}"#);
        assert_eq!(out.len(), 1);
    }

    // ---- upstream → client: inject indices --------------------------------

    #[test]
    fn injects_missing_indices_on_delta_events() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(
            &p,
            r#"{"type":"response.output_audio.delta","delta":"AAAA"}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["output_index"], 0);
        assert_eq!(v["content_index"], 0);
    }

    /// HIGH #2: events that the reference `_EVENTS_NEEDING_OUTPUT_INDEX` covers
    /// but the old `contains("delta")` heuristic missed must now receive
    /// indices. Each of these would crash a strict GA client without the
    /// index.
    #[test]
    fn injects_indices_on_non_delta_events_in_litellm_sets() {
        let p = XaiRealtimeProvider::new();
        for event_type in [
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.output_audio.done",
            "response.function_call_arguments.done",
        ] {
            let event = format!(r#"{{"type":"{event_type}"}}"#);
            let v = parse_out(&upstream(&p, &event));
            assert_eq!(
                v["output_index"], 0,
                "{event_type} must receive output_index"
            );
        }
        for event_type in [
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.output_audio.done",
        ] {
            let event = format!(r#"{{"type":"{event_type}"}}"#);
            let v = parse_out(&upstream(&p, &event));
            assert_eq!(
                v["content_index"], 0,
                "{event_type} must receive content_index"
            );
        }
    }

    /// HIGH #2: output-item-level events that require only `output_index`
    /// (not `content_index`) must not get a spurious content_index.
    #[test]
    fn output_item_events_get_only_output_index() {
        let p = XaiRealtimeProvider::new();
        let v = parse_out(&upstream(&p, r#"{"type":"response.output_item.added"}"#));
        assert_eq!(v["output_index"], 0);
        assert!(
            v.get("content_index").is_none(),
            "output_item events must not receive content_index"
        );
    }

    /// HIGH #2: an event type not in the reference sets must not receive indices.
    #[test]
    fn does_not_inject_indices_on_unlisted_event_types() {
        let p = XaiRealtimeProvider::new();
        let v = parse_out(&upstream(&p, r#"{"type":"response.created"}"#));
        assert!(v.get("output_index").is_none());
        assert!(v.get("content_index").is_none());
    }

    #[test]
    fn preserves_existing_indices_on_delta_events() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(
            &p,
            r#"{"type":"response.output_audio.delta","output_index":3,"content_index":2,"delta":"AAAA"}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["output_index"], 3);
        assert_eq!(v["content_index"], 2);
    }

    // ---- upstream → client: normalize empty usage -------------------------

    #[test]
    fn normalizes_empty_usage_to_null() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(&p, r#"{"type":"response.done","usage":{}}"#);
        let v = parse_out(&out);
        assert!(
            v["usage"].is_null(),
            "usage must be null, got {}",
            v["usage"]
        );
    }

    #[test]
    fn preserves_populated_usage() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(
            &p,
            r#"{"type":"response.done","usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["usage"]["prompt_tokens"], 5);
        assert_eq!(v["usage"]["total_tokens"], 12);
    }

    /// LOW #12: a partial `response.usage` object (non-empty but missing
    /// `*_token_details`) must be default-filled so strict GA clients that
    /// unconditionally index into the details subobject do not crash.
    #[test]
    fn normalizes_partial_response_usage_with_token_details() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(
            &p,
            r#"{"type":"response.done","response":{"usage":{"input_tokens":5}}}"#,
        );
        let v = parse_out(&out);
        let usage = &v["response"]["usage"];
        assert_eq!(usage["input_tokens"], 5);
        assert_eq!(usage["total_tokens"], 0);
        assert_eq!(usage["output_tokens"], 0);
        assert_eq!(usage["input_token_details"]["cached_tokens"], 0);
        assert_eq!(usage["input_token_details"]["text_tokens"], 0);
        assert_eq!(usage["input_token_details"]["audio_tokens"], 0);
        assert_eq!(usage["output_token_details"]["cached_tokens"], 0);
        assert_eq!(usage["output_token_details"]["text_tokens"], 0);
        assert_eq!(usage["output_token_details"]["audio_tokens"], 0);
    }

    /// LOW #12: on `response.done`, an empty `response.usage: {}` becomes
    /// the default GA shape (not null) per the reference `empty_as_null=False`
    /// for the done event.
    #[test]
    fn normalizes_empty_response_usage_on_done_to_default_ga_shape() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(&p, r#"{"type":"response.done","response":{"usage":{}}}"#);
        let v = parse_out(&out);
        let usage = &v["response"]["usage"];
        assert!(
            usage.is_object(),
            "response.done empty usage must become default GA shape, got {usage}"
        );
        assert!(usage.get("input_token_details").is_some());
        assert!(usage.get("output_token_details").is_some());
    }

    /// LOW #12: on `response.created`, an empty `response.usage: {}` becomes
    /// `null` (usage is optional on created, per the reference `empty_as_null=True`).
    #[test]
    fn normalizes_empty_response_usage_on_created_to_null() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(&p, r#"{"type":"response.created","response":{"usage":{}}}"#);
        let v = parse_out(&out);
        assert!(
            v["response"]["usage"].is_null(),
            "response.created empty usage must be null"
        );
    }

    // ---- upstream → client: fix tool role ---------------------------------

    #[test]
    fn fixes_tool_role_on_conversation_item_added() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(
            &p,
            r#"{"type":"conversation.item.added","item":{"type":"message","role":"tool","content":[]}}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["item"]["role"], "assistant");
    }

    #[test]
    fn leaves_assistant_role_unchanged() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(
            &p,
            r#"{"type":"conversation.item.added","item":{"type":"message","role":"user","content":[]}}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["item"]["role"], "user");
    }

    // ---- client → upstream / upstream echo: default create_response --------

    #[test]
    fn injects_create_response_in_client_session_update() {
        let p = XaiRealtimeProvider::new();
        let out = client(
            &p,
            r#"{"type":"session.update","session":{"turn_detection":{"type":"server_vad","threshold":0.5}}}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["session"]["turn_detection"]["create_response"], true);
        // Existing fields preserved.
        assert_eq!(v["session"]["turn_detection"]["threshold"], 0.5);
    }

    #[test]
    fn does_not_override_explicit_create_response() {
        let p = XaiRealtimeProvider::new();
        let out = client(
            &p,
            r#"{"type":"session.update","session":{"turn_detection":{"type":"server_vad","create_response":false}}}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["session"]["turn_detection"]["create_response"], false);
    }

    #[test]
    fn ignores_non_server_vad_turn_detection() {
        let p = XaiRealtimeProvider::new();
        let out = client(
            &p,
            r#"{"type":"session.update","session":{"turn_detection":{"type":"none"}}}"#,
        );
        let v = parse_out(&out);
        assert!(v["session"]["turn_detection"]["create_response"].is_null());
    }

    /// HIGH #3: GA-schema clients place turn_detection under
    /// `session.audio.input.turn_detection` (per the reference
    /// `_remap_beta_session_to_ga`). The injection must descend into this
    /// nested form as well as the top-level form.
    #[test]
    fn injects_create_response_in_nested_audio_input_turn_detection() {
        let p = XaiRealtimeProvider::new();
        let out = client(
            &p,
            r#"{"type":"session.update","session":{"audio":{"input":{"turn_detection":{"type":"server_vad","threshold":0.5}}}}}"#,
        );
        let v = parse_out(&out);
        assert_eq!(
            v["session"]["audio"]["input"]["turn_detection"]["create_response"],
            true
        );
        assert_eq!(
            v["session"]["audio"]["input"]["turn_detection"]["threshold"],
            0.5
        );
    }

    /// HIGH #3: both top-level and nested turn_detection blocks should be
    /// injected when both are present.
    #[test]
    fn injects_create_response_in_both_top_level_and_nested_turn_detection() {
        let p = XaiRealtimeProvider::new();
        let out = client(
            &p,
            r#"{"type":"session.update","session":{"turn_detection":{"type":"server_vad"},"audio":{"input":{"turn_detection":{"type":"server_vad"}}}}}"#,
        );
        let v = parse_out(&out);
        assert_eq!(v["session"]["turn_detection"]["create_response"], true);
        assert_eq!(
            v["session"]["audio"]["input"]["turn_detection"]["create_response"],
            true
        );
    }

    // ---- robustness --------------------------------------------------------

    #[test]
    fn malformed_upstream_json_passes_through() {
        let p = XaiRealtimeProvider::new();
        let out = upstream(&p, "not-json{{{");
        assert_eq!(out, vec!["not-json{{{".to_string()]);
    }

    #[test]
    fn malformed_client_json_passes_through() {
        let p = XaiRealtimeProvider::new();
        let out = client(&p, "oops");
        assert_eq!(out, vec!["oops".to_string()]);
    }

    // ---- connection setup: xAI realtime host vs explicit api_base ----------

    /// Without an explicit `api_base`, xAI must default to its own realtime
    /// host — inheriting the OpenAI identity default would send xAI
    /// credentials to OpenAI's endpoint.
    #[test]
    fn ws_url_defaults_to_xai_realtime_host() {
        let p = XaiRealtimeProvider::new();
        let url = p.ws_url(&dep("xai/grok-voice-latest"));
        assert!(
            url.starts_with("wss://api.x.ai/v1/realtime"),
            "expected xAI realtime host, got {url}"
        );
        assert!(url.contains("model=grok-voice-latest"), "got {url}");
    }

    /// An explicit `api_base` always wins (custom gateways / proxies).
    #[test]
    fn ws_url_honours_explicit_api_base() {
        let p = XaiRealtimeProvider::new();
        let mut deployment = dep("xai/grok-voice-latest");
        deployment.oxllm_params.api_base = Some("https://realtime.internal/v1".to_string());
        let url = p.ws_url(&deployment);
        assert!(
            url.starts_with("wss://realtime.internal/v1/realtime"),
            "expected the configured base, got {url}"
        );
    }

    #[test]
    fn ws_auth_headers_use_bearer_token() {
        let p = XaiRealtimeProvider::new();
        let headers = p.ws_auth_headers(&dep("xai/grok-realtime"));
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "Authorization");
        assert!(headers[0].1.contains("sk-test"));
    }

    /// LOW: xAI must delegate usage extraction to the identity OpenAI provider
    /// (its realtime API is wire-compatible). Without the override the trait
    /// default returns `None` and every session settles as zero-cost.
    #[test]
    fn extract_usage_delegates_to_openai_and_parses_response_done() {
        let p = XaiRealtimeProvider::new();
        let event = r#"{"type":"response.done","response":{"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12,"input_tokens_details":{"cached_tokens":2,"audio_tokens":3},"output_tokens_details":{"reasoning_tokens":1}}}}"#;
        let u = p.extract_usage(event).expect("xai must surface usage");
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 7);
        assert_eq!(u.total_tokens, 12);
        assert_eq!(u.cache_read_input_tokens, Some(2));
        assert_eq!(u.audio_input_tokens, Some(3));
        assert_eq!(u.reasoning_tokens, Some(1));
        // Events without usage surface as None (the inherited default shape).
        assert!(
            p.extract_usage(r#"{"type":"response.audio.delta","delta":"x"}"#)
                .is_none()
        );
        // LOW #8: null usage (common on response.created) must not produce a
        // zero summary — propagated via the delegated OpenAI extract_usage.
        assert!(
            p.extract_usage(r#"{"type":"response.created","response":{"usage":null}}"#)
                .is_none()
        );
        // LOW (C10): xAI's empty `usage: {}` quirk on response.created is
        // absence, not a zero report — must surface as None via the delegate.
        assert!(
            p.extract_usage(r#"{"type":"response.created","response":{"usage":{}}}"#)
                .is_none()
        );
    }

    // ---- upstream → client: back-fill missing `part` (MEDIUM #2) -----------

    /// MEDIUM: a `response.content_part.done` event that omits `part` must
    /// be back-filled. With no prior `.added` in this fresh provider, the
    /// default `{"type":"audio","transcript":""}` is used.
    #[test]
    fn backfills_missing_part_with_default_when_no_cached_entry() {
        let p = XaiRealtimeProvider::new();
        let v = parse_out(&upstream(
            &p,
            r#"{"type":"response.content_part.done","item_id":"i1","content_index":0}"#,
        ));
        let part = &v["part"];
        assert_eq!(part["type"], "audio");
        assert_eq!(part["transcript"], "");
    }

    /// MEDIUM: a `response.content_part.added` carrying `part` seeds the
    /// cache; a subsequent `response.content_part.done` (same key) missing
    /// `part` must back-fill from the cached value.
    #[test]
    fn backfills_part_from_prior_content_part_added() {
        let p = XaiRealtimeProvider::new();
        let added = r#"{"type":"response.content_part.added","response_id":"r1","item_id":"i1","content_index":0,"part":{"type":"audio","audio":"base64data"}}"#;
        // Drive the `.added` through the transform so the cache is seeded.
        let _ = upstream(&p, added);
        // `.done` omits `part` — must back-fill from the cached `.added` part.
        let v = parse_out(&upstream(
            &p,
            r#"{"type":"response.content_part.done","response_id":"r1","item_id":"i1","content_index":0}"#,
        ));
        assert_eq!(v["part"]["type"], "audio");
        assert_eq!(v["part"]["audio"], "base64data");
    }

    /// MEDIUM: `response.output_audio_transcript.done` updates the cache with
    /// its transcript; a later `content_part.done` missing `part` back-fills
    /// the cached part with the transcript merged in.
    #[test]
    fn backfills_part_merged_from_transcript_done() {
        let p = XaiRealtimeProvider::new();
        // Seed the cache from content_part.added with an audio part.
        let _ = upstream(
            &p,
            r#"{"type":"response.content_part.added","response_id":"r1","item_id":"i1","content_index":0,"part":{"type":"audio","audio":"aa"}}"#,
        );
        // Update the cached part's transcript via output_audio_transcript.done.
        let _ = upstream(
            &p,
            r#"{"type":"response.output_audio_transcript.done","response_id":"r1","item_id":"i1","content_index":0,"transcript":"hello"}"#,
        );
        // content_part.done missing part back-fills the cached, merged part.
        let v = parse_out(&upstream(
            &p,
            r#"{"type":"response.content_part.done","response_id":"r1","item_id":"i1","content_index":0}"#,
        ));
        assert_eq!(v["part"]["type"], "audio");
        assert_eq!(v["part"]["audio"], "aa");
        assert_eq!(v["part"]["transcript"], "hello");
    }

    /// MEDIUM: a `content_part.added` that already carries `part` is left
    /// unchanged (no spurious back-fill), and the cache is seeded.
    #[test]
    fn leaves_present_part_on_content_part_added_unchanged() {
        let p = XaiRealtimeProvider::new();
        let added = r#"{"type":"response.content_part.added","item_id":"i1","content_index":0,"part":{"type":"text","text":"hi"}}"#;
        let v = parse_out(&upstream(&p, added));
        assert_eq!(v["part"]["type"], "text");
        assert_eq!(v["part"]["text"], "hi");
    }

    /// LOW (C10): the content-part back-fill cache is capped — a talkative
    /// session with more distinct `(response_id, item_id, content_index)`
    /// triples than the cap must not grow the map without bound, and `.done`
    /// events for unretained keys still pass through (back-fill degrades to
    /// the default part).
    #[test]
    fn content_part_cache_is_capped_and_degrades_gracefully() {
        let p = XaiRealtimeProvider::new();
        for i in 0..(CONTENT_PART_CACHE_CAP + 10) {
            let added = format!(
                r#"{{"type":"response.content_part.added","response_id":"r{i}","item_id":"i","content_index":0,"part":{{"type":"audio","audio":"chunk{i}"}}}}"#
            );
            let _ = upstream(&p, &added);
        }
        let len = p.content_parts.lock().expect("cache lock").len();
        assert!(
            len <= CONTENT_PART_CACHE_CAP,
            "cache must stay within the cap, got {len}"
        );
        // A `.done` for a key retained early in the session still back-fills.
        let v = parse_out(&upstream(
            &p,
            r#"{"type":"response.content_part.done","response_id":"r0","item_id":"i","content_index":0}"#,
        ));
        assert_eq!(v["part"]["audio"], "chunk0");
        // A `.done` for a never-retained key (past the cap) degrades to the
        // default part instead of failing the transform.
        let v = parse_out(&upstream(
            &p,
            r#"{"type":"response.content_part.done","response_id":"r-uncached","item_id":"i","content_index":0}"#,
        ));
        assert_eq!(v["part"]["type"], "audio");
        assert_eq!(v["part"]["transcript"], "");
    }
}
