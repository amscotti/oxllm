//! Bedrock Nova Sonic realtime event transforms.
//!
//! Translates bidirectionally between the **OpenAI Realtime API** event
//! format (what the client speaks) and the **Bedrock Nova Sonic** event
//! format (carried by native HTTP/2 duplex transport or an explicit WS bridge).
//!
//! # Wire formats
//!
//! | Direction            | OpenAI (client)                     | Bedrock Nova Sonic (upstream)          |
//! |----------------------|-------------------------------------|----------------------------------------|
//! | Session setup        | `session.update`                    | `sessionStart` + `promptStart`         |
//! | Audio in             | `input_audio_buffer.append`         | `audioInput` (+ `contentStart`)        |
//! | Audio commit         | `input_audio_buffer.commit`         | `contentEnd`                           |
//! | Text in              | `conversation.item.create`          | `contentStart` + `textInput` + `contentEnd` |
//! | Tool result in       | `conversation.item.create` (output) | `contentStart` + `toolResult` + `contentEnd` |
//! | Trigger response     | `response.create`                   | automatic speech turn detection; text-only unsupported |
//! | Cancel response      | `response.cancel`                   | unsupported (Nova uses speech barge-in) |
//! | Session created      | `session.created`                   | `sessionStart`                         |
//! | Output begins        | `response.output_item.added`        | `contentStart` (role=ASSISTANT)        |
//! | Text delta           | `response.text.delta`               | `textOutput`                           |
//! | Audio delta          | `response.audio.delta`              | `audioOutput`                          |
//! | Output ends          | `response.output_item.done`         | `contentEnd`                           |
//! | Tool call            | function-call item + argument lifecycle | `contentStart(TOOL)` / `toolUse` / `contentEnd` |
//! | Response done        | `response.done`                     | `contentEnd` (END_TURN) / `promptEnd`  |
//!
//! # Audio sample rates
//!
//! OpenAI Realtime uses 24 kHz PCM16 for output; Bedrock Nova Sonic uses
//! 24 kHz for output and 16 kHz for input by default. The sample-rate
//! mapping is applied during `session.update` processing.
//!
//! # Statefulness
//!
//! Unlike the identity OpenAI provider, the Bedrock transform is **inherently
//! stateful**: it tracks the current prompt/content names (Nova Sonic
//! requires UUID-style names on every event), whether audio content has been
//! opened, the current response/output-item IDs for the upstream→client
//! direction, and accumulated text deltas. State is held behind a
//! [`std::sync::Mutex`] — one [`BedrockRealtimeTransform`] instance is
//! created per WebSocket session by the host, and both the client→upstream
//! and upstream→client pumps call into it cooperatively (never holding the
//! lock across an await point).
//!
//! # Limitations
//!
//! This module implements the **pure event transforms** only. The host
//! (`oxllm_server::realtime::bedrock`) is responsible for:
//!
//! - **Signed HTTP/2 duplex transport** for the official
//!   `POST /model/{modelId}/invoke-with-bidirectional-stream` operation. Native
//!   AWS runtime endpoints do not support a WebSocket GET upgrade.
//! - **AWS eventstream envelopes and chained request signatures**, handled by
//!   the host's official SDK transport. Decoded chunk bytes reach
//!   [`RealtimeProvider::transform_upstream_event`] as Nova JSON. An explicit
//!   WS(S) compatible bridge is a separate host transport, not the native API.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use oxllm_core::RealtimeProvider;
use oxllm_core::UsageSummary;
use oxllm_core::deployment::Deployment;
use oxllm_core::error::{Error, Result};
use percent_encoding::{AsciiSet, utf8_percent_encode};
use serde_json::{Value, json};

const REALTIME_MODEL_PATH_SEGMENT: &AsciiSet = &super::MODEL_PATH_SEGMENT.add(b':');

// ===========================================================================
// Constants — mirror the reference `BedrockRealtimeConfig` defaults.
// ===========================================================================

/// Default inference configuration.
const DEFAULT_MAX_TOKENS: u32 = 1024;
const DEFAULT_TOP_P: f64 = 0.9;
const DEFAULT_TEMPERATURE: f64 = 0.7;

/// Default audio output configuration.
const DEFAULT_OUTPUT_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_OUTPUT_SAMPLE_SIZE_BITS: u32 = 16;
const DEFAULT_OUTPUT_CHANNEL_COUNT: u32 = 1;
const DEFAULT_VOICE_ID: &str = "matthew";

/// Default audio input configuration.
const DEFAULT_INPUT_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_INPUT_SAMPLE_SIZE_BITS: u32 = 16;
const DEFAULT_INPUT_CHANNEL_COUNT: u32 = 1;

const OUTPUT_MEDIA_TYPE: &str = "audio/lpcm";
const OUTPUT_AUDIO_TYPE: &str = "SPEECH";
const OUTPUT_ENCODING: &str = "base64";
const INPUT_MEDIA_TYPE: &str = "audio/lpcm";
const INPUT_AUDIO_TYPE: &str = "SPEECH";
const INPUT_ENCODING: &str = "base64";
const TEXT_MEDIA_TYPE: &str = "text/plain";

// ===========================================================================
// ID generation — without the `uuid` crate (not a dependency of oxllm-providers).
// ===========================================================================

/// Monotonic counter for generating process-unique IDs.
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique hex ID with the given prefix (e.g. `"prompt"`, `"item"`).
///
/// Combines a nanosecond timestamp with a monotonic counter to guarantee
/// uniqueness within a process. The format is `{prefix}_{timestamp_hex}{counter_hex}`,
/// e.g. `prompt_18c5a3f0b2e1_0`. Nova Sonic only requires that prompt/content
/// names be unique strings — the exact format is not constrained.
fn unique_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}_{ts:x}{n:x}")
}

// ===========================================================================
// Session state — all mutable per-session fields.
// ===========================================================================

/// Mutable per-session state for the Bedrock realtime transform.
///
/// Held behind a `Mutex` inside [`BedrockRealtimeTransform`]. All transform
/// methods lock the mutex, mutate state, produce output events, and unlock
/// within a single synchronous call.
#[derive(Clone)]
struct SessionState {
    // ---- Nova Sonic content names ----
    /// The prompt name for this session (Nova Sonic groups all content under
    /// a single prompt name per turn).
    prompt_name: String,
    /// The audio content name used for `audioInput` / interactive audio.
    audio_content_name: String,

    // ---- Session lifecycle flags ----
    /// Whether `promptStart` has been sent (set by `session.update`).
    prompt_started: bool,
    /// Whether the client has supplied speech (required for response.create).
    client_audio_streamed: bool,
    /// Whether an audio `contentStart` is currently open upstream.
    audio_content_started: bool,
    /// Sample rate of the currently-open audio content (for mismatch detection).
    audio_content_sample_rate: u32,

    // ---- Inference configuration (updated by `session.update`) ----
    max_tokens: u32,
    top_p: f64,
    temperature: f64,

    // ---- Audio output configuration ----
    output_sample_rate: u32,
    voice_id: String,

    // ---- Audio input configuration ----
    input_sample_rate: u32,

    // ---- Upstream→client response tracking ----
    /// Current response ID (e.g. `resp_…`), set when ASSISTANT contentStart arrives.
    current_response_id: Option<String>,
    /// Current output item ID (e.g. `item_…`).
    current_output_item_id: Option<String>,
    /// Nova uses contentId on output (contentName on input/legacy adapters).
    current_content_id: Option<String>,
    current_tool_call: Option<Value>,
    completed_output: Vec<Value>,
    /// Current conversation ID (e.g. `conv_…`).
    current_conversation_id: Option<String>,
    /// Current delta type: `"text"`, `"audio"`, or `"tool"`.
    current_delta_type: Option<String>,
    /// Accumulated text deltas for `response.text.done`.
    current_delta_chunks: Vec<String>,
    /// Output index of the currently-open ASSISTANT item. Incremented per
    /// ASSISTANT `contentStart` within a response so multi-block turns
    /// (text → audio, text → toolUse, …) get distinct indices, matching
    /// OpenAI Realtime's per-item `output_index` semantics. Reset to 0 when
    /// a new `response_id` is minted.
    current_output_index: u32,
    /// Last usage report seen from upstream (a standalone `usage` event or
    /// inline on `completionEnd`/`promptEnd`). Threaded into the
    /// client-facing `response.done` so it carries real totals instead of
    /// hardcoded zeros. Taken (consumed) when `response.done` fires.
    last_usage: Option<UsageSummary>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            prompt_name: unique_id("prompt"),
            audio_content_name: unique_id("content"),
            prompt_started: false,
            client_audio_streamed: false,
            audio_content_started: false,
            audio_content_sample_rate: 0,
            max_tokens: DEFAULT_MAX_TOKENS,
            top_p: DEFAULT_TOP_P,
            temperature: DEFAULT_TEMPERATURE,
            output_sample_rate: DEFAULT_OUTPUT_SAMPLE_RATE,
            voice_id: DEFAULT_VOICE_ID.to_string(),
            input_sample_rate: DEFAULT_INPUT_SAMPLE_RATE,
            current_response_id: None,
            current_output_item_id: None,
            current_content_id: None,
            current_tool_call: None,
            completed_output: Vec::new(),
            current_conversation_id: None,
            current_delta_type: None,
            current_delta_chunks: Vec::new(),
            current_output_index: 0,
            last_usage: None,
        }
    }
}

// ===========================================================================
// BedrockRealtimeTransform
// ===========================================================================

/// Bedrock Nova Sonic realtime event transform.
///
/// Implements [`RealtimeProvider`] by translating OpenAI Realtime API events
/// to/from Bedrock Nova Sonic events. Create one instance per WebSocket
/// session.
///
/// See the [module docs](crate::bedrock::realtime) for the full event mapping
/// table and limitations.
pub struct BedrockRealtimeTransform {
    /// The bare model id (e.g. `amazon.nova-sonic-v1:0`), used in
    /// `session.created` events. Stored at construction since the trait's
    /// `transform_upstream_event` does not receive the model.
    model: String,
    /// Per-session mutable state.
    state: Mutex<SessionState>,
}

impl BedrockRealtimeTransform {
    /// Construct a new transform for the given model id.
    ///
    /// `model` should be the deployment's `oxllm_params.model` with any
    /// `bedrock/` prefix stripped (e.g. `amazon.nova-sonic-v1:0`).
    pub fn new(model: &str) -> Self {
        let bare = model.strip_prefix("bedrock/").unwrap_or(model);
        Self {
            model: bare.to_string(),
            state: Mutex::new(SessionState::default()),
        }
    }

    /// End a whole Nova session, not just one response. The host uses this
    /// documented sequence on disconnect before a bounded usage drain:
    /// <https://docs.aws.amazon.com/nova/latest/userguide/input-events>.html
    pub fn shutdown_events(&self) -> Vec<String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.prompt_started {
            return Vec::new();
        }
        let mut events = Vec::new();
        if state.audio_content_started {
            events.push(content_end_event(
                &state.prompt_name,
                &state.audio_content_name,
            ));
            state.audio_content_started = false;
        }
        events.push(prompt_end_event(&state.prompt_name));
        events.push(json!({"event":{"sessionEnd":{}}}).to_string());
        state.prompt_started = false;
        events
    }
}

impl Default for BedrockRealtimeTransform {
    fn default() -> Self {
        Self::new("amazon.nova-sonic-v1:0")
    }
}

impl RealtimeProvider for BedrockRealtimeTransform {
    fn transform_client_event(&self, event: &str) -> Result<Vec<String>> {
        let msg: Value = match serde_json::from_str(event) {
            Ok(v) => v,
            Err(e) => {
                return Err(Error::Serialization(format!(
                    "bedrock realtime: invalid JSON from client: {e} (event: {snip})",
                    // `get(..n)` returns None when `n` is not on a UTF-8 char
                    // boundary, avoiding a slice panic on multi-byte input.
                    snip = event.get(..200).unwrap_or(event)
                )));
            }
        };
        let event_type = msg.get("type").and_then(Value::as_str).unwrap_or("");
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let out = match event_type {
            "session.update" => state.transform_session_update(&msg),
            "input_audio_buffer.append" => state.transform_input_audio_append(&msg),
            "input_audio_buffer.commit" => state.transform_input_audio_commit(),
            "conversation.item.create" => state.transform_conversation_item_create(&msg),
            "response.create" => return state.transform_response_create(),
            "response.cancel" => return Err(Error::BadRequest(
                "response.cancel is not supported by Nova Sonic; interruption uses speech barge-in, or close the session".into()
            )),
            _ => {
                tracing::debug!(
                    event_type,
                    "bedrock realtime: unknown client event type, dropping"
                );
                Vec::new()
            }
        };
        Ok(out)
    }

    fn transform_upstream_event(&self, event: &str) -> Result<Vec<String>> {
        let msg: Value = match serde_json::from_str(event) {
            Ok(v) => v,
            Err(e) => {
                return Err(Error::Serialization(format!(
                    "bedrock realtime: invalid JSON from upstream: {e} (event: {snip})",
                    snip = event.get(..200).unwrap_or(event)
                )));
            }
        };
        // Bedrock events are wrapped: `{"event": { <eventName>: { ... } }}`.
        let event_obj = msg.get("event").unwrap_or(&msg);
        // Reuse the primary Bedrock adapter's exception handling. The host's
        // payload-only eventstream decoder may have discarded exception headers;
        // a remaining message-only error must still fail, not be silently dropped.
        if let Some(error) = super::bedrock_stream_error(event_obj) {
            return Err(error);
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        if event_obj.get("sessionStart").is_some() {
            return Ok(state.transform_session_start(&self.model));
        }
        if let Some(cs) = event_obj.get("contentStart") {
            return Ok(state.transform_content_start(cs));
        }
        if let Some(to) = event_obj.get("textOutput") {
            return Ok(state.transform_text_output(to));
        }
        if let Some(ao) = event_obj.get("audioOutput") {
            return Ok(state.transform_audio_output(ao));
        }
        if let Some(ce) = event_obj.get("contentEnd") {
            return Ok(state.transform_content_end(ce));
        }
        if let Some(tu) = event_obj.get("toolUse") {
            return state.transform_tool_use(tu);
        }
        // A standalone Nova Sonic `usage` event is metering-only (it has no
        // OpenAI Realtime equivalent — the client sees usage via
        // `response.done`). Capture it on the session state so the next
        // `response.done` carries real totals, then drop it from the client
        // stream. `extract_usage` independently surfaces it to the host's
        // metering hook.
        if event_obj.get("usage").is_some() || event_obj.get("usageEvent").is_some() {
            if let Some(u) = parse_bedrock_usage(event_obj) {
                if event_obj.get("usageEvent").is_some() {
                    match &mut state.last_usage {
                        Some(total) => total.merge(&u),
                        None => state.last_usage = Some(u),
                    }
                } else {
                    state.last_usage = Some(u);
                }
            }
            return Ok(Vec::new());
        }
        if event_obj.get("promptEnd").is_some() || event_obj.get("completionEnd").is_some() {
            // `completionEnd` / `promptEnd` may carry inline usage — capture
            // it before closing the response so `response.done` emits real
            // totals instead of zeros.
            if let Some(u) = parse_bedrock_usage(event_obj) {
                state.last_usage = Some(u);
            }
            return Ok(state.transform_prompt_end());
        }
        // Bedrock `error` events must not be swallowed silently — surface them
        // as OpenAI Realtime `error` events so the client learns the session
        // failed instead of observing a hung WebSocket. The Bedrock error
        // payload (`{"error": {"message": "…", "code": "…", …}}`) is mapped
        // onto the OpenAI Realtime error shape (`{"type":"error","error":…}`).
        if let Some(err) = event_obj.get("error") {
            return Ok(state.transform_upstream_error(err));
        }
        if event_obj.get("completionStart").is_some() || event_obj.get("sessionEnd").is_some() {
            return Ok(Vec::new());
        }
        // Unknown / unhandled upstream event — log so the drop is diagnosable,
        // then skip. Barge-in (`turnedStart`/`turnedEnd`), `sessionEnd`, and
        // any future event types land here until explicitly mapped.
        tracing::warn!(
            event_keys = ?event_obj.as_object().map(|o| o.keys().collect::<Vec<_>>()),
            "bedrock realtime: unknown upstream event, dropping"
        );
        Ok(Vec::new())
    }

    /// Extract token usage from a Nova Sonic usage-bearing upstream event.
    ///
    /// Native Nova `usageEvent` carries incremental speech/text counts in
    /// `details.delta` alongside cumulative session totals. Sum the deltas,
    /// never the cumulative totals (which would double-bill each update).
    /// Legacy compatible `usage` / inline reports are accepted as well.
    fn extract_usage(&self, upstream_event: &str) -> Option<UsageSummary> {
        let v: Value = serde_json::from_str(upstream_event).ok()?;
        let event_obj = v.get("event").unwrap_or(&v);
        parse_bedrock_usage(event_obj)
    }

    fn ws_url(&self, deployment: &Deployment) -> String {
        // Preserve an explicit bridge/native base. Without one this is an
        // HTTPS operation descriptor, not an AWS WebSocket endpoint.
        let region = deployment
            .oxllm_params
            .extra
            .get("aws_region_name")
            .or_else(|| deployment.oxllm_params.extra.get("aws_region"))
            .and_then(Value::as_str)
            .filter(|region| !region.is_empty())
            .unwrap_or("us-east-1");
        let default_base = format!("https://bedrock-runtime.{region}.amazonaws.com");
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(&default_base);
        let (base, fragment) = base
            .split_once('#')
            .map_or((base, None), |(base, fragment)| (base, Some(fragment)));
        let (base, query) = base
            .split_once('?')
            .map_or((base, None), |(base, query)| (base, Some(query)));
        let base = base.trim_end_matches('/');

        // Derive the bare model id from the deployment.
        let configured = &deployment.oxllm_params.model;
        let model = configured
            .strip_prefix("bedrock/")
            .or_else(|| configured.strip_prefix("bedrock_mantle/"))
            .unwrap_or(configured);
        // Normalize both suffix forms even when stacked, as the host does.
        let model = super::strip_model_suffixes(&super::strip_model_suffixes(model));
        let model = utf8_percent_encode(&model, REALTIME_MODEL_PATH_SEGMENT);

        // Explicit WS(S) bridges expose this operation path with their own
        // upgrade contract. Native HTTP(S) endpoints require a signed HTTP/2
        // POST; returning their URL here does not authorize a WS GET upgrade.
        let mut url = format!("{base}/model/{model}/invoke-with-bidirectional-stream");
        if let Some(query) = query {
            url.push('?');
            url.push_str(query);
        }
        if let Some(fragment) = fragment {
            url.push('#');
            url.push_str(fragment);
        }
        url
    }

    fn ws_auth_headers(&self, _deployment: &Deployment) -> Vec<(String, String)> {
        // Native duplex headers and chained event signatures are created by
        // the host's SDK. Empty static headers are not permission to dial an
        // unsigned runtime endpoint; explicit WS(S) bridges use host presigning.
        Vec::new()
    }
}

// ===========================================================================
// Client → Upstream transforms
// ===========================================================================

impl SessionState {
    /// `session.update` → `sessionStart` + `promptStart` (+ system prompt).
    fn transform_session_update(&mut self, msg: &Value) -> Vec<String> {
        let session = msg.get("session").unwrap_or(&Value::Null);

        // Update inference configuration.
        if let Some(t) = session
            .get("max_response_output_tokens")
            .and_then(Value::as_u64)
        {
            // Saturate rather than truncate: a caller-supplied u64 above
            // u32::MAX must not silently become a small/garbage value.
            self.max_tokens = u32::try_from(t).unwrap_or(u32::MAX);
        }
        if let Some(t) = session.get("temperature").and_then(Value::as_f64) {
            self.temperature = t;
        }
        if let Some(t) = session.get("top_p").and_then(Value::as_f64) {
            self.top_p = t;
        }

        // Update voice.
        if let Some(v) = session.get("voice").and_then(Value::as_str) {
            self.voice_id = v.to_string();
        }

        // Map audio formats to sample rates.
        if let Some(fmt) = session.get("output_audio_format").and_then(Value::as_str) {
            self.output_sample_rate = map_audio_format_to_sample_rate(fmt, true);
        }
        if let Some(fmt) = session.get("input_audio_format").and_then(Value::as_str) {
            self.input_sample_rate = map_audio_format_to_sample_rate(fmt, false);
        }

        // Allow direct override of sample rates (custom extension).
        if let Some(sr) = session
            .get("output_sample_rate_hertz")
            .and_then(Value::as_u64)
        {
            self.output_sample_rate = u32::try_from(sr).unwrap_or(u32::MAX);
        }
        if let Some(sr) = session
            .get("input_sample_rate_hertz")
            .and_then(Value::as_u64)
        {
            self.input_sample_rate = u32::try_from(sr).unwrap_or(u32::MAX);
        }

        let mut messages = Vec::new();

        // Nova Sonic prompt lifecycle: every `promptStart` must be balanced by
        // a `promptEnd` before a new prompt opens. OpenAI Realtime clients
        // routinely send `session.update` more than once (to change voice,
        // tools, turn detection, …), so on the second and subsequent calls
        // we close out the previous prompt first — emit a `contentEnd` for
        // any open audio content, then a `promptEnd`, then start a fresh
        // prompt name. Without this the upstream would either error or open a
        // second prompt context, corrupting the session.
        if self.prompt_started {
            if self.audio_content_started {
                messages.push(content_end_event(
                    &self.prompt_name,
                    &self.audio_content_name,
                ));
                self.audio_content_started = false;
                self.audio_content_name = unique_id("content");
            }
            messages.push(prompt_end_event(&self.prompt_name));
            // Open a new prompt under a fresh name.
            self.prompt_name = unique_id("prompt");
            self.prompt_started = false;
        }

        // 1. sessionStart — inference configuration.
        messages.push(
            json!({
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": self.max_tokens,
                            "topP": self.top_p,
                            "temperature": self.temperature,
                        }
                    }
                }
            })
            .to_string(),
        );

        // 2. promptStart — text/audio output configuration + optional tools.
        let mut prompt_start = json!({
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": TEXT_MEDIA_TYPE},
            "audioOutputConfiguration": {
                "mediaType": OUTPUT_MEDIA_TYPE,
                "sampleRateHertz": self.output_sample_rate,
                "sampleSizeBits": DEFAULT_OUTPUT_SAMPLE_SIZE_BITS,
                "channelCount": DEFAULT_OUTPUT_CHANNEL_COUNT,
                "voiceId": self.voice_id,
                "encoding": OUTPUT_ENCODING,
                "audioType": OUTPUT_AUDIO_TYPE,
            },
        });

        // Attach tool configuration if the session declares tools.
        if let Some(tools) = session.get("tools").and_then(Value::as_array) {
            let bedrock_tools = transform_tools_to_bedrock(tools);
            if !bedrock_tools.is_empty() {
                if let Some(obj) = prompt_start.as_object_mut() {
                    obj.insert(
                        "toolUseOutputConfiguration".to_string(),
                        json!({"mediaType": "application/json"}),
                    );
                    obj.insert(
                        "toolConfiguration".to_string(),
                        json!({"tools": bedrock_tools}),
                    );
                }
            }
        }

        messages.push(
            json!({
                "event": {
                    "promptStart": prompt_start,
                }
            })
            .to_string(),
        );
        self.prompt_started = true;

        // 3. System prompt (instructions) as a TEXT content trio.
        if let Some(instructions) = session.get("instructions").and_then(Value::as_str) {
            if !instructions.is_empty() {
                let content_name = unique_id("content");
                messages.extend(text_content_trio(
                    &self.prompt_name,
                    &content_name,
                    "SYSTEM",
                    false,
                    instructions,
                ));
            }
        }

        messages
    }

    /// `input_audio_buffer.append` → (optional contentEnd) + contentStart + audioInput.
    fn transform_input_audio_append(&mut self, msg: &Value) -> Vec<String> {
        self.client_audio_streamed = true;
        let mut messages = Vec::new();

        // If the sample rate changed since the audio content was opened,
        // close the old content and open a new one with a fresh name.
        if self.audio_content_started && self.audio_content_sample_rate != self.input_sample_rate {
            messages.push(content_end_event(
                &self.prompt_name,
                &self.audio_content_name,
            ));
            self.audio_content_started = false;
            self.audio_content_name = unique_id("content");
        }

        // Open audio content on first append (or after a sample-rate swap).
        if !self.audio_content_started {
            messages.push(
                json!({
                    "event": {
                        "contentStart": {
                            "promptName": self.prompt_name,
                            "contentName": self.audio_content_name,
                            "type": "AUDIO",
                            "interactive": true,
                            "role": "USER",
                            "audioInputConfiguration": {
                                "mediaType": INPUT_MEDIA_TYPE,
                                "sampleRateHertz": self.input_sample_rate,
                                "sampleSizeBits": DEFAULT_INPUT_SAMPLE_SIZE_BITS,
                                "channelCount": DEFAULT_INPUT_CHANNEL_COUNT,
                                "audioType": INPUT_AUDIO_TYPE,
                                "encoding": INPUT_ENCODING,
                            },
                        }
                    }
                })
                .to_string(),
            );
            self.audio_content_started = true;
            self.audio_content_sample_rate = self.input_sample_rate;
        }

        // Forward the audio chunk. The client already provides base64-encoded PCM.
        let audio_data = msg.get("audio").and_then(Value::as_str).unwrap_or("");
        messages.push(
            json!({
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": audio_data,
                    }
                }
            })
            .to_string(),
        );

        messages
    }

    /// `input_audio_buffer.commit` → contentEnd (for open audio content).
    fn transform_input_audio_commit(&mut self) -> Vec<String> {
        if !self.audio_content_started {
            return Vec::new();
        }
        self.audio_content_started = false;
        vec![content_end_event(
            &self.prompt_name,
            &self.audio_content_name,
        )]
    }

    /// `conversation.item.create` → text content trio or tool result trio.
    fn transform_conversation_item_create(&mut self, msg: &Value) -> Vec<String> {
        let item = msg.get("item").unwrap_or(&Value::Null);
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");

        match item_type {
            "function_call_output" => self.transform_tool_result(item),
            "message" => {
                let mut messages = Vec::new();
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for part in content {
                        if part.get("type").and_then(Value::as_str) == Some("input_text") {
                            let text = part.get("text").and_then(Value::as_str).unwrap_or("");
                            let content_name = unique_id("content");
                            messages.extend(text_content_trio(
                                &self.prompt_name,
                                &content_name,
                                "USER",
                                false,
                                text,
                            ));
                        }
                    }
                }
                messages
            }
            _ => Vec::new(),
        }
    }

    /// Transform a `function_call_output` item into Bedrock tool result events.
    fn transform_tool_result(&mut self, item: &Value) -> Vec<String> {
        let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
        let output = item.get("output");
        let output_str = match output {
            Some(Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => String::new(),
        };

        let content_name = unique_id("content");
        let mut messages = Vec::new();

        // contentStart for the tool result.
        messages.push(
            json!({
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "interactive": false,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": call_id,
                            "type": "TEXT",
                            "textInputConfiguration": {"mediaType": TEXT_MEDIA_TYPE},
                        },
                    }
                }
            })
            .to_string(),
        );

        // toolResult payload.
        messages.push(
            json!({
                "event": {
                    "toolResult": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "content": output_str,
                    }
                }
            })
            .to_string(),
        );

        // contentEnd.
        messages.push(content_end_event(&self.prompt_name, &content_name));
        messages
    }

    /// Nova v1 textInput supplies context, not an interactive speech turn.
    /// A synthetic tone is not a supported speech trigger and must not be
    /// injected (nor silently billed) on behalf of a text-only caller.
    fn transform_response_create(&mut self) -> Result<Vec<String>> {
        if !self.prompt_started {
            return Err(Error::BadRequest(
                "send session.update before response.create".into(),
            ));
        }
        if !self.client_audio_streamed {
            return Err(Error::BadRequest(
                "text-only response.create is not supported by Nova Sonic; supply speech using input_audio_buffer.append".into()
            ));
        }
        // Speech sessions already generate through Nova's automatic VAD.
        Ok(Vec::new())
    }
}

// ===========================================================================
// Upstream → Client transforms
// ===========================================================================

impl SessionState {
    /// `sessionStart` → `session.created`.
    fn transform_session_start(&mut self, model: &str) -> Vec<String> {
        let event_id = unique_id("event");
        let session_id = unique_id("sess");
        vec![
            json!({
                "type": "session.created",
                "event_id": event_id,
                "session": {
                    "id": session_id,
                    "object": "realtime.session",
                    "model": model,
                    "modalities": ["text", "audio"],
                }
            })
            .to_string(),
        ]
    }

    /// `contentStart` (ASSISTANT or TOOL) starts an output block. Nova's TOOL
    /// role is model-generated tool use, not an echo of a client tool result.
    ///
    /// SYSTEM/USER starts are input echoes/transcriptions, not assistant items.
    fn transform_content_start(&mut self, content_start: &Value) -> Vec<String> {
        let role = content_start
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("");
        let content_type = content_start
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("TEXT");
        if role != "ASSISTANT" && !(role == "TOOL" && content_type == "TOOL") {
            return Vec::new();
        }

        // Initialize the response/conversation IDs on the first ASSISTANT
        // contentStart of a response (after an END_TURN or promptEnd they are
        // `None`). For every subsequent ASSISTANT contentStart within the
        // same response, advance the output index so each content block gets
        // a distinct `output_index` and a fresh `item_id` — OpenAI Realtime
        // semantics assign a new item per output block (text → audio, text →
        // toolUse, …). Previously the index was pinned to 0 and the item id
        // reused, colliding multi-block turns on the client.
        let is_new_response = self.current_response_id.is_none();
        if is_new_response {
            self.current_response_id = Some(unique_id("resp"));
            self.current_conversation_id = Some(unique_id("conv"));
            self.current_output_index = 0;
        } else {
            self.current_output_index = self.current_output_index.saturating_add(1);
        }
        self.current_output_item_id = Some(unique_id("item"));
        self.current_content_id = content_start
            .get("contentId")
            .or_else(|| content_start.get("contentName"))
            .and_then(Value::as_str)
            .map(String::from);
        self.current_tool_call = None;

        let delta_type = match content_type {
            "TEXT" => "text",
            "TOOL" => "tool",
            _ => "audio",
        };
        self.current_delta_type = Some(delta_type.to_string());
        self.current_delta_chunks.clear();

        let response_id = self.current_response_id.clone().unwrap_or_default();
        let output_item_id = self.current_output_item_id.clone().unwrap_or_default();
        let conversation_id = self.current_conversation_id.clone().unwrap_or_default();
        let output_index = self.current_output_index;

        let mut messages = Vec::new();

        // `response.created` is emitted only at the start of a new response.
        if is_new_response {
            messages.push(
                json!({
                    "type": "response.created",
                    "event_id": unique_id("event"),
                    "response": {
                        "object": "realtime.response",
                        "id": response_id,
                        "status": "in_progress",
                        "output": [],
                        "conversation_id": conversation_id,
                    }
                })
                .to_string(),
            );
        }

        // toolUse carries the call ID and name; wait for those before emitting
        // a function_call item. Never label a TOOL block as audio/a message.
        if delta_type == "tool" {
            return messages;
        }

        // response.output_item.added
        messages.push(
            json!({
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": output_index,
                "item": {
                    "id": output_item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
            })
            .to_string(),
        );

        // response.content_part.added
        let part = if delta_type == "text" {
            json!({"type": "text", "text": ""})
        } else {
            json!({"type": "audio", "transcript": ""})
        };
        messages.push(
            json!({
                "type": "response.content_part.added",
                "content_index": 0,
                "output_index": output_index,
                "event_id": unique_id("event"),
                "item_id": output_item_id,
                "part": part,
                "response_id": response_id,
            })
            .to_string(),
        );

        messages
    }

    /// `textOutput` → `response.text.delta` (+ accumulate for text.done).
    fn transform_text_output(&mut self, text_output: &Value) -> Vec<String> {
        if !self.matches_content(text_output) || self.current_delta_type.as_deref() != Some("text")
        {
            return Vec::new();
        }
        let (Some(response_id), Some(output_item_id)) = (
            self.current_response_id.as_ref(),
            self.current_output_item_id.as_ref(),
        ) else {
            return Vec::new();
        };
        let output_index = self.current_output_index;

        let content = text_output
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("");
        self.current_delta_chunks.push(content.to_string());

        vec![
            json!({
                "type": "response.text.delta",
                "content_index": 0,
                "event_id": unique_id("event"),
                "item_id": output_item_id,
                "output_index": output_index,
                "response_id": response_id,
                "delta": content,
            })
            .to_string(),
        ]
    }

    /// `audioOutput` → `response.audio.delta`.
    fn transform_audio_output(&mut self, audio_output: &Value) -> Vec<String> {
        if !self.matches_content(audio_output)
            || self.current_delta_type.as_deref() != Some("audio")
        {
            return Vec::new();
        }
        let (Some(response_id), Some(output_item_id)) = (
            self.current_response_id.as_ref(),
            self.current_output_item_id.as_ref(),
        ) else {
            return Vec::new();
        };
        let output_index = self.current_output_index;

        let content = audio_output
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("");

        vec![
            json!({
                "type": "response.audio.delta",
                "content_index": 0,
                "event_id": unique_id("event"),
                "item_id": output_item_id,
                "output_index": output_index,
                "response_id": response_id,
                "delta": content,
            })
            .to_string(),
        ]
    }

    /// `contentEnd` → done events for the current content + output_item.done.
    ///
    /// Native completionEnd is the final response boundary; TOOL_USE and
    /// interruption finish the corresponding OpenAI response immediately.
    fn transform_content_end(&mut self, content_end: &Value) -> Vec<String> {
        if !self.matches_content(content_end) {
            return Vec::new();
        }
        let (Some(response_id), Some(output_item_id)) = (
            self.current_response_id.as_ref(),
            self.current_output_item_id.as_ref(),
        ) else {
            return Vec::new();
        };

        let response_id = response_id.clone();
        let output_item_id = output_item_id.clone();
        let delta_type = self.current_delta_type.clone();
        let accumulated_text = self.current_delta_chunks.join("");
        let output_index = self.current_output_index;

        let mut messages = Vec::new();

        match delta_type.as_deref() {
            Some("tool") => {
                if let Some(item) = &self.current_tool_call {
                    messages.push(json!({
                        "type":"response.function_call_arguments.done",
                        "event_id":unique_id("event"), "response_id":response_id,
                        "item_id":output_item_id, "output_index":output_index,
                        "call_id":item["call_id"], "name":item["name"], "arguments":item["arguments"]
                    }).to_string());
                } else {
                    return Vec::new();
                }
            }
            Some("text") => {
                // response.text.done
                messages.push(
                    json!({
                        "type": "response.text.done",
                        "content_index": 0,
                        "event_id": unique_id("event"),
                        "item_id": output_item_id,
                        "output_index": output_index,
                        "response_id": response_id,
                        "text": accumulated_text,
                    })
                    .to_string(),
                );
                // response.content_part.done (text)
                messages.push(
                    json!({
                        "type": "response.content_part.done",
                        "content_index": 0,
                        "event_id": unique_id("event"),
                        "item_id": output_item_id,
                        "output_index": output_index,
                        "part": {"type": "text", "text": accumulated_text},
                        "response_id": response_id,
                    })
                    .to_string(),
                );
            }
            Some("audio") => {
                // response.audio.done
                messages.push(
                    json!({
                        "type": "response.audio.done",
                        "content_index": 0,
                        "event_id": unique_id("event"),
                        "item_id": output_item_id,
                        "output_index": output_index,
                        "response_id": response_id,
                    })
                    .to_string(),
                );
                // response.content_part.done (audio)
                messages.push(
                    json!({
                        "type": "response.content_part.done",
                        "content_index": 0,
                        "event_id": unique_id("event"),
                        "item_id": output_item_id,
                        "output_index": output_index,
                        "part": {"type": "audio", "transcript": ""},
                        "response_id": response_id,
                    })
                    .to_string(),
                );
            }
            _ => {}
        }

        let item = self.current_tool_call.take().unwrap_or_else(|| {
            let content = if delta_type.as_deref() == Some("text") {
                json!([{"type":"text","text":accumulated_text}])
            } else {
                json!([{"type":"audio","transcript":""}])
            };
            json!({"id":output_item_id,"object":"realtime.item","type":"message",
                "status":"completed","role":"assistant","content":content})
        });
        // response.output_item.done
        messages.push(
            json!({
                "type": "response.output_item.done",
                "event_id": unique_id("event"),
                "output_index": output_index,
                "response_id": response_id,
                "item": item,
            })
            .to_string(),
        );

        // Reset delta chunks for the next content.
        self.current_delta_chunks.clear();
        self.current_output_item_id = None;
        self.current_content_id = None;
        self.current_delta_type = None;
        self.completed_output.push(item);

        // Native streams have a completionEnd after contentEnd; retain the
        // response until then so late usage is included. Legacy adapters lack
        // completionId and use contentEnd as the turn boundary. TOOL_USE ends
        // the OpenAI function-call response so clients can return tool output.
        let stop_reason = content_end.get("stopReason").and_then(Value::as_str);
        if stop_reason == Some("TOOL_USE")
            || (stop_reason == Some("END_TURN") && content_end.get("completionId").is_none())
        {
            messages.extend(response_done_events(self));
        } else if stop_reason == Some("INTERRUPTED") {
            messages.extend(response_done_events_with_status(self, "cancelled"));
        }

        messages
    }

    /// `toolUse.content` contains complete JSON arguments. Emit the added item
    /// and argument delta here; finish it at contentEnd(TOOL_USE).
    fn transform_tool_use(&mut self, tool_use: &Value) -> Result<Vec<String>> {
        if !self.matches_content(tool_use) || self.current_delta_type.as_deref() != Some("tool") {
            return Err(Error::Serialization(
                "Nova toolUse has no matching TOOL contentStart".into(),
            ));
        }
        let (Some(response_id), Some(output_item_id)) = (
            self.current_response_id.as_ref(),
            self.current_output_item_id.as_ref(),
        ) else {
            return Err(Error::Serialization(
                "Nova toolUse has no active response".into(),
            ));
        };
        let output_index = self.current_output_index;

        let call_id = tool_use
            .get("toolUseId")
            .and_then(Value::as_str)
            .unwrap_or("");
        let tool_name = tool_use
            .get("toolName")
            .and_then(Value::as_str)
            .unwrap_or("");
        let arguments = match tool_use.get("content").or_else(|| tool_use.get("input")) {
            Some(Value::String(text)) => {
                serde_json::from_str::<Value>(text).map_err(|_| {
                    Error::Serialization("invalid Nova tool-use JSON arguments".into())
                })?;
                text.clone()
            }
            Some(value) => value.to_string(),
            None => return Err(Error::Serialization("Nova toolUse has no arguments".into())),
        };
        if call_id.is_empty() || tool_name.is_empty() || self.current_tool_call.is_some() {
            return Err(Error::Serialization(
                "invalid or duplicate Nova toolUse".into(),
            ));
        }
        let item = json!({"id":output_item_id,"object":"realtime.item","type":"function_call",
            "status":"completed","call_id":call_id,"name":tool_name,"arguments":arguments});
        self.current_tool_call = Some(item.clone());
        let mut added = item;
        added["status"] = json!("in_progress");
        added["arguments"] = json!("");

        Ok(vec![
            json!({
                "type":"response.output_item.added", "event_id":unique_id("event"),
                "response_id":response_id, "output_index":output_index, "item":added
            })
            .to_string(),
            json!({
                "type": "response.function_call_arguments.delta",
                "event_id": unique_id("event"),
                "response_id": response_id,
                "item_id": output_item_id,
                "output_index": output_index,
                "call_id": call_id,
                "name": tool_name,
                "delta": arguments,
            })
            .to_string(),
        ])
    }

    fn matches_content(&self, event: &Value) -> bool {
        let actual = event
            .get("contentId")
            .or_else(|| event.get("contentName"))
            .and_then(Value::as_str);
        match (self.current_content_id.as_deref(), actual) {
            (Some(expected), Some(actual)) => expected == actual,
            _ => true,
        }
    }

    /// `promptEnd` / `completionEnd` → `response.done` + reset.
    fn transform_prompt_end(&mut self) -> Vec<String> {
        response_done_events(self)
    }

    /// Map a Bedrock upstream `error` event onto an OpenAI Realtime `error`
    /// event. The Bedrock error payload (`{"message": "…", "code": "…", …}`)
    /// is forwarded verbatim under `error`, with `type: "error"` added to
    /// match the OpenAI Realtime wire shape. This ensures a Nova Sonic
    /// failure surfaces to the client instead of silently stalling the
    /// WebSocket.
    fn transform_upstream_error(&self, err: &Value) -> Vec<String> {
        let mut error_payload = if let Some(obj) = err.as_object() {
            Value::Object(obj.clone())
        } else {
            json!({})
        };
        // Guarantee a `message` field exists so the client always has
        // something actionable.
        if let Value::Object(map) = &mut error_payload {
            if !map.contains_key("message") {
                if let Some(msg) = err
                    .get("errorMessage")
                    .and_then(Value::as_str)
                    .or_else(|| err.as_str())
                {
                    map.insert("message".to_string(), json!(msg));
                } else {
                    map.insert("message".to_string(), json!("bedrock realtime error"));
                }
            }
        }
        vec![
            json!({
                "type": "error",
                "error": error_payload,
            })
            .to_string(),
        ]
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Parse a Nova Sonic usage report from an upstream event object into a
/// [`UsageSummary`]. Native usageEvent carries speech/text deltas; legacy
/// adapters may use `usage` or inline completionEnd/promptEnd reports.
///
/// `inputTokens`/`outputTokens`/`totalTokens` map onto the OpenAI
/// `prompt_tokens`/`completion_tokens`/`total_tokens` fields; `cacheReadInputTokens`
/// surfaces as `cache_read_input_tokens`.
fn parse_bedrock_usage(event_obj: &Value) -> Option<UsageSummary> {
    if let Some(report) = event_obj.get("usageEvent") {
        let delta = report.pointer("/details/delta")?;
        let input_speech = delta.pointer("/input/speechTokens")?.as_u64()?;
        let input_text = delta.pointer("/input/textTokens")?.as_u64()?;
        let output_speech = delta.pointer("/output/speechTokens")?.as_u64()?;
        let output_text = delta.pointer("/output/textTokens")?.as_u64()?;
        let input = input_speech.saturating_add(input_text);
        let output = output_speech.saturating_add(output_text);
        return Some(UsageSummary {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input.saturating_add(output),
            audio_input_tokens: Some(input_speech),
            ..Default::default()
        });
    }
    let usage = event_obj
        .get("usage")
        .or_else(|| event_obj.get("completionEnd").and_then(|c| c.get("usage")))
        .or_else(|| event_obj.get("promptEnd").and_then(|p| p.get("usage")))?;
    if usage.get("inputTokens").and_then(Value::as_u64).is_none()
        && usage.get("outputTokens").and_then(Value::as_u64).is_none()
    {
        return None;
    }
    let input = usage
        .get("inputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("outputTokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total = usage
        .get("totalTokens")
        .and_then(Value::as_u64)
        .unwrap_or(input.saturating_add(output));
    let cache_read = usage
        .get("cacheReadInputTokens")
        .and_then(Value::as_u64)
        .filter(|&c| c > 0);
    Some(UsageSummary {
        prompt_tokens: input,
        completion_tokens: output,
        total_tokens: total,
        cache_read_input_tokens: cache_read,
        ..Default::default()
    })
}

/// Map an OpenAI audio format string to a Bedrock sample rate.
///
/// - `pcm16`: 24 kHz for output, 16 kHz for input.
/// - `g711_ulaw` / `g711_alaw`: 8 kHz.
fn map_audio_format_to_sample_rate(audio_format: &str, is_output: bool) -> u32 {
    match audio_format {
        "pcm16" => {
            if is_output {
                24_000
            } else {
                16_000
            }
        }
        "g711_ulaw" | "g711_alaw" => 8_000,
        _ => {
            if is_output {
                24_000
            } else {
                16_000
            }
        }
    }
}

/// Transform OpenAI tool definitions to Bedrock `toolSpec` format.
fn transform_tools_to_bedrock(tools: &[Value]) -> Vec<Value> {
    let mut bedrock_tools = Vec::with_capacity(tools.len());
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        // Realtime tools use flat name/description/parameters fields. Retain
        // support for the nested Chat Completions shape as an extension.
        let function = tool.get("function").unwrap_or(tool);
        let name = function.get("name").and_then(Value::as_str).unwrap_or("");
        let description = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        let empty = json!({});
        let parameters = function.get("parameters").unwrap_or(&empty);
        // Bedrock expects the input schema as a JSON string under `json`.
        let input_schema_json = serde_json::to_string(parameters).unwrap_or_else(|_| "{}".into());
        bedrock_tools.push(json!({
            "toolSpec": {
                "name": name,
                "description": description,
                "inputSchema": {"json": input_schema_json},
            }
        }));
    }
    bedrock_tools
}

/// Build a Bedrock `contentEnd` event JSON string.
fn content_end_event(prompt_name: &str, content_name: &str) -> String {
    json!({
        "event": {
            "contentEnd": {
                "promptName": prompt_name,
                "contentName": content_name,
            }
        }
    })
    .to_string()
}

/// Build a Bedrock `promptEnd` event JSON string. Nova Sonic requires every
/// `promptStart` to be balanced by a `promptEnd` before a new prompt can
/// open; emitting a second `promptStart` without one corrupts the session.
fn prompt_end_event(prompt_name: &str) -> String {
    json!({
        "event": {
            "promptEnd": {
                "promptName": prompt_name,
            }
        }
    })
    .to_string()
}

/// Build a TEXT content trio: `contentStart` + `textInput` + `contentEnd`.
///
/// Used for system instructions and user text input.
fn text_content_trio(
    prompt_name: &str,
    content_name: &str,
    role: &str,
    interactive: bool,
    text: &str,
) -> Vec<String> {
    vec![
        json!({
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "interactive": interactive,
                    "role": role,
                    "textInputConfiguration": {"mediaType": TEXT_MEDIA_TYPE},
                }
            }
        })
        .to_string(),
        json!({
            "event": {
                "textInput": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "content": text,
                }
            }
        })
        .to_string(),
        content_end_event(prompt_name, content_name),
    ]
}

/// Build the `response.done` event and reset response tracking state.
///
/// Returns an empty vec when there is no active response to close.
fn response_done_events(state: &mut SessionState) -> Vec<String> {
    response_done_events_with_status(state, "completed")
}

fn response_done_events_with_status(state: &mut SessionState, status: &str) -> Vec<String> {
    let (Some(response_id), Some(conversation_id)) = (
        state.current_response_id.take(),
        state.current_conversation_id.take(),
    ) else {
        // No active response — nothing to close. Also reset the rest.
        state.current_output_item_id = None;
        state.current_content_id = None;
        state.current_tool_call = None;
        state.completed_output.clear();
        state.current_delta_type = None;
        state.current_delta_chunks.clear();
        state.current_output_index = 0;
        state.last_usage = None;
        return Vec::new();
    };

    // Reset tracking for the next response.
    state.current_output_item_id = None;
    state.current_content_id = None;
    state.current_tool_call = None;
    state.current_delta_type = None;
    state.current_delta_chunks.clear();
    state.current_output_index = 0;
    let output = std::mem::take(&mut state.completed_output);
    // Unknown usage is null, not a fabricated zero report. The host meters
    // native usageEvent deltas independently even if they arrive after a turn.
    let usage = state.last_usage.take().map(|u| json!({
        "input_tokens":u.prompt_tokens,"output_tokens":u.completion_tokens,"total_tokens":u.total_tokens,
        "input_token_details":{"audio_tokens":u.audio_input_tokens}
    }));

    vec![
        json!({
            "type": "response.done",
            "event_id": unique_id("event"),
            "response": {
                "object": "realtime.response",
                "id": response_id,
                "status": status,
                "output": output,
                "conversation_id": conversation_id,
                "usage": usage,
            }
        })
        .to_string(),
    ]
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;

    // ---- helper: parse a JSON string into a Value ----
    fn parse(s: &str) -> Value {
        serde_json::from_str(s).expect("valid JSON")
    }

    // ---- helper: extract the Bedrock event type from an event JSON string ----
    fn bedrock_event_type(s: &str) -> String {
        let v: Value = serde_json::from_str(s).expect("valid JSON");
        let event = v.get("event").unwrap_or(&v);
        if let Some(obj) = event.as_object() {
            for key in [
                "sessionStart",
                "promptStart",
                "contentStart",
                "textInput",
                "contentEnd",
                "audioInput",
                "toolResult",
                "sessionEnd",
                "promptEnd",
            ] {
                if obj.contains_key(key) {
                    return key.to_string();
                }
            }
        }
        "unknown".to_string()
    }

    // ---- helper: extract the OpenAI event type ----
    fn openai_event_type(s: &str) -> String {
        let v: Value = serde_json::from_str(s).expect("valid JSON");
        v.get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string()
    }

    fn transform() -> BedrockRealtimeTransform {
        BedrockRealtimeTransform::new("amazon.nova-sonic-v1:0")
    }

    // =========================================================================
    // Robustness: malformed multi-byte input must not panic (HIGH #3)
    // =========================================================================

    /// A non-JSON client event whose 200th byte lands inside a multi-byte
    /// UTF-8 sequence must return an error, not panic. Previously the error
    /// branch sliced by byte index (`&event[..200]`), which panics with
    /// "byte index 200 is not a char boundary".
    #[test]
    fn transform_client_event_multibyte_non_json_does_not_panic() {
        let t = transform();
        // 199 ASCII bytes, then a 4-byte emoji straddling offset 200, then
        // trailing garbage so the total exceeds 200 bytes and is not JSON.
        let mut event = "x".repeat(199);
        event.push('🚀'); // 4 bytes: offsets 199..203
        event.push_str("{{{not-json");
        assert!(event.len() > 200);
        let res = t.transform_client_event(&event);
        assert!(
            res.is_err(),
            "malformed multi-byte input must error, not panic"
        );
        let err = res.unwrap_err();
        assert!(
            matches!(err, Error::Serialization(_)),
            "expected Serialization error, got {err:?}"
        );
    }

    /// Same robustness check for the upstream direction.
    #[test]
    fn transform_upstream_event_multibyte_non_json_does_not_panic() {
        let t = transform();
        let mut event = "y".repeat(199);
        event.push('🚀');
        event.push_str("{{{not-json");
        assert!(event.len() > 200);
        let res = t.transform_upstream_event(&event);
        assert!(res.is_err());
        assert!(
            matches!(res.unwrap_err(), Error::Serialization(_)),
            "expected Serialization error"
        );
    }

    // =========================================================================
    // Client → Upstream: session.update
    // =========================================================================

    #[test]
    fn session_update_produces_session_start_and_prompt_start() {
        let t = transform();
        let event = r#"{"type":"session.update","session":{"voice":"alloy"}}"#;
        let out = t.transform_client_event(event).unwrap();
        assert_eq!(
            out.len(),
            2,
            "session.update without instructions → 2 events"
        );
        assert_eq!(bedrock_event_type(&out[0]), "sessionStart");
        assert_eq!(bedrock_event_type(&out[1]), "promptStart");
    }

    #[test]
    fn session_update_with_instructions_adds_text_content_trio() {
        let t = transform();
        let event = r#"{"type":"session.update","session":{"instructions":"You are helpful."}}"#;
        let out = t.transform_client_event(event).unwrap();
        // sessionStart + promptStart + contentStart + textInput + contentEnd
        assert_eq!(out.len(), 5);
        assert_eq!(bedrock_event_type(&out[0]), "sessionStart");
        assert_eq!(bedrock_event_type(&out[1]), "promptStart");
        assert_eq!(bedrock_event_type(&out[2]), "contentStart");
        assert_eq!(bedrock_event_type(&out[3]), "textInput");
        assert_eq!(bedrock_event_type(&out[4]), "contentEnd");
    }

    #[test]
    fn session_update_includes_inference_config() {
        let t = transform();
        let event = r#"{"type":"session.update","session":{"max_response_output_tokens":500,"temperature":0.5}}"#;
        let out = t.transform_client_event(event).unwrap();
        let session_start = parse(&out[0]);
        let inference = &session_start["event"]["sessionStart"]["inferenceConfiguration"];
        assert_eq!(inference["maxTokens"].as_u64(), Some(500));
        assert_eq!(inference["temperature"].as_f64(), Some(0.5));
    }

    #[test]
    fn session_update_with_tools_attaches_tool_configuration() {
        let t = transform();
        let event = r#"{"type":"session.update","session":{"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}}"#;
        let out = t.transform_client_event(event).unwrap();
        let prompt_start = parse(&out[1]);
        let ps = &prompt_start["event"]["promptStart"];
        assert!(
            ps.get("toolConfiguration").is_some(),
            "must have toolConfiguration"
        );
        assert!(ps.get("toolUseOutputConfiguration").is_some());
        let tools = ps["toolConfiguration"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["toolSpec"]["name"].as_str(), Some("get_weather"));
    }

    #[test]
    fn session_update_pcm16_output_maps_to_24khz() {
        let t = transform();
        let event = r#"{"type":"session.update","session":{"output_audio_format":"pcm16"}}"#;
        let out = t.transform_client_event(event).unwrap();
        let prompt_start = parse(&out[1]);
        let sr =
            &prompt_start["event"]["promptStart"]["audioOutputConfiguration"]["sampleRateHertz"];
        assert_eq!(sr.as_u64(), Some(24_000));
    }

    /// A second `session.update` must NOT emit a bare `promptStart` without a
    /// preceding `promptEnd` — Nova Sonic requires every `promptStart` to be
    /// balanced by a `promptEnd` before a new prompt opens. Regression test
    /// for the duplicate-`promptStart` lifecycle bug.
    #[test]
    fn repeated_session_update_emits_prompt_end_before_new_prompt_start() {
        let t = transform();
        let event = r#"{"type":"session.update","session":{"voice":"alloy"}}"#;
        let first = t.transform_client_event(event).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(bedrock_event_type(&first[0]), "sessionStart");
        assert_eq!(bedrock_event_type(&first[1]), "promptStart");

        let second = t.transform_client_event(event).unwrap();
        // The second update must close the prior prompt first:
        //   promptEnd → sessionStart → promptStart
        assert_eq!(
            bedrock_event_type(&second[0]),
            "promptEnd",
            "second session.update must emit promptEnd before a new promptStart"
        );
        assert_eq!(bedrock_event_type(&second[1]), "sessionStart");
        assert_eq!(bedrock_event_type(&second[2]), "promptStart");

        // The new prompt must carry a fresh prompt name (distinct from the
        // first prompt's name).
        let first_ps = parse(&first[1]);
        let second_pe = parse(&second[0]);
        let first_name = first_ps["event"]["promptStart"]["promptName"]
            .as_str()
            .unwrap_or("");
        let ended_name = second_pe["event"]["promptEnd"]["promptName"]
            .as_str()
            .unwrap_or("");
        assert_eq!(
            first_name, ended_name,
            "promptEnd must close the prior prompt by name"
        );
        let second_ps = parse(&second[2]);
        let second_name = second_ps["event"]["promptStart"]["promptName"]
            .as_str()
            .unwrap_or("");
        assert_ne!(
            first_name, second_name,
            "new promptStart must use a fresh prompt name"
        );
    }

    // =========================================================================
    // Client → Upstream: input_audio_buffer.append
    // =========================================================================

    #[test]
    fn audio_append_first_time_opens_content_start() {
        let t = transform();
        // Must send session.update first to set prompt_started.
        t.transform_client_event(r#"{"type":"session.update","session":{}}"#)
            .unwrap();
        let event = r#"{"type":"input_audio_buffer.append","audio":"SGVsbG8="}"#;
        let out = t.transform_client_event(event).unwrap();
        // contentStart (AUDIO) + audioInput
        assert_eq!(out.len(), 2);
        assert_eq!(bedrock_event_type(&out[0]), "contentStart");
        assert_eq!(bedrock_event_type(&out[1]), "audioInput");
        let audio_input = parse(&out[1]);
        assert_eq!(
            audio_input["event"]["audioInput"]["content"].as_str(),
            Some("SGVsbG8=")
        );
    }

    #[test]
    fn audio_append_second_time_no_content_start() {
        let t = transform();
        t.transform_client_event(r#"{"type":"session.update","session":{}}"#)
            .unwrap();
        t.transform_client_event(r#"{"type":"input_audio_buffer.append","audio":"AAA="}"#)
            .unwrap();
        let out = t
            .transform_client_event(r#"{"type":"input_audio_buffer.append","audio":"BBB="}"#)
            .unwrap();
        // Only audioInput — content already open.
        assert_eq!(out.len(), 1);
        assert_eq!(bedrock_event_type(&out[0]), "audioInput");
    }

    // =========================================================================
    // Client → Upstream: input_audio_buffer.commit
    // =========================================================================

    #[test]
    fn audio_commit_closes_open_audio_content() {
        let t = transform();
        t.transform_client_event(r#"{"type":"session.update","session":{}}"#)
            .unwrap();
        t.transform_client_event(r#"{"type":"input_audio_buffer.append","audio":"AAA="}"#)
            .unwrap();
        let out = t
            .transform_client_event(r#"{"type":"input_audio_buffer.commit"}"#)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(bedrock_event_type(&out[0]), "contentEnd");
    }

    #[test]
    fn audio_commit_without_open_content_is_noop() {
        let t = transform();
        let out = t
            .transform_client_event(r#"{"type":"input_audio_buffer.commit"}"#)
            .unwrap();
        assert!(out.is_empty());
    }

    // =========================================================================
    // Client → Upstream: conversation.item.create
    // =========================================================================

    #[test]
    fn conversation_item_create_text_produces_trio() {
        let t = transform();
        let event = r#"{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Hello world"}]}}"#;
        let out = t.transform_client_event(event).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(bedrock_event_type(&out[0]), "contentStart");
        assert_eq!(bedrock_event_type(&out[1]), "textInput");
        assert_eq!(bedrock_event_type(&out[2]), "contentEnd");
        let text_input = parse(&out[1]);
        assert_eq!(
            text_input["event"]["textInput"]["content"].as_str(),
            Some("Hello world")
        );
    }

    #[test]
    fn conversation_item_create_tool_result_produces_trio() {
        let t = transform();
        let event = r#"{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_123","output":"sunny"}}"#;
        let out = t.transform_client_event(event).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(bedrock_event_type(&out[0]), "contentStart");
        assert_eq!(bedrock_event_type(&out[1]), "toolResult");
        assert_eq!(bedrock_event_type(&out[2]), "contentEnd");
        let cs = parse(&out[0]);
        assert_eq!(cs["event"]["contentStart"]["role"].as_str(), Some("TOOL"));
        assert_eq!(
            cs["event"]["contentStart"]["toolResultInputConfiguration"]["toolUseId"].as_str(),
            Some("call_123")
        );
        let tr = parse(&out[1]);
        assert_eq!(tr["event"]["toolResult"]["content"].as_str(), Some("sunny"));
    }

    // =========================================================================
    // Client → Upstream: response.create
    // =========================================================================

    #[test]
    fn text_only_response_create_is_explicitly_unsupported() {
        let t = transform();
        t.transform_client_event(r#"{"type":"session.update","session":{}}"#)
            .unwrap();
        let error = t
            .transform_client_event(r#"{"type":"response.create"}"#)
            .unwrap_err();
        assert!(matches!(error, Error::BadRequest(_)));
        assert!(error.to_string().contains("text-only"));
        assert!(
            !t.state.lock().unwrap().audio_content_started,
            "must not inject synthetic audio"
        );
    }

    #[test]
    fn response_create_suppressed_when_client_streamed_audio() {
        let t = transform();
        t.transform_client_event(r#"{"type":"session.update","session":{}}"#)
            .unwrap();
        t.transform_client_event(r#"{"type":"input_audio_buffer.append","audio":"AAA="}"#)
            .unwrap();
        let out = t
            .transform_client_event(r#"{"type":"response.create"}"#)
            .unwrap();
        assert!(
            out.is_empty(),
            "trigger audio suppressed when client streamed audio"
        );
    }

    #[test]
    fn response_create_before_session_update_errors() {
        let t = transform();
        let error = t
            .transform_client_event(r#"{"type":"response.create"}"#)
            .unwrap_err();
        assert!(error.to_string().contains("session.update"));
    }

    // =========================================================================
    // Client → Upstream: edge cases
    // =========================================================================

    #[test]
    fn invalid_json_client_event_returns_error() {
        let t = transform();
        // Malformed frames are now surfaced through the error channel instead
        // of being silently dropped (the host logs + drops at the pump).
        assert!(t.transform_client_event("not json{{{").is_err());
    }

    #[test]
    fn unknown_client_event_type_returns_empty() {
        let t = transform();
        let out = t
            .transform_client_event(r#"{"type":"some.unknown.event"}"#)
            .unwrap();
        assert!(out.is_empty());
    }

    // =========================================================================
    // Upstream → Client: sessionStart
    // =========================================================================

    #[test]
    fn upstream_session_start_produces_session_created() {
        let t = transform();
        let event = r#"{"event":{"sessionStart":{"inferenceConfiguration":{"maxTokens":1024}}}}"#;
        let out = t.transform_upstream_event(event).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(openai_event_type(&out[0]), "session.created");
        let v = parse(&out[0]);
        assert_eq!(
            v["session"]["model"].as_str(),
            Some("amazon.nova-sonic-v1:0")
        );
    }

    // =========================================================================
    // Upstream → Client: contentStart (ASSISTANT)
    // =========================================================================

    #[test]
    fn upstream_assistant_content_start_text_produces_three_events() {
        let t = transform();
        let event =
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT","contentName":"c1"}}}"#;
        let out = t.transform_upstream_event(event).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(openai_event_type(&out[0]), "response.created");
        assert_eq!(openai_event_type(&out[1]), "response.output_item.added");
        assert_eq!(openai_event_type(&out[2]), "response.content_part.added");
        let part = &parse(&out[2])["part"];
        assert_eq!(part["type"].as_str(), Some("text"));
    }

    #[test]
    fn upstream_assistant_content_start_audio_produces_audio_part() {
        let t = transform();
        let event =
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"AUDIO","contentName":"c1"}}}"#;
        let out = t.transform_upstream_event(event).unwrap();
        assert_eq!(out.len(), 3);
        let part = &parse(&out[2])["part"];
        assert_eq!(part["type"].as_str(), Some("audio"));
    }

    #[test]
    fn upstream_non_assistant_content_start_is_dropped() {
        let t = transform();
        let event = r#"{"event":{"contentStart":{"role":"USER","type":"TEXT"}}}"#;
        let out = t.transform_upstream_event(event).unwrap();
        assert!(out.is_empty());
    }

    // =========================================================================
    // Upstream → Client: error events
    // =========================================================================

    /// A Bedrock upstream `error` event must be surfaced as an OpenAI Realtime
    /// `error` event — NOT silently dropped. Without this the client would
    /// observe a hung WebSocket with no indication the session failed.
    #[test]
    fn upstream_error_event_surfaces_as_openai_error() {
        let t = transform();
        let event = r#"{"event":{"error":{"message":"rate limited","code":"429"}}}"#;
        let out = t.transform_upstream_event(event).unwrap();
        assert_eq!(out.len(), 1);
        let v = parse(&out[0]);
        assert_eq!(v["type"].as_str(), Some("error"));
        assert_eq!(v["error"]["message"].as_str(), Some("rate limited"));
        assert_eq!(v["error"]["code"].as_str(), Some("429"));
    }

    /// A Bedrock error event without a `message` field still produces an
    /// OpenAI-shaped error event with a fallback message, so the client is
    /// never left without any signal.
    #[test]
    fn upstream_error_event_without_message_uses_fallback() {
        let t = transform();
        let event = r#"{"event":{"error":{"code":"500"}}}"#;
        let out = t.transform_upstream_event(event).unwrap();
        assert_eq!(out.len(), 1);
        let v = parse(&out[0]);
        assert_eq!(v["type"].as_str(), Some("error"));
        assert!(v["error"]["message"].as_str().is_some());
    }

    #[test]
    fn header_stripped_eventstream_exceptions_fail_the_realtime_session() {
        let t = transform();
        assert!(matches!(
            t.transform_upstream_event(r#"{"message":"upstream failed"}"#),
            Err(Error::Upstream { status: 500, .. })
        ));
        assert!(matches!(
            t.transform_upstream_event(r#"{"throttlingException":{"message":"slow down"}}"#),
            Err(Error::Upstream { status: 429, .. })
        ));
    }

    // =========================================================================
    // Upstream → Client: textOutput / audioOutput
    // =========================================================================

    #[test]
    fn upstream_text_output_produces_text_delta() {
        let t = transform();
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
        )
        .unwrap();
        let out = t
            .transform_upstream_event(
                r#"{"event":{"textOutput":{"content":"Hello ","contentName":"c1"}}}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(openai_event_type(&out[0]), "response.text.delta");
        assert_eq!(parse(&out[0])["delta"].as_str(), Some("Hello "));
    }

    #[test]
    fn upstream_audio_output_produces_audio_delta() {
        let t = transform();
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"AUDIO"}}}"#,
        )
        .unwrap();
        let out = t
            .transform_upstream_event(
                r#"{"event":{"audioOutput":{"content":"base64data","contentName":"c1"}}}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(openai_event_type(&out[0]), "response.audio.delta");
        assert_eq!(parse(&out[0])["delta"].as_str(), Some("base64data"));
    }

    #[test]
    fn upstream_text_output_without_active_response_is_dropped() {
        let t = transform();
        let out = t
            .transform_upstream_event(r#"{"event":{"textOutput":{"content":"Hello"}}}"#)
            .unwrap();
        assert!(out.is_empty(), "must have contentStart first");
    }

    // =========================================================================
    // Upstream → Client: contentEnd
    // =========================================================================

    #[test]
    fn upstream_content_end_text_produces_done_events() {
        let t = transform();
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
        )
        .unwrap();
        t.transform_upstream_event(r#"{"event":{"textOutput":{"content":"Hi there"}}}"#)
            .unwrap();
        let out = t
            .transform_upstream_event(
                r#"{"event":{"contentEnd":{"contentName":"c1","stopReason":"END_TURN"}}}"#,
            )
            .unwrap();
        // text.done + content_part.done + output_item.done + response.done
        assert!(out.len() >= 4);
        assert_eq!(openai_event_type(&out[0]), "response.text.done");
        assert_eq!(parse(&out[0])["text"].as_str(), Some("Hi there"));
        assert_eq!(openai_event_type(&out[1]), "response.content_part.done");
        assert_eq!(openai_event_type(&out[2]), "response.output_item.done");
        assert_eq!(openai_event_type(&out[3]), "response.done");
    }

    #[test]
    fn upstream_content_end_audio_produces_audio_done() {
        let t = transform();
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"AUDIO"}}}"#,
        )
        .unwrap();
        let out = t
            .transform_upstream_event(
                r#"{"event":{"contentEnd":{"contentName":"c1","stopReason":"END_TURN"}}}"#,
            )
            .unwrap();
        // audio.done + content_part.done + output_item.done + response.done
        assert!(out.len() >= 3);
        assert_eq!(openai_event_type(&out[0]), "response.audio.done");
    }

    #[test]
    fn upstream_content_end_without_end_turn_no_response_done() {
        let t = transform();
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
        )
        .unwrap();
        let out = t
            .transform_upstream_event(r#"{"event":{"contentEnd":{"contentName":"c1"}}}"#)
            .unwrap();
        // text.done + content_part.done + output_item.done — NO response.done
        let types: Vec<_> = out.iter().map(|s| openai_event_type(s)).collect();
        assert!(types.contains(&"response.text.done".to_string()));
        assert!(types.contains(&"response.output_item.done".to_string()));
        assert!(!types.contains(&"response.done".to_string()));
    }

    // =========================================================================
    // Upstream → Client: toolUse
    // =========================================================================

    #[test]
    fn native_tool_use_emits_the_complete_function_call_lifecycle() {
        let t = transform();
        let start = t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"TOOL","type":"TOOL","contentId":"c1","completionId":"turn1"}}}"#
        ).unwrap();
        assert_eq!(start.len(), 1);
        assert_eq!(openai_event_type(&start[0]), "response.created");
        let arguments = t.transform_upstream_event(
            r#"{"event":{"toolUse":{"contentId":"c1","toolUseId":"tu_1","toolName":"get_weather","content":"{\"city\":\"SF\"}"}}}"#
        ).unwrap();
        assert_eq!(arguments.len(), 2);
        let added = parse(&arguments[0]);
        assert_eq!(added["type"], "response.output_item.added");
        assert_eq!(added["item"]["type"], "function_call");
        assert_eq!(added["item"]["call_id"], "tu_1");
        assert_eq!(added["item"]["name"], "get_weather");
        assert_eq!(added["item"]["arguments"], "");
        assert_eq!(parse(&arguments[1])["delta"], r#"{"city":"SF"}"#);
        let done = t.transform_upstream_event(
            r#"{"event":{"contentEnd":{"contentId":"c1","completionId":"turn1","type":"TOOL","stopReason":"TOOL_USE"}}}"#
        ).unwrap();
        assert_eq!(
            done.iter()
                .map(|event| openai_event_type(event))
                .collect::<Vec<_>>(),
            vec![
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.done"
            ]
        );
        let finished = parse(&done[1]);
        assert_eq!(finished["item"]["id"], added["item"]["id"]);
        assert_eq!(finished["response_id"], added["response_id"]);
        assert_eq!(finished["item"]["arguments"], r#"{"city":"SF"}"#);
        assert_eq!(parse(&done[2])["response"]["output"][0], finished["item"]);
        assert!(parse(&done[2])["response"]["usage"].is_null());
        let result = t.transform_client_event(
            r#"{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"tu_1","output":"sunny"}}"#
        ).unwrap();
        assert_eq!(
            parse(&result[0])["event"]["contentStart"]["toolResultInputConfiguration"]["toolUseId"],
            "tu_1"
        );
        assert_eq!(parse(&result[1])["event"]["toolResult"]["content"], "sunny");
    }

    #[test]
    fn malformed_tool_arguments_are_not_replaced_with_an_empty_object() {
        let t = transform();
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"TOOL","type":"TOOL","contentId":"c1"}}}"#,
        )
        .unwrap();
        assert!(t.transform_upstream_event(
            r#"{"event":{"toolUse":{"contentId":"c1","toolUseId":"tu_1","toolName":"get_weather","content":"not json"}}}"#
        ).is_err());
    }

    #[test]
    fn flat_realtime_tools_are_forwarded_to_nova() {
        let t = transform();
        let out = t.transform_client_event(r#"{"type":"session.update","session":{"tools":[{
            "type":"function","name":"weather","description":"Get weather","parameters":{"type":"object"}
        }]}}"#).unwrap();
        let tool =
            &parse(&out[1])["event"]["promptStart"]["toolConfiguration"]["tools"][0]["toolSpec"];
        assert_eq!(tool["name"], "weather");
        assert_eq!(tool["inputSchema"]["json"], r#"{"type":"object"}"#);
    }

    #[test]
    fn unsupported_cancel_does_not_mutate_session_and_shutdown_closes_input() {
        let t = transform();
        t.transform_client_event(r#"{"type":"session.update","session":{}}"#)
            .unwrap();
        let audio = t
            .transform_client_event(r#"{"type":"input_audio_buffer.append","audio":"AAAA"}"#)
            .unwrap();
        let error = t
            .transform_client_event(r#"{"type":"response.cancel"}"#)
            .unwrap_err();
        assert!(error.to_string().contains("not supported"));
        assert!(t.state.lock().unwrap().audio_content_started);
        let shutdown = t.shutdown_events();
        assert_eq!(
            shutdown
                .iter()
                .map(|event| bedrock_event_type(event))
                .collect::<Vec<_>>(),
            vec!["contentEnd", "promptEnd", "sessionEnd"]
        );
        assert_eq!(
            parse(&shutdown[0])["event"]["contentEnd"]["contentName"],
            parse(&audio[0])["event"]["contentStart"]["contentName"]
        );
        assert!(t.shutdown_events().is_empty());
    }

    // =========================================================================
    // Upstream → Client: promptEnd / completionEnd
    // =========================================================================

    #[test]
    fn upstream_prompt_end_produces_response_done() {
        let t = transform();
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
        )
        .unwrap();
        let out = t
            .transform_upstream_event(r#"{"event":{"promptEnd":{"promptName":"p1"}}}"#)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(openai_event_type(&out[0]), "response.done");
    }

    // =========================================================================
    // Upstream → Client: edge cases
    // =========================================================================

    #[test]
    fn invalid_json_upstream_event_returns_error() {
        let t = transform();
        assert!(t.transform_upstream_event("garbage{{{").is_err());
    }

    #[test]
    fn unknown_upstream_event_returns_empty() {
        let t = transform();
        let out = t
            .transform_upstream_event(r#"{"event":{"someUnknownEvent":{}}}"#)
            .unwrap();
        assert!(out.is_empty());
    }

    // =========================================================================
    // Full round-trip: session → audio → response
    // =========================================================================

    #[test]
    fn speech_session_round_trip() {
        let t = transform();

        // Client: session.update
        let out = t
            .transform_client_event(
                r#"{"type":"session.update","session":{"voice":"alloy","instructions":"Be brief."}}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 5); // sessionStart + promptStart + content trio

        // Client: real audio; generation is triggered by Nova's speech VAD.
        let out = t
            .transform_client_event(r#"{"type":"input_audio_buffer.append","audio":"AAAA"}"#)
            .unwrap();
        assert!(out.len() >= 2);

        // Upstream: sessionStart → session.created
        let out = t
            .transform_upstream_event(
                r#"{"event":{"sessionStart":{"inferenceConfiguration":{"maxTokens":1024}}}}"#,
            )
            .unwrap();
        assert_eq!(openai_event_type(&out[0]), "session.created");

        // Upstream: contentStart (ASSISTANT TEXT)
        let out = t
            .transform_upstream_event(
                r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
            )
            .unwrap();
        assert_eq!(out.len(), 3);

        // Upstream: textOutput
        let out = t
            .transform_upstream_event(r#"{"event":{"textOutput":{"content":"Hi!"}}}"#)
            .unwrap();
        assert_eq!(openai_event_type(&out[0]), "response.text.delta");

        // Upstream: contentEnd (END_TURN) → done events
        let out = t
            .transform_upstream_event(r#"{"event":{"contentEnd":{"stopReason":"END_TURN"}}}"#)
            .unwrap();
        let types: Vec<_> = out.iter().map(|s| openai_event_type(s)).collect();
        assert!(types.contains(&"response.text.done".to_string()));
        assert!(types.contains(&"response.done".to_string()));
    }

    // =========================================================================
    // ws_url / ws_auth_headers
    // =========================================================================

    fn dep(model: &str) -> Deployment {
        Deployment {
            model_name: model.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn ws_url_default_describes_native_http_operation_without_claiming_websocket_support() {
        let t = transform();
        let url = t.ws_url(&dep("bedrock/amazon.nova-sonic-v1:0"));
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-sonic-v1%3A0/invoke-with-bidirectional-stream"
        );
    }

    /// The operation descriptor uses the same suffix normalization as the host.
    #[test]
    fn ws_url_strips_throughput_and_context_suffixes() {
        let t = transform();
        for model in [
            "bedrock/us.amazon.nova-sonic-v1:0:51k",
            "bedrock/us.amazon.nova-sonic-v1:0[1m]",
            "bedrock/us.amazon.nova-sonic-v1:0:51k[1m]",
            "bedrock_mantle/us.amazon.nova-sonic-v1:0[1m]:51k",
        ] {
            assert_eq!(
                t.ws_url(&dep(model)),
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.amazon.nova-sonic-v1%3A0/invoke-with-bidirectional-stream"
            );
        }
    }

    #[test]
    fn ws_url_explicit_bridge_preserves_arn_prefix_query_and_scheme() {
        let arn = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/sonic";
        for scheme in ["ws", "wss", "http", "https"] {
            let mut configured = dep(&format!("bedrock/{arn}"));
            configured.oxllm_params.api_base = Some(format!(
                "{scheme}://bridge.example:8443/tenant/bedrock/?tenant=a%2Fb"
            ));
            configured.model_info = Some(oxllm_core::deployment::ModelInfo {
                base_model: Some("amazon.nova-sonic-v1:0".into()),
                ..Default::default()
            });
            let url = transform().ws_url(&configured);
            assert_eq!(
                url,
                format!(
                    "{scheme}://bridge.example:8443/tenant/bedrock/model/arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Aapplication-inference-profile%2Fsonic/invoke-with-bidirectional-stream?tenant=a%2Fb"
                )
            );
            let segment = url
                .split_once("/model/")
                .unwrap()
                .1
                .split_once("/invoke-with-bidirectional-stream")
                .unwrap()
                .0;
            assert_eq!(
                percent_encoding::percent_decode_str(segment)
                    .decode_utf8()
                    .unwrap(),
                arn
            );
            assert_eq!(configured.oxllm_params.model, format!("bedrock/{arn}"));
        }
    }

    #[test]
    fn ws_url_default_uses_configured_region_and_keeps_runtime_arn() {
        let arn = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/sonic";
        let mut configured = dep(&format!("bedrock/{arn}"));
        configured
            .oxllm_params
            .extra
            .insert("aws_region_name".into(), json!("eu-west-1"));
        configured
            .oxllm_params
            .extra
            .insert("base_model".into(), json!("amazon.nova-sonic-v1:0"));
        assert_eq!(
            transform().ws_url(&configured),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Aapplication-inference-profile%2Fsonic/invoke-with-bidirectional-stream"
        );
    }

    #[test]
    fn ws_auth_headers_empty_for_bedrock() {
        let t = transform();
        assert!(
            t.ws_auth_headers(&dep("bedrock/amazon.nova-sonic-v1:0"))
                .is_empty()
        );
    }

    // =========================================================================
    // map_audio_format_to_sample_rate
    // =========================================================================

    #[test]
    fn pcm16_output_is_24khz() {
        assert_eq!(map_audio_format_to_sample_rate("pcm16", true), 24_000);
    }

    #[test]
    fn pcm16_input_is_16khz() {
        assert_eq!(map_audio_format_to_sample_rate("pcm16", false), 16_000);
    }

    #[test]
    fn g711_is_8khz() {
        assert_eq!(map_audio_format_to_sample_rate("g711_ulaw", true), 8_000);
        assert_eq!(map_audio_format_to_sample_rate("g711_alaw", false), 8_000);
    }

    // =========================================================================
    // Metering: extract_usage + response.done usage threading (HIGH)
    // =========================================================================

    /// `extract_usage` returns `None` for events without a usage payload.
    #[test]
    fn extract_usage_returns_none_for_non_usage_event() {
        let t = transform();
        assert!(
            t.extract_usage(r#"{"event":{"contentStart":{"role":"ASSISTANT"}}}"#)
                .is_none()
        );
    }

    /// A standalone Nova Sonic `usage` event is parsed into a `UsageSummary`
    /// whose fields match the payload — this is what the host's metering hook
    /// accumulates to bill the session. Previously `extract_usage` defaulted
    /// to `None`, so every realtime session settled zero spend.
    #[test]
    fn extract_usage_parses_standalone_usage_event() {
        let t = transform();
        let event = r#"{"event":{"usage":{"inputTokens":12,"outputTokens":34,"totalTokens":46,"cacheReadInputTokens":5}}}"#;
        let u = t.extract_usage(event).expect("usage must be parsed");
        assert_eq!(u.prompt_tokens, 12);
        assert_eq!(u.completion_tokens, 34);
        assert_eq!(u.total_tokens, 46);
        assert_eq!(u.cache_read_input_tokens, Some(5));
    }

    /// Usage nested inside `completionEnd` is also recognised.
    #[test]
    fn extract_usage_parses_completion_end_usage() {
        let t = transform();
        let event = r#"{"event":{"completionEnd":{"usage":{"inputTokens":7,"outputTokens":9,"totalTokens":16}}}}"#;
        let u = t.extract_usage(event).expect("usage must be parsed");
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.completion_tokens, 9);
        assert_eq!(u.total_tokens, 16);
    }

    #[test]
    fn native_usage_deltas_are_metered_once_and_late_usage_precedes_response_done() {
        let t = transform();
        t.transform_upstream_event(r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT","contentId":"c","completionId":"r"}}}"#).unwrap();
        t.transform_upstream_event(
            r#"{"event":{"textOutput":{"contentId":"c","content":"Hello"}}}"#,
        )
        .unwrap();
        let ended = t.transform_upstream_event(r#"{"event":{"contentEnd":{"contentId":"c","completionId":"r","stopReason":"END_TURN"}}}"#).unwrap();
        assert!(
            !ended
                .iter()
                .any(|event| openai_event_type(event) == "response.done")
        );
        let mut total = UsageSummary::default();
        for cumulative in [100, 200] {
            let event = json!({"event":{"usageEvent":{
                "completionId":"r", "totalInputTokens":cumulative,
                "totalOutputTokens":cumulative, "totalTokens":cumulative * 2,
                "details":{"delta":{
                    "input":{"speechTokens":3,"textTokens":2},
                    "output":{"speechTokens":7,"textTokens":1}
                }}
            }}})
            .to_string();
            total.merge(&t.extract_usage(&event).unwrap());
            assert!(t.transform_upstream_event(&event).unwrap().is_empty());
        }
        assert_eq!(total.prompt_tokens, 10);
        assert_eq!(total.completion_tokens, 16);
        assert_eq!(total.audio_input_tokens, Some(6));
        assert_eq!(
            total.total_tokens, 26,
            "must sum deltas, not cumulative totals"
        );
        let done = t
            .transform_upstream_event(
                r#"{"event":{"completionEnd":{"completionId":"r","stopReason":"END_TURN"}}}"#,
            )
            .unwrap();
        let response = &parse(&done[0])["response"];
        assert_eq!(response["usage"]["total_tokens"], 26);
        assert_eq!(response["output"][0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn empty_usage_is_absent_and_user_content_does_not_finish_assistant_output() {
        let t = transform();
        assert!(t.extract_usage(r#"{"event":{"usage":{}}}"#).is_none());
        assert!(t.extract_usage(r#"{"event":{"usageEvent":{}}}"#).is_none());
        t.transform_upstream_event(r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT","contentId":"assistant"}}}"#).unwrap();
        assert!(
            t.transform_upstream_event(
                r#"{"event":{"textOutput":{"contentId":"user","content":"echo"}}}"#
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            t.transform_upstream_event(
                r#"{"event":{"contentEnd":{"contentId":"user","stopReason":"END_TURN"}}}"#
            )
            .unwrap()
            .is_empty()
        );
    }

    /// A realistic Nova Sonic event sequence must (a) surface usage to the
    /// metering hook via `extract_usage` and (b) thread the same non-zero
    /// totals into the client-facing `response.done`. Previously
    /// `response.done` hardcoded all-zero usage regardless of what upstream
    /// reported.
    #[test]
    fn realtime_session_surfaces_nonzero_usage_in_response_done() {
        let t = transform();

        // Drive a full turn:
        //   contentStart(ASSISTANT,TEXT) → textOutput → contentEnd(END_TURN)
        //   → a standalone usage event
        t.transform_upstream_event(
            r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
        )
        .unwrap();
        t.transform_upstream_event(r#"{"event":{"textOutput":{"content":"Hi"}}}"#)
            .unwrap();
        let done = t
            .transform_upstream_event(r#"{"event":{"contentEnd":{"stopReason":"END_TURN"}}}"#)
            .unwrap();
        // The usage event arrives after END_TURN.
        let usage_event =
            r#"{"event":{"usage":{"inputTokens":11,"outputTokens":22,"totalTokens":33}}}"#;
        assert!(
            t.extract_usage(usage_event).is_some(),
            "usage event must be surfaced to metering"
        );

        // Without a usage event before the legacy END_TURN it carries null
        // (none was reported yet); the metering hook still captures the
        // later usage event. Now drive a second turn where usage precedes
        // END_TURN so response.done carries real totals.
        let second = t
            .transform_upstream_event(
                r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
            )
            .unwrap();
        assert!(
            second
                .iter()
                .any(|s| openai_event_type(s) == "response.created"),
            "a new response must be created for the second turn"
        );

        // Feed the usage event BEFORE the END_TURN this time.
        let usage_out = t.transform_upstream_event(usage_event).unwrap();
        assert!(
            usage_out.is_empty(),
            "standalone usage event is metering-only, dropped from client stream"
        );

        let done2 = t
            .transform_upstream_event(r#"{"event":{"contentEnd":{"stopReason":"END_TURN"}}}"#)
            .unwrap();
        let response_done = done2
            .iter()
            .find(|s| openai_event_type(s) == "response.done")
            .expect("response.done must be emitted on END_TURN");
        let v = parse(response_done);
        assert_eq!(v["response"]["usage"]["input_tokens"], 11);
        assert_eq!(v["response"]["usage"]["output_tokens"], 22);
        assert_eq!(v["response"]["usage"]["total_tokens"], 33);

        // Sanity: the first turn's response.done was also emitted.
        assert!(
            done.iter().any(|s| openai_event_type(s) == "response.done"),
            "first turn must close with response.done"
        );
    }

    // =========================================================================
    // Multi-block realtime turns: distinct output ids / indices (MEDIUM)
    // =========================================================================

    /// A turn that emits two ASSISTANT content blocks (TEXT then AUDIO)
    /// before END_TURN must assign each block a distinct `item_id` and an
    /// incrementing `output_index` (0 then 1). Previously the second block
    /// reused the first's id and pinned `output_index: 0`, colliding on the
    /// client.
    #[test]
    fn multi_block_turn_assigns_distinct_item_ids_and_indices() {
        let t = transform();

        // First ASSISTANT content block (TEXT).
        let first = t
            .transform_upstream_event(
                r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"TEXT"}}}"#,
            )
            .unwrap();
        let added1 = first
            .iter()
            .find(|s| openai_event_type(s) == "response.output_item.added")
            .expect("first output_item.added");
        let v1 = parse(added1);
        let item1 = v1["item"]["id"].as_str().expect("item id").to_string();
        assert_eq!(v1["output_index"], 0);

        // Text delta for the first block.
        let delta1 = t
            .transform_upstream_event(r#"{"event":{"textOutput":{"content":"hello"}}}"#)
            .unwrap();
        let d1 = parse(&delta1[0]);
        assert_eq!(d1["output_index"], 0);
        assert_eq!(d1["item_id"], item1);

        // Second ASSISTANT content block (AUDIO) — NO END_TURN between.
        let second = t
            .transform_upstream_event(
                r#"{"event":{"contentStart":{"role":"ASSISTANT","type":"AUDIO"}}}"#,
            )
            .unwrap();
        // No new response.created for the second block (same response).
        assert!(
            !second
                .iter()
                .any(|s| openai_event_type(s) == "response.created"),
            "second block must not start a new response"
        );
        let added2 = second
            .iter()
            .find(|s| openai_event_type(s) == "response.output_item.added")
            .expect("second output_item.added");
        let v2 = parse(added2);
        let item2 = v2["item"]["id"].as_str().expect("item id").to_string();
        assert_eq!(v2["output_index"], 1, "second block must be index 1");
        assert_ne!(item1, item2, "each block gets a fresh item id");

        // Audio delta for the second block uses index 1 + the new item id.
        let delta2 = t
            .transform_upstream_event(r#"{"event":{"audioOutput":{"content":"AAAB"}}}"#)
            .unwrap();
        let d2 = parse(&delta2[0]);
        assert_eq!(d2["output_index"], 1);
        assert_eq!(d2["item_id"], item2);

        // Close both blocks; the END_TURN contentEnd emits output_item.done
        // for the second block at index 1, then response.done.
        let close = t
            .transform_upstream_event(r#"{"event":{"contentEnd":{"stopReason":"END_TURN"}}}"#)
            .unwrap();
        let done2 = close
            .iter()
            .find(|s| openai_event_type(s) == "response.output_item.done")
            .expect("output_item.done");
        assert_eq!(parse(done2)["output_index"], 1);
        assert!(
            close
                .iter()
                .any(|s| openai_event_type(s) == "response.done")
        );
    }

    // =========================================================================
    // Numeric safety: u64 → u32 saturating cast (LOW)
    // =========================================================================

    /// `max_response_output_tokens` above `u32::MAX` must saturate to
    /// `u32::MAX` rather than silently truncating to a small/garbage value.
    #[test]
    fn session_update_oversize_token_count_saturates_not_truncates() {
        let t = transform();
        let event = format!(
            r#"{{"type":"session.update","session":{{"max_response_output_tokens":{}}}}}"#,
            u64::MAX
        );
        let out = t.transform_client_event(&event).unwrap();
        let inference = &parse(&out[0])["event"]["sessionStart"]["inferenceConfiguration"];
        assert_eq!(
            inference["maxTokens"].as_u64(),
            Some(u64::from(u32::MAX)),
            "oversize token count must saturate to u32::MAX, not truncate"
        );
    }

    /// An oversize `output_sample_rate_hertz` saturates instead of truncating.
    #[test]
    fn session_update_oversize_sample_rate_saturates_not_truncates() {
        let t = transform();
        let event = format!(
            r#"{{"type":"session.update","session":{{"output_sample_rate_hertz":{}}}}}"#,
            u64::MAX
        );
        let out = t.transform_client_event(&event).unwrap();
        let sr =
            &parse(&out[1])["event"]["promptStart"]["audioOutputConfiguration"]["sampleRateHertz"];
        assert_eq!(
            sr.as_u64(),
            Some(u64::from(u32::MAX)),
            "oversize sample rate must saturate to u32::MAX, not truncate"
        );
    }
}
