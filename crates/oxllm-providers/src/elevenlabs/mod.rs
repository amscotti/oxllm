//! ElevenLabs provider — text-to-speech + speech-to-text.
//!
//! Wire targets:
//! - TTS:     `https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`
//! - STT:     `https://api.elevenlabs.io/v1/speech-to-text`
//!
//! ElevenLabs is an audio-only provider. It is **not** OpenAI-compatible and
//! has no chat/embeddings/rerank surface, so it implements only the
//! [`Provider`] trait and advertises [`Endpoint::AudioSpeech`] (TTS) and
//! [`Endpoint::AudioTranscriptions`] (STT). This module provides registration
//! and API-key auth only. OpenAI audio compatibility needs host-side mapping:
//! TTS uses `/text-to-speech/{voice_id}`, `text`, `model_id` and a query
//! `output_format`; STT uses `/speech-to-text` with multipart `file` and
//! `model_id` (plus `language_code` when supplied). Forwarding OpenAI paths
//! and payloads unchanged does not implement these native contracts.
//!
//! Auth note: ElevenLabs authenticates with a custom `xi-api-key: <key>`
//! header rather than the `Authorization` scheme. Because audio is served by
//! the passthrough surface (which has no typed audio trait), the scheme is
//! exposed as [`elevenlabs_auth_headers`] rather than a trait method.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::Endpoint;
use oxllm_core::provider::{Provider, ProviderId, ResolvedCredentials};

/// The default ElevenLabs base URL.
pub const DEFAULT_ELEVENLABS_BASE: &str = "https://api.elevenlabs.io/v1";

/// The ElevenLabs provider.
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsProvider;

/// Strip the `elevenlabs/` provider prefix from a model id.
pub fn strip_elevenlabs_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("elevenlabs", rest)) => rest,
        _ => model,
    }
}

/// Build ElevenLabs' auth headers from resolved credentials.
///
/// ElevenLabs authenticates with a custom `xi-api-key: <key>` header rather
/// than the `Authorization` scheme. Exposed as a public helper because audio
/// is served by the passthrough surface rather than a typed trait method.
pub fn elevenlabs_auth_headers(creds: &ResolvedCredentials, headers: &mut HashMap<String, String>) {
    let key = match creds {
        ResolvedCredentials::ApiKey { bearer } => bearer.as_str(),
        other => match other.best_effort_bearer() {
            Some(b) => b,
            None => return,
        },
    };
    headers.insert("xi-api-key".to_string(), key.to_string());
}

impl Provider for ElevenLabsProvider {
    fn id(&self) -> ProviderId {
        "elevenlabs"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(
            endpoint,
            Endpoint::AudioSpeech | Endpoint::AudioTranscriptions
        )
    }
}

#[cfg(test)]
mod tests;
