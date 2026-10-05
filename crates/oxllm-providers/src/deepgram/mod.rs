//! Deepgram provider — audio transcription.
//!
//! Wire target:
//! - transcription: `https://api.deepgram.com/v1/listen`
//!
//! Deepgram is an audio-only provider. It is **not** OpenAI-compatible and has
//! no chat/embeddings/rerank surface, so it implements only the [`Provider`]
//! trait and advertises [`Endpoint::AudioTranscriptions`]. This module provides
//! registration and API-key auth only. OpenAI audio compatibility still needs
//! host-side adaptation: `/listen` takes raw audio bytes (or JSON containing a
//! URL), model/options in the query, and returns a nested transcript under
//! `results.channels[].alternatives[]`. Forwarding OpenAI multipart unchanged
//! to `/audio/transcriptions` does not implement that contract.
//!
//! Auth note: Deepgram API keys use the `Token` scheme —
//! `Authorization: Token <key>`. Deepgram also supports temporary Bearer JWTs;
//! this adapter's credential mode is API keys. Because audio is served by
//! the passthrough surface (which has no typed audio trait), the scheme is
//! exposed as [`deepgram_auth_headers`] rather than a trait method.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::Endpoint;
use oxllm_core::provider::{Provider, ProviderId, ResolvedCredentials};

/// The default Deepgram base URL.
pub const DEFAULT_DEEPGRAM_BASE: &str = "https://api.deepgram.com/v1";

/// The Deepgram provider.
#[derive(Debug, Clone, Default)]
pub struct DeepgramProvider;

/// Strip the `deepgram/` provider prefix from a model id.
pub fn strip_deepgram_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("deepgram", rest)) => rest,
        _ => model,
    }
}

/// Build Deepgram's auth headers from resolved credentials.
///
/// Deepgram uses the `Token` auth scheme (`Authorization: Token <key>`), not
/// Bearer. Exposed as a public helper because audio is served by the
/// passthrough surface rather than a typed trait method.
pub fn deepgram_auth_headers(creds: &ResolvedCredentials, headers: &mut HashMap<String, String>) {
    let key = match creds {
        ResolvedCredentials::ApiKey { bearer } => bearer.as_str(),
        other => match other.best_effort_bearer() {
            Some(b) => b,
            None => return,
        },
    };
    headers.insert("authorization".to_string(), format!("Token {key}"));
}

impl Provider for DeepgramProvider {
    fn id(&self) -> ProviderId {
        "deepgram"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(endpoint, Endpoint::AudioTranscriptions)
    }
}

#[cfg(test)]
mod tests;
