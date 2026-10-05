use super::*;
use oxllm_core::Endpoint;
use oxllm_core::provider::{Provider, ResolvedCredentials};
use std::collections::HashMap;

// =============================================================================
// Ported from LiteLLM `tests/llm_translation/test_elevenlabs.py`.
//
// LiteLLM's ElevenLabs suite has three pieces: a live-call STT base class
// (`TestElevenLabsAudioTranscription`, model `elevenlabs/scribe_v1`), a
// diarize/`language_code` parameter-passthrough test (network-mocked), and a
// pure TTS transform suite (`TestElevenLabsTextToSpeechTransformation`). The
// ports below cover the observable, pure pieces: provider metadata, the
// `xi-api-key` auth scheme, the TTS/STT base URL, and model-prefix stripping.
// =============================================================================

/// Ported from `TestElevenLabsAudioTranscription.get_custom_llm_provider`:
/// the provider slug is `elevenlabs` and it advertises TTS (`AudioSpeech`)
/// and STT (`AudioTranscriptions`).
#[test]
fn provider_id_and_supports() {
    let p = ElevenLabsProvider;
    assert_eq!(p.id(), "elevenlabs");
    assert!(p.supports(Endpoint::AudioSpeech));
    assert!(p.supports(Endpoint::AudioTranscriptions));
    // ElevenLabs is audio-only.
    assert!(!p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
    assert!(!p.supports(Endpoint::Rerank));
    // No typed transform accessors: audio is served via passthrough.
    assert!(p.chat().is_none());
    assert!(p.embeddings().is_none());
    assert!(p.rerank().is_none());
}

/// Ported from `ElevenLabsTextToSpeechConfig.validate_environment` /
/// `ElevenLabsAudioTranscriptionConfig.validate_environment`: ElevenLabs
/// authenticates with a custom `xi-api-key: <key>` header (not `Authorization`).
#[test]
fn auth_uses_xi_api_key_header() {
    let creds = ResolvedCredentials::from_key("eleven-secret");
    let mut headers = HashMap::new();
    elevenlabs_auth_headers(&creds, &mut headers);
    assert_eq!(
        headers.get("xi-api-key").map(String::as_str),
        Some("eleven-secret")
    );
    // Must NOT also emit a Bearer Authorization header.
    assert!(!headers.contains_key("authorization"));
}

/// Ported from `ElevenLabsAudioTranscriptionConfig.get_complete_url` /
/// `ElevenLabsTextToSpeechConfig.TTS_BASE_URL`: the default base URL is
/// `https://api.elevenlabs.io/v1` (`/speech-to-text` and
/// `/text-to-speech/{voice_id}` are appended by the upstream call).
#[test]
fn default_base_url() {
    assert_eq!(DEFAULT_ELEVENLABS_BASE, "https://api.elevenlabs.io/v1");
}

/// Ported from `TestElevenLabsAudioTranscription.get_base_audio_transcription_call_args`:
/// the `elevenlabs/` prefix is stripped from the model id
/// (`elevenlabs/scribe_v1` → `scribe_v1`).
#[test]
fn strip_elevenlabs_prefix_cases() {
    assert_eq!(strip_elevenlabs_prefix("elevenlabs/scribe_v1"), "scribe_v1");
    assert_eq!(strip_elevenlabs_prefix("scribe_v1"), "scribe_v1");
    assert_eq!(
        strip_elevenlabs_prefix("elevenlabs/eleven_multilingual_v2"),
        "eleven_multilingual_v2"
    );
}

/// The provider must be registered under its slug in the default registry so
/// it can be selected as a deployment target.
#[test]
fn registered_in_default_registry() {
    let registry = crate::default_registry();
    let p = registry.get("elevenlabs").expect("elevenlabs registered");
    assert_eq!(p.id(), "elevenlabs");
    assert!(p.supports(Endpoint::AudioSpeech));
    assert!(p.supports(Endpoint::AudioTranscriptions));
}
