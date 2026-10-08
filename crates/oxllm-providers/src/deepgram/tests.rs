use super::*;
use oxllm_core::Endpoint;
use oxllm_core::provider::{Provider, ResolvedCredentials};
use std::collections::HashMap;

// =============================================================================
// Ported from the reference `tests/llm_translation/test_deepgram.py`.
//
// The reference Deepgram suite (`TestDeepgramAudioTranscription`) extends a
// live-call base class that hits the network for `deepgram/nova-2`. The ports
// below cover the observable, pure pieces: provider metadata, the `Token`
// (not Bearer) auth scheme, the `/listen` base URL, and model-prefix
// stripping.
// =============================================================================

/// Ported from `TestDeepgramAudioTranscription.get_custom_llm_provider`: the
/// provider slug is `deepgram` and it advertises audio transcription only.
#[test]
fn provider_id_and_supports() {
    let p = DeepgramProvider;
    assert_eq!(p.id(), "deepgram");
    assert!(p.supports(Endpoint::AudioTranscriptions));
    // Deepgram is audio-transcription-only.
    assert!(!p.supports(Endpoint::AudioSpeech));
    assert!(!p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
    // No typed transform accessors: audio is served via passthrough.
    assert!(p.chat().is_none());
    assert!(p.embeddings().is_none());
    assert!(p.rerank().is_none());
}

/// Ported from `DeepgramAudioTranscriptionConfig.validate_environment`:
/// Deepgram authenticates with `Authorization: Token <key>` — **not** Bearer.
#[test]
fn auth_uses_token_scheme_not_bearer() {
    let creds = ResolvedCredentials::from_key("dg-secret");
    let mut headers = HashMap::new();
    deepgram_auth_headers(&creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Token dg-secret")
    );
}

/// Ported from `DeepgramAudioTranscriptionConfig.get_complete_url`: the
/// default base URL is `https://api.deepgram.com/v1` (the `/listen` path and
/// `?model=` query are appended by the upstream call).
#[test]
fn default_base_url() {
    assert_eq!(DEFAULT_DEEPGRAM_BASE, "https://api.deepgram.com/v1");
}

/// Ported from `TestDeepgramAudioTranscription.get_base_audio_transcription_call_args`:
/// the `deepgram/` prefix is stripped from the model id (`deepgram/nova-2` → `nova-2`).
#[test]
fn strip_deepgram_prefix_cases() {
    assert_eq!(strip_deepgram_prefix("deepgram/nova-2"), "nova-2");
    assert_eq!(strip_deepgram_prefix("nova-2"), "nova-2");
    assert_eq!(
        strip_deepgram_prefix("deepgram/nova-3/general"),
        "nova-3/general"
    );
}

/// The provider must be registered under its slug in the default registry so
/// it can be selected as a deployment target.
#[test]
fn registered_in_default_registry() {
    let registry = crate::default_registry();
    let p = registry.get("deepgram").expect("deepgram registered");
    assert_eq!(p.id(), "deepgram");
    assert!(p.supports(Endpoint::AudioTranscriptions));
    assert!(!p.supports(Endpoint::AudioSpeech));
}
