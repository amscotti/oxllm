//! The endpoint kinds the proxy can serve.
//!
//! Each variant corresponds to a distinct OpenAI-shaped API surface.
//! Adding a new variant is a bounded, documented change.

/// The kinds of LLM endpoint the proxy supports.
///
/// Marked `#[non_exhaustive]` so that adding a new variant is not a breaking
/// change for downstream crates. Consumers outside the defining crate must
/// include a `_` wildcard arm in `match` expressions.
///
/// ```rust
/// use oxllm_core::Endpoint;
///
/// // Unit variants are constructible from outside the crate; only exhaustive
/// // matching is forbidden (a `_` arm is required).
/// let endpoint = Endpoint::ChatCompletions;
/// let is_chat = match endpoint {
///     Endpoint::ChatCompletions => true,
///     _ => false,
/// };
/// assert!(is_chat);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Endpoint {
    /// `POST /v1/chat/completions` — the primary endpoint.
    ChatCompletions,
    /// `POST /v1/completions` — legacy text completions.
    Completions,
    /// `POST /v1/embeddings`.
    Embeddings,
    /// `POST /v1/rerank`.
    Rerank,
    /// `POST /v1/images/generations`.
    ImagesGenerations,
    /// `POST /v1/audio/transcriptions`.
    AudioTranscriptions,
    /// `POST /v1/audio/speech`.
    AudioSpeech,
    /// `POST /v1/moderations`.
    Moderations,
    /// `POST /v1/responses` — OpenAI Responses API.
    Responses,
    /// `POST /v1/messages` — Anthropic Messages (accepted + translated).
    Messages,
    /// `GET /v1/realtime` — websocket.
    Realtime,
    /// `POST /v1/batches` — OpenAI Batches API (passthrough).
    Batches,
    /// `POST /v1/files` — OpenAI Files API (passthrough).
    Files,
    /// `POST /v1/fine_tuning/jobs` — OpenAI Fine-tuning API (passthrough).
    FineTuning,
    /// `POST /v1/assistants` — OpenAI Assistants API (passthrough).
    Assistants,
    /// `POST /v1/threads` — OpenAI Threads API (passthrough).
    Threads,
    /// `POST /utils/token_counter` — server-only token estimation.
    TokenCounter,
}
