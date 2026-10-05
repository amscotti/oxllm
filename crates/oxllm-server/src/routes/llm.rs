//! Core LLM route group: chat, completions, embeddings, messages, responses,
//! rerank, images, audio, moderations, models, and realtime.
//!
//! These routes share the same auth/allowed-IPs/auth-brake stack, which is
//! applied by [`build_router`][super::build_router] after merging the LLM and
//! platform groups.

use axum::Router;
use axum::routing::{delete, get, post};

use crate::handlers::{
    anthropic_batches, anthropic_count_tokens, anthropic_files, anthropic_messages_passthrough,
    chat, completions, embeddings, messages, models, passthrough, responses,
};
use crate::state::SharedState;

/// Register the core LLM (chat / completions / embeddings / messages /
/// responses / rerank / images / audio / moderations / models / realtime)
/// routes. The returned router carries no auth layers — `build_router`
/// applies the standard proxy auth stack to the merged group.
pub(crate) fn routes() -> Router<SharedState> {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/completions", post(completions::completions))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/embeddings", post(embeddings::embeddings))
        .route("/v1/messages", post(messages::messages))
        .route(
            "/v1/messages/count_tokens",
            post(anthropic_count_tokens::count_tokens),
        )
        .route(
            "/anthropic/v1/messages",
            post(anthropic_messages_passthrough::messages_passthrough),
        )
        // Anthropic Message Batches API (native `/v1/messages/batches` shape;
        // no route conflict with the OpenAI `/v1/batches` platform endpoints).
        .route(
            "/v1/messages/batches",
            post(anthropic_batches::create_batch),
        )
        .route("/v1/messages/batches", get(anthropic_batches::list_batches))
        .route(
            "/v1/messages/batches/{batch_id}",
            get(anthropic_batches::retrieve_batch),
        )
        .route(
            "/v1/messages/batches/{batch_id}/cancel",
            post(anthropic_batches::cancel_batch),
        )
        .route(
            "/v1/messages/batches/{batch_id}/results",
            get(anthropic_batches::retrieve_batch_results),
        )
        // Anthropic Files API. Mounted under `/anthropic/v1/files` to avoid
        // colliding with the OpenAI Files API (`/v1/files`) registered under
        // the `platform-apis` feature. Mirrors the `/anthropic/v1/messages`
        // native-passthrough convention.
        .route("/anthropic/v1/files", post(anthropic_files::upload_file))
        .route("/anthropic/v1/files", get(anthropic_files::list_files))
        .route(
            "/anthropic/v1/files/{file_id}",
            get(anthropic_files::retrieve_file),
        )
        .route(
            "/anthropic/v1/files/{file_id}/content",
            get(anthropic_files::download_file_content),
        )
        .route(
            "/anthropic/v1/files/{file_id}",
            delete(anthropic_files::delete_file),
        )
        .route("/v1/responses", post(responses::responses))
        .route("/v1/rerank", post(passthrough::rerank))
        .route(
            "/v1/images/generations",
            post(passthrough::images_generations),
        )
        .route("/v1/images/edits", post(passthrough::images_edits))
        .route(
            "/v1/images/variations",
            post(passthrough::images_variations),
        )
        .route(
            "/v1/audio/transcriptions",
            post(passthrough::audio_transcriptions),
        )
        .route(
            "/v1/audio/translations",
            post(passthrough::audio_translations),
        )
        .route("/v1/audio/speech", post(passthrough::audio_speech))
        .route("/v1/moderations", post(passthrough::moderations))
        .route("/v1/models", get(models::list_models))
        .route("/v1/models/{model}", get(models::retrieve_model))
        .route(
            "/v1/realtime",
            get(crate::realtime::handle_realtime_upgrade),
        )
}
