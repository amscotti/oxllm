//! Platform route group: batches, files, assistants, fine-tuning, and the
//! token-counter utility. These are management-style platform APIs that share
//! the standard proxy auth stack applied by
//! [`build_router`][super::build_router].
//!
//! Compiled only when the `platform-apis` Cargo feature is enabled. Slim
//! deployments can pass `--no-default-features` to produce a chat-only proxy.

#![cfg(feature = "platform-apis")]

use axum::Router;
use axum::routing::{delete, get, post};

use crate::handlers::{assistants, batches, files, fine_tuning, threads, token_counter};
use crate::state::SharedState;

/// Register the platform (batches / files / fine-tuning / assistants /
/// threads / token-counter) routes. The returned router carries no auth layers
/// `build_router` applies the standard proxy auth stack to the merged group.
pub(crate) fn routes() -> Router<SharedState> {
    Router::new()
        .route("/v1/batches", post(batches::create_batch))
        .route("/v1/batches", get(batches::list_batches))
        .route("/v1/batches/{batch_id}", get(batches::retrieve_batch))
        .route("/v1/batches/{batch_id}/cancel", post(batches::cancel_batch))
        .route("/v1/files", post(files::upload_file))
        .route("/v1/files", get(files::list_files))
        .route("/v1/files/{file_id}", get(files::retrieve_file))
        .route("/v1/files/{file_id}", delete(files::delete_file))
        .route(
            "/v1/files/{file_id}/content",
            get(files::retrieve_file_content),
        )
        .route(
            "/v1/fine_tuning/jobs",
            post(fine_tuning::create_fine_tuning_job),
        )
        .route(
            "/v1/fine_tuning/jobs",
            get(fine_tuning::list_fine_tuning_jobs),
        )
        .route(
            "/v1/fine_tuning/jobs/{job_id}",
            get(fine_tuning::retrieve_fine_tuning_job),
        )
        .route(
            "/v1/fine_tuning/jobs/{job_id}/cancel",
            post(fine_tuning::cancel_fine_tuning_job),
        )
        .route("/v1/assistants", post(assistants::create_assistant))
        .route("/v1/assistants", get(assistants::list_assistants))
        .route(
            "/v1/assistants/{assistant_id}",
            get(assistants::retrieve_assistant),
        )
        .route(
            "/v1/assistants/{assistant_id}",
            post(assistants::modify_assistant),
        )
        .route(
            "/v1/assistants/{assistant_id}",
            delete(assistants::delete_assistant),
        )
        .route("/v1/threads", post(threads::create_thread))
        .route("/v1/threads/runs", post(threads::create_run_v2))
        .route("/v1/threads/runs/stream", post(threads::create_run_stream))
        .route(
            "/v1/threads/{thread_id}",
            get(threads::retrieve_thread)
                .post(threads::modify_thread)
                .delete(threads::delete_thread),
        )
        .route(
            "/v1/threads/{thread_id}/messages",
            post(threads::create_message).get(threads::list_messages),
        )
        .route(
            "/v1/threads/{thread_id}/messages/{message_id}",
            get(threads::retrieve_message)
                .post(threads::modify_message)
                .delete(threads::delete_message),
        )
        .route(
            "/v1/threads/{thread_id}/runs",
            post(threads::create_run).get(threads::list_runs),
        )
        .route(
            "/v1/threads/{thread_id}/runs/{run_id}",
            get(threads::retrieve_run).post(threads::modify_run),
        )
        .route(
            "/v1/threads/{thread_id}/runs/{run_id}/cancel",
            post(threads::cancel_run),
        )
        .route("/utils/token_counter", post(token_counter::token_counter))
}
