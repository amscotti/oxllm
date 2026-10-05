//! Azure OpenAI legacy route aliases — the deployment- and engine-named
//! paths used by the Azure SDKs. These delegate to the existing handler
//! functions in [`crate::handlers::azure_legacy`].

use axum::Router;
use axum::routing::post;

use crate::handlers::azure_legacy;
use crate::state::SharedState;

/// Register the Azure legacy paths (`/openai/deployments/{model}/...` and
/// `/engines/{model}/...`). The returned router carries no auth layers
/// `build_router` applies the standard proxy auth stack to the merged group.
pub(crate) fn routes() -> Router<SharedState> {
    Router::new()
        .route(
            "/openai/deployments/{model}/chat/completions",
            post(azure_legacy::deployments_chat_completions),
        )
        .route(
            "/engines/{model}/chat/completions",
            post(azure_legacy::engines_chat_completions),
        )
        .route(
            "/openai/deployments/{model}/embeddings",
            post(azure_legacy::deployments_embeddings),
        )
        .route(
            "/openai/deployments/{model}/completions",
            post(azure_legacy::deployments_completions),
        )
}
