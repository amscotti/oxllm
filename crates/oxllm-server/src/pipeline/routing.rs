//! Checked host routing. Hard selectors survive model-group fallback, even
//! when the next group uses a different routing strategy.

use oxllm_core::router::{RequestContext, RoutingStrategy, RoutingStrategyKind, TagBased};

use crate::error::ApiError;
use crate::state::AppState;

const ROUTING_ERROR_CODE: &str = "routing_error";

fn routing_error(error: oxllm_core::Error) -> ApiError {
    let mut error = ApiError::from(error);
    error.code = Some(ROUTING_ERROR_CODE.to_string());
    error
}

pub(crate) fn is_routing_error(error: &ApiError) -> bool {
    error.code.as_deref() == Some(ROUTING_ERROR_CODE)
}

/// Materialize the primary group's default before cloning contexts for other
/// groups, so an operator-supplied hard default is not lost on fallback.
pub(crate) fn apply_default_tag(state: &AppState, ctx: &mut RequestContext) {
    if state.router.strategy_kind_for(&ctx.model) == RoutingStrategyKind::TagBased
        && ctx.tags.is_empty()
        && ctx.tag.as_deref().is_none_or(str::is_empty)
    {
        ctx.tag = state.router.config().default_tag.clone();
    }
}

pub(crate) fn try_pick_deployment_index(
    state: &AppState,
    ctx: &RequestContext,
) -> Result<Option<usize>, ApiError> {
    let hard_tags: Vec<String> = ctx
        .tags
        .iter()
        .map(String::as_str)
        .chain(ctx.tag.as_deref().filter(|_| ctx.tags.is_empty()))
        .filter(|tag| tag.len() > 1 && (tag.starts_with('&') || tag.starts_with('!')))
        .map(str::to_string)
        .collect();
    if hard_tags.is_empty()
        || state.router.strategy_kind_for(&ctx.model) == RoutingStrategyKind::TagBased
    {
        return state
            .router
            .try_pick_deployment_index(&state.deployments, ctx)
            .map_err(routing_error);
    }

    // Core enforces selectors before priority/pre-call filters for TagBased.
    // A fallback can select another strategy. Reuse the same checked tag
    // matcher to constrain its input rather than either discarding selectors
    // or changing the fallback's configured selection strategy.
    let mut hard_ctx = ctx.clone();
    hard_ctx.tags = hard_tags;
    hard_ctx.tag = None;
    let selector = TagBased::new();
    let available = state
        .router
        .deployments_for_model(&state.deployments, &ctx.model)
        .into_iter()
        .filter(|deployment| {
            !state
                .router
                .cooldown_store()
                .is_cooling(&state.router.cooldown_key(deployment))
        })
        .collect::<Vec<_>>();
    selector
        .try_pick(&available, &hard_ctx)
        .map_err(routing_error)?;
    if available.is_empty()
        && state.router.strategy_kind_for(&ctx.model) != RoutingStrategyKind::ComplexityRouter
    {
        return Ok(None);
    }

    let eligible_indices: Vec<usize> = state
        .deployments
        .iter()
        .enumerate()
        .filter_map(|(index, deployment)| {
            selector
                .try_pick(&[deployment], &hard_ctx)
                .ok()
                .flatten()
                .map(|_| index)
        })
        .collect();
    if eligible_indices.len() == state.deployments.len() {
        return state
            .router
            .try_pick_deployment_index(&state.deployments, ctx)
            .map_err(routing_error);
    }
    // Clone only on the cross-strategy hard-selector path. Original indices
    // are retained so the host still uses the live deployment for credentials,
    // pricing and accounting. Core's stores use stable deployment identities.
    let eligible = eligible_indices
        .iter()
        .map(|&index| state.deployments[index].clone())
        .collect::<Vec<_>>();
    state
        .router
        .try_pick_deployment_index(&eligible, ctx)
        .map(|index| index.map(|index| eligible_indices[index]))
        .map_err(routing_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_tags_filter_other_strategies_without_losing_original_indices() {
        let config = serde_json::from_value(serde_json::json!({
            "model_list": [
                {"model_name":"m","oxllm_params":{"model":"openai/gpt-4o","api_key":"first"},"model_info":{"tags":["blocked"]}},
                {"model_name":"m","oxllm_params":{"model":"openai/gpt-4o","api_key":"second"},"model_info":{"tags":["allowed"]}}
            ],
            "router_settings":{"routing_strategy":"round_robin"}
        })).unwrap();
        let state = AppState::with_registry(config, oxllm_providers::default_registry()).unwrap();
        let mut ctx = RequestContext::new("m");
        ctx.tags = vec!["&allowed".to_string()];
        assert_eq!(try_pick_deployment_index(&state, &ctx).unwrap(), Some(1));
        ctx.tags = vec!["&missing".to_string()];
        let error = try_pick_deployment_index(&state, &ctx).unwrap_err();
        assert!(is_routing_error(&error));
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
    }
}
