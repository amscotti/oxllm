//! Routing-state inspection endpoint: `GET /routing/state`.
//!
//! Returns a live snapshot of the router's runtime state: the effective
//! routing strategy (global plus per-group overrides), the registered
//! pre-call checks, and per-deployment cooling / health / bandit state.
//! None of the reported fields are secrets — the only config-derived
//! string per deployment is the provider model (e.g. `"openai/gpt-4o"`);
//! API keys are never included.

use std::collections::HashMap;
use std::time::Duration;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use oxllm_core::Deployment;
use oxllm_core::router::Router;
use serde_json::{Map, Value, json};

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::state::SharedState;

use super::require_master;

/// `GET /routing/state` — return the router's live state as JSON.
pub async fn get_routing_state(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let state = shared.load_full();
    Ok(Json(routing_state_json(&state.router, &state.deployments)).into_response())
}

/// Build the `GET /routing/state` response body from the live router and
/// the current deployment list.
///
/// Shape decisions:
/// - Deployment keys are the router's cooldown keys
///   (`{model_name}:{index}`, index = position in `deployments`) — the same
///   keys the orchestration loop stamps, so `cooling`, `healthy`, and
///   `bandit_cells` line up with what routing actually used.
/// - `bandit_cells` is omitted entirely for deployments with no recorded
///   cells: bandit state only exists once adaptive routing has run, and an
///   empty object would read as "adaptive but unobserved".
/// - `healthy` is the fail-open effective state (staleness applied); the
///   staleness check runs before `health_reason` is read, so a stale entry
///   reads healthy and reports no reason (`health_reason` is the last
///   recorded unhealthy reason, null when the store holds no entry).
fn routing_state_json(router: &Router, deployments: &[Deployment]) -> Value {
    let config = router.config();

    // Only groups with an explicit strategy override are listed; groups
    // without one follow the global `strategy`.
    let group_strategies: Map<String, Value> = config
        .routing_groups
        .iter()
        .filter_map(|(model, group)| {
            group
                .routing_strategy
                .map(|kind| (model.clone(), kind.as_str().into()))
        })
        .collect();

    // Group bandit cells by deployment key. Cell keys are exactly the
    // cooldown keys (`{model_name}:{provider_model}`) stamped by the
    // orchestration loop, so equality (not prefix matching) is the correct
    // join.
    let mut cells_by_key: HashMap<String, Map<String, Value>> = HashMap::new();
    for ((request_type, key), cell) in router.bandit_store().all_cells() {
        cells_by_key.entry(key).or_default().insert(
            request_type.as_str().to_string(),
            json!({
                "alpha": cell.alpha,
                "beta": cell.beta,
                // Recorded outcomes — reuse the cell's own accessor so a
                // future prior change propagates here automatically.
                "observations": cell.observations(),
            }),
        );
    }

    let staleness = Duration::from_secs(config.health_check_staleness_threshold_secs);
    let health = router.health_state();
    let cooldown = router.cooldown_store();

    let deployments: Vec<Value> = deployments
        .iter()
        .map(|dep| {
            let key = router.cooldown_key(dep);
            // `is_healthy` first: it applies staleness and lazily evicts
            // stale entries, so `reason` below reflects only live entries.
            let healthy = health.is_healthy(&key, staleness);
            let mut entry = json!({
                "key": key,
                "model_name": dep.model_name,
                "provider_model": dep.oxllm_params.model,
                "cooling": cooldown.is_cooling(&key),
                "healthy": healthy,
                "health_reason": health.reason(&key),
            });
            if let Some(cells) = cells_by_key.get(&key) {
                if let Some(obj) = entry.as_object_mut() {
                    obj.insert("bandit_cells".to_string(), Value::Object(cells.clone()));
                }
            }
            entry
        })
        .collect();

    json!({
        "strategy": config.routing_strategy.as_str(),
        "group_strategies": group_strategies,
        "pre_call_checks": router.pre_call_check_names(),
        "deployments": deployments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;
    use oxllm_core::router::{RequestType, RouterConfig, RouterGroupOverride, RoutingStrategyKind};

    fn dep(model_name: &str, provider_model: &str) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: provider_model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn default_router_serializes_expected_shape() {
        let router = Router::new(RouterConfig::default());
        let deps = vec![
            dep("gpt-4o", "openai/gpt-4o"),
            dep("gpt-4o", "azure/gpt-4o"),
        ];
        let v = routing_state_json(&router, &deps);

        assert_eq!(v["strategy"], "simple_shuffle");
        assert!(
            v["group_strategies"]
                .as_object()
                .expect("object")
                .is_empty()
        );
        assert_eq!(v["pre_call_checks"], json!([]));

        let deployments = v["deployments"].as_array().expect("array");
        assert_eq!(deployments.len(), 2);
        assert_eq!(deployments[0]["key"], router.cooldown_key(&deps[0]));
        assert_eq!(deployments[1]["key"], router.cooldown_key(&deps[1]));
        assert_eq!(deployments[0]["model_name"], "gpt-4o");
        assert_eq!(deployments[0]["provider_model"], "openai/gpt-4o");
        assert_eq!(deployments[1]["provider_model"], "azure/gpt-4o");
        assert_eq!(deployments[0]["cooling"], false);
        assert_eq!(deployments[0]["healthy"], true);
        assert!(deployments[0]["health_reason"].is_null());
        assert!(
            deployments[0].get("bandit_cells").is_none(),
            "no recorded cells → bandit_cells key omitted"
        );
    }

    #[test]
    fn group_override_and_runtime_state_are_reported() {
        let mut config = RouterConfig {
            enable_health_check_routing: true,
            ..RouterConfig::default()
        };
        config.routing_groups.insert(
            "gpt-4o".to_string(),
            RouterGroupOverride {
                routing_strategy: Some(RoutingStrategyKind::AdaptiveRouter),
                ..Default::default()
            },
        );
        let router = Router::new(config);
        let deps = vec![
            dep("gpt-4o", "openai/gpt-4o"),
            dep("claude-3", "anthropic/claude-3"),
        ];

        let d0_key = router.cooldown_key(&deps[0]);
        let d1_key = router.cooldown_key(&deps[1]);
        router.mark_cooldown(&d0_key);
        router
            .health_state()
            .mark_unhealthy(&d0_key, Some("dns".to_string()));
        // Two successes: alpha = 1 (prior) + 2 = 3, observations = 2.
        router
            .bandit_store()
            .record_success(&(RequestType::Code, d0_key.clone()));
        router
            .bandit_store()
            .record_success(&(RequestType::Code, d0_key.clone()));

        let v = routing_state_json(&router, &deps);

        assert_eq!(v["strategy"], "simple_shuffle");
        assert_eq!(v["group_strategies"]["gpt-4o"], "adaptive_router");
        assert_eq!(v["pre_call_checks"], json!(["health_check"]));

        let d0 = &v["deployments"][0];
        assert_eq!(d0["key"], d0_key);
        assert_eq!(d0["cooling"], true);
        assert_eq!(d0["healthy"], false);
        assert_eq!(d0["health_reason"], "dns");
        let code = &d0["bandit_cells"]["code"];
        assert_eq!(code["alpha"], 3.0);
        assert_eq!(code["beta"], 1.0);
        assert_eq!(code["observations"], 2.0);

        let d1 = &v["deployments"][1];
        assert_eq!(d1["key"], d1_key);
        assert_eq!(d1["cooling"], false);
        assert_eq!(d1["healthy"], true);
        assert!(d1["health_reason"].is_null());
        assert!(
            d1.get("bandit_cells").is_none(),
            "deployments without cells omit bandit_cells"
        );
    }
}
