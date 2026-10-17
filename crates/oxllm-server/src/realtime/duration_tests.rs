//! Numeric regressions for the catalog's live-duration rate cards. All elapsed
//! work is injected with Instant offsets; peers are local and never infer.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use axum::Extension;
use futures_util::{SinkExt, StreamExt};
use oxllm_core::{Hook, ResponseSummary};
use serde_json::json;
use tokio::sync::Notify;
use tokio_tungstenite::tungstenite::Message;

use super::*;
use crate::auth::AuthUser;
use crate::cache::InMemoryCache;
use crate::config::types::{ConfigModel, OxllmSettings, ProxyConfig, RouterSettings};
use crate::hooks::spend_keys::{deployment_spend_key, provider_spend_key};
use crate::orchestration::budget::reserve_router_budgets;
use crate::pipeline::EndpointPipeline;
use crate::realtime::session::SessionEnd;
use crate::spend_writer::SpendEntry;
use crate::state::StateBuilder;

const BASE: f64 = 0.25;
const WORK: Duration = Duration::from_secs(12);
const CARDS: [(&str, &str, f64); 2] = [
    ("xai/grok-voice-latest", "xai", 1.333333e-3),
    ("gpt-live-1", "openai", 8.333333e-4),
];

#[path = "window_tests.rs"]
mod window_tests;

#[derive(Default)]
struct Probe {
    events: Mutex<Vec<(&'static str, Option<f64>)>>,
    done: Notify,
}

#[async_trait::async_trait]
impl Hook for Probe {
    async fn post_success(&self, ctx: &CallContext, response: &ResponseSummary) {
        self.events.lock().unwrap().push((
            "success",
            crate::hooks::cost_fallback::fallback_cost(ctx, response),
        ));
        self.done.notify_one();
    }

    async fn post_failure(&self, ctx: &CallContext, _: &Error) {
        self.events
            .lock()
            .unwrap()
            .push(("failure", ctx.cost_override));
        self.done.notify_one();
    }

    async fn release_concurrency(&self, ctx: &CallContext) {
        self.events
            .lock()
            .unwrap()
            .push(("release", ctx.cost_override));
        self.done.notify_one();
    }
}

struct Fixture {
    state: Arc<AppState>,
    probe: Arc<Probe>,
    ledger: tokio::sync::mpsc::Receiver<SpendEntry>,
    spend_keys: Vec<String>,
}

fn state(model: &str, provider: &str, base: Option<String>) -> Fixture {
    state_for_group(
        model,
        provider,
        base,
        "voice",
        Arc::new(InMemoryCache::default()),
    )
}

fn state_for_group(
    model: &str,
    provider: &str,
    base: Option<String>,
    group: &str,
    cache: Arc<dyn Cache>,
) -> Fixture {
    state_for_group_with_window(model, provider, base, group, cache, "1d")
}

fn state_for_group_with_window(
    model: &str,
    provider: &str,
    base: Option<String>,
    group: &str,
    cache: Arc<dyn Cache>,
    window: &str,
) -> Fixture {
    let (spend_tx, ledger) = tokio::sync::mpsc::channel(8);
    let probe = Arc::new(Probe::default());
    let mut state = StateBuilder::new(ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: group.into(),
            oxllm_params: oxllm_core::OxllmParams {
                model: model.into(),
                api_key: Some("local-test".into()),
                api_base: base,
                timeout: Some(60.0),
                max_budget: Some(100.0),
                budget_duration: Some(window.into()),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            max_budget: Some(100.0),
            budget_duration: Some(window.into()),
            request_timeout: Some(60),
            model_max_budget: Some(HashMap::from([(group.into(), 100.0)])),
            ..Default::default()
        },
        general_settings: crate::config::types::GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: RouterSettings {
            provider_budget_config: Some(HashMap::from([(
                provider.into(),
                oxllm_core::router::ProviderBudgetConfig {
                    budget_limit: 100.0,
                    time_period: window.into(),
                },
            )])),
            ..Default::default()
        },
        extra: Default::default(),
    })
    .with_cache(cache)
    .with_spend_tx(Some(spend_tx.clone()))
    .build()
    .unwrap();
    // A duplicate ledger observer must not produce a second terminal row.
    state
        .hooks
        .push(Arc::new(crate::hooks::CostTrackingHook::with_spend_writer(
            Some(spend_tx),
        )));
    state.hooks.push(probe.clone());
    let deployment = &state.deployments[0];
    let spend_keys = vec![
        "spend:abc".into(),
        "spend:team:team".into(),
        "spend:org:org".into(),
        "spend:project:project".into(),
        "spend:user:user".into(),
        "spend:customer:customer".into(),
        "spend:tag:app".into(),
        "spend:tag:prod".into(),
        format!("spend:model:{group}"),
        deployment_spend_key(&oxllm_core::router::deployment_key(deployment)),
        provider_spend_key(provider),
    ];
    Fixture {
        state: Arc::new(state),
        probe,
        ledger,
        spend_keys,
    }
}

async fn reserved(model: &str, provider: &str) -> (Fixture, UpgradeAbortGuard) {
    let fixture = state(model, provider, None);
    reserve(fixture).await
}

async fn reserve(fixture: Fixture) -> (Fixture, UpgradeAbortGuard) {
    reserve_with_duration(fixture, Duration::from_secs(600)).await
}

fn session_context(deployment: &Deployment, duration: Duration) -> CallContext {
    let group = &deployment.model_name;
    CallContext {
        request_body: Arc::new(json!({"model":group,"stream":true,"max_tokens":100})),
        pricing_model: Some(deployment.oxllm_params.model.clone()),
        deployment_model: Some(deployment.oxllm_params.model.clone()),
        upstream_duration_estimate: Some(duration),
        team_id: Some("team".into()),
        team_max_budget: Some(100.0),
        org_id: Some("org".into()),
        org_max_budget: Some(100.0),
        project_id: Some("project".into()),
        project_max_budget: Some(100.0),
        user_id: Some("user".into()),
        user_max_budget: Some(100.0),
        end_user_id: Some("customer".into()),
        customer_max_budget: Some(100.0),
        request_tag: Some("app".into()),
        request_tags: vec!["app".into(), "prod".into()],
        tag_budgets: vec![("app".into(), 100.0), ("prod".into(), 100.0)],
        max_parallel_requests: Some(4),
        rpm: Some(20),
        ..CallContext::for_test(group)
    }
}

async fn reserve_with_duration(
    fixture: Fixture,
    duration: Duration,
) -> (Fixture, UpgradeAbortGuard) {
    let deployment = fixture.state.deployments[0].clone();
    let group = &deployment.model_name;
    seed_baselines(&fixture).await;
    let mut pipeline = EndpointPipeline {
        state: &fixture.state,
        call_ctx: session_context(&deployment, duration),
    };
    assert!(pipeline.pre_call().await.unwrap().is_none());
    let dep_key = oxllm_core::router::deployment_key(&deployment);
    let reservation =
        reserve_router_budgets(&fixture.state, &deployment, &dep_key, &pipeline.call_ctx)
            .await
            .unwrap();
    let estimate = pipeline.call_ctx.budget_estimate.unwrap();
    for key in &fixture.spend_keys {
        let spend = fixture.state.cache.counter_value(key).await.unwrap();
        assert!(
            (spend - BASE - estimate).abs() < 1e-12,
            "reservation for {key}: {spend}"
        );
    }
    let routing = RealtimeRoutingGuard::new(fixture.state.clone(), &deployment, group);
    let mut guard = UpgradeAbortGuard::new(
        fixture.state.clone(),
        pipeline.call_ctx,
        deployment,
        routing,
    );
    guard.set_router_budget(RealtimeRouterBudgetGuard::new(
        fixture.state.cache.clone(),
        reservation,
    ));
    (fixture, guard)
}

async fn seed_baselines(fixture: &Fixture) {
    for key in &fixture.spend_keys {
        fixture.state.cache.incr(key, BASE, None).await.unwrap();
    }
    fixture
        .state
        .cache
        .incr("parallel:abc", 1.0, None)
        .await
        .unwrap();
    fixture
        .state
        .cache
        .incr("rpm:abc", 1.0, None)
        .await
        .unwrap();
}

fn connected(guard: &mut UpgradeAbortGuard) {
    let started = guard.start_connection();
    guard.connection_result(Ok(()), started);
    guard
        .settlement
        .as_mut()
        .unwrap()
        .call_ctx
        .upstream_start_time = Some(Instant::now() - WORK);
}

#[derive(Clone, Copy)]
enum RolloverExit {
    Disconnect,
    UpstreamFailure,
    SetupFailure,
    PositiveAdjustment,
}

async fn exercise_rollover<Fut: std::future::Future<Output = ()>>(
    cache: Arc<dyn Cache>,
    expired_scope: usize,
    exit: RolloverExit,
    b_in_flight: bool,
    expire: impl FnOnce(String) -> Fut,
) {
    let (model, provider, rate) = CARDS[0];
    let bound = crate::realtime::REALTIME_DURATION_ESTIMATE;
    let fixture = state_for_group_with_window(model, provider, None, "voice", cache.clone(), "1m");
    let (mut fixture, mut guard) = reserve_with_duration(fixture, bound).await;
    if matches!(exit, RolloverExit::SetupFailure) {
        guard.start_connection();
    } else {
        connected(&mut guard);
    }
    let ctx = &mut guard.settlement.as_mut().unwrap().call_ctx;
    ctx.upstream_start_time = Some(Instant::now() - Duration::from_secs(70));
    if matches!(exit, RolloverExit::PositiveAdjustment) {
        ctx.cost_override = Some(700.0 * rate); // exceeds A's original 619r reservation
    }
    assert_eq!(ctx.budget_windows.as_ref().unwrap().len(), 9);
    let key = fixture.spend_keys[expired_scope].clone();
    expire(key.clone()).await;
    assert!(
        cache.counter_value(&key).await.is_none(),
        "the original window really expired"
    );

    let deployment = &fixture.state.deployments[0];
    let mut b = session_context(deployment, bound);
    b.request_id = "rollover-b".into();
    b.cost_override = Some(rate);
    let mut pipeline = EndpointPipeline {
        state: &fixture.state,
        call_ctx: b,
    };
    assert!(pipeline.pre_call().await.unwrap().is_none());
    let dep_key = oxllm_core::router::deployment_key(deployment);
    let reservation =
        reserve_router_budgets(&fixture.state, deployment, &dep_key, &pipeline.call_ctx)
            .await
            .unwrap();
    if !b_in_flight {
        crate::orchestration::budget::settle_router_spend(
            &fixture.state,
            deployment,
            &dep_key,
            Some(rate),
            &reservation,
        )
        .await;
        pipeline.post_success(model, None, None, provider).await;
        assert!((cache.counter_value(&key).await.unwrap() - rate).abs() < 1e-12);
    }
    let outcome = if matches!(exit, RolloverExit::SetupFailure) {
        Err(Error::Network(
            "setup failed after reservation window ended".into(),
        ))
    } else {
        let mut meter = guard.meter();
        meter.delivered_binary();
        let end = if matches!(exit, RolloverExit::UpstreamFailure) {
            SessionEnd::UpstreamError(Error::Network("failure after work".into()))
        } else {
            SessionEnd::ClientDisconnected
        };
        Ok(meter.finish(end))
    };
    await_settlement(guard.settle(outcome)).await;
    if b_in_flight {
        assert!(
            (cache.counter_value(&key).await.unwrap() - bound.as_secs_f64() * rate).abs() < 1e-12,
            "A must not consume B's outstanding reservation"
        );
        crate::orchestration::budget::settle_router_spend(
            &fixture.state,
            deployment,
            &dep_key,
            Some(rate),
            &reservation,
        )
        .await;
        pipeline.post_success(model, None, None, provider).await;
    }
    let rows = [
        fixture.ledger.try_recv().unwrap(),
        fixture.ledger.try_recv().unwrap(),
    ];
    let a = rows
        .iter()
        .find(|row| row.request_id.as_deref() == Some("req-1"))
        .unwrap();
    let b = rows
        .iter()
        .find(|row| row.request_id.as_deref() == Some("rollover-b"))
        .unwrap();
    assert_eq!(b.cost, Some(rate));
    let cost = a.cost.unwrap();
    match exit {
        RolloverExit::SetupFailure => {
            assert_eq!(cost, 0.0);
            assert_eq!(a.status.as_deref(), Some("failure"));
        }
        RolloverExit::PositiveAdjustment => assert_eq!(cost, 700.0 * rate),
        _ => assert!((70.0 * rate..71.0 * rate).contains(&cost)),
    }
    for (index, key) in fixture.spend_keys.iter().enumerate() {
        let expected = if index == expired_scope {
            rate
        } else {
            BASE + rate + cost
        };
        let after = cache.counter_value(key).await.unwrap();
        assert!(
            (after - expected).abs() < 1e-12,
            "{key}: A's window settlement changed {expected} to {after}"
        );
    }
    assert_eq!(cache.counter_value("parallel:abc").await, Some(1.0));
    assert!(
        fixture.ledger.try_recv().is_err(),
        "A and B each have one ledger row"
    );
}

#[tokio::test]
async fn catalog_duration_rollover_never_refunds_another_requests_recreated_window() {
    for expired_scope in 0..11 {
        let cache = Arc::new(InMemoryCache::default());
        let clock = cache.clone();
        exercise_rollover(
            cache,
            expired_scope,
            RolloverExit::Disconnect,
            false,
            |key| async move {
                clock.expire_counter_for_test(&key).await;
            },
        )
        .await;
    }
}

#[tokio::test]
async fn catalog_duration_rollover_failure_release_and_positive_adjustments_are_window_owned() {
    for exit in [
        RolloverExit::UpstreamFailure,
        RolloverExit::SetupFailure,
        RolloverExit::PositiveAdjustment,
    ] {
        for expired_scope in 0..11 {
            let cache = Arc::new(InMemoryCache::default());
            let clock = cache.clone();
            exercise_rollover(cache, expired_scope, exit, true, |key| async move {
                clock.expire_counter_for_test(&key).await;
            })
            .await;
        }
    }
}

#[cfg(feature = "redis")]
#[tokio::test]
async fn catalog_duration_rollover_redis_all_scopes_use_atomic_window_receipts() {
    let Ok(url) = std::env::var("OXLLM_REDIS_URL") else {
        eprintln!("SKIP disposable local Redis test: OXLLM_REDIS_URL is unset");
        return;
    };
    assert!(
        matches!(
            url::Url::parse(&url).unwrap().host_str(),
            Some("127.0.0.1" | "localhost" | "[::1]")
        ),
        "budget rollover tests require an isolated loopback Redis fixture"
    );
    for exit in [
        RolloverExit::Disconnect,
        RolloverExit::UpstreamFailure,
        RolloverExit::SetupFailure,
        RolloverExit::PositiveAdjustment,
    ] {
        for scope in 0..11 {
            let prefix = format!(
                "budget-rollover:{}",
                oxllm_core::CounterWindow::fresh().as_str()
            );
            let cache = Arc::new(
                oxllm_cache_redis::RedisCache::new(&url, &prefix)
                    .await
                    .unwrap(),
            );
            let client = redis::Client::open(url.as_str()).unwrap();
            let mut connection = client.get_multiplexed_async_connection().await.unwrap();
            exercise_rollover(cache, scope, exit, true, |key| async move {
                // Real Redis expiry, not DEL+SET or a local duration heuristic.
                // The companion metadata deliberately remains until recreation.
                let expired: i64 = redis::cmd("PEXPIREAT")
                    .arg(format!("{prefix}:{key}"))
                    .arg(1)
                    .query_async(&mut connection)
                    .await
                    .unwrap();
                assert_eq!(expired, 1);
            })
            .await;
        }
    }
}

async fn balances(fixture: &Fixture, cost: f64) {
    for key in &fixture.spend_keys {
        let actual = fixture.state.cache.counter_value(key).await.unwrap();
        assert!(
            (actual - (BASE + cost)).abs() < 1e-12,
            "{key}: {actual} != {}",
            BASE + cost
        );
    }
    assert_eq!(
        fixture.state.cache.counter_value("parallel:abc").await,
        Some(1.0)
    );
    assert_eq!(
        fixture.state.cache.counter_value("rpm:abc").await,
        Some(2.0)
    );
    let deployment = &fixture.state.deployments[0];
    assert_eq!(
        fixture
            .state
            .router
            .usage_store()
            .in_flight(&fixture.state.router.cooldown_key(deployment)),
        0
    );
}

async fn measured_row(fixture: &mut Fixture, rate: f64, event: &'static str) -> f64 {
    let row = tokio::time::timeout(Duration::from_secs(3), fixture.ledger.recv())
        .await
        .unwrap()
        .expect("terminal duration row");
    let pending = fixture.probe.events.lock().unwrap().is_empty();
    if pending {
        tokio::time::timeout(Duration::from_secs(3), fixture.probe.done.notified())
            .await
            .unwrap();
    }
    let cost = row
        .cost
        .expect("elapsed cost is known independently of token usage");
    let expected = WORK.as_secs_f64() * rate;
    assert!(
        (expected..expected + rate).contains(&cost),
        "catalog duration cost {cost} must be near {expected}"
    );
    assert_eq!(
        row.status.as_deref(),
        Some(if event == "success" {
            "success"
        } else {
            "partial"
        })
    );
    assert_eq!(
        fixture.probe.events.lock().unwrap().as_slice(),
        &[(event, Some(cost))]
    );
    assert!(
        fixture.ledger.try_recv().is_err(),
        "one terminal ledger row even with duplicate hooks"
    );
    balances(fixture, cost).await;
    cost
}

#[tokio::test]
async fn catalog_duration_group_name_cannot_make_a_token_partial_charge_complete() {
    let fixture = state_for_group(
        "openai/gpt-4o",
        "openai",
        None,
        "gpt-live-1",
        Arc::new(InMemoryCache::default()),
    );
    let (mut fixture, mut guard) = reserve(fixture).await;
    connected(&mut guard);
    let estimate = guard
        .settlement
        .as_ref()
        .unwrap()
        .call_ctx
        .budget_estimate
        .unwrap();
    let mut meter = guard.meter();
    meter.observe(&crate::realtime::OpenAIRealtimeProvider,
        r#"{"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}"#);
    meter.delivered_binary();
    await_settlement(guard.settle(Ok(meter.finish(SessionEnd::ClientDisconnected)))).await;
    let row = fixture.ledger.try_recv().unwrap();
    let reported_token_cost = 2.0 * 2.5e-6 + 3.0 * 10e-6;
    assert!((row.cost.unwrap() - reported_token_cost).abs() < 1e-12);
    for key in &fixture.spend_keys[..9] {
        let spend = fixture.state.cache.counter_value(key).await.unwrap();
        assert!(
            (spend - BASE - estimate).abs() < 1e-12,
            "token partial estimate must stand for {key}: {spend}"
        );
    }
}

#[tokio::test]
async fn catalog_duration_serving_card_cannot_fall_back_to_a_token_priced_group_name() {
    let fixture = state_for_group(
        CARDS[0].0,
        CARDS[0].1,
        None,
        "gpt-4o",
        Arc::new(InMemoryCache::default()),
    );
    let (mut fixture, mut guard) = reserve(fixture).await;
    connected(&mut guard);
    let mut meter = guard.meter();
    meter.observe(&crate::realtime::OpenAIRealtimeProvider,
        r#"{"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}"#);
    await_settlement(guard.settle(Ok(meter.finish(SessionEnd::UpstreamClosed)))).await;
    measured_row(&mut fixture, CARDS[0].2, "success").await;
}

#[tokio::test]
async fn catalog_duration_normal_close_settles_every_scope_to_the_same_cost() {
    for (model, provider, rate) in CARDS {
        let (mut fixture, mut guard) = reserved(model, provider).await;
        connected(&mut guard);
        let mut meter = guard.meter();
        meter.observe(&crate::realtime::OpenAIRealtimeProvider,
            r#"{"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}"#);
        await_settlement(guard.settle(Ok(meter.finish(SessionEnd::UpstreamClosed)))).await;
        measured_row(&mut fixture, rate, "success").await;
    }
}

#[tokio::test]
async fn catalog_duration_unmetered_client_close_settles_the_measured_not_reserved_cost() {
    for (model, provider, rate) in CARDS {
        let (mut fixture, mut guard) = reserved(model, provider).await;
        connected(&mut guard);
        let mut meter = guard.meter();
        meter.delivered_binary();
        await_settlement(guard.settle(Ok(meter.finish(SessionEnd::ClientDisconnected)))).await;
        measured_row(&mut fixture, rate, "release").await;
    }
}

#[tokio::test]
async fn catalog_duration_upstream_error_after_work_keeps_the_measured_charge_and_failure_callback()
{
    for (model, provider, rate) in CARDS {
        let (mut fixture, mut guard) = reserved(model, provider).await;
        connected(&mut guard);
        let mut meter = guard.meter();
        meter.delivered_binary();
        let end = SessionEnd::UpstreamError(Error::Network("after connected work".into()));
        await_settlement(guard.settle(Ok(meter.finish(end)))).await;
        measured_row(&mut fixture, rate, "failure").await;
    }
}

#[tokio::test]
async fn catalog_duration_infrastructure_abort_keeps_the_measured_charge_and_failure_callback() {
    for (model, provider, rate) in CARDS {
        let (mut fixture, mut guard) = reserved(model, provider).await;
        connected(&mut guard);
        guard.meter().delivered_binary();
        drop(guard);
        tokio::time::timeout(Duration::from_secs(3), fixture.probe.done.notified())
            .await
            .unwrap();
        measured_row(&mut fixture, rate, "failure").await;
    }
}

#[tokio::test]
async fn catalog_duration_preexisting_gpt_live_partial_also_settles_elapsed_cost() {
    let (model, provider, rate) = CARDS[1];
    let (mut fixture, mut guard) = reserved(model, provider).await;
    connected(&mut guard);
    let mut meter = guard.meter();
    meter.delivered_binary();
    await_settlement(guard.settle(Ok(meter.finish(SessionEnd::ClientDisconnected)))).await;
    measured_row(&mut fixture, rate, "release").await;
}

#[tokio::test]
async fn catalog_duration_preconnect_failures_never_bill_elapsed_setup_time() {
    for (model, provider, _) in CARDS {
        for error in [
            Error::Config("setup".into()),
            Error::Network("connect".into()),
        ] {
            for connected_first in [false, true] {
                let (mut fixture, mut guard) = reserved(model, provider).await;
                let started = guard.start_connection();
                if connected_first {
                    guard.connection_result(Ok(()), started);
                }
                guard
                    .settlement
                    .as_mut()
                    .unwrap()
                    .call_ctx
                    .upstream_start_time = Some(Instant::now() - Duration::from_secs(3600));
                await_settlement(guard.settle(Err(error.clone()))).await;
                let row = fixture.ledger.try_recv().unwrap();
                assert_eq!(row.status.as_deref(), Some("failure"));
                assert_eq!(row.cost, Some(0.0));
                for key in &fixture.spend_keys {
                    assert!(
                        (fixture.state.cache.counter_value(key).await.unwrap() - BASE).abs()
                            < 1e-12,
                        "{key}"
                    );
                }
                assert_eq!(
                    fixture.state.cache.counter_value("parallel:abc").await,
                    Some(1.0)
                );
                assert_eq!(
                    fixture.state.cache.counter_value("rpm:abc").await,
                    Some(if connected_first { 2.0 } else { 1.0 })
                );
                assert_eq!(
                    fixture.probe.events.lock().unwrap().as_slice(),
                    &[("failure", Some(0.0))]
                );
                assert!(fixture.ledger.try_recv().is_err());
            }
        }
    }
}

#[tokio::test]
async fn catalog_duration_explicit_zero_is_authoritative_on_partial_close() {
    for (model, provider, _) in CARDS {
        let (mut fixture, mut guard) = reserved(model, provider).await;
        connected(&mut guard);
        guard.settlement.as_mut().unwrap().call_ctx.cost_override = Some(0.0);
        let mut meter = guard.meter();
        meter.delivered_binary();
        await_settlement(guard.settle(Ok(meter.finish(SessionEnd::ClientDisconnected)))).await;
        let row = fixture.ledger.try_recv().unwrap();
        assert_eq!(row.cost, Some(0.0));
        assert_eq!(row.status.as_deref(), Some("partial"));
        balances(&fixture, 0.0).await;
        assert_eq!(
            fixture.probe.events.lock().unwrap().as_slice(),
            &[("release", Some(0.0))]
        );
    }
}

#[tokio::test]
async fn catalog_duration_override_failure_keeps_even_a_positive_settlement_delta() {
    let (mut fixture, mut guard) = reserved(CARDS[0].0, CARDS[0].1).await;
    connected(&mut guard);
    guard
        .settlement
        .as_mut()
        .unwrap()
        .call_ctx
        .deployment_pricing = Some(oxllm_core::DeploymentPricing {
        cost_per_second_rate: Some(2.0),
        input_cost_per_second: Some(3.0),
        output_cost_per_second: Some(4.0),
        ..Default::default()
    });
    let mut meter = guard.meter();
    meter.delivered_binary();
    await_settlement(
        guard.settle(Ok(meter.finish(SessionEnd::UpstreamError(Error::Network(
            "fault after work".into(),
        ))))),
    )
    .await;
    // The real charge exceeds the original catalog reservation here: failure
    // cleanup must not lose that positive debit, or refund the measured cost.
    measured_row(&mut fixture, 2.0, "failure").await;
}

#[tokio::test]
async fn catalog_duration_missing_upstream_timing_keeps_cost_unknown_not_admission_elapsed() {
    for (model, provider, rate) in CARDS {
        let (mut fixture, mut guard) = reserved(model, provider).await;
        connected(&mut guard);
        let ctx = &mut guard.settlement.as_mut().unwrap().call_ctx;
        ctx.upstream_start_time = None;
        ctx.start_time = Some(Instant::now() - Duration::from_secs(3600));
        let mut meter = guard.meter();
        meter.delivered_binary();
        await_settlement(guard.settle(Ok(meter.finish(SessionEnd::ClientDisconnected)))).await;
        let row = fixture.ledger.try_recv().unwrap();
        assert!(row.cost.is_none());
        assert!(row.total_tokens.is_none());
        assert_eq!(row.status.as_deref(), Some("partial"));
        balances(&fixture, 600.0 * rate).await;
        assert_eq!(
            fixture.probe.events.lock().unwrap().as_slice(),
            &[("release", None)]
        );
    }
}

async fn unknown_timing_with_reported_tokens(group: &str, partial: bool, failed: bool) {
    let fixture = state_for_group(
        CARDS[0].0,
        CARDS[0].1,
        None,
        group,
        Arc::new(InMemoryCache::default()),
    );
    let (mut fixture, mut guard) = reserve(fixture).await;
    connected(&mut guard);
    let ctx = &mut guard.settlement.as_mut().unwrap().call_ctx;
    ctx.upstream_start_time = None;
    ctx.start_time = Some(Instant::now() - Duration::from_secs(3600));
    let mut meter = guard.meter();
    meter.observe(&crate::realtime::OpenAIRealtimeProvider,
        r#"{"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}"#);
    if partial {
        meter.delivered_binary();
    }
    let end = if failed {
        SessionEnd::UpstreamError(Error::Network("missing timing after work".into()))
    } else if partial {
        SessionEnd::ClientDisconnected
    } else {
        SessionEnd::UpstreamClosed
    };
    await_settlement(guard.settle(Ok(meter.finish(end)))).await;
    let row = fixture.ledger.try_recv().unwrap();
    assert!(
        row.cost.is_none(),
        "tokens are not duration evidence: {:?}",
        row.cost
    );
    assert_eq!(row.total_tokens, Some(5));
    balances(
        &fixture,
        if partial && !failed {
            600.0 * CARDS[0].2
        } else {
            0.0
        },
    )
    .await;
    assert!(fixture.ledger.try_recv().is_err());
}

#[tokio::test]
async fn catalog_duration_unknown_timing_never_falls_back_to_a_token_group_price() {
    unknown_timing_with_reported_tokens("gpt-4o", false, false).await;
}

#[tokio::test]
async fn catalog_duration_unknown_timing_never_adds_router_nominal_cost_from_token_reports() {
    unknown_timing_with_reported_tokens("voice", false, false).await;
}

#[tokio::test]
async fn catalog_duration_unknown_timing_partial_keeps_estimates_even_with_reported_tokens() {
    unknown_timing_with_reported_tokens("gpt-4o", true, false).await;
}

#[tokio::test]
async fn catalog_duration_unknown_timing_failure_refunds_every_scope_without_inventing_cost() {
    for partial in [false, true] {
        unknown_timing_with_reported_tokens("voice", partial, true).await;
    }
}

#[tokio::test]
async fn catalog_duration_connected_control_only_closes_and_faults_still_have_elapsed_cost() {
    for (model, provider, rate) in CARDS {
        for failed in [false, true] {
            let (mut fixture, mut guard) = reserved(model, provider).await;
            connected(&mut guard);
            let end = if failed {
                SessionEnd::UpstreamError(Error::Network("connected session failed".into()))
            } else {
                SessionEnd::UpstreamClosed
            };
            let outcome = guard.meter().finish(end);
            assert!(outcome.usage.is_none() && !outcome.unmetered_output);
            await_settlement(guard.settle(Ok(outcome))).await;
            measured_row(
                &mut fixture,
                rate,
                if failed { "failure" } else { "success" },
            )
            .await;
        }
    }
}

#[tokio::test]
async fn catalog_duration_zero_rate_override_needs_no_invented_timing_or_tokens() {
    for failed in [false, true] {
        let (mut fixture, mut guard) = reserved(CARDS[0].0, CARDS[0].1).await;
        connected(&mut guard);
        let ctx = &mut guard.settlement.as_mut().unwrap().call_ctx;
        ctx.upstream_start_time = None;
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cost_per_second_rate: Some(0.0),
            input_cost_per_second: Some(3.0),
            ..Default::default()
        });
        let mut meter = guard.meter();
        meter.delivered_binary();
        let end = if failed {
            SessionEnd::TaskAborted
        } else {
            SessionEnd::ClientDisconnected
        };
        await_settlement(guard.settle(Ok(meter.finish(end)))).await;
        let row = fixture.ledger.try_recv().unwrap();
        assert_eq!(row.cost, Some(0.0));
        assert_eq!(row.status.as_deref(), Some("partial"));
        assert!(row.total_tokens.is_none());
        balances(&fixture, 0.0).await;
        assert!(fixture.ledger.try_recv().is_err());
    }
}

#[tokio::test]
async fn catalog_duration_partial_token_usage_does_not_limit_elapsed_settlement() {
    for (model, provider, rate) in CARDS {
        for failed in [false, true] {
            let (mut fixture, mut guard) = reserved(model, provider).await;
            connected(&mut guard);
            let mut meter = guard.meter();
            meter.observe(&crate::realtime::OpenAIRealtimeProvider,
                r#"{"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}"#);
            meter.delivered_text(
                r#"{"type":"response.text.delta","response_id":"pending","delta":"hello"}"#,
            );
            let end = if failed {
                SessionEnd::TaskAborted
            } else {
                SessionEnd::ClientDisconnected
            };
            await_settlement(guard.settle(Ok(meter.finish(end)))).await;
            measured_row(
                &mut fixture,
                rate,
                if failed { "failure" } else { "release" },
            )
            .await;
        }
    }
}

#[tokio::test]
async fn catalog_duration_live_estimate_is_resolved_before_existing_key_top_up_and_router_reserve()
{
    for (model, provider, rate) in CARDS {
        let fixture = state(model, provider, None);
        let deployment = &fixture.state.deployments[0];
        let mut pipeline = EndpointPipeline {
            state: &fixture.state,
            call_ctx: CallContext {
                pricing_model: Some(model.into()),
                upstream_duration_estimate: Some(Duration::from_secs(60)),
                team_id: Some("team".into()),
                team_max_budget: Some(100.0),
                ..CallContext::for_test("voice")
            },
        };
        assert!(pipeline.pre_call().await.unwrap().is_none());
        let prior = pipeline.call_ctx.budget_estimate.unwrap();
        assert!((prior - 60.0 * rate).abs() < 1e-12);
        let bound = crate::realtime::REALTIME_DURATION_ESTIMATE;
        crate::pipeline::prepare_deployment_pricing_with_duration_estimate(
            &fixture.state,
            &mut pipeline.call_ctx,
            deployment,
            &json!({"model":"voice"}),
            bound,
        )
        .await
        .unwrap();
        assert_eq!(pipeline.call_ctx.upstream_duration_estimate, Some(bound));
        let estimate = pipeline.call_ctx.budget_estimate.unwrap();
        assert!((estimate - bound.as_secs_f64() * rate).abs() < 1e-12);
        assert!(bound >= crate::realtime::REALTIME_MAX_SESSION_DURATION);
        let dep_key = oxllm_core::router::deployment_key(deployment);
        let reservation =
            reserve_router_budgets(&fixture.state, deployment, &dep_key, &pipeline.call_ctx)
                .await
                .unwrap();
        for key in [
            "spend:abc",
            "spend:team:team",
            "spend:model:voice",
            &fixture.spend_keys[9],
            &fixture.spend_keys[10],
        ] {
            let spend = fixture.state.cache.counter_value(key).await.unwrap();
            assert!(
                (spend - estimate).abs() < 1e-12,
                "{key}: {spend} != {estimate}"
            );
        }
        reservation.release(&fixture.state.cache).await;
        pipeline
            .post_failure(&Error::Network("before connection".into()))
            .await;
    }
}

#[tokio::test]
async fn catalog_duration_partial_settlement_survives_owner_abort_at_each_monetary_boundary() {
    for failed in [false, true] {
        for scope in ["deployment", "provider", "key", "team", "parallel"] {
            let cache = Arc::new(super::tests::DelayedCache::default());
            let fixture = state_for_group(CARDS[0].0, CARDS[0].1, None, "voice", cache.clone());
            let (mut fixture, mut guard) = reserve(fixture).await;
            let pause_key = match scope {
                "deployment" => fixture.spend_keys[9].clone(),
                "provider" => fixture.spend_keys[10].clone(),
                "key" => "spend:abc".into(),
                "team" => "spend:team:team".into(),
                _ => "parallel:abc".into(),
            };
            *cache.pause_key.lock().unwrap() = Some(pause_key);
            connected(&mut guard);
            let mut meter = guard.meter();
            meter.delivered_binary();
            let end = if failed {
                SessionEnd::TaskAborted
            } else {
                SessionEnd::ClientDisconnected
            };
            let outcome = meter.finish(end);
            let owner = tokio::spawn(async move {
                await_settlement(guard.settle(Ok(outcome))).await;
            });
            tokio::time::timeout(Duration::from_secs(3), cache.entered.notified())
                .await
                .unwrap();
            // The guard already transferred ownership to detached settlement;
            // no monetary refund may run again when its waiting owner aborts.
            owner.abort();
            assert!(owner.await.unwrap_err().is_cancelled());
            cache.resume.notify_one();
            measured_row(
                &mut fixture,
                CARDS[0].2,
                if failed { "failure" } else { "release" },
            )
            .await;
            for key in fixture
                .spend_keys
                .iter()
                .map(String::as_str)
                .chain(["parallel:abc"])
            {
                assert_eq!(
                    cache.decrements.lock().unwrap().get(key),
                    Some(&1),
                    "{scope}: {key}"
                );
            }
        }
    }
}

#[tokio::test]
async fn catalog_duration_upgrade_reserves_the_live_cap_despite_a_sixty_second_http_timeout() {
    for (model, provider, rate) in CARDS {
        let upstream = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let fixture = state(
            model,
            provider,
            Some(format!("http://{}", upstream.local_addr().unwrap())),
        );
        let peer = tokio::spawn(async move {
            let (stream, _) = upstream.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            socket
                .send(Message::Text(r#"{"type":"session.created"}"#.into()))
                .await
                .unwrap();
            while let Some(Ok(message)) = socket.next().await {
                if matches!(message, Message::Close(_))
                    || message
                        .to_text()
                        .is_ok_and(|s| s.contains("response.cancel"))
                {
                    let _ = socket.close(None).await;
                    break;
                }
            }
        });
        let auth = AuthUser {
            key_hash_hex: "abc".into(),
            team_id: Some("team".into()),
            team_max_budget: Some(100.0),
            ..Default::default()
        };
        let shared = Arc::new(arc_swap::ArcSwap::from(fixture.state.clone()));
        let app = axum::Router::new()
            .route(
                "/v1/realtime",
                axum::routing::get(crate::realtime::handle_realtime_upgrade),
            )
            .layer(Extension(auth))
            .with_state(shared);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "ws://{}/v1/realtime?model=voice",
            listener.local_addr().unwrap()
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut client, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        let event = tokio::time::timeout(Duration::from_secs(3), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(event.to_text().unwrap().contains("session.created"));
        for key in [
            "spend:abc",
            "spend:team:team",
            "spend:model:voice",
            &fixture.spend_keys[9],
            &fixture.spend_keys[10],
        ] {
            let reserved = fixture.state.cache.counter_value(key).await.unwrap();
            assert!(
                reserved >= 600.0 * rate,
                "{model} {key} reserved {reserved}, must cover at least {}",
                600.0 * rate
            );
        }
        client.close(None).await.unwrap();
        tokio::time::timeout(Duration::from_secs(3), peer)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), fixture.probe.done.notified())
            .await
            .unwrap();
        server.abort();
    }
}
