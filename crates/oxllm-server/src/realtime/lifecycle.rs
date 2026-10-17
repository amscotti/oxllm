//! Single-owner realtime settlement. Task cancellation drops the live owner;
//! normal teardown transfers it to a detached task before the first await.

use std::sync::Arc;
use std::time::Instant;

use oxllm_core::{Cache, CallContext, Deployment, Error};
use tokio::runtime::Handle;
use tokio::task::JoinHandle;

use super::connect::ConnectError;
use super::routing::RealtimeRoutingGuard;
use super::session::{SessionMeter, SessionOutcome};
use crate::orchestration::budget::RouterBudgetReservation;
use crate::state::AppState;

/// Owns an acquired router reservation until it is transferred to settlement.
/// Taking the reservation, rather than marking it retained after a debit,
/// prevents cancellation from refunding a partially settled counter twice.
pub(super) struct RealtimeRouterBudgetGuard {
    cache: Arc<dyn Cache>,
    reservation: Option<RouterBudgetReservation>,
    runtime: Option<Handle>,
}

impl RealtimeRouterBudgetGuard {
    pub(super) fn new(cache: Arc<dyn Cache>, reservation: RouterBudgetReservation) -> Self {
        Self {
            cache,
            reservation: Some(reservation),
            runtime: Handle::try_current().ok(),
        }
    }

    pub(super) fn into_reservation(mut self) -> RouterBudgetReservation {
        self.reservation.take().unwrap_or_default()
    }
}

impl Drop for RealtimeRouterBudgetGuard {
    fn drop(&mut self) {
        let Some(reservation) = self.reservation.take() else {
            return;
        };
        if let Some(runtime) = &self.runtime {
            let cache = self.cache.clone();
            // Handle::spawn also works when Drop runs outside a runtime's
            // entered context. A closing runtime cannot run async cleanup,
            // but must never turn Drop into a panic.
            let _task = runtime.spawn(async move {
                reservation.release(&cache).await;
            });
        } else {
            tracing::warn!("no runtime available for realtime router budget cleanup");
        }
    }
}

struct Settlement {
    state: Arc<AppState>,
    call_ctx: CallContext,
    deployment: Deployment,
    model: String,
    provider: String,
    dep_key: String,
    meter: SessionMeter,
    routing: Option<RealtimeRoutingGuard>,
    router_budget: Option<RealtimeRouterBudgetGuard>,
}

/// Cleanup remains owned before upgrade, throughout connect/pump, and until
/// terminal settlement has been moved into an independently driven task.
pub(super) struct UpgradeAbortGuard {
    settlement: Option<Settlement>,
    runtime: Option<Handle>,
}

impl UpgradeAbortGuard {
    pub(super) fn new(
        state: Arc<AppState>,
        call_ctx: CallContext,
        deployment: Deployment,
        routing: RealtimeRoutingGuard,
    ) -> Self {
        let provider = deployment.provider_slug().to_string();
        // Pricing labels can carry throughput/context tiers. Only the adapter
        // strips those from the wire selector; billing retains the rate card.
        let model = super::openai::bare_model(&deployment).to_string();
        Self {
            runtime: Handle::try_current().ok(),
            settlement: Some(Settlement {
                state,
                call_ctx,
                deployment,
                model,
                provider,
                dep_key: routing.dep_key.clone(),
                meter: SessionMeter::default(),
                routing: Some(routing),
                router_budget: None,
            }),
        }
    }

    pub(super) fn set_router_budget(&mut self, guard: RealtimeRouterBudgetGuard) {
        if let Some(settlement) = &mut self.settlement {
            settlement.router_budget = Some(guard);
        }
    }

    pub(super) fn meter(&self) -> SessionMeter {
        self.settlement
            .as_ref()
            .map(|settlement| settlement.meter.clone())
            .unwrap_or_default()
    }

    pub(super) fn start_connection(&mut self) -> Instant {
        let started = Instant::now();
        if let Some(settlement) = &mut self.settlement {
            settlement.call_ctx.upstream_start_time = Some(started);
        }
        started
    }

    /// Called synchronously at the actual handshake result, including from the
    /// Bedrock adapter. No success is inferred from merely entering on_upgrade.
    pub(super) fn connection_result(
        &mut self,
        result: Result<(), &ConnectError>,
        started: Instant,
    ) {
        let Some(settlement) = &mut self.settlement else {
            return;
        };
        let Some(routing) = &mut settlement.routing else {
            return;
        };
        match result {
            Ok(()) => {
                settlement.call_ctx.reached_upstream = true;
                routing.connected(started.elapsed());
            }
            Err(error) => {
                settlement.call_ctx.reached_upstream =
                    matches!(&error.error, Error::Upstream { .. });
                routing.connect_failed(error);
            }
        }
    }

    /// Transfer both budget owners before returning a cancellable join future.
    /// Aborting the awaiting upgrade task drops only the handle, not settlement.
    pub(super) fn settle(
        mut self,
        outcome: oxllm_core::Result<SessionOutcome>,
    ) -> Option<JoinHandle<()>> {
        self.dispatch(outcome, true)
    }

    fn dispatch(
        &mut self,
        outcome: oxllm_core::Result<SessionOutcome>,
        observe_terminal: bool,
    ) -> Option<JoinHandle<()>> {
        let mut settlement = self.settlement.take()?;
        if let Some(mut routing) = settlement.routing.take() {
            if observe_terminal {
                routing.finish(&outcome);
            }
            // The upstream work has ended. Free the gauge synchronously, even
            // if the cache or a hook stalls the later monetary settlement.
            drop(routing);
        }
        let Some(runtime) = &self.runtime else {
            tracing::warn!("no runtime available for realtime key budget cleanup");
            return None;
        };
        Some(runtime.spawn(async move {
            let router_budget = settlement.router_budget.take().unwrap_or_else(|| {
                RealtimeRouterBudgetGuard::new(
                    settlement.state.cache.clone(),
                    RouterBudgetReservation::default(),
                )
            });
            super::settle_reserved_realtime_session(
                &settlement.state,
                &settlement.call_ctx,
                &settlement.deployment,
                &settlement.model,
                &settlement.provider,
                &settlement.dep_key,
                outcome,
                router_budget,
            )
            .await;
        }))
    }
}

impl Drop for UpgradeAbortGuard {
    fn drop(&mut self) {
        let Some(settlement) = &self.settlement else {
            return;
        };
        let connected = settlement
            .routing
            .as_ref()
            .is_some_and(RealtimeRoutingGuard::is_connected);
        let outcome = if connected {
            Ok(settlement.meter.aborted_outcome())
        } else {
            Err(Error::Network(
                "realtime task aborted before upstream handshake completion".into(),
            ))
        };
        // Infrastructure cancellation is not evidence against the deployment.
        // Any real handshake failure was already recorded by connection_result.
        let _task = self.dispatch(outcome, connected);
    }
}

pub(super) async fn await_settlement(task: Option<JoinHandle<()>>) {
    if let Some(task) = task {
        if let Err(error) = task.await {
            tracing::warn!(%error, "realtime settlement task failed");
        }
    }
}

#[cfg(test)]
#[path = "duration_tests.rs"]
mod duration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use futures_util::{SinkExt, StreamExt};
    use oxllm_core::{Hook, IncrResult, ResponseSummary};
    use serde_json::json;
    use tokio::sync::Notify;
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    use crate::cache::InMemoryCache;
    use crate::config::types::{
        ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings, RoutingKnobs,
    };
    use crate::hooks::{BudgetHook, RateLimitHook};
    use crate::orchestration::budget::reserve_router_budgets;
    use crate::realtime::session::SessionEnd;
    use crate::state::StateBuilder;

    const BASE_SPEND: f64 = 0.25;
    const ACTUAL_COST: f64 = 2.0 * 0.01 + 3.0 * 0.02;

    /// Apply a real counter debit, then withhold its completion/ack. Aborting
    /// settlement at this boundary previously refunded an already-debited
    /// router reservation or abandoned the remaining key/parallel cleanup.
    #[derive(Default)]
    pub(super) struct DelayedCache {
        inner: InMemoryCache,
        pub(super) pause_key: Mutex<Option<String>>,
        pub(super) entered: Notify,
        pub(super) resume: Notify,
        pub(super) decrements: Mutex<HashMap<String, usize>>,
    }

    impl DelayedCache {
        async fn after_decrement(&self, key: &str) {
            *self
                .decrements
                .lock()
                .unwrap()
                .entry(key.into())
                .or_default() += 1;
            let pause = {
                let mut pause_key = self.pause_key.lock().unwrap();
                if pause_key.as_deref() == Some(key) {
                    pause_key.take();
                    true
                } else {
                    false
                }
            };
            if pause {
                self.entered.notify_one();
                self.resume.notified().await;
            }
        }
    }

    #[async_trait::async_trait]
    impl Cache for DelayedCache {
        async fn get(&self, key: &str) -> Option<String> {
            self.inner.get(key).await
        }
        async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
            self.inner.set(key, value, ttl).await;
        }
        async fn delete(&self, key: &str) {
            self.inner.delete(key).await;
        }
        async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> oxllm_core::Result<f64> {
            self.inner.incr(key, by, ttl).await
        }
        async fn incr_if_below(
            &self,
            key: &str,
            delta: f64,
            ceiling: f64,
            ttl: Option<Duration>,
        ) -> oxllm_core::Result<IncrResult> {
            self.inner.incr_if_below(key, delta, ceiling, ttl).await
        }
        async fn counter_value(&self, key: &str) -> Option<f64> {
            self.inner.counter_value(key).await
        }
        async fn decr_if_exists(&self, key: &str, by: f64) -> oxllm_core::Result<bool> {
            let applied = self.inner.decr_if_exists(key, by).await?;
            self.after_decrement(key).await;
            Ok(applied)
        }

        async fn incr_if_below_in_window(
            &self,
            key: &str,
            delta: f64,
            ceiling: f64,
            ttl: Option<Duration>,
            expected: Option<&oxllm_core::CounterWindow>,
        ) -> oxllm_core::Result<oxllm_core::WindowedIncrResult> {
            self.inner
                .incr_if_below_in_window(key, delta, ceiling, ttl, expected)
                .await
        }

        async fn incr_in_window(
            &self,
            key: &str,
            delta: f64,
            window: &oxllm_core::CounterWindow,
        ) -> oxllm_core::Result<bool> {
            let applied = self.inner.incr_in_window(key, delta, window).await?;
            if applied && delta < 0.0 {
                self.after_decrement(key).await;
            }
            Ok(applied)
        }
    }

    #[derive(Default)]
    struct CompletionProbe {
        successes: AtomicUsize,
        failures: AtomicUsize,
        concurrency_releases: AtomicUsize,
        done: Notify,
    }

    #[async_trait::async_trait]
    impl Hook for CompletionProbe {
        async fn post_success(&self, _: &CallContext, _: &ResponseSummary) {
            self.successes.fetch_add(1, Ordering::SeqCst);
            self.done.notify_one();
        }
        async fn post_failure(&self, _: &CallContext, _: &Error) {
            self.failures.fetch_add(1, Ordering::SeqCst);
            self.done.notify_one();
        }
        async fn release_concurrency(&self, _: &CallContext) {
            self.concurrency_releases.fetch_add(1, Ordering::SeqCst);
            self.done.notify_one();
        }
    }

    struct Fixture {
        state: Arc<AppState>,
        cache: Arc<DelayedCache>,
        probe: Arc<CompletionProbe>,
        store_key: String,
        dep_spend: String,
        estimate: f64,
        spend_rx: tokio::sync::mpsc::Receiver<crate::spend_writer::SpendEntry>,
    }

    async fn reserved_session() -> (Fixture, UpgradeAbortGuard) {
        let cache = Arc::new(DelayedCache::default());
        let probe = Arc::new(CompletionProbe::default());
        let (spend_tx, spend_rx) = tokio::sync::mpsc::channel(4);
        let mut state = StateBuilder::new(ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "rt-alias".into(),
                oxllm_params: oxllm_core::OxllmParams {
                    model: "openai/rt-upstream".into(),
                    api_key: Some("test-key".into()),
                    max_budget: Some(100.0),
                    input_cost_per_token: Some(0.01),
                    output_cost_per_token: Some(0.02),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings {
                max_budget: Some(100.0),
                ..Default::default()
            },
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                routing: RoutingKnobs {
                    allowed_fails: Some(0),
                    ..Default::default()
                },
                enable_health_check_routing: Some(true),
                health_check_failure_threshold: Some(1),
                provider_budget_config: Some(HashMap::from([(
                    "openai".into(),
                    oxllm_core::router::ProviderBudgetConfig {
                        budget_limit: 100.0,
                        time_period: "1d".into(),
                    },
                )])),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        })
        .with_cache(cache.clone())
        .with_spend_tx(Some(spend_tx))
        .build()
        .unwrap();
        // Use real admission/settlement hooks with a last-hook completion
        // barrier; no metrics or exporter can hide a double monetary release.
        state.hooks = vec![
            Arc::new(BudgetHook::new(cache.clone(), Some(100.0), "1d")),
            Arc::new(RateLimitHook::new(cache.clone(), None, None)),
            probe.clone(),
        ];
        let state = Arc::new(state);
        let deployment = state.deployments[0].clone();
        let dep_key = oxllm_core::router::deployment_key(&deployment);
        let dep_spend = format!("spend:dep:{dep_key}");
        for key in ["spend:abc", dep_spend.as_str(), "spend:provider:openai"] {
            cache.incr(key, BASE_SPEND, None).await.unwrap();
        }
        // A different live request owns these baselines. Keeping them detects
        // duplicate decrements that a zero-clamped gauge would conceal.
        cache.incr("parallel:abc", 1.0, None).await.unwrap();
        cache.incr("rpm:abc", 1.0, None).await.unwrap();
        let mut pipeline = crate::pipeline::EndpointPipeline {
            state: &state,
            call_ctx: CallContext {
                request_body: Arc::new(json!({"model":"rt-alias","max_tokens":100})),
                deployment_pricing: deployment.oxllm_params.pricing_override(),
                pricing_model: Some(deployment.oxllm_params.model.clone()),
                max_parallel_requests: Some(4),
                rpm: Some(20),
                ..CallContext::for_test("rt-alias")
            },
        };
        assert!(pipeline.pre_call().await.unwrap().is_none());
        let estimate = pipeline.call_ctx.budget_estimate.unwrap();
        assert!(estimate > ACTUAL_COST);
        let reservation = reserve_router_budgets(&state, &deployment, &dep_key, &pipeline.call_ctx)
            .await
            .unwrap();
        let store_key = state.router.cooldown_key(&deployment);
        state.router.record_in_flight(&store_key, 1);
        let routing = RealtimeRoutingGuard::new(state.clone(), &deployment, "rt-alias");
        let mut guard =
            UpgradeAbortGuard::new(state.clone(), pipeline.call_ctx, deployment, routing);
        guard.set_router_budget(RealtimeRouterBudgetGuard::new(cache.clone(), reservation));
        (
            Fixture {
                state,
                cache,
                probe,
                store_key,
                dep_spend,
                estimate,
                spend_rx,
            },
            guard,
        )
    }

    async fn wait_done(fixture: &Fixture) {
        tokio::time::timeout(Duration::from_secs(3), fixture.probe.done.notified())
            .await
            .expect("detached settlement must complete");
    }

    async fn assert_balances(fixture: &Fixture, key_cost: f64, router_cost: f64, rpm: f64) {
        for (key, expected) in [
            ("spend:abc", BASE_SPEND + key_cost),
            (fixture.dep_spend.as_str(), BASE_SPEND + router_cost),
            ("spend:provider:openai", BASE_SPEND + router_cost),
            ("parallel:abc", 1.0),
            ("rpm:abc", rpm),
        ] {
            let actual = fixture.cache.counter_value(key).await.unwrap();
            assert!(
                (actual - expected).abs() < 1e-9,
                "{key}: {actual} != {expected}"
            );
        }
        assert_eq!(
            fixture
                .state
                .router
                .usage_store()
                .in_flight(&fixture.store_key),
            1
        );
    }

    #[tokio::test]
    async fn abort_before_upstream_handshake_refunds_reservations_and_releases_slots_once() {
        use tokio::io::AsyncReadExt;
        let (fixture, mut guard) = reserved_session().await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let request = format!("ws://{}/realtime", listener.local_addr().unwrap())
            .into_client_request()
            .unwrap();
        let owner = tokio::spawn(async move {
            let started = guard.start_connection();
            let result = super::super::connect::connect(request, true).await;
            let outcome = match result {
                Ok(_) => {
                    guard.connection_result(Ok(()), started);
                    Ok(guard.meter().finish(SessionEnd::UpstreamClosed))
                }
                Err(error) => {
                    guard.connection_result(Err(&error), started);
                    Err(error.error)
                }
            };
            await_settlement(guard.settle(outcome)).await;
        });
        let (mut stalled_peer, _) = listener.accept().await.unwrap();
        tokio::time::timeout(
            Duration::from_secs(3),
            stalled_peer.read_exact(&mut [0u8; 1]),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            fixture
                .state
                .router
                .usage_store()
                .in_flight(&fixture.store_key),
            2
        );
        assert!(
            fixture
                .state
                .router
                .latency_store()
                .p50(&fixture.store_key)
                .is_none()
        );
        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        assert_eq!(
            fixture
                .state
                .router
                .usage_store()
                .in_flight(&fixture.store_key),
            1
        );
        wait_done(&fixture).await;
        assert_balances(&fixture, 0.0, 0.0, 1.0).await;
        assert_eq!(fixture.probe.failures.load(Ordering::SeqCst), 1);
        assert_eq!(fixture.probe.successes.load(Ordering::SeqCst), 0);
        assert!(
            fixture
                .state
                .router
                .health_store()
                .reason(&fixture.store_key)
                .is_none()
        );
        assert!(
            !fixture
                .state
                .router
                .cooldown_store()
                .is_cooling(&fixture.store_key)
        );
        for key in [
            "spend:abc",
            fixture.dep_spend.as_str(),
            "spend:provider:openai",
            "parallel:abc",
            "rpm:abc",
        ] {
            assert_eq!(
                fixture.cache.decrements.lock().unwrap().get(key),
                Some(&1),
                "{key}"
            );
        }
    }

    #[tokio::test]
    async fn abort_after_upgrade_preserves_partial_operator_accounting_and_refunds_the_key() {
        for (reported_usage, disconnect_first) in [(false, false), (true, false), (false, true)] {
            let (mut fixture, mut guard) = reserved_session().await;
            let (socket, mut client, server) = super::super::tests::session_socket_pair().await;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let request = format!("ws://{}/realtime", listener.local_addr().unwrap())
                .into_client_request()
                .unwrap();
            let checkpoint = Arc::new(Notify::new());
            let peer_checkpoint = checkpoint.clone();
            let draining = Arc::new(Notify::new());
            let peer_draining = draining.clone();
            let peer = tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                let mut upstream = tokio_tungstenite::accept_async(stream).await.unwrap();
                if reported_usage {
                    upstream
                        .send(Message::Text(
                            json!({"type":"response.done","response":{"id":"first","usage":{
                                "input_tokens":2,"output_tokens":3,"total_tokens":5
                            }}})
                            .to_string()
                            .into(),
                        ))
                        .await
                        .unwrap();
                }
                upstream.send(Message::Text(
                    json!({"type":"response.text.delta","response_id":"pending","delta":"hello"})
                        .to_string().into(),
                )).await.unwrap();
                while let Some(Ok(message)) = upstream.next().await {
                    if message.to_text().ok() == Some("checkpoint") {
                        peer_checkpoint.notify_one();
                    } else if message
                        .to_text()
                        .ok()
                        .is_some_and(|text| text.contains("response.cancel"))
                    {
                        // Withhold final usage/close so the owner is aborted
                        // inside the real client-disconnect drain boundary.
                        peer_draining.notify_one();
                    }
                }
            });
            let owner = tokio::spawn(async move {
                let started = guard.start_connection();
                let upstream = super::super::connect::connect(request, true).await.unwrap();
                guard.connection_result(Ok(()), started);
                let outcome = super::super::proxy(
                    socket,
                    upstream,
                    Box::new(super::super::OpenAIRealtimeProvider),
                    guard.meter(),
                )
                .await;
                await_settlement(guard.settle(Ok(outcome))).await;
            });
            tokio::time::timeout(Duration::from_secs(3), async {
                while let Some(Ok(message)) = client.next().await {
                    if message
                        .to_text()
                        .ok()
                        .is_some_and(|text| text.contains("response.text.delta"))
                    {
                        break;
                    }
                }
                // The upstream acknowledgement proves the pump passed its
                // delivered-output checkpoint before the hard abort.
                client
                    .send(Message::Text("checkpoint".into()))
                    .await
                    .unwrap();
                checkpoint.notified().await;
            })
            .await
            .unwrap();
            let latency = fixture
                .state
                .router
                .latency_store()
                .p50(&fixture.store_key)
                .unwrap();
            if disconnect_first {
                client.close(None).await.unwrap();
                tokio::time::timeout(Duration::from_secs(3), draining.notified())
                    .await
                    .unwrap();
            }
            owner.abort();
            assert!(owner.await.unwrap_err().is_cancelled());
            wait_done(&fixture).await;
            let router_cost = if reported_usage {
                ACTUAL_COST
            } else {
                fixture.estimate
            };
            let key_cost = if disconnect_first {
                fixture.estimate
            } else {
                0.0
            };
            assert_balances(&fixture, key_cost, router_cost, 2.0).await;
            assert_eq!(
                fixture.probe.failures.load(Ordering::SeqCst),
                usize::from(!disconnect_first)
            );
            assert_eq!(
                fixture.probe.concurrency_releases.load(Ordering::SeqCst),
                usize::from(disconnect_first)
            );
            assert_eq!(
                fixture.state.router.latency_store().p50(&fixture.store_key),
                Some(latency)
            );
            assert!(
                fixture
                    .state
                    .router
                    .health_store()
                    .reason(&fixture.store_key)
                    .is_none()
            );
            assert!(
                !fixture
                    .state
                    .router
                    .cooldown_store()
                    .is_cooling(&fixture.store_key)
            );
            let entry = fixture.spend_rx.recv().await.unwrap();
            assert_eq!(entry.status.as_deref(), Some("partial"));
            assert_eq!(entry.total_tokens, reported_usage.then_some(5));
            assert_eq!(entry.cost.is_some(), reported_usage);
            if let Some(cost) = entry.cost {
                assert!(
                    (cost - ACTUAL_COST).abs() < 1e-12,
                    "reported partial cost must match router settlement"
                );
            }
            assert!(
                fixture.spend_rx.try_recv().is_err(),
                "one terminal realtime record"
            );
            tokio::time::timeout(Duration::from_secs(3), peer)
                .await
                .unwrap()
                .unwrap();
            server.abort();
        }
    }

    #[tokio::test]
    async fn abort_during_delayed_settlement_cannot_double_refund_or_leak_parallel_slots() {
        for delay_scope in ["deployment", "provider", "key", "parallel"] {
            let (fixture, mut guard) = reserved_session().await;
            let delay_key = match delay_scope {
                "deployment" => fixture.dep_spend.clone(),
                "provider" => "spend:provider:openai".into(),
                "key" => "spend:abc".into(),
                _ => "parallel:abc".into(),
            };
            *fixture.cache.pause_key.lock().unwrap() = Some(delay_key);
            let started = guard.start_connection();
            guard.connection_result(Ok(()), started);
            let mut meter = guard.meter();
            meter.observe(&super::super::OpenAIRealtimeProvider,
                r#"{"type":"response.done","response":{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}"#);
            let outcome = meter.finish(SessionEnd::UpstreamClosed);
            let owner = tokio::spawn(async move {
                await_settlement(guard.settle(Ok(outcome))).await;
            });
            tokio::time::timeout(Duration::from_secs(3), fixture.cache.entered.notified())
                .await
                .unwrap();
            // Monetary settlement is blocked after applying a debit, but the
            // socket occupancy has already been released synchronously.
            assert_eq!(
                fixture
                    .state
                    .router
                    .usage_store()
                    .in_flight(&fixture.store_key),
                1
            );
            owner.abort();
            assert!(owner.await.unwrap_err().is_cancelled());
            fixture.cache.resume.notify_one();
            wait_done(&fixture).await;
            assert_balances(&fixture, ACTUAL_COST, ACTUAL_COST, 2.0).await;
            assert_eq!(fixture.probe.successes.load(Ordering::SeqCst), 1);
            assert_eq!(fixture.probe.failures.load(Ordering::SeqCst), 0);
            for key in [
                "spend:abc",
                fixture.dep_spend.as_str(),
                "spend:provider:openai",
                "parallel:abc",
            ] {
                assert_eq!(
                    fixture.cache.decrements.lock().unwrap().get(key),
                    Some(&1),
                    "{delay_scope}: {key}"
                );
            }
        }
    }

    #[test]
    fn guards_dropped_outside_an_entered_runtime_use_the_captured_handle() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (fixture, guard) = runtime.block_on(reserved_session());
        // No entered runtime here: tokio::spawn would panic in this Drop.
        drop(guard);
        assert_eq!(
            fixture
                .state
                .router
                .usage_store()
                .in_flight(&fixture.store_key),
            1
        );
        runtime.block_on(async {
            wait_done(&fixture).await;
            assert_balances(&fixture, 0.0, 0.0, 1.0).await;
        });
        let router_guard = runtime.block_on(async {
            RealtimeRouterBudgetGuard::new(
                fixture.cache.clone(),
                RouterBudgetReservation::default(),
            )
        });
        drop(router_guard);
        runtime.block_on(tokio::task::yield_now());
    }
}
