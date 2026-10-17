//! Deterministic admission/top-up/settlement interleavings. The paused cache
//! returns receipts from the real in-memory implementation; no ownership is
//! inferred by the mock itself.

use super::*;
use oxllm_core::{CounterWindow, IncrResult, WindowedIncrResult};

enum Gate {
    RejectReservation { key: String, top_up: bool },
    AppliedAdjustment(String),
}

#[derive(Default)]
struct GatedCache {
    inner: InMemoryCache,
    gate: Mutex<Option<Gate>>,
    entered: Notify,
    resume: Notify,
}

impl GatedCache {
    fn take_reservation(&self, key: &str, top_up: bool) -> bool {
        let mut gate = self.gate.lock().unwrap();
        if matches!(gate.as_ref(), Some(Gate::RejectReservation { key: wanted, top_up: conditional }) if wanted == key && *conditional == top_up)
        {
            gate.take();
            true
        } else {
            false
        }
    }

    async fn after_adjustment(&self, key: &str) {
        let pause = {
            let mut gate = self.gate.lock().unwrap();
            if matches!(gate.as_ref(), Some(Gate::AppliedAdjustment(wanted)) if wanted == key) {
                gate.take();
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
impl Cache for GatedCache {
    async fn get(&self, key: &str) -> Option<String> {
        self.inner.get(key).await
    }
    async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
        self.inner.set(key, value, ttl).await;
    }
    async fn delete(&self, key: &str) {
        self.inner.delete(key).await;
    }
    async fn incr(&self, key: &str, delta: f64, ttl: Option<Duration>) -> oxllm_core::Result<f64> {
        self.inner.incr(key, delta, ttl).await
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
    async fn decr_if_exists(&self, key: &str, delta: f64) -> oxllm_core::Result<bool> {
        self.inner.decr_if_exists(key, delta).await
    }
    async fn counter_value(&self, key: &str) -> Option<f64> {
        self.inner.counter_value(key).await
    }
    async fn incr_if_below_in_window(
        &self,
        key: &str,
        delta: f64,
        ceiling: f64,
        ttl: Option<Duration>,
        expected: Option<&CounterWindow>,
    ) -> oxllm_core::Result<WindowedIncrResult> {
        if self.take_reservation(key, expected.is_some()) {
            self.entered.notify_one();
            self.resume.notified().await;
            return Ok(WindowedIncrResult::Rejected(ceiling));
        }
        self.inner
            .incr_if_below_in_window(key, delta, ceiling, ttl, expected)
            .await
    }
    async fn incr_in_window(
        &self,
        key: &str,
        delta: f64,
        window: &CounterWindow,
    ) -> oxllm_core::Result<bool> {
        let applied = self.inner.incr_in_window(key, delta, window).await?;
        if applied && delta < 0.0 {
            self.after_adjustment(key).await;
        }
        Ok(applied)
    }
}

fn fixture(cache: Arc<GatedCache>) -> Fixture {
    state_for_group_with_window(CARDS[0].0, CARDS[0].1, None, "voice", cache, "1m")
}

async fn wait_entered(cache: &GatedCache) {
    tokio::time::timeout(Duration::from_secs(3), cache.entered.notified())
        .await
        .unwrap();
}

async fn complete_b(state: &Arc<AppState>) {
    let deployment = &state.deployments[0];
    let mut ctx = session_context(deployment, crate::realtime::REALTIME_DURATION_ESTIMATE);
    ctx.request_id = "rollover-b".into();
    ctx.cost_override = Some(CARDS[0].2);
    let mut pipeline = EndpointPipeline {
        state,
        call_ctx: ctx,
    };
    assert!(pipeline.pre_call().await.unwrap().is_none());
    let dep_key = oxllm_core::router::deployment_key(deployment);
    let reservation = reserve_router_budgets(state, deployment, &dep_key, &pipeline.call_ctx)
        .await
        .unwrap();
    crate::orchestration::budget::settle_router_spend(
        state,
        deployment,
        &dep_key,
        Some(CARDS[0].2),
        &reservation,
    )
    .await;
    pipeline
        .post_success(CARDS[0].0, None, None, CARDS[0].1)
        .await;
}

async fn assert_b_survives(fixture: &mut Fixture, stale: usize, a_cost: f64, detached: bool) {
    if detached {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let notified = fixture.probe.done.notified();
                if fixture.probe.events.lock().unwrap().len() >= 2 {
                    break;
                }
                notified.await;
            }
        })
        .await
        .unwrap();
    }
    let rows = tokio::time::timeout(Duration::from_secs(3), async {
        [
            fixture.ledger.recv().await.unwrap(),
            fixture.ledger.recv().await.unwrap(),
        ]
    })
    .await
    .unwrap();
    let a = rows
        .iter()
        .find(|row| row.request_id.as_deref() == Some("req-1"))
        .unwrap();
    let b = rows
        .iter()
        .find(|row| row.request_id.as_deref() == Some("rollover-b"))
        .unwrap();
    assert_eq!(b.cost, Some(CARDS[0].2));
    assert!((a.cost.unwrap() - a_cost).abs() < 1e-12);
    for (index, key) in fixture.spend_keys.iter().enumerate() {
        let expected = if index == stale {
            CARDS[0].2
        } else {
            BASE + CARDS[0].2 + a_cost
        };
        let actual = fixture.state.cache.counter_value(key).await.unwrap();
        assert!(
            (actual - expected).abs() < 1e-12,
            "{key}: {actual} != {expected}"
        );
    }
    assert_eq!(
        fixture.state.cache.counter_value("parallel:abc").await,
        Some(1.0)
    );
    assert!(
        fixture.ledger.try_recv().is_err(),
        "one terminal row per request"
    );
}

#[tokio::test]
async fn budget_window_late_admission_rejection_cannot_rollback_b_in_a_recreated_scope() {
    // The final model gate rejects after every earlier key/team/... scope
    // reserved. Each earlier scope is independently expired/recreated by B.
    for stale in 0..8 {
        let cache = Arc::new(GatedCache::default());
        let mut fixture = fixture(cache.clone());
        seed_baselines(&fixture).await;
        *cache.gate.lock().unwrap() = Some(Gate::RejectReservation {
            key: "spend:model:voice".into(),
            top_up: false,
        });
        let state = fixture.state.clone();
        let owner = tokio::spawn(async move {
            let ctx = session_context(
                &state.deployments[0],
                crate::realtime::REALTIME_DURATION_ESTIMATE,
            );
            let mut pipeline = EndpointPipeline {
                state: &state,
                call_ctx: ctx,
            };
            assert!(pipeline.pre_call().await.is_err());
        });
        wait_entered(&cache).await;
        cache
            .inner
            .expire_counter_for_test(&fixture.spend_keys[stale])
            .await;
        complete_b(&fixture.state).await;
        cache.resume.notify_one();
        owner.await.unwrap();
        assert_b_survives(&mut fixture, stale, 0.0, false).await;
    }
}

#[tokio::test]
async fn budget_window_router_admission_rollback_cannot_refund_b_deployment_reservation() {
    let cache = Arc::new(GatedCache::default());
    let mut fixture = fixture(cache.clone());
    seed_baselines(&fixture).await;
    *cache.gate.lock().unwrap() = Some(Gate::RejectReservation {
        key: "spend:provider:xai".into(),
        top_up: false,
    });
    let state = fixture.state.clone();
    let owner = tokio::spawn(async move {
        let deployment = state.deployments[0].clone();
        let mut pipeline = EndpointPipeline {
            state: &state,
            call_ctx: session_context(&deployment, crate::realtime::REALTIME_DURATION_ESTIMATE),
        };
        assert!(pipeline.pre_call().await.unwrap().is_none());
        let routing = RealtimeRoutingGuard::new(state.clone(), &deployment, "voice");
        let guard = UpgradeAbortGuard::new(
            state.clone(),
            pipeline.call_ctx,
            deployment.clone(),
            routing,
        );
        let dep_key = oxllm_core::router::deployment_key(&deployment);
        assert!(
            reserve_router_budgets(
                &state,
                &deployment,
                &dep_key,
                &guard.settlement.as_ref().unwrap().call_ctx
            )
            .await
            .is_err()
        );
        await_settlement(guard.settle(Err(Error::BudgetExceeded("router admission".into())))).await;
    });
    wait_entered(&cache).await;
    cache
        .inner
        .expire_counter_for_test(&fixture.spend_keys[9])
        .await;
    complete_b(&fixture.state).await;
    cache.resume.notify_one();
    owner.await.unwrap();
    assert_b_survives(&mut fixture, 9, 0.0, false).await;
}

#[tokio::test]
async fn budget_window_top_up_rollback_and_failure_release_do_not_touch_b_replacements() {
    for stale in 0..8 {
        let cache = Arc::new(GatedCache::default());
        let (mut fixture, mut guard) = reserve(fixture(cache.clone())).await;
        *cache.gate.lock().unwrap() = Some(Gate::RejectReservation {
            key: "spend:model:voice".into(),
            top_up: true,
        });
        let owner = tokio::spawn(async move {
            let settlement = guard.settlement.as_mut().unwrap();
            assert!(
                crate::pipeline::prepare_deployment_pricing_with_duration_estimate(
                    &settlement.state,
                    &mut settlement.call_ctx,
                    &settlement.deployment,
                    &json!({"model":"voice"}),
                    crate::realtime::REALTIME_DURATION_ESTIMATE,
                )
                .await
                .is_err()
            );
            await_settlement(guard.settle(Err(Error::BudgetExceeded("top-up rejected".into()))))
                .await;
        });
        wait_entered(&cache).await;
        cache
            .inner
            .expire_counter_for_test(&fixture.spend_keys[stale])
            .await;
        complete_b(&fixture.state).await;
        cache.resume.notify_one();
        owner.await.unwrap();
        assert_b_survives(&mut fixture, stale, 0.0, false).await;
    }
}

#[tokio::test]
async fn budget_window_owner_abort_after_applied_adjustment_cannot_refund_b_new_window() {
    for stale in 0..11 {
        let cache = Arc::new(GatedCache::default());
        let (mut fixture, mut guard) = reserve_with_duration(
            fixture(cache.clone()),
            crate::realtime::REALTIME_DURATION_ESTIMATE,
        )
        .await;
        *cache.gate.lock().unwrap() =
            Some(Gate::AppliedAdjustment(fixture.spend_keys[stale].clone()));
        connected(&mut guard);
        // A known snapshot keeps the monetary barrier independent of elapsed
        // scheduling; numeric catalog-duration resolution is covered above.
        let cost = 70.0 * CARDS[0].2;
        guard.settlement.as_mut().unwrap().call_ctx.cost_override = Some(cost);
        let mut meter = guard.meter();
        meter.delivered_binary();
        let outcome = meter.finish(SessionEnd::UpstreamError(Error::Network(
            "after work".into(),
        )));
        let owner = tokio::spawn(async move {
            await_settlement(guard.settle(Ok(outcome))).await;
        });
        wait_entered(&cache).await;
        cache
            .inner
            .expire_counter_for_test(&fixture.spend_keys[stale])
            .await;
        complete_b(&fixture.state).await;
        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        cache.resume.notify_one();
        assert_b_survives(&mut fixture, stale, cost, true).await;
    }
}

#[tokio::test]
async fn budget_window_top_up_of_a_replaced_first_scope_fails_closed_without_debiting_b() {
    let cache = Arc::new(GatedCache::default());
    let (mut fixture, mut guard) = reserve(fixture(cache.clone())).await;
    cache
        .inner
        .expire_counter_for_test(&fixture.spend_keys[0])
        .await;
    complete_b(&fixture.state).await;
    let settlement = guard.settlement.as_mut().unwrap();
    let old_estimate = settlement.call_ctx.budget_estimate;
    let error = crate::pipeline::prepare_deployment_pricing_with_duration_estimate(
        &settlement.state,
        &mut settlement.call_ctx,
        &settlement.deployment,
        &json!({"model":"voice"}),
        crate::realtime::REALTIME_DURATION_ESTIMATE,
    )
    .await
    .unwrap_err();
    // A stale admission receipt is a retryable race, not an exhausted cap —
    // it must classify as RateLimited so clients retry instead of giving up.
    assert!(
        matches!(error, Error::RateLimited(ref message) if message.contains("window changed")),
        "expected retryable RateLimited, got {error:?}"
    );
    assert_eq!(settlement.call_ctx.budget_estimate, old_estimate);
    assert!(
        (cache.counter_value(&fixture.spend_keys[0]).await.unwrap() - CARDS[0].2).abs() < 1e-12
    );
    await_settlement(guard.settle(Err(error))).await;
    assert_b_survives(&mut fixture, 0, 0.0, false).await;
}
