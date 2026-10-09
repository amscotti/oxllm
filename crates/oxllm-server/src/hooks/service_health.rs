//! Bounded dependency-health notifications for configured service callbacks.

use std::collections::HashMap;
use std::time::Duration;

use oxllm_core::ServiceFailureEvent;
use tokio::time::Instant;

use crate::state::{AppState, SharedState};

const CHECK_INTERVAL: Duration = Duration::from_secs(15);
const REPEAT_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Default)]
struct ServiceHealthMonitor {
    last_failure: HashMap<String, Instant>,
}

impl ServiceHealthMonitor {
    async fn inspect(&mut self, state: &AppState) {
        if state.is_shutting_down()
            || !state
                .hooks
                .iter()
                .any(|hook| hook.supports_service_events())
        {
            self.last_failure.clear();
            return;
        }
        let checks = crate::handlers::health::dependency_checks(state).await;
        self.last_failure.retain(|service, _| {
            checks.get(service).and_then(serde_json::Value::as_str) == Some("error")
        });
        for (service, status) in checks {
            if status.as_str() != Some("error")
                || self
                    .last_failure
                    .get(&service)
                    .is_some_and(|last| last.elapsed() < REPEAT_INTERVAL)
            {
                continue;
            }
            let event = ServiceFailureEvent {
                service: service.clone(),
                message: "configured backend is unavailable: health check failed or timed out"
                    .into(),
            };
            crate::hooks::runner::run_service_failure_hooks(&state.hooks, &event).await;
            self.last_failure.insert(service, Instant::now());
        }
    }
}

/// Follow live state so enabling/disabling callbacks during reload takes effect.
/// Without service observers the worker performs no backend probes.
pub(crate) async fn run(shared: SharedState, mut shutdown: tokio::sync::oneshot::Receiver<()>) {
    let mut interval = tokio::time::interval(CHECK_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut monitor = ServiceHealthMonitor::default();
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => return,
            _ = interval.tick() => {}
        }
        let state = shared.load_full();
        if state.is_shutting_down() {
            return;
        }
        monitor.inspect(&state).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use oxllm_core::{Cache, Hook};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct ProbeCache {
        healthy: AtomicBool,
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Cache for ProbeCache {
        async fn get(&self, _: &str) -> Option<String> {
            None
        }
        async fn set(&self, _: &str, _: String, _: Option<Duration>) {}
        async fn delete(&self, _: &str) {}
        async fn incr(&self, _: &str, _: f64, _: Option<Duration>) -> oxllm_core::Result<f64> {
            Ok(0.0)
        }
        async fn ping(&self) -> bool {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.healthy.load(Ordering::SeqCst)
        }
    }

    struct Recorder(Arc<Mutex<Vec<String>>>);
    #[async_trait]
    impl Hook for Recorder {
        async fn post_service_failure(&self, event: &ServiceFailureEvent) {
            self.0.lock().unwrap().push(event.service.clone());
        }
        fn supports_service_events(&self) -> bool {
            true
        }
    }

    fn state(cache: Arc<ProbeCache>) -> AppState {
        let config = crate::config::parse_config_str(
            "model_list: []\noxllm_settings:\n  cache: {type: redis}\n",
        )
        .unwrap();
        crate::state::StateBuilder::new(config)
            .with_cache(cache)
            .with_cache_backend(crate::state::CacheBackendKind::Redis)
            .build()
            .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn failures_are_deduplicated_repeated_and_rearmed_after_recovery() {
        let cache = Arc::new(ProbeCache {
            healthy: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        });
        let mut state = state(cache.clone());
        let events = Arc::new(Mutex::new(Vec::new()));
        state.hooks = vec![Arc::new(Recorder(events.clone()))];
        let mut monitor = ServiceHealthMonitor::default();
        monitor.inspect(&state).await;
        monitor.inspect(&state).await;
        assert_eq!(*events.lock().unwrap(), ["redis"]);
        tokio::time::advance(REPEAT_INTERVAL).await;
        monitor.inspect(&state).await;
        assert_eq!(events.lock().unwrap().len(), 2);
        cache.healthy.store(true, Ordering::SeqCst);
        monitor.inspect(&state).await;
        cache.healthy.store(false, Ordering::SeqCst);
        monitor.inspect(&state).await;
        assert_eq!(events.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn no_service_observers_or_shutdown_means_no_backend_probes() {
        let cache = Arc::new(ProbeCache {
            healthy: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        });
        let mut state = state(cache.clone());
        let mut monitor = ServiceHealthMonitor::default();
        monitor.inspect(&state).await;
        assert_eq!(cache.calls.load(Ordering::SeqCst), 0);
        state.hooks = vec![Arc::new(Recorder(Arc::new(Mutex::new(Vec::new()))))];
        state.shutdown.store(true, Ordering::Release);
        monitor.inspect(&state).await;
        assert_eq!(cache.calls.load(Ordering::SeqCst), 0);
        let shared = Arc::new(arc_swap::ArcSwap::from_pointee(state));
        let (tx, rx) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn(run(shared, rx));
        let _ = tx.send(());
        tokio::time::timeout(Duration::from_secs(1), worker)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn datadog_service_callback_posts_real_dependency_events_through_callback_wrapper() {
        let intake = httpmock::MockServer::start_async().await;
        let wire = intake
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/api/v2/logs")
                    .header("DD-API-KEY", "synthetic-key")
                    .body_contains("\"dependency\":\"redis\"")
                    .body_contains("\"event\":\"service_failure\"");
                then.status(202);
            })
            .await;
        let config = crate::config::parse_config_str(&format!("model_list: []\noxllm_settings:\n  cache: {{type: redis}}\n  callbacks: [datadog]\n  service_callback: [datadog]\n  datadog_config: {{api_key: synthetic-key, endpoint: {}}}\ngeneral_settings:\n  allow_private_api_base: true\n", intake.base_url())).unwrap();
        let cache = Arc::new(ProbeCache {
            healthy: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
        });
        let state = crate::state::StateBuilder::new(config)
            .with_cache(cache)
            .with_cache_backend(crate::state::CacheBackendKind::Redis)
            .build()
            .unwrap();
        assert!(
            state
                .hooks
                .iter()
                .any(|hook| hook.supports_service_events())
        );
        let mut monitor = ServiceHealthMonitor::default();
        monitor.inspect(&state).await;
        monitor.inspect(&state).await;
        for _ in 0..80 {
            if wire.hits_async().await >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        wire.assert_hits_async(1).await;
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn configured_database_failure_reaches_service_observers() {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .unwrap();
        db.migrate().await.unwrap();
        let config = crate::config::parse_config_str(
            "model_list: []\ngeneral_settings: {database_url: 'sqlite::memory:'}\n",
        )
        .unwrap();
        let mut state = crate::state::StateBuilder::new(config)
            .with_db(Some(db.clone()))
            .build()
            .unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        state.hooks = vec![Arc::new(Recorder(events.clone()))];
        db.pool().close().await;
        ServiceHealthMonitor::default().inspect(&state).await;
        assert_eq!(*events.lock().unwrap(), ["db"]);
    }
}
