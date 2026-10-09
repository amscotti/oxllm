//! Synchronous ownership of deployment in-flight slots across cancellable awaits.

use std::sync::Arc;

use oxllm_core::router::Router;

pub(super) struct InFlightGuard {
    router: Arc<Router>,
    key: String,
    active: bool,
}

impl InFlightGuard {
    pub(super) fn new(router: Arc<Router>, key: &str) -> Self {
        let mut guard = Self {
            router,
            key: key.into(),
            active: false,
        };
        guard.reserve();
        guard
    }

    pub(super) fn reserve(&mut self) {
        if !self.active {
            self.router.record_in_flight(&self.key, 1);
            self.active = true;
        }
    }

    pub(super) fn release(&mut self) {
        if self.active {
            self.router.record_in_flight(&self.key, -1);
            self.active = false;
        }
    }

    /// The detached pump now owns the matching decrement.
    pub(super) fn hand_off(&mut self) {
        self.active = false;
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.release();
    }
}
