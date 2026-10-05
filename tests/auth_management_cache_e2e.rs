//! Auth-cache mutation interleavings and virtual-key cache-control policies.

mod common;

use std::error::Error as StdError;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use httpmock::MockServer;
use oxllm_core::{Cache, Error};
use oxllm_server::app::build_router;
use oxllm_server::auth::hash_key_hex;
use oxllm_server::cache::{DualCache, InMemoryCache};
use oxllm_server::state::StateBuilder;
use oxllm_server::{Database, SharedState};
use serde_json::{Value, json};
use tokio::sync::{Barrier, Mutex};

type TestResult<T = ()> = Result<T, Box<dyn StdError + Send + Sync>>;
type CacheFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const MASTER_KEY: &str = "sk-master-auth-cache-controls";
const DEADLINE: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Set,
    Delete,
}

struct Pause {
    operation: Operation,
    key: String,
    reached: Arc<Barrier>,
    resume: Arc<Barrier>,
}

struct Gate {
    reached: Arc<Barrier>,
    resume: Arc<Barrier>,
}

impl Gate {
    async fn reached(&self) -> TestResult {
        tokio::time::timeout(DEADLINE, self.reached.wait()).await?;
        Ok(())
    }

    async fn resume(&self) -> TestResult {
        tokio::time::timeout(DEADLINE, self.resume.wait()).await?;
        Ok(())
    }
}

/// A deterministic late-write gate plus a failed-cache-write fixture. Reads
/// retain warm memory during an outage, as DualCache can; failed deletes and
/// missed broadcasts leave those old snapshots intact on purpose.
struct ControlledCache {
    memory: InMemoryCache,
    pause: Mutex<Option<Pause>>,
    drop_writes: AtomicBool,
}

impl ControlledCache {
    fn new() -> Self {
        Self {
            memory: InMemoryCache::default(),
            pause: Mutex::new(None),
            drop_writes: AtomicBool::new(false),
        }
    }

    async fn arm(&self, operation: Operation, key: String) -> Gate {
        let gate = Gate {
            reached: Arc::new(Barrier::new(2)),
            resume: Arc::new(Barrier::new(2)),
        };
        *self.pause.lock().await = Some(Pause {
            operation,
            key,
            reached: gate.reached.clone(),
            resume: gate.resume.clone(),
        });
        gate
    }

    async fn pause(&self, operation: Operation, key: &str) {
        let pause = {
            let mut slot = self.pause.lock().await;
            if slot
                .as_ref()
                .is_some_and(|pause| pause.operation == operation && pause.key == key)
            {
                slot.take()
            } else {
                None
            }
        };
        if let Some(pause) = pause {
            pause.reached.wait().await;
            pause.resume.wait().await;
        }
    }
}

// Spell the async-trait ABI directly so the root integration crate needs no
// additional dependency just to decorate the production Cache trait.
impl Cache for ControlledCache {
    fn get<'a, 'b, 'c>(&'a self, key: &'b str) -> CacheFuture<'c, Option<String>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(async move { self.memory.get(key).await })
    }

    fn set<'a, 'b, 'c>(
        &'a self,
        key: &'b str,
        value: String,
        ttl: Option<Duration>,
    ) -> CacheFuture<'c, ()>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(async move {
            self.pause(Operation::Set, key).await;
            if !self.drop_writes.load(Ordering::SeqCst) {
                self.memory.set(key, value, ttl).await;
            }
        })
    }

    fn delete<'a, 'b, 'c>(&'a self, key: &'b str) -> CacheFuture<'c, ()>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(async move {
            self.pause(Operation::Delete, key).await;
            if !self.drop_writes.load(Ordering::SeqCst) {
                self.memory.delete(key).await;
            }
        })
    }

    fn incr<'a, 'b, 'c>(
        &'a self,
        key: &'b str,
        by: f64,
        ttl: Option<Duration>,
    ) -> CacheFuture<'c, Result<f64, Error>>
    where
        'a: 'c,
        'b: 'c,
        Self: 'c,
    {
        Box::pin(async move {
            if self.drop_writes.load(Ordering::SeqCst) {
                return Err(Error::Network("test cache outage".to_string()));
            }
            self.memory.incr(key, by, ttl).await
        })
    }
}

struct Proxy {
    inner: common::OxllmHandle,
    server: tokio::task::JoinHandle<std::io::Result<()>>,
    shared: SharedState,
}

impl Proxy {
    fn url(&self, path: &str) -> String {
        self.inner.url(path)
    }

    async fn shutdown(self) -> TestResult {
        let _ = self.inner.shutdown_tx.send(());
        let result = tokio::time::timeout(DEADLINE, self.server).await?;
        result??;
        Ok(())
    }
}

async fn fresh_db() -> TestResult<Arc<Database>> {
    let db = Database::connect("sqlite::memory:").await?;
    db.migrate().await?;
    Ok(db)
}

async fn start_proxy(
    mock: &MockServer,
    db: Arc<Database>,
    cache: Arc<dyn Cache>,
) -> TestResult<Proxy> {
    let mut config = common::proxy_config(&mock.base_url());
    config.general_settings.master_key = Some(MASTER_KEY.to_string());
    config.general_settings.allow_private_api_base = Some(true);
    let state = StateBuilder::new(config)
        .with_cache(cache)
        .with_db(Some(db))
        .build()?;
    let shared = Arc::new(ArcSwap::from_pointee(state));
    let router = build_router(shared.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    Ok(Proxy {
        inner: common::OxllmHandle { addr, shutdown_tx },
        server,
        shared,
    })
}

fn mock_chat(mock: &MockServer) -> httpmock::Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200).json_body(json!({
            "id":"auth-cache-test", "object":"chat.completion", "created":1,
            "model":"gpt-4o",
            "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
        }));
    })
}

fn chat_body() -> Value {
    json!({"model":"gpt-4o","messages":[{"role":"user","content":"cache policy"}]})
}

async fn master_post(proxy: &Proxy, path: &str, body: Value) -> TestResult<reqwest::Response> {
    Ok(common::http_client()
        .post(proxy.url(path))
        .bearer_auth(MASTER_KEY)
        .json(&body)
        .send()
        .await?)
}

async fn create_key(proxy: &Proxy, body: Value) -> TestResult<(String, String)> {
    let controls = body.get("allowed_cache_controls").cloned();
    let response = master_post(proxy, "/key/generate", body).await?;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await?;
    if let Some(controls) = controls {
        assert_eq!(body["allowed_cache_controls"], controls);
    }
    let key = body["key"].as_str().ok_or("missing generated key")?;
    let id = body["key_id"].as_str().ok_or("missing generated key id")?;
    Ok((key.to_string(), id.to_string()))
}

async fn chat(url: String, key: String, body: Value) -> TestResult<reqwest::Response> {
    Ok(common::http_client()
        .post(url)
        .bearer_auth(key)
        .json(&body)
        .send()
        .await?)
}

async fn chat_status(proxy: &Proxy, key: &str, body: Value) -> TestResult<u16> {
    Ok(
        chat(proxy.url("/v1/chat/completions"), key.to_string(), body)
            .await?
            .status()
            .as_u16(),
    )
}

async fn key_info(proxy: &Proxy, id: &str) -> TestResult<Value> {
    let response = common::http_client()
        .get(proxy.url("/key/info"))
        .bearer_auth(MASTER_KEY)
        .query(&[("key_id", id)])
        .send()
        .await?;
    assert_eq!(response.status(), 200);
    Ok(response.json().await?)
}

#[tokio::test]
async fn late_key_fill_cannot_resurrect_block_delete_purge_or_limit_update() -> TestResult {
    for (path, patch, expected) in [
        ("/key/block", json!({}), 401),
        ("/key/delete", json!({}), 401),
        ("/key/purge", json!({}), 401),
        ("/key/update", json!({"rpm":0}), 429),
    ] {
        let mock = MockServer::start();
        let upstream = mock_chat(&mock);
        let cache = Arc::new(ControlledCache::new());
        let proxy = start_proxy(&mock, fresh_db().await?, cache.clone()).await?;
        let (key, id) = create_key(&proxy, json!({})).await?;
        cache
            .memory
            .set("resp:unrelated", "keep".into(), None)
            .await;
        cache.memory.incr("spend:unrelated", 7.0, None).await?;

        let gate = cache
            .arm(
                Operation::Set,
                format!("vk:{}", hash_key_hex(key.as_bytes())),
            )
            .await;
        let request = tokio::spawn(chat(
            proxy.url("/v1/chat/completions"),
            key.clone(),
            chat_body(),
        ));
        gate.reached().await?; // DB snapshot read; old cache write has not landed.
        let mut mutation = patch;
        mutation["key_id"] = json!(id);
        assert_eq!(master_post(&proxy, path, mutation).await?.status(), 200);
        gate.resume().await?; // Old snapshot writes AFTER completed invalidation.
        let response = tokio::time::timeout(DEADLINE, request).await???;
        assert_eq!(response.status(), expected, "late fill after {path}");
        assert_eq!(
            chat_status(&proxy, &key, chat_body()).await?,
            expected,
            "later request after {path}"
        );
        assert_eq!(
            cache.memory.get("resp:unrelated").await.as_deref(),
            Some("keep")
        );
        assert_eq!(
            cache.memory.counter_value("spend:unrelated").await,
            Some(7.0)
        );
        upstream.assert_hits(0);
        proxy.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn late_scope_fills_observe_completed_blocks_and_limit_updates() -> TestResult {
    for (create_path, scope, key_scope, update_path, update, cache_key, expected) in [
        (
            "/team/add",
            json!({"team_id":"scope","team_name":"Scope"}),
            json!({"team_id":"scope"}),
            "/team/block",
            json!({"team_id":"scope"}),
            "vk_team:scope",
            401,
        ),
        (
            "/user/add",
            json!({"user_id":"scope","user_name":"Scope"}),
            json!({"user_id":"scope"}),
            "/user/update",
            json!({"user_id":"scope","max_budget":0.0}),
            "vk_user:scope",
            429,
        ),
        (
            "/organization/add",
            json!({"org_id":"scope","org_name":"Scope"}),
            json!({"org_id":"scope"}),
            "/organization/update",
            json!({"org_id":"scope","rpm":0}),
            "vk_org:scope",
            429,
        ),
        (
            "/project/add",
            json!({"project_id":"scope","project_name":"Scope"}),
            json!({"project_id":"scope"}),
            "/project/update",
            json!({"project_id":"scope","rpm":0}),
            "vk_project:scope",
            429,
        ),
        (
            "/customer/new",
            json!({"end_user_id":"scope"}),
            json!({}),
            "/customer/block",
            json!({"end_user_id":"scope"}),
            "vk_customer:scope",
            403,
        ),
        (
            "/tag/new",
            json!({"name":"scope"}),
            json!({}),
            "/tag/update",
            json!({"name":"scope","max_budget":0.0}),
            "vk_tag:scope",
            429,
        ),
    ] {
        let mock = MockServer::start();
        let upstream = mock_chat(&mock);
        let cache = Arc::new(ControlledCache::new());
        let proxy = start_proxy(&mock, fresh_db().await?, cache.clone()).await?;
        assert_eq!(master_post(&proxy, create_path, scope).await?.status(), 200);
        let (key, _) = create_key(&proxy, key_scope).await?;
        let mut body = chat_body();
        if cache_key.starts_with("vk_customer:") {
            body["user"] = json!("scope");
        }
        if cache_key.starts_with("vk_tag:") {
            body["metadata"] = json!({"tags":["scope"]});
        }
        let gate = cache.arm(Operation::Set, cache_key.to_string()).await;
        let request = tokio::spawn(chat(
            proxy.url("/v1/chat/completions"),
            key.clone(),
            body.clone(),
        ));
        gate.reached().await?;
        assert_eq!(
            master_post(&proxy, update_path, update).await?.status(),
            200
        );
        gate.resume().await?;
        let response = tokio::time::timeout(DEADLINE, request).await???;
        assert_eq!(response.status(), expected, "late fill of {cache_key}");
        assert_eq!(chat_status(&proxy, &key, body).await?, expected);
        upstream.assert_hits(0);
        proxy.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn bulk_reassignment_commits_before_invalidation_and_fences_late_fill() -> TestResult {
    for operation in [Operation::Delete, Operation::Set] {
        let mock = MockServer::start();
        let upstream = mock_chat(&mock);
        let db = fresh_db().await?;
        let cache = Arc::new(ControlledCache::new());
        let proxy = start_proxy(&mock, db.clone(), cache.clone()).await?;
        for (path, body) in [
            ("/team/add", json!({"team_id":"old","team_name":"Old"})),
            ("/team/add", json!({"team_id":"new","team_name":"New"})),
            ("/team/block", json!({"team_id":"new"})),
            ("/user/add", json!({"user_id":"owner","user_name":"Owner"})),
        ] {
            assert_eq!(master_post(&proxy, path, body).await?.status(), 200);
        }
        let (key, id) = create_key(&proxy, json!({"team_id":"old","user_id":"owner"})).await?;
        if operation == Operation::Delete {
            assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 200);
        }
        let gate = cache
            .arm(operation, format!("vk:{}", hash_key_hex(key.as_bytes())))
            .await;
        if operation == Operation::Delete {
            let url = proxy.url("/team/member/add");
            let mutation = tokio::spawn(async move {
                common::http_client()
                    .post(url)
                    .bearer_auth(MASTER_KEY)
                    .json(&json!({"user_id":"owner","team_id":"new"}))
                    .send()
                    .await
            });
            gate.reached().await?;
            let record = db.get_key(&id).await?.ok_or("missing reassigned key")?;
            assert_eq!(
                record.team_id.as_deref(),
                Some("new"),
                "invalidation must follow commit"
            );
            assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 401);
            gate.resume().await?;
            let response = tokio::time::timeout(DEADLINE, mutation).await???;
            assert_eq!(response.status(), 200);
            assert_eq!(response.json::<Value>().await?["keys_affected"], 1);
            upstream.assert_hits(1);
        } else {
            let request = tokio::spawn(chat(
                proxy.url("/v1/chat/completions"),
                key.clone(),
                chat_body(),
            ));
            gate.reached().await?;
            let response = master_post(
                &proxy,
                "/team/member/add",
                json!({"user_id":"owner","team_id":"new"}),
            )
            .await?;
            assert_eq!(response.status(), 200);
            assert_eq!(response.json::<Value>().await?["keys_affected"], 1);
            gate.resume().await?;
            let response = tokio::time::timeout(DEADLINE, request).await???;
            assert_eq!(
                response.status(),
                401,
                "late fill must reload the new blocked team"
            );
            upstream.assert_hits(0);
        }
        assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 401);
        proxy.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn cache_outage_and_recovery_cannot_resurrect_a_revoked_warm_snapshot() -> TestResult {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let cache = Arc::new(ControlledCache::new());
    let proxy = start_proxy(&mock, fresh_db().await?, cache.clone()).await?;
    let (key, id) = create_key(&proxy, json!({})).await?;
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 200);
    let cache_key = format!("vk:{}", hash_key_hex(key.as_bytes()));
    let old = cache
        .memory
        .get(&cache_key)
        .await
        .ok_or("missing warm snapshot")?;
    cache.drop_writes.store(true, Ordering::SeqCst);
    assert_eq!(
        master_post(&proxy, "/key/block", json!({"key_id":id}))
            .await?
            .status(),
        200
    );
    assert_eq!(
        cache.memory.get(&cache_key).await.as_deref(),
        Some(old.as_str()),
        "failed delete retains stale memory"
    );
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 401);
    cache.drop_writes.store(false, Ordering::SeqCst);
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 401);
    upstream.assert_hits(1);
    proxy.shutdown().await
}

#[tokio::test]
async fn dual_cache_cannot_trust_a_locally_stale_version_after_peer_revocation() -> TestResult {
    let mut backends = vec![None];
    if let Some(redis) = common::connect_redis("auth-cache-version").await {
        backends.push(Some(redis));
    }
    for redis in backends {
        let mock = MockServer::start();
        let upstream = mock_chat(&mock);
        let db = fresh_db().await?;
        let memory_a = Arc::new(InMemoryCache::default());
        let memory_b = Arc::new(InMemoryCache::default());
        let replica_a = start_proxy(
            &mock,
            db.clone(),
            Arc::new(DualCache::new(memory_a, redis.clone())),
        )
        .await?;
        let replica_b =
            start_proxy(&mock, db, Arc::new(DualCache::new(memory_b.clone(), redis))).await?;
        let (key, id) = create_key(&replica_a, json!({})).await?;
        assert_eq!(chat_status(&replica_b, &key, chat_body()).await?, 200);
        let cache_key = format!("vk:{}", hash_key_hex(key.as_bytes()));
        let old = memory_b
            .get(&cache_key)
            .await
            .ok_or("missing peer snapshot")?;
        // No subscriber is installed in this fixture: deliberately miss pub/sub.
        assert_eq!(
            master_post(&replica_a, "/key/block", json!({"key_id":id}))
                .await?
                .status(),
            200
        );
        assert_eq!(
            memory_b.get(&cache_key).await.as_deref(),
            Some(old.as_str())
        );
        assert_eq!(chat_status(&replica_b, &key, chat_body()).await?, 401);
        upstream.assert_hits(1);
        replica_a.shutdown().await?;
        replica_b.shutdown().await?;
    }
    Ok(())
}

#[tokio::test]
async fn late_missing_key_fill_cannot_hide_a_committed_key_creation() -> TestResult {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let db = fresh_db().await?;
    let cache = Arc::new(ControlledCache::new());
    let proxy = start_proxy(&mock, db.clone(), cache.clone()).await?;
    let key = "sk-late-negative-fill";
    let hash = hash_key_hex(key.as_bytes());
    let cache_key = format!("vk:{hash}");
    let gate = cache.arm(Operation::Set, cache_key.clone()).await;
    let request = tokio::spawn(chat(
        proxy.url("/v1/chat/completions"),
        key.to_string(),
        chat_body(),
    ));
    gate.reached().await?;
    let now = oxllm_server::now_iso();
    let record = serde_json::from_value(json!({
        "key_id":"created-during-fill", "key_hash":hash,
        "created_at":now, "updated_at":now, "is_active":true
    }))?;
    db.create_key(&record).await?;
    let state = proxy.shared.load_full();
    oxllm_server::auth::invalidate_cache_key(&state, &cache_key).await;
    gate.resume().await?;
    let response = tokio::time::timeout(DEADLINE, request).await???;
    assert_eq!(response.status(), 200);
    assert_eq!(chat_status(&proxy, key, chat_body()).await?, 200);
    upstream.assert_hits(2);
    proxy.shutdown().await
}

#[tokio::test]
async fn db_versions_detect_policy_changes_even_with_an_unchanged_timestamp() -> TestResult {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let db = fresh_db().await?;
    let proxy = start_proxy(&mock, db.clone(), Arc::new(InMemoryCache::default())).await?;
    let (key, id) = create_key(&proxy, json!({})).await?;
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 200);
    let mut record = db.get_key(&id).await?.ok_or("missing key")?;
    // A timestamp-only "generation" would still match the warm snapshot.
    record.rpm = Some(0);
    assert!(db.update_key(&record).await?);
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 429);
    upstream.assert_hits(1);
    proxy.shutdown().await
}

#[tokio::test]
async fn warm_auth_snapshot_cannot_bypass_an_unreachable_db_version_authority() -> TestResult {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let db = fresh_db().await?;
    let proxy = start_proxy(&mock, db.clone(), Arc::new(InMemoryCache::default())).await?;
    let (key, _) = create_key(&proxy, json!({})).await?;
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 200);
    let cache_key = format!("vk:{}", hash_key_hex(key.as_bytes()));
    assert!(
        proxy
            .shared
            .load_full()
            .cache
            .get(&cache_key)
            .await
            .is_some()
    );
    db.pool().close().await;
    for _ in 0..2 {
        assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 503);
    }
    upstream.assert_hits(1);
    proxy.shutdown().await
}

#[tokio::test]
async fn virtual_key_cache_controls_reach_management_auth_and_cache_hook() -> TestResult {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let db = fresh_db().await?;
    let proxy = start_proxy(&mock, db.clone(), Arc::new(InMemoryCache::default())).await?;
    let (key, id) = create_key(&proxy, json!({"allowed_cache_controls":["insert"]})).await?;
    assert_eq!(
        key_info(&proxy, &id).await?["allowed_cache_controls"],
        json!(["insert"])
    );
    assert_eq!(
        db.get_key(&id)
            .await?
            .ok_or("missing key")?
            .allowed_cache_controls
            .as_deref(),
        Some("[\"insert\"]")
    );
    let mut body = chat_body();
    body["cache"] = json!({"insert":true});
    assert_eq!(chat_status(&proxy, &key, body.clone()).await?, 200);
    let cached = chat(proxy.url("/v1/chat/completions"), key.clone(), body.clone()).await?;
    assert_eq!(cached.status(), 200);
    assert!(cached.headers().contains_key("x-oxllm-cache-hit"));

    for controls in [
        json!({"insert":true,"no-cache":true}),
        json!({"insert":false,"no-store":false}),
    ] {
        body["cache"] = controls;
        // Caller-supplied metadata cannot override the stamped auth policy.
        body["allowed_cache_controls"] = json!(["insert", "no-cache", "no-store"]);
        let response = chat(proxy.url("/v1/chat/completions"), key.clone(), body.clone()).await?;
        assert_eq!(response.status(), 400);
        assert!(response.text().await?.contains("not allowed"));
    }
    upstream.assert_hits(1);

    for patch in [
        json!({"key_id":id,"key_alias":"renamed"}),
        json!({"key_id":id,"allowed_cache_controls":null}),
    ] {
        let response = master_post(&proxy, "/key/update", patch).await?;
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.json::<Value>().await?["allowed_cache_controls"],
            json!(["insert"])
        );
    }
    let response = master_post(
        &proxy,
        "/key/update",
        json!({"key_id":id,"allowed_cache_controls":[]}),
    )
    .await?;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.json::<Value>().await?["allowed_cache_controls"],
        json!([])
    );
    assert_eq!(
        key_info(&proxy, &id).await?["allowed_cache_controls"],
        json!([])
    );
    assert_eq!(
        db.get_key(&id)
            .await?
            .ok_or("missing key")?
            .allowed_cache_controls
            .as_deref(),
        Some("[]")
    );
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 200);
    body["cache"] = json!({"insert":true});
    assert_eq!(
        chat_status(&proxy, &key, body.clone()).await?,
        400,
        "[] is deny-all, not unrestricted"
    );

    let response = master_post(
        &proxy,
        "/key/update",
        json!({"key_id":id,"clear_allowed_cache_controls":true}),
    )
    .await?;
    assert_eq!(response.status(), 200);
    assert!(response.json::<Value>().await?["allowed_cache_controls"].is_null());
    assert!(
        db.get_key(&id)
            .await?
            .ok_or("missing key")?
            .allowed_cache_controls
            .is_none()
    );
    body["cache"] = json!({"use_cache":true,"no-cache":true,"no-store":true});
    assert_eq!(chat_status(&proxy, &key, body.clone()).await?, 200);
    let (unrestricted, _) = create_key(&proxy, json!({})).await?;
    assert_eq!(
        chat_status(&proxy, &unrestricted, body).await?,
        200,
        "absent policy remains unrestricted"
    );
    proxy.shutdown().await
}

#[tokio::test]
async fn cache_control_validation_rejects_unsupported_names_without_mutating_keys() -> TestResult {
    let mock = MockServer::start();
    let db = fresh_db().await?;
    let proxy = start_proxy(&mock, db.clone(), Arc::new(InMemoryCache::default())).await?;
    for controls in [
        json!(["ttl"]),
        json!(["no_cache"]),
        json!(["*"]),
        json!([" insert"]),
        json!(["insert", "max-age"]),
    ] {
        let response = master_post(
            &proxy,
            "/key/generate",
            json!({"allowed_cache_controls":controls}),
        )
        .await?;
        assert_eq!(response.status(), 400);
        assert!(response.text().await?.contains("allowed_cache_controls"));
    }
    assert!(db.list_keys(None).await?.is_empty());
    let (_, id) = create_key(&proxy, json!({"allowed_cache_controls":["insert"]})).await?;
    for patch in [
        json!({"key_id":id,"allowed_cache_controls":["unsupported"],"is_active":false}),
        json!({"key_id":id,"allowed_cache_controls":[],"clear_allowed_cache_controls":true}),
    ] {
        assert_eq!(
            master_post(&proxy, "/key/update", patch).await?.status(),
            400
        );
        let record = db.get_key(&id).await?.ok_or("missing key")?;
        assert!(record.is_active);
        assert_eq!(
            record.allowed_cache_controls.as_deref(),
            Some("[\"insert\"]")
        );
    }
    proxy.shutdown().await
}

#[tokio::test]
async fn all_supported_cache_controls_work_and_policy_survives_rotation() -> TestResult {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let proxy = start_proxy(&mock, fresh_db().await?, Arc::new(InMemoryCache::default())).await?;
    let controls = ["insert", "use_cache", "no-cache", "no-store"];
    assert_eq!(oxllm_server::auth::SUPPORTED_CACHE_CONTROLS, &controls);
    let (old_key, id) = create_key(&proxy, json!({"allowed_cache_controls":controls})).await?;
    let mut body = chat_body();
    body["cache"] = json!({"insert":true,"use_cache":true,"no-cache":true,"no-store":true});
    for _ in 0..2 {
        assert_eq!(chat_status(&proxy, &old_key, body.clone()).await?, 200);
    }
    upstream.assert_hits(2);
    let response = master_post(&proxy, "/key/rotate", json!({"key_id":id})).await?;
    assert_eq!(response.status(), 200);
    let rotation: Value = response.json().await?;
    let key = rotation["key"]
        .as_str()
        .ok_or("missing rotated key")?
        .to_string();
    assert_eq!(chat_status(&proxy, &old_key, body.clone()).await?, 401);
    assert_eq!(chat_status(&proxy, &key, body.clone()).await?, 200);
    assert_eq!(
        key_info(&proxy, &id).await?["allowed_cache_controls"],
        json!(controls)
    );
    body["cache"]["unsupported"] = json!(true);
    assert_eq!(chat_status(&proxy, &key, body).await?, 400);
    upstream.assert_hits(3);
    proxy.shutdown().await
}

#[tokio::test]
async fn corrupt_persisted_cache_control_policy_fails_closed_on_explicit_controls() -> TestResult {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let db = fresh_db().await?;
    let proxy = start_proxy(&mock, db.clone(), Arc::new(InMemoryCache::default())).await?;
    let (key, id) = create_key(&proxy, json!({})).await?;
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 200);
    let mut record = db.get_key(&id).await?.ok_or("missing key")?;
    record.allowed_cache_controls = Some("not-json".to_string());
    assert!(db.update_key(&record).await?);
    let mut body = chat_body();
    body["cache"] = json!({"insert":true});
    assert_eq!(chat_status(&proxy, &key, body).await?, 400);
    assert_eq!(chat_status(&proxy, &key, chat_body()).await?, 200);
    upstream.assert_hits(2);
    proxy.shutdown().await
}
