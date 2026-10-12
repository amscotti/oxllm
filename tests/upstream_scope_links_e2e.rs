//! Native management scope-link contracts over HTTP with SQLite persistence.

mod common;

use std::sync::Arc;
use std::time::Duration;

use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use oxllm_server::{Database, OxllmParams};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-scope-links-master-key";
const LITERAL_MODELS: &str = "[\"  \",\" scope-model \",\"gpt-*\"]";

struct Proxy {
    db: Arc<Database>,
    client: reqwest::Client,
    address: std::net::SocketAddr,
    shutdown: tokio::sync::oneshot::Sender<()>,
    server: tokio::task::JoinHandle<()>,
}

impl Proxy {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }

    async fn post(&self, path: &str, body: Value) -> (u16, Value) {
        let response = self
            .client
            .post(self.url(path))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap())
    }

    async fn ok(&self, path: &str, body: Value) -> Value {
        let (status, body) = self.post(path, body).await;
        assert_eq!(status, 200, "{path}: {body}");
        body
    }

    async fn reject(&self, path: &str, request: Value, status: u16, message: &str) {
        let (actual, body) = self.post(path, request).await;
        assert_eq!(actual, status, "{path}: {body}");
        assert_eq!(
            body.as_object().unwrap().len(),
            1,
            "native envelope: {body}"
        );
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"].as_str().unwrap().contains(message),
            "{body}"
        );
    }

    async fn models(&self, key: &str) -> Vec<Value> {
        let response = self
            .client
            .get(self.url("/v1/models"))
            .bearer_auth(key)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        body["data"].as_array().unwrap().clone()
    }

    async fn finish(self) {
        let _ = self.shutdown.send(());
        tokio::time::timeout(Duration::from_secs(5), self.server)
            .await
            .unwrap()
            .unwrap();
        self.db.pool().close().await;
    }
}

async fn proxy() -> Proxy {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    db.migrate().await.unwrap();
    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "scope-model".into(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".into(),
                api_key: Some("synthetic-provider-key".into()),
                ..Default::default()
            },
            ..Default::default()
        }],
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.into()),
            ..Default::default()
        },
        oxllm_settings: OxllmSettings::default(),
        router_settings: Default::default(),
        extra: Default::default(),
    };
    let shared = build_shared_state_with_db(config, None, Some(db.clone())).unwrap();
    let router = build_router(shared);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown, receiver) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = receiver.await;
            })
            .await
            .unwrap();
    });
    Proxy {
        db,
        client: common::http_client(),
        address,
        shutdown,
        server,
    }
}

async fn seed(proxy: &Proxy) {
    for team in ["team-a", "team-b"] {
        proxy
            .ok("/team/add", json!({"team_id":team,"team_name":team}))
            .await;
    }
    proxy
        .ok("/user/add", json!({"user_id":"user","user_name":"User"}))
        .await;
    for (project, team) in [
        ("owned-a", Some("team-a")),
        ("owned-b", Some("team-b")),
        ("shared", None),
    ] {
        proxy
            .ok(
                "/project/add",
                json!({
                    "project_id":project,"project_name":project,"team_id":team,"max_budget":50
                }),
            )
            .await;
    }
}

async fn key_snapshot(proxy: &Proxy, id: &str) -> Value {
    serde_json::to_value(proxy.db.get_key(id).await.unwrap().unwrap()).unwrap()
}

async fn project_snapshot(proxy: &Proxy, id: &str) -> Value {
    serde_json::to_value(proxy.db.get_project(id).await.unwrap().unwrap()).unwrap()
}

#[tokio::test]
async fn keys_enforce_effective_team_project_pairs_and_preserve_literal_policies() {
    let proxy = proxy().await;
    seed(&proxy).await;
    for team in [Value::Null, json!("team-b")] {
        proxy
            .reject(
                "/key/generate",
                json!({"team_id":team,"project_id":"owned-a"}),
                400,
                "belongs to team",
            )
            .await;
    }
    assert!(proxy.db.list_keys(None).await.unwrap().is_empty());
    let created = proxy.ok("/key/generate", json!({
        "team_id":"team-a","project_id":"owned-a","user_id":"user","rpm_limit":7,"tpm_limit":90
    })).await;
    let id = created["key_id"].as_str().unwrap();
    let mut imported = proxy.db.get_key(id).await.unwrap().unwrap();
    imported.models = Some(LITERAL_MODELS.into());
    imported.allowed_cache_controls = Some("[]".into());
    proxy.db.update_key(&imported).await.unwrap();
    let before = key_snapshot(&proxy, id).await;

    for update in [
        json!({"key_id":id,"team_id":"team-b","key_alias":"must-not-persist","max_budget":0}),
        json!({"key_id":id,"team_id":""}),
        json!({"key_id":id,"project_id":"owned-b"}),
    ] {
        proxy
            .reject("/key/update", update, 400, "belongs to team")
            .await;
        assert_eq!(key_snapshot(&proxy, id).await, before);
    }
    let literal: Value = serde_json::from_str(LITERAL_MODELS).unwrap();
    let mut team = proxy.db.get_team("team-a").await.unwrap().unwrap();
    team.models = Some(LITERAL_MODELS.into());
    proxy.db.update_team(&team).await.unwrap();
    proxy
        .ok("/team/update", json!({"team_id":"team-a","models":literal}))
        .await;
    assert_eq!(
        proxy.db.get_team("team-a").await.unwrap().unwrap().models,
        Some(LITERAL_MODELS.into())
    );
    let replay = proxy
        .ok("/key/update", json!({"key_id":id,"models":literal}))
        .await;
    assert_eq!(replay["models"], literal);
    assert_eq!(key_snapshot(&proxy, id).await["models"], LITERAL_MODELS);
    for path in ["/key/block", "/key/unblock"] {
        proxy.ok(path, json!({"key_id":id})).await;
        assert_eq!(key_snapshot(&proxy, id).await["models"], LITERAL_MODELS);
    }
    let moved = proxy
        .ok(
            "/key/update",
            json!({"key_id":id,"team_id":"team-b","project_id":"owned-b"}),
        )
        .await;
    assert_eq!(moved["team_id"], "team-b");
    assert_eq!(moved["project_id"], "owned-b");
    assert_eq!(moved["models"], literal);
    assert_eq!(moved["rpm"], 7);
    assert_eq!(moved["tpm"], 90);
    let detached = proxy
        .ok(
            "/key/update",
            json!({"key_id":id,"project_id":"","team_id":""}),
        )
        .await;
    assert!(detached["project_id"].is_null() && detached["team_id"].is_null());
    for team in [Value::Null, json!("team-a"), json!("team-b")] {
        proxy
            .ok(
                "/key/generate",
                json!({"team_id":team,"project_id":"shared"}),
            )
            .await;
    }

    let response = proxy
        .client
        .post(proxy.url("/key/update"))
        .bearer_auth(created["key"].as_str().unwrap())
        .json(&json!({"key_id":id,"project_id":"owned-a"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401, "management remains master-only");
    assert!(key_snapshot(&proxy, id).await["project_id"].is_null());
    proxy.finish().await;
}

#[tokio::test]
async fn bulk_moves_and_project_reparents_are_atomic_and_refresh_auth_after_commit() {
    let proxy = proxy().await;
    seed(&proxy).await;
    let owned = proxy
        .ok(
            "/key/generate",
            json!({"team_id":"team-a","project_id":"owned-a","user_id":"user"}),
        )
        .await;
    let shared = proxy
        .ok(
            "/key/generate",
            json!({"team_id":"team-a","project_id":"shared","user_id":"user"}),
        )
        .await;
    let ids = [
        owned["key_id"].as_str().unwrap(),
        shared["key_id"].as_str().unwrap(),
    ];
    let snapshots = [
        key_snapshot(&proxy, ids[0]).await,
        key_snapshot(&proxy, ids[1]).await,
    ];
    for key in [&owned, &shared] {
        assert!(!proxy.models(key["key"].as_str().unwrap()).await.is_empty());
    }
    for (path, team) in [
        ("/team/member/add", "team-b"),
        ("/team/member/delete", "team-a"),
    ] {
        proxy
            .reject(
                path,
                json!({"team_id":team,"user_id":"user"}),
                400,
                "belongs to team",
            )
            .await;
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(key_snapshot(&proxy, id).await, snapshots[index]);
        }
    }
    let before_project = project_snapshot(&proxy, "owned-a").await;
    proxy
        .reject(
            "/project/update",
            json!({"project_id":"owned-a","team_id":"team-b","max_budget":0}),
            400,
            "keys outside team",
        )
        .await;
    assert_eq!(project_snapshot(&proxy, "owned-a").await, before_project);

    proxy
        .ok("/key/update", json!({"key_id":ids[0],"project_id":""}))
        .await;
    proxy
        .ok(
            "/team/update",
            json!({"team_id":"team-b","models":["not-configured"]}),
        )
        .await;
    let moved = proxy
        .ok(
            "/team/member/add",
            json!({"team_id":"team-b","user_id":"user"}),
        )
        .await;
    assert_eq!(moved["keys_affected"], 2);
    for (index, key) in [&owned, &shared].iter().enumerate() {
        assert_eq!(key_snapshot(&proxy, ids[index]).await["team_id"], "team-b");
        assert!(
            proxy.models(key["key"].as_str().unwrap()).await.is_empty(),
            "cached team-a must be evicted after the bulk commit"
        );
    }
    proxy
        .ok(
            "/team/member/delete",
            json!({"team_id":"team-b","user_id":"user"}),
        )
        .await;
    for key in [&owned, &shared] {
        assert!(!proxy.models(key["key"].as_str().unwrap()).await.is_empty());
    }
    proxy
        .reject(
            "/project/update",
            json!({"project_id":"shared","team_id":"team-a"}),
            400,
            "keys outside team",
        )
        .await;
    proxy
        .ok(
            "/team/member/add",
            json!({"team_id":"team-a","user_id":"user"}),
        )
        .await;
    proxy
        .ok(
            "/project/update",
            json!({"project_id":"shared","team_id":"team-a"}),
        )
        .await;
    proxy
        .reject(
            "/team/member/delete",
            json!({"team_id":"team-a","user_id":"user"}),
            400,
            "belongs to team",
        )
        .await;
    proxy.ok("/team/delete", json!({"team_id":"team-a"})).await;
    assert!(project_snapshot(&proxy, "shared").await["team_id"].is_null());
    for (index, key) in [&owned, &shared].iter().enumerate() {
        assert!(key_snapshot(&proxy, ids[index]).await["team_id"].is_null());
        assert!(!proxy.models(key["key"].as_str().unwrap()).await.is_empty());
    }
    proxy.finish().await;
}

#[tokio::test]
async fn missing_and_deleted_entities_keep_native_errors_without_partial_writes() {
    let proxy = proxy().await;
    seed(&proxy).await;
    for field in ["team_id", "user_id", "org_id", "project_id"] {
        proxy
            .reject("/key/generate", json!({(field):"absent"}), 400, field)
            .await;
    }
    assert!(proxy.db.list_keys(None).await.unwrap().is_empty());
    for field in ["team_id", "org_id"] {
        proxy
            .reject(
                "/project/add",
                json!({"project_id":"bad-project","project_name":"Bad",(field):"absent"}),
                400,
                "not found",
            )
            .await;
        assert!(proxy.db.get_project("bad-project").await.unwrap().is_none());
    }
    for path in ["/team/member/add", "/team/member/delete"] {
        proxy
            .reject(
                path,
                json!({"team_id":"absent","user_id":"user"}),
                404,
                "team",
            )
            .await;
        proxy
            .reject(
                path,
                json!({"team_id":"team-a","user_id":"absent"}),
                404,
                "user",
            )
            .await;
    }
    proxy
        .reject(
            "/key/update",
            json!({"key_id":"absent","project_id":"absent"}),
            404,
            "key",
        )
        .await;
    proxy
        .reject(
            "/project/update",
            json!({"project_id":"absent","team_id":"team-a"}),
            404,
            "project",
        )
        .await;
    let key = proxy
        .ok(
            "/key/generate",
            json!({"team_id":"team-a","project_id":"owned-a","user_id":"user"}),
        )
        .await;
    let id = key["key_id"].as_str().unwrap();
    proxy
        .ok("/project/delete", json!({"project_id":"owned-a"}))
        .await;
    let updated = proxy
        .ok(
            "/key/update",
            json!({"key_id":id,"key_alias":"after-project-delete"}),
        )
        .await;
    assert!(updated["project_id"].is_null());
    let before = key_snapshot(&proxy, id).await;
    proxy
        .reject(
            "/key/update",
            json!({"key_id":id,"project_id":"owned-a","key_alias":"rejected"}),
            400,
            "project_id",
        )
        .await;
    assert_eq!(key_snapshot(&proxy, id).await, before);
    proxy
        .reject(
            "/project/delete",
            json!({"project_id":"owned-a"}),
            404,
            "project",
        )
        .await;
    proxy.ok("/user/delete", json!({"user_id":"user"})).await;
    let updated = proxy
        .ok(
            "/key/update",
            json!({"key_id":id,"key_alias":"after-user-delete"}),
        )
        .await;
    assert!(updated["user_id"].is_null());
    proxy
        .reject(
            "/team/member/add",
            json!({"team_id":"team-a","user_id":"user"}),
            404,
            "user",
        )
        .await;
    proxy.ok("/team/delete", json!({"team_id":"team-a"})).await;
    let updated = proxy
        .ok(
            "/key/update",
            json!({"key_id":id,"key_alias":"after-team-delete"}),
        )
        .await;
    assert!(updated["team_id"].is_null());
    proxy
        .reject(
            "/key/update",
            json!({"key_id":id,"team_id":"team-a"}),
            400,
            "team_id",
        )
        .await;
    proxy.ok("/key/purge", json!({"key_id":id})).await;
    for path in [
        "/key/update",
        "/key/block",
        "/key/delete",
        "/key/purge",
        "/key/rotate",
    ] {
        proxy.reject(path, json!({"key_id":id}), 404, "key").await;
    }

    let blank = proxy
        .ok(
            "/project/add",
            json!({"project_id":"blank-owner","project_name":"Blank","team_id":"","org_id":""}),
        )
        .await;
    assert_eq!(blank["project_id"], "blank-owner");
    assert!(project_snapshot(&proxy, "blank-owner").await["team_id"].is_null());
    proxy
        .ok("/key/generate", json!({"project_id":"blank-owner"}))
        .await;
    proxy
        .reject(
            "/project/update",
            json!({"project_id":"blank-owner","team_id":"team-b"}),
            400,
            "keys outside team",
        )
        .await;
    proxy.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_http_attachments_reparents_and_deletes_cannot_leave_foreign_links() {
    let proxy = proxy().await;
    seed(&proxy).await;
    for index in 0..8 {
        let project = format!("race-project-{index}");
        proxy
            .ok(
                "/project/add",
                json!({"project_id":project,"project_name":"Shared"}),
            )
            .await;
        let (attached, reparented) = tokio::join!(
            proxy.post(
                "/key/generate",
                json!({"team_id":"team-a","project_id":project})
            ),
            proxy.post(
                "/project/update",
                json!({"project_id":project,"team_id":"team-b"})
            )
        );
        let mut statuses = [attached.0, reparented.0];
        statuses.sort();
        assert_eq!(statuses, [200, 400], "{attached:?} / {reparented:?}");
        let created = if attached.0 == 200 {
            assert!(project_snapshot(&proxy, &project).await["team_id"].is_null());
            attached.1
        } else {
            proxy
                .ok(
                    "/key/generate",
                    json!({"team_id":"team-b","project_id":project}),
                )
                .await
        };
        let id = created["key_id"].as_str().unwrap();
        let (deleted, updated) = tokio::join!(
            proxy.post("/project/delete", json!({"project_id":project})),
            proxy.post(
                "/key/update",
                json!({"key_id":id,"key_alias":"metadata-only"})
            )
        );
        assert_eq!(deleted.0, 200, "{deleted:?}");
        assert_eq!(updated.0, 200, "{updated:?}");
        let current = proxy.db.get_key(id).await.unwrap().unwrap();
        assert!(current.project_id.is_none());
        assert_eq!(current.key_alias.as_deref(), Some("metadata-only"));
    }
    proxy.finish().await;
}

#[tokio::test]
async fn dangling_project_blockers_are_bad_requests_without_deleting_existing_teams() {
    let proxy = proxy().await;
    seed(&proxy).await;
    let created = proxy
        .ok(
            "/key/generate",
            json!({"team_id":"team-a","user_id":"user"}),
        )
        .await;
    let id = created["key_id"].as_str().unwrap();
    let mut imported = proxy.db.get_key(id).await.unwrap().unwrap();
    imported.project_id = Some("absent-project".into());
    proxy.db.update_key(&imported).await.unwrap();
    let before = key_snapshot(&proxy, id).await;
    for (path, request) in [
        ("/team/delete", json!({"team_id":"team-a"})),
        (
            "/team/member/add",
            json!({"team_id":"team-b","user_id":"user"}),
        ),
    ] {
        proxy
            .reject(path, request, 400, "project 'absent-project' not found")
            .await;
        assert!(proxy.db.get_team("team-a").await.unwrap().is_some());
        assert_eq!(
            key_snapshot(&proxy, id).await,
            before,
            "rejection must not partially detach the key"
        );
    }
    proxy
        .reject(
            "/team/delete",
            json!({"team_id":"absent-team"}),
            404,
            "team",
        )
        .await;
    proxy
        .ok("/key/update", json!({"key_id":id,"project_id":""}))
        .await;
    proxy.ok("/team/delete", json!({"team_id":"team-a"})).await;
    assert!(key_snapshot(&proxy, id).await["team_id"].is_null());
    proxy.finish().await;
}
