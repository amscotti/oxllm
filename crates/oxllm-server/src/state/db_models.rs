//! DB-stored model merging (`db` feature only).
//!
//! [`refresh_db_models`] re-reads the `models` table and merges the rows
//! additively on top of the YAML-derived deployments, swapping the result
//! into the shared state atomically. [`db_record_to_deployment`] converts one
//! stored record, decrypting the API key when a [`CredentialCipher`] is
//! configured and resolving `os.environ/` references the same way the YAML
//! loader does.

use std::sync::Arc;

use oxllm_core::Deployment;

use crate::state::SharedState;

/// Build the merged deployment list from a config's YAML `model_list` plus
/// all DB-stored models, skipping DB rows that shadow a YAML deployment or
/// fail to deserialize/resolve. This is the shared core used by both
/// [`refresh_db_models`] (management-API / pub/sub path, CAS-swapped onto
/// the live state) and the config hot-reload path in
/// [`crate::config::watcher::do_reload`] (merged into the new state
/// *before* the atomic swap so there is no transient window where DB-only
/// models are absent).
pub(crate) async fn merge_db_deployments(
    config: &crate::config::ProxyConfig,
    db: &std::sync::Arc<oxllm_db::Database>,
    cipher: Option<&crate::crypto::CredentialCipher>,
) -> anyhow::Result<Vec<Deployment>> {
    let records = db.list_all_models_db().await?;
    let environment = crate::config::loader::resolved_environment();

    let allow_private_api_base = config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);

    let mut deployments: Vec<Deployment> = config
        .model_list
        .iter()
        .map(|cm| cm.clone().into())
        .collect();

    let mut existing_names: std::collections::HashSet<String> =
        deployments.iter().map(|d| d.model_name.clone()).collect();
    let yaml_names: std::collections::HashSet<String> = existing_names.iter().cloned().collect();

    for record in records {
        if crate::config::loader::raw_model_info_supports_environment(
            record.model_info.as_ref(),
            &environment,
        )
        .is_ok_and(|active| !active)
        {
            tracing::info!(model_name = %record.model_name, environment = %environment,
                "skipping inactive DB deployment before credential resolution");
            continue;
        }
        match db_record_to_deployment(&record, allow_private_api_base, cipher) {
            Ok(dep) => {
                if !crate::config::loader::deployment_supports_environment(&dep, &environment) {
                    tracing::info!(model_name = %dep.model_name, environment = %environment,
                        "skipping DB deployment outside its supported environments");
                    continue;
                }
                if existing_names.contains(&dep.model_name) {
                    let source = if yaml_names.contains(&dep.model_name) {
                        "YAML"
                    } else {
                        "an earlier DB"
                    };
                    tracing::warn!(
                        model_id = %record.model_id,
                        model_name = %dep.model_name,
                        "DB model duplicates {source} deployment with the same model_name; skipping"
                    );
                    continue;
                }
                existing_names.insert(dep.model_name.clone());
                deployments.push(dep);
            }
            Err(e) => {
                tracing::warn!(
                    model_id = %record.model_id,
                    error = %e,
                    "skipping DB model that failed to deserialize / resolve"
                );
            }
        }
    }

    Ok(deployments)
}

/// Reload DB-stored models, merge them additively on top of the YAML-derived
/// deployments from the current state, and swap the result in atomically.
///
/// Called after a model add/update/delete and after a config hot-reload (so
/// DB models survive YAML edits). When no database is configured this is a
/// no-op. Individual DB models whose stored JSON fails to deserialize or
/// resolve (e.g. references an unset `os.environ/` var) are skipped with a
/// warning — one bad row must not break the whole proxy.
///
/// Publishing uses `ArcSwap::compare_and_swap` in an RCU retry loop: the new
/// state is installed only if no concurrent writer (a config hot-reload, or
/// another `refresh_db_models`) swapped the live `Arc` between our
/// [`ArcSwap::load_full`] and the publish. A plain [`ArcSwap::store`] here
/// would let a management-API refresh silently revert a just-applied config
/// reload — we would overwrite the fresh config/router/hooks with our clone of
/// the pre-reload state, carrying only the new deployments Vec. On contention
/// the work is redone from the now-current state so DB models are always
/// re-merged on top of the live config.
pub async fn refresh_db_models(shared: &SharedState) -> anyhow::Result<()> {
    loop {
        let current = shared.load_full();
        let Some(db) = current.db.clone() else {
            return Ok(());
        };

        let deployments =
            merge_db_deployments(&current.config, &db, current.credential_cipher.as_ref()).await?;

        // Build the candidate next state BEFORE pruning. The prune is a
        // destructive retain-only sweep over the router runtime stores
        // (`current.router` is the same `Arc<Router>` that `with_deployments`
        // carries into the published state — shared via
        // `Router::rebuild_from_stores` on reload). If we pruned BEFORE the
        // compare-and-swap and the CAS then lost (a concurrent config reload
        // swapped the state between `load_full` and publish), we would have
        // applied a prune computed from the STALE merged deployment list to
        // the SHARED stores — incorrectly dropping cooldown/latency/usage/
        // health/bandit entries for models that exist in the winning config
        // but not in our stale list. The retain-only sweep cannot restore
        // them on retry. Pruning AFTER a winning CAS closes the window: the
        // brief lag between the swap and the prune is harmless — stale entries
        // just consume a little memory; the router never picks a removed
        // deployment because it is absent from the live `deployments` list
        // handed to `pick_deployment` at request time.
        let next = current.with_deployments(deployments);
        let prev = shared.compare_and_swap(&current, Arc::new(next));
        if Arc::ptr_eq(&current, &prev) {
            // We won the compare-and-swap: our update landed and no concurrent
            // writer can have been overwritten by it. NOW sweep the router
            // runtime stores for deployments that are no longer present — e.g.
            // a DB model deleted via the management API. The config hot-reload
            // path does this in `assemble`, but `refresh_db_models` bypasses
            // `assemble`, so without this sweep a deleted DB model leaks its
            // cooldown/latency/usage/health/bandit entries forever (the stores
            // are plain `HashMap`s with no self-eviction). It also orphans the
            // positional cooldown/health/bandit keys (`"{model_name}:{index}"`)
            // when a deletion shifts every later DB model's index, leaving the
            // shifted deployments un-cooled under their new keys. Reload the
            // just-published state to prune against its (our) deployments
            // without cloning the vec before the move into `with_deployments`.
            let live = shared.load_full();
            live.router.prune_stale_store_keys(&live.deployments);
            return Ok(());
        }
        // The live state changed under us between `load_full` and the publish.
        // Retry from the now-current state so the next attempt re-merges DB
        // models on top of whatever config is now live (e.g. a fresh reload).
    }
}

/// Convert a stored [`oxllm_db::ModelConfigRecord`] into a runtime
/// [`Deployment`], resolving `os.environ/` references in the stored JSON the
/// same way the YAML config loader does.
///
/// When a [`crate::crypto::CredentialCipher`] is configured, encrypted secret
/// fields in `oxllm_params` are decrypted first (so a decrypted
/// `os.environ/` reference still resolves afterwards). This symmetrically
/// mirrors [`crate::handlers::management::models_mgmt::encrypt_stored_secrets`],
/// covering every field the redaction denylist catches (`api_key`,
/// `aws_secret_access_key`, `aws_session_token`, `private_key`,
/// `service_account_json`, `password`, custom `authorization` headers, …).
pub(crate) fn db_record_to_deployment(
    record: &oxllm_db::ModelConfigRecord,
    allow_private_api_base: bool,
    cipher: Option<&crate::crypto::CredentialCipher>,
) -> anyhow::Result<Deployment> {
    let mut params_value = record.oxllm_params.clone();
    decrypt_stored_secrets(&mut params_value, cipher, &record.model_id);
    let oxllm_params: oxllm_core::deployment::OxllmParams =
        crate::config::loader::deserialize_env_value(params_value)?;

    if let Some(api_base) = oxllm_params.api_base.as_deref() {
        crate::config::validate_api_base(api_base, allow_private_api_base)?;
    }

    let model_info = match record.model_info.clone() {
        Some(v) => Some(crate::config::loader::deserialize_env_value(v)?),
        None => None,
    };

    Ok(Deployment {
        model_name: record.model_name.clone(),
        oxllm_params,
        model_info,
    })
}

/// Decrypt every encrypted secret credential embedded in `oxllm_params` in
/// place when a cipher is configured. This is the symmetric counterpart of
/// [`crate::handlers::management::models_mgmt::encrypt_stored_secrets`]: it
/// walks the same denylist of secret key names (via the shared
/// [`should_redact`](crate::handlers::management::should_redact) predicate)
/// plus every string inside a secret container (see
/// [`is_secret_container`](crate::handlers::management::is_secret_container))
/// and attempts to decrypt each blob-like string value.
///
/// A decryption failure is non-fatal: the stored value is kept as-is (treated
/// as plaintext) so a rotated or lost `ENCRYPTION_KEY` cannot break model
/// loading, and rows written before encryption was enabled keep working. The
/// failure is warned with `model_id` so an operator seeing intermittent
/// provider-side 401s can correlate them to the DB row with the undecryptable
/// secret (the upstream receives the raw ciphertext blob, not a real value).
/// The warning re-fires on every `refresh_db_models` cycle, so a key-rotation
/// mismatch surfaces without a process restart.
fn decrypt_stored_secrets(
    params: &mut serde_json::Value,
    cipher: Option<&crate::crypto::CredentialCipher>,
    model_id: &str,
) {
    let Some(cipher) = cipher else { return };
    decrypt_secrets_inner(params, cipher, model_id, 0, None, false);
}

fn decrypt_secrets_inner(
    value: &mut serde_json::Value,
    cipher: &crate::crypto::CredentialCipher,
    model_id: &str,
    depth: u32,
    parent: Option<&str>,
    secret_container: bool,
) {
    if depth >= crate::handlers::management::REDACT_MAX_DEPTH {
        return;
    }
    use crate::handlers::management::{is_secret_container, should_redact};
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                let secret =
                    secret_container || is_secret_container(k, parent) || should_redact(k, v);
                if secret {
                    if let serde_json::Value::String(stored) = v {
                        match crate::crypto::decrypt_api_key(Some(cipher), stored) {
                            Ok(plain) => {
                                *stored = plain;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    model_id = %model_id,
                                    key = %k,
                                    error = %e,
                                    "failed to decrypt stored oxllm_params secret field; \
                                     the raw ciphertext will be sent upstream and provider calls \
                                     for this model will likely fail. Treating the stored value \
                                     as plaintext (check ENCRYPTION_KEY / key rotation)."
                                );
                            }
                        }
                    } else {
                        decrypt_secrets_inner(v, cipher, model_id, depth + 1, Some(k), true);
                    }
                } else {
                    decrypt_secrets_inner(v, cipher, model_id, depth + 1, Some(k), false);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                decrypt_secrets_inner(item, cipher, model_id, depth + 1, parent, secret_container);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decrypt_stored_secrets_round_trip() {
        let cipher = crate::crypto::CredentialCipher::from_passphrase("state-round-trip");
        let blob = crate::crypto::encrypt_api_key(Some(&cipher), "sk-db-key").expect("encrypt");
        let mut params = json!({"model": "openai/gpt-4o", "api_key": blob});
        decrypt_stored_secrets(&mut params, Some(&cipher), "model-1");
        assert_eq!(params["api_key"], "sk-db-key");
    }

    /// C06 MEDIUM: secret-container symmetry. The write path encrypts every
    /// string inside `extra_headers` (and `headers` under `extra_body`); the
    /// load path must decrypt them back so provider calls see real header
    /// values, while non-secret siblings stay untouched.
    #[test]
    fn decrypt_stored_secrets_round_trips_secret_containers() {
        let cipher = crate::crypto::CredentialCipher::from_passphrase("state-containers");
        let enc = |v: &str| crate::crypto::encrypt_api_key(Some(&cipher), v).expect("encrypt");
        let mut params = json!({
            "model": "openai/gpt-4o",
            "extra_headers": {
                "X-Backend-Key": enc("hunter2"),
                "X-Benign": enc("header-value")
            },
            "extra_body": {
                "headers": { "X-Signature": enc("sig-secret") },
                "conversation_id": "conv-keep"
            }
        });
        decrypt_stored_secrets(&mut params, Some(&cipher), "model-containers");
        assert_eq!(params["extra_headers"]["X-Backend-Key"], "hunter2");
        assert_eq!(params["extra_headers"]["X-Benign"], "header-value");
        assert_eq!(params["extra_body"]["headers"]["X-Signature"], "sig-secret");
        assert_eq!(params["extra_body"]["conversation_id"], "conv-keep");
        // Plaintext container values (pre-encryption rows) pass through.
        let mut legacy = json!({
            "model": "openai/gpt-4o",
            "extra_headers": { "X-Plain": "plain-value" }
        });
        decrypt_stored_secrets(&mut legacy, Some(&cipher), "model-legacy");
        assert_eq!(legacy["extra_headers"]["X-Plain"], "plain-value");
    }

    #[test]
    fn decrypt_stored_secrets_passthrough_cases() {
        // No cipher → value untouched.
        let mut params = json!({"api_key": "sk-plain"});
        decrypt_stored_secrets(&mut params, None, "model-1");
        assert_eq!(params["api_key"], "sk-plain");

        // No api_key field → untouched.
        let cipher = crate::crypto::CredentialCipher::from_passphrase("state-passthrough");
        let mut params = json!({"model": "openai/gpt-4o"});
        decrypt_stored_secrets(&mut params, Some(&cipher), "model-2");
        assert!(params.get("api_key").is_none());

        // Plaintext value (pre-encryption row) → untouched.
        let mut params = json!({"api_key": "sk-legacy-plaintext"});
        decrypt_stored_secrets(&mut params, Some(&cipher), "model-3");
        assert_eq!(params["api_key"], "sk-legacy-plaintext");

        // Blob encrypted under a DIFFERENT key → kept as-is (non-fatal). The
        // warning now carries the model_id so operators can correlate the
        // resulting provider-side 401s to the row; the stored value is still
        // the raw ciphertext blob, not plaintext.
        let other = crate::crypto::CredentialCipher::from_passphrase("state-other-key");
        let blob = crate::crypto::encrypt_api_key(Some(&other), "sk-x").expect("encrypt");
        let mut params = json!({"api_key": blob.clone()});
        decrypt_stored_secrets(&mut params, Some(&cipher), "model-4");
        assert_eq!(params["api_key"], blob);
    }

    /// C30 MEDIUM: at-rest encryption now covers every secret field, not just
    /// `api_key`. This round-trips a Bedrock-style record with
    /// `aws_secret_access_key` / `aws_session_token` / `private_key` /
    /// `authorization` / `service_account_json` to assert all of them encrypt
    /// on the write path and decrypt back on the load path.
    #[test]
    fn decrypt_stored_secrets_round_trips_all_secret_fields() {
        let cipher = crate::crypto::CredentialCipher::from_passphrase("state-all-secrets");

        // Simulate what encrypt_stored_secrets produced: every secret field
        // encrypted under the cipher, non-secret fields left plaintext.
        let enc_aws_secret =
            crate::crypto::encrypt_api_key(Some(&cipher), "AWSSECRET123").expect("encrypt");
        let enc_aws_session =
            crate::crypto::encrypt_api_key(Some(&cipher), "AQoSESSION").expect("encrypt");
        let enc_private_key =
            crate::crypto::encrypt_api_key(Some(&cipher), "-----BEGIN PRIVATE KEY-----")
                .expect("encrypt");
        let enc_authorization =
            crate::crypto::encrypt_api_key(Some(&cipher), "Bearer secret-token").expect("encrypt");
        let enc_service_account =
            crate::crypto::encrypt_api_key(Some(&cipher), "{\"private_key\":\"pk\"}")
                .expect("encrypt");

        let mut params = json!({
            "model": "bedrock/anthropic.claude-3",
            "api_key": crate::crypto::encrypt_api_key(Some(&cipher), "sk-openai").expect("encrypt"),
            "aws_secret_access_key": enc_aws_secret,
            "aws_session_token": enc_aws_session,
            "private_key": enc_private_key,
            "authorization": enc_authorization,
            "service_account_json": enc_service_account,
            "headers": {
                "Authorization": enc_authorization,
                "Content-Type": "application/json"
            },
            "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "weight": 2
        });

        decrypt_stored_secrets(&mut params, Some(&cipher), "model-all");

        assert_eq!(params["api_key"], "sk-openai");
        assert_eq!(params["aws_secret_access_key"], "AWSSECRET123");
        assert_eq!(params["aws_session_token"], "AQoSESSION");
        assert_eq!(params["private_key"], "-----BEGIN PRIVATE KEY-----");
        assert_eq!(params["authorization"], "Bearer secret-token");
        assert_eq!(params["service_account_json"], "{\"private_key\":\"pk\"}");
        // Nested header Authorization also decrypted.
        assert_eq!(params["headers"]["Authorization"], "Bearer secret-token");
        // Non-secret fields survive untouched.
        assert_eq!(params["model"], "bedrock/anthropic.claude-3");
        assert_eq!(
            params["api_base"],
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        );
        assert_eq!(params["weight"], 2);
        assert_eq!(params["headers"]["Content-Type"], "application/json");
    }

    /// LOW: two DB rows with the same `model_name` — the second must be
    /// skipped (first-wins), and the result must contain exactly one
    /// deployment for that name.
    #[tokio::test]
    async fn merge_db_deployments_skips_duplicate_db_model_name() {
        use crate::config::types::{GeneralSettings, ProxyConfig};

        let db = std::sync::Arc::new(
            oxllm_db::Database::connect("sqlite::memory:")
                .await
                .expect("connect"),
        );
        // Seed the pre-claim schema: legacy duplicates must survive an upgrade,
        // while current management writes are atomically prevented from adding them.
        sqlx::query("CREATE TABLE models (model_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, \
            oxllm_params TEXT NOT NULL, model_info TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
            .execute(db.pool()).await.expect("legacy models table");

        let record_a = oxllm_db::types::ModelConfigRecord {
            model_id: "db-a".to_string(),
            model_name: "dup-model".to_string(),
            oxllm_params: json!({"model": "openai/gpt-4o", "api_key": "sk-a"}),
            model_info: None,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
        };
        let record_b = oxllm_db::types::ModelConfigRecord {
            model_id: "db-b".to_string(),
            model_name: "dup-model".to_string(),
            oxllm_params: json!({"model": "openai/gpt-4o", "api_key": "sk-b"}),
            model_info: None,
            created_at: "2025-01-02T00:00:00Z".to_string(),
            updated_at: "2025-01-02T00:00:00Z".to_string(),
        };
        db.create_model(&record_a).await.expect("insert a");
        db.create_model(&record_b).await.expect("insert b");
        db.migrate().await.expect("migrate legacy models");

        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: crate::config::types::RouterSettings::default(),
            extra: serde_json::Map::new(),
        };

        let deployments = merge_db_deployments(&config, &db, None)
            .await
            .expect("merge");

        let dups: Vec<_> = deployments
            .iter()
            .filter(|d| d.model_name == "dup-model")
            .collect();
        assert_eq!(
            dups.len(),
            1,
            "duplicate DB model_name must be skipped (first-wins)"
        );
    }

    /// LOW: a DB row whose `model_name` collides with a YAML deployment must
    /// also be skipped (the YAML deployment wins).
    #[tokio::test]
    async fn merge_db_deployments_db_row_shadowing_yaml_skipped() {
        use crate::config::types::{
            ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings,
        };
        use oxllm_core::deployment::OxllmParams;

        let db = std::sync::Arc::new(
            oxllm_db::Database::connect("sqlite::memory:")
                .await
                .expect("connect"),
        );
        db.migrate().await.expect("migrate");
        let record = oxllm_db::types::ModelConfigRecord {
            model_id: "db-1".to_string(),
            model_name: "yaml-model".to_string(),
            oxllm_params: json!({"model": "openai/gpt-4o", "api_key": "sk-db"}),
            model_info: None,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
        };
        db.create_model(&record).await.expect("insert");

        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "yaml-model".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("sk-yaml".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };

        let deployments = merge_db_deployments(&config, &db, None)
            .await
            .expect("merge");
        assert_eq!(
            deployments.len(),
            1,
            "DB row shadowing a YAML model must be skipped"
        );
    }

    /// MEDIUM: `list_models_db` (the API-facing variant) caps at 1 000 rows;
    /// when the `models` table holds more than 1 000 rows, the default `None`
    /// limit silently truncates the oldest DB-stored models. The internal
    /// merge path (`merge_db_deployments`) must use the uncapped
    /// `list_all_models_db` so every routable model is merged — otherwise
    /// models beyond the 1 000-row cap become silently unroutable (requests
    /// return "model not found" with no warning anywhere). This inserts 1 002
    /// rows and asserts all 1 002 appear in the merged result.
    #[tokio::test]
    async fn merge_db_deployments_does_not_truncate_beyond_1000_rows() {
        use crate::config::types::{GeneralSettings, ProxyConfig};

        let db = std::sync::Arc::new(
            oxllm_db::Database::connect("sqlite::memory:")
                .await
                .expect("connect"),
        );
        db.migrate().await.expect("migrate");

        const N: usize = 1002;
        for i in 0..N {
            let record = oxllm_db::types::ModelConfigRecord {
                model_id: format!("db-{i}"),
                model_name: format!("m-{i}"),
                oxllm_params: json!({"model": "openai/gpt-4o", "api_key": "sk"}),
                model_info: None,
                created_at: format!("2025-01-01T00:00:{i:02}Z"),
                updated_at: format!("2025-01-01T00:00:{i:02}Z"),
            };
            db.create_model(&record).await.expect("insert");
        }

        // Sanity: the capped API variant truncates at 1 000 rows.
        let capped = db.list_models_db(None).await.expect("list");
        assert_eq!(
            capped.len(),
            1000,
            "list_models_db(None) must cap at 1000 rows (the bug condition)"
        );
        // The uncapped variant sees every row.
        let uncapped = db.list_all_models_db().await.expect("list all");
        assert_eq!(uncapped.len(), N, "list_all_models_db must be uncapped");

        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: crate::config::types::RouterSettings::default(),
            extra: serde_json::Map::new(),
        };

        let deployments = merge_db_deployments(&config, &db, None)
            .await
            .expect("merge");
        assert_eq!(
            deployments.len(),
            N,
            "merge_db_deployments must include every DB row, not truncate at 1000"
        );
    }

    /// LOW: `refresh_db_models` must prune the router runtime stores only
    /// AFTER a successful compare-and-swap, not before. When the CAS loses to
    /// a concurrent config reload that added a NEW YAML model, a pre-CAS prune
    /// using the stale merged list would incorrectly drop the new model's
    /// previously-accumulated cooldown entry from the shared stores (the prune
    /// is retain-only and cannot restore it on retry).
    ///
    /// Deterministically forcing a CAS loss requires a concurrent writer that
    /// swaps the state between `load_full` and `compare_and_swap` — not
    /// feasible in a unit test. Instead this test asserts the post-CAS prune
    /// invariant directly: after a winning `refresh_db_models` with a DB
    /// wired, the router stores are pruned against the NEW merged deployments
    /// (stale entries removed, live entries retained). This is the property
    /// the post-CAS placement preserves; the pre-CAS placement violated it
    /// under contention because the prune ran with a stale list against
    /// shared stores even when the CAS lost.
    #[tokio::test]
    async fn refresh_db_models_prunes_stale_entries_after_winning_cas() {
        use crate::config::types::{
            ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings,
        };
        use crate::state::AppState;
        use oxllm_core::deployment::OxllmParams;

        // `Database::connect` already returns `Arc<Database>` — do NOT
        // double-wrap, we need to assign it to `AppState::db: Option<Arc<Database>>`.
        let db: std::sync::Arc<oxllm_db::Database> = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");

        // Seed the DB with one model "db-keep" that will be in the merged list.
        let db_record = oxllm_db::types::ModelConfigRecord {
            model_id: "db-1".to_string(),
            model_name: "db-keep".to_string(),
            oxllm_params: json!({"model": "openai/gpt-4o", "api_key": "sk"}),
            model_info: None,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
        };
        db.create_model(&db_record).await.expect("insert");

        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "yaml-keep".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("sk".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };

        let state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("build state");

        // Seed cooldown failure-count entries for three models:
        //  - "yaml-keep": in the merged list → must survive.
        //  - "db-keep": in the merged list → must survive.
        //  - "stale-gone": NOT in the merged list → must be pruned.
        let yaml_dep = state
            .deployments
            .iter()
            .find(|d| d.model_name == "yaml-keep")
            .expect("yaml-keep present")
            .clone();
        let yaml_key = state.router.cooldown_key(&yaml_dep);
        // An arbitrary key that no real deployment maps to (cooldown keys are
        // hashes of model_name + serialized params, so a literal string never
        // collides). It stands in for a retired model's leftover store entry.
        let stale_key = "stale-gone-key";
        let cooldown = state.router.cooldown_arc();
        cooldown.record_failure(&yaml_key, 500, std::time::Duration::from_secs(60));
        cooldown.record_failure(stale_key, 500, std::time::Duration::from_secs(60));
        assert!(cooldown.failure_count(&yaml_key, 500) > 0, "yaml seeded");
        assert!(cooldown.failure_count(stale_key, 500) > 0, "stale seeded");

        // Publish the state, wire the DB, then run refresh_db_models.
        let shared: SharedState = std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state));
        // The state builder already wired the DB when `db` feature is on; but
        // AppState::with_registry doesn't inject a DB, so install one on the
        // published state by rebuilding with the DB attached via a CAS-friendly
        // path: clone, set db, store.
        {
            let mut seeded = (*shared.load_full()).clone();
            seeded.db = Some(db);
            shared.store(std::sync::Arc::new(seeded));
        }

        refresh_db_models(&shared).await.expect("refresh");

        // After refresh, the DB model "db-keep" is merged in. Prune should
        // keep entries for yaml-keep and db-keep, drop stale-gone.
        let after = shared.load_full();
        let after_cd = after.router.cooldown_arc();
        assert!(
            after_cd.failure_count(&yaml_key, 500) > 0,
            "yaml-keep cooldown must survive (it is in the merged list)"
        );
        assert!(
            after_cd.failure_count(stale_key, 500) == 0,
            "stale-gone cooldown must be pruned (it is not in the merged list)"
        );
        let db_dep = after
            .deployments
            .iter()
            .find(|d| d.model_name == "db-keep")
            .expect("db-keep merged");
        let db_key = after.router.cooldown_key(db_dep);
        assert!(
            after.deployments.iter().any(|d| d.model_name == "db-keep"),
            "db-keep must be present in merged deployments"
        );
        let _ = db_key;
    }
}
