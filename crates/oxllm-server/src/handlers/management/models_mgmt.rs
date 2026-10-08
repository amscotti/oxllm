//! Model management endpoints.

use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use oxllm_core::Deployment;
use serde::Deserialize;
use serde_json::json;

use crate::auth::AuthUser;
use crate::crypto::CredentialCipher;
use crate::error::ApiError;
use crate::state::SharedState;

/// Sentinel cache-invalidation key used to signal cross-replica model-list
/// refresh. Published on the existing `oxllm:invalidate` pub/sub channel
/// after management-API model mutations; peer replicas' subscriber tasks
/// detect this key and call `refresh_db_models` to pick up the change
/// without a restart.
const MODEL_LIST_INVALIDATE_KEY: &str = "__model_list__";

/// Broadcast the model-list change to peer replicas via the Redis pub/sub
/// channel **and** refresh this replica's local deployment list.
///
/// - The pub/sub publish (`__model_list__`) is attempted **first** and
///   unconditionally. A transient local-refresh failure must not strand a
///   committed DB row on every other replica: peers re-reading the shared DB
///   is what makes them correct, so notifying them comes first. The publish
///   itself is best-effort at the cache layer (Redis may be down or
///   unconfigured; the failure is logged inside `publish_invalidation`).
/// - The local [`crate::state::refresh_db_models`] runs second; its error is
///   returned to the caller so a 500 surfaces the local degradation, while
///   peers have already been notified.
///
/// Without this ordering, a `list_models_db` blip right after a successful
/// `db.create_model` would leave a durable row that no replica loads until
/// restart — and since there is no fallback DB poll on the peer side in some
/// configurations, the row would stay invisible fleet-wide.
async fn refresh_and_propagate(shared: &SharedState) -> Result<(), ApiError> {
    shared
        .load()
        .cache
        .publish_invalidation(MODEL_LIST_INVALIDATE_KEY)
        .await;
    crate::state::refresh_db_models(shared)
        .await
        .map_err(|e| ApiError::internal(format!("failed to refresh models: {e}")))?;
    Ok(())
}

use super::{
    db_err, load_db, random_id, redact_secrets, require_master, should_redact, truncate_scope_list,
};

#[derive(Debug, Deserialize)]
pub struct ModelInfoQuery {
    pub model: String,
}

/// `GET /model/list` — list configured models with provider metadata.
///
/// Merges YAML deployments (static) with DB-stored deployments (dynamic) via
/// [`crate::state::refresh_db_models`] is a no-op here — the merged list is
/// already live in `AppState.deployments` because every write endpoint calls
/// `refresh_db_models` after mutating the DB.
///
/// One row is emitted per distinct `model_name`. The singular `provider`
/// field reports the first deployment's provider (kept for backward
/// compatibility), and `providers` lists every provider backing the group
/// (deduped, sorted) so a failover topology such as `gpt-4o` served by both
/// OpenAI and Azure is fully visible. Previously the group was deduped to its
/// first deployment and the single `provider` was whichever entry happened to
/// be first — non-deterministic w.r.t. config edits and misleading for an
/// operator auditing which providers serve a group.
///
/// Rows backed by a `models`-table record additionally carry `model_id` (the
/// DB primary key, e.g. `model-a1b2…`), because `POST /model/delete` and
/// `POST /model/update` address deployments by that id — never by the
/// client-facing `model_name`. YAML-configured rows carry no `model_id`, so
/// clients can tell deletable (DB-backed) models from config-managed ones.
///
/// The response is capped at [`super::SCOPE_LIST_LIMIT`] rows and carries a
/// top-level `truncated` boolean (the admin table reads it to warn that the
/// list is incomplete), matching the scope list endpoints.
pub async fn list_models(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let state = shared.load_full();
    let mut data = model_list_rows(&state.deployments);
    // Cap the response at the shared list limit and report truncation, matching
    // the scope list endpoints. The admin UI reads the top-level `truncated`
    // flag to warn that the table (and its per-model enrichment) is incomplete
    // beyond the cap.
    let truncated = truncate_scope_list(&mut data);
    // Attach the DB row id for DB-backed model names. A failure to read the
    // models table must not fail the whole list (the deployment list is
    // already live without it); the rows simply render without `model_id`,
    // and mutation endpoints remain reachable by clients that know the id.
    if let Some(db) = state.db.clone() {
        match db.list_models_db(None).await {
            Ok(records) => attach_db_model_ids(&mut data, &records),
            Err(e) => {
                tracing::warn!(error = %e, "failed to read models table for /model/list ids");
            }
        }
    }
    Ok(Json(json!({ "object": "list", "data": data, "truncated": truncated })).into_response())
}

/// Insert `model_id` (the `models` table primary key) into every list row
/// whose `id` (the `model_name`) matches a DB record. Rows without a DB
/// record (YAML-configured deployments) keep no `model_id`. Duplicate
/// `model_name`s in the table resolve to the newest record
/// (`list_models_db` orders `created_at DESC`, so the first seen wins).
fn attach_db_model_ids(data: &mut [serde_json::Value], records: &[oxllm_db::ModelConfigRecord]) {
    let mut by_name: HashMap<&str, &str> = HashMap::new();
    for record in records {
        by_name
            .entry(record.model_name.as_str())
            .or_insert(record.model_id.as_str());
    }
    for row in data.iter_mut() {
        let Some(name) = row.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        if let Some(model_id) = by_name.get(name) {
            row.as_object_mut()
                .expect("model_list_rows emits objects")
                .insert("model_id".to_string(), json!(model_id));
        }
    }
}

/// Build the `data` array for [`list_models`] from a deployment slice.
/// Extracted so the dedup/aggregation logic is unit-testable without a full
/// `AppState`. See [`list_models`] for the `provider` / `providers` contract.
fn model_list_rows(deployments: &[Deployment]) -> Vec<serde_json::Value> {
    // Preserve first-seen order of model names (YAML order, then DB order).
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    for d in deployments.iter().filter(|d| d.model_name != "*") {
        let provider = d.provider_slug().to_string();
        let entry = groups.entry(d.model_name.clone()).or_default();
        if !entry.contains(&provider) {
            entry.push(provider);
        }
        if !order.contains(&d.model_name) {
            order.push(d.model_name.clone());
        }
    }

    order
        .into_iter()
        .map(|model_name| {
            let mut providers = groups.remove(&model_name).unwrap_or_default();
            providers.sort();
            let first = providers.first().cloned().unwrap_or_default();
            json!({
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "oxllm",
                "provider": first,
                "providers": providers,
            })
        })
        .collect()
}

/// `GET /model/info?model=...` — get pricing/limits info for a model.
///
/// When a `model_name` is backed by multiple deployments (a failover group),
/// the **first** matching deployment's pricing is reported. This is
/// intentional and pinned by tests: `GET /model/list` surfaces every provider
/// backing a group, while this endpoint reports a single representative
/// pricing snapshot. Operators that need the full per-deployment topology
/// should use `GET /config`. The reference parity call (per-deployment pricing
/// array) is left to a future change.
pub async fn model_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<ModelInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let state = shared.load_full();
    let body = model_info_body(&state.deployments, &q.model)
        .ok_or_else(|| ApiError::model_not_found(&q.model))?;
    Ok(Json(body).into_response())
}

/// Build the `GET /model/info` response body for `model_name` from the
/// deployment list. Extracted (mirroring [`model_list_rows`]) so the
/// first-match selection is unit-testable without a full `AppState`. Returns
/// `None` when no deployment matches. See [`model_info`] for the
/// first-deployment contract on multi-deployment groups.
fn model_info_body(deployments: &[Deployment], model_name: &str) -> Option<serde_json::Value> {
    let deployment = deployments.iter().find(|d| d.model_name == model_name)?;
    let pricing = oxllm_core::lookup_pricing(&deployment.oxllm_params.model);
    Some(match pricing {
        Some(p) => json!({
            "model": deployment.model_name,
            "provider": deployment.provider_slug(),
            "provider_model": deployment.oxllm_params.model,
            "input_cost_per_token": p.input_cost_per_token,
            "output_cost_per_token": p.output_cost_per_token,
            "max_input_tokens": p.max_input_tokens,
            "max_output_tokens": p.max_output_tokens,
            "price_provider": p.provider,
        }),
        None => json!({
            "model": deployment.model_name,
            "provider": deployment.provider_slug(),
            "provider_model": deployment.oxllm_params.model,
            "pricing": "unknown",
        }),
    })
}

/// Request body for `POST /model/new`.
#[derive(Debug, Deserialize)]
pub struct AddModelRequest {
    pub model_name: String,
    #[serde(alias = "litellm_params")]
    pub oxllm_params: serde_json::Value,
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
}

/// `POST /model/new` — add a model deployment to the DB.
///
/// Stores the oxllm_params/model_info JSON in the DB. `os.environ/`
/// references are resolved at load time (when the merged deployment list is
/// rebuilt), not at write time. When `ENCRYPTION_KEY` is set, every secret
/// field in `oxllm_params` (not just `api_key`) is encrypted (AES-256-GCM)
/// before storage and decrypted again at load time — see
/// [`encrypt_stored_secrets`].
///
/// **One deployment per `model_name` via this API (intentional).** Unlike
/// the reference, which lets operators add multiple deployments under a shared
/// `model_name` via `POST /model/new` to build load-balanced / failover
/// groups, oxllm rejects a request whose `model_name` already exists in the
/// live deployment list (YAML or DB) with a 400. Multi-deployment routing
/// groups are configured statically via the YAML `model_list`. This keeps the
/// dynamic API to an atomically enforced one-row-per-name invariant; relaxing it
/// would require revisiting
/// `merge_db_deployments`' first-wins policy.
pub async fn add_model(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<AddModelRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_model_entry(&req.model_name, &req.oxllm_params)?;
    let (state, db) = load_db(&shared, &auth).await?;

    // A DB row whose model_name collides with an existing deployment is
    // silently dropped by `refresh_db_models` (it skips rows that shadow a
    // YAML deployment or a previously-loaded DB row). Detect the collision up
    // front so the operator gets a 400 instead of a durable-but-unusable row.
    let name_taken = state
        .deployments
        .iter()
        .any(|d| d.model_name == req.model_name);
    if name_taken {
        return Err(ApiError::bad_request(format!(
            "a deployment for model_name '{}' already exists; a DB row shadowing \
             an existing deployment is never loaded by the router",
            req.model_name
        )));
    }

    let mut oxllm_params = req.oxllm_params;
    encrypt_stored_secrets(
        &mut oxllm_params,
        state.credential_cipher.as_ref(),
        UndecryptableBlob::EncryptAnyway,
    )?;

    let model_id = format!("model-{}", random_id());
    let now = oxllm_db::now_iso();
    let record = oxllm_db::ModelConfigRecord {
        model_id: model_id.clone(),
        model_name: req.model_name,
        oxllm_params,
        model_info: req.model_info,
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_model(&record).await.map_err(model_write_error)?;

    // Run the cross-replica publish + local refresh, capturing any error so
    // the rollback checks below still run (C65 LOW: a refresh failure must
    // not strand a committed-but-unloadable row — that is exactly the case
    // the rollback exists for). The publish inside `refresh_and_propagate`
    // fires unconditionally before the local refresh, so peers are notified
    // even when the local refresh errors.
    let refresh_err = refresh_and_propagate(&shared).await.err();
    // If the local refresh failed (transient DB read), retry once before the
    // landed check so a transient error does not skip the rollback of an
    // unloadable row. A second failure leaves the row durable; a loadable row
    // self-heals via the 30s fallback poll on this and peer replicas.
    let refresh_err = if refresh_err.is_some() {
        crate::state::refresh_db_models(&shared)
            .await
            .map_err(|e| ApiError::internal(format!("failed to refresh models: {e}")))
            .err()
    } else {
        refresh_err
    };

    // `refresh_db_models` skips individual bad rows with only a `tracing::warn!`
    // — rows whose `api_base` fails the SSRF policy (private CIDR without
    // `allow_private_api_base`), rows with an undecryptable `api_key` after a
    // key rotation, rows whose stored JSON fails to deserialize. The row is
    // durable in the DB but invisible to the router, so the operator would get
    // a 200 here and only discover the model 404s on first use. Re-read the
    // live deployments and verify the new model actually landed; if not, roll
    // back the row and surface a clear error. Only trust this check when the
    // refresh succeeded — a refresh failure leaves the live list stale.
    if refresh_err.is_none() {
        let after = shared.load_full();
        let expected = crate::state::db_record_to_deployment(
            &record,
            after
                .config
                .general_settings
                .allow_private_api_base
                .unwrap_or(false),
            after.credential_cipher.as_ref(),
        )
        .ok();
        let inactive = crate::config::loader::raw_model_info_supports_environment(
            record.model_info.as_ref(),
            &crate::config::loader::resolved_environment(),
        )
        .is_ok_and(|active| !active);
        let landed = !after
            .config
            .model_list
            .iter()
            .any(|model| model.model_name == record.model_name)
            && (inactive
                || expected.is_some_and(|expected| {
                    !crate::config::loader::deployment_supports_environment(
                        &expected,
                        &crate::config::loader::resolved_environment(),
                    ) || after.deployments.iter().any(|deployment| {
                        deployment.model_name == record.model_name
                            && after.router.cooldown_key(deployment)
                                == after.router.cooldown_key(&expected)
                    })
                }));
        if !landed {
            let _ = db.delete_model(&model_id).await;
            return Err(ApiError::bad_request(format!(
                "model '{}' was stored but dropped from the live deployment list; \
                 common causes: api_base rejected by SSRF policy, undecryptable \
                 api_key, or malformed oxllm_params. The stored row has been rolled \
                 back.",
                record.model_name
            )));
        }
    }

    if let Some(e) = refresh_err {
        return Err(e);
    }

    tracing::info!(model_id = %model_id, "model added via management API");
    Ok(Json(json!({ "model_id": model_id, "status": "created" })).into_response())
}

/// Request body for `POST /model/update`.
#[derive(Debug, Deserialize)]
pub struct UpdateModelRequest {
    pub model_id: String,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    #[serde(alias = "litellm_params")]
    pub oxllm_params: Option<serde_json::Value>,
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
    /// When true, clear the stored `model_info`.
    #[serde(default)]
    pub clear_model_info: bool,
}

/// `POST /model/update` — update an existing DB-stored model.
///
/// YAML-derived models cannot be edited here (they are reloaded from the
/// config file). Only `model_id`s that live in the `models` table are
/// mutable; updating a YAML-only id returns 404.
pub async fn update_model(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateModelRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_model(&req.model_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("model '{}' not found", req.model_id)))?;

    if let Some(name) = req.model_name {
        validate_model_name(&name)?;
        record.model_name = name;
    }
    if let Some(new_params) = req.oxllm_params {
        // Deep-merge the new params into the stored ones so a partial update
        // (e.g. tweaking `weight` or `rpm`) does not silently drop previously
        // stored secrets (`api_key`, `aws_secret_access_key`, custom headers,
        // …). A caller that wants to clear a field sends an explicit `null`
        // for it. This mirrors the reference `update_config`, which merges
        // per-section rather than replacing the whole block. Validation runs
        // on the merged result (not the raw patch) so a partial patch that
        // omits `model` is accepted as long as the stored record carries one.
        let mut merged = record.oxllm_params.clone();
        deep_merge_json(&mut merged, new_params);
        validate_model_params(&merged)?;
        // SkipAndWarn: the merged params may carry legacy blobs loaded from
        // the stored row; a blob-like value no ring key decrypts must not be
        // re-wrapped into a double-encrypted dead credential.
        encrypt_stored_secrets(
            &mut merged,
            state.credential_cipher.as_ref(),
            UndecryptableBlob::SkipAndWarn,
        )?;
        record.oxllm_params = merged;
    }
    if req.clear_model_info {
        record.model_info = None;
    } else if let Some(info) = req.model_info {
        record.model_info = Some(info);
    }
    record.updated_at = oxllm_db::now_iso();

    crate::state::db_record_to_deployment(
        &record,
        state
            .config
            .general_settings
            .allow_private_api_base
            .unwrap_or(false),
        state.credential_cipher.as_ref(),
    )
    .map_err(|_| {
        ApiError::bad_request(
            "model configuration cannot be loaded; verify parameter types and endpoint policy",
        )
    })?;
    let collision = state
        .config
        .model_list
        .iter()
        .any(|model| model.model_name == record.model_name)
        || db
            .list_all_models_db()
            .await
            .map_err(db_err)?
            .iter()
            .any(|model| model.model_id != req.model_id && model.model_name == record.model_name);
    if collision {
        return Err(ApiError::bad_request(
            "model_name already belongs to another configured deployment",
        ));
    }

    db.update_model(&req.model_id, &record)
        .await
        .map_err(model_write_error)?;
    // Retry the refresh once on a transient local-read failure (mirroring
    // `add_model` / `delete_model`). The DB mutation is already durable; a
    // transient blip should not surface a 500 when the row is committed and
    // the in-memory list will self-heal on the next fallback poll / peer
    // publish. Peers were already notified (publish fires before local
    // refresh inside `refresh_and_propagate`).
    if refresh_and_propagate(&shared).await.is_err() {
        crate::state::refresh_db_models(&shared)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "failed to refresh models");
                ApiError::internal("failed to refresh models")
            })?;
    }

    tracing::info!(model_id = %req.model_id, "model updated via management API");
    // Redact secrets before serializing the stored record back to the caller.
    // At-rest encryption (when `ENCRYPTION_KEY` is set) covers every secret
    // field in `oxllm_params` via `encrypt_stored_secrets`; this redaction
    // pass is a defense-in-depth that keeps the response safe even when
    // encryption is disabled or a row predates encryption. This is the only
    // endpoint that emits DB-stored `oxllm_params`; `add_model` returns
    // `{model_id, status}` and the read paths already redact via the same
    // shared helper.
    let mut response = serde_json::to_value(&record)
        .map_err(|e| ApiError::internal(format!("failed to serialize model record: {e}")))?;
    redact_secrets(&mut response);
    Ok(Json(response).into_response())
}

/// Request body for `POST /model/delete`.
#[derive(Debug, Deserialize)]
pub struct DeleteModelRequest {
    pub model_id: String,
}

/// DB alias claims close the create/rename race; an expected name conflict is
/// a client error, not a failed storage service.
fn model_write_error(error: anyhow::Error) -> ApiError {
    if let Some(conflict) = error.downcast_ref::<oxllm_db::ModelNameConflict>() {
        return ApiError::bad_request(format!(
            "model_name '{}' already exists",
            conflict.model_name
        ));
    }
    db_err(error)
}

/// `POST /model/delete` — remove a DB-stored model deployment.
pub async fn delete_model(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteModelRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    // Delete directly and interpret the rows-affected count, avoiding the
    // check-then-delete TOCTOU the previous existence pre-check had (two
    // concurrent deletes of the same id raced: both passed the pre-check, the
    // second affected 0 rows). 0 rows now maps cleanly to 404.
    let rows = db.delete_model(&req.model_id).await.map_err(db_err)?;
    if rows == 0 {
        return Err(ApiError::not_found(format!(
            "model '{}' not found",
            req.model_id
        )));
    }
    // Retry the refresh once on a transient local-read failure (mirroring
    // `add_model`). The DB mutation is already durable; a transient blip in
    // `list_models_db` should not surface a 500 when the row is gone and the
    // in-memory list will self-heal on the next 30s fallback poll / peer
    // publish. Peers were already notified because `refresh_and_propagate`
    // publishes before the local refresh.
    if refresh_and_propagate(&shared).await.is_err() {
        crate::state::refresh_db_models(&shared)
            .await
            .map_err(|e| ApiError::internal(format!("failed to refresh models: {e}")))?;
    }

    tracing::info!(model_id = %req.model_id, "model deleted via management API");
    Ok(Json(json!({ "model_id": req.model_id, "status": "deleted" })).into_response())
}

/// Recursively merge `from` into `into`: for two objects, merge key-by-key
/// (recursing into shared keys); for any other shape (or a type mismatch), the
/// `from` value replaces the `into` value. Used by [`update_model`] so a
/// partial `oxllm_params` patch preserves previously-stored fields (notably
/// secrets) instead of replacing the whole block. An explicit `null` in `from`
/// clears the field in `into`.
fn deep_merge_json(into: &mut serde_json::Value, from: serde_json::Value) {
    match (into, from) {
        (serde_json::Value::Object(target), serde_json::Value::Object(source)) => {
            for (k, v) in source {
                if let Some(existing) = target.get_mut(&k) {
                    deep_merge_json(existing, v);
                } else {
                    target.insert(k, v);
                }
            }
        }
        (target, source) => *target = source,
    }
}

/// How [`encrypt_stored_secrets`] treats a blob-like value that no key in
/// the ring can decrypt — an un-prefixed legacy blob whose rotation key was
/// dropped, or (far more likely on a fresh write) plaintext that merely
/// looks like base64. The two are indistinguishable by construction, so each
/// call site picks the fail-safe that matches where the value came from.
#[derive(Clone, Copy)]
enum UndecryptableBlob {
    /// Fresh write (`/model/new`): the value came straight from the caller,
    /// so blob-like plaintext (a base64-shaped provider key) must still be
    /// encrypted, never stored as-is.
    EncryptAnyway,
    /// Merged write (`/model/update`): the value may be a legacy blob loaded
    /// from the stored row. Re-wrapping it would store ciphertext-inside-
    /// ciphertext that peels back to a dead blob upstream, so it is left
    /// untouched with a loud warning.
    SkipAndWarn,
}

/// Encrypt every secret credential embedded in `oxllm_params` in place before
/// the record is written to the DB. No-op when encryption is disabled
/// (`ENCRYPTION_KEY` unset).
///
/// This mirrors the reference model-storage helper, which runs
/// `encrypt_value_helper` over every deployment-params value. oxllm encrypts
/// every **string** value whose key matches the denylist shared with
/// [`redact_secrets`] (substring match, case-insensitive) — so `api_key`,
/// `aws_secret_access_key`, `aws_session_token`, `private_key`,
/// `service_account_json`, `password`, custom `authorization` headers, and any
/// `sk-`-prefixed value under any key are all encrypted at rest — plus every
/// string inside a secret container such as `extra_headers` (see
/// [`super::is_secret_container`]), whose header names (`X-Backend-Key`, …)
/// the denylist can never enumerate. Non-secret fields (`model`, `weight`,
/// numeric pricing, `api_base`, …) are left plaintext so they stay queryable.
///
/// "Already encrypted" is decided exactly: a value carrying the `oxenc1:`
/// magic prefix, or any blob-like value that successfully decrypts under the
/// cipher (covers legacy blobs written before the prefix existed). A blob-like
/// value that fails to decrypt is resolved by `mode` (see
/// [`UndecryptableBlob`]). Recursion is capped at
/// [`super::REDACT_MAX_DEPTH`] for stack safety on caller-controlled nesting.
fn encrypt_stored_secrets(
    params: &mut serde_json::Value,
    cipher: Option<&CredentialCipher>,
    mode: UndecryptableBlob,
) -> Result<(), ApiError> {
    let Some(cipher) = cipher else {
        return Ok(());
    };
    encrypt_secrets_inner(params, cipher, mode, 0, None, false)
}

fn encrypt_secrets_inner(
    value: &mut serde_json::Value,
    cipher: &CredentialCipher,
    mode: UndecryptableBlob,
    depth: u32,
    parent: Option<&str>,
    secret_container: bool,
) -> Result<(), ApiError> {
    if depth >= super::REDACT_MAX_DEPTH {
        return Ok(());
    }
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                let secret = secret_container
                    || super::is_secret_container(k, parent)
                    || should_redact(k, v);
                if secret {
                    if let serde_json::Value::String(s) = v {
                        if CredentialCipher::looks_encrypted(s) {
                            continue;
                        }
                        let blob_like = CredentialCipher::is_blob_like(s);
                        if blob_like && cipher.decrypt(s).is_ok() {
                            continue;
                        }
                        if blob_like && matches!(mode, UndecryptableBlob::SkipAndWarn) {
                            tracing::warn!(
                                key = %k,
                                "stored value looks like a legacy encrypted blob but no key \
                                 in the ring decrypts it; leaving it untouched to avoid \
                                 double-wrapping. Re-add the old key via ENCRYPTION_KEY_PREVIOUS \
                                 and update again to migrate it to the current key."
                            );
                            continue;
                        }
                        let encrypted =
                            crate::crypto::encrypt_api_key(Some(cipher), s).map_err(|e| {
                                tracing::warn!(
                                    error = %e,
                                    key = %k,
                                    "failed to encrypt oxllm_params secret field"
                                );
                                match e {
                                    crate::crypto::CryptoError::PlaintextTooLarge(_) => {
                                        ApiError::bad_request(format!(
                                            "oxllm_params field '{k}' is too large to encrypt \
                                             at rest ({e})"
                                        ))
                                    }
                                    _ => ApiError::internal(
                                        "failed to encrypt oxllm_params secret field",
                                    ),
                                }
                            })?;
                        *s = encrypted;
                    } else {
                        encrypt_secrets_inner(v, cipher, mode, depth + 1, Some(k), true)?;
                    }
                } else {
                    encrypt_secrets_inner(v, cipher, mode, depth + 1, Some(k), false)?;
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                encrypt_secrets_inner(item, cipher, mode, depth + 1, parent, secret_container)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Validate the two fields every stored model needs: a non-empty
/// `model_name` and a `oxllm_params.model` that is a non-empty string.
fn validate_model_entry(name: &str, params: &serde_json::Value) -> Result<(), ApiError> {
    validate_model_name(name)?;
    validate_model_params(params)
}

fn validate_model_name(name: &str) -> Result<(), ApiError> {
    if name.trim().is_empty() {
        return Err(ApiError::bad_request("model_name must be non-empty"));
    }
    Ok(())
}

fn validate_model_params(params: &serde_json::Value) -> Result<(), ApiError> {
    let model = params.get("model").and_then(|v| v.as_str()).unwrap_or("");
    if model.trim().is_empty() {
        return Err(ApiError::bad_request(
            "oxllm_params.model must be a non-empty string",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[test]
    fn validate_rejects_empty_model_name() {
        let err = validate_model_entry("", &json!({"model": "openai/gpt-4o"}))
            .expect_err("empty name rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("model_name"));
    }

    #[test]
    fn validate_rejects_empty_model() {
        let err =
            validate_model_entry("x", &json!({"model": ""})).expect_err("empty model rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("oxllm_params.model"));
    }

    #[test]
    fn validate_rejects_missing_model() {
        let err = validate_model_entry("x", &json!({})).expect_err("missing model rejected");
        assert!(err.message.contains("oxllm_params.model"));
    }

    #[test]
    fn validate_accepts_well_formed_params() {
        validate_model_entry("gpt-4o", &json!({"model": "openai/gpt-4o"}))
            .expect("well-formed params accepted");
    }

    #[test]
    fn encrypt_stored_secrets_noop_without_cipher() {
        let mut params = json!({"model": "openai/gpt-4o", "api_key": "sk-plain"});
        encrypt_stored_secrets(&mut params, None, UndecryptableBlob::EncryptAnyway)
            .expect("no cipher never fails");
        assert_eq!(params["api_key"], "sk-plain");
    }

    #[test]
    fn encrypt_stored_secrets_noop_without_api_key_field() {
        let cipher = CredentialCipher::from_passphrase("mgmt-test");
        let mut params = json!({"model": "openai/gpt-4o"});
        encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
            .expect("no api_key is fine");
        assert!(params.get("api_key").is_none());
    }

    #[test]
    fn encrypt_stored_secrets_encrypts_plaintext_key() {
        let cipher = CredentialCipher::from_passphrase("mgmt-round-trip");
        let mut params = json!({"model": "openai/gpt-4o", "api_key": "sk-secret-key"});
        encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
            .expect("encrypt succeeds");
        let stored = params["api_key"].as_str().expect("still a string");
        assert_ne!(stored, "sk-secret-key");
        assert!(CredentialCipher::looks_encrypted(stored));
        // Round-trips back to the original key.
        let plain =
            crate::crypto::decrypt_api_key(Some(&cipher), stored).expect("decrypt succeeds");
        assert_eq!(plain, "sk-secret-key");
    }

    #[test]
    fn encrypt_stored_secrets_skips_already_encrypted_value() {
        let cipher = CredentialCipher::from_passphrase("mgmt-double-encrypt");
        let blob = cipher.encrypt("sk-original").expect("encrypt");
        let mut params = json!({"model": "openai/gpt-4o", "api_key": blob});
        encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
            .expect("skip is not an error");
        // Untouched: a single decrypt still recovers the original key.
        assert_eq!(params["api_key"].as_str().unwrap(), blob);
        let plain = crate::crypto::decrypt_api_key(Some(&cipher), &blob).expect("decrypt succeeds");
        assert_eq!(plain, "sk-original");
    }

    #[test]
    fn encrypt_stored_secrets_encrypts_base64_shaped_plaintext() {
        // Regression: a plaintext key that merely *looks* like base64 must be
        // encrypted, not skipped as "already encrypted".
        let cipher = CredentialCipher::from_passphrase("mgmt-b64-false-pos");
        let fake = base64::engine::general_purpose::STANDARD.encode(b"x".repeat(48));
        assert!(fake.len() >= 40);
        assert!(!CredentialCipher::looks_encrypted(&fake));
        let mut params = json!({"model": "openai/gpt-4o", "api_key": fake.clone()});
        encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
            .expect("encrypt succeeds");
        let stored = params["api_key"].as_str().expect("still a string");
        assert_ne!(stored, fake);
        assert!(CredentialCipher::looks_encrypted(stored));
        let plain =
            crate::crypto::decrypt_api_key(Some(&cipher), stored).expect("decrypt succeeds");
        assert_eq!(plain, fake);
    }

    // ---- C30 MEDIUM: at-rest encryption covers ALL secret fields -----------
    //
    // Previously only `oxllm_params.api_key` was encrypted at rest; every other
    // provider credential (`aws_secret_access_key`, `aws_session_token`,
    // `private_key`, `service_account_json`, `password`, custom
    // `Authorization` headers) was stored plaintext in the DB. The reference
    // encrypts every deployment-params value; oxllm now encrypts every secret
    // the redaction denylist catches. This test asserts no plaintext survives
    // the encrypt pass for the full set and that every value round-trips.

    #[test]
    fn encrypt_stored_secrets_encrypts_every_known_secret_field() {
        let cipher = CredentialCipher::from_passphrase("mgmt-all-fields");
        let mut params = json!({
            "model": "bedrock/anthropic.claude-3",
            "api_key": "sk-openai",
            "aws_secret_access_key": "AWSSECRET123",
            "aws_session_token": "AQoSESSION",
            "private_key": "-----BEGIN PRIVATE KEY-----",
            "service_account_json": "{\"private_key\":\"pk\"}",
            "password": "hunter2",
            "authorization": "Bearer secret-token",
            "headers": {
                "Authorization": "Bearer header-secret",
                "Content-Type": "application/json"
            },
            "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "weight": 2
        });

        encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
            .expect("encrypt succeeds");

        let serialized = params.to_string();
        // No plaintext secret survives the encrypt pass.
        assert!(
            !serialized.contains("sk-openai"),
            "api_key plaintext leaked: {serialized}"
        );
        assert!(
            !serialized.contains("AWSSECRET123"),
            "aws_secret_access_key plaintext leaked: {serialized}"
        );
        assert!(
            !serialized.contains("AQoSESSION"),
            "aws_session_token plaintext leaked: {serialized}"
        );
        assert!(
            !serialized.contains("-----BEGIN PRIVATE KEY-----"),
            "private_key plaintext leaked: {serialized}"
        );
        assert!(
            !serialized.contains("\"private_key\":\"pk\""),
            "service_account_json plaintext leaked: {serialized}"
        );
        assert!(
            !serialized.contains("hunter2"),
            "password plaintext leaked: {serialized}"
        );
        assert!(
            !serialized.contains("Bearer secret-token"),
            "authorization plaintext leaked: {serialized}"
        );
        assert!(
            !serialized.contains("Bearer header-secret"),
            "nested header Authorization plaintext leaked: {serialized}"
        );
        // Non-secret fields survive.
        assert_eq!(params["model"], "bedrock/anthropic.claude-3");
        assert_eq!(
            params["api_base"],
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        );
        assert_eq!(params["weight"], 2);
        assert_eq!(params["headers"]["Content-Type"], "application/json");

        // Every encrypted value round-trips to its original plaintext.
        assert_eq!(decrypt(&cipher, &params["api_key"]), "sk-openai",);
        assert_eq!(
            decrypt(&cipher, &params["aws_secret_access_key"]),
            "AWSSECRET123",
        );
        assert_eq!(decrypt(&cipher, &params["aws_session_token"]), "AQoSESSION",);
        assert_eq!(
            decrypt(&cipher, &params["private_key"]),
            "-----BEGIN PRIVATE KEY-----",
        );
        assert_eq!(
            decrypt(&cipher, &params["service_account_json"]),
            "{\"private_key\":\"pk\"}",
        );
        assert_eq!(decrypt(&cipher, &params["password"]), "hunter2");
        assert_eq!(
            decrypt(&cipher, &params["authorization"]),
            "Bearer secret-token",
        );
        assert_eq!(
            decrypt(&cipher, &params["headers"]["Authorization"]),
            "Bearer header-secret",
        );
    }

    #[test]
    fn encrypt_stored_secrets_leaves_non_secret_string_fields_plaintext() {
        // The denylist restricts to credential-shaped string keys; non-secret
        // string fields (model, api_base, provider config) stay plaintext so
        // they remain queryable.
        let cipher = CredentialCipher::from_passphrase("mgmt-non-secret");
        let mut params = json!({
            "model": "openai/gpt-4o",
            "api_base": "https://api.openai.com/v1",
            "region": "us-east-1",
            "weight": 1
        });
        encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
            .expect("encrypt succeeds");
        assert_eq!(params["model"], "openai/gpt-4o");
        assert_eq!(params["api_base"], "https://api.openai.com/v1");
        assert_eq!(params["region"], "us-east-1");
        assert_eq!(params["weight"], 1);
    }

    fn decrypt(cipher: &CredentialCipher, value: &serde_json::Value) -> String {
        let stored = value.as_str().expect("encrypted value is a string");
        crate::crypto::decrypt_api_key(Some(cipher), stored).expect("decrypt succeeds")
    }

    // ---- C06 MEDIUM: secret containers (extra_headers) --------------------
    //
    // Header maps sent to a provider carry credentials under arbitrary header
    // names (`X-Backend-Key`) that the key-name denylist can never enumerate;
    // previously such values were stored plaintext despite ENCRYPTION_KEY
    // being set (the reference encrypts every deployment-params value). Every string
    // inside `extra_headers` — and inside `headers` nested under `extra_body`
    // — is now treated as secret.

    #[test]
    fn encrypt_stored_secrets_encrypts_extra_headers_and_extra_body_headers() {
        let cipher = CredentialCipher::from_passphrase("mgmt-containers");
        let mut params = json!({
            "model": "openai/gpt-4o",
            "extra_headers": {
                "Authorization": "Bearer extra-header-token",
                "X-Backend-Key": "hunter2"
            },
            "extra_body": {
                "headers": { "X-Signature": "sig-secret" },
                "conversation_id": "conv-keep"
            }
        });
        encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
            .expect("encrypt succeeds");
        let serialized = params.to_string();
        assert!(!serialized.contains("hunter2"), "leaked: {serialized}");
        assert!(
            !serialized.contains("extra-header-token"),
            "leaked: {serialized}"
        );
        assert!(!serialized.contains("sig-secret"), "leaked: {serialized}");
        // Non-secret siblings stay plaintext / queryable.
        assert_eq!(params["extra_body"]["conversation_id"], "conv-keep");
        // Every container value round-trips.
        assert_eq!(
            decrypt(&cipher, &params["extra_headers"]["X-Backend-Key"]),
            "hunter2"
        );
        assert_eq!(
            decrypt(&cipher, &params["extra_headers"]["Authorization"]),
            "Bearer extra-header-token"
        );
        assert_eq!(
            decrypt(&cipher, &params["extra_body"]["headers"]["X-Signature"]),
            "sig-secret"
        );
    }

    // ---- C06 LOW: no silent re-wrap of legacy blobs on /model/update -------

    #[test]
    fn encrypt_stored_secrets_does_not_rewrap_undecryptable_legacy_blob_on_update() {
        // Simulate a legacy (pre-magic) row: a blob written under key A with
        // the prefix stripped, whose rotation key was since dropped.
        let key_a = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let key_b = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
        let cipher_a = CredentialCipher::from_passphrase(key_a);
        let cipher_b = CredentialCipher::from_passphrase(key_b);
        let blob = cipher_a.encrypt("sk-legacy").expect("encrypt under key a");
        let legacy = blob.trim_start_matches("oxenc1:").to_string();
        assert!(!CredentialCipher::looks_encrypted(&legacy));
        assert!(CredentialCipher::is_blob_like(&legacy));
        assert!(cipher_b.decrypt(&legacy).is_err());

        // Update path (SkipAndWarn): the undecryptable legacy blob is left
        // untouched — never wrapped inside a second blob that would peel back
        // to dead ciphertext upstream.
        let mut params = json!({"model": "openai/gpt-4o", "api_key": legacy.clone()});
        encrypt_stored_secrets(&mut params, Some(&cipher_b), UndecryptableBlob::SkipAndWarn)
            .expect("skip is not an error");
        assert_eq!(params["api_key"].as_str().unwrap(), legacy);

        // Add path (EncryptAnyway) keeps the older contract: blob-like input
        // from a fresh caller is treated as plaintext and encrypted.
        let mut fresh = json!({"model": "openai/gpt-4o", "api_key": legacy.clone()});
        encrypt_stored_secrets(
            &mut fresh,
            Some(&cipher_b),
            UndecryptableBlob::EncryptAnyway,
        )
        .expect("encrypt succeeds");
        assert!(CredentialCipher::looks_encrypted(
            fresh["api_key"].as_str().unwrap()
        ));
    }

    #[test]
    fn encrypt_stored_secrets_oversize_secret_fails_loudly() {
        // A secret larger than the symmetric decrypt ceiling must fail the
        // write with a 400 (PlaintextTooLarge), never store a blob this
        // module could not decrypt back.
        let cipher = CredentialCipher::from_passphrase("mgmt-oversize");
        let huge = "k".repeat(crate::crypto::MAX_PLAINTEXT_LEN + 1);
        let mut params = json!({"model": "openai/gpt-4o", "api_key": huge});
        let err =
            encrypt_stored_secrets(&mut params, Some(&cipher), UndecryptableBlob::EncryptAnyway)
                .expect_err("oversize secret must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("too large"));
        assert_eq!(
            params["api_key"].as_str().unwrap().len(),
            crate::crypto::MAX_PLAINTEXT_LEN + 1
        );
    }

    // ---- list_models: multi-provider aggregation (C30 LOW) -----------------

    fn dep(model_name: &str, provider_model: &str) -> Deployment {
        serde_json::from_str(
            &json!({
                "model_name": model_name,
                "oxllm_params": {"model": provider_model},
            })
            .to_string(),
        )
        .expect("parse Deployment")
    }

    #[test]
    fn model_list_rows_surfaces_all_providers_for_shared_model_name() {
        // Failover topology: one model_name backed by OpenAI + Azure. The old
        // HashSet dedup kept only the first deployment and reported a single,
        // order-dependent `provider`.
        let deps = vec![
            dep("gpt-4o", "openai/gpt-4o"),
            dep("gpt-4o", "azure/gpt-4o"),
        ];
        let rows = model_list_rows(&deps);
        assert_eq!(rows.len(), 1, "one row per model_name");
        let row = &rows[0];
        assert_eq!(row["id"].as_str(), Some("gpt-4o"));
        let providers = row["providers"].as_array().expect("providers array");
        assert_eq!(providers.len(), 2, "both providers visible");
        assert_eq!(providers[0].as_str(), Some("azure"));
        assert_eq!(providers[1].as_str(), Some("openai"));
        // The singular `provider` is the (sorted) first, not whatever happened
        // to be first in the deployment list.
        assert_eq!(row["provider"].as_str(), Some("azure"));
    }

    #[test]
    fn model_list_rows_preserves_first_seen_order_and_skips_wildcard() {
        let deps = vec![
            dep("claude-3", "anthropic/claude-3"),
            dep("gpt-4o", "openai/gpt-4o"),
            dep("gpt-4o", "azure/gpt-4o"),
            dep("*", "openai/fallback"),
        ];
        let rows = model_list_rows(&deps);
        let ids: Vec<&str> = rows.iter().filter_map(|r| r["id"].as_str()).collect();
        assert_eq!(
            ids,
            vec!["claude-3", "gpt-4o"],
            "first-seen order, wildcard dropped"
        );
        // claude-3 has a single provider.
        let claude = &rows[0];
        assert_eq!(claude["providers"].as_array().unwrap().len(), 1);
        assert_eq!(claude["provider"].as_str(), Some("anthropic"));
    }

    #[test]
    fn model_list_rows_dedups_repeated_provider() {
        // Two deployments with the same provider under the same model_name
        // collapse to a single entry in `providers`.
        let deps = vec![
            dep("gpt-4o", "openai/gpt-4o"),
            dep("gpt-4o", "openai/gpt-4o-2024-08-06"),
        ];
        let rows = model_list_rows(&deps);
        let providers = rows[0]["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].as_str(), Some("openai"));
    }

    #[test]
    fn attach_db_model_ids_marks_only_db_backed_rows() {
        // A model added via POST /model/new is deletable by its DB id; a
        // YAML-configured model is not. /model/list must distinguish them so
        // clients (the admin UI's delete button) never send a model_name
        // where the DB primary key is required.
        let deps = vec![
            dep("db-model", "openai/gpt-4o"),
            dep("yaml-model", "anthropic/claude-3"),
        ];
        let mut rows = model_list_rows(&deps);
        let now = oxllm_db::now_iso();
        let records = vec![oxllm_db::ModelConfigRecord {
            model_id: "model-abc123".to_string(),
            model_name: "db-model".to_string(),
            oxllm_params: json!({"model": "openai/gpt-4o"}),
            model_info: None,
            created_at: now.clone(),
            updated_at: now,
        }];
        attach_db_model_ids(&mut rows, &records);
        assert_eq!(rows[0]["model_id"].as_str(), Some("model-abc123"));
        assert!(
            rows[1].get("model_id").is_none(),
            "YAML-configured rows carry no model_id"
        );
    }

    #[test]
    fn model_list_caps_at_scope_limit_and_reports_truncation() {
        // One row over the cap must surface `truncated: true` to /model/list
        // (and therefore the admin table) rather than silently dropping rows.
        let deps: Vec<Deployment> = (0..crate::handlers::management::SCOPE_LIST_LIMIT + 1)
            .map(|i| dep(&format!("model-{i}"), "openai/gpt-4o"))
            .collect();
        let mut rows = model_list_rows(&deps);
        let truncated = truncate_scope_list(&mut rows);
        assert!(truncated, "one row over the cap must report truncation");
        assert_eq!(rows.len(), crate::handlers::management::SCOPE_LIST_LIMIT);
    }

    // ---- update_model response redaction (C30 MEDIUM) ---------------------
    //
    // At-rest encryption only covers `oxllm_params.api_key`; every other
    // provider secret is stored verbatim and would leak via POST /model/update.
    // The handler now routes the stored record through the shared
    // `redact_secrets` before serializing. This test pins the redaction for
    // the Bedrock/Vertex-style secret fields the report called out.

    #[test]
    fn redact_secrets_covers_update_model_response_secret_fields() {
        let mut record = json!({
            "model_id": "model-abc",
            "model_name": "bedrock-claude",
            "oxllm_params": {
                "model": "bedrock/anthropic.claude-3",
                "api_key": "sk-should-be-redacted",
                "aws_secret_access_key": "AWSSECRET123",
                "aws_session_token": "AQoEXAMPLEH4aoAH0eNSPE",
                "authorization": "Bearer secret-token",
                "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "weight": 2
            },
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z"
        });
        redact_secrets(&mut record);
        let serialized = record.to_string();
        assert!(!serialized.contains("sk-should-be-redacted"));
        assert!(!serialized.contains("AWSSECRET123"));
        assert!(!serialized.contains("AQoEXAMPLEH4aoAH0eNSPE"));
        assert!(!serialized.contains("secret-token"));
        assert!(serialized.contains("***"));
        // Non-secret fields survive.
        assert_eq!(
            record["oxllm_params"]["api_base"],
            "https://bedrock-runtime.us-east-1.amazonaws.com"
        );
        assert_eq!(record["oxllm_params"]["weight"], 2);
        assert_eq!(record["model_id"], "model-abc");
    }

    // ---- refresh_and_propagate publishes before local refresh (C65 MEDIUM) --
    //
    // Pins the contract that a management mutation always notifies peers
    // (publish fires unconditionally, before the local refresh), so a
    // transient local-refresh failure can never strand a committed DB row
    // on every other replica. The local-refresh error path itself is
    // verified by code inspection: `refresh_and_propagate` calls
    // `publish_invalidation` first, then `refresh_db_models`, and returns
    // the latter's `Err` to the caller while the publish has already fired.

    /// A minimal Cache whose `publish_invalidation` records every key it was
    /// called with. Used to assert the `__model_list__` publish fires.
    struct PublishingCache {
        published: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl oxllm_core::Cache for PublishingCache {
        async fn get(&self, _key: &str) -> Option<String> {
            None
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<std::time::Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(
            &self,
            _key: &str,
            _by: f64,
            _ttl: Option<std::time::Duration>,
        ) -> Result<f64, oxllm_core::Error> {
            Ok(0.0)
        }
        async fn publish_invalidation(&self, key: &str) {
            self.published
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(key.to_string());
        }
    }

    fn shared_with_publishing_cache() -> (
        std::sync::Arc<arc_swap::ArcSwap<crate::state::AppState>>,
        std::sync::Arc<PublishingCache>,
    ) {
        let cache = std::sync::Arc::new(PublishingCache {
            published: std::sync::Mutex::new(Vec::new()),
        });
        let state = crate::state::AppState::with_registry_cache_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
            cache.clone(),
            None,
            None,
        )
        .expect("state");
        (
            std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state)),
            cache,
        )
    }

    #[tokio::test]
    async fn refresh_and_propagate_publishes_even_when_no_db_configured() {
        // No DB → `refresh_db_models` short-circuits with Ok(()) because
        // `current.db` is None. The publish must still fire exactly once,
        // proving it is unconditional (it is not gated on the refresh path).
        let (shared, cache) = shared_with_publishing_cache();
        refresh_and_propagate(&shared).await.expect("ok");
        let published = cache
            .published
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        assert_eq!(
            published,
            vec!["__model_list__".to_string()],
            "publish must fire exactly once with the model-list sentinel, \
             independent of the local-refresh outcome"
        );
    }

    // ---- deep_merge_json (C30 MEDIUM: partial update must not wipe secrets) --
    //
    // update_model now deep-merges the new oxllm_params into the stored ones,
    // so a PATCH that tweaks a non-secret field cannot silently drop a stored
    // api_key / aws_secret_access_key. Explicit `null` still clears a field.

    #[test]
    fn deep_merge_preserves_secrets_on_partial_patch() {
        let mut stored = json!({
            "model": "openai/gpt-4o",
            "api_key": "oxenc1:encrypted-blob",
            "aws_secret_access_key": "AWSSECRET",
            "weight": 1
        });
        let patch = json!({ "weight": 2 });
        deep_merge_json(&mut stored, patch);
        assert_eq!(stored["model"], "openai/gpt-4o");
        assert_eq!(stored["api_key"], "oxenc1:encrypted-blob");
        assert_eq!(stored["aws_secret_access_key"], "AWSSECRET");
        assert_eq!(stored["weight"], 2);
    }

    #[test]
    fn deep_merge_overwrites_when_caller_provides_value() {
        let mut stored = json!({ "api_key": "old", "weight": 1 });
        let patch = json!({ "api_key": "sk-new-plaintext" });
        deep_merge_json(&mut stored, patch);
        assert_eq!(stored["api_key"], "sk-new-plaintext");
        assert_eq!(stored["weight"], 1);
    }

    #[test]
    fn deep_merge_explicit_null_clears_field() {
        let mut stored = json!({ "api_key": "old", "weight": 1 });
        let patch = json!({ "api_key": serde_json::Value::Null });
        deep_merge_json(&mut stored, patch);
        assert!(stored["api_key"].is_null());
        assert_eq!(stored["weight"], 1);
    }

    // ---- model_info: first-deployment pricing on multi-deployment groups -----
    //
    // Pins the documented behaviour: when a model_name is backed by multiple
    // deployments, model_info reports the FIRST matching deployment's pricing
    // (a single representative snapshot). list_models surfaces every provider;
    // operators needing the full topology use GET /config.

    #[test]
    fn model_info_body_reports_first_deployment_for_shared_model_name() {
        let deps = vec![
            dep("gpt-4o", "openai/gpt-4o"),
            dep("gpt-4o", "azure/gpt-4o"),
        ];
        let body = model_info_body(&deps, "gpt-4o").expect("first deployment found");
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["provider"], "openai");
        assert_eq!(body["provider_model"], "openai/gpt-4o");
    }

    #[test]
    fn model_info_body_returns_none_for_unknown_model() {
        let deps = vec![dep("gpt-4o", "openai/gpt-4o")];
        assert!(model_info_body(&deps, "missing").is_none());
    }

    #[test]
    fn deep_merge_recurses_into_nested_objects() {
        let mut stored = json!({
            "extra": { "keep": 1, "replace": "old" }
        });
        let patch = json!({ "extra": { "replace": "new", "added": true } });
        deep_merge_json(&mut stored, patch);
        assert_eq!(stored["extra"]["keep"], 1);
        assert_eq!(stored["extra"]["replace"], "new");
        assert_eq!(stored["extra"]["added"], true);
    }

    #[tokio::test]
    async fn refresh_and_propagate_publishes_before_local_refresh_returns() {
        // The publish must happen *before* refresh_and_propagate returns
        // control to the caller (so a caller that observed an error still
        // knows peers were notified). We assert the ordering by checking the
        // publish trace is non-empty immediately after the call returns,
        // whether the call succeeded or errored. With no DB configured the
        // call succeeds and the trace must contain exactly one publish.
        let (shared, cache) = shared_with_publishing_cache();
        let outcome = refresh_and_propagate(&shared).await;
        assert!(outcome.is_ok(), "no DB → refresh_db_models is a no-op Ok");
        let count = cache
            .published
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        assert_eq!(
            count, 1,
            "publish must have fired by the time refresh_and_propagate returns"
        );
    }
}
