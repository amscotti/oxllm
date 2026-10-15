//! Auth middleware.
//!
//! Master-key auth by default. When `database_url` is configured, virtual API
//! keys are looked up in the DB. JWT auth is checked third (after master and
//! virtual keys). If no master key is configured, auth is
//! skipped (open mode for development).

pub mod jwt;

#[cfg(feature = "db")]
mod cache;

use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::error::ApiError;
use crate::state::SharedState;

pub(crate) const MIN_AUTH_SECRET_LEN: usize = 16;

/// Cache controls understood by `CacheHook`. Management accepts these exact
/// names; an absent allowlist is unrestricted, while an empty one denies all
/// explicitly supplied controls.
pub const SUPPORTED_CACHE_CONTROLS: &[&str] = &["insert", "use_cache", "no-cache", "no-store"];

#[cfg(feature = "db")]
pub(crate) fn validate_allowed_cache_controls(controls: &[String]) -> Result<(), ApiError> {
    for control in controls {
        if !SUPPORTED_CACHE_CONTROLS.contains(&control.as_str()) {
            return Err(ApiError::bad_request(format!(
                "unsupported allowed_cache_controls value '{control}'; supported controls: {}",
                SUPPORTED_CACHE_CONTROLS.join(", ")
            )));
        }
    }
    Ok(())
}

/// Decode the key's JSON TEXT policy without collapsing `[]` into None.
/// Invalid persisted policies deny all explicit controls rather than silently
/// granting an unrestricted cache policy.
#[cfg(feature = "db")]
pub(crate) fn parse_cache_controls_column(raw: Option<&str>) -> Option<Vec<String>> {
    let raw = raw?;
    match serde_json::from_str::<Vec<String>>(raw) {
        Ok(controls) if validate_allowed_cache_controls(&controls).is_ok() => Some(controls),
        _ => {
            tracing::warn!(
                "invalid allowed_cache_controls column; denying all explicit cache controls"
            );
            Some(Vec::new())
        }
    }
}

/// Reject configured authentication that cannot be enforced. This is shared
/// by config validation and state assembly, including programmatic reloads.
pub(crate) fn validate_auth_config(
    settings: &crate::config::types::GeneralSettings,
) -> anyhow::Result<()> {
    let development = crate::config::loader::environment_is_development();
    if settings.allow_short_master_key == Some(true) && !development {
        anyhow::bail!(
            "general_settings.allow_short_master_key is set but the environment is not \
             development; unset the flag or set OXLLM_ENVIRONMENT or LITELLM_ENVIRONMENT \
             to development"
        );
    }
    let short_master_ok = settings.allow_short_master_key == Some(true) && development;
    for (name, secret) in [
        ("master_key", settings.master_key.as_deref()),
        ("jwt_secret", settings.jwt_secret.as_deref()),
    ] {
        if let Some(secret) = secret {
            let blank = secret.trim().is_empty();
            let too_short = secret.len() < MIN_AUTH_SECRET_LEN;
            let allowed = name == "master_key" && short_master_ok && !blank;
            if (blank || too_short) && !allowed {
                anyhow::bail!(
                    "general_settings.{name} is blank or too short; require at least {MIN_AUTH_SECRET_LEN} bytes"
                );
            }
        }
    }
    if settings.jwt_secret.is_some() && jwt::parse_algorithm(&settings.jwt_algorithm).is_none() {
        anyhow::bail!(
            "general_settings.jwt_algorithm must be a supported HMAC algorithm (HS256, HS384 or HS512)"
        );
    }
    if settings
        .database_url
        .as_deref()
        .is_some_and(|url| url.trim().is_empty())
    {
        anyhow::bail!("general_settings.database_url must not be blank");
    }
    Ok(())
}

/// The authenticated principal, stored in request extensions.
#[derive(Debug, Clone, Default)]
pub struct AuthUser {
    /// SHA-256 hash of the provided key (never the raw key).
    pub key_hash: [u8; 32],
    /// Lowercase hex encoding of `key_hash`, precomputed at construction time.
    pub key_hash_hex: String,
    /// True when the principal authenticated as the configured master key
    /// (bypasses all checks and is the only key permitted on management
    /// endpoints).
    pub is_master: bool,
    /// Virtual-key id (when authenticated against a generated key in the DB).
    pub key_id: Option<String>,
    /// Team id attached to the virtual key, if any.
    pub team_id: Option<String>,
    /// User id attached to the virtual key, if any.
    pub user_id: Option<String>,
    /// Org id attached to the virtual key, if any.
    pub org_id: Option<String>,
    /// Project id attached to the virtual key, if any.
    pub project_id: Option<String>,
    /// Per-key budget override (USD) — key spend bucket only.
    pub max_budget: Option<f64>,
    /// Per-key budget window (e.g. `"30d"`).
    pub budget_duration: Option<String>,
    /// Per-key requests-per-minute limit.
    pub rpm: Option<u32>,
    /// Per-key tokens-per-minute limit.
    pub tpm: Option<u32>,
    /// Team max budget when the key belongs to a team with its own ceiling.
    pub team_max_budget: Option<f64>,
    /// Team RPM when configured on the team record.
    pub team_rpm: Option<u32>,
    /// Team TPM when configured on the team record.
    pub team_tpm: Option<u32>,
    /// Team budget window (e.g. `"1d"`). When set, takes precedence over the
    /// key's window for the team spend counter TTL.
    pub team_budget_duration: Option<String>,
    /// Org max budget when the key belongs to an org with its own ceiling.
    pub org_max_budget: Option<f64>,
    /// Org RPM when configured on the org record.
    pub org_rpm: Option<u32>,
    /// Org TPM when configured on the org record.
    pub org_tpm: Option<u32>,
    /// Org budget window (e.g. `"1d"`). When set, takes precedence over the
    /// key's window for the org spend counter TTL.
    pub org_budget_duration: Option<String>,
    /// Project max budget when the key belongs to a project with a ceiling.
    pub project_max_budget: Option<f64>,
    /// Project RPM when configured on the project record.
    pub project_rpm: Option<u32>,
    /// Project TPM when configured on the project record.
    pub project_tpm: Option<u32>,
    /// Project budget window (e.g. `"1d"`). When set, takes precedence over
    /// the key's window for the project spend counter TTL.
    pub project_budget_duration: Option<String>,
    /// User max budget when the key is attached to a user with a ceiling.
    pub user_max_budget: Option<f64>,
    /// User budget window (e.g. `"7d"`). When set, takes precedence over the
    /// key's window for the user spend counter TTL.
    pub user_budget_duration: Option<String>,
    /// Concurrent request gauge limit (from org/budget records). `None` =
    /// unlimited.
    pub max_parallel_requests: Option<u32>,
    /// Allowed-model patterns attached to the key (JSON-decoded from
    /// `KeyRecord::models`). `None` or empty means "all models allowed".
    pub key_models: Option<Vec<String>>,
    /// Allowed-model patterns attached to the key's team. `None` or empty
    /// means the team imposes no restriction.
    pub team_models: Option<Vec<String>>,
    /// Allowed-model patterns attached to the key's user. `None` or empty
    /// means the user imposes no restriction.
    pub user_models: Option<Vec<String>>,
    /// Per-key cache-control allowlist. None is unrestricted; Some([]) denies
    /// every explicitly supplied control but permits requests without controls.
    pub allowed_cache_controls: Option<Vec<String>>,
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

impl AuthUser {
    /// An open-mode (no auth) user.
    pub fn open() -> Self {
        let key_hash = [0u8; 32];
        Self {
            key_hash_hex: bytes_to_hex(&key_hash),
            key_hash,
            ..Default::default()
        }
    }

    /// The master-key principal.
    pub fn master(key_hash: [u8; 32]) -> Self {
        Self {
            key_hash_hex: bytes_to_hex(&key_hash),
            key_hash,
            is_master: true,
            ..Default::default()
        }
    }

    /// A virtual-key principal loaded from the DB.
    pub fn virtual_key(key_hash: [u8; 32], info: VirtualKeyInfo) -> Self {
        let limits = info.scope_limits();
        let max_parallel = i64_rate_to_u32(info.max_parallel_requests);
        let mut user = Self {
            key_hash_hex: bytes_to_hex(&key_hash),
            key_hash,
            is_master: false,
            key_id: Some(info.key_id),
            team_id: info.team_id,
            user_id: info.user_id,
            org_id: info.org_id,
            project_id: info.project_id,
            key_models: info.key_models,
            team_models: info.team_models,
            user_models: info.user_models,
            allowed_cache_controls: info.allowed_cache_controls,
            max_parallel_requests: max_parallel,
            ..Default::default()
        };
        user.set_scope_limits(&limits);
        user
    }

    /// Lowercase hex encoding of the key hash, suitable for use as a cache key
    /// component (e.g. `spend:<hash>`). Never reveals the raw key.
    pub fn key_hash_hex(&self) -> &str {
        &self.key_hash_hex
    }

    /// All five scopes' limits, grouped. Scopes without a matching column
    /// (user rate limits) keep `None` fields.
    pub fn scope_limits(&self) -> oxllm_core::AllScopeLimits {
        oxllm_core::AllScopeLimits {
            key: oxllm_core::ScopeLimits {
                max_budget: self.max_budget,
                budget_duration: self.budget_duration.clone(),
                rpm: self.rpm,
                tpm: self.tpm,
            },
            team: oxllm_core::ScopeLimits {
                max_budget: self.team_max_budget,
                budget_duration: self.team_budget_duration.clone(),
                rpm: self.team_rpm,
                tpm: self.team_tpm,
            },
            user: oxllm_core::ScopeLimits {
                max_budget: self.user_max_budget,
                budget_duration: self.user_budget_duration.clone(),
                ..Default::default()
            },
            org: oxllm_core::ScopeLimits {
                max_budget: self.org_max_budget,
                budget_duration: self.org_budget_duration.clone(),
                rpm: self.org_rpm,
                tpm: self.org_tpm,
            },
            project: oxllm_core::ScopeLimits {
                max_budget: self.project_max_budget,
                budget_duration: self.project_budget_duration.clone(),
                rpm: self.project_rpm,
                tpm: self.project_tpm,
            },
        }
    }

    /// Expand grouped limits onto the flat per-scope fields. Scopes without
    /// a matching column (user rate limits) are ignored for RPM/TPM.
    pub fn set_scope_limits(&mut self, limits: &oxllm_core::AllScopeLimits) {
        self.max_budget = limits.key.max_budget;
        self.budget_duration = limits.key.budget_duration.clone();
        self.rpm = limits.key.rpm;
        self.tpm = limits.key.tpm;
        self.team_max_budget = limits.team.max_budget;
        self.team_budget_duration = limits.team.budget_duration.clone();
        self.team_rpm = limits.team.rpm;
        self.team_tpm = limits.team.tpm;
        self.user_max_budget = limits.user.max_budget;
        self.user_budget_duration = limits.user.budget_duration.clone();
        self.org_max_budget = limits.org.max_budget;
        self.org_budget_duration = limits.org.budget_duration.clone();
        self.org_rpm = limits.org.rpm;
        self.org_tpm = limits.org.tpm;
        self.project_max_budget = limits.project.max_budget;
        self.project_budget_duration = limits.project.budget_duration.clone();
        self.project_rpm = limits.project.rpm;
        self.project_tpm = limits.project.tpm;
    }
}

/// Budget/rate-limit settings extracted from a virtual key row.
///
/// Key-level limits (`max_budget`, `rpm`, `tpm`) are **not** merged with team
/// or user ceilings; those live in the dedicated `team_*` / `user_*` fields so
/// hooks can check the correct shared buckets.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct VirtualKeyInfo {
    pub key_id: String,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub org_id: Option<String>,
    pub project_id: Option<String>,
    pub max_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
    pub team_max_budget: Option<f64>,
    pub team_rpm: Option<i64>,
    pub team_tpm: Option<i64>,
    pub team_budget_duration: Option<String>,
    pub org_max_budget: Option<f64>,
    pub org_rpm: Option<i64>,
    pub org_tpm: Option<i64>,
    pub org_budget_duration: Option<String>,
    pub project_max_budget: Option<f64>,
    pub project_rpm: Option<i64>,
    pub project_tpm: Option<i64>,
    pub project_budget_duration: Option<String>,
    pub user_max_budget: Option<f64>,
    pub user_budget_duration: Option<String>,
    /// Concurrent request gauge (from org record). `None` = unlimited.
    pub max_parallel_requests: Option<i64>,
    pub expires_at: Option<String>,
    pub is_active: bool,
    /// Allowed-model patterns from the key record (parsed from the
    /// `models` JSON column). `None`/empty = unrestricted.
    pub key_models: Option<Vec<String>>,
    /// Allowed-model patterns from the team record. `None`/empty = unrestricted.
    pub team_models: Option<Vec<String>>,
    /// Allowed-model patterns from the user record. `None`/empty = unrestricted.
    pub user_models: Option<Vec<String>>,
    /// The key's cache-control policy, inherited by AuthUser and CallContext.
    /// Missing legacy fields remain unrestricted; [] is a deny-all policy.
    #[serde(default)]
    pub allowed_cache_controls: Option<Vec<String>>,
}

/// Convert a DB `i64` rate column to the `u32` enforcement type. Values above
/// `u32::MAX` clamp to `u32::MAX` (closest representable ceiling) rather than
/// collapsing to `None` — a row written outside the management API (direct DB
/// edit, migration, future writer) must not silently *disable* a
/// billing-adjacent control. Negative values collapse to `None` (unlimited);
/// they are rejected by management `validate_limits` and pinned as `None` by
/// `scope_limits_negative_rates_collapse_to_none`.
fn i64_rate_to_u32(v: Option<i64>) -> Option<u32> {
    v.filter(|&v| v >= 0).map(|v| v.min(u32::MAX as i64) as u32)
}

impl VirtualKeyInfo {
    /// All five scopes' limits, grouped. DB rates are stored as `i64` and
    /// coerced per `i64_rate_to_u32`.
    pub fn scope_limits(&self) -> oxllm_core::AllScopeLimits {
        oxllm_core::AllScopeLimits {
            key: oxllm_core::ScopeLimits {
                max_budget: self.max_budget,
                budget_duration: self.budget_duration.clone(),
                rpm: i64_rate_to_u32(self.rpm),
                tpm: i64_rate_to_u32(self.tpm),
            },
            team: oxllm_core::ScopeLimits {
                max_budget: self.team_max_budget,
                budget_duration: self.team_budget_duration.clone(),
                rpm: i64_rate_to_u32(self.team_rpm),
                tpm: i64_rate_to_u32(self.team_tpm),
            },
            user: oxllm_core::ScopeLimits {
                max_budget: self.user_max_budget,
                budget_duration: self.user_budget_duration.clone(),
                ..Default::default()
            },
            org: oxllm_core::ScopeLimits {
                max_budget: self.org_max_budget,
                budget_duration: self.org_budget_duration.clone(),
                rpm: i64_rate_to_u32(self.org_rpm),
                tpm: i64_rate_to_u32(self.org_tpm),
            },
            project: oxllm_core::ScopeLimits {
                max_budget: self.project_max_budget,
                budget_duration: self.project_budget_duration.clone(),
                rpm: i64_rate_to_u32(self.project_rpm),
                tpm: i64_rate_to_u32(self.project_tpm),
            },
        }
    }
}

/// Compute the SHA-256 hash of a byte string.
pub fn hash_key(input: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(input);
    hasher.finalize().into()
}

/// Hex-encode a hash as a lowercase string (used as the lookup key for virtual
/// keys in the DB and the key cache).
pub fn hash_key_hex(input: &[u8]) -> String {
    bytes_to_hex(&hash_key(input))
}

#[cfg(feature = "db")]
const VK_POSITIVE_TTL: std::time::Duration = std::time::Duration::from_secs(300);
#[cfg(feature = "db")]
const VK_NEGATIVE_TTL: std::time::Duration = std::time::Duration::from_secs(60);
/// Tombstone written when the key-row DB read itself failed. Distinct from
/// versioned missing-record snapshots so a later cache hit can tell "the DB
/// was unreachable" (→ 503, brake untouched). Its short TTL only denies
/// admission; it never authorizes a principal from unverified data.
#[cfg(feature = "db")]
const VK_DB_ERROR: &str = "__oxllm_db_error__";

/// Outcome of resolving a presented credential against the virtual-key store.
///
/// Distinguishes an *authentication* failure (the credential is wrong — 401,
/// counts toward the auth brake) from a *server-side* failure (the backing
/// store was unreachable — 503, must not arm the brake: legitimate clients
/// retrying through a DB flap cannot be allowed to exhaust their failure
/// budget and get 429-blocked).
#[cfg(feature = "db")]
#[derive(Debug)]
enum LookupOutcome {
    /// The credential resolved to an authenticated principal.
    Found(Box<AuthUser>),
    /// The credential is genuinely invalid (unknown, expired, inactive, or
    /// rejected by policy such as a blocked team). Terminal only after the
    /// JWT fallback also fails; counts toward the auth brake.
    Invalid,
    /// A server-side storage failure made resolution impossible. The request
    /// fails with 503 and does not count toward the auth brake.
    ServerError,
}

/// Clear the affected auth snapshot and broadcast its invalidation after the
/// management DB mutation commits. Every key/scope cache read and fill uses
/// the DB-authoritative content-version protocol in `auth::cache`: neither a
/// late fill nor a missed delete/pub-sub message can resurrect an old policy.
/// Redis/local cache generations are never trusted, including in DualCache.
/// Only this key is evicted; response and spend caches remain independent.
pub async fn invalidate_cache_key(state: &crate::state::AppState, cache_key: &str) {
    state.cache.delete(cache_key).await;
    state.cache.publish_invalidation(cache_key).await;
}

/// The auth middleware function.
///
/// - If neither a master key nor JWT validation is configured **and** no
///   database with virtual keys is configured, auth is skipped (open mode).
///   When a database is configured, virtual keys are expected even without a
///   master key/JWT, so open mode is never entered in that case.
/// - Extracts credentials from (in order):
/// 1. `Authorization: Bearer <key>` (OpenAI SDK / curl)
/// 2. `x-api-key: <key>` (Anthropic SDK)
/// 3. `?authorization=` query (WebSocket / realtime fallback)
/// - If the key matches the master key (constant-time), authenticates as master.
/// - Otherwise, when a database is configured, looks up the key hash against
///   virtual keys (with DB-versioned 5-minute snapshots). Inactive keys are rejected.
/// - On any failure, returns a 401 with the OpenAI error envelope.
///
/// Credential sources, in order: `Authorization: Bearer <key>`, then
/// `x-api-key`/`api-key`. The `?authorization=` query parameter is accepted
/// **only** on the `/v1/realtime` WebSocket route — browsers cannot set
/// headers during a WS upgrade. Accepting it elsewhere would leak API keys
/// into access logs (#34).
pub async fn auth_middleware(
    State(shared): State<SharedState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let state = shared.load_full();
    if state.master_key_hash.is_none() && state.jwt_validator.is_none() && state.db.is_none() {
        let mut req = request;
        req.extensions_mut().insert(AuthUser::open());
        return next.run(req).await;
    }

    // Backstop brake check. Every production route stack that mounts this
    // middleware also mounts the outer `auth_brake_guard`
    // (routes/middleware.rs, routes/management.rs), which is the authoritative
    // gate and returns 429 before this middleware runs. This also protects
    // middleware wired standalone (e.g. tests), so a custom stack never falls back to an
    // unthrottled auth loop. See [`brake_entry_response`].
    if let Some(blocked) = brake_entry_response(&state, &headers, request.extensions()).await {
        return blocked;
    }
    let token = extract_auth_token(&headers, request.uri());
    let Some(token) = token else {
        // No credentials at all — return 401 WITHOUT recording a failure.
        // Counting missing-credential requests would let any unauthenticated
        // probe inflate the auth-brake counter (e.g. health checks, browsers
        // sending a preflight without auth). Only an attempt that presents a
        // token which fails to match is a real auth failure worth braking.
        return ApiError::unauthorized().into_response();
    };
    let provided_hash = hash_key(token.as_bytes());

    if let Some(expected_hash) = &state.master_key_hash {
        if provided_hash.ct_eq(expected_hash).into() {
            let mut req = request;
            req.extensions_mut().insert(AuthUser::master(provided_hash));
            return next.run(req).await;
        }
    }

    #[cfg(feature = "db")]
    {
        // Derive the hex from the hash already computed above instead of
        // re-hashing the token inside `lookup_virtual_key` (the previous path
        // SHA-256'd the same token twice on every virtual-key auth).
        let provided_hash_hex = bytes_to_hex(&provided_hash);
        match lookup_virtual_key(&state, &provided_hash_hex, provided_hash).await {
            LookupOutcome::Found(user) => {
                let mut req = request;
                req.extensions_mut().insert(*user);
                return next.run(req).await;
            }
            LookupOutcome::Invalid => {}
            LookupOutcome::ServerError => {
                // A server-side DB failure is not an authentication failure:
                // the credential may be perfectly valid, we just could not
                // verify it. Return 503 (mirroring `apply_customer_context`)
                // so load balancers retry / alert, and do NOT record a brake
                // failure — clients retrying through a DB flap must not be
                // 429-blocked on top of the outage.
                return ApiError::upstream(503, "unable to verify api key; try again")
                    .into_response();
            }
        }
    }

    match jwt::claims_to_user_if_valid(&state.jwt_validator, &token).await {
        Err(()) => {
            // The jti replay cache (shared Redis) was unreachable while
            // `jwt_replay_protection` is enabled. Fail closed on admission,
            // but with the same doctrine as the DB-outage paths above: a 503
            // server error, NOT a 401, and no auth-brake failure recorded —
            // the token may be perfectly valid, and clients retrying through
            // the cache flap must not be 429-blocked on top of the outage.
            return ApiError::upstream(503, "unable to verify jwt; try again").into_response();
        }
        Ok(None) => {}
        Ok(Some(user)) => {
            #[cfg(feature = "db")]
            let user = match jwt::enrich_jwt_user_with_db(&state, user).await {
                Ok(Some(user)) => user,
                Ok(None) => {
                    record_failure(&state, &headers, request.extensions()).await;
                    return ApiError::unauthorized().into_response();
                }
                Err(()) => {
                    // Same doctrine as the virtual-key path above: a DB outage
                    // during JWT enrichment is a 503, not an auth failure, and
                    // must not arm the brake.
                    return ApiError::upstream(503, "unable to verify jwt limits; try again")
                        .into_response();
                }
            };
            let mut req = request;
            req.extensions_mut().insert(user);
            return next.run(req).await;
        }
    }

    record_failure(&state, &headers, request.extensions()).await;
    ApiError::unauthorized().into_response()
}

/// Resolve the per-client IP used for auth-brake enforcement, or `None` when
/// no connection peer is available or the peer is a shared private/proxy
/// address that cannot attribute the request to a real client.
///
/// Delegates to [`crate::security::auth_brake_ip`], which returns `None` for
/// the peerless case (no `ConnectInfo` — e.g. the server was not started with
/// `into_make_service_with_connect_info`) and for an untrusted peer in a
/// private / shared range. In both cases keying the brake would collapse many
/// distinct clients into one bucket (`"unknown"`, or the reverse proxy's own
/// address when `trusted_proxies` is unconfigured) — a global self-DoS where
/// a single client trips the threshold and blocks everyone. Returning `None`
/// lets callers skip enforcement entirely.
fn brake_ip(
    headers: &HeaderMap,
    extensions: &axum::http::Extensions,
    trusted_proxies: &[ipnet::IpNet],
) -> Option<String> {
    let peer = extensions
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    crate::security::auth_brake_ip(headers, peer, trusted_proxies)
}

async fn record_failure(
    state: &crate::state::AppState,
    headers: &HeaderMap,
    extensions: &axum::http::Extensions,
) {
    if let Some(brake) = &state.auth_brake {
        if let Some(ip) = brake_ip(headers, extensions, &state.trusted_proxies) {
            brake.record_failure(&ip).await;
        }
    }
}

/// Defensive auth-brake entry backstop. Every production route stack that
/// mounts an auth middleware also mounts the outer `auth_brake_guard`
/// (routes/middleware.rs, routes/management.rs), which is the authoritative
/// gate and returns 429 before the middleware runs. This helper also protects
/// middleware wired standalone (e.g. tests, custom stacks), so a custom stack
/// never falls back to an unthrottled auth loop. Its skip diagnostic shares
/// the outer guard's one-shot gate to avoid duplicate warnings. It is
/// shared by [`auth_middleware`] and [`management_auth_middleware`] so the two
/// cannot diverge.
///
/// Returns `Some(429 response)` when the caller's IP is currently blocked
/// (caller must short-circuit); `None` when the request may proceed. A
/// peerless / untrusted-shared request (no attributable IP) skips enforcement
/// to avoid a shared "unknown" bucket self-DoS (see [`brake_ip`]).
async fn brake_entry_response(
    state: &crate::state::AppState,
    headers: &HeaderMap,
    extensions: &axum::http::Extensions,
) -> Option<Response> {
    let brake = state.auth_brake.as_ref()?;
    let ip = match brake_ip(headers, extensions, &state.trusted_proxies) {
        Some(ip) => ip,
        None => {
            // A peer may exist but be a shared/proxy address. Describe failed
            // attribution accurately and share the outer guard's warning gate.
            let peer = extensions
                .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                .map(|ci| ci.0);
            crate::security::log_auth_brake_skip(peer, &state.trusted_proxies);
            return None;
        }
    };
    if brake.is_blocked(&ip).await {
        tracing::warn!(ip = %ip, "auth brake: IP blocked");
        let mut err = ApiError::rate_limited("too many auth failures; blocked").into_response();
        // Report the *remaining* block time, not the full window.
        // See `auth_brake_guard` in `security.rs` for the rationale.
        let retry_after = brake
            .remaining_block_secs(&ip)
            .await
            .unwrap_or_else(|| brake.block_secs());
        if let Ok(v) = axum::http::HeaderValue::from_str(&retry_after.to_string()) {
            err.headers_mut().insert("retry-after", v);
        }
        Some(err)
    } else {
        None
    }
}

/// Management-endpoint auth: only the configured master key is accepted.
pub async fn management_auth_middleware(
    State(shared): State<SharedState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    let state = shared.load_full();
    // Same defensive auth-brake entry backstop as `auth_middleware`: when this
    // middleware is wired standalone (no outer `auth_brake_guard`), a blocked IP
    // must still be 429'd instead of being allowed to hammer management
    // endpoints. Production stacks mount the guard, so this is belt-and-braces.
    if let Some(blocked) = brake_entry_response(&state, &headers, request.extensions()).await {
        return blocked;
    }
    let Some(expected_hash) = &state.master_key_hash else {
        return ApiError::unauthorized().into_response();
    };
    let Some(token) = extract_bearer_token(&headers).or_else(|| extract_api_key_header(&headers))
    else {
        // No credentials at all — return 401 WITHOUT recording a failure.
        // Counting missing-credential requests would let any keyless probe
        // (monitoring agents, scanners) inflate the auth-brake counter,
        // pushing a shared IP toward the block threshold. Only an attempt
        // that presents a token which fails to match is a real auth failure
        // worth braking — matching the LLM `auth_middleware` path.
        return ApiError::unauthorized().into_response();
    };
    let provided_hash = hash_key(token.as_bytes());
    if provided_hash.ct_eq(expected_hash).into() {
        let mut req = request;
        req.extensions_mut().insert(AuthUser::master(provided_hash));
        next.run(req).await
    } else {
        record_failure(&state, &headers, request.extensions()).await;
        ApiError::unauthorized().into_response()
    }
}

/// Resolve a virtual key with DB-versioned 5-minute positive snapshots and
/// 60-second negative snapshots, returning
/// [`LookupOutcome::Found`] only when the key exists and is active.
///
/// Genuinely invalid credentials (unknown, expired, inactive, blocked team)
/// yield [`LookupOutcome::Invalid`]; backing-store failures yield
/// [`LookupOutcome::ServerError`] so the middleware can answer 503 without
/// arming the auth brake (see [`LookupOutcome`]).
///
/// Team and user records are resolved on every request via their own short
/// caches (`vk_team:{id}` / `vk_user:{id}`). Team/user limit updates therefore
/// invalidate only their own entries, not every virtual-key snapshot. Missing
/// key/team/user rows are tombstoned briefly. Every snapshot (including an
/// absence) requires a fresh DB version check before it can be reused.
///
/// # Cross-replica invalidation
///
/// The shared DB is the version authority, including when Redis or pub/sub is
/// unavailable. Management mutations call [`invalidate_cache_key`] after
/// commit to evict and broadcast just the affected snapshot. A stale local
/// DualCache entry, late fill, or missed broadcast cannot bypass the DB version
/// check. Cache errors fall back to the DB; DB errors fail closed as 503.
#[cfg(feature = "db")]
async fn lookup_virtual_key(
    state: &crate::state::AppState,
    hash_hex: &str,
    provided_hash: [u8; 32],
) -> LookupOutcome {
    let Some(db) = &state.db else {
        return LookupOutcome::Invalid;
    };
    let cache_key = format!("vk:{}", hash_hex);

    let info = match cache::load(
        state,
        &cache_key,
        || db.get_key_by_hash(hash_hex),
        |record: oxllm_db::KeyRecord| {
            let info = VirtualKeyInfo {
                key_id: record.key_id,
                team_id: record.team_id,
                user_id: record.user_id,
                org_id: record.org_id,
                project_id: record.project_id,
                max_budget: record.max_budget,
                budget_duration: record.budget_duration,
                rpm: record.rpm,
                tpm: record.tpm,
                expires_at: record.expires_at,
                is_active: record.is_active,
                key_models: parse_models_column(record.models.as_deref()),
                allowed_cache_controls: parse_cache_controls_column(
                    record.allowed_cache_controls.as_deref(),
                ),
                ..Default::default()
            };
            virtual_key_valid(&info).then_some(info)
        },
    )
    .await
    {
        Ok(Some(info)) if virtual_key_valid(&info) => info,
        Ok(_) => return LookupOutcome::Invalid,
        Err(()) => return LookupOutcome::ServerError,
    };

    // H18: fan out the team/user/org/project limit lookups concurrently.
    // Each is an independent cache-or-DB read, so running them in parallel
    // removes 3 round-trips of latency when scopes are configured.
    let (team_limits, user_limits, org_limits, project_limits) = tokio::join!(
        async {
            match info.team_id.as_deref() {
                Some(team_id) => team_limits_for(state, db, team_id).await,
                None => Ok(None),
            }
        },
        async {
            match info.user_id.as_deref() {
                Some(user_id) => user_limits_for(state, db, user_id).await,
                None => Ok(None),
            }
        },
        async {
            match info.org_id.as_deref() {
                Some(org_id) => org_limits_for(state, db, org_id).await,
                None => Ok(None),
            }
        },
        async {
            match info.project_id.as_deref() {
                Some(project_id) => project_limits_for(state, db, project_id).await,
                None => Ok(None),
            }
        },
    );

    // Fail closed on scope-limit DB errors: `team_limits_for` & co. return
    // `Err` when their backing read fails, and swallowing it here would merge
    // as "no ceiling" — silently disabling the team/org/project/user budget
    // and RPM/TPM caps for the error window. The key-level DB read on the same
    // request already fails closed, so a scope read must too. The failure is
    // surfaced as `ServerError` (503), not a 401: the credential was fine.
    let team_limits = match team_limits {
        Ok(limits) => limits,
        Err(()) => return scope_lookup_failed(&info, "team"),
    };
    // A blocked team rejects every key in it, regardless of the key's own
    // `is_active` flag.
    // This is separate from per-key `/key/block` so `/team/unblock` does not
    // re-enable individually-compromised keys. Fail closed (reject).
    //
    // The key's own `vk:{hash}` entry is intentionally NOT tombstoned here.
    // `/team/unblock` (handlers/management/teams.rs) only invalidates the
    // `vk_team:{id}` limits cache. A block is a team policy, not an invalid key
    // row: keep the base key snapshot independent so unblock only reloads the
    // team policy. Its DB content version also changes if eviction/broadcast
    // fails, so a blocked or unblocked team never waits for a snapshot TTL.
    if let Some(ref tl) = team_limits {
        if tl.blocked {
            tracing::info!(
                key_id = %info.key_id,
                team_id = %info.team_id.as_deref().unwrap_or(""),
                "key rejected: team is blocked"
            );
            return LookupOutcome::Invalid;
        }
    }
    let user_limits = match user_limits {
        Ok(limits) => limits,
        Err(()) => return scope_lookup_failed(&info, "user"),
    };
    let org_limits = match org_limits {
        Ok(limits) => limits,
        Err(()) => return scope_lookup_failed(&info, "org"),
    };
    let project_limits = match project_limits {
        Ok(limits) => limits,
        Err(()) => return scope_lookup_failed(&info, "project"),
    };
    if (info.team_id.is_some() && team_limits.is_none())
        || (info.user_id.is_some() && user_limits.is_none())
        || (info.org_id.is_some() && org_limits.is_none())
        || (info.project_id.is_some() && project_limits.is_none())
    {
        return LookupOutcome::Invalid;
    }

    let mut info = info;
    info.team_max_budget = team_limits.as_ref().and_then(|t| t.max_budget);
    info.team_budget_duration = team_limits.as_ref().and_then(|t| t.budget_duration.clone());
    info.team_rpm = team_limits.as_ref().and_then(|t| t.rpm);
    info.team_tpm = team_limits.as_ref().and_then(|t| t.tpm);
    info.org_max_budget = org_limits.as_ref().and_then(|o| o.max_budget);
    info.org_rpm = org_limits.as_ref().and_then(|o| o.rpm);
    info.org_tpm = org_limits.as_ref().and_then(|o| o.tpm);
    info.org_budget_duration = org_limits.as_ref().and_then(|o| o.budget_duration.clone());
    info.project_max_budget = project_limits.as_ref().and_then(|p| p.max_budget);
    info.project_rpm = project_limits.as_ref().and_then(|p| p.rpm);
    info.project_tpm = project_limits.as_ref().and_then(|p| p.tpm);
    info.project_budget_duration = project_limits
        .as_ref()
        .and_then(|p| p.budget_duration.clone());
    info.user_max_budget = user_limits.as_ref().and_then(|u| u.max_budget);
    info.user_budget_duration = user_limits.as_ref().and_then(|u| u.budget_duration.clone());
    info.team_models = team_limits.as_ref().and_then(|t| t.models.clone());
    info.user_models = user_limits.as_ref().and_then(|u| u.models.clone());
    // Org concurrent-request gauge (budgets table is not key-linked yet).
    info.max_parallel_requests = org_limits.as_ref().and_then(|o| o.max_parallel_requests);
    // The key's `budget_duration` is its OWN spend-window TTL (counter
    // `spend:{key_hash}`) and must not inherit the team/user window. Each
    // scope governs its own counter independently (`spend:team:*`,
    // `spend:user:*`); feeding a parent scope's window into the key scope
    // would either reset the key counter too early (lost metering → under-
    // billing) or never (over-blocking) depending on which window won.
    // `info.budget_duration` was already populated from the key row above;
    // leave it untouched. See docs/architecture.md "Budget hierarchy".

    LookupOutcome::Found(Box::new(AuthUser::virtual_key(provided_hash, info)))
}

/// Log a scope-limit DB failure and fail the request closed with a server
/// error (503, brake untouched). Billing-control scope caps must never
/// silently degrade to "unlimited", and a backing-store outage is not an
/// authentication failure.
#[cfg(feature = "db")]
fn scope_lookup_failed(info: &VirtualKeyInfo, scope: &str) -> LookupOutcome {
    tracing::warn!(
        scope,
        key_id = %info.key_id,
        "scope-limit DB lookup failed; failing closed (rejecting this request)"
    );
    LookupOutcome::ServerError
}

#[cfg(feature = "db")]
async fn team_limits_for(
    state: &crate::state::AppState,
    db: &std::sync::Arc<oxllm_db::Database>,
    team_id: &str,
) -> Result<Option<TeamLimits>, ()> {
    cache::load(
        state,
        &format!("vk_team:{team_id}"),
        || db.get_team(team_id),
        |record: oxllm_db::TeamRecord| {
            Some(TeamLimits {
                max_budget: record.max_budget,
                budget_duration: record.budget_duration,
                rpm: record.rpm,
                tpm: record.tpm,
                models: parse_models_column(record.models.as_deref()),
                blocked: record.blocked,
            })
        },
    )
    .await
}

#[cfg(feature = "db")]
async fn user_limits_for(
    state: &crate::state::AppState,
    db: &std::sync::Arc<oxllm_db::Database>,
    user_id: &str,
) -> Result<Option<UserLimits>, ()> {
    cache::load(
        state,
        &format!("vk_user:{user_id}"),
        || db.get_user(user_id),
        |record: oxllm_db::UserRecord| {
            Some(UserLimits {
                max_budget: record.max_budget,
                budget_duration: record.budget_duration,
                models: parse_models_column(record.models.as_deref()),
            })
        },
    )
    .await
}

#[cfg(feature = "db")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct TeamLimits {
    max_budget: Option<f64>,
    budget_duration: Option<String>,
    rpm: Option<i64>,
    tpm: Option<i64>,
    /// JSON-decoded allowed-model patterns from the team record.
    models: Option<Vec<String>>,
    /// Whether the team is administratively blocked (`team.blocked`).
    /// `#[serde(default)]` so cached entries from before the `blocked` column
    /// was added deserialize as "not blocked".
    #[serde(default)]
    blocked: bool,
}

#[cfg(feature = "db")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct UserLimits {
    max_budget: Option<f64>,
    budget_duration: Option<String>,
    /// JSON-decoded allowed-model patterns from the user record.
    models: Option<Vec<String>>,
}

#[cfg(feature = "db")]
async fn org_limits_for(
    state: &crate::state::AppState,
    db: &std::sync::Arc<oxllm_db::Database>,
    org_id: &str,
) -> Result<Option<OrgLimits>, ()> {
    cache::load(
        state,
        &format!("vk_org:{org_id}"),
        || db.get_organization(org_id),
        |record: oxllm_db::OrganizationRecord| {
            Some(OrgLimits {
                max_budget: record.max_budget,
                budget_duration: record.budget_duration,
                rpm: record.rpm,
                tpm: record.tpm,
                max_parallel_requests: record.max_parallel_requests,
            })
        },
    )
    .await
}

#[cfg(feature = "db")]
async fn project_limits_for(
    state: &crate::state::AppState,
    db: &std::sync::Arc<oxllm_db::Database>,
    project_id: &str,
) -> Result<Option<ProjectLimits>, ()> {
    cache::load(
        state,
        &format!("vk_project:{project_id}"),
        || db.get_project(project_id),
        |record: oxllm_db::ProjectRecord| {
            Some(ProjectLimits {
                max_budget: record.max_budget,
                budget_duration: record.budget_duration,
                rpm: record.rpm,
                tpm: record.tpm,
            })
        },
    )
    .await
}

#[cfg(feature = "db")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct OrgLimits {
    max_budget: Option<f64>,
    budget_duration: Option<String>,
    rpm: Option<i64>,
    tpm: Option<i64>,
    /// Concurrent request gauge from the org record.
    #[serde(default)]
    max_parallel_requests: Option<i64>,
}

#[cfg(feature = "db")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct ProjectLimits {
    max_budget: Option<f64>,
    budget_duration: Option<String>,
    rpm: Option<i64>,
    tpm: Option<i64>,
}

/// Cached end-user / customer limits (request-body `"user"` field).
///
/// Unlike team/user/org/project, customer identity is request-scoped (from
/// the OpenAI body `user` field), so this cache is looked up from the
/// pipeline after body parse — not from `lookup_virtual_key`.
#[cfg(feature = "db")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct CustomerLimits {
    pub max_budget: Option<f64>,
    pub budget_duration: Option<String>,
    pub blocked: bool,
    /// Model patterns that this end-user must not call (`*` wildcards ok).
    #[serde(default)]
    pub model_blocklist: Option<Vec<String>>,
}

/// Cache key for customer limits: `vk_customer:{end_user_id}`.
#[cfg(feature = "db")]
pub fn customer_cache_key(end_user_id: &str) -> String {
    format!("vk_customer:{end_user_id}")
}

/// Resolve end-user / customer limits by id, with positive/negative cache.
///
/// Returns:
/// - `Ok(Some(limits))` when a customer row exists
/// - `Ok(None)` when no such customer (request passes without customer scope)
/// - `Err(())` on DB error (caller should fail closed — a blocked customer
///   must not slip through during a flaky-DB window)
// `Err(())` is the documented fail-closed sentinel for a DB outage.
#[cfg(feature = "db")]
#[allow(clippy::result_unit_err)]
pub async fn customer_limits_for(
    state: &crate::state::AppState,
    db: &std::sync::Arc<oxllm_db::Database>,
    end_user_id: &str,
) -> Result<Option<CustomerLimits>, ()> {
    cache::load(
        state,
        &customer_cache_key(end_user_id),
        || db.get_customer(end_user_id),
        |record: oxllm_db::CustomerRecord| {
            Some(CustomerLimits {
                max_budget: record.max_budget,
                budget_duration: record.budget_duration,
                blocked: record.blocked,
                model_blocklist: parse_models_column(record.model_blocklist.as_deref()),
            })
        },
    )
    .await
}

/// Cached limits for normalized request tags from every supported carrier.
///
/// Request-scoped like customers: looked up from the pipeline after the
/// carriers are parsed, not from `lookup_virtual_key`.
#[cfg(feature = "db")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct TagLimits {
    pub max_budget: Option<f64>,
}

/// Cache key for tag limits: `vk_tag:{name}`.
#[cfg(feature = "db")]
pub fn tag_cache_key(name: &str) -> String {
    format!("vk_tag:{name}")
}

/// Resolve tag limits by name, with positive/negative cache.
///
/// Returns:
/// - `Ok(Some(limits))` when a tag row exists
/// - `Ok(None)` when no such tag (request passes without tag budget scope)
/// - `Err(())` on DB error (caller should fail closed when a budget may apply)
// `Err(())` is the documented fail-closed sentinel for a DB outage.
#[cfg(feature = "db")]
#[allow(clippy::result_unit_err)]
pub async fn tag_limits_for(
    state: &crate::state::AppState,
    db: &std::sync::Arc<oxllm_db::Database>,
    name: &str,
) -> Result<Option<TagLimits>, ()> {
    cache::load(
        state,
        &tag_cache_key(name),
        || db.get_tag(name),
        |record: oxllm_db::TagRecord| {
            Some(TagLimits {
                max_budget: record.max_budget,
            })
        },
    )
    .await
}

/// A virtual key is valid only when it is active and not expired.
#[cfg(feature = "db")]
fn virtual_key_valid(info: &VirtualKeyInfo) -> bool {
    if !info.is_active {
        return false;
    }
    if is_expired(&info.expires_at) {
        return false;
    }
    true
}

/// Reserved fail-closed policy marker. ACL and blocklist consumers recognize
/// it as unconditional denial, never as an ordinary model pattern.
pub(crate) const MODELS_ACL_DENY_ALL: &str = "\0oxllm-acl-deny-all";

/// Decode the `models` JSON column (`["gpt-4o", "claude-*"]`) into a vec.
///
/// Returns `None` when the column is missing or empty so the caller treats it
/// as "no restriction". On parse failure, fails **closed**: returns a
/// non-empty deny-all sentinel so corrupt ACL rows never grant unrestricted
/// model access.
#[cfg(feature = "db")]
pub(crate) fn parse_models_column(raw: Option<&str>) -> Option<Vec<String>> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    match serde_json::from_str::<Vec<String>>(raw) {
        Ok(v) if !v.is_empty() => Some(v),
        Ok(_) => None,
        Err(e) => {
            tracing::error!(
                error = %e,
                raw = raw,
                "failed to parse models column; denying all models (fail-closed)"
            );
            Some(vec![MODELS_ACL_DENY_ALL.to_string()])
        }
    }
}

/// Returns `true` when `expires_at` is set and parses to a timestamp in the
/// past. Returns `false` only when `expires_at` is unset (no expiry). When
/// `expires_at` is set but unparseable, returns `true` (fail closed) — a
/// malformed expiry is treated as expired so the key is rejected rather than
/// allowed indefinitely.
#[cfg(feature = "db")]
fn is_expired(expires_at: &Option<String>) -> bool {
    let Some(exp) = expires_at.as_deref() else {
        return false;
    };
    match exp.parse::<jiff::Timestamp>() {
        Ok(ts) => jiff::Timestamp::now() > ts,
        Err(e) => {
            tracing::warn!(
            expires_at = exp,
            error = %e,
            "virtual key has unparseable expires_at; treating as expired"
            );
            true
        }
    }
}

/// Extract the auth token from a request using the standard credential
/// sources: `Authorization: Bearer` header, then `x-api-key`/`api-key`.
///
/// The `?authorization=` query parameter is consulted **only** when `path`
/// refers to the realtime WebSocket route (`/v1/realtime`). Browser WebSocket
/// clients cannot set headers during the initial HTTP upgrade, so the query
/// param is the only transport for the credential in that case. Accepting it
/// on any other route would cause API keys to leak into access logs (#34).
fn extract_auth_token(headers: &HeaderMap, uri: &axum::http::Uri) -> Option<String> {
    let token = extract_bearer_token(headers).or_else(|| extract_api_key_header(headers));
    if token.is_none() && is_realtime_path(uri.path()) {
        extract_query_authorization(uri)
    } else {
        token
    }
}

/// Whether the request path targets the realtime WebSocket route.
///
/// Matches `/v1/realtime` exactly or any sub-path like `/v1/realtime/{model}`.
/// Does **not** match unrelated paths like `/v1/realtime-foo` (which a naive
/// `starts_with` check would incorrectly accept).
///
/// Paths containing dot-segments (`.`/`..`) never qualify, so a crafted URI
/// like `/v1/realtime/../chat/completions?authorization=…` cannot trick this
/// check into extracting the `authorization` query param (which would leak
/// the credential into access logs on the rerouted non-realtime path).
fn is_realtime_path(path: &str) -> bool {
    if path.split('/').any(|seg| seg == ".." || seg == ".") {
        return false;
    }
    path == "/v1/realtime" || path.starts_with("/v1/realtime/")
}

/// Extract the bearer token from the `Authorization` header.
///
/// Scheme matching is case-insensitive per RFC 7235 (`Bearer`, `bearer`,
/// `BEARER`, …). Per RFC 6750 §2.1 the credential is a single token; values
/// with trailing whitespace-delimited content (e.g. `"Bearer sk-test extra"`)
/// are rejected.
fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let auth_value = headers.get(axum::http::header::AUTHORIZATION)?;
    let auth_str = auth_value.to_str().ok()?;
    let (scheme, token) = auth_str.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let token = token.trim();
    // Reject values containing internal whitespace: the b64token grammar
    // (RFC 6750 §2.1) does not allow it, and accepting it would let
    // misformed clients pass surprising values through the parser.
    if token.is_empty() || token.contains(char::is_whitespace) {
        return None;
    }
    Some(token.to_string())
}

/// Extract the token from Anthropic-style `x-api-key` (or `api-key`) headers.
/// Enables the official Anthropic SDK to authenticate when pointing
/// `base_url` at oxllm.
fn extract_api_key_header(headers: &HeaderMap) -> Option<String> {
    for name in ["x-api-key", "api-key"] {
        if let Some(value) = headers.get(name).and_then(|v| v.to_str().ok()) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Extract a token from the `authorization` query parameter. Used as a
/// fallback for WebSocket clients (e.g. the realtime endpoint) that cannot
/// set headers during the initial HTTP upgrade.
fn extract_query_authorization(uri: &axum::http::Uri) -> Option<String> {
    let query = uri.query()?;
    for pair in query.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        if key == "authorization" {
            let decoded = percent_decode(value);
            if !decoded.is_empty() {
                return Some(decoded);
            }
        }
    }
    None
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_val(bytes[i + 1]);
            let lo = hex_val(bytes[i + 2]);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        // RFC 3986: `+` is a literal sub-delimiter in a URI, not an encoded
        // space (that is `application/x-www-form-urlencoded` semantics, which
        // does not apply to bearer tokens). A standard base64 key or some
        // provider key formats contain `+`; converting it to a space silently
        // corrupts the credential and fails auth. `%2B` still decodes to `+`.
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_default()
}

fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let h1 = hash_key(b"sk-test");
        let h2 = hash_key(b"sk-test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_key_hex_is_lowercase_hex() {
        let hex = hash_key_hex(b"sk-test");
        assert_eq!(hex.len(), 64);
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn master_principal_flags_is_master() {
        let h = hash_key(b"sk-master");
        let user = AuthUser::master(h);
        assert!(user.is_master);
        assert!(user.key_id.is_none());
    }

    #[test]
    fn open_principal_is_not_master() {
        let user = AuthUser::open();
        assert!(!user.is_master);
    }

    #[test]
    fn virtual_key_carries_metadata() {
        let h = hash_key(b"sk-vk");
        let info = VirtualKeyInfo {
            key_id: "k-1".to_string(),
            team_id: Some("t-1".to_string()),
            max_budget: Some(5.0),
            budget_duration: Some("30d".to_string()),
            rpm: Some(100),
            tpm: Some(10_000),
            team_max_budget: Some(50.0),
            team_rpm: Some(500),
            is_active: true,
            allowed_cache_controls: Some(vec!["insert".to_string()]),
            ..Default::default()
        };
        let user = AuthUser::virtual_key(h, info);
        assert!(!user.is_master);
        assert_eq!(user.key_id.as_deref(), Some("k-1"));
        assert_eq!(user.team_id.as_deref(), Some("t-1"));
        assert_eq!(user.max_budget, Some(5.0));
        assert_eq!(user.rpm, Some(100));
        assert_eq!(
            user.allowed_cache_controls,
            Some(vec!["insert".to_string()])
        );
    }

    #[test]
    fn key_hash_hex_is_lowercase_hex() {
        let user = AuthUser::master(hash_key(b"sk-test"));
        let hex = user.key_hash_hex();
        assert_eq!(hex.len(), 64);
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn key_hash_hex_differs_for_different_keys() {
        let a = AuthUser::master(hash_key(b"sk-a"));
        let b = AuthUser::master(hash_key(b"sk-b"));
        assert_ne!(a.key_hash_hex(), b.key_hash_hex());
    }

    #[test]
    fn hash_differs_for_different_keys() {
        let h1 = hash_key(b"sk-test");
        let h2 = hash_key(b"sk-other");
        assert_ne!(h1, h2);
    }

    #[test]
    fn constant_time_compare_equal() {
        let h1 = hash_key(b"sk-master");
        let h2 = hash_key(b"sk-master");
        assert!(bool::from(h1.ct_eq(&h2)));
    }

    #[test]
    fn constant_time_compare_not_equal() {
        let h1 = hash_key(b"sk-master");
        let h2 = hash_key(b"sk-attacker");
        assert!(!bool::from(h1.ct_eq(&h2)));
    }

    #[test]
    fn extract_bearer_token_valid() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-test123".parse().unwrap());
        let token = extract_bearer_token(&headers);
        assert_eq!(token.as_deref(), Some("sk-test123"));
    }

    #[test]
    fn extract_api_key_header_anthropic_style() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-ant-style".parse().unwrap());
        assert_eq!(
            extract_api_key_header(&headers).as_deref(),
            Some("sk-ant-style")
        );
    }

    #[test]
    fn extract_api_key_header_prefers_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "from-x".parse().unwrap());
        headers.insert("api-key", "from-api".parse().unwrap());
        assert_eq!(extract_api_key_header(&headers).as_deref(), Some("from-x"));
    }

    #[test]
    fn extract_bearer_token_case_insensitive_scheme() {
        for value in [
            "Bearer sk-mixed",
            "bearer sk-mixed",
            "BEARER sk-mixed",
            "BeArEr sk-mixed",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", value.parse().unwrap());
            let token = extract_bearer_token(&headers);
            assert_eq!(
                token.as_deref(),
                Some("sk-mixed"),
                "failed for Authorization: {value}"
            );
        }
    }

    #[test]
    fn extract_bearer_token_missing_header() {
        let headers = HeaderMap::new();
        assert!(extract_bearer_token(&headers).is_none());
    }

    #[test]
    fn extract_bearer_token_wrong_scheme() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Basic abc".parse().unwrap());
        assert!(extract_bearer_token(&headers).is_none());
    }

    #[test]
    fn extract_bearer_token_empty_token() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer ".parse().unwrap());
        assert!(extract_bearer_token(&headers).is_none());
    }

    #[test]
    fn extract_query_authorization_present() {
        let uri: axum::http::Uri = "/v1/realtime?model=gpt-4o-realtime&authorization=sk-test"
            .parse()
            .unwrap();
        assert_eq!(
            extract_query_authorization(&uri).as_deref(),
            Some("sk-test")
        );
    }

    #[test]
    fn extract_query_authorization_first_param() {
        let uri: axum::http::Uri = "/v1/realtime?authorization=sk-first&model=gpt-4o-realtime"
            .parse()
            .unwrap();
        assert_eq!(
            extract_query_authorization(&uri).as_deref(),
            Some("sk-first")
        );
    }

    #[test]
    fn extract_query_authorization_absent() {
        let uri: axum::http::Uri = "/v1/realtime?model=gpt-4o-realtime".parse().unwrap();
        assert!(extract_query_authorization(&uri).is_none());
    }

    #[test]
    fn extract_query_authorization_no_query() {
        let uri: axum::http::Uri = "/v1/realtime".parse().unwrap();
        assert!(extract_query_authorization(&uri).is_none());
    }

    /// A bare `?authorization` (no `=`) is not a key/value pair and must not
    /// be mis-parsed as a token.
    #[test]
    fn extract_query_authorization_bare_key_ignored() {
        let uri: axum::http::Uri = "/v1/realtime?authorization&model=gpt-4o".parse().unwrap();
        assert!(extract_query_authorization(&uri).is_none());
    }

    #[test]
    fn extract_query_authorization_percent_decoded() {
        let uri: axum::http::Uri = "/v1/realtime?authorization=sk%2Dtest".parse().unwrap();
        assert_eq!(
            extract_query_authorization(&uri).as_deref(),
            Some("sk-test")
        );
    }

    #[test]
    fn extract_query_authorization_empty_value() {
        let uri: axum::http::Uri = "/v1/realtime?authorization=".parse().unwrap();
        assert!(extract_query_authorization(&uri).is_none());
    }

    /// A literal `+` in the `authorization` query param must pass through
    /// unchanged (RFC 3986 treats `+` as a sub-delimiter, not an encoded
    /// space). Standard base64 keys and some provider key formats contain `+`;
    /// the previous form-urlencoded decoding converted it to a space, silently
    /// corrupting the credential. `%2B` is the percent-encoded form of `+`.
    #[test]
    fn extract_query_authorization_plus_is_literal() {
        let uri: axum::http::Uri = "/v1/realtime?authorization=sk+plus".parse().unwrap();
        assert_eq!(
            extract_query_authorization(&uri).as_deref(),
            Some("sk+plus"),
            "literal '+' must pass through, not become a space"
        );
        let uri: axum::http::Uri = "/v1/realtime?authorization=sk%2Bplus".parse().unwrap();
        assert_eq!(
            extract_query_authorization(&uri).as_deref(),
            Some("sk+plus"),
            "%2B must decode to '+'"
        );
    }

    // --- extract_auth_token: realtime-only query-param fallback (#34) -----

    #[test]
    fn extract_auth_token_query_param_rejected_on_chat_completions() {
        // An API key in the URL must NOT authenticate on non-realtime routes.
        // Such keys leak into access logs. See #34.
        let headers = HeaderMap::new();
        let uri: axum::http::Uri = "/v1/chat/completions?authorization=sk-test"
            .parse()
            .unwrap();
        assert!(
            extract_auth_token(&headers, &uri).is_none(),
            "query-param auth must not be accepted on /v1/chat/completions"
        );
    }

    #[test]
    fn extract_auth_token_query_param_rejected_on_other_routes() {
        let headers = HeaderMap::new();
        // `/v1/realtime-foo` must NOT be accepted — it is not a realtime route.
        // Only `/v1/realtime` and its sub-paths (e.g. `/v1/realtime/{model}`)
        // qualify for query-param auth.
        for path in [
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/models",
            "/realtime",
            "/v1/real",
            "/v1/realtime-foo",
        ] {
            let uri: axum::http::Uri = format!("{path}?authorization=sk-test").parse().unwrap();
            assert!(
                extract_auth_token(&headers, &uri).is_none(),
                "query-param auth must not be accepted on {path}"
            );
        }
    }

    #[test]
    fn extract_auth_token_query_param_accepted_on_realtime() {
        // The realtime WebSocket upgrade is the one route where browsers
        // cannot set headers, so the query param is the only transport.
        let headers = HeaderMap::new();
        let uri: axum::http::Uri = "/v1/realtime?model=gpt-4o-realtime&authorization=sk-test"
            .parse()
            .unwrap();
        assert_eq!(
            extract_auth_token(&headers, &uri).as_deref(),
            Some("sk-test")
        );
    }

    #[test]
    fn extract_auth_token_query_param_accepted_on_realtime_subpath() {
        // Any path under /v1/realtime (e.g. session-scoped routes) is allowed.
        let headers = HeaderMap::new();
        let uri: axum::http::Uri = "/v1/realtime/sess_abc?authorization=sk-test"
            .parse()
            .unwrap();
        assert_eq!(
            extract_auth_token(&headers, &uri).as_deref(),
            Some("sk-test")
        );
    }

    #[test]
    fn extract_auth_token_prefers_bearer_header_over_query() {
        // When both a Bearer header and a query param are present, the header
        // wins (and the query param is never consulted on non-realtime routes).
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-header".parse().unwrap());
        let uri: axum::http::Uri = "/v1/chat/completions?authorization=sk-query"
            .parse()
            .unwrap();
        assert_eq!(
            extract_auth_token(&headers, &uri).as_deref(),
            Some("sk-header")
        );
    }

    #[test]
    fn extract_auth_token_prefers_api_key_header_over_query_on_chat() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-header".parse().unwrap());
        let uri: axum::http::Uri = "/v1/chat/completions?authorization=sk-query"
            .parse()
            .unwrap();
        assert_eq!(
            extract_auth_token(&headers, &uri).as_deref(),
            Some("sk-header")
        );
    }

    #[test]
    fn extract_auth_token_returns_none_when_no_credentials() {
        let headers = HeaderMap::new();
        let uri: axum::http::Uri = "/v1/chat/completions".parse().unwrap();
        assert!(extract_auth_token(&headers, &uri).is_none());
    }

    #[test]
    fn extract_auth_token_realtime_without_query_returns_none() {
        let headers = HeaderMap::new();
        let uri: axum::http::Uri = "/v1/realtime".parse().unwrap();
        assert!(extract_auth_token(&headers, &uri).is_none());
    }

    /// M8: a request with NO auth header must return 401 WITHOUT recording an
    /// auth-brake failure. Previously the no-credentials path called
    /// `record_failure`, which let any unauthenticated probe (health checks,
    /// CORS preflight, scanners) inflate the per-IP failure counter and trip
    /// the brake. Only a presented-but-non-matching token should count.
    #[tokio::test]
    async fn missing_auth_header_does_not_record_failure() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        // A master_key is configured, so the state must carry an auth brake.
        let brake = {
            let state = shared.load_full();
            state
                .auth_brake
                .clone()
                .expect("auth_brake must be present when master_key is configured")
        };
        assert_eq!(
            brake.tracked_ips(),
            0,
            "no failures recorded before any request"
        );

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared.clone(),
                auth_middleware,
            ));

        // Request carries NO Authorization header and no ?api_key=.
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::UNAUTHORIZED,
            "missing credentials must yield 401"
        );

        // The core assertion: the brake must NOT have recorded a failure for
        // a no-credentials request. `tracked_ips` stays 0 because
        // `record_failure` was never called.
        assert_eq!(
            brake.tracked_ips(),
            0,
            "a request with no auth header must not increment the auth-brake counter"
        );

        // Sanity-check the positive path in the same fixture: a presented but
        // wrong token MUST record a failure, proving the brake still arms.
        // A real peer address is attached so `brake_ip` attributes the failure
        // (peerless requests now skip enforcement to avoid a shared "unknown"
        // bucket — see `auth_brake_skipped_when_no_peer_address`).
        let app2 = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));
        let peer: std::net::SocketAddr = "8.8.8.8:4321".parse().expect("peer");
        let req2 = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", "Bearer sk-definitely-wrong")
            .extension(axum::extract::ConnectInfo(peer))
            .body(Body::empty())
            .expect("request");
        let resp2 = app2.oneshot(req2).await.expect("response");
        assert_eq!(resp2.status(), axum::http::StatusCode::UNAUTHORIZED);
        assert!(
            brake.tracked_ips() >= 1,
            "a presented-but-wrong token must record a failure (got {})",
            brake.tracked_ips()
        );
    }

    /// A request with no `ConnectInfo` (peer address) must NOT record an
    /// auth-brake failure, even when a wrong token is presented. Without a
    /// peer, `client_ip` would resolve to the shared `"unknown"` bucket, so
    /// recording there would let any peerless client trip the brake for ALL
    /// peerless clients — a global self-DoS under misconfiguration (server
    /// not started with `into_make_service_with_connect_info`).
    #[tokio::test]
    async fn auth_brake_skipped_when_no_peer_address() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let brake = {
            let state = shared.load_full();
            state
                .auth_brake
                .clone()
                .expect("auth_brake must be present when master_key is configured")
        };

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        // Present a wrong token but attach NO ConnectInfo → peerless.
        for _ in 0..5 {
            let req = Request::builder()
                .method("GET")
                .uri("/healthz")
                .header("authorization", "Bearer sk-wrong")
                .body(Body::empty())
                .expect("request");
            let resp = app.clone().oneshot(req).await.expect("response");
            assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
        }

        // No peer → enforcement skipped → nothing tracked. The shared
        // "unknown" bucket is never populated, so a peerless client cannot
        // block other peerless clients.
        assert_eq!(
            brake.tracked_ips(),
            0,
            "peerless wrong-token requests must not populate the shared \"unknown\" bucket"
        );
    }

    /// The auth brake must not collapse into a global block behind an
    /// unconfigured reverse proxy. With no `trusted_proxy_cidrs` configured,
    /// every request behind a proxy resolves to the proxy's own (private)
    /// address; enforcing the brake there would let a single failing client
    /// block every other client behind that proxy for `block_secs`. The brake
    /// must refuse to record/enforce on an untrusted shared peer instead.
    #[tokio::test]
    async fn auth_brake_skipped_for_untrusted_shared_peer() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let brake = {
            let state = shared.load_full();
            state
                .auth_brake
                .clone()
                .expect("auth_brake must be present when master_key is configured")
        };

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared.clone(),
                crate::security::auth_brake_guard,
            ))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        // Reverse proxy in front (private peer), `trusted_proxy_cidrs` unset:
        // multiple distinct XFF clients share the proxy's address. Failures
        // must not populate a shared bucket.
        let peer: std::net::SocketAddr = "10.0.0.5:8080".parse().expect("peer");
        for xff in ["198.51.100.1", "198.51.100.2", "198.51.100.3"] {
            for _ in 0..5 {
                let req = Request::builder()
                    .method("GET")
                    .uri("/healthz")
                    .header("authorization", "Bearer sk-wrong")
                    .header("x-forwarded-for", xff)
                    .extension(axum::extract::ConnectInfo(peer))
                    .body(Body::empty())
                    .expect("request");
                let resp = app.clone().oneshot(req).await.expect("response");
                assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
            }
        }
        assert_eq!(
            brake.tracked_ips(),
            0,
            "failures behind an untrusted shared proxy must not populate a shared bucket"
        );
    }

    /// With a trusted proxy configured, distinct XFF client IPs sharing one
    /// peer address must get distinct brake buckets: 10 failures from one
    /// client must not 429 a different client behind the same proxy.
    #[tokio::test]
    async fn auth_brake_keeps_xff_clients_separate_when_proxy_trusted() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                trusted_proxy_cidrs: Some(vec!["10.0.0.0/8".to_string()]),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared.clone(),
                crate::security::auth_brake_guard,
            ))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let peer: std::net::SocketAddr = "10.0.0.5:8080".parse().expect("peer");
        let request = |xff: &str, bearer: &str| {
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .header("authorization", format!("Bearer {bearer}"))
                .header("x-forwarded-for", xff)
                .extension(axum::extract::ConnectInfo(peer))
                .body(Body::empty())
                .expect("request")
        };

        // Client A exhausts the failure budget.
        for _ in 0..10 {
            let resp = app
                .clone()
                .oneshot(request("198.51.100.1", "sk-wrong"))
                .await
                .expect("response");
            assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
        }
        let resp = app
            .clone()
            .oneshot(request("198.51.100.1", "sk-wrong"))
            .await
            .expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            "client A must be blocked after exhausting its own failure budget"
        );

        // Client B (same proxy peer, distinct XFF) is unaffected.
        let resp = app
            .clone()
            .oneshot(request("198.51.100.2", "sk-test-master-key"))
            .await
            .expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "a distinct XFF client behind the same proxy must not be blocked by another client's failures"
        );
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn open_mode_is_bypassed_when_db_configured() {
        // Regression: when a database (virtual keys) is configured but no
        // master key / JWT is, the middleware must NOT enter open mode — a
        // request without credentials must be rejected with 401 rather than
        // authenticated as an open principal.
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let db = fresh_db().await;
        // Config carries no master_key and no jwt_secret → master_key_hash and
        // jwt_validator are both None, but db.is_some().
        let shared = crate::app::build_shared_state_with_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            None,
            Some(db),
        )
        .expect("shared state");

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::UNAUTHORIZED,
            "db configured without master key/JWT must require auth (no open mode)"
        );
    }

    #[cfg(feature = "db")]
    async fn db_state(db: std::sync::Arc<oxllm_db::Database>) -> crate::state::AppState {
        crate::state::AppState::with_registry_cache_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
            std::sync::Arc::new(crate::cache::InMemoryCache::default()),
            None,
            Some(db),
        )
        .expect("state")
    }

    #[cfg(feature = "db")]
    async fn fresh_db() -> std::sync::Arc<oxllm_db::Database> {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");
        db
    }

    #[cfg(feature = "db")]
    fn expect_found(outcome: LookupOutcome) -> AuthUser {
        match outcome {
            LookupOutcome::Found(user) => *user,
            other => panic!("expected LookupOutcome::Found, got {other:?}"),
        }
    }

    #[cfg(feature = "db")]
    fn key_record(
        key_hash: &str,
        team_id: Option<&str>,
        user_id: Option<&str>,
    ) -> oxllm_db::KeyRecord {
        let now = oxllm_db::now_iso();
        oxllm_db::KeyRecord {
            key_id: format!("key-{}", key_hash.chars().take(8).collect::<String>()),
            key_hash: key_hash.to_string(),
            key_alias: None,
            team_id: team_id.map(str::to_string),
            user_id: user_id.map(str::to_string),
            org_id: None,
            project_id: None,
            max_budget: Some(10.0),
            budget_duration: Some("30d".to_string()),
            rpm: Some(100),
            tpm: Some(1000),
            created_at: now.clone(),
            updated_at: now,
            expires_at: None,
            is_active: true,
            models: None,
            allowed_cache_controls: None,
        }
    }

    #[cfg(feature = "db")]
    async fn cached_auth_value(
        state: &crate::state::AppState,
        cache_key: &str,
    ) -> Option<serde_json::Value> {
        let raw = state.cache.get(cache_key).await?;
        serde_json::from_str::<serde_json::Value>(&raw)
            .ok()?
            .get("value")
            .cloned()
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn missing_virtual_key_tombstone_is_versioned_and_replaced_on_create() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-missing";
        let hash = hash_key(token.as_bytes());
        let hex = hash_key_hex(token.as_bytes());
        assert!(matches!(
            lookup_virtual_key(&state, &hex, hash).await,
            LookupOutcome::Invalid
        ));
        assert_eq!(
            cached_auth_value(&state, &format!("vk:{hex}")).await,
            Some(serde_json::Value::Null)
        );
        db.create_key(&key_record(&hex, None, None))
            .await
            .expect("create key");
        assert!(
            matches!(
                lookup_virtual_key(&state, &hex, hash).await,
                LookupOutcome::Found(_)
            ),
            "the shared DB version must make a newly generated key reachable immediately"
        );
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn team_limits_resolve_outside_virtual_key_snapshot() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-team";
        let key_hash = hash_key_hex(token.as_bytes());
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-1".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(5.0),
            budget_duration: Some("1d".to_string()),
            rpm: Some(10),
            tpm: Some(100),
            created_at: now.clone(),
            updated_at: now.clone(),
            models: None,
            blocked: false,
        })
        .await
        .expect("team");
        db.create_key(&key_record(&key_hash, Some("team-1"), None))
            .await
            .expect("key");
        let user =
            expect_found(lookup_virtual_key(&state, &key_hash, hash_key(token.as_bytes())).await);
        assert_eq!(user.team_max_budget, Some(5.0));

        db.update_team(&oxllm_db::TeamRecord {
            team_id: "team-1".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(1.0),
            budget_duration: Some("1d".to_string()),
            rpm: Some(20),
            tpm: Some(200),
            created_at: now.clone(),
            updated_at: now,
            models: None,
            blocked: false,
        })
        .await
        .expect("update");
        invalidate_cache_key(&state, "vk_team:team-1").await;
        let user =
            expect_found(lookup_virtual_key(&state, &key_hash, hash_key(token.as_bytes())).await);
        assert_eq!(user.team_max_budget, Some(1.0));
        assert_eq!(user.team_rpm, Some(20));
    }

    /// A scope-limit DB error must fail the request closed. `team_limits_for`
    /// previously swallowed `get_team` errors as `None`, which `lookup_virtual_key`
    /// merged into "no ceiling" — silently disabling the team budget/RPM/TPM
    /// caps for the error window while the key-level read on the same request
    /// failed closed. Regression: an `Err` from a scope read must reject the
    /// request (Err → 503), never produce an `AuthUser` with the caps absent.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn scope_limit_db_error_fails_closed() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-err".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(5.0),
            budget_duration: None,
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now,
            models: None,
            blocked: false,
        })
        .await
        .expect("team");
        assert!(matches!(
            team_limits_for(&state, &db, "team-err").await,
            Ok(Some(_))
        ));
        // A warm scope snapshot is insufficient when the version authority is
        // unreachable: its policy may have changed on another replica.
        db.pool().close().await;
        assert!(
            matches!(team_limits_for(&state, &db, "team-err").await, Err(())),
            "a scope-limit DB error must fail closed as a server error, not degrade to unlimited"
        );
    }

    /// MEDIUM: a server-side DB failure during virtual-key verification must
    /// surface as 503 (not a misleading 401 `invalid_api_key`) and must NOT
    /// count toward the per-IP auth brake. Previously `lookup_virtual_key`
    /// collapsed "DB read failed" into `None`, so a transient DB outage on a
    /// cache-cold valid key produced 401 **and** armed the brake — legitimate
    /// clients retrying through the flap could exhaust the failure budget and
    /// get 429-blocked, converting an infra outage into client lockout.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn db_error_on_virtual_key_returns_503_and_does_not_arm_brake() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let db = fresh_db().await;
        let shared = crate::app::build_shared_state_with_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: crate::config::types::GeneralSettings {
                    master_key: Some("sk-test-master-key".to_string()),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            None,
            Some(db.clone()),
        )
        .expect("shared state");

        let brake = {
            let state = shared.load_full();
            state
                .auth_brake
                .clone()
                .expect("auth_brake must be present when master_key is configured")
        };

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        // Warm the negative path first, while the DB is up: a genuinely wrong
        // token from peer A must 401 and arm the brake (proving the brake
        // still functions for real auth failures after this fix).
        let peer_a: std::net::SocketAddr = "8.8.8.8:4321".parse().expect("peer");
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", "Bearer sk-definitely-wrong")
            .extension(axum::extract::ConnectInfo(peer_a))
            .body(Body::empty())
            .expect("request");
        let resp = app.clone().oneshot(req).await.expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
        assert_eq!(brake.tracked_ips(), 1, "the wrong token must arm the brake");

        // Seed a VALID key row, then poison the pool so the cache-cold lookup
        // for it fails with a connection error.
        let token = "sk-db-outage";
        let hex = hash_key_hex(token.as_bytes());
        db.create_key(&key_record(&hex, None, None))
            .await
            .expect("create key");
        db.pool().close().await;

        // Same-shaped request from a distinct peer B with the VALID key: the
        // failure is server-side, so it must be a 503 and must not add peer B
        // to the brake (the count stays at A's genuine failure).
        let peer_b: std::net::SocketAddr = "9.9.9.9:4321".parse().expect("peer");
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", format!("Bearer {token}"))
            .extension(axum::extract::ConnectInfo(peer_b))
            .body(Body::empty())
            .expect("request");
        let resp = app.clone().oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "a DB outage during virtual-key verification must yield 503, not 401"
        );
        assert_eq!(
            brake.tracked_ips(),
            1,
            "a server-side DB failure must not count toward the auth brake"
        );
        assert!(
            !brake.is_blocked("9.9.9.9").await,
            "peer B must not be blocked by an infra outage"
        );

        // The 1s DB-error tombstone keeps serving 503 (not a misleading 401)
        // for repeat presentations within the blip window.
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", format!("Bearer {token}"))
            .extension(axum::extract::ConnectInfo(peer_b))
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "cached DB-error tombstone must keep yielding 503"
        );
        assert_eq!(
            brake.tracked_ips(),
            1,
            "tombstone-served 503s must not count toward the auth brake"
        );
    }

    /// A virtual key whose `expires_at` elapses while its positive-cache entry
    /// is live must be rejected on the cached path: `lookup_virtual_key`
    /// re-checks `virtual_key_valid` on every cached hit, so an expired key
    /// cannot keep authenticating until the 300s cache TTL expires.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn cached_key_expiry_rechecked_on_cached_path() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-expiring";
        let key_hash = hash_key_hex(token.as_bytes());
        let mut record = key_record(&key_hash, None, None);
        record.expires_at = Some("2000-01-01T00:00:00Z".to_string());
        db.create_key(&record).await.expect("expired key");
        let info = VirtualKeyInfo {
            key_id: record.key_id.clone(),
            is_active: true,
            expires_at: record.expires_at.clone(),
            ..Default::default()
        };
        let version = hash_key_hex(&serde_json::to_vec(&Some(&record)).expect("serialize record"));
        state
            .cache
            .set(
                &format!("vk:{key_hash}"),
                serde_json::json!({"version":version,"value":info}).to_string(),
                Some(VK_POSITIVE_TTL),
            )
            .await;
        assert!(
            matches!(
                lookup_virtual_key(&state, &key_hash, hash_key(token.as_bytes())).await,
                LookupOutcome::Invalid
            ),
            "an expired key must be rejected even while its positive-cache entry is live"
        );
    }

    /// The key's `budget_duration` is its OWN spend-window TTL (counter
    /// `spend:{key_hash}`) and must NOT inherit the team/user window. Each
    /// scope governs its own counter (see docs/architecture.md "Budget
    /// hierarchy"); a key with `max_budget` but no `budget_duration` is
    /// intended as a lifetime/non-resetting cap and must stay that way even
    /// when the parent team defines a window.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn key_budget_duration_does_not_inherit_team_window() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-windowless";
        let key_hash = hash_key_hex(token.as_bytes());
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-win".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(5.0),
            budget_duration: Some("1d".to_string()),
            rpm: Some(10),
            tpm: Some(100),
            created_at: now.clone(),
            updated_at: now.clone(),
            models: None,
            blocked: false,
        })
        .await
        .expect("team");
        // Key has a budget cap but NO window of its own.
        db.create_key(&oxllm_db::KeyRecord {
            key_id: "key-windowless".to_string(),
            key_hash: key_hash.clone(),
            key_alias: None,
            team_id: Some("team-win".to_string()),
            user_id: None,
            org_id: None,
            project_id: None,
            max_budget: Some(10.0),
            budget_duration: None,
            rpm: Some(100),
            tpm: Some(1000),
            created_at: now.clone(),
            updated_at: now,
            expires_at: None,
            is_active: true,
            models: None,
            allowed_cache_controls: None,
        })
        .await
        .expect("key");
        let user =
            expect_found(lookup_virtual_key(&state, &key_hash, hash_key(token.as_bytes())).await);
        assert_eq!(
            user.budget_duration, None,
            "key with no budget_duration must not inherit the team's window"
        );
        assert_eq!(user.max_budget, Some(10.0));
        assert_eq!(user.team_max_budget, Some(5.0));
    }

    #[cfg(feature = "db")]
    #[test]
    fn is_expired_past_timestamp_is_expired() {
        assert!(is_expired(&Some("2000-01-01T00:00:00Z".to_string())));
    }

    #[cfg(feature = "db")]
    #[test]
    fn is_expired_future_timestamp_is_not_expired() {
        assert!(!is_expired(&Some("2999-01-01T00:00:00Z".to_string())));
    }

    #[cfg(feature = "db")]
    #[test]
    fn is_expired_none_is_not_expired() {
        assert!(!is_expired(&None));
    }

    #[cfg(feature = "db")]
    #[test]
    fn is_expired_unparseable_is_expired() {
        assert!(is_expired(&Some("not-a-date".to_string())));
    }

    #[cfg(feature = "db")]
    #[test]
    fn virtual_key_valid_rejects_inactive() {
        let info = VirtualKeyInfo {
            key_id: "k-1".to_string(),
            is_active: false,
            expires_at: None,
            ..Default::default()
        };
        assert!(!virtual_key_valid(&info));
    }

    #[cfg(feature = "db")]
    #[test]
    fn virtual_key_valid_rejects_expired() {
        let info = VirtualKeyInfo {
            key_id: "k-1".to_string(),
            is_active: true,
            expires_at: Some("2000-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        assert!(!virtual_key_valid(&info));
    }

    #[cfg(feature = "db")]
    #[test]
    fn virtual_key_valid_rejects_unparseable_expiry() {
        let info = VirtualKeyInfo {
            key_id: "k-1".to_string(),
            is_active: true,
            expires_at: Some("not-a-date".to_string()),
            ..Default::default()
        };
        assert!(!virtual_key_valid(&info));
    }

    #[cfg(feature = "db")]
    #[test]
    fn parse_models_column_missing_and_empty_are_unrestricted() {
        assert!(parse_models_column(None).is_none());
        assert!(parse_models_column(Some("")).is_none());
        assert!(parse_models_column(Some("   ")).is_none());
        assert!(parse_models_column(Some("[]")).is_none());
    }

    #[cfg(feature = "db")]
    #[test]
    fn parse_models_column_valid_list() {
        let parsed = parse_models_column(Some(r#"["gpt-4o","claude-*"]"#)).expect("parsed");
        assert_eq!(parsed, vec!["gpt-4o".to_string(), "claude-*".to_string()]);
    }

    /// Corrupt `models` JSON must fail closed (deny-all), never grant
    /// unrestricted model access.
    #[cfg(feature = "db")]
    #[test]
    fn parse_models_column_unparseable_fails_closed() {
        let denied = parse_models_column(Some("not-json")).expect("deny-all sentinel");
        assert_eq!(denied, vec![MODELS_ACL_DENY_ALL.to_string()]);
        // The ACL check treats the marker as an unconditional denial.
        let auth = AuthUser::open();
        let err = crate::pipeline::check_model_acl(&auth, "gpt-4o", Some(&denied), None, None);
        assert!(err.is_err(), "deny-all sentinel must block every model");
    }

    #[cfg(feature = "db")]
    #[test]
    fn virtual_key_valid_accepts_active_and_unexpired() {
        let info = VirtualKeyInfo {
            key_id: "k-1".to_string(),
            is_active: true,
            expires_at: None,
            ..Default::default()
        };
        assert!(virtual_key_valid(&info));
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn invalidate_cache_key_clears_local_cache() {
        let db = fresh_db().await;
        let state = db_state(db).await;
        state.cache.set("vk:abc", "payload".to_string(), None).await;
        invalidate_cache_key(&state, "vk:abc").await;
        assert!(
            state.cache.get("vk:abc").await.is_none(),
            "invalidate_cache_key must clear the local cache entry"
        );
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn invalidate_cache_key_clears_negative_cache_entry() {
        let db = fresh_db().await;
        let state = db_state(db).await;
        assert!(matches!(
            lookup_virtual_key(&state, "deadbeef", [0; 32]).await,
            LookupOutcome::Invalid
        ));
        assert!(state.cache.get("vk:deadbeef").await.is_some());
        invalidate_cache_key(&state, "vk:deadbeef").await;
        assert!(state.cache.get("vk:deadbeef").await.is_none());
    }

    // -----------------------------------------------------------------------
    // Twin-inspired edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn hash_key_empty_input() {
        let h = hash_key(b"");
        // SHA-256 of empty input is well-known:
        // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(h[0], 0xe3);
        assert_eq!(h[31], 0x55);
    }

    #[test]
    fn hash_key_hex_empty_input() {
        let hex = hash_key_hex(b"");
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn constant_time_compare_self_always_equal() {
        let h = hash_key(b"sk-any-key");
        assert!(bool::from(h.ct_eq(&h)));
    }

    #[test]
    fn hex_val_valid_digits() {
        assert_eq!(hex_val(b'0'), Some(0));
        assert_eq!(hex_val(b'9'), Some(9));
        assert_eq!(hex_val(b'a'), Some(10));
        assert_eq!(hex_val(b'f'), Some(15));
        assert_eq!(hex_val(b'A'), Some(10));
        assert_eq!(hex_val(b'F'), Some(15));
    }

    #[test]
    fn hex_val_invalid_returns_none() {
        assert_eq!(hex_val(b'z'), None);
        assert_eq!(hex_val(b'@'), None);
        assert_eq!(hex_val(b'/'), None);
    }

    #[test]
    fn percent_decode_plus_is_literal() {
        // RFC 3986: `+` is a literal sub-delimiter in a URI, not an encoded
        // space. Converting it (form-urlencoded semantics) would corrupt a
        // base64 key or a provider key containing `+`. Space-as-plus is sent
        // as `%2B`... well, `%20` for space; `%2B` is the encoding of `+`.
        assert_eq!(percent_decode("hello+world"), "hello+world");
        // `%2B` still decodes to a literal `+` (the correct way to transport it).
        assert_eq!(percent_decode("sk%2Bplus"), "sk+plus");
        // `%20` decodes to a space (the standard URI encoding of space).
        assert_eq!(percent_decode("hello%20world"), "hello world");
    }

    #[test]
    fn percent_decode_no_encoding() {
        assert_eq!(percent_decode("hello"), "hello");
    }

    #[test]
    fn percent_decode_empty() {
        assert_eq!(percent_decode(""), "");
    }

    #[test]
    fn extract_bearer_token_whitespace_trimmed() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer   sk-padded   ".parse().unwrap());
        let token = extract_bearer_token(&headers);
        assert_eq!(token.as_deref(), Some("sk-padded"));
    }

    #[test]
    fn extract_api_key_header_absent_returns_none() {
        let headers = HeaderMap::new();
        assert!(extract_api_key_header(&headers).is_none());
    }

    #[test]
    fn extract_api_key_header_empty_value_ignored() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "".parse().unwrap());
        assert!(extract_api_key_header(&headers).is_none());
    }

    #[test]
    fn extract_api_key_header_whitespace_only_ignored() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "   ".parse().unwrap());
        assert!(extract_api_key_header(&headers).is_none());
    }

    #[test]
    fn extract_auth_token_prefers_bearer_over_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-bearer".parse().unwrap());
        headers.insert("x-api-key", "sk-apikey".parse().unwrap());
        let uri: axum::http::Uri = "/v1/chat/completions".parse().unwrap();
        let token = extract_auth_token(&headers, &uri);
        assert_eq!(token.as_deref(), Some("sk-bearer"));
    }

    #[test]
    fn extract_auth_token_falls_back_to_x_api_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "sk-fallback".parse().unwrap());
        let uri: axum::http::Uri = "/v1/chat/completions".parse().unwrap();
        let token = extract_auth_token(&headers, &uri);
        assert_eq!(token.as_deref(), Some("sk-fallback"));
    }

    #[test]
    fn is_realtime_path_exact_match() {
        assert!(is_realtime_path("/v1/realtime"));
    }

    #[test]
    fn is_realtime_path_subpath() {
        assert!(is_realtime_path("/v1/realtime/gpt-4o-realtime"));
    }

    #[test]
    fn is_realtime_path_no_false_positive() {
        assert!(!is_realtime_path("/v1/realtime-foo"));
        assert!(!is_realtime_path("/realtime"));
        assert!(!is_realtime_path("/v1/chat/completions"));
    }

    #[test]
    fn bytes_to_hex_known_value() {
        let bytes = [0xde, 0xad, 0xbe, 0xef];
        let hex = bytes_to_hex(&bytes);
        assert_eq!(hex, "deadbeef");
    }

    #[test]
    fn bytes_to_hex_upper_bound() {
        let bytes = [0xff; 4];
        let hex = bytes_to_hex(&bytes);
        assert_eq!(hex, "ffffffff");
    }

    #[test]
    fn virtual_key_info_default_is_inactive() {
        let info = VirtualKeyInfo::default();
        assert!(!info.is_active);
        assert!(info.key_id.is_empty());
    }

    #[test]
    fn scope_limits_negative_rates_collapse_to_none() {
        // C27: a negative team `rpm`/`tpm` is silently coerced to `None`
        // (unlimited) by the `u32::try_from(v).ok()` conversion in
        // `scope_limits()` — `u32::try_from(-1i64)` is `Err`. This is the
        // root cause the management handlers now defend against with
        // `validate_limits` (rejecting negative values at create/update time
        // with a 400). This test pins the coercion so a regression in either
        // the validation or the resolution is caught.
        let info = VirtualKeyInfo {
            is_active: true,
            team_rpm: Some(-1),
            team_tpm: Some(-5),
            rpm: Some(-1),
            ..Default::default()
        };
        let limits = info.scope_limits();
        assert_eq!(limits.team.rpm, None, "negative team_rpm must not enforce");
        assert_eq!(limits.team.tpm, None, "negative team_tpm must not enforce");
        assert_eq!(limits.key.rpm, None, "negative key rpm must not enforce");
    }

    /// LOW: DB rates above `u32::MAX` (written by direct DB edit, a
    /// migration, or a future writer — management `validate_limits` rejects
    /// them at the API boundary) must clamp to `u32::MAX`, the closest
    /// representable ceiling, instead of collapsing to `None` and silently
    /// *disabling* a billing-adjacent control.
    #[test]
    fn scope_limits_out_of_range_rates_clamp_to_u32_max() {
        let info = VirtualKeyInfo {
            is_active: true,
            rpm: Some(5_000_000_000),
            tpm: Some(i64::MAX),
            team_rpm: Some(u32::MAX as i64 + 1),
            org_tpm: Some(9_000_000_000_000),
            ..Default::default()
        };
        let limits = info.scope_limits();
        assert_eq!(
            limits.key.rpm,
            Some(u32::MAX),
            "rpm above u32::MAX must clamp, not vanish"
        );
        assert_eq!(limits.key.tpm, Some(u32::MAX));
        assert_eq!(limits.team.rpm, Some(u32::MAX));
        assert_eq!(limits.org.tpm, Some(u32::MAX));
        // In-range values still pass through unchanged.
        let info = VirtualKeyInfo {
            is_active: true,
            rpm: Some(100),
            ..Default::default()
        };
        assert_eq!(info.scope_limits().key.rpm, Some(100));
    }

    #[test]
    fn auth_user_open_key_hash_is_all_zeros() {
        let user = AuthUser::open();
        assert_eq!(user.key_hash, [0u8; 32]);
        assert!(!user.is_master);
    }

    #[test]
    fn auth_user_master_key_hash_carries_provided_hash() {
        let h = hash_key(b"sk-master-key");
        let user = AuthUser::master(h);
        assert_eq!(user.key_hash, h);
        assert!(user.is_master);
    }

    // -----------------------------------------------------------------------
    // End-to-end middleware tests (coverage gaps: JWT through the stack,
    // realtime query-param auth, header precedence, array-aud, brake shapes)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn jwt_authenticates_through_middleware_stack() {
        use crate::auth::jwt::JwtClaims;
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use jsonwebtoken::EncodingKey;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                jwt_secret: Some("jwt-secret-at-least-16".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        // A handler that echoes the claim-derived limits so the middleware
        // actually landing an AuthUser is observable end-to-end.
        let app = Router::new()
            .route(
                "/healthz",
                get(|auth: axum::Extension<AuthUser>| async move {
                    format!(
                        "user={:?} budget={:?} rpm={:?} tpm={:?}",
                        auth.user_id, auth.max_budget, auth.rpm, auth.tpm
                    )
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &JwtClaims {
                sub: "user-1".to_string(),
                team_id: Some("team-1".to_string()),
                max_budget: Some(5.0),
                rpm: Some(100),
                tpm: Some(10_000),
                exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
                jti: None,
                iss: None,
            },
            &EncodingKey::from_secret(b"jwt-secret-at-least-16"),
        )
        .expect("encode jwt");

        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let s = String::from_utf8_lossy(&body);
        assert!(
            s.contains("user=Some(\"user-1\")"),
            "AuthUser must carry the JWT user id: {s}"
        );
        assert!(
            s.contains("budget=Some(5.0)"),
            "claim budget must land on AuthUser: {s}"
        );
        assert!(
            s.contains("rpm=Some(100)"),
            "claim rpm must land on AuthUser: {s}"
        );
    }

    #[tokio::test]
    async fn realtime_query_param_auth_through_middleware() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let app = Router::new()
            .route("/v1/realtime", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/v1/realtime?model=gpt-4o-realtime&authorization=sk-test-master-key")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "the realtime WebSocket query-param credential must authenticate through the middleware"
        );
    }

    #[tokio::test]
    async fn middleware_prefers_bearer_over_x_api_key() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        // Correct Bearer + wrong x-api-key → authenticates (Bearer wins).
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", "Bearer sk-test-master-key")
            .header("x-api-key", "sk-wrong")
            .body(Body::empty())
            .expect("request");
        let resp = app.clone().oneshot(req).await.expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);

        // Wrong Bearer + correct x-api-key → 401 (the Bearer token is the one
        // tried, so the correct x-api-key never gets a chance).
        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", "Bearer sk-wrong")
            .header("x-api-key", "sk-test-master-key")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::UNAUTHORIZED,
            "Authorization: Bearer must take precedence over x-api-key"
        );
    }

    /// A JWT carrying an array-valued `aud` (the Auth0/Keycloak/Azure AD form)
    /// must authenticate through the real middleware, not just `validate()`.
    #[tokio::test]
    async fn jwt_with_array_aud_authenticates_through_middleware() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use jsonwebtoken::EncodingKey;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                jwt_secret: Some("jwt-secret-at-least-16".to_string()),
                jwt_audience: Some("oxllm".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let claims = crate::auth::jwt::JwtClaims {
            sub: "aud-user".to_string(),
            team_id: None,
            max_budget: None,
            rpm: None,
            tpm: None,
            exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
            jti: None,
            iss: None,
        };
        let mut value = serde_json::to_value(claims).expect("claims to value");
        value["aud"] = serde_json::json!(["oxllm", "internal"]);
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &value,
            &EncodingKey::from_secret(b"jwt-secret-at-least-16"),
        )
        .expect("encode jwt");

        let req = Request::builder()
            .method("GET")
            .uri("/healthz")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "array-valued aud with a matching entry must authenticate through the middleware"
        );
    }

    /// A presented-but-wrong token must record an auth-brake failure whether
    /// or not it is JWT-shaped — the two differ only in which validation path
    /// rejects them, and both must count toward the failure budget.
    #[tokio::test]
    async fn wrong_tokens_record_brake_failure_regardless_of_shape() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let brake = {
            let state = shared.load_full();
            state
                .auth_brake
                .clone()
                .expect("auth_brake must be present when master_key is configured")
        };

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let peer: std::net::SocketAddr = "8.8.8.8:4321".parse().expect("peer");
        for token in [
            "sk-plain-wrong",
            // JWT-shaped (three dot-separated segments) but invalid signature.
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.invalid-signature",
        ] {
            let req = Request::builder()
                .method("GET")
                .uri("/healthz")
                .header("authorization", format!("Bearer {token}"))
                .extension(axum::extract::ConnectInfo(peer))
                .body(Body::empty())
                .expect("request");
            let resp = app.clone().oneshot(req).await.expect("response");
            assert_eq!(resp.status(), axum::http::StatusCode::UNAUTHORIZED);
        }
        assert!(
            brake.tracked_ips() >= 1,
            "both JWT-shaped and non-JWT-shaped wrong tokens must record an auth-brake failure"
        );
    }

    // -----------------------------------------------------------------------
    // C03 regression tests
    // -----------------------------------------------------------------------

    /// HIGH: a team `budget_duration` configured on the team record must
    /// propagate through `lookup_virtual_key` onto `AuthUser.team_budget_duration`,
    /// independent of the key's own `budget_duration`. Previously the team
    /// window was fetched from the DB and then silently dropped, leaving the
    /// team spend counter to inherit the key's window.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn team_budget_window_propagates_independently_of_key_window() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-team-window";
        let key_hash = hash_key_hex(token.as_bytes());
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-window".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(100.0),
            budget_duration: Some("1d".to_string()),
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now.clone(),
            models: None,
            blocked: false,
        })
        .await
        .expect("team");
        // Key carries a DIFFERENT (longer) window than the team. The team
        // counter must use the team's `"1d"`, not the key's `"30d"`.
        db.create_key(&oxllm_db::KeyRecord {
            key_id: "key-team-window".to_string(),
            key_hash: key_hash.clone(),
            key_alias: None,
            team_id: Some("team-window".to_string()),
            user_id: None,
            org_id: None,
            project_id: None,
            max_budget: Some(50.0),
            budget_duration: Some("30d".to_string()),
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now,
            expires_at: None,
            is_active: true,
            models: None,
            allowed_cache_controls: None,
        })
        .await
        .expect("key");
        let user =
            expect_found(lookup_virtual_key(&state, &key_hash, hash_key(token.as_bytes())).await);
        assert_eq!(
            user.budget_duration.as_deref(),
            Some("30d"),
            "key window must be the key's own"
        );
        assert_eq!(
            user.team_budget_duration.as_deref(),
            Some("1d"),
            "team window must propagate from the team record, not the key's"
        );
        assert_eq!(user.team_max_budget, Some(100.0));
        // The grouped view must agree (HIGH fix removed the hard-coded None).
        let limits = user.scope_limits();
        assert_eq!(limits.team.budget_duration.as_deref(), Some("1d"));
    }

    /// Expired keys retain versioned negative snapshots. Those snapshots may
    /// not substitute for the shared version authority during a DB outage.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn expired_key_is_negative_cached() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-expired-cache";
        let hex = hash_key_hex(token.as_bytes());
        let hash = hash_key(token.as_bytes());
        let mut record = key_record(&hex, None, None);
        record.expires_at = Some("2000-01-01T00:00:00Z".to_string());
        db.create_key(&record).await.expect("create key");
        // First lookup hits the DB and rejects the expired key.
        assert!(matches!(
            lookup_virtual_key(&state, &hex, hash).await,
            LookupOutcome::Invalid
        ));
        // The invalid key now has a versioned missing-value snapshot.
        assert_eq!(
            cached_auth_value(&state, &format!("vk:{hex}")).await,
            Some(serde_json::Value::Null),
            "expired key must be negative-cached after the first DB read"
        );
        // A DB outage cannot establish whether the expired row was renewed.
        // Fail closed as a server error rather than trusting an old version.
        db.pool().close().await;
        assert!(
            matches!(
                lookup_virtual_key(&state, &hex, hash).await,
                LookupOutcome::ServerError
            ),
            "a cached tombstone must not bypass the DB version authority"
        );
    }

    /// LOW: an inactive (not expired) key must also be negative-cached.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn inactive_key_is_negative_cached() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-inactive-cache";
        let hex = hash_key_hex(token.as_bytes());
        let hash = hash_key(token.as_bytes());
        let mut record = key_record(&hex, None, None);
        record.is_active = false;
        db.create_key(&record).await.expect("create key");
        assert!(matches!(
            lookup_virtual_key(&state, &hex, hash).await,
            LookupOutcome::Invalid
        ));
        assert_eq!(
            cached_auth_value(&state, &format!("vk:{hex}")).await,
            Some(serde_json::Value::Null),
            "inactive key must be negative-cached after the first DB read"
        );
    }

    /// MEDIUM: a JWT carrying a URL-style `iss` and a `team_id` claim must have
    /// its DB team ceiling overlaid by `enrich_jwt_user_with_db`. The previous
    /// `:`-separated `scoped_id` encoding could not be reversed for URL issuers
    /// (the first colon after `jwt:` is the scheme separator), so the raw id was
    /// never recovered and DB enrichment was a silent no-op — every real-world
    /// IdP integration lost team/user budget caps, RPM/TPM limits, and model
    /// restrictions for JWT-authenticated callers.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn enrich_jwt_user_with_db_overlays_team_for_url_issuer() {
        use crate::auth::jwt::{JwtClaims, claims_to_user};

        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-1".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(10.0),
            budget_duration: Some("1d".to_string()),
            rpm: Some(50),
            tpm: Some(500),
            created_at: now.clone(),
            updated_at: now,
            models: None,
            blocked: false,
        })
        .await
        .expect("team");

        let claims = JwtClaims {
            sub: "user-1".to_string(),
            team_id: Some("team-1".to_string()),
            max_budget: None,
            rpm: None,
            tpm: None,
            exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
            jti: None,
            iss: Some("https://idp.example".to_string()),
        };
        let user = claims_to_user(claims);
        // Sanity: the team id arrived issuer-scoped (percent-escaped
        // separator — see `jwt::SCOPE_SEP`).
        let scoped_team = user.team_id.clone().expect("team_id");
        assert!(
            scoped_team.starts_with("jwt:https://idp.example"),
            "scoped team id must carry the URL issuer: {scoped_team}"
        );

        let enriched = jwt::enrich_jwt_user_with_db(&state, user)
            .await
            .expect("team lookup must not hit a server error")
            .expect("unblocked team must enrich");
        assert_eq!(
            enriched.team_id.as_deref(),
            Some("team-1"),
            "after enrichment the DB team id must replace the scoped form"
        );
        assert_eq!(
            enriched.team_max_budget,
            Some(10.0),
            "DB team max_budget must bind for a URL-style issuer"
        );
        assert_eq!(enriched.team_rpm, Some(50));
        assert_eq!(enriched.team_budget_duration.as_deref(), Some("1d"));
    }

    /// MEDIUM: a corrupt `models` JSON column on the DB team record must fail
    /// CLOSED on the JWT enrichment path, identical to the virtual-key path
    /// (`parse_models_column`). Previously the JWT path used a separate
    /// `parse_models_json` that returned `None` on parse failure, leaving
    /// `team_models` at its default `None` (unrestricted) — so the same corrupt
    /// row denied all models to virtual-key callers while granting JWT callers
    /// unrestricted access. The shared `parse_models_column` now feeds both
    /// paths, so corrupt rows yield the deny-all sentinel here too.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn enrich_jwt_user_with_db_corrupt_models_fails_closed() {
        use crate::auth::jwt::{JwtClaims, claims_to_user};

        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-corrupt".to_string(),
            team_name: "Team".to_string(),
            max_budget: None,
            budget_duration: None,
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now,
            models: Some("{not json}".to_string()),
            blocked: false,
        })
        .await
        .expect("team");

        let claims = JwtClaims {
            sub: "user-1".to_string(),
            team_id: Some("team-corrupt".to_string()),
            max_budget: None,
            rpm: None,
            tpm: None,
            exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
            jti: None,
            iss: Some("https://idp.example".to_string()),
        };
        let user = claims_to_user(claims);
        let enriched = jwt::enrich_jwt_user_with_db(&state, user)
            .await
            .expect("team lookup must not hit a server error")
            .expect("corrupt models must still enrich (deny-all ACL)");

        let team_models = enriched.team_models.expect("team_models set");
        assert_eq!(
            team_models,
            vec![MODELS_ACL_DENY_ALL.to_string()],
            "corrupt models column must fail closed to the deny-all sentinel, not unrestricted"
        );
    }

    /// A JWT whose `team_id` matches a blocked team must be rejected, matching
    /// virtual-key auth. `/team/block` is supposed to stop every principal in
    /// the team; overlaying budget/RPM without honoring `blocked` would leave
    /// JWT traffic running after the team is blocked.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn enrich_jwt_user_with_db_rejects_blocked_team() {
        use crate::auth::jwt::{JwtClaims, claims_to_user};

        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-blocked".to_string(),
            team_name: "Blocked".to_string(),
            max_budget: Some(10.0),
            budget_duration: None,
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now,
            models: None,
            blocked: true,
        })
        .await
        .expect("team");

        let claims = JwtClaims {
            sub: "user-1".to_string(),
            team_id: Some("team-blocked".to_string()),
            max_budget: None,
            rpm: None,
            tpm: None,
            exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
            jti: None,
            iss: Some("https://idp.example".to_string()),
        };
        let user = claims_to_user(claims);
        assert!(
            matches!(jwt::enrich_jwt_user_with_db(&state, user).await, Ok(None)),
            "JWT for a blocked team must be rejected"
        );
    }

    /// JWT enrichment uses the same versioned scope helpers as virtual keys.
    /// Warm snapshots cannot bypass the shared version authority during an
    /// outage: a peer may have blocked the team or changed its limits.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn jwt_enrichment_warm_scope_cache_requires_reachable_version_authority() {
        use crate::auth::jwt::JwtClaims;
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use jsonwebtoken::EncodingKey;
        use tower::ServiceExt;

        let db = fresh_db().await;
        let now = oxllm_db::now_iso();
        db.create_team(&oxllm_db::TeamRecord {
            team_id: "team-jwt-cache".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(10.0),
            budget_duration: Some("1d".to_string()),
            rpm: Some(50),
            tpm: Some(500),
            created_at: now.clone(),
            updated_at: now,
            models: None,
            blocked: false,
        })
        .await
        .expect("team");

        let shared = crate::app::build_shared_state_with_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: crate::config::types::GeneralSettings {
                    jwt_secret: Some("jwt-secret-at-least-16".to_string()),
                    ..Default::default()
                },
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            None,
            Some(db.clone()),
        )
        .expect("shared state");

        let app = Router::new()
            .route(
                "/healthz",
                get(|auth: axum::Extension<AuthUser>| async move {
                    format!("team={:?} budget={:?}", auth.team_id, auth.team_max_budget)
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &JwtClaims {
                sub: "user-jwt-cache".to_string(),
                team_id: Some("team-jwt-cache".to_string()),
                max_budget: None,
                rpm: None,
                tpm: None,
                exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
                jti: None,
                iss: Some("https://idp.example".to_string()),
            },
            &EncodingKey::from_secret(b"jwt-secret-at-least-16"),
        )
        .expect("encode jwt");

        let request = || {
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .expect("request")
        };

        // First request resolves the team (and user) rows from the DB and
        // warms the vk_team:/vk_user: caches.
        let resp = app.clone().oneshot(request()).await.expect("response");
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let s = String::from_utf8_lossy(&body);
        assert!(
            s.contains("team=Some(\"team-jwt-cache\")"),
            "DB team id must overlay the scoped form: {s}"
        );
        assert!(
            s.contains("budget=Some(10.0)"),
            "DB team budget must bind on the first request: {s}"
        );

        // Poison the DB: a warm snapshot cannot prove that its version is
        // still current, so the second request must fail closed as 503.
        db.pool().close().await;
        let resp = app.oneshot(request()).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "warm scope snapshots require the shared version authority"
        );
    }

    /// MEDIUM #1 (JWT side): a scope-lookup DB error during JWT enrichment
    /// must fail closed as `Err(())` (surfaced by the middleware as 503, not
    /// an auth failure). Covers both the team branch and the user-only branch
    /// (a `sub` always drives a user lookup).
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn enrich_jwt_user_db_error_fails_closed_as_server_error() {
        use crate::auth::jwt::{JwtClaims, claims_to_user};

        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        db.pool().close().await;

        let base = || JwtClaims {
            sub: "user-1".to_string(),
            team_id: None,
            max_budget: None,
            rpm: None,
            tpm: None,
            exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
            jti: None,
            iss: Some("https://idp.example".to_string()),
        };

        // User-only branch: `sub` drives the user lookup, which fails closed.
        let user = claims_to_user(base());
        assert!(
            matches!(jwt::enrich_jwt_user_with_db(&state, user).await, Err(())),
            "a user-lookup DB error must fail closed as a server error"
        );

        // Team branch (team lookup fails first).
        let mut claims = base();
        claims.team_id = Some("team-1".to_string());
        let user = claims_to_user(claims);
        assert!(
            matches!(jwt::enrich_jwt_user_with_db(&state, user).await, Err(())),
            "a team-lookup DB error must fail closed as a server error"
        );
    }

    /// LOW: a key in a blocked team must be rejected, and `/team/unblock` must
    /// re-enable it immediately — not after the 60s key-tombstone window. The
    /// team-block path no longer tombstones the key's own `vk:{hash}` entry;
    /// rejection relies solely on the team-limits cache, which unblock clears.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn key_re_authenticates_immediately_after_team_unblock() {
        let db = fresh_db().await;
        let state = db_state(db.clone()).await;
        let token = "sk-team-unblock";
        let hex = hash_key_hex(token.as_bytes());
        let hash = hash_key(token.as_bytes());
        let now = oxllm_db::now_iso();

        let team = oxllm_db::TeamRecord {
            team_id: "team-unblock".to_string(),
            team_name: "Team".to_string(),
            max_budget: Some(5.0),
            budget_duration: None,
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now.clone(),
            models: None,
            blocked: false,
        };
        db.create_team(&team).await.expect("team");
        db.create_key(&key_record(&hex, Some("team-unblock"), None))
            .await
            .expect("key");

        // Baseline: the key authenticates while the team is unblocked.
        assert!(
            matches!(
                lookup_virtual_key(&state, &hex, hash).await,
                LookupOutcome::Found(_)
            ),
            "key must authenticate before its team is blocked"
        );

        // Block the team (as /team/block does) and clear the team-limits cache
        // (as the management handler does via invalidate_cache_key).
        let mut blocked = team.clone();
        blocked.blocked = true;
        db.update_team(&blocked).await.expect("block");
        invalidate_cache_key(&state, "vk_team:team-unblock").await;
        assert!(
            matches!(
                lookup_virtual_key(&state, &hex, hash).await,
                LookupOutcome::Invalid
            ),
            "key must be rejected while its team is blocked"
        );

        // Unblock the team (as /team/unblock does) and clear the team-limits
        // cache only — NOT the key's vk:{hash} entry.
        let mut unblocked = team.clone();
        unblocked.blocked = false;
        db.update_team(&unblocked).await.expect("unblock");
        invalidate_cache_key(&state, "vk_team:team-unblock").await;
        let user = expect_found(lookup_virtual_key(&state, &hex, hash).await);
        assert_eq!(user.team_id.as_deref(), Some("team-unblock"));
    }

    /// LOW: `management_auth_middleware` must accept the master key delivered
    /// via `x-api-key` (the Anthropic SDK convention), not just via
    /// `Authorization: Bearer`.
    #[tokio::test]
    async fn management_auth_accepts_x_api_key() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let app = Router::new()
            .route("/management/keys", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                management_auth_middleware,
            ));

        let req = Request::builder()
            .method("GET")
            .uri("/management/keys")
            .header("x-api-key", "sk-test-master-key")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "management routes must accept the master key via x-api-key"
        );
    }

    /// LOW: `management_auth_middleware` must enforce the auth brake on entry
    /// when wired standalone (no outer `auth_brake_guard`), matching
    /// `auth_middleware`. Without this backstop, a blocked IP could hammer
    /// management endpoints unthrottled — the failure counter increments but is
    /// never consulted, so the block never engages. Production stacks mount the
    /// guard, so this only matters for tests/custom stacks.
    #[tokio::test]
    async fn management_auth_brake_backstop_blocks_after_failures() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use tower::ServiceExt;

        let shared = crate::app::build_shared_state(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                master_key: Some("sk-test-master-key".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .expect("shared state");

        let brake = {
            let state = shared.load_full();
            state
                .auth_brake
                .clone()
                .expect("auth_brake must be present when master_key is configured")
        };

        // Standalone stack: NO outer `auth_brake_guard`, only the middleware.
        let app = Router::new()
            .route("/management/keys", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                management_auth_middleware,
            ));

        let peer: std::net::SocketAddr = "8.8.8.8:4321".parse().expect("peer");
        // Exhaust the failure budget (default policy: 10 fails / 60s). Each
        // wrong-master-key request must record a failure via `record_failure`.
        for _ in 0..10 {
            let req = Request::builder()
                .method("GET")
                .uri("/management/keys")
                .header("authorization", "Bearer sk-wrong")
                .extension(axum::extract::ConnectInfo(peer))
                .body(Body::empty())
                .expect("request");
            let resp = app.clone().oneshot(req).await.expect("response");
            assert_eq!(
                resp.status(),
                axum::http::StatusCode::UNAUTHORIZED,
                "wrong master key must be rejected (401) while the IP is not yet blocked"
            );
        }

        // The next request from the same IP must be blocked at the entry
        // backstop (429), not authenticated or 401'd.
        let req = Request::builder()
            .method("GET")
            .uri("/management/keys")
            .header("authorization", "Bearer sk-test-master-key")
            .extension(axum::extract::ConnectInfo(peer))
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            "a blocked IP must be 429'd at the management-auth entry backstop, \
             even with a correct master key and no outer auth_brake_guard"
        );
        assert!(brake.is_blocked("8.8.8.8").await);
    }

    /// MEDIUM: with `jwt_replay_protection` active, an outage of the shared
    /// jti replay cache (e.g. Redis down) must fail *closed but as a server
    /// error*: 503, not 401, and no auth-brake failure recorded. Previously
    /// the outage was mapped onto the generic auth-failure path
    /// (`Err(InvalidToken)` → `Ok(None)` here), so `auth_middleware` answered
    /// 401 while calling `record_failure` — during a Redis flap every
    /// legitimate JWT client racked up per-IP auth-brake failures and got
    /// 429-blocked on top of the outage, violating the fail-503-without-
    /// arming-the-brake doctrine used for the DB paths.
    #[tokio::test]
    async fn jwt_replay_cache_outage_answers_503_without_arming_auth_brake() {
        use crate::auth::jwt::{JwtValidator, ReplayError};
        use crate::state::StateBuilder;
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use axum::routing::get;
        use jsonwebtoken::EncodingKey;
        use oxllm_core::{Cache, Error, IncrResult};
        use tower::ServiceExt;

        /// Stand-in for the shared jti replay store being unreachable.
        struct FailingCache;
        #[async_trait::async_trait]
        impl Cache for FailingCache {
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
            ) -> Result<f64, Error> {
                Err(Error::Network("store unreachable".to_string()))
            }
            async fn incr_if_below(
                &self,
                _key: &str,
                _delta: f64,
                _ceiling: f64,
                _ttl: Option<std::time::Duration>,
            ) -> Result<IncrResult, Error> {
                Err(Error::Network("store unreachable".to_string()))
            }
        }

        let validator = std::sync::Arc::new(
            JwtValidator::new(
                "jwt-secret-at-least-16",
                jsonwebtoken::Algorithm::HS256,
                None,
                None,
            )
            .expect("validator")
            .with_jti_cache(Some(
                std::sync::Arc::new(FailingCache) as std::sync::Arc<dyn Cache>
            )),
        );
        // Sanity: the injected cache fails as a cache outage, not a replay.
        let probe = sample_jwt_claims_for_outage_test();
        assert_eq!(
            validator.check_replay(&probe).await,
            Err(ReplayError::CacheUnavailable)
        );

        let state = StateBuilder::new(crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                jwt_secret: Some("jwt-secret-at-least-16".to_string()),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        })
        .with_jwt_validator(validator)
        .build()
        .expect("state");
        let brake = state
            .auth_brake
            .clone()
            .expect("auth brake must be armed when jwt_secret is configured");
        let shared: crate::state::SharedState =
            std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state));

        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                shared,
                auth_middleware,
            ));

        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &sample_jwt_claims_for_outage_test(),
            &EncodingKey::from_secret(b"jwt-secret-at-least-16"),
        )
        .expect("encode jwt");

        let peer: std::net::SocketAddr = "8.8.8.4:4321".parse().expect("peer");
        // Far beyond the default failure budget (10 fails / 60s window): if
        // the outage recorded auth failures, the IP would have been
        // 429-blocked long before the loop ends.
        for _ in 0..25 {
            let req = Request::builder()
                .method("GET")
                .uri("/healthz")
                .header("authorization", format!("Bearer {token}"))
                .extension(axum::extract::ConnectInfo(peer))
                .body(Body::empty())
                .expect("request");
            let resp = app.clone().oneshot(req).await.expect("response");
            assert_eq!(
                resp.status(),
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "a jti-cache outage must fail closed as 503, not 401"
            );
        }
        assert!(
            !brake.is_blocked("8.8.8.4").await,
            "a jti-cache outage must not arm the auth brake"
        );
    }

    /// Claims helper for `jwt_replay_cache_outage_answers_503_without_arming_auth_brake`.
    fn sample_jwt_claims_for_outage_test() -> crate::auth::jwt::JwtClaims {
        crate::auth::jwt::JwtClaims {
            sub: "user-1".to_string(),
            team_id: None,
            max_budget: None,
            rpm: None,
            tpm: None,
            exp: (jiff::Timestamp::now().as_second() + 3600) as u64,
            jti: Some("outage-jti".to_string()),
            iss: None,
        }
    }

    /// LOW: a path-traversal-style URI must NOT qualify as a realtime path,
    /// so the `authorization` query param is not extracted (and thus cannot
    /// leak into access logs when the request is rerouted). This pins the
    /// defensive check in `is_realtime_path` against `.`/`..` segments.
    #[test]
    fn query_param_rejected_on_path_traversal_attempt() {
        let headers = HeaderMap::new();
        let uri: axum::http::Uri = "/v1/realtime/../chat/completions?authorization=sk-leaked"
            .parse()
            .expect("uri parses");
        assert!(
            extract_auth_token(&headers, &uri).is_none(),
            "path-traversal URI must not trigger realtime query-param extraction"
        );
        // The defensive guard is on `is_realtime_path` itself.
        assert!(
            !is_realtime_path("/v1/realtime/../chat/completions"),
            "is_realtime_path must reject dot-segment paths"
        );
        // Legitimate sub-paths still qualify.
        assert!(is_realtime_path("/v1/realtime/gpt-4o-realtime"));
    }

    /// LOW: per RFC 6750 §2.1 the Bearer credential is a single token; values
    /// with trailing whitespace-delimited content (`"Bearer sk-test extra"`)
    /// must be rejected rather than silently truncated.
    #[test]
    fn extract_bearer_token_rejects_trailing_content() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer sk-test extra".parse().unwrap());
        assert!(
            extract_bearer_token(&headers).is_none(),
            "Bearer values with trailing content must be rejected (RFC 6750 §2.1)"
        );
    }
}
