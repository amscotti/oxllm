//! JWT (JSON Web Token) authentication.
//!
//! JWT is an alternative auth method, checked after the master key and
//! virtual-key paths. A request authenticates via JWT when:
//!
//! 1. `general_settings.jwt_secret` is configured.
//! 2. The bearer token is shaped like a JWT (three dot-separated base64
//!    segments).
//! 3. The signature verifies against the configured secret and the `exp`
//!    claim has not elapsed.
//!
//! Claims of interest: `sub` (user id), `team_id`, and optional budget/rate
//! overrides (`budget`, `rpm`, `tpm`).
//!
//! # Replay protection semantics (burn on presentation)
//!
//! When `jwt_replay_protection` is enabled, a token's `jti` is claimed *at
//! authentication time* — before DB enrichment and before the model-ACL /
//! budget / rate-limit hooks run. A token whose first request is subsequently
//! rejected downstream (429 rate limit, 403 model ACL, 503 DB flap) is
//! therefore consumed: every retry is rejected as a replay for the token's
//! remaining lifetime. This one-presentation semantics is deliberate — it is
//! the strictest reading of replay protection (and the only one enforceable
//! without allowing unbounded concurrent in-flight duplicates). Operators who
//! need retry-friendly clients should leave `jwt_replay_protection` off or
//! mint short-lived tokens. See [`JwtValidator::check_replay`].

use std::sync::Arc;
use std::time::Duration;

use jsonwebtoken::Algorithm;
use jsonwebtoken::DecodingKey;
use jsonwebtoken::Validation;
use jsonwebtoken::errors::Error as JwtError;
use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

use oxllm_core::{Cache, IncrResult};

use crate::auth::{AuthUser, VirtualKeyInfo, hash_key, hash_key_hex};

/// Deserialize a JWT `exp` (a `NumericDate`, RFC 7519 §2) as whole seconds,
/// tolerating the optional fractional part some federated IdPs emit. RFC 7519
/// permits a NumericDate to "contain a fractional part"; `serde` rejects
/// fractional JSON numbers for an integer target type, so a token with
/// `"exp": 1735689600.5` would otherwise fail to decode and be rejected with a
/// generic parse error — leaving the IdP silently unable to authenticate. A
/// fractional value is truncated toward zero; negative / non-finite values
/// (which are malformed for an expiry) are rejected so the field stays
/// fail-closed for anything but a positive time.
fn deserialize_exp<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let n = serde_json::Number::deserialize(deserializer)?;
    if let Some(u) = n.as_u64() {
        return Ok(u);
    }
    if let Some(f) = n.as_f64() {
        if f.is_finite() && f >= 0.0 {
            return Ok(f as u64);
        }
    }
    Err(Error::custom("exp must be a non-negative numeric value"))
}

/// Deserialize an optional `rpm`/`tpm` claim, tolerating integral-valued
/// floats (`"rpm": 100.0`): some federated IdPs emit every JSON number as a
/// float, and a plain `Option<u32>` would fail the whole token for a legal
/// JSON number — stricter than every other numeric claim in the struct.
/// Non-integral, negative, or out-of-`u32`-range values are rejected (fail
/// closed) rather than truncated, so a rate limit can never be silently
/// rounded.
fn deserialize_rate<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let Some(n) = Option::<serde_json::Number>::deserialize(deserializer)? else {
        return Ok(None);
    };
    if let Some(u) = n.as_u64() {
        return u32::try_from(u)
            .map(Some)
            .map_err(|_| Error::custom("rate claim is out of range"));
    }
    if let Some(f) = n.as_f64() {
        if f.is_finite() && f >= 0.0 && f.fract() == 0.0 {
            return u32::try_from(f as u64)
                .map(Some)
                .map_err(|_| Error::custom("rate claim is out of range"));
        }
    }
    Err(Error::custom(
        "rate claim must be a non-negative integral number",
    ))
}

/// JWT claims expected on the wire. Unknown claims are ignored.
///
/// `aud` is intentionally **not** a field here: RFC 7519 §4.1.3 permits `aud`
/// to be either a String or an array of Strings, and typing it as
/// `Option<String>` would make `serde` reject array-valued `aud` tokens (common
/// from Auth0/Keycloak/Azure AD) before audience validation even runs. The
/// `jsonwebtoken` `Validation` still checks `aud` against the raw payload when
/// `jwt_audience` is configured, so omitting the field here does not weaken
/// audience enforcement. Any `aud` present in the token is discarded by serde's
/// default unknown-field handling.
#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub struct JwtClaims {
    /// Subject — the user id.
    pub sub: String,
    /// Team id (custom claim).
    #[serde(default)]
    pub team_id: Option<String>,
    /// Per-token budget override in USD (custom claim).
    #[serde(default, rename = "budget")]
    pub max_budget: Option<f64>,
    /// Per-token requests-per-minute limit (custom claim). Accepts
    /// integral-valued floats (see `deserialize_rate`).
    #[serde(default, deserialize_with = "deserialize_rate")]
    pub rpm: Option<u32>,
    /// Per-token tokens-per-minute limit (custom claim). Accepts
    /// integral-valued floats (see `deserialize_rate`).
    #[serde(default, deserialize_with = "deserialize_rate")]
    pub tpm: Option<u32>,
    /// Expiration (unix seconds). Required by `Validation`. Accepts the
    /// RFC 7519 fractional form, truncated toward zero (see `deserialize_exp`).
    #[serde(deserialize_with = "deserialize_exp")]
    pub exp: u64,
    /// JWT ID (`jti`). Used for optional replay protection when
    /// `jwt_replay_protection` is enabled and a Redis cache is configured.
    #[serde(default)]
    pub jti: Option<String>,
    /// Issuer. Used (with `aud` as fallback) to scope the derived key hash so
    /// two issuers with colliding `sub` namespaces cannot share buckets.
    #[serde(default)]
    pub iss: Option<String>,
}

/// Why [`JwtValidator::check_replay`] rejected a token.
///
/// The two cases must be answered differently by callers: a genuine replay
/// is an *authentication* failure (401, counts toward the auth brake), while
/// a jti-cache outage is a *server-side* failure — the token may be
/// perfectly valid, the gateway simply cannot enforce replay protection
/// while the shared store is down (503, must NOT arm the brake, mirroring
/// the DB-outage doctrine in `crate::auth::auth_middleware`). Collapsing
/// the outage onto the replay/auth-failure path previously made every
/// legitimate JWT client rack up per-IP auth-brake failures during a Redis
/// flap and get 429-blocked on top of the outage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayError {
    /// A duplicate presentation of the same `jti` was detected.
    Replay,
    /// The jti replay cache could not be reached. The request still fails
    /// closed (replay protection must not silently degrade to off), but as
    /// a server error, not an auth failure.
    CacheUnavailable,
}

/// A configured JWT validator (secret + algorithm + optional audience).
#[derive(Clone)]
pub struct JwtValidator {
    decoding: DecodingKey,
    validation: Validation,
    jti_cache: Option<Arc<dyn Cache>>,
}

impl JwtValidator {
    /// Build a validator from config. `secret` is the raw HMAC secret bytes.
    pub fn new(
        secret: &str,
        algorithm: Algorithm,
        audience: Option<&str>,
        issuer: Option<&str>,
    ) -> Result<Self, JwtError> {
        let decoding = DecodingKey::from_secret(secret.as_bytes());
        let mut validation = Validation::new(algorithm);
        // `exp` is enforced by default in `Validation::new`; set it
        // explicitly so future library defaults cannot silently disable it.
        validation.validate_exp = true;
        // Reject tokens used before their `nbf` (Not Before) claim. The
        // claim is optional — when absent, validation passes; when present
        // and in the future, the token is rejected (modulo leeway).
        validation.validate_nbf = true;
        // Tolerate ~30s of clock drift between issuer and verifier to avoid
        // spurious `exp`/`nbf` failures across federated IdPs. `check_replay`
        // derives its marker TTL from the same constant so the marker always
        // outlives the acceptance window.
        validation.leeway = LEEWAY_SECS;
        // `exp` must always be present; without it a token never expires.
        // When an issuer is configured, `iss` must be present too: a token
        // without an `iss` claim would otherwise validate and land on the
        // legacy *unscoped* spend/RPM buckets (see `claims_to_user`), merging
        // with any other no-`iss` IdP sharing the secret. Rejecting it keeps
        // the configured-issuer deployment fully scoped. `aud` gets the same
        // treatment: jsonwebtoken only checks `aud` *when the claim is
        // present*, so without requiring it a validly-signed token with no
        // `aud` would validate — silently defeating the audience scope the
        // operator configured (PyJWT/the reference reject it with
        // `MissingRequiredClaimError("aud")`).
        let mut required = vec!["exp".to_string()];
        if issuer.is_some() {
            required.push("iss".to_string());
        }
        if audience.is_some() {
            required.push("aud".to_string());
        }
        validation.set_required_spec_claims(&required);
        if let Some(aud) = audience {
            validation.set_audience(&[aud]);
        } else {
            validation.validate_aud = false;
        }
        if let Some(iss) = issuer {
            validation.set_issuer(&[iss]);
        }
        Ok(Self {
            decoding,
            validation,
            jti_cache: None,
        })
    }

    pub fn with_jti_cache(mut self, cache: Option<Arc<dyn Cache>>) -> Self {
        self.jti_cache = cache;
        self
    }

    /// Returns `true` when a shared jti replay cache is wired. Used by
    /// `state::build` tests to verify that the cache backend (not the config
    /// string) gates replay protection.
    #[cfg(test)]
    pub(crate) fn has_jti_cache(&self) -> bool {
        self.jti_cache.is_some()
    }

    /// Decode + validate a token. Returns the claims on success.
    pub fn validate(&self, token: &str) -> Result<JwtClaims, JwtError> {
        let data = jsonwebtoken::decode::<JwtClaims>(token, &self.decoding, &self.validation)?;
        Ok(data.claims)
    }

    /// Atomically claim the token's `jti` so a second presentation of the
    /// same token is rejected for the token's (leeway-extended) lifetime.
    ///
    /// The claim happens **on presentation, not on accepted use**: the jti is
    /// consumed during authentication, before DB enrichment
    /// (`enrich_jwt_user_with_db`) and before the model-ACL / budget /
    /// rate-limit hooks run. A request that is rejected downstream after a
    /// successful authentication still burns the token — retries are
    /// replays. This is deliberate (see the module docs for the rationale);
    /// The reference has no equivalent control to mirror here.
    ///
    /// Failures are typed ([`ReplayError`]): a detected replay is an auth
    /// failure; a cache outage is a server-side failure that callers must
    /// surface as 503 without arming the auth brake.
    pub async fn check_replay(&self, claims: &JwtClaims) -> Result<(), ReplayError> {
        // An empty/whitespace `jti` carries no replay signal and is treated as
        // absent: its hash is a constant, so an unchecked `Some("")` would
        // collapse every empty-`jti` token onto one bucket and the first such
        // token would reject all subsequent ones (a fail-closed 401 for any
        // IdP that emits a blank `jti`).
        let Some(jti) = claims.jti.as_deref().filter(|j| !j.trim().is_empty()) else {
            return Ok(());
        };
        let Some(cache) = &self.jti_cache else {
            return Ok(());
        };
        // Hash the JTI *scoped by issuer and subject* before using it as a
        // cache key. RFC 7519 §4.1.7 scopes `jti` uniqueness per issuer, and
        // this gateway supports multiple IdPs sharing one `jwt_secret`; two
        // IdPs that both start their jti counters at `"1"` would otherwise
        // collide on a raw `jti:seen:1` bucket and a legitimate token from the
        // second IdP would be rejected as a replay. The whole triple is hashed
        // because the `jti`/`sub` claims are attacker-influenced and unbounded
        // in length — SHA-256 yields a fixed 64-char hex key regardless of
        // input. Components are joined with a NUL (`\0`) separator rather than
        // `:` so a colon embedded in `sub`/`jti` (only the signing-key holder
        // can mint these, so this is a self-DoS at worst — and only on the
        // legacy no-`iss` path) cannot make two distinct `(sub, jti)` pairs
        // produce the same scope string and collide.
        let scope = format!(
            "{}\0{}\0{}",
            claims.iss.as_deref().unwrap_or_default(),
            claims.sub,
            jti
        );
        let jti_hash = hash_key_hex(scope.as_bytes());
        let key = format!("jti:seen:{jti_hash}");
        // Compute the JTI cache TTL from the JWT `exp` claim. Uses
        // `SystemTime::now()` rather than a monotonic clock because the TTL is
        // expressed in wall-clock seconds (the same frame `exp` is measured
        // in). Accepted as a known minor trade-off (L12): an NTP step
        // backwards could marginally extend the cache window, but the `.max(1)`
        // floor prevents underflow/panic and NTP jumps are rare, small, and
        // bounded by the already-short remaining token lifetime — the worst
        // case is a one-shot few-second over-retention of a seen-JTI marker,
        // not a correctness regression (replay protection is monotonic on the
        // cache claim, not on the clock).
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Clamp the TTL to a sane ceiling. `exp` is provider-issued and
        // unclamped here (only `expires_at` clamps it, at `claims_to_virtual_key_info`);
        // a validly-signed token with an absurd `exp` (e.g. `u64::MAX`) would
        // otherwise build `Duration::from_secs(u64::MAX)`, panicking
        // `Instant + Duration` in the in-memory counter mirror and overflowing
        // Redis `PEXPIRE`. A token that outlives the marker simply becomes
        // re-presentable after the window elapses, which is acceptable for
        // replay protection. The TTL is measured from `exp + LEEWAY_SECS`,
        // not `exp`: `validate()` accepts the token until then (clock-drift
        // leeway), so a marker expiring at `exp` would let a replay presented
        // in the `(exp, exp + 30s]` window validate while its marker is
        // already gone.
        let ttl = claims
            .exp
            .saturating_add(LEEWAY_SECS)
            .saturating_sub(now)
            .clamp(1, MAX_JTI_TTL_SECS);
        // Atomically claim the JTI in a single compare-and-set: the first
        // caller increments the counter 0 → 1 (`Allowed`); every subsequent
        // caller is `Rejected`. This closes the check-then-set TOCTOU window
        // of a `get`-then-`set` sequence, where two concurrent requests
        // carrying the same `jti` could both observe "not seen" before either
        // write landed, letting a duplicate presentation slip through. Backing
        // stores with atomic primitives (Redis Lua, in-memory write lock)
        // override `incr_if_below` for true cross-call atomicity.
        match cache
            .incr_if_below(&key, 1.0, 1.0, Some(Duration::from_secs(ttl)))
            .await
        {
            Ok(IncrResult::Allowed(_)) => Ok(()),
            Ok(IncrResult::Rejected(_)) => {
                tracing::warn!(jti_hash = %jti_hash, "jwt replay detected");
                Err(ReplayError::Replay)
            }
            Err(e) => {
                // Fail closed: replay protection is a security control, so a
                // cache failure (e.g. Redis unreachable) rejects the request
                // rather than risk a duplicate presentation slipping through.
                // The rejection is typed `CacheUnavailable` (not folded into
                // the replay/auth-failure path) so callers answer 503 without
                // recording an auth-brake failure: an outage must not
                // 429-block legitimate clients retrying through it.
                tracing::warn!(error = %e, jti_hash = %jti_hash, "jti replay cache check failed");
                Err(ReplayError::CacheUnavailable)
            }
        }
    }
}

impl std::fmt::Debug for JwtValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JwtValidator")
            .field("validation", &self.validation)
            .finish()
    }
}

/// Parse the algorithm string from config. Returns `None` (fail closed) when
/// `jwt_algorithm` names a non-HMAC algorithm or fails to parse; an unset
/// `jwt_algorithm` keeps the HS256 default.
///
/// Only the **HMAC** family (`HS256` / `HS384` / `HS512`) is supported, because
/// `JwtValidator::new` builds its `DecodingKey` via `from_secret`, which yields
/// an HMAC key. An asymmetric algorithm (`RS256`/`ES256`/`EdDSA`/…) would
/// construct successfully but then reject every token at `validate()` time with
/// a cryptic `InvalidKeyFormat` — and worse, a migrating operator pasting an
/// IdP *public key* into `jwt_secret` would get HS256 verification keyed on a
/// guessable value. So instead of silently downgrading to HS256 (which would
/// weaken the configured control), a non-HMAC or unparseable `jwt_algorithm`
/// returns `None` and the caller disables JWT auth entirely.
///
/// `Algorithm::None` (the "none" / unsecured-JWT algorithm) is rejected up
/// front and *upgraded* to HS256 with a warning: accepting it would let tokens
/// with no signature authenticate, and upgrading (never downgrading) is the
/// safe direction. jsonwebtoken 10.x no longer exposes an `Algorithm::None`
/// variant and `FromStr` refuses "none", so this check is belt-and-braces: it
/// produces a clearer warning than the generic parse-error path and makes the
/// security intent explicit.
pub fn parse_algorithm(s: &Option<String>) -> Option<Algorithm> {
    let Some(s) = s.as_deref() else {
        return Some(Algorithm::HS256);
    };
    if s.eq_ignore_ascii_case("none") {
        tracing::warn!(
            algorithm = %s,
            "jwt_algorithm 'none' is insecure and not allowed; using HS256"
        );
        return Some(Algorithm::HS256);
    }
    match s.parse::<Algorithm>() {
        Ok(a) if a.family() == jsonwebtoken::AlgorithmFamily::Hmac => Some(a),
        Ok(_) => {
            tracing::error!(
                algorithm = %s,
                "non-HMAC jwt_algorithm is not supported (only HS256/HS384/HS512 are \
                 constructable from an HMAC secret); JWT auth disabled"
            );
            None
        }
        Err(e) => {
            tracing::error!(
                algorithm = %s,
                error = %e,
                "invalid jwt_algorithm; JWT auth disabled"
            );
            None
        }
    }
}

/// Clock-drift leeway (seconds) tolerated for `exp`/`nbf` validation. Shared
/// with [`JwtValidator::check_replay`], which must keep the jti marker alive
/// for the whole acceptance window: `validate()` admits a token until
/// `exp + LEEWAY_SECS`, so a marker expiring at `exp` would let a replay
/// presented inside the leeway window slip past.
const LEEWAY_SECS: u64 = 30;

/// Upper bound (seconds) for a JTI replay-cache marker TTL. A validly-signed
/// token can carry an `exp` far beyond any real lifetime (e.g. `u64::MAX`);
/// the replay TTL derives from `exp - now` and an unclamped value would
/// overflow `Instant` in the in-memory counter mirror and Redis `PEXPIRE`.
/// The ceiling is far below `Instant`'s representable range.
const MAX_JTI_TTL_SECS: u64 = 7 * 24 * 60 * 60;

/// Upper bound (USD) for a JWT `budget` claim. Values above this are absurd
/// magnitudes (e.g. `1e308`) that would otherwise be stored verbatim and,
/// while `f64` saturation makes them simply never bind, they pollute counters
/// and reporting with pathological magnitudes. Collapsing them to `None`
/// (unlimited) mirrors the non-finite/negative handling.
const MAX_SANITIZED_BUDGET: f64 = 1e12;

/// A token "looks like" a JWT: three non-empty dot-separated segments.
pub fn looks_like_jwt(token: &str) -> bool {
    let mut count = 0usize;
    for part in token.split('.') {
        if part.is_empty() {
            return false;
        }
        count += 1;
    }
    count == 3
}

/// One-shot gate for the unscoped-JWT warning in [`build_validator`]. Without
/// it a long-running process would re-warn on every hot reload that rebuilds
/// the validator. Mirrors `PEERLESS_BRAKE_WARNED` in `auth/mod.rs`.
static UNSCOPED_JWT_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Warn (once) when JWT auth is enabled with neither `jwt_audience` nor
/// `jwt_issuer` configured: any validly-signed token from any minter sharing
/// the secret is then accepted. The reference emits the same warning
/// (`handle_jwt.py::_build_decode_kwargs`). Returns whether this call emitted
/// the warning, so tests can exercise the gate without touching global state.
fn warn_unscoped_jwt_once(
    warned: &std::sync::atomic::AtomicBool,
    audience: Option<&str>,
    issuer: Option<&str>,
) -> bool {
    if audience.is_some() || issuer.is_some() {
        return false;
    }
    if warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    tracing::warn!(
        "jwt auth is enabled but neither jwt_audience nor jwt_issuer is configured; \
         tokens minted by any application that shares the same signing secret will be \
         accepted. set jwt_audience (and ideally jwt_issuer) to scope this deployment"
    );
    true
}

/// Build the in-memory shared validator from config, or `None` when JWT is
/// not configured (no `jwt_secret`) or the configuration is unusable
/// (blank secret, non-HMAC / unparseable algorithm). Configuration that would
/// silently weaken the control — an HMAC secret keyed on the empty string, or
/// an asymmetric algorithm downgraded to HS256 — fails closed instead.
pub fn build_validator(
    secret: &Option<String>,
    algorithm: &Option<String>,
    audience: &Option<String>,
    issuer: &Option<String>,
    jti_cache: Option<Arc<dyn Cache>>,
) -> Option<Arc<JwtValidator>> {
    let secret = secret.as_deref()?;
    if secret.trim().is_empty() {
        tracing::error!("jwt_secret is empty or whitespace-only; JWT auth disabled");
        return None;
    }
    // Minimum entropy: short HMAC secrets make forgery trivial. Reject
    // secrets shorter than 16 bytes (operators should use ≥32 random bytes).
    const MIN_JWT_SECRET_LEN: usize = super::MIN_AUTH_SECRET_LEN;
    if secret.len() < MIN_JWT_SECRET_LEN {
        tracing::error!(
            len = secret.len(),
            min = MIN_JWT_SECRET_LEN,
            "jwt_secret is too short; JWT auth disabled (use at least {MIN_JWT_SECRET_LEN} characters)"
        );
        return None;
    }
    let algorithm = parse_algorithm(algorithm)?;
    warn_unscoped_jwt_once(&UNSCOPED_JWT_WARNED, audience.as_deref(), issuer.as_deref());
    match JwtValidator::new(secret, algorithm, audience.as_deref(), issuer.as_deref()) {
        Ok(v) => Some(Arc::new(v.with_jti_cache(jti_cache))),
        Err(e) => {
            tracing::error!(error = %e, "failed to build JWT validator; JWT auth disabled");
            None
        }
    }
}

/// Convert validated JWT claims into an [`AuthUser`]. The key hash is derived
/// from the JWT subject (`sub`) rather than the token bytes, so it remains
/// stable across token refreshes (which change `exp`/`iat` and thus the token
/// bytes). This keeps per-user budget/rate-limit/cache buckets coherent.
///
/// The hash input is scoped by the issuer (`iss`) so two identity providers
/// with colliding `sub` namespaces (e.g. `user-1` from two different IdPs)
/// cannot map to the same bucket. `aud` is intentionally **not** used: it is
/// only validated when `jwt_audience` is configured and otherwise varies per
/// client, which would fragment a single user's bucket. The
/// `team_id`/`user_id` carried on the principal are scoped identically (see
/// `claims_to_virtual_key_info`) so spend attribution buckets cannot merge
/// across IdPs or collide with DB team/user ids.
///
/// Tokens without an `iss` claim keep the legacy unscoped `jwt:{sub}` shape
/// for backward compatibility. This is only reachable when `jwt_issuer` is
/// **not** configured — `JwtValidator::new` makes `iss` a required claim
/// whenever an issuer is configured, so a configured-issuer deployment never
/// produces an unscoped bucket. A deployment of two no-`iss` IdPs sharing one
/// `jwt_secret` is unsupported: their spend/RPM buckets merge on equal `sub`
/// (and they are mutually trusting anyway, since both hold the signing key).
///
/// # Scope of enforcement (important for operators)
///
/// JWT principals are bounded **only** by their per-token claims
/// (`budget`, `rpm`, `tpm`). Team / org / project / user ceilings configured in
/// the DB are **NOT** enforced on this path, unlike the DB virtual-key path
/// which fans out `team_limits_for` and populates all six scopes. Team spend is
/// **NOT** attributed either: `team_max_budget` is left `None` here, and the
/// budget hook only writes the `spend:team:{id}` counter when a team cap is
/// configured, so a JWT carrying a `team_id` claim does not appear under its
/// team in `/spend/teams` reporting. (The per-token `budget` claim still binds
/// at the key scope via `max_budget`.) This is intentional — JWT callers are
/// treated as external and self-scoped — but it is asymmetric with virtual
/// keys: if you need DB-backed team caps or team spend attribution to bind JWT
/// callers, populate `team_*`/`org_*`/`project_*`/`user_*` fields (including
/// `team_max_budget`) here via a DB lookup analogous to `lookup_virtual_key`.
///
/// # Claim sanitisation
///
/// Defensively sanitises attacker/issuer-influenced fields so a misbehaving
/// issuer cannot neuter the per-token ceiling or cause bucket collisions:
///
/// - `max_budget` must be finite, `>= 0.0`, and `<= MAX_SANITIZED_BUDGET`;
///   `NaN`/`Infinity`/negative/absurd values collapse to `None` (unlimited). A
///   `NaN` budget would otherwise make every `spend >= max_budget` comparison
///   evaluate `false`, effectively disabling the ceiling.
/// - An empty/all-whitespace `team_id` collapses to `None` so spend is not
///   attributed to a phantom empty-id team.
/// - `exp` outside jiff's representable range is clamped to `Timestamp::MAX`
///   (rather than producing an empty `expires_at` string) so the field remains
///   a parseable RFC-3339 timestamp for any future consumer.
///
/// The empty-`sub` rejection (which would collapse every anonymous token onto
/// one bucket) is performed by the caller in [`claims_to_user_if_valid`],
/// because this function is infallible by contract.
pub fn claims_to_user(claims: JwtClaims) -> AuthUser {
    // The hash input is NUL-joined (`jwt:{iss}\0{sub}`) so a colon embedded in
    // `iss`/`sub` cannot make two distinct `(iss, sub)` pairs collapse onto one
    // `spend:{key_hash}` / rate-limit bucket — the same hardening applied to the
    // jti replay scope (`check_replay`) and `scoped_id`. Only reachable when the
    // signing-key holder mints crafted claims (self-DoS at worst), but kept
    // consistent with the hardened paths. The no-`iss` branch keeps the legacy
    // `jwt:{sub}` shape so existing unscoped deployments' buckets are unchanged.
    let hash_input = match claims.iss.as_ref() {
        Some(scope) => format!("jwt:{}\0{}", scope, claims.sub),
        None => format!("jwt:{}", claims.sub),
    };
    let key_hash = hash_key(hash_input.as_bytes());
    let info = claims_to_virtual_key_info(claims);
    AuthUser::virtual_key(key_hash, info)
}

/// Build the [`VirtualKeyInfo`] for a set of JWT claims, applying the
/// sanitisation rules documented on [`claims_to_user`]. Extracted so the
/// resulting `expires_at` / `max_budget` / `team_id` shaping is unit-testable
/// (these fields are otherwise consumed inside [`AuthUser::virtual_key`] and
/// not observable from the returned [`AuthUser`]).
fn claims_to_virtual_key_info(claims: JwtClaims) -> VirtualKeyInfo {
    let iss = claims.iss.as_deref();
    let team_id = claims.team_id.filter(|t| !t.trim().is_empty());
    let max_budget = claims
        .max_budget
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= MAX_SANITIZED_BUDGET);
    let expires_at = jiff::Timestamp::from_second(i64::try_from(claims.exp).unwrap_or(i64::MAX))
        .unwrap_or(jiff::Timestamp::MAX)
        .to_string();
    // The `key_hash` bucket is issuer-scoped (`jwt:{iss}\0{sub}`, hashed
    // before it is ever stored) so two IdPs with colliding `sub` namespaces
    // cannot share budget/rate-limit buckets. The `team_id`/`user_id` that
    // flow into spend attribution (`spend:team:{id}` / `spend:user:{id}`, via
    // `AuthUser` → `CallContext`) must be scoped the same way: otherwise two
    // IdPs minting the same `sub` (or a JWT whose `team_id` matches a real DB
    // team id) would merge their per-user/per-team spend, and a JWT `team_id`
    // could attribute traffic to an unrelated DB team in `/spend/teams`.
    // Embedding the issuer keeps every JWT identity namespace disjoint from
    // every other IdP and from DB ids. `scoped_id` joins the issuer and the
    // id with the NUL-free [`SCOPE_SEP`] escape so the encoding round-trips
    // through `strip_jwt_scope_prefix` even for URL-style issuers (which
    // contain colons) *and* survives Postgres `TEXT`, which rejects NUL
    // bytes. `key_id` uses the same `jwt:{iss}{SCOPE_SEP}{sub}` shape
    // (legacy `jwt:{sub}` without an issuer): it flows into
    // `SpendEntry.key_id` → `spend_logs.key_id` and every telemetry backend,
    // where a `:`-joined encoding would let `("a", "b:c")` and `("a:b", "c")`
    // merge their spend-log attribution — the same colon-collision the
    // `key_hash`/`scoped_id` hardening closes.
    VirtualKeyInfo {
        key_id: match iss {
            Some(iss) => scoped_id(Some(iss), &claims.sub),
            None => format!("jwt:{}", claims.sub),
        },
        team_id: team_id.map(|team| scoped_id(iss, &team)),
        user_id: Some(scoped_id(iss, &claims.sub)),
        max_budget,
        rpm: claims.rpm.map(|v| v as i64),
        tpm: claims.tpm.map(|v| v as i64),
        expires_at: Some(expires_at),
        is_active: true,
        ..Default::default()
    }
}

/// Separator joining the issuer and the id in a JWT-scoped identity. It is
/// a percent-escape *sequence* (not a raw NUL byte) so the encoded identity
/// is NUL-free: `key_id`/`team_id`/`user_id` flow into the `spend_logs`
/// `TEXT` columns (`SpendEntry` → `SpendLog`), and Postgres rejects NUL
/// bytes with `invalid byte sequence for encoding "UTF8": 0x00` — which
/// previously failed the whole spend-log insert batch for every JWT
/// carrying an `iss` claim. Because [`percent_encode_scope_component`]
/// escapes `%` itself, the sequence can only ever appear as the separator.
const SCOPE_SEP: &str = "%00";

/// Percent-encode the characters that would break the [`SCOPE_SEP`]-joined
/// scoped-id encoding: `%` (the escape character itself) and NUL. Everything
/// else passes through verbatim — the encoding only needs to keep the
/// separator unambiguous, not be URL-safe. NUL is escaped even though
/// [`claims_to_user_if_valid`] rejects NUL-bearing claims, so the encoding
/// stays round-trippable if that guard is ever relaxed.
fn percent_encode_scope_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '%' => out.push_str("%25"),
            '\0' => out.push_str(SCOPE_SEP),
            _ => out.push(c),
        }
    }
    out
}

/// Inverse of [`percent_encode_scope_component`]. Only the `%25`/`%00`
/// escapes the encoder emits are recognised; a `%` that is not part of a
/// known escape is preserved verbatim so a foreign string cannot panic the
/// decoder. Only needed on the DB-enrichment path (`strip_jwt_scope_prefix`).
#[cfg(feature = "db")]
fn percent_decode_scope_component(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            match [bytes[i + 1], bytes[i + 2]] {
                [b'2', b'5'] => {
                    out.push(b'%');
                    i += 3;
                    continue;
                }
                [b'0', b'0'] => {
                    out.push(0);
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    // The input is valid UTF-8 and the decoder only substitutes ASCII
    // escapes in place — it never splits a multi-byte sequence.
    String::from_utf8(out).expect("percent-decode preserves UTF-8")
}

/// Prefix a JWT-derived identity with its issuer so identities minted by
/// different IdPs can never collide. Tokens without an `iss` claim keep the
/// legacy unscoped shape for backward compatibility.
///
/// The issuer and id are percent-encoded (see
/// [`percent_encode_scope_component`]) and joined with the [`SCOPE_SEP`]
/// escape sequence, so the encoding is simultaneously:
///
/// - **NUL-free** — the derived ids flow into `spend_logs` `TEXT` columns
///   and Postgres rejects NUL bytes (see [`SCOPE_SEP`]); and
/// - **unambiguously reversible** — issuer values are overwhelmingly URLs
///   (`https://accounts.google.com`, …) and URLs contain colons in both the
///   scheme and an optional port, so the old `:`-separated encoding
///   (`jwt:{iss}:{id}`) could not be reversed by
///   [`strip_jwt_scope_prefix`], which `enrich_jwt_user_with_db` relies on
///   to recover the raw id for DB lookups — the first colon after `jwt:` is
///   the URL scheme separator, not the `iss`/`id` boundary, so DB
///   enrichment silently no-op'd for every URL-style issuer.
///
/// This mirrors the jti-scope hardening (see [`JwtValidator::check_replay`]),
/// which stays NUL-*joined* because it is hashed before it is ever stored.
fn scoped_id(iss: Option<&str>, id: &str) -> String {
    match iss {
        Some(iss) => format!(
            "jwt:{}{SCOPE_SEP}{}",
            percent_encode_scope_component(iss),
            percent_encode_scope_component(id)
        ),
        None => id.to_string(),
    }
}

/// If JWT auth is configured and `token` is a valid JWT, return the
/// authenticated user. Returns `Ok(None)` when JWT is not configured, the
/// token is not JWT-shaped, validation fails, the token's `sub` claim is
/// empty, or a replay was detected — an *authentication* failure callers
/// surface as 401 (and count toward the auth brake).
///
/// Returns `Err(())` when the jti replay cache could not be reached while
/// replay protection is enabled: the token may be perfectly valid, so this is
/// a *server-side* failure — callers must answer 503 and must NOT record an
/// auth-brake failure (the same doctrine `auth_middleware` applies to DB
/// outages). See [`ReplayError::CacheUnavailable`].
///
/// Validation failures are logged at `debug!` rather than `warn!` because any
/// bearer token with two dots (a legitimate virtual key, deliberate garbage, or
/// an unauthenticated scanner) passes `looks_like_jwt` and lands here — logging
/// `warn!` per request would inflate logs without conveying anything
/// actionable. The replay-detected branch in [`JwtValidator::check_replay`]
/// keeps its `warn!` because a replay is genuinely security-relevant.
///
/// Callers with a database should follow with
/// `enrich_jwt_user_with_db` so DB team/user/org ceilings bind JWT
/// principals the same way virtual keys do.
// `Err(())` is a deliberate fail-closed sentinel (verification/cache outage
// could not be distinguished from invalid); callers translate it to 503.
#[allow(clippy::result_unit_err)]
pub async fn claims_to_user_if_valid(
    validator: &Option<std::sync::Arc<JwtValidator>>,
    token: &str,
) -> Result<Option<AuthUser>, ()> {
    let Some(validator) = validator.as_ref() else {
        return Ok(None);
    };
    if !looks_like_jwt(token) {
        return Ok(None);
    }
    let claims = match validator.validate(token) {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(error = %e, "jwt validation failed");
            return Ok(None);
        }
    };
    // Reject tokens whose `sub` is empty or all-whitespace: every such token
    // from a given issuer would hash to the same `jwt:{iss}\0` bucket, causing
    // cross-caller budget/rate-limit collisions. Only the signing-key holder
    // can mint these, but the collision semantics are surprising and defensive
    // validation is cheap.
    if claims.sub.trim().is_empty() {
        tracing::debug!("jwt rejected: empty 'sub' claim");
        return Ok(None);
    }
    // Reject NUL bytes in the claims that flow into `scoped_id`
    // (`iss`/`sub`/`team_id`). Those identities are separator-joined by
    // `scoped_id` and recovered by `strip_jwt_scope_prefix`; the
    // percent-encoding keeps the round-trip NUL-safe, but JSON (RFC 8259)
    // permits `\u0000` in strings and a NUL in `iss`/`sub` could still
    // collide the NUL-joined jti replay-scope hash for crafted claim pairs
    // (see [`JwtValidator::check_replay`]). No real IdP emits NUL in these
    // claims, so rejecting is cheap defence-in-depth.
    if claims.sub.contains('\0')
        || claims.iss.as_deref().is_some_and(|s| s.contains('\0'))
        || claims.team_id.as_deref().is_some_and(|s| s.contains('\0'))
    {
        tracing::debug!("jwt rejected: NUL byte in sub/iss/team_id claim");
        return Ok(None);
    }
    match validator.check_replay(&claims).await {
        Ok(()) => {}
        Err(ReplayError::Replay) => {
            tracing::warn!("jwt rejected: replayed jti presented");
            return Ok(None);
        }
        Err(ReplayError::CacheUnavailable) => {
            // Server-side failure: reject, but as a 503 without an auth-brake
            // record (see the function docs).
            tracing::warn!("jwt jti replay cache unavailable; failing closed as server error");
            return Err(());
        }
    }
    Ok(Some(claims_to_user(claims)))
}

/// When a database is configured, overlay DB team/user ceilings onto a JWT
/// principal so JWT callers respect the same scope caps as virtual keys.
///
/// Team/user records are resolved through the shared cached helpers
/// (`team_limits_for` / `user_limits_for`, positive 300s /
/// negative 60s snapshots). Each hit validates its content version against the
/// shared DB, just like virtual-key auth, so a late fill or lost invalidation
/// cannot restore stale team/user policy. A warm snapshot cannot bypass a DB
/// outage; the version authority must remain reachable.
///
/// Returns:
/// - `Ok(Some(user))` — proceed with the (possibly enriched) principal
/// - `Ok(None)` — reject: the referenced team is administratively blocked
/// - `Err(())` — a scope lookup failed (fail closed: do not proceed with
///   claim-only unlimited ceilings while the DB is unreachable). Callers
///   surface this as a 503, not an auth failure.
///
/// Identity for spend attribution stays issuer-scoped (from
/// [`claims_to_user`]); DB ids are looked up using the **raw** claim values
/// stripped of the `jwt:{iss}{SCOPE_SEP}` prefix when present. When a DB row
/// is found, its budget/RPM/TPM/models fields replace the claim-only defaults
/// for that scope. A missing row (`Ok(None)` inner) leaves the claim-only
/// principal in place.
// `Err(())` is a deliberate fail-closed sentinel for a DB outage; the caller
// translates it to a 503 rather than treating the principal as invalid.
#[cfg(feature = "db")]
#[allow(clippy::result_unit_err)]
pub async fn enrich_jwt_user_with_db(
    state: &crate::state::AppState,
    mut user: AuthUser,
) -> Result<Option<AuthUser>, ()> {
    let Some(db) = state.db.as_ref() else {
        return Ok(Some(user));
    };

    // Recover raw ids: claims_to_user may have prefixed `jwt:{iss}{SCOPE_SEP}`.
    let raw_team = user
        .team_id
        .as_deref()
        .map(strip_jwt_scope_prefix)
        .filter(|s| !s.is_empty());
    let raw_user = user
        .user_id
        .as_deref()
        .map(strip_jwt_scope_prefix)
        .filter(|s| !s.is_empty());

    if let Some(team_id) = raw_team {
        match super::team_limits_for(state, db, &team_id).await {
            Ok(Some(limits)) => {
                if limits.blocked {
                    tracing::info!(team_id = %team_id, "JWT rejected: team is blocked");
                    return Ok(None);
                }
                // Prefer DB id for spend attribution so JWT + VK share buckets.
                // `team_limits_for` resolves by primary key, so the row's id
                // equals the requested one.
                user.team_id = Some(team_id.to_string());
                user.team_max_budget = limits.max_budget.or(user.team_max_budget);
                user.team_budget_duration =
                    limits.budget_duration.clone().or(user.team_budget_duration);
                user.team_rpm = super::i64_rate_to_u32(limits.rpm).or(user.team_rpm);
                user.team_tpm = super::i64_rate_to_u32(limits.tpm).or(user.team_tpm);
                if let Some(models) = limits.models {
                    user.team_models = Some(models);
                }
            }
            Ok(None) => {}
            Err(()) => {
                tracing::warn!(team_id = %team_id, "JWT DB team lookup failed; rejecting request");
                return Err(());
            }
        }
    }

    if let Some(user_id) = raw_user {
        match super::user_limits_for(state, db, &user_id).await {
            Ok(Some(limits)) => {
                user.user_id = Some(user_id.to_string());
                user.user_max_budget = limits.max_budget.or(user.user_max_budget);
                user.user_budget_duration =
                    limits.budget_duration.clone().or(user.user_budget_duration);
                if let Some(models) = limits.models {
                    user.user_models = Some(models);
                }
            }
            Ok(None) => {}
            Err(()) => {
                tracing::warn!(user_id = %user_id, "JWT DB user lookup failed; rejecting request");
                return Err(());
            }
        }
    }

    Ok(Some(user))
}

/// Strip a leading `jwt:{iss}` prefix ([`SCOPE_SEP`]-separated) if present
/// and percent-decode the remainder so DB lookups use the raw id. The
/// inverse of [`scoped_id`]. Returns an owned `String` because decoding may
/// allocate. Splitting on the escape sequence (rather than `:`) keeps the
/// round-trip unambiguous for URL-style issuers, which contain colons in the
/// scheme/port.
#[cfg(feature = "db")]
fn strip_jwt_scope_prefix(id: &str) -> String {
    // Format produced by `scoped_id`: `jwt:{iss}{SCOPE_SEP}{raw}` or
    // unscoped `raw`.
    if let Some(rest) = id.strip_prefix("jwt:") {
        if let Some((_, raw)) = rest.split_once(SCOPE_SEP) {
            return percent_decode_scope_component(raw);
        }
    }
    id.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::EncodingKey;
    use oxllm_core::Error;

    /// A [`Cache`](oxllm_core::Cache) whose counter operations always fail —
    /// stands in for the shared jti replay store (Redis) being unreachable.
    struct FailingCache;
    #[async_trait::async_trait]
    impl Cache for FailingCache {
        async fn get(&self, _key: &str) -> Option<String> {
            None
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
            Err(Error::Network("store unreachable".to_string()))
        }
        async fn incr_if_below(
            &self,
            _key: &str,
            _delta: f64,
            _ceiling: f64,
            _ttl: Option<Duration>,
        ) -> Result<IncrResult, Error> {
            Err(Error::Network("store unreachable".to_string()))
        }
    }

    fn encode(claims: &JwtClaims, secret: &str) -> String {
        jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("encode jwt")
    }

    fn sample_claims(exp_offset_secs: i64) -> JwtClaims {
        let exp = (jiff::Timestamp::now().as_second() + exp_offset_secs) as u64;
        JwtClaims {
            sub: "user-1".to_string(),
            team_id: Some("team-1".to_string()),
            max_budget: Some(5.0),
            rpm: Some(100),
            tpm: Some(10_000),
            exp,
            jti: None,
            iss: None,
        }
    }

    /// Encode a JWT whose payload is `claims` plus an extra `aud` string claim.
    /// Used to exercise audience handling without re-adding `aud` to
    /// `JwtClaims` (which is intentionally field-less — see the struct docs).
    fn encode_with_aud(claims: &JwtClaims, aud: &str, secret: &str) -> String {
        let mut value = serde_json::to_value(claims).expect("claims to value");
        value["aud"] = serde_json::Value::String(aud.to_string());
        jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            &value,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("encode jwt")
    }

    /// Encode a JWT whose payload is `claims` plus an array-valued `aud` claim
    /// (`"aud": [...]`), the form Auth0/Keycloak/Azure AD emit when multiple
    /// resource servers are registered.
    fn encode_with_aud_array(claims: &JwtClaims, aud: &[&str], secret: &str) -> String {
        let mut value = serde_json::to_value(claims).expect("claims to value");
        value["aud"] = serde_json::Value::Array(
            aud.iter()
                .map(|s| serde_json::Value::String(s.to_string()))
                .collect(),
        );
        jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            &value,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .expect("encode jwt")
    }

    #[test]
    fn looks_like_jwt_detects_three_segments() {
        assert!(looks_like_jwt("a.b.c"));
        assert!(looks_like_jwt("header.payload.signature"));
        assert!(!looks_like_jwt("sk-ox-abc"));
        assert!(!looks_like_jwt("a.b"));
        assert!(!looks_like_jwt("a.b.c.d"));
        assert!(!looks_like_jwt(""));
        assert!(!looks_like_jwt("a..c"));
    }

    #[test]
    fn valid_token_validates() {
        let claims = sample_claims(3600);
        let token = encode_with_aud(&claims, "oxllm", "secret");
        let v =
            JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None).expect("validator");
        let decoded = v.validate(&token).expect("valid");
        assert_eq!(decoded.sub, "user-1");
        assert_eq!(decoded.team_id.as_deref(), Some("team-1"));
    }

    #[test]
    fn expired_token_rejected() {
        let claims = sample_claims(-3600);
        let token = encode(&claims, "secret");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(v.validate(&token).is_err());
    }

    #[test]
    fn wrong_secret_rejected() {
        let claims = sample_claims(3600);
        let token = encode(&claims, "secret");
        let v = JwtValidator::new("other", Algorithm::HS256, None, None).expect("validator");
        assert!(v.validate(&token).is_err());
    }

    #[test]
    fn wrong_audience_rejected() {
        let claims = sample_claims(3600);
        let token = encode_with_aud(&claims, "oxllm", "secret");
        let v =
            JwtValidator::new("secret", Algorithm::HS256, Some("other"), None).expect("validator");
        assert!(v.validate(&token).is_err());
    }

    #[test]
    fn no_audience_configured_skips_aud_check() {
        let claims = sample_claims(3600);
        let token = encode_with_aud(&claims, "oxllm", "secret");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(v.validate(&token).is_ok());
    }

    /// When `jwt_audience` is configured, a token with NO `aud` claim must be
    /// rejected: jsonwebtoken only checks `aud` when the claim is present, so
    /// a validly-signed token from any minter of the shared secret that
    /// simply omits `aud` would otherwise validate — silently defeating the
    /// audience scope the operator configured. Mirrors the `iss` treatment
    /// (`issuer_configured_requires_iss_claim`); PyJWT/the reference raise
    /// `MissingRequiredClaimError("aud")` in exactly this case.
    #[test]
    fn audience_configured_requires_aud_claim() {
        let claims = sample_claims(3600);
        let token = encode(&claims, "secret");
        let v =
            JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None).expect("validator");
        assert!(
            v.validate(&token).is_err(),
            "a no-aud token must be rejected when jwt_audience is configured"
        );
    }

    #[test]
    fn wrong_issuer_rejected() {
        let mut claims = sample_claims(3600);
        claims.iss = Some("https://idp-a.example".to_string());
        let token = encode(&claims, "secret");
        let v = JwtValidator::new(
            "secret",
            Algorithm::HS256,
            None,
            Some("https://idp-b.example"),
        )
        .expect("validator");
        assert!(v.validate(&token).is_err());
    }

    #[test]
    fn correct_issuer_accepted() {
        let mut claims = sample_claims(3600);
        claims.iss = Some("https://idp-a.example".to_string());
        let token = encode(&claims, "secret");
        let v = JwtValidator::new(
            "secret",
            Algorithm::HS256,
            None,
            Some("https://idp-a.example"),
        )
        .expect("validator");
        assert!(v.validate(&token).is_ok());
    }

    #[test]
    fn no_issuer_configured_skips_iss_check() {
        let mut claims = sample_claims(3600);
        claims.iss = Some("https://any-issuer.example".to_string());
        let token = encode(&claims, "secret");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(v.validate(&token).is_ok());
    }

    /// When `jwt_issuer` is configured, a token with NO `iss` claim must be
    /// rejected: jsonwebtoken's `set_issuer` only checks `iss` *when present*,
    /// so without `required_spec_claims` a missing `iss` would validate and
    /// land on the legacy *unscoped* spend/RPM buckets — merging with any other
    /// no-`iss` IdP sharing the secret.
    #[test]
    fn issuer_configured_requires_iss_claim() {
        let claims = sample_claims(3600);
        let token = encode(&claims, "secret");
        let v = JwtValidator::new(
            "secret",
            Algorithm::HS256,
            None,
            Some("https://idp-a.example"),
        )
        .expect("validator");
        assert!(
            v.validate(&token).is_err(),
            "a no-iss token must be rejected when jwt_issuer is configured"
        );
    }

    #[test]
    fn parse_algorithm_defaults_when_unset() {
        assert_eq!(parse_algorithm(&None), Some(Algorithm::HS256));
        assert_eq!(
            parse_algorithm(&Some("HS512".into())),
            Some(Algorithm::HS512)
        );
        assert_eq!(
            parse_algorithm(&Some("HS384".into())),
            Some(Algorithm::HS384)
        );
    }

    #[test]
    fn parse_algorithm_rejects_none_algorithm() {
        // "none" (the unsecured-JWT algorithm) must never be honoured — an
        // unsigned token would let any attacker authenticate. jsonwebtoken 10.x
        // no longer exposes an `Algorithm::None` variant (and `FromStr` refuses
        // "none"), so `parse_algorithm` belts-and-braces upgrades it to HS256
        // (an upgrade, never a downgrade of the configured control).
        let parsed = parse_algorithm(&Some("none".into()));
        assert_eq!(parsed, Some(Algorithm::HS256));
    }

    /// An unparseable `jwt_algorithm` must fail closed (return `None`, JWT
    /// disabled), never silently default to HS256: a typo like `"HS265"` that
    /// was coerced to HS256 would leave the operator believing their stated
    /// algorithm was in force.
    #[test]
    fn parse_algorithm_fails_closed_on_unparseable() {
        assert_eq!(parse_algorithm(&Some("bogus".into())), None);
        assert_eq!(parse_algorithm(&Some("HS265".into())), None);
    }

    /// Case variants and whitespace-padded values that jsonwebtoken's
    /// case-sensitive `FromStr` rejects must fail closed too. `"NONE"` is the
    /// exception: the `none` rejection is case-insensitive, and the safe
    /// direction there is the HS256 upgrade.
    #[test]
    fn parse_algorithm_case_variants() {
        assert_eq!(
            parse_algorithm(&Some("NONE".into())),
            Some(Algorithm::HS256)
        );
        assert_eq!(parse_algorithm(&Some("Hs256".into())), None);
        assert_eq!(parse_algorithm(&Some(" HS256 ".into())), None);
    }

    #[test]
    fn build_validator_none_when_no_secret() {
        assert!(build_validator(&None, &None, &None, &None, None).is_none());
    }

    #[test]
    fn build_validator_some_when_secret() {
        let v = build_validator(
            &Some("long-enough-secret!".into()),
            &None,
            &None,
            &None,
            None,
        );
        assert!(v.is_some());
    }

    #[test]
    fn build_validator_fails_closed_on_short_secret() {
        let v = build_validator(&Some("short".into()), &None, &None, &None, None);
        assert!(
            v.is_none(),
            "secrets shorter than 16 chars must disable JWT auth"
        );
    }

    /// A configured non-HMAC / unparseable `jwt_algorithm` must disable JWT
    /// auth entirely (fail closed), never silently downgrade to HS256: an
    /// operator migrating from an existing proxy plausibly sets `RS256` and pastes the
    /// IdP public key into `jwt_secret`, and HS256 verification keyed on that
    /// public value would let anyone forge tokens.
    #[tokio::test]
    async fn build_validator_fails_closed_on_non_hmac_algorithm() {
        for alg in ["RS256", "PS384", "HS265", "garbage"] {
            let v = build_validator(
                &Some("long-enough-secret!".into()),
                &Some(alg.into()),
                &None,
                &None,
                None,
            );
            assert!(
                v.is_none(),
                "jwt_algorithm '{alg}' must disable JWT auth, not build an HS256 validator"
            );
        }
        // A token signed with the secret-as-HMAC must not authenticate when
        // `jwt_algorithm` is asymmetric: JWT auth is disabled, so the token
        // falls through to no authenticated user.
        let claims = sample_claims(3600);
        let token = encode(&claims, "secret");
        assert!(
            claims_to_user_if_valid(&None, &token)
                .await
                .unwrap()
                .is_none(),
            "a token must not authenticate against a failed-closed RS256 config"
        );
    }

    /// An empty / whitespace-only `jwt_secret` must disable JWT auth (fail
    /// closed): the empty string is trivially guessable, so an HS256 validator
    /// keyed on it would let any client mint valid tokens.
    #[tokio::test]
    async fn build_validator_fails_closed_on_blank_secret() {
        for secret in ["", "   "] {
            let v = build_validator(&Some(secret.into()), &None, &None, &None, None);
            assert!(
                v.is_none(),
                "blank jwt_secret must disable JWT auth (got {secret:?})"
            );
        }
        // A token signed with the empty key must not authenticate (JWT disabled).
        let claims = sample_claims(3600);
        let token = encode(&claims, "");
        assert!(
            claims_to_user_if_valid(&None, &token)
                .await
                .unwrap()
                .is_none()
        );
    }

    /// The unscoped-JWT warning must fire exactly once per gate, and only
    /// when neither `jwt_audience` nor `jwt_issuer` is configured. Mirrors
    /// The reference decode-kwargs warning (behaviour parity; the log
    /// itself is not captured here — the gate logic is what needs pinning).
    #[test]
    fn unscoped_jwt_warning_fires_once_and_only_when_unscoped() {
        use std::sync::atomic::AtomicBool;

        let gate = AtomicBool::new(false);
        assert!(
            !warn_unscoped_jwt_once(&gate, Some("oxllm"), None),
            "audience configured → no warning"
        );
        assert!(
            !warn_unscoped_jwt_once(&gate, None, Some("https://idp.example")),
            "issuer configured → no warning"
        );
        assert!(
            warn_unscoped_jwt_once(&gate, None, None),
            "unscoped deployment → warning fires"
        );
        assert!(
            !warn_unscoped_jwt_once(&gate, None, None),
            "warning must not fire twice on the same gate"
        );
    }

    #[test]
    fn claims_to_user_extracts_fields() {
        let claims = sample_claims(3600);
        let user = claims_to_user(claims);
        assert!(!user.is_master);
        assert!(
            user.key_id
                .as_deref()
                .unwrap_or_default()
                .starts_with("jwt:")
        );
        assert_eq!(user.user_id.as_deref(), Some("user-1"));
        assert_eq!(user.team_id.as_deref(), Some("team-1"));
        assert_eq!(user.max_budget, Some(5.0));
        assert_eq!(user.rpm, Some(100));
        assert_eq!(user.tpm, Some(10_000));
    }

    #[test]
    fn token_without_aud_validates_when_no_audience_configured() {
        let claims = sample_claims(3600);
        let token = encode(&claims, "test-jwt-secret-16");
        let v = build_validator(
            &Some("test-jwt-secret-16".into()),
            &None,
            &None,
            &None,
            None,
        )
        .expect("validator");
        assert!(v.validate(&token).is_ok());
    }

    #[test]
    fn jwt_hash_is_stable_across_token_refreshes() {
        // Two JWTs with the same `sub` but different `exp` must produce the
        // same `key_hash`, otherwise budget/rate-limit/cache buckets reset on
        // every token refresh.
        let claims1 = sample_claims(3600);
        let claims2 = sample_claims(7200);
        assert_ne!(
            claims1.exp, claims2.exp,
            "test setup: exp values must differ"
        );
        assert_eq!(claims1.sub, claims2.sub, "test setup: same subject");

        let token1 = encode(&claims1, "secret");
        let token2 = encode(&claims2, "secret");
        assert_ne!(token1, token2, "test setup: tokens must differ");

        let user1 = claims_to_user(claims1);
        let user2 = claims_to_user(claims2);
        assert_eq!(
            user1.key_hash, user2.key_hash,
            "same sub must hash to the same key_hash across token refreshes"
        );
        assert_eq!(
            user1.key_hash_hex, user2.key_hash_hex,
            "hex encoding must also match"
        );
    }

    #[test]
    fn jwt_hash_scoped_by_iss_prevents_cross_issuer_collisions() {
        // Two issuers with colliding `sub` namespaces (e.g. `user-1` from two
        // different IdPs) must NOT map to the same bucket.
        let mut claims_a = sample_claims(3600);
        claims_a.iss = Some("https://idp-a.example".to_string());
        let mut claims_b = sample_claims(3600);
        claims_b.iss = Some("https://idp-b.example".to_string());
        assert_eq!(claims_a.sub, claims_b.sub, "test setup: same subject");

        let user_a = claims_to_user(claims_a);
        let user_b = claims_to_user(claims_b);
        assert_ne!(
            user_a.key_hash, user_b.key_hash,
            "same sub from different issuers must not share a key_hash"
        );
        assert_ne!(
            user_a.key_hash_hex, user_b.key_hash_hex,
            "hex encoding must also differ"
        );
    }

    /// Two tokens with an equal `sub` minted by different issuers must not
    /// share ANY bucket: not the derived `key_hash` (already pinned by
    /// `jwt_hash_scoped_by_iss_prevents_cross_issuer_collisions`) and not the
    /// `user_id`/`team_id` that drive the `spend:user:*` / `spend:team:*`
    /// attribution counters. Previously only the key bucket was scoped; the
    /// verbatim `sub`/`team_id` flowed into spend aggregation and could merge
    /// per-user/per-team spend across identity providers (or collide with a
    /// real DB team id).
    #[test]
    fn jwt_user_and_team_ids_scoped_by_issuer_so_spend_buckets_cannot_merge() {
        use crate::hooks::spend_keys::{team_spend_key, user_spend_key};

        let mut claims_a = sample_claims(3600);
        claims_a.iss = Some("https://idp-a.example".to_string());
        let mut claims_b = sample_claims(3600);
        claims_b.iss = Some("https://idp-b.example".to_string());
        assert_eq!(claims_a.sub, claims_b.sub, "test setup: same subject");
        assert_eq!(claims_a.team_id, claims_b.team_id, "test setup: same team");

        let user_a = claims_to_user(claims_a);
        let user_b = claims_to_user(claims_b);

        assert_ne!(user_a.key_hash, user_b.key_hash);
        let (user_a_id, user_b_id) = (
            user_a.user_id.as_deref().expect("user_id"),
            user_b.user_id.as_deref().expect("user_id"),
        );
        let (team_a_id, team_b_id) = (
            user_a.team_id.as_deref().expect("team_id"),
            user_b.team_id.as_deref().expect("team_id"),
        );
        assert_ne!(
            user_a_id, user_b_id,
            "user attribution ids must be issuer-scoped"
        );
        assert_ne!(
            team_a_id, team_b_id,
            "team attribution ids must be issuer-scoped"
        );
        assert_ne!(user_spend_key(user_a_id), user_spend_key(user_b_id));
        assert_ne!(team_spend_key(team_a_id), team_spend_key(team_b_id));

        // A JWT `team_id` must never collide with the unscoped DB team ids
        // (`team-<hex>`) used by virtual keys.
        assert!(
            !team_a_id.starts_with("team-"),
            "JWT team attribution id must be namespaced away from DB team ids"
        );
        // Without `iss`, the legacy unscoped shapes are preserved.
        let claims_legacy = sample_claims(3600);
        let user_legacy = claims_to_user(claims_legacy);
        assert_eq!(user_legacy.user_id.as_deref(), Some("user-1"));
        assert_eq!(user_legacy.team_id.as_deref(), Some("team-1"));
    }

    /// `scoped_id` → `strip_jwt_scope_prefix` must round-trip the raw id even
    /// when the issuer is a URL containing a scheme separator and a port. The
    /// previous `:`-separated encoding could not be unambiguously reversed for
    /// URL issuers — the first colon after `jwt:` is the URL scheme separator,
    /// not the `iss`/`id` boundary — so `enrich_jwt_user_with_db` looked up
    /// garbage ids like `//idp.example:8443:team-1` and DB enrichment was a
    /// silent no-op for every real-world IdP.
    ///
    /// Components containing the escape syntax itself (`%00`, `%25`) must also
    /// round-trip: `%` is escaped by the encoder, so a literal `%00` inside a
    /// component can never be mistaken for the separator.
    #[cfg(feature = "db")]
    #[test]
    fn scoped_id_round_trips_through_strip_for_url_issuer() {
        for iss in [
            "https://idp.example:8443",
            "https://accounts.google.com",
            "https://login.microsoftonline.com/v2.0",
            "secret",
        ] {
            let scoped = scoped_id(Some(iss), "team-1");
            assert_eq!(
                strip_jwt_scope_prefix(&scoped),
                "team-1",
                "round-trip must recover the raw id for issuer {iss:?} (got scoped={scoped:?})"
            );
            let scoped_user = scoped_id(Some(iss), "user-9");
            assert_eq!(strip_jwt_scope_prefix(&scoped_user), "user-9");
        }
        // Escape-syntax-looking components round-trip verbatim.
        for (iss, id) in [("evil%00fake", "team-%25"), ("a%25b", "%00")] {
            assert_eq!(
                strip_jwt_scope_prefix(scoped_id(Some(iss), id).as_str()),
                id,
                "components containing the escape syntax must round-trip"
            );
        }
        // No issuer → unscoped → strip is a no-op.
        assert_eq!(
            strip_jwt_scope_prefix(scoped_id(None, "team-1").as_str()),
            "team-1"
        );
        // An unscoped (no `jwt:` prefix) id passes through untouched.
        assert_eq!(strip_jwt_scope_prefix("team-1"), "team-1");
    }

    #[test]
    fn jwt_hash_ignores_aud_when_audience_not_validated() {
        // `aud` is only validated when `jwt_audience` is configured. When it
        // isn't, two JWTs with the same `sub` but different `aud` must
        // produce the SAME key_hash — otherwise a single user's
        // budget/rate-limit/cache buckets fragment across audience values.
        // Note: `aud` is injected directly into the token payload via
        // `encode_with_aud` because it is intentionally not a field of
        // `JwtClaims` (see the struct docs for why).
        let claims_a = sample_claims(3600);
        let claims_b = sample_claims(3600);
        assert_eq!(claims_a.sub, claims_b.sub, "test setup: same subject");

        let token_a = encode_with_aud(&claims_a, "aud-a", "secret");
        let token_b = encode_with_aud(&claims_b, "aud-b", "secret");
        assert_ne!(token_a, token_b, "test setup: tokens must differ");
        let user_a = claims_to_user(claims_a);
        let user_b = claims_to_user(claims_b);
        assert_eq!(
            user_a.key_hash, user_b.key_hash,
            "unvalidated aud must not influence the derived key_hash"
        );
        assert_eq!(
            user_a.key_hash_hex, user_b.key_hash_hex,
            "hex encoding must also match"
        );
    }

    #[test]
    fn jwt_hash_without_iss_or_aud_keeps_legacy_shape() {
        let claims = sample_claims(3600);
        let expected = hash_key(format!("jwt:{}", claims.sub).as_bytes());
        let user = claims_to_user(claims);
        assert_eq!(
            user.key_hash, expected,
            "no iss/aud must keep the legacy jwt:{{sub}} hash"
        );
    }

    #[tokio::test]
    async fn jti_replay_detected_with_cache() {
        use crate::cache::InMemoryCache;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let v = JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None)
            .expect("validator")
            .with_jti_cache(Some(cache));

        let mut claims = sample_claims(3600);
        claims.jti = Some("token-123".to_string());
        let _token = encode(&claims, "secret");

        v.check_replay(&claims).await.expect("first use ok");
        let err = v.check_replay(&claims).await;
        assert!(
            err.is_err(),
            "second presentation of same jti must be rejected as replay"
        );
    }

    /// JTI replay protection follows *burn-on-presentation* semantics: the
    /// jti is claimed inside `claims_to_user_if_valid` (authentication time),
    /// before DB enrichment or any downstream hook runs. A token whose first
    /// request is subsequently rejected (429/403/503) is consumed — the retry
    /// is a replay. This test pins that ordering: the burn happens at auth
    /// time regardless of what the caller does with the authenticated user.
    #[tokio::test]
    async fn jti_burned_at_authentication_time_before_any_downstream_hook() {
        use crate::cache::InMemoryCache;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let validator = Arc::new(
            JwtValidator::new("secret", Algorithm::HS256, None, None)
                .expect("validator")
                .with_jti_cache(Some(cache)),
        );

        let mut claims = sample_claims(3600);
        claims.jti = Some("burn-jti".to_string());
        let token = encode(&claims, "secret");

        assert!(
            claims_to_user_if_valid(&Some(validator.clone()), &token)
                .await
                .unwrap()
                .is_some(),
            "first presentation authenticates and burns the jti"
        );
        assert!(
            claims_to_user_if_valid(&Some(validator), &token)
                .await
                .unwrap()
                .is_none(),
            "retry must be rejected as a replay even though no downstream \
             hook ever accepted the first request"
        );
    }

    #[tokio::test]
    async fn jti_replay_passthrough_without_cache() {
        let v =
            JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None).expect("validator");

        let mut claims = sample_claims(3600);
        claims.jti = Some("token-456".to_string());

        v.check_replay(&claims)
            .await
            .expect("no cache => always ok");
        v.check_replay(&claims)
            .await
            .expect("no cache => repeated use still ok");
    }

    #[tokio::test]
    async fn jti_replay_passthrough_without_jti_claim() {
        use crate::cache::InMemoryCache;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let v = JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None)
            .expect("validator")
            .with_jti_cache(Some(cache));

        let claims = sample_claims(3600);
        assert!(claims.jti.is_none());

        v.check_replay(&claims)
            .await
            .expect("token without jti => no replay check");
    }

    /// The jti marker's TTL is derived from `exp + LEEWAY_SECS`, not `exp`:
    /// `validate()` admits a token until `exp + 30s` (clock-drift leeway), so
    /// a marker expiring at `exp` would let a replay presented in the
    /// `(exp, exp + 30s]` window validate while its marker is already gone.
    /// (The marker is still finite — bounded by [`MAX_JTI_TTL_SECS`] — it
    /// simply must outlive the acceptance window.)
    #[tokio::test]
    async fn jti_marker_ttl_tracks_exp_plus_leeway() {
        use crate::cache::InMemoryCache;

        let cache: Arc<InMemoryCache> = Arc::new(InMemoryCache::default());
        let cache_dyn: Arc<dyn Cache> = cache.clone();
        let v = JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None)
            .expect("validator")
            .with_jti_cache(Some(cache_dyn));

        let mut claims = sample_claims(2);
        claims.jti = Some("short-lived".to_string());

        v.check_replay(&claims).await.expect("first use ok");
        let err = v.check_replay(&claims).await;
        assert!(err.is_err(), "replay within TTL must be rejected");

        let keys = cache.counter_keys().await;
        assert_eq!(keys.len(), 1, "exactly one replay marker");
        let expires_at = cache
            .counter_expires_at(&keys[0])
            .await
            .expect("marker present");
        let remaining = expires_at.saturating_duration_since(std::time::Instant::now());
        // exp = now+2 at claim time, so the marker must live to ≈ exp + 30s.
        // The pre-leeway marker expired at exp (≈2 s from claim); a >15 s
        // bound distinguishes the two without a flaky tight window.
        assert!(
            remaining > Duration::from_secs(15),
            "marker must outlive exp by the 30s leeway window, got {remaining:?}"
        );
    }

    /// A token expired by *less* than the 30 s leeway must still validate
    /// (clock-drift tolerance across federated IdPs). The companion
    /// `expired_token_rejected` only pins the far-past (−3600 s) case.
    #[test]
    fn exp_expired_within_leeway_still_validates() {
        let claims = sample_claims(-10);
        let token = encode(&claims, "secret");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(
            v.validate(&token).is_ok(),
            "expiry 10s in the past must sit inside the 30s leeway window"
        );
    }

    /// A token expired by *more* than the leeway must be rejected — the
    /// leeway is a clock-drift tolerance, not a lifetime extension.
    #[test]
    fn exp_expired_beyond_leeway_rejected() {
        let claims = sample_claims(-61);
        let token = encode(&claims, "secret");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(
            v.validate(&token).is_err(),
            "expiry 61s in the past must fall outside the 30s leeway window"
        );
    }

    /// L11: the JTI must be hashed (not stored verbatim) when used as a cache
    /// key. This guards against attacker-supplied oversized/high-entropy JTIs
    /// bloating the cache. The hash covers the `iss`/`sub`-scoped triple, so a
    /// verbatim implementation would store `"jti:seen:token-123"`.
    #[tokio::test]
    async fn jti_cache_key_is_hashed_not_verbatim() {
        use crate::auth::hash_key_hex;
        use crate::cache::InMemoryCache;

        let cache: Arc<InMemoryCache> = Arc::new(InMemoryCache::default());
        let cache_dyn: Arc<dyn Cache> = cache.clone();
        let v = JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None)
            .expect("validator")
            .with_jti_cache(Some(cache_dyn));

        let mut claims = sample_claims(3600);
        claims.jti = Some("token-123".to_string());
        v.check_replay(&claims).await.expect("first use ok");

        // The stored key must be the hash of the NUL-joined {iss}\0{sub}\0{jti}
        // scope (no `iss` here, so the first component is empty), not the raw
        // JTI.
        let expected_key = format!("jti:seen:{}", hash_key_hex(b"\0user-1\0token-123"));
        assert_eq!(
            cache.get(&expected_key).await.as_deref(),
            Some("1"),
            "JTI must be hashed before use as a cache key"
        );
        // And the verbatim key must NOT exist.
        let verbatim_key = "jti:seen:token-123".to_string();
        assert!(
            cache.get(&verbatim_key).await.is_none(),
            "verbatim JTI must not be stored as the cache key"
        );
    }

    /// Concurrent presentation of the same `jti` must admit exactly one
    /// caller. This pins the atomic claim in `check_replay`: a non-atomic
    /// `get`-then-`set` would let two or more of the concurrent callers all
    /// observe "not seen" before either write landed.
    #[tokio::test]
    async fn jti_replay_concurrent_admits_exactly_one() {
        use crate::cache::InMemoryCache;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let v = JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None)
            .expect("validator")
            .with_jti_cache(Some(cache));

        let mut claims = sample_claims(3600);
        claims.jti = Some("concurrent-jti".to_string());

        let n = 32;
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let v = v.clone();
            let claims = claims.clone();
            handles.push(tokio::spawn(async move {
                v.check_replay(&claims).await.is_ok()
            }));
        }
        let mut ok = 0;
        for h in handles {
            if h.await.expect("join") {
                ok += 1;
            }
        }
        assert_eq!(
            ok, 1,
            "exactly one concurrent caller must win the jti claim (got {ok})"
        );
    }

    /// RFC 7519 §4.1.7 scopes `jti` uniqueness per issuer. Two IdPs sharing
    /// one `jwt_secret` that both start their jti counters at `"1"` must not
    /// collide on a single replay bucket — a legitimate token from the second
    /// IdP would otherwise be rejected as a replay.
    #[tokio::test]
    async fn jti_replay_scoped_by_issuer_and_subject() {
        use crate::cache::InMemoryCache;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None)
            .expect("validator")
            .with_jti_cache(Some(cache));

        let mut claims_a = sample_claims(3600);
        claims_a.iss = Some("https://idp-a.example".to_string());
        claims_a.jti = Some("1".to_string());
        let mut claims_b = sample_claims(3600);
        claims_b.iss = Some("https://idp-b.example".to_string());
        claims_b.jti = Some("1".to_string());
        let mut claims_other_sub = sample_claims(3600);
        claims_other_sub.iss = Some("https://idp-a.example".to_string());
        claims_other_sub.sub = "user-2".to_string();
        claims_other_sub.jti = Some("1".to_string());

        v.check_replay(&claims_a).await.expect("issuer A jti=1 ok");
        v.check_replay(&claims_b)
            .await
            .expect("same jti, different issuer must not be a replay");
        v.check_replay(&claims_other_sub)
            .await
            .expect("same jti/issuer, different sub must not be a replay");
        // And each scope still rejects a genuine repeat of itself.
        assert!(v.check_replay(&claims_a).await.is_err());
        assert!(v.check_replay(&claims_b).await.is_err());
    }

    /// An empty / whitespace-only `jti` is treated as absent: its hash is a
    /// constant, so an unchecked `Some("")` would collapse every such token
    /// onto one bucket and reject all subsequent ones. Replay must be skipped
    /// instead, so repeated empty-`jti` presentations all pass.
    #[tokio::test]
    async fn jti_replay_skipped_for_empty_jti() {
        use crate::cache::InMemoryCache;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None)
            .expect("validator")
            .with_jti_cache(Some(cache));

        for jti in [Some("".to_string()), Some("   ".to_string())] {
            let mut claims = sample_claims(3600);
            claims.jti = jti;
            v.check_replay(&claims).await.expect("empty jti: ok");
            v.check_replay(&claims).await.expect("empty jti: still ok");
        }
    }

    /// A validly-signed token with an absurd `exp` (e.g. `u64::MAX`) must not
    /// panic the replay path: the TTL is clamped to [`MAX_JTI_TTL_SECS`] before
    /// it ever reaches `Instant + Duration` or Redis `PEXPIRE`.
    #[tokio::test]
    async fn jti_replay_huge_exp_does_not_panic_and_clamps_ttl() {
        use crate::cache::InMemoryCache;

        let cache: Arc<InMemoryCache> = Arc::new(InMemoryCache::default());
        let cache_dyn: Arc<dyn Cache> = cache.clone();
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None)
            .expect("validator")
            .with_jti_cache(Some(cache_dyn));

        let mut claims = sample_claims(3600);
        claims.exp = u64::MAX;
        claims.jti = Some("huge-exp-jti".to_string());

        v.check_replay(&claims)
            .await
            .expect("huge-exp first use ok");
        assert!(
            v.check_replay(&claims).await.is_err(),
            "replay of a huge-exp jti must still be rejected"
        );

        // The marker must carry a bounded TTL (well below Instant's range),
        // not a ~u64::MAX-second window.
        let keys = cache.counter_keys().await;
        assert_eq!(keys.len(), 1, "exactly one replay marker");
        let expires_at = cache
            .counter_expires_at(&keys[0])
            .await
            .expect("marker present");
        let ttl = expires_at.saturating_duration_since(std::time::Instant::now());
        assert!(
            ttl <= Duration::from_secs(MAX_JTI_TTL_SECS),
            "replay TTL must be clamped to MAX_JTI_TTL_SECS, got {ttl:?}"
        );
        assert!(
            ttl > Duration::from_secs(MAX_JTI_TTL_SECS - 60),
            "huge-exp replay TTL should sit at the clamp ceiling, got {ttl:?}"
        );
    }

    /// `exp` above `i64::MAX` must not wrap to a 1969 timestamp nor panic. The
    /// value is only surfaced via `VirtualKeyInfo.expires_at` (never
    /// re-validated — the real `exp` check ran in `validate`), but clamping
    /// keeps the field sane for any future consumer. The clamped `expires_at`
    /// must be a parseable RFC-3339 timestamp (not an empty string), so a
    /// future consumer running it through `is_expired` does not fail closed.
    #[test]
    fn claims_to_user_clamps_overflowing_exp() {
        let mut claims = sample_claims(3600);
        claims.exp = u64::MAX;
        // Must not panic on the `u64 → i64` conversion (previously `as i64`
        // wrapped to a negative/1969 value).
        let user = claims_to_user(claims);
        assert_eq!(user.user_id.as_deref(), Some("user-1"));
    }

    #[test]
    fn claims_to_virtual_key_info_clamps_overflowing_exp_to_parseable_rfc3339() {
        let mut claims = sample_claims(3600);
        claims.exp = u64::MAX;
        // `expires_at` must be a parseable RFC-3339 timestamp. Previously
        // `jiff::Timestamp::from_second(i64::MAX)` errored and
        // `.unwrap_or_default()` produced an empty string — which would make
        // `is_expired` fail closed (treat as expired) if ever reused.
        let info = claims_to_virtual_key_info(claims);
        let expires_at = info.expires_at.as_deref().expect("expires_at set");
        assert!(!expires_at.is_empty(), "expires_at must not be empty");
        let ts: jiff::Timestamp = expires_at
            .parse()
            .expect("expires_at must parse as RFC-3339");
        // Clamping lands on jiff's max representable instant, not a 1969 wrap.
        assert!(
            ts > jiff::Timestamp::now(),
            "clamped far-future exp must remain in the future, got {ts}"
        );
    }

    /// Non-HMAC algorithms (`RS256`/`ES256`/`EdDSA`/…) must fail closed at
    /// parse time. `JwtValidator::new` builds its `DecodingKey` via
    /// `from_secret` (HMAC only), so an asymmetric algorithm would construct
    /// successfully but reject every token at `validate()` time with a cryptic
    /// `InvalidKeyFormat` — and an operator pasting an IdP *public key* into
    /// `jwt_secret` would get HS256 verification keyed on a guessable value.
    /// Returning `None` (JWT disabled) rather than downgrading never weakens
    /// the configured control.
    #[test]
    fn parse_algorithm_fails_closed_on_non_hmac() {
        for alg in [
            "RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "EdDSA",
        ] {
            assert_eq!(
                parse_algorithm(&Some(alg.into())),
                None,
                "non-HMAC algorithm '{alg}' must fail closed"
            );
        }
        // HMAC variants pass through unchanged.
        assert_eq!(
            parse_algorithm(&Some("HS256".into())),
            Some(Algorithm::HS256)
        );
        assert_eq!(
            parse_algorithm(&Some("HS384".into())),
            Some(Algorithm::HS384)
        );
        assert_eq!(
            parse_algorithm(&Some("HS512".into())),
            Some(Algorithm::HS512)
        );
    }

    /// RFC 7519 §4.1.3 permits `aud` to be a String or an array of Strings.
    /// Auth0/Keycloak/Azure AD emit arrays when multiple resource servers are
    /// registered. The token must validate both when `jwt_audience` is unset
    /// (audience irrelevant) and when one of the array entries matches the
    /// configured audience. Previously `JwtClaims.aud: Option<String>` made
    /// serde reject arrays before audience validation even ran.
    #[test]
    fn array_valued_aud_validates_when_no_audience_configured() {
        let claims = sample_claims(3600);
        let token = encode_with_aud_array(&claims, &["oxllm", "internal"], "secret");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(v.validate(&token).is_ok());
    }

    #[test]
    fn array_valued_aud_validates_when_one_entry_matches_configured_audience() {
        let claims = sample_claims(3600);
        let token = encode_with_aud_array(&claims, &["internal", "oxllm"], "secret");
        let v =
            JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None).expect("validator");
        assert!(v.validate(&token).is_ok());
    }

    #[test]
    fn array_valued_aud_rejected_when_no_entry_matches_configured_audience() {
        let claims = sample_claims(3600);
        let token = encode_with_aud_array(&claims, &["internal", "other"], "secret");
        let v =
            JwtValidator::new("secret", Algorithm::HS256, Some("oxllm"), None).expect("validator");
        assert!(v.validate(&token).is_err());
    }

    /// A `NaN` budget claim would make every `spend >= max_budget` comparison
    /// evaluate `false`, silently neutering the per-token ceiling (effective
    /// unlimited spend). Non-finite or negative values collapse to `None`.
    #[test]
    fn claims_sanitise_non_finite_or_negative_max_budget() {
        let mut claims = sample_claims(3600);
        claims.max_budget = Some(f64::NAN);
        assert!(claims_to_virtual_key_info(claims).max_budget.is_none());

        let mut claims = sample_claims(3600);
        claims.max_budget = Some(f64::INFINITY);
        assert!(claims_to_virtual_key_info(claims).max_budget.is_none());

        let mut claims = sample_claims(3600);
        claims.max_budget = Some(-1.0);
        assert!(claims_to_virtual_key_info(claims).max_budget.is_none());

        // Absurd but finite magnitudes (near f64::MAX) collapse to unlimited
        // rather than being stored verbatim — they would never bind anyway,
        // but must not panic or pollute counters with pathological values.
        let mut claims = sample_claims(3600);
        claims.max_budget = Some(1e308);
        assert!(
            claims_to_virtual_key_info(claims).max_budget.is_none(),
            "over-large finite budget must behave as effectively-unlimited"
        );

        // Finite and non-negative values pass through.
        let mut claims = sample_claims(3600);
        claims.max_budget = Some(0.0);
        assert_eq!(claims_to_virtual_key_info(claims).max_budget, Some(0.0));
    }

    /// An empty/all-whitespace `team_id` must collapse to `None` so spend is
    /// not attributed to a phantom empty-id team.
    #[test]
    fn claims_sanitise_empty_team_id() {
        let mut claims = sample_claims(3600);
        claims.team_id = Some(String::new());
        assert!(claims_to_virtual_key_info(claims).team_id.is_none());

        let mut claims = sample_claims(3600);
        claims.team_id = Some("   ".to_string());
        assert!(claims_to_virtual_key_info(claims).team_id.is_none());

        // A non-empty team id is preserved.
        let mut claims = sample_claims(3600);
        claims.team_id = Some("team-1".to_string());
        assert_eq!(
            claims_to_virtual_key_info(claims).team_id.as_deref(),
            Some("team-1")
        );
    }

    /// A token whose `sub` is empty or all-whitespace must be rejected at the
    /// `claims_to_user_if_valid` boundary: every such token from a given issuer
    /// would hash to the same `jwt:{iss}\0` bucket, causing cross-caller
    /// budget/rate-limit collisions.
    #[tokio::test]
    async fn empty_sub_claim_rejected_by_claims_to_user_if_valid() {
        let validator = build_validator(
            &Some("test-jwt-secret-16".into()),
            &None,
            &None,
            &None,
            None,
        )
        .expect("validator configured");

        let mut claims = sample_claims(3600);
        claims.sub = String::new();
        let token = encode(&claims, "test-jwt-secret-16");
        assert!(
            claims_to_user_if_valid(&Some(validator.clone()), &token)
                .await
                .unwrap()
                .is_none(),
            "empty sub must be rejected"
        );

        let mut claims = sample_claims(3600);
        claims.sub = "   ".to_string();
        let token = encode(&claims, "test-jwt-secret-16");
        assert!(
            claims_to_user_if_valid(&Some(validator), &token)
                .await
                .unwrap()
                .is_none(),
            "whitespace-only sub must be rejected"
        );
    }

    /// A token signed with a different HMAC algorithm than the validator's
    /// configured algorithm must be rejected. This pins the algorithm-binding
    /// guarantee that prevents `alg` confusion attacks (an HS384 token against
    /// an HS256-configured validator, or vice versa).
    #[test]
    fn mismatched_hmac_algorithm_rejected() {
        let claims = sample_claims(3600);
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS384),
            &claims,
            &EncodingKey::from_secret("secret".as_bytes()),
        )
        .expect("encode hs384 jwt");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(
            v.validate(&token).is_err(),
            "HS384 token must be rejected by an HS256-configured validator"
        );
    }

    /// A token whose `nbf` (Not Before) is in the future must be rejected.
    /// `validate_nbf = true` is set in `JwtValidator::new` but was previously
    /// never exercised by tests (all sample claims omitted `nbf`).
    #[test]
    fn nbf_in_future_rejected() {
        let mut value = serde_json::to_value(sample_claims(3600)).expect("claims to value");
        // `nbf` 1 hour in the future, beyond the 30s leeway.
        let nbf = jiff::Timestamp::now().as_second() + 3600;
        value["nbf"] = serde_json::Value::from(nbf);
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            &value,
            &EncodingKey::from_secret("secret".as_bytes()),
        )
        .expect("encode jwt with nbf");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        assert!(
            v.validate(&token).is_err(),
            "token with future nbf (beyond leeway) must be rejected"
        );
    }

    /// RFC 7519 §2 permits `exp` to carry a fractional part. Plain `serde`
    /// rejects a fractional JSON number for an integer target type, so a token
    /// with `"exp": 1735689600.5` from a federated IdP would otherwise fail to
    /// decode and the IdP would be silently unable to authenticate.
    /// `deserialize_exp` truncates toward zero, so such a token authenticates
    /// with the whole-second expiry.
    #[test]
    fn fractional_exp_token_accepted_with_truncation() {
        let mut value = serde_json::to_value(sample_claims(3600)).expect("claims to value");
        let now = jiff::Timestamp::now().as_second();
        let fractional = now as f64 + 3600.5;
        value["exp"] =
            serde_json::Value::from(serde_json::Number::from_f64(fractional).expect("finite f64"));
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            &value,
            &EncodingKey::from_secret("secret".as_bytes()),
        )
        .expect("encode jwt with fractional exp");
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");
        let claims = v
            .validate(&token)
            .expect("fractional exp must validate (truncated to whole seconds)");
        assert_eq!(
            claims.exp,
            (now + 3600) as u64,
            "fractional exp must be truncated toward zero"
        );
    }

    /// IdPs that emit every JSON number as a float (`"rpm": 100.0`, legal
    /// JSON) must not have their whole token rejected — the integral value is
    /// accepted, mirroring the fractional-`exp` tolerance. Non-integral,
    /// negative, or over-`u32` values are rejected fail-closed so a rate
    /// limit can never be silently rounded.
    #[test]
    fn rate_claims_accept_integral_floats_reject_garbage() {
        let encode_with_claim = |rate: serde_json::Value| {
            let mut value = serde_json::to_value(sample_claims(3600)).expect("claims to value");
            value["rpm"] = rate;
            jsonwebtoken::encode(
                &jsonwebtoken::Header::new(Algorithm::HS256),
                &value,
                &EncodingKey::from_secret("secret".as_bytes()),
            )
            .expect("encode jwt")
        };
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None).expect("validator");

        for rate in [100.0_f64, 1e2] {
            let token = encode_with_claim(serde_json::Value::from(rate));
            let claims = v
                .validate(&token)
                .unwrap_or_else(|e| panic!("integral float rpm {rate} must validate: {e}"));
            assert_eq!(claims.rpm, Some(100), "integral float must parse to u32");
        }
        for rate in [
            serde_json::Value::from(100.5_f64),
            serde_json::Value::from(-5_i64),
            serde_json::Value::from(4_294_967_296_u64),
        ] {
            let token = encode_with_claim(rate.clone());
            assert!(
                v.validate(&token).is_err(),
                "non-integral/negative/out-of-range rpm must fail closed: {rate}"
            );
        }
    }

    /// A colon embedded in `sub`/`jti` must not make two distinct `(sub, jti)`
    /// pairs collide on the same replay scope. Only the legacy no-`iss` path is
    /// reachable (a configured `jwt_issuer` requires `iss`), and only the
    /// signing-key holder controls these claims, so this is a self-DoS at worst
    /// — but the NUL-separated scope encoding makes it impossible regardless.
    #[tokio::test]
    async fn jti_replay_scope_unambiguous_for_colon_in_sub() {
        use crate::cache::InMemoryCache;

        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None)
            .expect("validator")
            .with_jti_cache(Some(cache));

        let mut claims_a = sample_claims(3600);
        claims_a.sub = "a:b".to_string();
        claims_a.jti = Some("c".to_string());
        let mut claims_b = sample_claims(3600);
        claims_b.sub = "a".to_string();
        claims_b.jti = Some("b:c".to_string());

        v.check_replay(&claims_a).await.expect("first token ok");
        v.check_replay(&claims_b)
            .await
            .expect("distinct (sub, jti) pair must not be treated as a replay");
        // And each scope still rejects a genuine repeat of itself.
        assert!(v.check_replay(&claims_a).await.is_err());
        assert!(v.check_replay(&claims_b).await.is_err());
    }

    /// A cache outage during the JTI replay check must fail closed (reject)
    /// rather than risk admitting a replayed token — but as the *distinct*
    /// `CacheUnavailable` error, not the `Replay`/auth-failure outcome, so
    /// `auth_middleware` can answer 503 without arming the auth brake.
    #[tokio::test]
    async fn jti_replay_fails_closed_on_cache_error() {
        let cache: Arc<dyn Cache> = Arc::new(FailingCache);
        let v = JwtValidator::new("secret", Algorithm::HS256, None, None)
            .expect("validator")
            .with_jti_cache(Some(cache));

        let mut claims = sample_claims(3600);
        claims.jti = Some("err-jti".to_string());
        assert_eq!(
            v.check_replay(&claims).await,
            Err(ReplayError::CacheUnavailable),
            "a cache error must reject the token (fail closed) with the \
             distinct CacheUnavailable error, not a generic auth failure"
        );
    }

    /// A jti-cache outage surfacing through `claims_to_user_if_valid` must be
    /// distinguishable from a validation failure: `Err(())` (a server-side
    /// failure the middleware answers with 503, no auth-brake record) instead
    /// of `Ok(None)` (an authentication failure the middleware answers with
    /// 401 while counting toward the auth brake). Folding the outage onto the
    /// auth-failure path let a Redis flap 429-block every legitimate JWT
    /// client via the per-IP auth brake.
    #[tokio::test]
    async fn jti_cache_outage_is_a_server_error_not_an_auth_failure() {
        let validator = Arc::new(
            JwtValidator::new("secret", Algorithm::HS256, None, None)
                .expect("validator")
                .with_jti_cache(Some(Arc::new(FailingCache) as Arc<dyn Cache>)),
        );

        let mut claims = sample_claims(3600);
        claims.jti = Some("outage-jti".to_string());
        let token = encode(&claims, "secret");
        assert!(
            claims_to_user_if_valid(&Some(validator), &token)
                .await
                .is_err(),
            "a jti-cache outage must surface as Err (server error), not as \
             Ok(None) (auth failure)"
        );
    }

    /// Distinct `(iss, sub)` pairs that would collide under a `:`-joined
    /// encoding must produce distinct `key_hash` buckets. `iss="a", sub="b:c"`
    /// and `iss="a:b", sub="c"` both flatten to `"jwt:a:b:c"` with a colon
    /// separator, merging their spend/rate-limit buckets; the NUL-separated
    /// hash input keeps them distinct, matching the jti-scope / `scoped_id`
    /// hardening. Only reachable when the signing-key holder mints crafted
    /// claims (self-DoS at worst), but pinned for consistency with the
    /// hardened paths.
    #[test]
    fn claims_to_user_key_hash_unambiguous_for_colon_in_iss_or_sub() {
        let mut claims_a = sample_claims(3600);
        claims_a.iss = Some("a".to_string());
        claims_a.sub = "b:c".to_string();

        let mut claims_b = sample_claims(3600);
        claims_b.iss = Some("a:b".to_string());
        claims_b.sub = "c".to_string();

        let user_a = claims_to_user(claims_a);
        let user_b = claims_to_user(claims_b);
        assert_ne!(
            user_a.key_hash, user_b.key_hash,
            "distinct (iss, sub) pairs must not share a key_hash despite colon collision"
        );
        assert_ne!(
            user_a.key_hash_hex, user_b.key_hash_hex,
            "hex encoding must also differ"
        );
    }

    /// `key_id` must use the same unambiguous separation as the key-hash input
    /// and `scoped_id`: a `:`-joined `jwt:{iss}:{sub}` lets `("a", "b:c")` and
    /// `("a:b", "c")` collide on one `SpendEntry.key_id`, merging their
    /// spend-log and telemetry attribution. The no-`iss` legacy shape
    /// (`jwt:{sub}`) is unchanged.
    #[test]
    fn key_id_unambiguous_for_colon_in_iss_or_sub() {
        let mut claims_a = sample_claims(3600);
        claims_a.iss = Some("a".to_string());
        claims_a.sub = "b:c".to_string();

        let mut claims_b = sample_claims(3600);
        claims_b.iss = Some("a:b".to_string());
        claims_b.sub = "c".to_string();

        let user_a = claims_to_user(claims_a);
        let user_b = claims_to_user(claims_b);
        assert_ne!(
            user_a.key_id, user_b.key_id,
            "distinct (iss, sub) pairs must not share a key_id despite colon collision"
        );
        // Pin the issuer-scoped encoding shape explicitly.
        assert_eq!(user_a.key_id.as_deref(), Some("jwt:a%00b:c"));
        // Legacy no-issuer shape is preserved for existing deployments.
        let legacy = claims_to_user(sample_claims(3600));
        assert_eq!(legacy.key_id.as_deref(), Some("jwt:user-1"));
    }

    /// Every JWT-derived id that flows into the `spend_logs` table
    /// (`key_id` / `team_id` / `user_id`, via `SpendEntry` → `SpendLog`) must
    /// be NUL-free: Postgres `TEXT` rejects NUL bytes with `invalid byte
    /// sequence for encoding "UTF8": 0x00`, which fails the whole batched
    /// INSERT — including other principals' rows — so with the db feature on
    /// Postgres, every spend-log insert for a JWT carrying an `iss` claim
    /// previously failed. The percent-escaped `SCOPE_SEP` separator keeps the
    /// ids NUL-free while staying unambiguous.
    #[test]
    fn jwt_derived_ids_are_nul_free_for_postgres_text() {
        let mut claims = sample_claims(3600);
        claims.iss = Some("https://idp.example".to_string());
        let user = claims_to_user(claims);
        for (field, id) in [
            ("key_id", user.key_id.as_deref()),
            ("team_id", user.team_id.as_deref()),
            ("user_id", user.user_id.as_deref()),
        ] {
            let Some(id) = id else {
                panic!("{field} must be set for issuer-scoped claims")
            };
            assert!(
                !id.contains('\0'),
                "{field} must be NUL-free for Postgres TEXT spend-log columns: {id:?}"
            );
        }
        // Pin the encoding shape: separator is the `%00` escape, not a NUL.
        assert_eq!(
            user.key_id.as_deref(),
            Some("jwt:https://idp.example%00user-1")
        );
        assert_eq!(
            user.user_id.as_deref(),
            Some("jwt:https://idp.example%00user-1")
        );
        assert_eq!(
            user.team_id.as_deref(),
            Some("jwt:https://idp.example%00team-1")
        );
    }

    /// A token carrying a NUL byte in `sub`/`iss`/`team_id` must be rejected at
    /// the `claims_to_user_if_valid` boundary. Those claims are separator-joined
    /// by `scoped_id` and recovered by `strip_jwt_scope_prefix`; JSON (RFC
    /// 8259) permits `\u0000` in strings. The percent-encoding keeps the
    /// round-trip NUL-safe, but a NUL in `iss`/`sub` could still collide the
    /// NUL-joined jti replay-scope hash for crafted claim pairs, so rejecting
    /// stays cheap defence-in-depth.
    #[tokio::test]
    async fn claims_to_user_if_valid_rejects_nul_in_claims() {
        let validator = build_validator(
            &Some("test-jwt-secret-16".into()),
            &None,
            &None,
            &None,
            None,
        )
        .expect("validator configured");

        let cases: [(&str, String, Option<String>, Option<String>); 3] = [
            ("sub", "evil\0fake".to_string(), None, None),
            (
                "iss",
                "user-1".to_string(),
                Some("evil\0fake".to_string()),
                None,
            ),
            (
                "team_id",
                "user-1".to_string(),
                None,
                Some("evil\0fake".to_string()),
            ),
        ];
        for (field, sub, iss, team) in cases {
            let mut claims = sample_claims(3600);
            claims.sub = sub;
            claims.iss = iss;
            claims.team_id = team;
            let token = encode(&claims, "test-jwt-secret-16");
            assert!(
                claims_to_user_if_valid(&Some(validator.clone()), &token)
                    .await
                    .unwrap()
                    .is_none(),
                "NUL in {field} must be rejected"
            );
        }
    }
}
