//! Signed-cookie session auth with embedded CSRF token, and per-IP login
//! rate-limiting.
//!
//! Session cookie value: `<session_id>.<expires_unix>.<csrf_b64>.<hmac_hex>`
//! where the HMAC is `HMAC-SHA256(session_secret,
//! "<session_id>.<expires_unix>.<csrf_b64>")`. Verification recomputes the tag
//! and uses constant-time comparison via `hmac::Mac::verify_slice`. The cookie
//! stores **no** master key — only a random session id and CSRF token.
//!
//! The CSRF token is exposed to templates (via `<meta name="csrf-token">`) so
//! HTMX can echo it back as `X-CSRF-Token` on state-changing requests. The
//! `middleware::require_csrf` layer enforces the check.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::app::AppState;

type HmacSha256 = Hmac<Sha256>;

/// Name of the session cookie.
pub const SESSION_COOKIE: &str = "oxllm_admin_session";

/// Cookie carrying the pre-login CSRF token minted by the login form.
/// Unlike the session CSRF (delivered via an authenticated session cookie),
/// this token must be verifiable *before* any session exists, so the login
/// form sets it in a short-lived signed cookie and echoes the token into a
/// hidden form field. `POST /login` verifies the pair, which blocks
/// cross-site forced logins (login CSRF) even though the login route is
/// mounted on the public router.
pub const LOGIN_CSRF_COOKIE: &str = "oxllm_admin_login_csrf";

/// CSRF header expected on POST/DELETE requests.
pub const CSRF_HEADER: &str = "X-CSRF-Token";

/// A freshly minted session, ready to be written into a `Set-Cookie` header.
pub struct Session {
    pub cookie_value: String,
    pub max_age_secs: u64,
    pub csrf_token: String,
}

/// Parsed fields of a verified session cookie.
#[derive(Clone, Debug)]
pub struct SessionInfo {
    pub session_id: String,
    pub csrf_token: String,
}

/// Issue a new signed session valid for `ttl_hours`. The session embeds a
/// random CSRF token (32 bytes, base64).
pub fn create_session(secret: &str, ttl_hours: u64) -> Session {
    let mut id_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut id_bytes);
    let session_id = hex_encode(&id_bytes);
    let mut csrf_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut csrf_bytes);
    let csrf_token = base64_encode(&csrf_bytes);
    let max_age_secs = ttl_hours.saturating_mul(3600);
    let expires = unix_now().saturating_add(max_age_secs);
    let payload = format!("{session_id}.{expires}.{csrf_token}");
    let tag = sign(secret, payload.as_bytes());
    let cookie_value = format!("{payload}.{tag}");
    Session {
        cookie_value,
        max_age_secs,
        csrf_token,
    }
}

/// Render the full `Set-Cookie` header value for a live session. When
/// `secure` is `true`, the `Secure` attribute is added so the cookie is sent
/// only over HTTPS.
pub fn session_cookie_header(session: &Session, secure: bool) -> String {
    let secure_attr = if secure { "; Secure" } else { "" };
    format!(
        "{}={}; HttpOnly; SameSite=Strict; Path=/; Max-Age={}{}",
        SESSION_COOKIE, session.cookie_value, session.max_age_secs, secure_attr
    )
}

/// A `Set-Cookie` value that immediately expires the session cookie.
pub fn clear_cookie_header(secure: bool) -> String {
    let secure_attr = if secure { "; Secure" } else { "" };
    format!(
        "{}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{}",
        SESSION_COOKIE, secure_attr
    )
}

/// Mint a pre-login CSRF token. Returns `(token, set_cookie_value)` where
/// the cookie is signed with the session secret (`<token>.<hmac_hex>`) so a
/// forged form field cannot pass verification without the server-issued
/// cookie. Short `Max-Age` (10 min) bounds replay of an abandoned token.
pub fn create_login_csrf(secret: &str, secure: bool) -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let token = base64_encode(&bytes);
    let tag = sign(secret, token.as_bytes());
    let secure_attr = if secure { "; Secure" } else { "" };
    let cookie = format!(
        "{}={}.{}; HttpOnly; SameSite=Strict; Path=/; Max-Age=600{}",
        LOGIN_CSRF_COOKIE, token, tag, secure_attr
    );
    (token, cookie)
}

/// Verify a `POST /login` CSRF submission: the hidden form field must match
/// the token in the server-issued (signed) login-CSRF cookie.
pub fn login_csrf_ok(headers: &axum::http::HeaderMap, secret: &str, presented: &str) -> bool {
    let Some(value) = headers
        .get(axum::http::header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|header| find_cookie(header, LOGIN_CSRF_COOKIE))
    else {
        return false;
    };
    let Some((token, tag_hex)) = value.split_once('.') else {
        return false;
    };
    let Ok(tag) = hex_decode(tag_hex.as_bytes()) else {
        return false;
    };
    if !verify_tag(secret, token.as_bytes(), &tag) {
        return false;
    }
    !token.is_empty() && ct_eq(token.as_bytes(), presented.as_bytes())
}

/// Verify a session cookie value against the secret. Returns `true` only when
/// the signature is valid and the session has not expired.
pub fn verify_session(secret: &str, cookie_value: &str) -> bool {
    parse_session(secret, cookie_value).is_some()
}

/// Parse and verify a session cookie value, returning the embedded session id
/// and CSRF token on success.
pub fn parse_session(secret: &str, cookie_value: &str) -> Option<SessionInfo> {
    let mut parts = cookie_value.split('.');
    let session_id = parts.next()?;
    let expires_str = parts.next()?;
    let csrf_token = parts.next()?;
    let tag_hex = parts.next()?;
    if parts.next().is_some() {
        return None;
    }

    // Verify the HMAC first, then check expiry on the authenticated payload.
    // The opposite order makes the "expired" rejection observably faster than
    // the "bad signature" path, leaking a passive signal about stolen cookies
    // (a tampered expiry fails HMAC anyway, so the bind is still correct, but
    // an attacker can distinguish expired-but-signed from tampered).
    let payload = format!("{session_id}.{expires_str}.{csrf_token}");
    let tag_bytes = hex_decode(tag_hex.as_bytes()).ok()?;
    if !verify_tag(secret, payload.as_bytes(), &tag_bytes) {
        return None;
    }

    let expires = expires_str.parse::<u64>().ok()?;
    if unix_now() >= expires {
        return None;
    }

    Some(SessionInfo {
        session_id: session_id.to_string(),
        csrf_token: csrf_token.to_string(),
    })
}

/// Extract the CSRF token from a verified session cookie header. Returns
/// `None` when the cookie is missing or invalid.
pub fn csrf_token_from_headers(headers: &axum::http::HeaderMap, secret: &str) -> Option<String> {
    headers
        .get(axum::http::header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|header| find_cookie(header, SESSION_COOKIE))
        .and_then(|value| parse_session(secret, value).map(|i| i.csrf_token))
}

/// Axum middleware: reject any request without a valid session by redirecting
/// to `/login`. On success, the parsed CSRF token is attached as a request
/// extension so handlers and the CSRF middleware can read it. Public routes
/// are mounted on a separate router that never sees this middleware.
pub async fn require_session(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let info = req
        .headers()
        .get(axum::http::header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|header| find_cookie(header, SESSION_COOKIE))
        .and_then(|value| parse_session(&state.config.session_secret, value));

    if let Some(info) = info {
        req.extensions_mut().insert(SessionCsrf(info.csrf_token));
        next.run(req).await
    } else {
        let clear = clear_cookie_header(state.config.cookie_secure);
        if crate::app::is_htmx(req.headers()) {
            // A 303 would be followed inside the XHR and swap the full login
            // document into a table. HTMX only processes HX-Redirect on the
            // non-redirect response itself.
            return (
                [(axum::http::header::SET_COOKIE.as_str(), clear.as_str())],
                crate::app::htmx_redirect("/login"),
            )
                .into_response();
        }
        (
            StatusCode::SEE_OTHER,
            [("location", "/login")],
            [(axum::http::header::SET_COOKIE.as_str(), clear.as_str())],
        )
            .into_response()
    }
}

/// Request extension carrying the CSRF token of the verified session.
/// Handlers consume it via `axum::Extension` to render it into the page.
#[derive(Clone, Debug)]
pub struct SessionCsrf(pub String);

fn find_cookie<'a>(header: &'a str, name: &str) -> Option<&'a str> {
    let prefix = format!("{name}=");
    for cookie in header.split(';') {
        let trimmed = cookie.trim();
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str()) {
            return Some(rest);
        }
    }
    None
}

fn sign(secret: &str, payload: &[u8]) -> String {
    let tag = match make_mac(secret) {
        Some(mut mac) => {
            mac.update(payload);
            mac.finalize().into_bytes().to_vec()
        }
        None => Vec::new(),
    };
    hex_encode(&tag)
}

fn verify_tag(secret: &str, payload: &[u8], tag: &[u8]) -> bool {
    let mut mac = match make_mac(secret) {
        Some(m) => m,
        None => return false,
    };
    mac.update(payload);
    mac.verify_slice(tag).is_ok()
}

fn make_mac(secret: &str) -> Option<HmacSha256> {
    HmacSha256::new_from_slice(secret.as_bytes()).ok()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn hex_decode(hex: &[u8]) -> Result<Vec<u8>, ()> {
    if hex.len() % 2 != 0 {
        return Err(());
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.chunks_exact(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, ()> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(()),
    }
}

const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
        out.push(B64_ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(B64_ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(B64_ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    out
}

/// Constant-time byte comparison backed by the audited `subtle` crate.
///
/// `subtle::ConstantTimeEq` compares the raw bytes with no short-circuit on
/// value mismatch and needs no hashing, so the *contents* of neither side leak
/// through timing. It does, however, take an early length branch: slices of
/// unequal length are rejected before any byte is compared. The sole
/// secret-length comparison in this crate is the login password check
/// (`pages::auth::login`), where an attacker timing many requests could in
/// principle recover the *password length*. This is the standard limitation of
/// byte-wise CT comparison and is remotely unexploitable in practice — the
/// signal is dwarfed by network jitter and the password is also rate-limited
/// by `LoginBrake`. Accepted trade-off: the simplicity and auditability of
/// comparing plaintext bytes outweighs the marginal length signal. This is the
/// same primitive `oxllm-server` uses for master-key comparison, keeping both
/// surfaces on one audited comparison path.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}

/// Upper bound on the number of distinct IPs tracked. Without a cap, a
/// rotating-IP attacker hitting `/login` could grow the map without limit.
/// Expired entries are evicted as the cap is approached; if live (blocked)
/// entries alone fill the map, new IPs are simply not tracked (fail-open) so
/// tracked offenders are never evicted to make room.
pub(crate) const LOGIN_BRAKE_MAX_ENTRIES: usize = 10_000;

/// Per-IP login-failure rate limiter. After `max_fails` failures within
/// `window_secs`, the IP is blocked for `block_secs`. In-memory only (per
/// process); a fleet-wide impl would need Redis.
#[derive(Clone)]
pub struct LoginBrake {
    inner: Arc<Mutex<HashMap<String, FailureRecord>>>,
    max_fails: u32,
    window: Duration,
    block_secs: u64,
    max_entries: usize,
}

#[derive(Clone, Copy)]
struct FailureRecord {
    count: u32,
    window_start: Instant,
    blocked_at: Option<Instant>,
}

impl FailureRecord {
    fn reset(now: Instant) -> Self {
        Self {
            count: 0,
            window_start: now,
            blocked_at: None,
        }
    }
}

impl LoginBrake {
    pub fn new(max_fails: u32, window_secs: u64, block_secs: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            max_fails: max_fails.max(1),
            window: Duration::from_secs(window_secs.max(1)),
            // Clamp like the proxy `AuthBrake`: a 0-second block would make
            // `block_until == blocked_at`, silently disabling blocking.
            block_secs: block_secs.max(1),
            max_entries: LOGIN_BRAKE_MAX_ENTRIES,
        }
    }

    /// Test-only constructor with an explicit map cap, used to verify bounded
    /// memory without driving 10k+ entries through the limiter.
    #[cfg(test)]
    pub(crate) fn with_max_entries(
        max_fails: u32,
        window_secs: u64,
        block_secs: u64,
        max_entries: usize,
    ) -> Self {
        let mut b = Self::new(max_fails, window_secs, block_secs);
        b.max_entries = max_entries.max(1);
        b
    }

    fn block_until(record: &FailureRecord, block_secs: u64) -> Option<Instant> {
        record.blocked_at.and_then(|at| {
            at.checked_add(Duration::from_secs(block_secs))
                .or_else(|| at.checked_add(Duration::from_secs(30 * 24 * 60 * 60)))
        })
    }

    /// Whether a record is stale enough to reclaim: a block that has elapsed,
    /// or (when not blocked) a failure window that has rolled over.
    fn is_expired(record: &FailureRecord, now: Instant, window: Duration, block_secs: u64) -> bool {
        if let Some(until) = Self::block_until(record, block_secs) {
            return now >= until;
        }
        now.duration_since(record.window_start) >= window
    }

    /// Returns `true` while the IP is currently blocked.
    ///
    /// This is a read-only check: it never inserts a new entry, so probes from
    /// previously-unseen IPs cannot themselves grow the map.
    pub fn is_blocked(&self, ip: &str) -> bool {
        // Fail closed on mutex poison: after a panic in the critical section,
        // treat every IP as blocked until process restart rather than silently
        // disabling login rate limiting.
        let Ok(mut guard) = self.inner.lock() else {
            return true;
        };
        let now = Instant::now();
        let window = self.window;
        let block_secs = self.block_secs;
        let Some(entry) = guard.get_mut(ip) else {
            return false;
        };
        if let Some(until) = Self::block_until(entry, block_secs) {
            if now < until {
                return true;
            }
            *entry = FailureRecord::reset(now);
            return false;
        }
        if now.duration_since(entry.window_start) >= window {
            *entry = FailureRecord::reset(now);
        }
        entry.count >= self.max_fails
    }

    /// Record a login failure for `ip`. When the count crosses the threshold,
    /// the IP is marked blocked.
    pub fn record_failure(&self, ip: &str) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        let now = Instant::now();
        let window = self.window;
        let block_secs = self.block_secs;
        let max_entries = self.max_entries;

        // Bound memory: once the map reaches the cap, drop expired entries. If
        // it is still full (all live/blocked), stop tracking new IPs rather
        // than evicting an offender mid-block.
        if guard.len() >= max_entries {
            guard.retain(|_, r| !Self::is_expired(r, now, window, block_secs));
        }
        if guard.len() >= max_entries && !guard.contains_key(ip) {
            return;
        }

        let entry = guard
            .entry(ip.to_string())
            .or_insert(FailureRecord::reset(now));
        if let Some(until) = Self::block_until(entry, block_secs) {
            if now < until {
                return;
            }
            *entry = FailureRecord::reset(now);
        } else if now.duration_since(entry.window_start) >= window {
            *entry = FailureRecord::reset(now);
        }
        entry.count = entry.count.saturating_add(1);
        if entry.count >= self.max_fails && entry.blocked_at.is_none() {
            entry.blocked_at = Some(now);
        }
        tracing::warn!(
        ip = %ip,
        count = entry.count,
        max = self.max_fails,
        blocked = entry.blocked_at.is_some(),
        "login failure recorded"
        );
    }

    /// Block duration in seconds, for the `Retry-After` header.
    pub fn block_secs(&self) -> u64 {
        self.block_secs
    }

    /// Remaining block time (seconds) for an IP, or 0 when not blocked.
    ///
    /// Fails closed on mutex poison (matching `is_blocked`): under poison every
    /// IP is treated as blocked, so the `Retry-After` header reflects the
    /// configured block window rather than returning 0 (which would disagree
    /// with `is_blocked` and mislead callers into thinking the block has
    /// elapsed).
    pub fn retry_after(&self, ip: &str) -> u64 {
        let Ok(guard) = self.inner.lock() else {
            return self.block_secs;
        };
        let now = Instant::now();
        let block_secs = self.block_secs;
        guard
            .get(ip)
            .and_then(|r| Self::block_until(r, block_secs))
            .filter(|until| *until > now)
            .map(|until| until.duration_since(now).as_secs().max(1))
            .unwrap_or(0)
    }

    /// Number of tracked IPs (test/diagnostics only).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod extreme_block_regressions {
    use super::*;

    #[test]
    fn an_extreme_block_duration_cannot_panic_or_disable_blocking() {
        let brake = LoginBrake::new(1, 60, u64::MAX);
        brake.record_failure("client");
        assert!(brake.is_blocked("client"));
    }
}

impl std::fmt::Debug for LoginBrake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoginBrake")
            .field("max_fails", &self.max_fails)
            .field("window", &self.window)
            .field("block_secs", &self.block_secs)
            .finish()
    }
}

/// Resolve the client IP from the request.
///
/// By default the connection peer (`ConnectInfo`) is authoritative and
/// forwarded headers are ignored — any client can spoof `X-Forwarded-For` /
/// `X-Real-Ip` to bypass the login rate brake.
///
/// When `trusted_proxies` is non-empty AND the connection peer lands inside
/// one of the configured CIDR ranges, the admin UI is presumed to sit behind
/// a trusted reverse proxy / load balancer (the documented Dockerfile
/// topology: `--listen 0.0.0.0:8080` behind a proxy). In that topology the
/// peer is always the proxy's own address, so keying the brake on the peer
/// would collapse every distinct client into a single bucket — both useless
/// against brute force and a self-DoS once the threshold trips. To recover
/// per-client attribution, walk `X-Forwarded-For` right-to-left, skipping
/// entries that are themselves trusted proxies, and use the first non-trusted
/// IP-shaped entry (the real client). The leftmost entries are spoofable, so
/// the walk stops at the first non-trusted hop and never reaches them. Falls
/// back to `X-Real-IP` (validated as an IP) when XFF is absent, then to the
/// peer IP, then to `"unknown"` when no peer is available (e.g. unit tests).
///
/// Mirrors `oxllm_server::security::client_ip` so both surfaces agree on
/// attribution.
pub fn client_ip(
    headers: &axum::http::HeaderMap,
    peer: Option<std::net::SocketAddr>,
    trusted_proxies: &[ipnet::IpNet],
) -> String {
    let peer_ip = peer.map(|sa| sa.ip());
    let peer_is_trusted = match peer_ip {
        Some(ip) => trusted_proxies.iter().any(|cidr| cidr.contains(&ip)),
        None => false,
    };
    if peer_is_trusted {
        if let Some(client) = client_from_xff(headers, trusted_proxies) {
            return client;
        }
        if let Some(xri) = headers
            .get("x-real-ip")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim())
            .filter(|s| s.parse::<std::net::IpAddr>().is_ok())
        {
            return xri.to_string();
        }
    }
    match peer_ip {
        Some(ip) => ip.to_string(),
        None => "unknown".to_string(),
    }
}

/// Walk an `X-Forwarded-For` chain right-to-left, returning the first
/// non-trusted IP-shaped entry (the real client). Returns `None` when no
/// `X-Forwarded-For` header is present or no entry parses to an IP. Duplicate
/// headers are merged into one comma-separated string first (per RFC 7230
/// §3.2.2), so a misconfigured proxy that splits XFF across two headers can't
/// hide the rightmost hop from a single `headers.get` read.
fn client_from_xff(
    headers: &axum::http::HeaderMap,
    trusted_proxies: &[ipnet::IpNet],
) -> Option<String> {
    let mut joined = String::new();
    for value in headers.get_all("x-forwarded-for") {
        // A single non-UTF-8 header must not abort the whole walk: bailing
        // here would collapse every client behind a trusted proxy into one
        // LoginBrake bucket. Skip unreadable headers instead.
        let Some(s) = value.to_str().ok() else {
            continue;
        };
        if !joined.is_empty() {
            joined.push_str(", ");
        }
        joined.push_str(s);
    }
    if joined.is_empty() {
        return None;
    }
    for hop in joined.split(',').rev() {
        let trimmed = hop.trim();
        let Ok(ip) = trimmed.parse::<std::net::IpAddr>() else {
            continue;
        };
        let trusted = trusted_proxies.iter().any(|cidr| cidr.contains(&ip));
        if !trusted {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Parse the config's `trusted_proxies` strings into `IpNet` values, skipping
/// (and warning on) any that don't parse. Empty / unset yields an empty list,
/// which restores the pre-trust peer-only behaviour.
pub fn parse_trusted_proxies(raw: &[String]) -> Vec<ipnet::IpNet> {
    raw.iter()
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return None;
            }
            match trimmed.parse::<ipnet::IpNet>() {
                Ok(net) => Some(net),
                Err(e) => {
                    tracing::warn!(cidr = %trimmed, error = %e, "ignoring unparseable trusted_proxies entry");
                    None
                }
            }
        })
        .collect()
}

/// Constant-time byte comparison re-exported for handlers and middleware.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    constant_time_eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "test-secret-very-long-and-secure";

    #[test]
    fn roundtrip_session_is_valid() {
        let s = create_session(SECRET, 1);
        assert!(verify_session(SECRET, &s.cookie_value));
    }

    #[test]
    fn parse_session_extracts_csrf_token() {
        let s = create_session(SECRET, 1);
        let info = parse_session(SECRET, &s.cookie_value).expect("valid session");
        assert_eq!(info.csrf_token, s.csrf_token);
        assert!(!info.csrf_token.is_empty());
    }

    #[test]
    fn csrf_token_is_base64_44_chars() {
        let s = create_session(SECRET, 1);
        assert_eq!(s.csrf_token.len(), 44);
        assert!(s.csrf_token.ends_with('='));
    }

    #[test]
    fn wrong_secret_rejects() {
        let s = create_session(SECRET, 1);
        assert!(!verify_session("other-secret", &s.cookie_value));
    }

    #[test]
    fn tampered_expiry_rejected() {
        let s = create_session(SECRET, 1);
        let mut parts: Vec<&str> = s.cookie_value.split('.').collect();
        parts[1] = "99999999999";
        let tampered = parts.join(".");
        assert!(!verify_session(SECRET, &tampered));
    }

    #[test]
    fn tampered_csrf_rejected() {
        let s = create_session(SECRET, 1);
        let mut parts: Vec<&str> = s.cookie_value.split('.').collect();
        parts[2] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
        let tampered = parts.join(".");
        assert!(!verify_session(SECRET, &tampered));
    }

    #[test]
    fn expired_session_rejected() {
        let mut s = create_session(SECRET, 1);
        let mut parts = s.cookie_value.split('.').collect::<Vec<_>>().into_iter();
        let session_id = parts.next().unwrap().to_string();
        let payload = format!("{session_id}.1.{}", s.csrf_token);
        let tag = sign(SECRET, payload.as_bytes());
        s.cookie_value = format!("{payload}.{tag}");
        assert!(!verify_session(SECRET, &s.cookie_value));
    }

    /// Regression guard for the HMAC-before-expiry ordering fix. Both an
    /// expired-but-genuinely-signed cookie and a tampered cookie of the same
    /// shape must be rejected; the expired path must not short-circuit before
    /// `verify_tag` runs (which would leak a passive timing signal that an
    /// attacker could use to distinguish expired-but-signed cookies from
    /// tampered ones).
    #[test]
    fn parse_session_rejects_expired_and_tampered_uniformly() {
        let s = create_session(SECRET, 1);

        // Expired-but-signed: HMAC verifies, expiry check then rejects.
        let session_id = s.cookie_value.split('.').next().unwrap().to_string();
        let expired_payload = format!("{session_id}.1.{}", s.csrf_token);
        let expired_tag = sign(SECRET, expired_payload.as_bytes());
        let expired_cookie = format!("{expired_payload}.{expired_tag}");
        assert!(
            parse_session(SECRET, &expired_cookie).is_none(),
            "expired-but-signed cookie must be rejected"
        );

        // Tampered cookie of the same length: HMAC rejects.
        let mut tampered_parts: Vec<String> =
            s.cookie_value.split('.').map(|p| p.to_string()).collect();
        // Flip the first hex byte of the tag without changing length.
        let tag = tampered_parts[3].clone();
        let first = tag.as_bytes()[0];
        let flipped = if first == b'0' { b'1' } else { b'0' };
        let mut bytes = tag.into_bytes();
        bytes[0] = flipped;
        tampered_parts[3] = String::from_utf8(bytes).unwrap();
        let tampered_cookie = tampered_parts.join(".");
        assert_eq!(tampered_cookie.len(), s.cookie_value.len());
        assert!(
            parse_session(SECRET, &tampered_cookie).is_none(),
            "tampered cookie must be rejected"
        );
    }

    #[test]
    fn malformed_cookie_rejected() {
        assert!(!verify_session(SECRET, "garbage"));
        assert!(!verify_session(SECRET, "a.b"));
        assert!(!verify_session(SECRET, "a.b.c"));
        assert!(!verify_session(SECRET, "a.b.c.d.e"));
    }

    #[test]
    fn find_cookie_extracts_value() {
        let header = "other=1; oxllm_admin_session=val-123; foo=bar";
        assert_eq!(find_cookie(header, SESSION_COOKIE), Some("val-123"));
        assert!(find_cookie("none=here", SESSION_COOKIE).is_none());
    }

    #[test]
    fn hex_roundtrip() {
        let bytes = [0u8, 255, 16, 171, 1];
        let enc = hex_encode(&bytes);
        let dec = hex_decode(enc.as_bytes()).unwrap();
        assert_eq!(dec, bytes);
        assert!(hex_decode(b"abc").is_err());
        assert!(hex_decode(b"xy").is_err());
    }

    #[test]
    fn base64_roundtrip_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn session_cookie_header_includes_attributes() {
        let s = create_session(SECRET, 1);
        let h = session_cookie_header(&s, false);
        assert!(h.contains("HttpOnly"));
        assert!(h.contains("SameSite=Strict"));
        assert!(!h.contains("Secure"));
        let h_secure = session_cookie_header(&s, true);
        assert!(h_secure.contains("Secure"));
    }

    #[test]
    fn clear_cookie_header_includes_attributes() {
        let h = clear_cookie_header(false);
        assert!(h.contains("Max-Age=0"));
        assert!(!h.contains("Secure"));
        let h_secure = clear_cookie_header(true);
        assert!(h_secure.contains("Secure"));
    }

    #[test]
    fn csrf_token_from_headers_reads_cookie() {
        let s = create_session(SECRET, 1);
        let headers = axum::http::HeaderMap::from_iter([(
            axum::http::header::COOKIE,
            format!("{}={}", SESSION_COOKIE, s.cookie_value)
                .parse()
                .unwrap(),
        )]);
        assert_eq!(
            csrf_token_from_headers(&headers, SECRET),
            Some(s.csrf_token)
        );
    }

    #[test]
    fn login_brake_blocks_after_threshold_and_recovers() {
        let brake = LoginBrake::new(3, 60, 300);
        for _ in 0..3 {
            assert!(!brake.is_blocked("1.1.1.1"));
            brake.record_failure("1.1.1.1");
        }
        assert!(brake.is_blocked("1.1.1.1"));
        assert!(brake.retry_after("1.1.1.1") > 0);
        let other = "2.2.2.2";
        assert!(!brake.is_blocked(other));
        assert_eq!(brake.retry_after(other), 0);
    }

    #[test]
    fn login_brake_clamps_zero_block_secs() {
        let brake = LoginBrake::new(1, 60, 0);
        assert_eq!(brake.block_secs(), 1, "block_secs=0 must clamp to 1");
        brake.record_failure("1.1.1.1");
        assert!(
            brake.is_blocked("1.1.1.1"),
            "clamped block must actually hold the IP"
        );
    }

    #[test]
    fn login_brake_retry_after_fails_closed_on_poison() {
        let brake = LoginBrake::new(3, 60, 300);
        // Poison the shared inner mutex by panicking while it is held. Cloning
        // the `Arc` gives the thread a handle to the *same* mutex, so the
        // poison is visible through `brake` after the thread dies.
        let inner = brake.inner.clone();
        let handle = std::thread::spawn(move || {
            let _guard = inner.lock().unwrap();
            panic!("intentional poison for test");
        });
        let _ = handle.join();
        // Under poison `is_blocked` returns true (fail-closed); `retry_after`
        // must agree by returning a non-zero value (the configured block
        // window), not 0.
        assert!(brake.is_blocked("1.1.1.1"), "is_blocked must fail closed");
        let retry = brake.retry_after("1.1.1.1");
        assert!(
            retry > 0,
            "retry_after must fail closed (non-zero) under poison, got {retry}"
        );
    }

    #[test]
    fn login_brake_distinct_ips_independent() {
        let brake = LoginBrake::new(2, 60, 300);
        brake.record_failure("1.1.1.1");
        brake.record_failure("2.2.2.2");
        assert!(!brake.is_blocked("1.1.1.1"));
        assert!(!brake.is_blocked("2.2.2.2"));
        brake.record_failure("1.1.1.1");
        assert!(brake.is_blocked("1.1.1.1"));
        assert!(!brake.is_blocked("2.2.2.2"));
    }

    #[test]
    fn is_blocked_does_not_grow_map_for_unseen_ips() {
        let brake = LoginBrake::new(5, 60, 300);
        for i in 0..1000 {
            let _ = brake.is_blocked(&format!("10.0.0.{i}"));
        }
        assert_eq!(brake.len(), 0);
    }

    #[test]
    fn record_failure_caps_map_size_under_many_distinct_ips() {
        let cap = 50;
        let brake = LoginBrake::with_max_entries(5, 60, 300, cap);
        for i in 0..5_000u32 {
            brake.record_failure(&format!("10.{}.{}.{}", i / 65536, (i / 256) % 256, i % 256));
        }
        assert!(
            brake.len() <= cap,
            "map grew to {} (cap {cap})",
            brake.len()
        );
        assert!(brake.len() > 0);
    }

    #[test]
    fn client_ip_uses_peer_when_no_trusted_proxies() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-real-ip", "9.9.9.9".parse().unwrap());
        headers.insert("x-forwarded-for", "8.8.8.8, 10.0.0.1".parse().unwrap());
        let peer: std::net::SocketAddr = "203.0.113.5:443".parse().unwrap();
        // No trusted proxies: spoofed headers must be ignored, peer wins.
        assert_eq!(client_ip(&headers, Some(peer), &[]), "203.0.113.5");
    }

    #[test]
    fn client_ip_returns_unknown_without_peer() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-forwarded-for", "8.8.8.8".parse().unwrap());
        assert_eq!(client_ip(&headers, None, &[]), "unknown");
    }

    #[test]
    fn client_ip_honours_xff_when_peer_is_trusted() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut headers = axum::http::HeaderMap::new();
        // Right-most non-trusted entry is the real client (9.9.9.9); the
        // trailing 10.0.0.1 hop is itself a trusted proxy and is skipped.
        headers.insert(
            "x-forwarded-for",
            "8.8.8.8, 9.9.9.9, 10.0.0.1".parse().unwrap(),
        );
        let peer: std::net::SocketAddr = "10.0.0.1:443".parse().unwrap();
        assert_eq!(client_ip(&headers, Some(peer), &trusted), "9.9.9.9");
    }

    #[test]
    fn client_ip_falls_back_to_x_real_ip_when_xff_absent() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-real-ip", "198.51.100.7".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.1:443".parse().unwrap();
        assert_eq!(client_ip(&headers, Some(peer), &trusted), "198.51.100.7");
    }

    #[test]
    fn client_ip_ignores_spoofed_xff_when_peer_untrusted() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-forwarded-for", "9.9.9.9".parse().unwrap());
        // Peer is from a public range, not in trusted_proxies: XFF must be
        // ignored so an attacker cannot pick their own brake bucket.
        let peer: std::net::SocketAddr = "203.0.113.7:443".parse().unwrap();
        assert_eq!(client_ip(&headers, Some(peer), &trusted), "203.0.113.7");
    }

    #[test]
    fn client_ip_rejects_non_ip_xff_and_falls_back_to_peer() {
        let trusted: Vec<ipnet::IpNet> = vec!["10.0.0.0/8".parse().unwrap()];
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-forwarded-for", "not-an-ip, still-bogus".parse().unwrap());
        let peer: std::net::SocketAddr = "10.0.0.1:443".parse().unwrap();
        assert_eq!(client_ip(&headers, Some(peer), &trusted), "10.0.0.1");
    }

    #[test]
    fn parse_trusted_proxies_skips_invalid_entries() {
        let raw = vec![
            "10.0.0.0/8".to_string(),
            "".to_string(),
            "not-a-cidr".to_string(),
            "  127.0.0.0/8  ".to_string(),
        ];
        let parsed = parse_trusted_proxies(&raw);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].to_string(), "10.0.0.0/8");
        assert_eq!(parsed[1].to_string(), "127.0.0.0/8");
    }

    #[test]
    fn login_csrf_roundtrip_and_rejection() {
        let (token, cookie) = create_login_csrf(SECRET, false);
        let value = cookie.split(';').next().unwrap();
        let headers = axum::http::HeaderMap::from_iter([(
            axum::http::header::COOKIE,
            format!("other=1; {value}").parse().unwrap(),
        )]);
        assert!(login_csrf_ok(&headers, SECRET, &token));
        assert!(!login_csrf_ok(&headers, SECRET, "wrong-token"));
        assert!(!login_csrf_ok(&headers, "other-secret", &token));
        // Missing cookie -> rejected.
        let empty = axum::http::HeaderMap::new();
        assert!(!login_csrf_ok(&empty, SECRET, &token));
        // Tampered cookie value -> rejected.
        let tampered = format!("{}.deadbeef", token);
        let headers = axum::http::HeaderMap::from_iter([(
            axum::http::header::COOKIE,
            format!("{LOGIN_CSRF_COOKIE}={tampered}").parse().unwrap(),
        )]);
        assert!(!login_csrf_ok(&headers, SECRET, &token));
    }

    #[test]
    fn ct_eq_handles_lengths_and_equality() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"abcd"));
        assert!(ct_eq(b"", b""));
    }

    /// A single non-UTF-8 X-Forwarded-For header must not abort the XFF
    /// walk: bailing there collapses every client behind a trusted proxy
    /// into one LoginBrake bucket.
    #[test]
    fn client_from_xff_skips_non_utf8_header() {
        use axum::http::HeaderValue;
        let mut headers = axum::http::HeaderMap::new();
        headers.append(
            "x-forwarded-for",
            HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
        );
        headers.append(
            "x-forwarded-for",
            HeaderValue::from_static("203.0.113.7, 10.0.0.1"),
        );
        let trusted = parse_trusted_proxies(&["10.0.0.0/8".to_string()]);
        assert_eq!(
            client_from_xff(&headers, &trusted),
            Some("203.0.113.7".to_string())
        );
    }
}
