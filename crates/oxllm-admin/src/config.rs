//! Admin server configuration.
//!
//! Loaded once at startup from a YAML file (with `os.environ/` resolution,
//! matching the proxy config pattern) and/or environment variables. Environment
//! variables override file values; `os.environ/` inside the file is resolved at
//! load time only.

use anyhow::{Context, Result, bail};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Admin server configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct AdminConfig {
    /// Bind address, e.g. `"127.0.0.1:8080"`.
    pub listen: String,
    /// Proxy management API base URL, e.g. `"http://127.0.0.1:4000"`.
    /// HTTP(S) only; URL credentials, query strings and fragments are rejected.
    /// Use HTTPS for remote endpoints; private-network HTTP remains supported.
    pub proxy_url: String,
    /// Master key used to authenticate against the proxy. Server-side only.
    #[serde(default)]
    #[serde(skip_serializing)]
    pub master_key: String,
    /// Secret used to sign session cookies. Generated at load time if absent.
    #[serde(default)]
    #[serde(skip_serializing)]
    pub session_secret: String,
    /// Session lifetime in hours.
    #[serde(default = "default_ttl")]
    pub session_ttl_hours: u64,
    /// Optional login password gate. When `None`, login is auto-accepted.
    #[serde(default)]
    #[serde(skip_serializing)]
    pub login_password: Option<String>,
    /// Trusted proxy CIDRs for `X-Forwarded-*` parsing (future use).
    #[serde(default)]
    pub trusted_proxies: Vec<String>,
    /// When `true`, session cookies are emitted with the `Secure` attribute so
    /// browsers send them only over HTTPS. Defaults to `true` (safer-by-
    /// default). Browser-facing HTTPS is required outside browser localhost
    /// exceptions, even if the reverse proxy's backend connection is HTTP.
    /// For trusted HTTP-only development, explicitly set
    /// `ADMIN_COOKIE_SECURE=false` (YAML: `cookie_secure: false`). Forwarded
    /// headers never automatically weaken this setting.
    #[serde(default = "default_cookie_secure")]
    pub cookie_secure: bool,
    /// Per-IP login attempt limit before the IP is blocked.
    #[serde(default = "default_login_max_fails")]
    pub login_max_fails: u32,
    /// Window (seconds) in which `login_max_fails` failures trigger a block.
    #[serde(default = "default_login_window_secs")]
    pub login_window_secs: u64,
    /// Block duration (seconds) once the failure threshold is crossed.
    #[serde(default = "default_login_block_secs")]
    pub login_block_secs: u64,
}

/// Manual `Debug` impl that redacts secret fields. The `serde(skip_serializing)`
/// attributes protect the secrets from JSON/YAML serialization, but a derived
/// `Debug` would print `master_key`, `session_secret`, and `login_password` in
/// plaintext — one `tracing::debug!(?config)` away from leaking the master key
/// (which authenticates every management call) and the session HMAC secret
/// (which lets an attacker forge admin sessions). All three are replaced with
/// `"<redacted>"` here so the asymmetry between `skip_serializing` and `Debug`
/// cannot bite later.
impl std::fmt::Debug for AdminConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdminConfig")
            .field("listen", &self.listen)
            .field("proxy_url", &self.proxy_url)
            .field("master_key", &"<redacted>")
            .field("session_secret", &"<redacted>")
            .field("session_ttl_hours", &self.session_ttl_hours)
            .field(
                "login_password",
                &self.login_password.as_ref().map(|_| "<redacted>"),
            )
            .field("trusted_proxies", &self.trusted_proxies)
            .field("cookie_secure", &self.cookie_secure)
            .field("login_max_fails", &self.login_max_fails)
            .field("login_window_secs", &self.login_window_secs)
            .field("login_block_secs", &self.login_block_secs)
            .finish()
    }
}

fn default_login_max_fails() -> u32 {
    5
}

fn default_cookie_secure() -> bool {
    true
}

fn default_login_window_secs() -> u64 {
    60
}

fn default_login_block_secs() -> u64 {
    300
}

fn default_ttl() -> u64 {
    24
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8080".to_string(),
            proxy_url: "http://127.0.0.1:4000".to_string(),
            master_key: String::new(),
            session_secret: String::new(),
            session_ttl_hours: default_ttl(),
            login_password: None,
            trusted_proxies: Vec::new(),
            cookie_secure: default_cookie_secure(),
            login_max_fails: default_login_max_fails(),
            login_window_secs: default_login_window_secs(),
            login_block_secs: default_login_block_secs(),
        }
    }
}

/// Load admin config from a YAML file (if `path` is given and exists) and then
/// apply environment-variable overrides.
pub fn load(path: Option<&str>) -> Result<AdminConfig> {
    let mut cfg = match path {
        Some(p) => load_file(p)?,
        None => AdminConfig::default(),
    };

    apply_env_overrides(&mut cfg);

    if cfg.master_key.trim().is_empty() {
        bail!("admin config: `master_key` is required (set OXLLM_MASTER_KEY)");
    }
    cfg.proxy_url = validate_proxy_url(&cfg.proxy_url)?.to_string();
    if cfg.session_secret.trim().is_empty() {
        cfg.session_secret = generate_secret();
        tracing::warn!(
            "admin config: `session_secret` not set — generated a random one. \
 Set ADMIN_SESSION_SECRET for stable sessions across restarts."
        );
    } else if cfg.session_secret.len() < 16 {
        bail!(
            "admin config: `session_secret` is too short ({} chars); \
 require at least 16 characters (set ADMIN_SESSION_SECRET)",
            cfg.session_secret.len()
        );
    }

    // Refuse to start unauthenticated on a non-loopback bind.
    enforce_bind_password_guard(&cfg)?;

    Ok(cfg)
}

/// Validate before constructing the bearer-authenticated client. Error messages
/// intentionally omit the raw URL: a mistaken userinfo/query value may contain
/// a credential. Base paths are allowed for reverse-proxied private deployments.
pub(crate) fn validate_proxy_url(raw: &str) -> Result<reqwest::Url> {
    if raw.trim().is_empty() {
        bail!("admin config: `proxy_url` is required (set OXLLM_PROXY_URL)");
    }
    let url = reqwest::Url::parse(raw.trim()).map_err(|_| {
        anyhow::anyhow!("admin config: `proxy_url` must be an absolute HTTP(S) URL")
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        bail!("admin config: `proxy_url` must be an absolute HTTP(S) URL");
    }
    let authority = raw
        .trim()
        .split_once("://")
        .map(|(_, rest)| rest.split(['/', '?', '#']).next().unwrap_or(""));
    if !url.username().is_empty()
        || url.password().is_some()
        || authority.is_some_and(|value| value.contains('@'))
    {
        bail!(
            "admin config: `proxy_url` must not contain URL credentials; \
             set OXLLM_MASTER_KEY for management authentication"
        );
    }
    if url.query().is_some() || url.fragment().is_some() {
        bail!("admin config: `proxy_url` must not contain a query string or fragment");
    }
    Ok(url)
}

/// No DNS is performed for a diagnostic. An HTTP hostname may resolve privately;
/// do not reject it or mistake it for a proven public address.
pub(crate) fn proxy_transport_warning(url: &reqwest::Url) -> Option<&'static str> {
    if url.scheme() != "http" {
        return None;
    }
    let host = url.host_str().unwrap_or("").trim_matches(['[', ']']);
    let local = host.eq_ignore_ascii_case("localhost")
        || match host.parse::<std::net::IpAddr>() {
            Ok(std::net::IpAddr::V4(ip)) => {
                ip.is_loopback() || ip.is_private() || ip.is_link_local()
            }
            Ok(std::net::IpAddr::V6(ip)) => {
                ip.is_loopback() || ip.is_unique_local() || ip.is_unicast_link_local()
            }
            Err(_) => false,
        };
    if local {
        None
    } else {
        Some(
            "proxy_url uses HTTP: management credentials travel in cleartext. \
             Use HTTPS for remote endpoints, or ensure this hostname resolves \
             within a trusted private network.",
        )
    }
}

pub(crate) fn log_transport_diagnostics(cfg: &AdminConfig, proxy_url: &reqwest::Url) {
    if cfg.cookie_secure {
        tracing::info!(
            "Secure admin cookies are enabled. Use browser-facing HTTPS; \
             repeated sign-in expiry over HTTP can mean the cookie was blocked. \
             For trusted HTTP-only development, set ADMIN_COOKIE_SECURE=false \
             (YAML: cookie_secure: false)."
        );
    } else {
        tracing::warn!(
            "Secure admin cookies are explicitly disabled; use this setting \
             only for trusted HTTP deployments. Set ADMIN_COOKIE_SECURE=true \
             for browser-facing HTTPS."
        );
    }
    if let Some(message) = proxy_transport_warning(proxy_url) {
        tracing::warn!("{message}");
    }
}

/// Refuse to run unauthenticated on a non-loopback bind. Without a password,
/// any login POST succeeds (see app.rs login handler) and the UI is open.
///
/// This runs inside [`load`] after file/env resolution, and **again** from
/// `crate::serve` after CLI overrides are applied — otherwise `--listen
/// 0.0.0.0:8080` (as hard-coded in `Dockerfile.admin`) would bypass the
/// check, since the override lands after `load` has already validated.
pub(crate) fn enforce_bind_password_guard(cfg: &AdminConfig) -> Result<()> {
    let password_unset = cfg
        .login_password
        .as_deref()
        .map(|p| p.trim().is_empty())
        .unwrap_or(true);
    if password_unset && !is_loopback_listen(&cfg.listen) {
        bail!(
            "admin config: `login_password` is required when binding to a \
 non-loopback address ({:?}). Set ADMIN_UI_PASSWORD, or bind to 127.0.0.1.",
            cfg.listen
        );
    }
    Ok(())
}

/// Returns `true` if the listen address binds to a loopback interface only.
///
/// Parses the listen string as a `SocketAddr` first, which handles bracketed
/// and bare IPv6 forms correctly (e.g. `--listen ::1` parses to host `::1`,
/// whereas the old `rsplit_once(':')` heuristic yielded `::` and wrongly
/// classified a genuine loopback bind as exposed). Falls back to a hostname
/// check (`localhost`, `127.x.x.x`) for strings that don't parse (e.g.
/// `localhost:8080`, or a host the resolver would handle).
fn is_loopback_listen(listen: &str) -> bool {
    if let Ok(addr) = listen.trim().parse::<std::net::SocketAddr>() {
        return addr.ip().is_loopback();
    }
    // Try to complete a bare IPv6 literal like `::1` into a SocketAddr
    // (bracketed so the port split is unambiguous; `::1:0` alone would be
    // parsed as the IPv6 address ::1:0, not host ::1 port 0).
    if let Ok(addr) = format!("[{listen}]:0").parse::<std::net::SocketAddr>() {
        return addr.ip().is_loopback();
    }
    let host = listen.rsplit_once(':').map(|(h, _)| h).unwrap_or(listen);
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod bind_guard_regressions {
    use super::*;

    #[test]
    fn a_hostname_beginning_with_127_is_not_a_loopback_bind() {
        let config = AdminConfig {
            listen: "127.0.0.1.evil.test:8080".into(),
            ..Default::default()
        };
        assert!(enforce_bind_password_guard(&config).is_err());
        assert!(is_loopback_listen("127.0.0.2:8080"));
    }
}

fn load_file(path: &str) -> Result<AdminConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read admin config file: {path}"))?;
    parse_str(&content).with_context(|| format!("failed to parse admin config from: {path}"))
}

/// Parse a config from a YAML string, resolving `os.environ/` references once.
pub fn parse_str(yaml: &str) -> Result<AdminConfig> {
    let mut value: Value = serde_yml::from_str(yaml).context("failed to parse YAML")?;
    resolve_env_vars(&mut value)?;
    serde_json::from_value(value).context("failed to deserialize admin config")
}

fn apply_env_overrides(cfg: &mut AdminConfig) {
    if let Ok(v) = std::env::var("OXLLM_ADMIN_LISTEN") {
        if !v.trim().is_empty() {
            cfg.listen = v;
        }
    }
    if let Ok(v) = std::env::var("OXLLM_PROXY_URL") {
        if !v.trim().is_empty() {
            cfg.proxy_url = v;
        }
    }
    if let Ok(v) = std::env::var("OXLLM_MASTER_KEY") {
        if !v.trim().is_empty() {
            cfg.master_key = v;
        }
    }
    if let Ok(v) = std::env::var("ADMIN_SESSION_SECRET") {
        if !v.trim().is_empty() {
            cfg.session_secret = v;
        }
    }
    if let Ok(v) = std::env::var("ADMIN_UI_PASSWORD") {
        cfg.login_password = Some(v);
    }
    if let Ok(v) = std::env::var("ADMIN_COOKIE_SECURE") {
        cfg.cookie_secure = parse_bool(&v).unwrap_or(default_cookie_secure());
    }
    if let Ok(v) = std::env::var("ADMIN_LOGIN_MAX_FAILS") {
        if let Ok(n) = v.parse::<u32>() {
            if n > 0 {
                cfg.login_max_fails = n;
            }
        }
    }
    if let Ok(v) = std::env::var("ADMIN_LOGIN_WINDOW_SECS") {
        if let Ok(n) = v.parse::<u64>() {
            if n > 0 {
                cfg.login_window_secs = n;
            }
        }
    }
    if let Ok(v) = std::env::var("ADMIN_LOGIN_BLOCK_SECS") {
        if let Ok(n) = v.parse::<u64>() {
            if n > 0 {
                cfg.login_block_secs = n;
            }
        }
    }
}

fn parse_bool(value: &str) -> Result<bool, ()> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(()),
    }
}

fn generate_secret() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Recursively resolve `os.environ/NAME` string references in the JSON tree.
fn resolve_env_vars(value: &mut Value) -> Result<()> {
    let mut errors: Vec<String> = Vec::new();
    resolve_inner(value, String::new(), &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        bail!(
            "admin config references unset environment variable at: {}",
            errors.join(", ")
        );
    }
}

fn resolve_inner(value: &mut Value, path: String, errors: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                resolve_inner(val, child_path, errors);
            }
        }
        Value::Array(arr) => {
            for (i, val) in arr.iter_mut().enumerate() {
                resolve_inner(val, format!("{path}[{i}]"), errors);
            }
        }
        Value::String(s) => {
            if let Some(env_name) = s.strip_prefix("os.environ/") {
                match std::env::var(env_name) {
                    Ok(val) => {
                        *s = val;
                    }
                    Err(_) => {
                        errors.push(path);
                    }
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[allow(unsafe_code)]
    fn set_env(name: &str, value: &str) {
        unsafe { std::env::set_var(name, value) };
    }

    #[allow(unsafe_code)]
    fn unset_env(name: &str) {
        unsafe { std::env::remove_var(name) };
    }

    #[test]
    fn parses_yaml_config() {
        let _g = ENV_LOCK.lock().unwrap();
        unset_env("OXLLM_ADMIN_LISTEN");
        unset_env("OXLLM_PROXY_URL");
        unset_env("OXLLM_MASTER_KEY");
        unset_env("ADMIN_SESSION_SECRET");
        unset_env("ADMIN_UI_PASSWORD");
        unset_env("ADMIN_COOKIE_SECURE");
        unset_env("ADMIN_LOGIN_MAX_FAILS");
        unset_env("ADMIN_LOGIN_WINDOW_SECS");
        unset_env("ADMIN_LOGIN_BLOCK_SECS");
        let yaml = "\
listen: \"0.0.0.0:9999\"
proxy_url: \"http://localhost:4000\"
master_key: sk-test-master
session_secret: abcdef
session_ttl_hours: 12
";
        let cfg = parse_str(yaml).expect("config should parse");
        assert_eq!(cfg.listen, "0.0.0.0:9999");
        assert_eq!(cfg.proxy_url, "http://localhost:4000");
        assert_eq!(cfg.master_key, "sk-test-master");
        assert_eq!(cfg.session_secret, "abcdef");
        assert_eq!(cfg.session_ttl_hours, 12);
        assert!(cfg.login_password.is_none());
        assert!(cfg.trusted_proxies.is_empty());
        assert!(cfg.cookie_secure);
        assert_eq!(cfg.login_max_fails, 5);
        assert_eq!(cfg.login_window_secs, 60);
        assert_eq!(cfg.login_block_secs, 300);
    }

    #[test]
    fn resolves_os_environ_references() {
        let _g = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_ADMIN_TEST_KEY", "sk-resolved-xyz");
        let yaml = "master_key: os.environ/OXLLM_ADMIN_TEST_KEY\nproxy_url: http://p:4000\nlisten: 127.0.0.1:8080\n";
        let cfg = parse_str(yaml).expect("env ref should resolve");
        assert_eq!(cfg.master_key, "sk-resolved-xyz");
        unset_env("OXLLM_ADMIN_TEST_KEY");
    }

    #[test]
    fn missing_env_ref_reports_path() {
        let _g = ENV_LOCK.lock().unwrap();
        unset_env("OXLLM_ADMIN_DEFINITELY_MISSING");
        let yaml = "master_key: os.environ/OXLLM_ADMIN_DEFINITELY_MISSING\nproxy_url: http://p:4000\nlisten: 127.0.0.1:8080\n";
        let err = parse_str(yaml).expect_err("missing env should error");
        assert!(err.to_string().contains("master_key"));
        assert!(
            !err.to_string().contains("OXLLM_ADMIN_DEFINITELY_MISSING"),
            "must not leak env var name"
        );
    }

    #[test]
    fn env_overrides_take_precedence() {
        let _g = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_ADMIN_LISTEN", "1.2.3.4:5555");
        set_env("OXLLM_MASTER_KEY", "sk-from-env");
        set_env("ADMIN_SESSION_SECRET", "from-env-secret");
        let mut c = AdminConfig {
            master_key: "placeholder".to_string(),
            session_secret: "placeholder".to_string(),
            ..AdminConfig::default()
        };
        apply_env_overrides(&mut c);
        assert_eq!(c.listen, "1.2.3.4:5555");
        assert_eq!(c.master_key, "sk-from-env");
        assert_eq!(c.session_secret, "from-env-secret");
        unset_env("OXLLM_ADMIN_LISTEN");
        unset_env("OXLLM_MASTER_KEY");
        unset_env("ADMIN_SESSION_SECRET");
    }

    #[test]
    fn missing_master_key_is_fatal() {
        let _g = ENV_LOCK.lock().unwrap();
        unset_env("OXLLM_MASTER_KEY");
        unset_env("ADMIN_SESSION_SECRET");
        unset_env("OXLLM_PROXY_URL");
        unset_env("OXLLM_ADMIN_LISTEN");
        unset_env("ADMIN_UI_PASSWORD");
        unset_env("ADMIN_COOKIE_SECURE");
        unset_env("ADMIN_LOGIN_MAX_FAILS");
        unset_env("ADMIN_LOGIN_WINDOW_SECS");
        unset_env("ADMIN_LOGIN_BLOCK_SECS");
        let err = load(None).expect_err("missing master_key must error");
        assert!(err.to_string().contains("master_key"));
    }

    #[test]
    fn missing_session_secret_is_generated() {
        let _g = ENV_LOCK.lock().unwrap();
        set_env("OXLLM_MASTER_KEY", "sk-x");
        unset_env("ADMIN_SESSION_SECRET");
        unset_env("OXLLM_PROXY_URL");
        unset_env("OXLLM_ADMIN_LISTEN");
        unset_env("ADMIN_UI_PASSWORD");
        unset_env("ADMIN_COOKIE_SECURE");
        let cfg = load(None).expect("should load with generated secret");
        assert!(!cfg.session_secret.is_empty(), "secret must be generated");
        unset_env("OXLLM_MASTER_KEY");
    }

    #[test]
    fn env_overrides_set_cookie_secure_and_login_limits() {
        let _g = ENV_LOCK.lock().unwrap();
        set_env("ADMIN_COOKIE_SECURE", "true");
        set_env("ADMIN_LOGIN_MAX_FAILS", "7");
        set_env("ADMIN_LOGIN_WINDOW_SECS", "120");
        set_env("ADMIN_LOGIN_BLOCK_SECS", "600");
        let mut c = AdminConfig {
            master_key: "placeholder".to_string(),
            session_secret: "placeholder".to_string(),
            ..AdminConfig::default()
        };
        apply_env_overrides(&mut c);
        assert!(c.cookie_secure);
        assert_eq!(c.login_max_fails, 7);
        assert_eq!(c.login_window_secs, 120);
        assert_eq!(c.login_block_secs, 600);
        unset_env("ADMIN_COOKIE_SECURE");
        unset_env("ADMIN_LOGIN_MAX_FAILS");
        unset_env("ADMIN_LOGIN_WINDOW_SECS");
        unset_env("ADMIN_LOGIN_BLOCK_SECS");
    }

    #[test]
    fn env_overrides_reject_zero_login_limits() {
        let _g = ENV_LOCK.lock().unwrap();
        set_env("ADMIN_LOGIN_MAX_FAILS", "0");
        set_env("ADMIN_LOGIN_WINDOW_SECS", "0");
        set_env("ADMIN_LOGIN_BLOCK_SECS", "0");
        let mut c = AdminConfig {
            master_key: "placeholder".to_string(),
            session_secret: "placeholder".to_string(),
            ..AdminConfig::default()
        };
        apply_env_overrides(&mut c);
        assert_eq!(c.login_max_fails, 5, "0 must be ignored (default stays)");
        assert_eq!(c.login_window_secs, 60);
        assert_eq!(
            c.login_block_secs, 300,
            "ADMIN_LOGIN_BLOCK_SECS=0 must be rejected for consistency with siblings"
        );
        unset_env("ADMIN_LOGIN_MAX_FAILS");
        unset_env("ADMIN_LOGIN_WINDOW_SECS");
        unset_env("ADMIN_LOGIN_BLOCK_SECS");
    }

    #[test]
    fn parse_bool_accepts_common_forms() {
        assert_eq!(parse_bool("true"), Ok(true));
        assert_eq!(parse_bool("1"), Ok(true));
        assert_eq!(parse_bool("YES"), Ok(true));
        assert_eq!(parse_bool("off"), Ok(false));
        assert_eq!(parse_bool("0"), Ok(false));
        assert!(parse_bool("maybe").is_err());
    }

    #[test]
    fn proxy_urls_allow_private_http_and_base_paths() {
        for raw in [
            "http://127.0.0.1:4000",
            "http://[::1]:4000",
            "http://10.1.2.3:4000/gateway/",
            "http://proxy:4000",
            "https://proxy.example.test/management/",
            "http://203.0.113.1:4000",
        ] {
            assert!(validate_proxy_url(raw).is_ok(), "rejected {raw}");
        }
    }

    #[test]
    fn proxy_urls_reject_invalid_schemes_credentials_and_suffixes_without_leaking_them() {
        for raw in [
            "",
            "/proxy",
            "ftp://proxy.example.test",
            "file:///tmp/socket",
            "http://user:secret-proxy-password@proxy.example.test",
            "https://user@proxy.example.test",
            "https://@proxy.example.test",
            "https://proxy.example.test?token=secret-proxy-password",
            "https://proxy.example.test#secret-proxy-password",
        ] {
            let err = validate_proxy_url(raw).expect_err("invalid proxy URL accepted");
            assert!(err.to_string().contains("proxy_url"));
            assert!(!err.to_string().contains("secret-proxy-password"));
        }
    }

    #[test]
    fn cleartext_remote_urls_warn_without_rejecting_private_deployments() {
        for raw in [
            "http://localhost:4000",
            "http://127.0.0.2:4000",
            "http://[::1]:4000",
            "http://10.1.2.3:4000",
            "http://192.168.2.1:4000",
            "http://[fd00::1]:4000",
            "https://proxy.example.test",
        ] {
            assert!(proxy_transport_warning(&validate_proxy_url(raw).unwrap()).is_none());
        }
        for raw in ["http://203.0.113.1:4000", "http://proxy:4000"] {
            let url = validate_proxy_url(raw).expect("HTTP must remain supported");
            let warning = proxy_transport_warning(&url).expect("cleartext diagnostic");
            assert!(warning.contains("HTTPS"));
            assert!(warning.contains("trusted private network"));
        }
    }

    #[test]
    fn serve_revalidates_proxy_url_after_cli_override() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_admin_env();
        set_env("OXLLM_MASTER_KEY", "sk-test");
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let result = rt.block_on(crate::serve(
            None,
            None,
            Some("https://user:secret-proxy-password@proxy.example.test".to_string()),
        ));
        unset_env("OXLLM_MASTER_KEY");
        let err = result.expect_err("CLI override must be validated before binding");
        assert!(err.to_string().contains("proxy_url"));
        assert!(!err.to_string().contains("secret-proxy-password"));
    }

    #[test]
    fn debug_redacts_all_secret_fields() {
        let cfg = AdminConfig {
            listen: "127.0.0.1:8080".to_string(),
            proxy_url: "http://127.0.0.1:4000".to_string(),
            master_key: "sk-super-secret-master".to_string(),
            session_secret: "this-is-the-session-hmac-secret".to_string(),
            login_password: Some("hunter2-login-pw".to_string()),
            ..AdminConfig::default()
        };
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("AdminConfig"), "debug should name the struct");
        assert!(
            dbg.contains("<redacted>"),
            "debug should show redaction marker"
        );
        assert!(
            !dbg.contains("sk-super-secret-master"),
            "master_key leaked into Debug: {dbg}"
        );
        assert!(
            !dbg.contains("this-is-the-session-hmac-secret"),
            "session_secret leaked into Debug: {dbg}"
        );
        assert!(
            !dbg.contains("hunter2-login-pw"),
            "login_password leaked into Debug: {dbg}"
        );
        // Non-secret fields remain visible.
        assert!(dbg.contains("127.0.0.1:8080"));
    }

    #[test]
    fn debug_redacts_secrets_when_login_password_is_none() {
        let cfg = AdminConfig {
            master_key: "sk-secret".to_string(),
            session_secret: "session-secret".to_string(),
            login_password: None,
            ..AdminConfig::default()
        };
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("sk-secret"), "master_key leaked: {dbg}");
        assert!(
            !dbg.contains("session-secret"),
            "session_secret leaked: {dbg}"
        );
    }

    fn clear_admin_env() {
        unset_env("OXLLM_ADMIN_LISTEN");
        unset_env("OXLLM_PROXY_URL");
        unset_env("OXLLM_MASTER_KEY");
        unset_env("ADMIN_SESSION_SECRET");
        unset_env("ADMIN_UI_PASSWORD");
        unset_env("ADMIN_COOKIE_SECURE");
        unset_env("ADMIN_LOGIN_MAX_FAILS");
        unset_env("ADMIN_LOGIN_WINDOW_SECS");
        unset_env("ADMIN_LOGIN_BLOCK_SECS");
    }

    #[test]
    fn config_rejects_empty_password_on_non_loopback() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_admin_env();
        set_env("OXLLM_MASTER_KEY", "sk-test");
        set_env("OXLLM_ADMIN_LISTEN", "0.0.0.0:8080");
        // password intentionally unset
        let err = load(None).expect_err("must refuse unauthenticated non-loopback bind");
        let msg = err.to_string();
        assert!(msg.contains("login_password"), "msg: {msg}");
        unset_env("OXLLM_MASTER_KEY");
        unset_env("OXLLM_ADMIN_LISTEN");
    }

    #[test]
    fn config_accepts_empty_password_on_loopback() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_admin_env();
        set_env("OXLLM_MASTER_KEY", "sk-test");
        set_env("OXLLM_ADMIN_LISTEN", "127.0.0.1:8080");
        let cfg = load(None).expect("loopback bind without password is fine");
        assert_eq!(cfg.listen, "127.0.0.1:8080");
        assert!(cfg.login_password.is_none());
        unset_env("OXLLM_MASTER_KEY");
        unset_env("OXLLM_ADMIN_LISTEN");
    }

    #[test]
    fn config_accepts_password_on_non_loopback() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_admin_env();
        set_env("OXLLM_MASTER_KEY", "sk-test");
        set_env("OXLLM_ADMIN_LISTEN", "0.0.0.0:8080");
        set_env("ADMIN_UI_PASSWORD", "s3cret");
        let cfg = load(None).expect("password + non-loopback bind should load");
        assert_eq!(cfg.listen, "0.0.0.0:8080");
        assert_eq!(cfg.login_password.as_deref(), Some("s3cret"));
        unset_env("OXLLM_MASTER_KEY");
        unset_env("OXLLM_ADMIN_LISTEN");
        unset_env("ADMIN_UI_PASSWORD");
    }

    #[test]
    fn config_rejects_empty_string_password_on_non_loopback() {
        // `ADMIN_UI_PASSWORD=` (empty) from docker-compose must NOT be accepted
        // as a valid password on a non-loopback bind.
        let _g = ENV_LOCK.lock().unwrap();
        clear_admin_env();
        set_env("OXLLM_MASTER_KEY", "sk-test");
        set_env("OXLLM_ADMIN_LISTEN", "0.0.0.0:8080");
        set_env("ADMIN_UI_PASSWORD", "");
        let err = load(None).expect_err("empty password must be rejected on non-loopback");
        assert!(err.to_string().contains("login_password"));
        unset_env("OXLLM_MASTER_KEY");
        unset_env("OXLLM_ADMIN_LISTEN");
        unset_env("ADMIN_UI_PASSWORD");
    }

    #[test]
    fn is_loopback_listen_classifies_correctly() {
        assert!(is_loopback_listen("127.0.0.1:8080"));
        assert!(is_loopback_listen("127.0.0.2:8080"));
        assert!(is_loopback_listen("[::1]:8080"));
        assert!(is_loopback_listen("localhost:8080"));
        // Bare IPv6 loopback (no port): the old rsplit_once(':') heuristic
        // saw host "::" and wrongly treated this as an exposed bind.
        assert!(is_loopback_listen("::1"));
        assert!(!is_loopback_listen("0.0.0.0:8080"));
        assert!(!is_loopback_listen("::8080"));
        assert!(!is_loopback_listen("192.168.1.5:8080"));
        assert!(!is_loopback_listen("10.0.0.1:8080"));
    }

    #[test]
    fn bind_password_guard_accepts_non_loopback_with_password() {
        let cfg = AdminConfig {
            listen: "0.0.0.0:8080".to_string(),
            login_password: Some("s3cret".to_string()),
            ..AdminConfig::default()
        };
        enforce_bind_password_guard(&cfg).expect("password + non-loopback must pass the guard");
    }

    /// F5: `serve --listen 0.0.0.0:8080` (as hard-coded in `Dockerfile.admin`)
    /// must NOT bypass the non-loopback password guard. The CLI override is
    /// applied *after* `config::load` has validated, so `serve` re-runs the
    /// guard on the final listen address before binding.
    #[test]
    fn serve_rejects_listen_override_bypassing_password_guard() {
        let _g = ENV_LOCK.lock().unwrap();
        clear_admin_env();
        // The config file supplies master_key and a loopback bind (passes the
        // guard inside `load` on its own); the `--listen` override flips the
        // final bind to non-loopback, which must be rejected.
        let path = std::env::temp_dir().join(format!(
            "oxllm_admin_serve_f5_{}_{}.yaml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let yaml = "\
listen: \"127.0.0.1:8080\"
proxy_url: \"http://127.0.0.1:4000\"
master_key: sk-file-master
";
        std::fs::write(&path, yaml).expect("write temp admin config");
        // Sync test + `block_on` (rather than `#[tokio::test]` + `.await`) so
        // the env-serializing `ENV_LOCK` guard is never held across an await
        // point (clippy::await_holding_lock). `serve` fails on the guard
        // before any I/O, so a bare current-thread runtime is sufficient.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build current-thread runtime");
        let result = rt.block_on(crate::serve(
            Some(path.to_string_lossy().into_owned()),
            Some("0.0.0.0:8080".to_string()),
            None,
        ));
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("serve must refuse a non-loopback --listen without a password");
        let msg = err.to_string();
        assert!(msg.contains("login_password"), "msg: {msg}");
    }
}
