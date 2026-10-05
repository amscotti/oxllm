//! Shared helpers for the `oxllm` CLI binaries.
//!
//! This lib target exists so the keygen/key utilities can be unit-tested
//! independently of the `oxllm` binary entrypoint (and so
//! `cargo test -p oxllm --lib --bins` has a lib target to run).

/// Generate a random virtual key: `sk-ox-<40 hex chars>` (20 random bytes).
pub fn generate_random_key() -> String {
    use rand::Rng;
    const KEY_PREFIX: &str = "sk-ox-";
    let mut rng = rand::rng();
    let bytes: [u8; 20] = rng.random();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("{KEY_PREFIX}{hex}")
}

/// 16 hex chars of randomness for `key_id` suffixes.
pub fn random_id() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bytes: [u8; 8] = rng.random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase hex SHA-256 digest of `input`.
pub fn sha256_hex(input: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input);
    let h = hasher.finalize();
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// Normalize `sqlite://path.db` → `sqlite://path.db` and bare
/// `path.db` / `/abs/path.db` into `sqlite://` form so users can pass either
/// form on the CLI. Non-sqlite URLs are returned unchanged, except MySQL
/// URLs, which are rejected with a clear error: MySQL support is not
/// established, and passing the URL through would only fail later with a
/// confusing connection error.
pub fn canonical_sqlite_url(url: String) -> anyhow::Result<String> {
    if url.starts_with("sqlite:") {
        return Ok(url);
    }
    if url.starts_with("mysql://") || url.starts_with("mariadb://") {
        anyhow::bail!(
            "unsupported database: MySQL/MariaDB is not supported; use sqlite:// or postgres://"
        );
    }
    if url.starts_with("postgres://") || url.starts_with("postgresql://") {
        return Ok(url);
    }
    Ok(format!("sqlite://{url}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_key_has_prefix_and_sufficient_entropy() {
        let key = generate_random_key();
        assert!(key.starts_with("sk-ox-"));
        assert!(key.len() >= 46, "prefix (6) + 40 hex chars = 46");
    }

    #[test]
    fn random_ids_are_unique() {
        let a = random_id();
        let b = random_id();
        assert_ne!(a, b);
    }

    #[test]
    fn sha256_hex_is_lowercase_hex() {
        let h = sha256_hex(b"hello");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            h,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn canonical_sqlite_url_passes_through_sqlite_scheme() {
        assert_eq!(
            canonical_sqlite_url("sqlite::memory:".to_string()).unwrap(),
            "sqlite::memory:"
        );
        assert_eq!(
            canonical_sqlite_url("sqlite://oxllm.db".to_string()).unwrap(),
            "sqlite://oxllm.db"
        );
    }

    #[test]
    fn canonical_sqlite_url_wraps_bare_path() {
        assert_eq!(
            canonical_sqlite_url("oxllm.db".to_string()).unwrap(),
            "sqlite://oxllm.db"
        );
        assert_eq!(
            canonical_sqlite_url("/var/lib/oxllm.db".to_string()).unwrap(),
            "sqlite:///var/lib/oxllm.db"
        );
    }

    #[test]
    fn canonical_sqlite_url_passes_through_postgres() {
        assert_eq!(
            canonical_sqlite_url("postgres://u:p@host/db".to_string()).unwrap(),
            "postgres://u:p@host/db"
        );
    }

    #[test]
    fn canonical_sqlite_url_rejects_mysql() {
        // MySQL support is not established; reject early with a clear error
        // instead of a confusing downstream connection failure.
        for url in ["mysql://u:p@host/db", "mariadb://u:p@host/db"] {
            let err = canonical_sqlite_url(url.to_string()).unwrap_err();
            assert!(
                err.to_string().contains("unsupported database"),
                "unexpected error for {url}: {err}"
            );
        }
    }
}
