//! AES-256-GCM credential encryption for at-rest provider API keys.
//!
//! # Opt-in
//!
//! Encryption is opt-in: when `ENCRYPTION_KEY` is unset, the model
//! repository stores credentials as plaintext (backward compatible).
//!
//! # Key derivation
//!
//! `ENCRYPTION_KEY` is interpreted strictly; only the following two
//! forms are accepted (anything else disables encryption with a loud
//! error log, so operators do not silently end up with weak keys):
//! 1. **32-byte hex** (64 hex chars) → the raw bytes become the key.
//! 2. **32-byte base64** (44 base64 chars including padding) → the
//!    decoded bytes become the key.
//!
//! Generate a key with `openssl rand -hex 32` or `openssl rand -base64 32`.
//!
//! A programmatic `CredentialCipher::from_passphrase` constructor
//! additionally accepts arbitrary input and derives a key from it via a
//! single unsalted SHA-256 pass. That path is **not** a KDF and is compiled
//! out of shipped builds (`#[cfg(test)]` only): a human passphrase is cheap
//! to brute-force, so only the strict env/raw-key paths are exposed publicly.
//!
//! # Key rotation
//!
//! Set `ENCRYPTION_KEY_PREVIOUS` (comma-separated) to one or more old
//! keys during a rotation. The cipher then holds the previous keys plus
//! the current key; new writes use the current key and reads transparently
//! decrypt blobs written under any retained key. Remove the previous keys
//! once all blobs have been re-written under the new key.
//!
//! # Wire format
//!
//! `encrypt` returns `MAGIC || base64(nonce || ciphertext || tag)` — the
//! canonical AES-256-GCM single-blob layout prefixed with the `oxenc1:`
//! magic so blobs are unambiguously identifiable. The nonce is 12 bytes,
//! generated fresh per call via `OsRng`. Legacy blobs written without the
//! magic prefix are still readable by [`CredentialCipher::decrypt`].

use aes_gcm::aead::{Aead, KeyInit, OsRng as AeadOsRng, rand_core::RngCore};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use base64::Engine;
#[cfg(test)]
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

const NONCE_LEN: usize = 12;
/// GCM authentication tag length.
const TAG_LEN: usize = 16;
/// Minimum encrypted blob: nonce + tag (empty plaintext).
const MIN_BLOB_LEN: usize = NONCE_LEN + TAG_LEN;
/// Minimum base64 length of an encrypted blob payload: `4 * ceil(28 / 3) = 40`.
const MIN_BLOB_B64_LEN: usize = 4 * MIN_BLOB_LEN.div_ceil(3);
/// Upper bound on a base64 blob we are willing to decode. Well above any sane
/// provider-key length; this is a defense-in-depth ceiling so a pathologically
/// large stored value is rejected cheaply instead of being decoded into a
/// `Vec<u8>` purely to test validity. Real credentials are a few KB at most.
const MAX_BLOB_B64_LEN: usize = 8 * 1024;
/// Maximum plaintext [`CredentialCipher::encrypt`] accepts. Derived from
/// `MAX_BLOB_B64_LEN` so every blob `encrypt` writes is guaranteed to sit
/// within the decode ceiling [`CredentialCipher::decrypt`] enforces: a
/// larger plaintext would be stored as a blob this module could never read
/// back. `4 * ceil(n / 3)` inverts standard base64, and the nonce + tag
/// bytes are part of the blob.
pub(crate) const MAX_PLAINTEXT_LEN: usize = MAX_BLOB_B64_LEN.div_ceil(4) * 3 - NONCE_LEN - TAG_LEN;
/// Magic prefix tagging every encrypted value, so plaintext that merely
/// *looks* like base64 is never mistaken for a blob.
const MAGIC: &str = "oxenc1:";

#[derive(Debug)]
pub enum CryptoError {
    /// The stored value is not valid standard base64.
    InvalidBase64(String),
    /// The decoded blob is shorter than nonce + tag.
    CiphertextTooShort(usize),
    /// The stored blob exceeds the maximum base64 length we are willing to
    /// decode.
    CiphertextTooLarge(usize),
    /// AEAD verification failed (wrong key or corrupted ciphertext).
    AuthenticationFailed,
    /// AEAD decryption succeeded but the plaintext is not valid UTF-8.
    DecodingFailed,
    /// AEAD encryption failed (only reachable with plaintext > 64 GiB).
    EncryptionFailed,
    /// The plaintext exceeds the largest value whose blob `decrypt` could
    /// ever read back, so encrypting it would store a permanently
    /// undecryptable blob.
    PlaintextTooLarge(usize),
    /// The key ring is empty, so no key is available to encrypt or decrypt.
    EmptyKeyRing,
}

impl std::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CryptoError::InvalidBase64(e) => write!(f, "invalid base64 in ciphertext: {e}"),
            CryptoError::CiphertextTooShort(len) => {
                write!(
                    f,
                    "ciphertext too short ({len} bytes, minimum {MIN_BLOB_LEN})"
                )
            }
            CryptoError::CiphertextTooLarge(len) => {
                write!(
                    f,
                    "ciphertext too large ({len} chars, maximum {MAX_BLOB_B64_LEN})"
                )
            }
            CryptoError::AuthenticationFailed => {
                write!(
                    f,
                    "AES-GCM authentication failed (wrong key or corrupted ciphertext)"
                )
            }
            CryptoError::DecodingFailed => write!(f, "decrypted plaintext is not valid UTF-8"),
            CryptoError::EncryptionFailed => write!(f, "AES-GCM encryption failed"),
            CryptoError::PlaintextTooLarge(len) => write!(
                f,
                "plaintext too large ({len} bytes, maximum {MAX_PLAINTEXT_LEN}); \
                 encrypting it would store a blob that can never be decrypted"
            ),
            CryptoError::EmptyKeyRing => write!(f, "cipher key ring is empty"),
        }
    }
}

impl std::error::Error for CryptoError {}

#[derive(Clone)]
pub struct CredentialCipher {
    /// Ordered key ring. The last entry is the current (encrypt) key; all
    /// entries are tried on decrypt so blobs written under a previous key
    /// remain readable during rotation. Wrapped in [`Zeroizing`] so the raw
    /// key bytes are wiped when the cipher is dropped.
    keys: Zeroizing<Vec<[u8; 32]>>,
}

impl std::fmt::Debug for CredentialCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CredentialCipher").finish_non_exhaustive()
    }
}

impl CredentialCipher {
    #[must_use]
    pub fn from_raw_key(key: [u8; 32]) -> Self {
        Self {
            keys: Zeroizing::new(vec![key]),
        }
    }

    #[must_use]
    pub fn from_env() -> Option<Self> {
        Self::from_env_values(
            std::env::var("ENCRYPTION_KEY").ok().as_deref(),
            std::env::var("ENCRYPTION_KEY_PREVIOUS").ok().as_deref(),
        )
    }

    /// Pure core of [`Self::from_env`], split out so tests can exercise every
    /// branch without mutating process env (`std::env::set_var`/`remove_var`
    /// are `unsafe` in edition 2024 and this crate is `#![deny(unsafe_code)]`).
    ///
    /// `current` is the active `ENCRYPTION_KEY`; `previous` is the optional
    /// comma-separated `ENCRYPTION_KEY_PREVIOUS` retained for rotation.
    /// Both are parsed strictly (64-hex or 44-base64 only); a malformed
    /// current key disables encryption (`None`), and malformed previous keys
    /// are skipped with a warning.
    #[must_use]
    fn from_env_values(current: Option<&str>, previous: Option<&str>) -> Option<Self> {
        let raw = current?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            tracing::warn!(
                "ENCRYPTION_KEY is set but empty — credentials will be stored in plaintext. \
                 Set a non-empty value to enable AES-256-GCM encryption at rest."
            );
            return None;
        }
        let current_key = match parse_strict(trimmed) {
            Some(k) => k,
            None => {
                tracing::error!(
                    "ENCRYPTION_KEY is not a valid 64-char hex or 44-char base64 key; \
                     credential encryption is DISABLED (plaintext storage). \
                     Generate a key with `openssl rand -hex 32` or `openssl rand -base64 32`."
                );
                return None;
            }
        };
        let mut keys: Vec<[u8; 32]> = Vec::new();
        if let Some(prev) = previous {
            for candidate in prev.split(',') {
                let candidate = candidate.trim();
                if candidate.is_empty() {
                    continue;
                }
                match parse_strict(candidate) {
                    Some(k) => {
                        if k == current_key || keys.contains(&k) {
                            tracing::warn!(
                                "ENCRYPTION_KEY_PREVIOUS contains a duplicate key \
                                 (equal to ENCRYPTION_KEY or listed twice); keeping a \
                                 single ring entry."
                            );
                        } else {
                            keys.push(k);
                        }
                    }
                    None => tracing::warn!(
                        "ENCRYPTION_KEY_PREVIOUS contains an entry that is not a valid \
                         64-char hex or 44-char base64 key; skipping it."
                    ),
                }
            }
        }
        keys.push(current_key);
        Some(Self {
            keys: Zeroizing::new(keys),
        })
    }

    /// Derive a cipher from an arbitrary passphrase. Kept infallible for
    /// programmatic/test callers: hex and base64 keys are used verbatim, any
    /// other input is reduced via a single unsalted SHA-256 pass — which is
    /// **not** a KDF and must not be used for real secrets (prefer
    /// [`Self::from_env`] / [`Self::from_raw_key`]). An empty passphrase is
    /// accepted but logs a warning, since it derives a well-known constant key.
    ///
    /// Test-only: compiled out of shipped builds so the weak SHA-256 fallback
    /// is never reachable from production code.
    #[must_use]
    #[cfg(test)]
    pub fn from_passphrase(passphrase: &str) -> Self {
        if passphrase.is_empty() {
            tracing::warn!(
                "CredentialCipher::from_passphrase called with an empty string; \
                 deriving the well-known SHA-256(\"\") key. Pass a real key."
            );
        }
        let key = derive_key(passphrase);
        Self::from_raw_key(key)
    }

    pub fn encrypt(&self, plaintext: &str) -> Result<String, CryptoError> {
        let key = self.keys.last().ok_or(CryptoError::EmptyKeyRing)?;
        if plaintext.len() > MAX_PLAINTEXT_LEN {
            return Err(CryptoError::PlaintextTooLarge(plaintext.len()));
        }
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let mut nonce_bytes = [0u8; NONCE_LEN];
        AeadOsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = Zeroizing::new(
            cipher
                .encrypt(nonce, plaintext.as_bytes())
                .map_err(|_| CryptoError::EncryptionFailed)?,
        );
        let mut blob = Zeroizing::new(Vec::with_capacity(NONCE_LEN + ciphertext.len()));
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);
        Ok(format!(
            "{MAGIC}{}",
            base64::engine::general_purpose::STANDARD.encode(&blob[..])
        ))
    }

    /// Decrypt a value produced by [`Self::encrypt`]. Both the current
    /// magic-prefixed format and legacy (un-prefixed) blobs are accepted;
    /// every key in the ring is tried until one authenticates. The ring is
    /// walked current-key-first so the common case — a blob written under
    /// the current key — succeeds after a single AEAD pass. The raw
    /// decrypted buffer is zeroized before UTF-8 conversion; the returned
    /// `String` is a separate heap allocation that cannot be scrubbed in
    /// place, so callers holding a decrypted secret should drop it promptly.
    pub fn decrypt(&self, ciphertext_b64: &str) -> Result<String, CryptoError> {
        if self.keys.is_empty() {
            return Err(CryptoError::EmptyKeyRing);
        }
        let trimmed = ciphertext_b64.trim();
        let payload = trimmed.strip_prefix(MAGIC).unwrap_or(trimmed);
        if payload.len() > MAX_BLOB_B64_LEN {
            return Err(CryptoError::CiphertextTooLarge(payload.len()));
        }
        let blob = Zeroizing::new(
            base64::engine::general_purpose::STANDARD
                .decode(payload)
                .map_err(|e| CryptoError::InvalidBase64(e.to_string()))?,
        );
        if blob.len() < MIN_BLOB_LEN {
            return Err(CryptoError::CiphertextTooShort(blob.len()));
        }
        let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
        let nonce = Nonce::from_slice(nonce_bytes);
        for key in self.keys.iter().rev() {
            let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
            if let Ok(plaintext) = cipher.decrypt(nonce, ciphertext) {
                // `str::from_utf8` borrows, so the `Zeroizing` wrapper stays
                // alive and wipes the decrypted bytes even on the error path.
                let plaintext = Zeroizing::new(plaintext);
                return match std::str::from_utf8(&plaintext) {
                    Ok(s) => Ok(s.to_owned()),
                    Err(_) => Err(CryptoError::DecodingFailed),
                };
            }
        }
        Err(CryptoError::AuthenticationFailed)
    }

    /// Cheap, unambiguous check: could `value` be a blob produced by
    /// [`Self::encrypt`]? Only values carrying the `oxenc1:` magic prefix
    /// qualify, so plaintext that merely happens to be valid base64 is never
    /// mis-classified as "already encrypted". Use `Self::is_blob_like` when
    /// you also need to recognise legacy (un-prefixed) blobs for decryption.
    #[must_use]
    pub fn looks_encrypted(value: &str) -> bool {
        value.starts_with(MAGIC)
    }

    /// Broader guard used on the decrypt path: `true` for both current
    /// magic-prefixed blobs and legacy (un-prefixed) base64 blobs whose shape
    /// matches the old format. Surrounding whitespace is trimmed so the gate
    /// agrees with [`Self::decrypt`], which tolerates padded input.
    #[must_use]
    pub(crate) fn is_blob_like(value: &str) -> bool {
        let value = value.trim();
        if Self::looks_encrypted(value) {
            return true;
        }
        if value.len() < MIN_BLOB_B64_LEN {
            return false;
        }
        if value.len() > MAX_BLOB_B64_LEN {
            return false;
        }
        base64::engine::general_purpose::STANDARD
            .decode(value)
            .is_ok()
    }
}

/// Strict key parser used by the env path: accepts only a 64-char hex key or
/// a 44-char base64 key. Returns `None` for anything else (no fallback).
fn parse_strict(raw: &str) -> Option<[u8; 32]> {
    if let Ok(key) = hex_decode_32(raw) {
        return Some(key);
    }
    if raw.len() == 44 {
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(raw) {
            if bytes.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&bytes);
                return Some(key);
            }
        }
    }
    None
}

#[cfg(test)]
fn derive_key(passphrase: &str) -> [u8; 32] {
    if let Some(key) = parse_strict(passphrase) {
        return key;
    }
    tracing::warn!(
        "Deriving a cipher key from a non-key input via a single unsalted SHA-256 pass; \
         this is NOT a key derivation function and is weak against brute-force. \
         Use a 64-char hex key or 44-char base64 key instead."
    );
    let mut hasher = Sha256::new();
    hasher.update(passphrase.as_bytes());
    let digest = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    key
}

fn hex_decode_32(hex: &str) -> Result<[u8; 32], &'static str> {
    let mut out = [0u8; 32];
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return Err("hex string must be exactly 64 chars");
    }
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, &'static str> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err("invalid hex char"),
    }
}

/// Encrypt a credential for storage. With `None` cipher the plaintext is
/// returned unchanged (plaintext storage, backward compatible).
pub fn encrypt_api_key(
    cipher: Option<&CredentialCipher>,
    api_key: &str,
) -> Result<String, CryptoError> {
    match cipher {
        Some(c) => c.encrypt(api_key),
        None => Ok(api_key.to_string()),
    }
}

/// Decrypt a stored credential.
///
/// * `None` cipher → the value is returned unchanged (plaintext storage).
/// * A value that does not look like an encrypted blob → returned unchanged
///   (backward compatibility with rows written before encryption was enabled).
/// * A blob that fails to decrypt → `Err`. Callers decide what to do: fail
///   the operation, or log and treat the stored value as plaintext.
///
/// # Legacy plaintext that looks like a blob
///
/// When a cipher is configured, any non-magic value that is ≥ 40 chars of valid
/// standard base64 is treated as a legacy (un-prefixed) blob and AEAD
/// decryption is attempted. A legacy *plaintext* provider key that happens to
/// be ≥ 40 chars of valid standard base64 (no `-`/`_`, correct padding) will
/// therefore fail authentication and surface as `AuthenticationFailed`; the
/// raw value is then sent upstream and the provider returns 401. The `oxenc1:`
/// magic prefix exists precisely to avoid this for *new* writes, so the
/// ambiguous window only affects rows written before encryption was enabled.
/// Most real keys (`sk-…`, containing `-`) are not standard base64 and pass
/// through cleanly. Operators seeing unexplained 401s after enabling
/// encryption should re-write the affected rows under the magic prefix (e.g.
/// via a one-shot re-encrypt migration) to close the ambiguous legacy window.
pub fn decrypt_api_key(
    cipher: Option<&CredentialCipher>,
    stored_value: &str,
) -> Result<String, CryptoError> {
    let Some(c) = cipher else {
        return Ok(stored_value.to_string());
    };
    if !CredentialCipher::is_blob_like(stored_value) {
        return Ok(stored_value.to_string());
    }
    c.decrypt(stored_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twin_encrypt_decrypt_round_trip() {
        let cipher = CredentialCipher::from_passphrase("test-passphrase-oxllm");
        let plaintext = "sk-provider-key-1234567890";
        let encrypted = cipher.encrypt(plaintext).expect("encrypt must succeed");
        assert_ne!(encrypted, plaintext);
        let decrypted = cipher
            .decrypt(&encrypted)
            .expect("round-trip decrypt must succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn twin_empty_plaintext_round_trip() {
        let cipher = CredentialCipher::from_passphrase("empty-plaintext");
        let encrypted = cipher.encrypt("").expect("encrypt must succeed");
        assert_ne!(encrypted, "");
        let decrypted = cipher.decrypt(&encrypted).expect("decrypt must succeed");
        assert_eq!(decrypted, "");
    }

    #[test]
    fn twin_decrypt_wrong_key_fails() {
        let cipher_a = CredentialCipher::from_passphrase("key-a");
        let cipher_b = CredentialCipher::from_passphrase("key-b");
        let encrypted = cipher_a.encrypt("sk-secret").expect("encrypt must succeed");
        let result = cipher_b.decrypt(&encrypted);
        assert!(matches!(result, Err(CryptoError::AuthenticationFailed)));
    }

    #[test]
    fn twin_tampered_ciphertext_fails_authentication() {
        let cipher = CredentialCipher::from_passphrase("tamper-test");
        let encrypted = cipher
            .encrypt("sk-untampered")
            .expect("encrypt must succeed");
        let mut blob = base64::engine::general_purpose::STANDARD
            .decode(encrypted.strip_prefix(MAGIC).unwrap_or(&encrypted))
            .expect("our own output is valid base64");
        // Flip one bit inside the ciphertext body (past the 12-byte nonce).
        blob[NONCE_LEN] ^= 0x01;
        let tampered = format!(
            "{MAGIC}{}",
            base64::engine::general_purpose::STANDARD.encode(&blob)
        );
        let result = cipher.decrypt(&tampered);
        assert!(matches!(result, Err(CryptoError::AuthenticationFailed)));
    }

    #[test]
    fn twin_encrypt_produces_different_ciphertext() {
        let cipher = CredentialCipher::from_passphrase("nonce-test");
        let plaintext = "sk-same-input";
        let a = cipher.encrypt(plaintext).expect("encrypt must succeed");
        let b = cipher.encrypt(plaintext).expect("encrypt must succeed");
        assert_ne!(a, b);
        assert_eq!(cipher.decrypt(&a).expect("decrypt a"), plaintext);
        assert_eq!(cipher.decrypt(&b).expect("decrypt b"), plaintext);
    }

    #[test]
    fn twin_from_env_value_variants() {
        // Unset → no cipher.
        assert!(CredentialCipher::from_env_values(None, None).is_none());
        // Set-but-empty → no cipher (warning logged).
        assert!(CredentialCipher::from_env_values(Some(""), None).is_none());
        // A human passphrase is no longer accepted by the env path.
        assert!(CredentialCipher::from_env_values(Some("a-passphrase"), None).is_none());
        // A valid hex key is accepted.
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert!(CredentialCipher::from_env_values(Some(hex), None).is_some());
        // A valid base64 key is accepted.
        let b64 = base64::engine::general_purpose::STANDARD.encode([0x11u8; 32]);
        assert_eq!(b64.len(), 44);
        assert!(CredentialCipher::from_env_values(Some(&b64), None).is_some());
    }

    #[test]
    fn derive_key_rejects_human_passphrase_form() {
        // Human passphrase → strict env path refuses (None).
        assert!(CredentialCipher::from_env_values(Some("a-human-passphrase"), None).is_none());
        // 64-char string with one non-hex char → not a valid key → None.
        let bad_hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg";
        assert_eq!(bad_hex.len(), 64);
        assert!(CredentialCipher::from_env_values(Some(bad_hex), None).is_none());
        // 44-char string that is not valid base64 → None.
        let bad_b64 = "!".repeat(44);
        assert!(CredentialCipher::from_env_values(Some(&bad_b64), None).is_none());
        // Valid hex with a distinct previous (rotation) → Some, two keys.
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let prev = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
        let c = CredentialCipher::from_env_values(Some(hex), Some(prev)).expect("both valid");
        assert_eq!(c.keys.len(), 2);
    }

    #[test]
    fn derive_key_falls_back_to_sha256_for_invalid_shape() {
        // from_passphrase (programmatic path) still falls through to SHA-256
        // when the shape matches but the content is invalid.
        let bad = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg";
        let cipher = CredentialCipher::from_passphrase(bad);
        let mut hasher = Sha256::new();
        hasher.update(bad.as_bytes());
        let mut key = [0u8; 32];
        key.copy_from_slice(&hasher.finalize());
        let cipher_manual = CredentialCipher::from_raw_key(key);
        let ct = cipher.encrypt("round-trip").expect("encrypt must succeed");
        assert_eq!(cipher_manual.decrypt(&ct).expect("decrypt"), "round-trip");
    }

    #[test]
    fn twin_no_cipher_helpers_pass_plaintext_through() {
        assert_eq!(
            encrypt_api_key(None, "sk-plaintext").expect("no cipher never fails"),
            "sk-plaintext"
        );
        assert_eq!(
            decrypt_api_key(None, "sk-plaintext").expect("no cipher never fails"),
            "sk-plaintext"
        );
    }

    #[test]
    fn twin_hex_passphrase_used_as_raw_bytes() {
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let cipher_hex = CredentialCipher::from_passphrase(hex);
        let mut raw = [0u8; 32];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let hi = hex_nibble(chunk[0]).unwrap();
            let lo = hex_nibble(chunk[1]).unwrap();
            raw[i] = (hi << 4) | lo;
        }
        let cipher_raw = CredentialCipher::from_raw_key(raw);
        let ct = cipher_hex.encrypt("sk-test").expect("encrypt must succeed");
        assert_eq!(cipher_raw.decrypt(&ct).expect("decrypt"), "sk-test");
    }

    #[test]
    fn twin_base64_passphrase_used_as_raw_bytes() {
        let raw: [u8; 32] = std::array::from_fn(|i| i as u8);
        let b64 = base64::engine::general_purpose::STANDARD.encode(raw);
        assert_eq!(b64.len(), 44);
        let cipher_b64 = CredentialCipher::from_passphrase(&b64);
        let cipher_raw = CredentialCipher::from_raw_key(raw);
        let ct = cipher_b64.encrypt("sk-b64").expect("encrypt must succeed");
        assert_eq!(cipher_raw.decrypt(&ct).expect("decrypt"), "sk-b64");
    }

    #[test]
    fn twin_passphrase_sha256_fallback() {
        let cipher = CredentialCipher::from_passphrase("my-simple-passphrase");
        let mut hasher = Sha256::new();
        hasher.update(b"my-simple-passphrase");
        let mut key = [0u8; 32];
        key.copy_from_slice(&hasher.finalize());
        let cipher_manual = CredentialCipher::from_raw_key(key);
        let ct = cipher.encrypt("round-trip").expect("encrypt must succeed");
        assert_eq!(cipher_manual.decrypt(&ct).expect("decrypt"), "round-trip");
    }

    #[test]
    fn twin_decrypt_invalid_base64() {
        let cipher = CredentialCipher::from_passphrase("err-test");
        let result = cipher.decrypt("!!!not-base64!!!");
        assert!(matches!(result, Err(CryptoError::InvalidBase64(_))));
    }

    #[test]
    fn twin_decrypt_short_ciphertext() {
        let cipher = CredentialCipher::from_passphrase("short-test");
        let short = base64::engine::general_purpose::STANDARD.encode(b"0123456789");
        let result = cipher.decrypt(&short);
        assert!(matches!(result, Err(CryptoError::CiphertextTooShort(_))));
    }

    #[test]
    fn twin_looks_encrypted_threshold() {
        // The magic prefix — not length/base64 — is now what marks a blob.
        assert!(!CredentialCipher::looks_encrypted(&"A".repeat(39)));
        // Bare base64 of any length is NOT recognised as a blob, even if long.
        assert!(!CredentialCipher::looks_encrypted(&"A".repeat(40)));
        assert!(!CredentialCipher::looks_encrypted(&"A".repeat(64)));
        assert!(!CredentialCipher::looks_encrypted(&"!".repeat(44)));
        assert!(!CredentialCipher::looks_encrypted("sk-short"));
        // Only the magic prefix qualifies.
        assert!(CredentialCipher::looks_encrypted("oxenc1:AAAA"));
        assert!(!CredentialCipher::looks_encrypted(" oxenc1:AAAA"));
    }

    #[test]
    fn twin_looks_encrypted_accepts_real_blobs() {
        let cipher = CredentialCipher::from_passphrase("looks-test");
        // Even the empty plaintext yields a 28-byte blob → 40 base64 chars,
        // prefixed with the 7-byte magic.
        let empty = cipher.encrypt("").expect("encrypt must succeed");
        assert_eq!(empty.len(), MAGIC.len() + MIN_BLOB_B64_LEN);
        assert!(CredentialCipher::looks_encrypted(&empty));
        let encrypted = cipher
            .encrypt("sk-a-real-key")
            .expect("encrypt must succeed");
        assert!(encrypted.len() > MAGIC.len() + MIN_BLOB_B64_LEN);
        assert!(CredentialCipher::looks_encrypted(&encrypted));
    }

    #[test]
    fn looks_encrypted_false_positive_on_base64_plaintext() {
        // A >= 40-char standard-base64 plaintext that is NOT a real blob.
        let fake = base64::engine::general_purpose::STANDARD.encode(b"x".repeat(48));
        assert!(fake.len() >= MIN_BLOB_B64_LEN);
        // Prefix-based heuristic → not treated as an encrypted blob.
        assert!(!CredentialCipher::looks_encrypted(&fake));
        // And it encrypts (instead of being skipped as "already encrypted").
        let cipher = CredentialCipher::from_passphrase("false-pos");
        let stored = cipher.encrypt(&fake).expect("encrypt must succeed");
        assert_ne!(stored, fake);
        assert!(stored.starts_with(MAGIC));
        assert_eq!(cipher.decrypt(&stored).expect("decrypt"), fake);
    }

    #[test]
    fn looks_encrypted_and_decrypt_agree_on_legacy_blob_shape() {
        let cipher = CredentialCipher::from_passphrase("agree-test");
        // Legacy-shaped base64 (no prefix) that is not a real blob: it is
        // `is_blob_like` (decrypt is attempted) but fails authentication, so
        // `decrypt_api_key` returns Err rather than silently passing it
        // through as plaintext.
        let fake = base64::engine::general_purpose::STANDARD.encode(b"z".repeat(48));
        assert!(CredentialCipher::is_blob_like(&fake));
        assert!(!CredentialCipher::looks_encrypted(&fake));
        assert!(matches!(
            decrypt_api_key(Some(&cipher), &fake),
            Err(CryptoError::AuthenticationFailed)
        ));
    }

    #[test]
    fn decrypt_handles_whitespace_padding() {
        let cipher = CredentialCipher::from_passphrase("whitespace-test");
        let blob = cipher.encrypt("sk-trimmed").expect("encrypt must succeed");
        // A trailing newline / surrounding spaces must not break decrypt;
        // `is_blob_like` trims too, so decrypt_api_key recognises padded
        // blobs rather than passing the raw ciphertext through.
        let padded = format!("  {}\n", blob);
        assert_eq!(
            cipher.decrypt(&padded).expect("trimmed decrypt"),
            "sk-trimmed"
        );
    }

    #[test]
    fn decrypt_with_previous_key_after_rotation() {
        let key_a = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let key_b = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
        // Encrypt under key A (the soon-to-be-previous key).
        let cipher_a = CredentialCipher::from_env_values(Some(key_a), None).expect("key a");
        let blob = cipher_a
            .encrypt("sk-rotated")
            .expect("encrypt must succeed");
        // Rotate: current = B, previous = A. Old blob must still decrypt.
        let cipher_rot =
            CredentialCipher::from_env_values(Some(key_b), Some(key_a)).expect("rotated");
        assert_eq!(
            cipher_rot.decrypt(&blob).expect("old blob decrypts"),
            "sk-rotated"
        );
    }

    #[test]
    fn encrypt_uses_latest_key_only() {
        let key_a = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let key_b = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
        let cipher_rot =
            CredentialCipher::from_env_values(Some(key_b), Some(key_a)).expect("rotated");
        let blob = cipher_rot.encrypt("sk-new").expect("encrypt must succeed");
        // New blob decrypts with the B-only cipher.
        let cipher_b = CredentialCipher::from_env_values(Some(key_b), None).expect("key b");
        assert_eq!(
            cipher_b.decrypt(&blob).expect("decrypt with new key"),
            "sk-new"
        );
        // New blob does NOT decrypt with the A-only cipher.
        let cipher_a = CredentialCipher::from_env_values(Some(key_a), None).expect("key a");
        assert!(matches!(
            cipher_a.decrypt(&blob),
            Err(CryptoError::AuthenticationFailed)
        ));
    }

    #[test]
    fn from_passphrase_empty_warns_without_panicking() {
        // from_passphrase stays infallible; empty logs a warning but returns
        // a (weak, well-known) cipher rather than panicking.
        let cipher = CredentialCipher::from_passphrase("");
        let blob = cipher.encrypt("x").expect("encrypt must succeed");
        assert_eq!(cipher.decrypt(&blob).expect("decrypt"), "x");
    }

    #[test]
    fn twin_decrypt_api_key_passthrough_for_plaintext() {
        let cipher = CredentialCipher::from_passphrase("mixed-db");
        assert_eq!(
            decrypt_api_key(Some(&cipher), "sk-short").expect("plaintext passthrough"),
            "sk-short"
        );
    }

    #[test]
    fn twin_decrypt_api_key_returns_err_on_tampered_blob() {
        let cipher = CredentialCipher::from_passphrase("err-propagation");
        let encrypted = cipher.encrypt("sk-key").expect("encrypt must succeed");
        let mut blob = base64::engine::general_purpose::STANDARD
            .decode(encrypted.strip_prefix(MAGIC).unwrap_or(&encrypted))
            .expect("our own output is valid base64");
        // Flip one bit inside the GCM tag (last 16 bytes).
        let last = blob.len() - 1;
        blob[last] ^= 0x01;
        let tampered = format!(
            "{MAGIC}{}",
            base64::engine::general_purpose::STANDARD.encode(&blob)
        );
        let result = decrypt_api_key(Some(&cipher), &tampered);
        assert!(matches!(result, Err(CryptoError::AuthenticationFailed)));
    }

    #[test]
    fn twin_no_cipher_round_trips_plaintext() {
        let key = "sk-no-cipher-12345";
        let stored = encrypt_api_key(None, key).expect("no cipher never fails");
        assert_eq!(stored, key);
        assert_eq!(
            decrypt_api_key(None, &stored).expect("no cipher never fails"),
            key
        );
    }

    #[test]
    fn twin_cipher_round_trips_via_helpers() {
        let cipher = CredentialCipher::from_passphrase("helpers-test");
        let key = "sk-helper-roundtrip";
        let stored = encrypt_api_key(Some(&cipher), key).expect("encrypt must succeed");
        assert_ne!(stored, key);
        assert_eq!(
            decrypt_api_key(Some(&cipher), &stored).expect("decrypt must succeed"),
            key
        );
    }

    #[test]
    fn twin_debug_does_not_leak_key() {
        let cipher = CredentialCipher::from_passphrase("secret-passphrase-debug");
        let s = format!("{cipher:?}");
        assert!(!s.contains("secret-passphrase-debug"));
    }

    #[test]
    fn debug_does_not_leak_key_bytes() {
        let raw = [0xabu8; 32];
        let cipher = CredentialCipher::from_raw_key(raw);
        let s = format!("{cipher:?}");
        assert!(!s.contains("ab"));
        assert!(!s.contains("171"));
    }

    #[test]
    fn decrypt_api_key_handles_whitespace_padded_blob() {
        let cipher = CredentialCipher::from_passphrase("pad-api-key");
        let blob = cipher.encrypt("sk-padded").expect("encrypt must succeed");
        let padded = format!("  {blob}\n");
        // The gate (is_blob_like) must trim so it agrees with decrypt; the
        // padded blob must round-trip through the full helper, not be passed
        // through as raw ciphertext.
        assert_eq!(
            decrypt_api_key(Some(&cipher), &padded).expect("round-trip"),
            "sk-padded"
        );
    }

    #[test]
    fn decrypt_api_key_whitespace_padded_tampered_errors() {
        let cipher = CredentialCipher::from_passphrase("pad-tamper");
        let blob = cipher.encrypt("sk-tamper").expect("encrypt must succeed");
        let mut raw = base64::engine::general_purpose::STANDARD
            .decode(blob.strip_prefix(MAGIC).unwrap())
            .expect("our own output is valid base64");
        // Flip one bit inside the ciphertext body (past the 12-byte nonce).
        raw[NONCE_LEN] ^= 0x01;
        let padded = format!(
            "  {MAGIC}{}\n",
            base64::engine::general_purpose::STANDARD.encode(&raw)
        );
        // Whitespace padding must not cause the tampered blob to be silently
        // passed through as plaintext.
        assert!(matches!(
            decrypt_api_key(Some(&cipher), &padded),
            Err(CryptoError::AuthenticationFailed)
        ));
    }

    #[test]
    fn decrypt_magic_only_empty_payload_is_short() {
        let cipher = CredentialCipher::from_passphrase("magic-only");
        assert!(matches!(
            cipher.decrypt("oxenc1:"),
            Err(CryptoError::CiphertextTooShort(0))
        ));
    }

    #[test]
    fn from_env_values_skips_malformed_previous_entries() {
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let prev = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
        // Mixed previous list: a malformed entry and an empty slot are skipped
        // (warn!), the single valid previous key plus the current key remain.
        let mixed = format!("garbage,,{prev},another_bad");
        let c = CredentialCipher::from_env_values(Some(hex), Some(&mixed)).expect("current valid");
        assert_eq!(c.keys.len(), 2);
        // The retained previous key still decrypts a blob written under it.
        let prev_only = CredentialCipher::from_env_values(Some(prev), None).expect("prev");
        let blob = prev_only.encrypt("sk-mix").expect("encrypt must succeed");
        assert_eq!(c.decrypt(&blob).expect("decrypt"), "sk-mix");
    }

    #[test]
    fn from_env_values_dedupes_ring_keys() {
        let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let other = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
        // Current key repeated in PREVIOUS → a single ring entry (warn).
        let c = CredentialCipher::from_env_values(Some(hex), Some(hex)).expect("valid");
        assert_eq!(c.keys.len(), 1);
        // A previous key listed twice → one entry, plus the current key.
        let c = CredentialCipher::from_env_values(Some(hex), Some(&format!("{other},{other}")))
            .expect("valid");
        assert_eq!(c.keys.len(), 2);
        // Distinct previous + current → two entries.
        let c = CredentialCipher::from_env_values(Some(hex), Some(other)).expect("valid");
        assert_eq!(c.keys.len(), 2);
    }

    #[test]
    fn parse_strict_accepts_uppercase_hex() {
        let lower = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let upper = lower.to_uppercase();
        let c_lower = CredentialCipher::from_env_values(Some(lower), None).expect("lower");
        let c_upper = CredentialCipher::from_env_values(Some(&upper), None).expect("upper");
        // Both parse to the same raw key, so a blob written by one decrypts
        // with the other.
        let blob = c_lower.encrypt("sk-case").expect("encrypt must succeed");
        assert_eq!(c_upper.decrypt(&blob).expect("decrypt"), "sk-case");
    }

    #[test]
    fn encrypt_empty_key_ring_errors_instead_of_panicking() {
        let cipher = CredentialCipher {
            keys: Zeroizing::new(Vec::new()),
        };
        assert!(matches!(
            cipher.encrypt("x"),
            Err(CryptoError::EmptyKeyRing)
        ));
    }

    #[test]
    fn decrypt_empty_key_ring_errors_with_empty_key_ring() {
        // Mirrors `encrypt_empty_key_ring_errors_instead_of_panicking`: with an
        // empty ring `decrypt` must report `EmptyKeyRing` (a misconfiguration
        // signal) rather than falling through to `AuthenticationFailed`, so
        // the two paths agree on the same invariant violation.
        let cipher = CredentialCipher {
            keys: Zeroizing::new(Vec::new()),
        };
        assert!(matches!(
            cipher.decrypt("oxenc1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="),
            Err(CryptoError::EmptyKeyRing)
        ));
    }

    #[test]
    fn decrypt_rejects_oversize_blob_without_allocating() {
        // A 1 MiB standard-base64 string is well above `MAX_BLOB_B64_LEN`; it
        // must short-circuit to an error before the base64 decoder ever runs,
        // so no multi-megabyte `Vec<u8>` is allocated just to test validity.
        let cipher = CredentialCipher::from_passphrase("oversize-test");
        let oversize = "A".repeat(1024 * 1024);
        assert!(oversize.len() > MAX_BLOB_B64_LEN);
        assert!(matches!(
            cipher.decrypt(&oversize),
            Err(CryptoError::CiphertextTooLarge(n)) if n == oversize.len()
        ));
    }

    #[test]
    fn encrypt_enforces_symmetric_plaintext_cap() {
        // The cap is exactly the largest plaintext whose blob still fits the
        // 8 KiB decode ceiling `decrypt` enforces, so a value at the cap must
        // round-trip and a value one byte over must be rejected up front
        // instead of being stored as a permanently undecryptable blob.
        let cipher = CredentialCipher::from_passphrase("cap-test");
        let at_cap = "x".repeat(MAX_PLAINTEXT_LEN);
        let blob = cipher.encrypt(&at_cap).expect("at-cap plaintext encrypts");
        let payload = blob.strip_prefix(MAGIC).expect("magic prefix");
        assert!(payload.len() <= MAX_BLOB_B64_LEN);
        assert_eq!(
            cipher.decrypt(&blob).expect("round-trip").len(),
            MAX_PLAINTEXT_LEN
        );
        let over = "x".repeat(MAX_PLAINTEXT_LEN + 1);
        assert!(matches!(
            cipher.encrypt(&over),
            Err(CryptoError::PlaintextTooLarge(n)) if n == over.len()
        ));
    }

    #[test]
    fn decrypt_reports_oversize_magic_blob_as_too_large() {
        // Hand-constructed via the raw AEAD (the public `encrypt` now rejects
        // oversize plaintext): a magic-prefixed blob whose base64 payload
        // exceeds MAX_BLOB_B64_LEN. `is_blob_like` still recognises it (the
        // magic short-circuits before the length gate), so `decrypt` — not
        // plaintext passthrough — must reject it, with the honest
        // `CiphertextTooLarge` label.
        let cipher = CredentialCipher::from_passphrase("oversize-magic");
        let raw_key = cipher.keys[0];
        let aes = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&raw_key));
        let mut nonce_bytes = [0u8; NONCE_LEN];
        AeadOsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let payload = [0x41u8; MAX_PLAINTEXT_LEN + 64];
        let ciphertext = aes
            .encrypt(nonce, payload.as_ref())
            .expect("aes encrypt must succeed");
        let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);
        let stored = format!(
            "{MAGIC}{}",
            base64::engine::general_purpose::STANDARD.encode(&blob)
        );
        assert!(CredentialCipher::is_blob_like(&stored));
        assert!(matches!(
            cipher.decrypt(&stored),
            Err(CryptoError::CiphertextTooLarge(_))
        ));
        assert!(matches!(
            decrypt_api_key(Some(&cipher), &stored),
            Err(CryptoError::CiphertextTooLarge(_))
        ));
    }

    #[test]
    fn is_blob_like_rejects_oversize_input() {
        // `is_blob_like` is the gate `decrypt_api_key` uses to decide whether
        // to attempt decryption. An oversize value must be rejected as
        // "not a blob" so `decrypt_api_key` passes it through as plaintext
        // instead of decoding megabytes just to test `.is_ok()`.
        let oversize = "A".repeat(MAX_BLOB_B64_LEN + 1);
        assert!(!CredentialCipher::is_blob_like(&oversize));
    }

    #[test]
    fn decrypt_api_key_legacy_base64_plaintext_fails_closed() {
        // A legacy plaintext that is coincidentally valid standard base64 of
        // ≥ MIN_BLOB_B64_LEN is treated as a legacy blob and AEAD decryption
        // is attempted, which fails authentication. This pins the documented
        // legacy-support tradeoff: such a value fails closed rather than being
        // silently passed through. New writes use the `oxenc1:` magic prefix
        // to avoid the ambiguity entirely.
        let cipher = CredentialCipher::from_passphrase("legacy-edge");
        let legacy_plaintext = base64::engine::general_purpose::STANDARD.encode(b"p".repeat(48));
        assert!(legacy_plaintext.len() >= MIN_BLOB_B64_LEN);
        assert!(!CredentialCipher::looks_encrypted(&legacy_plaintext));
        assert!(CredentialCipher::is_blob_like(&legacy_plaintext));
        assert!(matches!(
            decrypt_api_key(Some(&cipher), &legacy_plaintext),
            Err(CryptoError::AuthenticationFailed)
        ));
    }

    #[test]
    fn decrypt_non_utf8_plaintext_returns_decoding_failed() {
        // Construct a blob whose decrypted payload is not valid UTF-8 by
        // encrypting raw bytes through the AEAD directly (the public `encrypt`
        // API is str-only). This is the only way to reach `DecodingFailed`:
        // GCM authenticates, so corrupting a real blob yields
        // `AuthenticationFailed`, not garbage plaintext.
        let cipher = CredentialCipher::from_passphrase("non-utf8-test");
        let raw_key = cipher.keys[0];
        let aes = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&raw_key));
        let mut nonce_bytes = [0u8; NONCE_LEN];
        AeadOsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        // 0xFF is invalid as the first byte of a 2-byte UTF-8 sequence.
        let payload = [0xFFu8, 0xFE, 0xFD, 0x80];
        let ciphertext = aes
            .encrypt(nonce, payload.as_ref())
            .expect("aes encrypt must succeed");
        let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);
        let stored = format!(
            "{MAGIC}{}",
            base64::engine::general_purpose::STANDARD.encode(&blob)
        );
        assert!(matches!(
            cipher.decrypt(&stored),
            Err(CryptoError::DecodingFailed)
        ));
    }
}
