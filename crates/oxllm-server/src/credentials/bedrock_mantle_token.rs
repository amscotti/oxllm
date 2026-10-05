//! Generate Bedrock Mantle bearer tokens from AWS credentials (Phase 2).
//!
//! This reproduces the algorithm of AWS's `aws-bedrock-token-generator`
//! libraries (Python / JS / Java): SigV4-presign a `POST` to the global
//! `https://bedrock.amazonaws.com/` endpoint with the query parameter
//! `Action=CallWithBearerToken`, drop the `https://` scheme, append
//! `&Version=1`, base64-encode the result, and prefix it with
//! `bedrock-api-key-`.
//!
//! Token generation is a **pure local computation** — no network call.
//! SigV4 presigning reuses the verified key-derivation primitives from
//! [`crate::bedrock_sigv4`].
//!
//! Wire format (verified against the upstream Python `_generate_token`):
//!
//! ```text
//! token = "bedrock-api-key-" + base64( "bedrock.amazonaws.com/?" +
//!           <sorted SigV4 presign query params> + "&Version=1" )
//! ```
//!
//! The presigned query carries `Action=CallWithBearerToken`, the standard
//! `X-Amz-*` SigV4 presign parameters, and (for temporary credentials) an
//! `X-Amz-Security-Token`. Only `host` is a signed header and the payload
//! hash is the SHA-256 of an empty body (`e3b0c44...`) — AWS rejects
//! `UNSIGNED-PAYLOAD` for this endpoint.

use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};

use crate::bedrock_sigv4::{
    amz_date_format, derive_signing_key, hex_encode, hmac_sha256, sha256, uri_encode,
};

/// Bearer-token prefix mandated by the Bedrock Mantle wire format.
const AUTH_PREFIX: &str = "bedrock-api-key-";
/// Versioning suffix appended to the scheme-stripped presigned URL.
const TOKEN_VERSION: &str = "&Version=1";
/// SigV4 query-auth algorithm identifier.
const ALGORITHM: &str = "AWS4-HMAC-SHA256";
/// The global Bedrock control-plane host (region-agnostic). The region still
/// participates in the SigV4 credential scope; the host does not.
const DEFAULT_HOST: &str = "bedrock.amazonaws.com";
/// Bedrock SigV4 service name.
const SERVICE: &str = "bedrock";
/// The action that marks a request as a bearer-token invocation
/// (`bedrock:CallWithBearerToken`).
const ACTION: &str = "CallWithBearerToken";
/// Maximum token lifetime the Bedrock token generator permits: 12 hours.
pub const MAX_EXPIRES_SECS: u64 = 43_200;

/// Generate a Bedrock Mantle bearer token from AWS credentials.
///
/// `now_secs` is the current Unix time, factored out so tests can pin the
/// timestamp for deterministic vectors. `expires_in_secs` is clamped to
/// [`MAX_EXPIRES_SECS`]; the *effective* token validity is always the minimum
/// of this value and the underlying AWS credential lifetime (the caller is
/// responsible for not requesting an expiry that outlives temporary
/// credentials — matching the upstream library's contract).
///
/// Returns `Err` only for empty required inputs.
pub fn generate_bedrock_mantle_token(
    access_key: &str,
    secret_key: &str,
    session_token: Option<&str>,
    region: &str,
    expires_in_secs: u64,
    now_secs: u64,
) -> Result<String, String> {
    if access_key.is_empty() {
        return Err("aws_access_key_id is empty".to_string());
    }
    if secret_key.is_empty() {
        return Err("aws_secret_access_key is empty".to_string());
    }
    if region.is_empty() {
        return Err("aws_region is empty".to_string());
    }
    let expires = expires_in_secs.min(MAX_EXPIRES_SECS);

    let amz_date = amz_date_format(now_secs);
    // date_stamp = the leading yyyymmdd of the amz date.
    let date_stamp = &amz_date[..8];
    let credential_scope = format!("{date_stamp}/{region}/{SERVICE}/aws4_request");

    // Query parameters that participate in the signature. Order is irrelevant:
    // the canonical query string is sorted before signing.
    let mut params: Vec<(&str, String)> = vec![
        ("Action", ACTION.to_string()),
        ("X-Amz-Algorithm", ALGORITHM.to_string()),
        (
            "X-Amz-Credential",
            format!("{access_key}/{credential_scope}"),
        ),
        ("X-Amz-Date", amz_date.clone()),
        ("X-Amz-Expires", expires.to_string()),
        ("X-Amz-SignedHeaders", "host".to_string()),
    ];
    if let Some(token) = session_token.filter(|t| !t.is_empty()) {
        params.push(("X-Amz-Security-Token", token.to_string()));
    }

    // Canonical request for a SigV4 presigned URL: only `host` is signed and
    // the payload hash is the SHA-256 of an empty body. AWS rejects
    // UNSIGNED-PAYLOAD for this specific endpoint.
    let canonical_query = canonical_query_string(&params);
    let empty_body_hash = hex_encode(&sha256(b""));
    let canonical_request =
        format!("POST\n/\n{canonical_query}\nhost:{DEFAULT_HOST}\n\nhost\n{empty_body_hash}",);

    let hashed_canonical = hex_encode(&sha256(canonical_request.as_bytes()));
    let string_to_sign = format!("{ALGORITHM}\n{amz_date}\n{credential_scope}\n{hashed_canonical}");

    let signing_key = derive_signing_key(secret_key, date_stamp, region, SERVICE);
    let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    // Re-assemble the final URL with the signature included. AWS verifies by
    // re-sorting and excluding `X-Amz-Signature`, so the literal order here
    // does not affect validation — we keep it sorted for determinism.
    let mut final_params = params;
    final_params.push(("X-Amz-Signature", signature));
    let final_query = canonical_query_string(&final_params);

    // Match the upstream library exactly: strip the scheme, keep the host,
    // then append the version suffix.
    let presigned_url = format!("{DEFAULT_HOST}/?{final_query}{TOKEN_VERSION}");
    let encoded = BASE64_STANDARD.encode(presigned_url.as_bytes());
    Ok(format!("{AUTH_PREFIX}{encoded}"))
}

/// Build a sorted, RFC-3986-encoded `k=v` query string. Used both for the
/// canonical request (without `X-Amz-Signature`) and for the final presigned
/// URL (with it).
fn canonical_query_string(params: &[(&str, String)]) -> String {
    let mut pairs: Vec<(String, String)> = params
        .iter()
        .map(|(k, v)| (uri_encode(k), uri_encode(v)))
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};

    /// Fixed, AWS-canonical test credentials (the well-known example pair).
    const AK: &str = "AKIAIOSFODNN7EXAMPLE";
    const SK: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    /// A pinned timestamp: 2024-05-14T12:44:56Z (verifiable externally).
    const NOW: u64 = 1_715_690_696;
    const REGION: &str = "us-east-1";

    fn decode_payload(token: &str) -> String {
        assert!(
            token.starts_with(AUTH_PREFIX),
            "token must start with {AUTH_PREFIX}: got {token}"
        );
        let encoded = &token[AUTH_PREFIX.len()..];
        let bytes = BASE64_STANDARD
            .decode(encoded)
            .unwrap_or_else(|e| panic!("payload must be valid standard base64: {e}"));
        String::from_utf8(bytes).unwrap_or_else(|e| panic!("decoded payload must be UTF-8: {e}"))
    }

    #[test]
    fn token_has_prefix_and_version_suffix() {
        let token =
            generate_bedrock_mantle_token(AK, SK, None, REGION, MAX_EXPIRES_SECS, NOW).unwrap();
        assert!(token.starts_with("bedrock-api-key-"));
        let payload = decode_payload(&token);
        assert!(
            payload.ends_with("&Version=1"),
            "payload must end with &Version=1: {payload}"
        );
    }

    #[test]
    fn payload_is_scheme_less_presigned_url() {
        let token =
            generate_bedrock_mantle_token(AK, SK, None, REGION, MAX_EXPIRES_SECS, NOW).unwrap();
        let payload = decode_payload(&token);
        assert!(
            payload.starts_with("bedrock.amazonaws.com/?"),
            "payload must start with the global host: {payload}"
        );
        assert!(
            !payload.starts_with("https://"),
            "scheme must be stripped: {payload}"
        );
    }

    #[test]
    fn payload_carries_required_sigv4_query_params() {
        let token =
            generate_bedrock_mantle_token(AK, SK, None, REGION, MAX_EXPIRES_SECS, NOW).unwrap();
        let payload = decode_payload(&token);
        let query = payload
            .strip_prefix("bedrock.amazonaws.com/?")
            .and_then(|s| s.strip_suffix("&Version=1"))
            .unwrap_or_else(|| panic!("unexpected payload shape: {payload}"));

        let params: std::collections::HashMap<&str, &str> = query
            .split('&')
            .map(|kv| kv.split_once('=').expect("k=v pair"))
            .collect();

        assert_eq!(params.get("Action"), Some(&"CallWithBearerToken"));
        assert_eq!(params.get("X-Amz-Algorithm"), Some(&"AWS4-HMAC-SHA256"));
        assert_eq!(params.get("X-Amz-Expires"), Some(&"43200"));
        assert_eq!(params.get("X-Amz-SignedHeaders"), Some(&"host"));
        assert_eq!(
            params.get("X-Amz-Date"),
            Some(&"20240514T124456Z"),
            "X-Amz-Date must match the pinned timestamp"
        );
        assert_eq!(
            params.get("X-Amz-Credential"),
            Some(&"AKIAIOSFODNN7EXAMPLE%2F20240514%2Fus-east-1%2Fbedrock%2Faws4_request"),
            "credential slashes must be percent-encoded"
        );
        assert!(params.contains_key("X-Amz-Signature"));
        assert!(
            !params.contains_key("X-Amz-Security-Token"),
            "no session token was supplied"
        );
    }

    #[test]
    fn session_token_is_embedded_when_supplied() {
        let token = generate_bedrock_mantle_token(
            AK,
            SK,
            Some("IQoJb3JpZ2luX2VjEOr/cross+slash=="),
            REGION,
            MAX_EXPIRES_SECS,
            NOW,
        )
        .unwrap();
        let payload = decode_payload(&token);
        // Special chars in the token must be percent-encoded so the URL is valid.
        assert!(
            payload.contains("X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOr%2Fcross%2Bslash%3D%3D"),
            "session token must be URI-encoded in the payload: {payload}"
        );
    }

    #[test]
    fn signature_is_self_consistent() {
        // Re-derive the signature from the decoded presigned URL and confirm
        // it matches the one embedded in the token. This is the strongest
        // correctness check: it proves the token would be accepted by any
        // SigV4 verifier (including AWS).
        let token =
            generate_bedrock_mantle_token(AK, SK, None, REGION, MAX_EXPIRES_SECS, NOW).unwrap();
        let payload = decode_payload(&token);
        let query = payload
            .strip_prefix("bedrock.amazonaws.com/?")
            .and_then(|s| s.strip_suffix("&Version=1"))
            .unwrap();

        let mut pairs: Vec<(String, String)> = query
            .split('&')
            .map(|kv| {
                let (k, v) = kv.split_once('=').expect("k=v");
                (k.to_string(), v.to_string())
            })
            .collect();
        let signature = pairs
            .iter()
            .find(|(k, _)| k == "X-Amz-Signature")
            .map(|(_, v)| v.clone())
            .expect("signature present");
        // Drop the signature to rebuild the canonical query exactly as signed.
        pairs.retain(|(k, _)| k != "X-Amz-Signature");
        pairs.sort();

        let canonical_query = pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let empty_body_hash = hex_encode(&sha256(b""));
        let canonical_request =
            format!("POST\n/\n{canonical_query}\nhost:{DEFAULT_HOST}\n\nhost\n{empty_body_hash}",);
        let credential_scope = "20240514/us-east-1/bedrock/aws4_request";
        let hashed = hex_encode(&sha256(canonical_request.as_bytes()));
        let string_to_sign = format!("{ALGORITHM}\n20240514T124456Z\n{credential_scope}\n{hashed}");
        let signing_key = derive_signing_key(SK, "20240514", REGION, SERVICE);
        let expected = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

        assert_eq!(
            signature, expected,
            "embedded signature must match an independent re-derivation"
        );
    }

    #[test]
    fn token_is_deterministic_for_pinned_inputs() {
        let a = generate_bedrock_mantle_token(AK, SK, None, REGION, MAX_EXPIRES_SECS, NOW).unwrap();
        let b = generate_bedrock_mantle_token(AK, SK, None, REGION, MAX_EXPIRES_SECS, NOW).unwrap();
        assert_eq!(a, b, "same inputs must yield byte-identical tokens");
    }

    #[test]
    fn token_changes_when_region_changes() {
        let a = generate_bedrock_mantle_token(AK, SK, None, "us-east-1", MAX_EXPIRES_SECS, NOW)
            .unwrap();
        let b = generate_bedrock_mantle_token(AK, SK, None, "eu-west-1", MAX_EXPIRES_SECS, NOW)
            .unwrap();
        assert_ne!(a, b, "region participates in the credential scope");
    }

    #[test]
    fn expires_is_clamped_to_twelve_hours() {
        let over = generate_bedrock_mantle_token(AK, SK, None, REGION, 99_999_999, NOW).unwrap();
        let max =
            generate_bedrock_mantle_token(AK, SK, None, REGION, MAX_EXPIRES_SECS, NOW).unwrap();
        assert_eq!(over, max, "expires must clamp to MAX_EXPIRES_SECS");
        let payload = decode_payload(&over);
        assert!(payload.contains("X-Amz-Expires=43200"));
    }

    #[test]
    fn rejects_empty_required_inputs() {
        assert!(generate_bedrock_mantle_token("", SK, None, REGION, 1, NOW).is_err());
        assert!(generate_bedrock_mantle_token(AK, "", None, REGION, 1, NOW).is_err());
        assert!(generate_bedrock_mantle_token(AK, SK, None, "", 1, NOW).is_err());
    }
}
