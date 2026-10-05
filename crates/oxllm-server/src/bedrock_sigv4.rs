//! AWS SigV4 request signing for Bedrock upstream calls.
//!
//! Implements [AWS Signature Version 4](https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html)
//! from scratch using only `sha2` + `hmac`, keeping the dependency surface
//! minimal (no `aws-sigv4` / `aws-sdk-*` crates).
//!
//! The host calls [`sign_request`] after serializing the body and before
//! sending the HTTP request. Signing adds `host`, `x-amz-date`,
//! `x-amz-content-sha256`, optionally `x-amz-security-token`, and the final
//! `Authorization` header to the supplied header list — all lowercase, ready
//! for reqwest's case-insensitive `HeaderMap`.
//!
//! The signing key derivation is documented in the AWS reference:
//!
//! ```text
//! kDate    = HMAC("AWS4" + secret_key, date_yyyymmdd)
//! kRegion  = HMAC(kDate,    region)
//! kService = HMAC(kRegion,  service)
//! kSigning = HMAC(kService, "aws4_request")
//! signature = HMAC(kSigning, string_to_sign)
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

/// The SigV4 algorithm identifier (constant for every signed request).
const ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// Sign a request in place.
///
/// `headers` is mutated to receive the canonical SigV4 headers (`host`,
/// `x-amz-date`, `x-amz-content-sha256`, optionally `x-amz-security-token`)
/// plus the final `Authorization` entry. Caller-supplied headers are kept;
/// any pre-existing header with the same lowercase name is overwritten so
/// the signed set is deterministic.
///
/// `service` is `"bedrock"` for Bedrock runtime calls; the function is
/// generic so it can be reused for STS / S3 in the future.
///
/// Returns `Err(message)` when the URL cannot be parsed or lacks a host —
/// the caller must propagate this so the request never goes out unsigned.
#[allow(clippy::too_many_arguments)]
pub fn sign_request(
    method: &str,
    url: &str,
    headers: &mut Vec<(String, String)>,
    body: &[u8],
    access_key: &str,
    secret_key: &str,
    region: &str,
    service: &str,
    session_token: Option<&str>,
) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|_| "invalid signing URL".to_string())?;
    let host = parsed
        .host_str()
        .ok_or_else(|| "signing URL has no host".to_string())?
        .to_string();
    // reqwest/hyper serialize the `Host` header with the port whenever one is
    // explicitly present in the URL (notably non-default ports). The signed
    // `host` value must match or AWS re-derives a different canonical header
    // and rejects the signature.
    let host = match parsed.port() {
        Some(port) => format!("{host}:{port}"),
        None => host,
    };
    // Bedrock uses the ordinary SigV4 path rules: encode the wire path again
    // for the canonical request. Decoding first would turn an ARN's encoded
    // slash into a separator and sign a different resource. Query parameters
    // are encoded only once, separately below. S3 has distinct path rules.
    let raw_path = parsed.path();
    let path = if raw_path.is_empty() { "/" } else { raw_path };
    let query = parsed.query().unwrap_or("");

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Honour a caller-supplied `x-amz-date` so callers (and tests) can pin
    // the timestamp; otherwise stamp the current UTC time. A pinned value must
    // be a valid `yyyyMMddTHHmmssZ` basic ISO-8601 date, otherwise the derived
    // credential scope (`{yyyymmdd}/...`) would be garbage and AWS would
    // reject the signature with no local hint.
    let caller_amz_date = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("x-amz-date"))
        .map(|(_, v)| v.clone());
    let amz_date = match caller_amz_date.as_deref() {
        None => amz_date_format(now),
        Some(d) if is_valid_amz_date(d) => d.to_string(),
        Some(d) => {
            return Err(format!(
                "caller-supplied x-amz-date is not a valid yyyyMMddTHHmmssZ value: {d:?}"
            ));
        }
    };
    // `is_valid_amz_date` guarantees at least 8 leading digits, so the date
    // stamp slice is always in range.
    let date_stamp = amz_date[..8].to_string();

    let payload_hash = hex_encode(&sha256(body));

    // Lowercase every existing header name so the canonical-headers build is
    // case-insensitive (AWS requires lowercase keys).
    for (k, _) in headers.iter_mut() {
        *k = k.to_ascii_lowercase();
    }
    // A caller-supplied `authorization` header would be folded into the
    // signed-headers set, making the signature circular (the signature
    // depends on an authorization value that itself contains the
    // signature). AWS would then reject with `SignatureDoesNotMatch`. No
    // current caller sets it, but `sign_request` is `pub` so this is
    // defense-in-depth.
    headers.retain(|(k, _)| k != "authorization");
    // Drop duplicate names so the signed-headers list never contains the
    // same name twice (`SignedHeaders=host;host` is rejected by AWS with
    // `SignatureDoesNotMatch`). Keeps the FIRST occurrence of each name,
    // matching the existing upsert semantic of "first slot wins".
    dedupe_headers(headers);
    upsert(headers, "host", host.as_str());
    upsert(headers, "x-amz-date", amz_date.as_str());
    upsert(headers, "x-amz-content-sha256", payload_hash.as_str());
    if let Some(token) = session_token.filter(|t| !t.is_empty()) {
        upsert(headers, "x-amz-security-token", token);
    }

    // Canonical headers: sorted by name, "name:value\n" for each, with the
    // value trimmed of surrounding whitespace.
    let mut sorted: Vec<(&str, &str)> = headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let canonical_headers: String = sorted
        .iter()
        .map(|(k, v)| format!("{k}:{}\n", v.trim()))
        .collect();
    let signed_headers: String = sorted.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(";");

    let canonical_request = format!(
        "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
        method = method.to_ascii_uppercase(),
        canonical_uri = canonical_uri(path),
        canonical_query = canonical_query_string(query),
    );

    let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
    let hashed_canonical = hex_encode(&sha256(canonical_request.as_bytes()));
    let string_to_sign =
        format!("{ALGORITHM}\n{amz_date}\n{credential_scope}\n{hashed_canonical}",);

    let signing_key = derive_signing_key(secret_key, &date_stamp, region, service);
    let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    let authorization = format!(
        "{ALGORITHM} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
    );

    upsert(headers, "authorization", authorization.as_str());
    Ok(())
}

/// Derive the SigV4 signing key per the AWS spec.
pub(crate) fn derive_signing_key(
    secret_key: &str,
    date_stamp: &str,
    region: &str,
    service: &str,
) -> Vec<u8> {
    let k_date = hmac_sha256(
        format!("AWS4{secret_key}").as_bytes(),
        date_stamp.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// Insert or replace `name` in the header list. The list is treated as
/// case-insensitive on the (already lowercased) `name`.
fn upsert(headers: &mut Vec<(String, String)>, name: &str, value: &str) {
    if let Some(slot) = headers.iter_mut().find(|(k, _)| k == name) {
        slot.1 = value.to_string();
    } else {
        headers.push((name.to_string(), value.to_string()));
    }
}

/// Remove duplicate header names, keeping the FIRST occurrence of each.
/// Guards against a caller passing both `Host: a` and `host: b` — after
/// lowercasing both become `host`, and without this dedupe [`upsert`] would
/// only overwrite the first slot, leaving two `host` entries that emit
/// `SignedHeaders=host;host` (rejected by AWS).
fn dedupe_headers(headers: &mut Vec<(String, String)>) {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    headers.retain(|(k, _)| seen.insert(k.clone()));
}

/// Format the current UTC time as AWS's `yyyyMMddTHHmmssZ` "amz date".
pub(crate) fn amz_date_format(secs: u64) -> String {
    let (year, month, day, hour, min, sec) = decompose_utc(secs);
    format!("{year:04}{month:02}{day:02}T{hour:02}{min:02}{sec:02}Z",)
}

/// Decompose a Unix timestamp (seconds) into UTC calendar components.
fn decompose_utc(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    // Days since epoch and seconds-of-day.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = (rem / 3600) as u32;
    let min = ((rem % 3600) / 60) as u32;
    let sec = (rem % 60) as u32;

    // Civil-from-days algorithm (Howard Hinnant). Days are relative to the
    // Unix epoch (1970-01-01).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = (if m <= 2 { y + 1 } else { y }) as u32;
    (year, m, d, hour, min, sec)
}

/// Canonical URI: URI-encode each path segment but keep `/` as `/`.
/// AWS uses RFC 3986 unreserved characters untouched and percent-encodes the
/// rest. Bedrock paths are `/model/<id>/invoke` where `<id>` contains dots
/// and colons; colons are NOT in the unreserved set (`A-Za-z0-9-._~`) and
/// must be encoded (`%3A`) or AWS rejects the signature.
fn canonical_uri(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for &byte in path.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    if out.is_empty() { "/".to_string() } else { out }
}

/// Percent-decode query escapes before their single canonical encoding.
/// A stray percent sign remains literal and is subsequently escaped.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_digit(bytes[i + 1]), hex_digit(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Validate a caller-supplied `x-amz-date` is the AWS basic ISO-8601 shape
/// `yyyyMMddTHHmmssZ`. The leading 8 digits are used as the SigV4 date stamp
/// in the credential scope, so a malformed value must be rejected up front.
fn is_valid_amz_date(date: &str) -> bool {
    let b = date.as_bytes();
    if !(b.len() == 16
        && b[..8].iter().all(|c| c.is_ascii_digit())
        && b[8] == b'T'
        && b[9..15].iter().all(|c| c.is_ascii_digit())
        && b[15] == b'Z')
    {
        return false;
    }
    // Validate ranges, not just shape: a shape-valid but out-of-range value
    // (e.g. month 99) would produce a garbage credential scope / date stamp
    // that AWS rejects with SignatureDoesNotMatch — with no local hint.
    let month = (b[4] - b'0') as u32 * 10 + (b[5] - b'0') as u32;
    let day = (b[6] - b'0') as u32 * 10 + (b[7] - b'0') as u32;
    let hour = (b[9] - b'0') as u32 * 10 + (b[10] - b'0') as u32;
    let minute = (b[11] - b'0') as u32 * 10 + (b[12] - b'0') as u32;
    let second = (b[13] - b'0') as u32 * 10 + (b[14] - b'0') as u32;
    let leap = {
        let year: u32 = std::str::from_utf8(&b[..4])
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
    };
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    (1..=12).contains(&month)
        && (1..=days_in_month).contains(&day)
        && hour < 24
        && minute < 60
        && second < 60
}

/// Canonical query string: URI-encode keys and values, then sort by key.
///
/// Each `k`/`v` is percent-decoded once before re-encoding. `sign_request` feeds
/// `url::Url::query()` (which returns the **already percent-encoded** query
/// string) into this function; without the decode step, `%20` would become
/// `%2520` and AWS would reject the signature with `SignatureDoesNotMatch`.
/// A stray `%` that is not a valid escape is left untouched by the decode
/// pass and subsequently encoded to `%25` by [`uri_encode`].
fn canonical_query_string(query: &str) -> String {
    if query.is_empty() {
        return String::new();
    }
    let mut pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| p.split_once('=').unwrap_or((p, "")))
        .map(|(k, v)| {
            (
                uri_encode(&percent_decode(k)),
                uri_encode(&percent_decode(v)),
            )
        })
        .collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// Validate an AWS region name. Mirrors LiteLLM's
/// `_VALID_AWS_REGION_PATTERN = re.compile(r"\A[a-z0-9-]+\Z")`
/// (`litellm/llms/bedrock/base_aws_llm.py:45`): ASCII lowercase letter /
/// digit / hyphen, non-empty, capped at 30 chars to reject stray input.
///
/// A region containing `/`, spaces, or uppercase letters would silently
/// corrupt the SigV4 credential scope (`{date}/{region}/{service}/aws4_request`)
/// and produce an obscure `SignatureDoesNotMatch` from AWS. The error message
/// is intentionally length-only and never echoes the raw region (defense in
/// depth against any future caller-controlled input leaking into logs).
pub fn validate_aws_region(region: &str) -> Result<(), String> {
    if region.is_empty()
        || region.len() > 30
        || !region
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err(format!(
            "invalid AWS region: must match [a-z0-9-]+ (got {} chars)",
            region.len()
        ));
    }
    Ok(())
}

/// RFC 3986 URI encoding for query keys/values.
pub(crate) fn uri_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &byte in input.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    out
}

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// HMAC-SHA256 (RFC 2104), implemented directly on top of [`sha256`] so the
/// function is infallible. The `hmac` crate's `new_from_slice` returns
/// `Result` only because its `Mac` trait is shared with fixed-key-length
/// MACs; HMAC itself accepts any key length (including empty), so the former
/// `.expect` could never fire — but it sat on caller-supplied material
/// (`aws_secret_access_key`), which violates the project's no-panic-on-input
/// convention. Computing HMAC from the already-audited SHA-256 primitive
/// removes that path entirely. Validated against the AWS SigV4 test vector in
/// [`tests::sigv4_known_aws_test_vector`].
const HMAC_BLOCK_SIZE: usize = 64;

pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    // Keys longer than the block size are first hashed; shorter keys are
    // zero-padded to the block size.
    let mut block_key = [0u8; HMAC_BLOCK_SIZE];
    if key.len() > HMAC_BLOCK_SIZE {
        let h = sha256(key);
        block_key[..h.len()].copy_from_slice(&h);
    } else {
        block_key[..key.len()].copy_from_slice(key);
    }

    let mut inner_key = [0u8; HMAC_BLOCK_SIZE];
    let mut outer_key = [0u8; HMAC_BLOCK_SIZE];
    for i in 0..HMAC_BLOCK_SIZE {
        inner_key[i] = block_key[i] ^ 0x36;
        outer_key[i] = block_key[i] ^ 0x5c;
    }

    let mut inner = Sha256::new();
    inner.update(inner_key);
    inner.update(data);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(outer_key);
    outer.update(inner_hash);
    outer.finalize().to_vec()
}

pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// SigV4 service name for the Bedrock runtime.
const BEDROCK_SERVICE: &str = "bedrock";
/// Presigned-URL lifetime for a WebSocket upgrade (15 minutes). AWS caps WS
/// presign at a few minutes; 900s matches LiteLLM's realtime presign.
const WS_PRESIGN_EXPIRES_SECS: u64 = 900;

/// Presign the legacy WebSocket-compatible bridge shape via query signing.
///
/// Native Bedrock uses signed HTTP/2 POST eventstreams, handled by the SDK
/// transport. These URL helpers are only for explicitly configured WS(S)
/// bridges that expose compatible framing and query authentication.
///
/// The signed URL is:
///
/// ```text
/// wss://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/invoke-with-bidirectional-stream
///   ?X-Amz-Algorithm=AWS4-HMAC-SHA256
///   &X-Amz-Credential={access_key}/{credential_scope}
///   &X-Amz-Date={amz_date}
///   &X-Amz-Expires=900
///   &X-Amz-SignedHeaders=host
///   &[X-Amz-Security-Token=...]
///   &X-Amz-Signature={signature}
/// ```
///
/// The canonical request signs `GET` with only `host` as a signed header and
/// `UNSIGNED-PAYLOAD` as the payload hash (the WebSocket upgrade has no body).
/// `model_id` should be the bare id (e.g. `amazon.nova-sonic-v1:0`); its `:`
/// is percent-encoded in both the canonical and final URL paths.
///
/// Reuses `derive_signing_key`, `hmac_sha256`, `sha256`, `hex_encode`,
/// `uri_encode`, and `amz_date_format` — the same verified primitives used
/// by [`sign_request`] and the Mantle bearer-token generator.
pub fn presign_bedrock_ws_url(
    access_key: &str,
    secret_key: &str,
    session_token: Option<&str>,
    region: &str,
    model_id: &str,
) -> Result<String, String> {
    presign_bedrock_ws_url_for_base(
        access_key,
        secret_key,
        session_token,
        region,
        model_id,
        None,
    )
}

/// Presign a runtime endpoint override while retaining its host, port, prefix
/// and non-signature query parameters in both the wire URL and signature.
pub(crate) fn presign_bedrock_ws_url_for_base(
    access_key: &str,
    secret_key: &str,
    session_token: Option<&str>,
    region: &str,
    model_id: &str,
    api_base: Option<&str>,
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
    validate_aws_region(region)?;
    if model_id.is_empty() {
        return Err("model_id is empty".to_string());
    }

    let default_base = format!("https://bedrock-runtime.{region}.amazonaws.com");
    let mut endpoint = url::Url::parse(api_base.unwrap_or(&default_base))
        .map_err(|_| "invalid Bedrock runtime endpoint".to_string())?;
    if !matches!(endpoint.scheme(), "http" | "https" | "ws" | "wss")
        || endpoint.host_str().is_none()
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || endpoint.fragment().is_some()
    {
        return Err("Bedrock runtime endpoint must be an HTTP(S)/WS(S) URL without credentials or a fragment".into());
    }
    let scheme = if matches!(endpoint.scheme(), "http" | "ws") {
        "ws"
    } else {
        "wss"
    };
    endpoint
        .set_scheme(scheme)
        .map_err(|_| "invalid Bedrock WebSocket scheme".to_string())?;
    let host = match endpoint.port() {
        Some(port) => format!("{}:{port}", endpoint.host_str().unwrap_or_default()),
        None => endpoint.host_str().unwrap_or_default().to_string(),
    };
    let prefix = endpoint.path().trim_end_matches('/');
    let path = format!(
        "{prefix}/model/{}/invoke-with-bidirectional-stream",
        uri_encode(model_id)
    );
    endpoint.set_path(&path);
    let path = endpoint.path().to_string();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let amz_date = amz_date_format(now);
    let date_stamp = &amz_date[..8];
    let credential_scope = format!("{date_stamp}/{region}/{BEDROCK_SERVICE}/aws4_request");

    // Query parameters that participate in the signature. Order is irrelevant:
    // the canonical query string is sorted before signing.
    let mut params: Vec<(String, String)> = endpoint
        .query_pairs()
        .filter(|(key, _)| !key.to_ascii_lowercase().starts_with("x-amz-"))
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    params.extend([
        ("X-Amz-Algorithm".into(), ALGORITHM.to_string()),
        (
            "X-Amz-Credential".into(),
            format!("{access_key}/{credential_scope}"),
        ),
        ("X-Amz-Date".into(), amz_date.clone()),
        ("X-Amz-Expires".into(), WS_PRESIGN_EXPIRES_SECS.to_string()),
        ("X-Amz-SignedHeaders".into(), "host".to_string()),
    ]);
    if let Some(token) = session_token.filter(|t| !t.is_empty()) {
        params.push(("X-Amz-Security-Token".into(), token.to_string()));
    }

    // Canonical request for a SigV4 presigned URL: only `host` is signed and
    // the payload hash is `UNSIGNED-PAYLOAD` (the WS upgrade has no body).
    let canonical_query = presign_canonical_query(&params);
    let canonical_request = format!(
        "GET\n{canonical_path}\n{canonical_query}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD",
        canonical_path = canonical_uri(&path),
    );

    let hashed_canonical = hex_encode(&sha256(canonical_request.as_bytes()));
    let string_to_sign =
        format!("{ALGORITHM}\n{amz_date}\n{credential_scope}\n{hashed_canonical}",);

    let signing_key = derive_signing_key(secret_key, date_stamp, region, BEDROCK_SERVICE);
    let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    // Re-assemble the final URL with the signature included. AWS re-sorts the
    // query on verification (excluding `X-Amz-Signature`), so literal order is
    // irrelevant; we keep it sorted for determinism.
    let mut final_params = params;
    final_params.push(("X-Amz-Signature".into(), signature));
    let final_query = presign_canonical_query(&final_params);

    endpoint.set_query(Some(&final_query));
    Ok(endpoint.to_string())
}

/// Build a sorted, RFC-3986-encoded `k=v` query string. Used both for the
/// canonical request (without `X-Amz-Signature`) and for the final presigned
/// URL (with it).
fn presign_canonical_query(params: &[(String, String)]) -> String {
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

    #[test]
    fn amz_date_shape_and_ranges_validated() {
        assert!(is_valid_amz_date("20150830T123600Z"));
        assert!(is_valid_amz_date("20240229T000000Z")); // leap day
        assert!(!is_valid_amz_date("20230229T000000Z")); // non-leap Feb 29
        assert!(!is_valid_amz_date("20151330T123600Z")); // month 13
        assert!(!is_valid_amz_date("20150832T123600Z")); // day 32
        assert!(!is_valid_amz_date("99991231T999999Z")); // hour/sec 99
        assert!(!is_valid_amz_date("20150830T126000Z")); // minute 60
        assert!(!is_valid_amz_date("2015083T123600Z")); // bad shape
    }

    /// Known AWS SigV4 test vector verified against a Python reference
    /// implementation using `hashlib` + `hmac`. The canonical request and
    /// string-to-sign are the AWS "get-vanilla" test shape: a GET request
    /// with no payload, two headers (`host`, `x-amz-date`), and the standard
    /// AWS test credentials.
    #[test]
    fn sigv4_known_aws_test_vector() {
        let secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
        let access_key = "AKIAIOSFODNN7EXAMPLE";
        let region = "us-east-1";
        let service = "service";

        let method = "GET";
        let url = "https://example.amazonaws.com/";
        let body: Vec<u8> = Vec::new();
        let mut headers: Vec<(String, String)> = Vec::new();
        upsert(&mut headers, "host", "example.amazonaws.com");
        upsert(&mut headers, "x-amz-date", "20150830T123600Z");

        // Build the canonical request manually using the same helpers the
        // production signing path uses so we can assert each intermediate
        // value against the AWS spec.
        let parsed = url::Url::parse(url).unwrap();
        let path = parsed.path();
        let path = if path.is_empty() { "/" } else { path };
        let query = parsed.query().unwrap_or("");
        let payload_hash = hex_encode(&sha256(&body));

        let mut sorted: Vec<(&str, &str)> = headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        let canonical_headers: String = sorted
            .iter()
            .map(|(k, v)| format!("{k}:{}\n", v.trim()))
            .collect();
        let signed_headers: String = sorted.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(";");

        let canonical_request = format!(
            "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}",
            method = method,
            canonical_uri = canonical_uri(path),
            canonical_query = canonical_query_string(query),
        );
        assert_eq!(
            canonical_request,
            "GET\n/\n\n\
             host:example.amazonaws.com\n\
             x-amz-date:20150830T123600Z\n\n\
             host;x-amz-date\n\
             e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let credential_scope = format!("20150830/{region}/{service}/aws4_request");
        let hashed = hex_encode(&sha256(canonical_request.as_bytes()));
        let string_to_sign = format!("{ALGORITHM}\n20150830T123600Z\n{credential_scope}\n{hashed}");
        assert_eq!(
            string_to_sign,
            "AWS4-HMAC-SHA256\n\
             20150830T123600Z\n\
             20150830/us-east-1/service/aws4_request\n\
             bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63"
        );

        let signing_key = derive_signing_key(secret_key, "20150830", region, service);
        let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));
        // Reference value computed with Python's `hashlib`+`hmac`:
        //   k = HMAC(b"AWS4"+secret, b"20150830")
        //   k = HMAC(k, b"us-east-1"); k = HMAC(k, b"service")
        //   k = HMAC(k, b"aws4_request")
        //   sig = HMAC(k, string_to_sign)
        assert_eq!(
            signature,
            "ea21d6f05e96a897f6000a1a293f0a5bf0f92a00343409e820dce329ca6365ea"
        );

        // Verify the production `sign_request` produces a valid authorization
        // header whose credential scope and signature format are well-formed.
        // The full signature differs from the manual value above because the
        // production path also adds `x-amz-content-sha256` to the signed
        // header set, which changes the canonical request. The expected
        // signature below was verified against a Python reference impl.
        let mut signed: Vec<(String, String)> = Vec::new();
        upsert(&mut signed, "host", "example.amazonaws.com");
        upsert(&mut signed, "x-amz-date", "20150830T123600Z");
        sign_request(
            method,
            url,
            &mut signed,
            &body,
            access_key,
            secret_key,
            region,
            service,
            None,
        )
        .expect("signing should succeed");
        let auth = signed
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert!(
            auth.starts_with(&format!(
                "AWS4-HMAC-SHA256 Credential={access_key}/20150830/{region}/{service}/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature="
            )),
            "unexpected credential scope/signed headers: {auth}"
        );
        assert!(
            auth.ends_with("b0e9826b8e27230263689c913533611258ba50a1cf46f2c0ae5eea5c777359c2"),
            "production signature must match the Python reference: got {auth}"
        );
    }

    /// `sign_request` populates every canonical SigV4 header and the final
    /// `Authorization` header.
    #[test]
    fn sign_request_adds_required_headers() {
        let mut headers: Vec<(String, String)> = Vec::new();
        sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke",
            &mut headers,
            b"{\"hello\":\"world\"}",
            "AKIATEST",
            "secret",
            "us-east-1",
            "bedrock",
            Some("session-token"),
        )
        .expect("signing should succeed");

        let keys: Vec<&str> = headers.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"host"));
        assert!(keys.contains(&"x-amz-date"));
        assert!(keys.contains(&"x-amz-content-sha256"));
        assert!(keys.contains(&"x-amz-security-token"));
        assert!(keys.contains(&"authorization"));

        let auth = headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIATEST/"));
        assert!(auth.contains("SignedHeaders="));
        assert!(auth.contains("Signature="));

        let date = headers
            .iter()
            .find(|(k, _)| k == "x-amz-date")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert_eq!(date.len(), "yyyyMMddTHHmmssZ".len());
        assert!(date.ends_with('Z'));
    }

    /// `sign_request` is deterministic for the same inputs (no time-dependence
    /// leak into anything except `x-amz-date` / the signature).
    #[test]
    fn sign_request_deterministic_except_for_time() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            &mut a,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");
        // Tiny sleep to dodge sub-second identical timestamps (rare on most
        // clocks but possible at second granularity).
        std::thread::sleep(std::time::Duration::from_millis(1100));
        sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            &mut b,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");
        // x-amz-content-sha256 for `{}` is stable.
        let a_hash = a.iter().find(|(k, _)| k == "x-amz-content-sha256").unwrap();
        let b_hash = b.iter().find(|(k, _)| k == "x-amz-content-sha256").unwrap();
        assert_eq!(a_hash, b_hash);
    }

    #[test]
    fn canonical_uri_passes_unreserved_through_unchanged() {
        assert_eq!(
            canonical_uri("/model/anthropic.claude-3-5/invoke"),
            "/model/anthropic.claude-3-5/invoke"
        );
    }

    #[test]
    fn canonical_uri_encodes_colon_in_model_id() {
        // Bedrock model ids contain `:` (e.g. `...-v2:0`). SigV4 unreserved
        // set is `A-Za-z0-9-._~`, so `:` must be `%3A` or AWS rejects the
        // signature.
        assert_eq!(
            canonical_uri("/model/anthropic.claude-3-5-sonnet-20241022-v2:0/invoke"),
            "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/invoke"
        );
    }

    #[test]
    fn canonical_uri_encodes_spaces() {
        assert_eq!(canonical_uri("/a b"), "/a%20b");
    }

    #[test]
    fn canonical_query_sorts_and_encodes() {
        assert_eq!(canonical_query_string("b=2&a=1&a=2"), "a=1&a=2&b=2");
        assert_eq!(canonical_query_string("k=a b"), "k=a%20b");
    }

    /// A pre-encoded `%XX` in the query must single-encode, not double-encode
    /// to `%25XX`. This mirrors [`sign_request_single_encodes_pre_encoded_path`]
    /// for the path; `url::Url::query()` returns the already-encoded form, so
    /// [`canonical_query_string`] percent-decodes each k/v once before
    /// re-encoding.
    #[test]
    fn canonical_query_string_single_encodes_pre_encoded() {
        // Input arrives from `url::Url::query()` already percent-encoded.
        // AWS re-derives the canonical request from the decoded parameters
        // and expects the single-encoded form on the wire.
        assert_eq!(canonical_query_string("k=a%20b&c=d%2Fe"), "c=d%2Fe&k=a%20b");
        // Flag-only / empty values stay empty (no spurious `=` insertion).
        assert_eq!(canonical_query_string("flag&k=v"), "flag=&k=v");
    }

    #[test]
    fn validate_aws_region_accepts_valid_regions() {
        assert!(validate_aws_region("us-east-1").is_ok());
        assert!(validate_aws_region("us-gov-west-1").is_ok());
        assert!(validate_aws_region("cn-north-1").is_ok());
        assert!(validate_aws_region("ap-south-1").is_ok());
        // 30 chars (the cap) is still ok.
        assert!(validate_aws_region(&"a".repeat(30)).is_ok());
    }

    #[test]
    fn validate_aws_region_rejects_invalid_regions() {
        assert!(validate_aws_region("").is_err(), "empty");
        assert!(validate_aws_region("us-east-1 ").is_err(), "trailing space");
        assert!(validate_aws_region("us-east-1/").is_err(), "slash");
        assert!(validate_aws_region("US-EAST-1").is_err(), "uppercase");
        assert!(validate_aws_region("us east 1").is_err(), "embedded space");
        assert!(validate_aws_region("us-east-1\n").is_err(), "newline");
        assert!(validate_aws_region(&"a".repeat(31)).is_err(), "too long");
    }

    #[test]
    fn amz_date_format_matches_aws_iso8601_basic() {
        // 2024-05-14T12:44:56Z = 1715690696 (verifiable externally).
        assert_eq!(amz_date_format(1_715_690_696), "20240514T124456Z");
    }

    #[test]
    fn decompose_utc_handles_epoch_zero() {
        assert_eq!(decompose_utc(0), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn decompose_utc_handles_leap_year_boundary() {
        // 2020-02-29T00:00:00Z = 1582934400.
        assert_eq!(decompose_utc(1_582_934_400), (2020, 2, 29, 0, 0, 0));
    }

    #[test]
    fn upsert_replaces_existing_key() {
        let mut headers = vec![("host".to_string(), "old".to_string())];
        upsert(&mut headers, "host", "new");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].1, "new");
    }

    #[test]
    fn upsert_appends_when_missing() {
        let mut headers: Vec<(String, String)> = Vec::new();
        upsert(&mut headers, "host", "example.com");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "host");
    }

    #[test]
    fn sign_request_errors_on_unparseable_url() {
        // M9: a bad URL must surface an error, not silently no-op and send
        // the request unsigned.
        let mut headers: Vec<(String, String)> = Vec::new();
        let res = sign_request(
            "POST",
            "not a url",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        );
        assert!(res.is_err(), "expected Err for unparseable url");
        assert!(
            headers.is_empty(),
            "no headers should be added when signing fails"
        );
    }

    #[test]
    fn sign_request_errors_on_url_without_host() {
        let mut headers: Vec<(String, String)> = Vec::new();
        let res = sign_request(
            "POST",
            "file:///etc/hostname",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        );
        assert!(res.is_err(), "expected Err for url without host");
    }

    /// The serialized wire path is encoded again for ordinary SigV4 services.
    #[test]
    fn sign_request_double_encodes_serialized_spaces_in_path() {
        let body = b"{}";
        let mut headers: Vec<(String, String)> = Vec::new();
        upsert(&mut headers, "x-amz-date", "20150830T123600Z");
        sign_request(
            "POST",
            "https://example.amazonaws.com/a b",
            &mut headers,
            body,
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");

        let auth = headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        let actual_sig = auth
            .rsplit("Signature=")
            .next()
            .expect("authorization has a signature");

        // The URL serializer first emits %20; the signing layer emits %2520.
        let payload_hash = hex_encode(&sha256(body));
        let canonical_headers = format!(
            "host:example.amazonaws.com\nx-amz-content-sha256:{payload_hash}\nx-amz-date:20150830T123600Z\n"
        );
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        let canonical_request =
            format!("POST\n/a%2520b\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}");
        let credential_scope = "20150830/us-east-1/bedrock/aws4_request";
        let hashed = hex_encode(&sha256(canonical_request.as_bytes()));
        let string_to_sign = format!("{ALGORITHM}\n20150830T123600Z\n{credential_scope}\n{hashed}");
        let signing_key = derive_signing_key("sk", "20150830", "us-east-1", "bedrock");
        let expected_sig = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

        assert_eq!(
            actual_sig, expected_sig,
            "canonical path must encode the serialized wire path"
        );
    }

    /// Escaped segment bytes keep their wire identity during canonical encoding.
    #[test]
    fn sign_request_preserves_pre_encoded_path_identity() {
        let body = b"{}";
        let mut headers: Vec<(String, String)> = Vec::new();
        upsert(&mut headers, "x-amz-date", "20150830T123600Z");
        sign_request(
            "POST",
            "https://example.amazonaws.com/a%20b",
            &mut headers,
            body,
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");
        let auth = headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        let actual_sig = auth.rsplit("Signature=").next().unwrap();

        // Encoded spaces and literal spaces serialize to the same wire path.
        let payload_hash = hex_encode(&sha256(body));
        let canonical_headers = format!(
            "host:example.amazonaws.com\nx-amz-content-sha256:{payload_hash}\nx-amz-date:20150830T123600Z\n"
        );
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        let canonical_request =
            format!("POST\n/a%2520b\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}");
        let credential_scope = "20150830/us-east-1/bedrock/aws4_request";
        let hashed = hex_encode(&sha256(canonical_request.as_bytes()));
        let string_to_sign = format!("{ALGORITHM}\n20150830T123600Z\n{credential_scope}\n{hashed}");
        let signing_key = derive_signing_key("sk", "20150830", "us-east-1", "bedrock");
        let expected_sig = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));
        assert_eq!(
            actual_sig, expected_sig,
            "canonical encoding must preserve escaped segment boundaries"
        );
    }

    /// The signed `host` header must include an explicit (non-default) port,
    /// matching how reqwest/hyper serialize the `Host` header.
    #[test]
    fn sign_request_includes_port_in_host_header() {
        let mut headers: Vec<(String, String)> = Vec::new();
        sign_request(
            "POST",
            "https://bedrock.local:8443/model/x/invoke",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");
        let host = headers
            .iter()
            .find(|(k, _)| k == "host")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert_eq!(
            host, "bedrock.local:8443",
            "host header must include the port"
        );
    }

    /// No explicit port → host header carries no port (preserves default-port
    /// behavior and the known AWS test vector).
    #[test]
    fn sign_request_omits_port_when_default() {
        let mut headers: Vec<(String, String)> = Vec::new();
        sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");
        let host = headers
            .iter()
            .find(|(k, _)| k == "host")
            .map(|(_, v)| v.as_str())
            .unwrap();
        assert_eq!(host, "bedrock-runtime.us-east-1.amazonaws.com");
    }

    /// A malformed caller-supplied `x-amz-date` must be rejected rather than
    /// producing a garbage credential scope that AWS silently rejects.
    #[test]
    fn sign_request_rejects_malformed_amz_date() {
        let mut headers: Vec<(String, String)> =
            vec![("x-amz-date".to_string(), "bad".to_string())];
        let res = sign_request(
            "POST",
            "https://example.amazonaws.com/x",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        );
        assert!(res.is_err(), "malformed x-amz-date must be rejected");
    }

    /// Flag-only (value-less) query parameters must appear in the canonical
    /// query string with an empty value, not be silently dropped.
    #[test]
    fn canonical_query_keeps_flag_only_params() {
        assert_eq!(canonical_query_string("a=1&flag&b=2"), "a=1&b=2&flag=");
    }

    /// End-to-end check that [`sign_request`] single-encodes a pre-encoded
    /// query string. `url::Url::query()` returns the already-encoded form
    /// (`?k=a%20b&c=d%2Fe`); [`canonical_query_string`] must percent-decode
    /// each k/v once before re-encoding so the canonical request AWS
    /// re-derives matches the one we signed.
    #[test]
    fn sign_request_single_encodes_query_params() {
        let body = b"{}";
        let mut headers: Vec<(String, String)> = Vec::new();
        upsert(&mut headers, "x-amz-date", "20150830T123600Z");
        sign_request(
            "POST",
            "https://example.amazonaws.com/?k=a%20b&c=d%2Fe",
            &mut headers,
            body,
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");
        let auth = headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        let actual_sig = auth.rsplit("Signature=").next().unwrap();

        // Independently rebuild the canonical request with the CORRECT
        // single-encoded canonical query (`c=d%2Fe&k=a%20b`, sorted).
        let payload_hash = hex_encode(&sha256(body));
        let canonical_headers = format!(
            "host:example.amazonaws.com\nx-amz-content-sha256:{payload_hash}\nx-amz-date:20150830T123600Z\n"
        );
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        let canonical_request = format!(
            "POST\n/\nc=d%2Fe&k=a%20b\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );
        let credential_scope = "20150830/us-east-1/bedrock/aws4_request";
        let hashed = hex_encode(&sha256(canonical_request.as_bytes()));
        let string_to_sign = format!("{ALGORITHM}\n20150830T123600Z\n{credential_scope}\n{hashed}");
        let signing_key = derive_signing_key("sk", "20150830", "us-east-1", "bedrock");
        let expected_sig = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));
        assert_eq!(
            actual_sig, expected_sig,
            "query must be single-encoded, not double-encoded"
        );
    }

    /// A caller that accidentally passes the same header name in two cases    /// (`Host` + `host`) must not produce a malformed `SignedHeaders=host;host`.
    /// [`dedupe_headers`] keeps the first occurrence after lowercasing.
    #[test]
    fn sign_request_dedupes_duplicate_header_names() {
        let mut headers: Vec<(String, String)> = vec![
            ("Host".to_string(), "a".to_string()),
            ("host".to_string(), "b".to_string()),
        ];
        sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");
        let host_count = headers.iter().filter(|(k, _)| k == "host").count();
        assert_eq!(host_count, 1, "exactly one host header: {headers:?}");
        // And the SignedHeaders list contains `host` exactly once.
        let auth = headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        let signed = auth
            .split("SignedHeaders=")
            .nth(1)
            .and_then(|s| s.split(',').next())
            .unwrap_or("");
        let host_in_signed = signed.split(';').filter(|h| *h == "host").count();
        assert_eq!(
            host_in_signed, 1,
            "SignedHeaders must list host once: {auth}"
        );
    }

    /// An empty session token must not produce an `x-amz-security-token`
    /// header, matching the guard already present in the presign / mantle
    /// paths. AWS treats an empty `x-amz-security-token` as a malformed /
    /// stale-token indicator and rejects with `InvalidClientTokenId`.
    #[test]
    fn sign_request_omits_security_token_when_empty() {
        let mut headers: Vec<(String, String)> = Vec::new();
        sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            Some(""),
        )
        .expect("signing should succeed");
        assert!(
            !headers.iter().any(|(k, _)| k == "x-amz-security-token"),
            "empty session token must not attach a security-token header: {headers:?}"
        );
        let auth = headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        let signed = auth
            .split("SignedHeaders=")
            .nth(1)
            .and_then(|s| s.split(',').next())
            .unwrap_or("");
        assert!(
            !signed.split(';').any(|h| h == "x-amz-security-token"),
            "empty session token must not appear in SignedHeaders: {auth}"
        );
    }

    /// A caller-supplied `authorization` header must NOT participate in the
    /// signed-headers set. If it did, the signature would be circular (it
    /// depends on an authorization value that itself contains the signature),
    /// and AWS would reject the request. The guard strips any pre-existing
    /// `authorization` (in any case) before computing the signed-headers set.
    /// Defense-in-depth: no current caller sets it, but `sign_request` is pub.
    #[test]
    fn sign_request_strips_pre_existing_authorization_header() {
        let mut headers: Vec<(String, String)> = vec![
            (
                "Authorization".to_string(),
                "Bearer caller-supplied".to_string(),
            ),
            ("X-Amz-Date".to_string(), "20150830T123600Z".to_string()),
        ];
        sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/invoke",
            &mut headers,
            b"{}",
            "AK",
            "sk",
            "us-east-1",
            "bedrock",
            None,
        )
        .expect("signing should succeed");

        // Exactly one authorization entry (the one sign_request added).
        let auths: Vec<&str> = headers
            .iter()
            .filter(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str())
            .collect();
        assert_eq!(auths.len(), 1, "expected a single authorization header");
        assert!(
            auths[0].starts_with("AWS4-HMAC-SHA256 "),
            "the caller-supplied bearer must be overwritten by the SigV4 value: {}",
            auths[0]
        );
        assert!(
            !auths[0].contains("caller-supplied"),
            "stale caller authorization leaked into the output: {}",
            auths[0]
        );

        // The signed-headers list must NOT include authorization.
        let signed = auths[0]
            .split("SignedHeaders=")
            .nth(1)
            .and_then(|s| s.split(',').next())
            .unwrap_or("");
        assert!(
            !signed.split(';').any(|h| h == "authorization"),
            "authorization must not be a signed header: {auths:?}"
        );
    }

    // =========================================================================
    // WebSocket URL presigning (realtime).
    // =========================================================================

    /// Re-derive the signature from a presigned URL and return it, so a test
    /// can compare it against the embedded `X-Amz-Signature`. The strongest
    /// correctness check: proves the URL would be accepted by any SigV4
    /// verifier (including AWS).
    fn rederive_signature(presigned_url: &str) -> String {
        let parsed = url::Url::parse(presigned_url).expect("presigned url parses");
        let host = match parsed.port() {
            Some(port) => format!("{}:{port}", parsed.host_str().expect("host")),
            None => parsed.host_str().expect("host").to_string(),
        };
        let path = canonical_uri(parsed.path());
        // query_pairs() returns percent-DECODED strings; the canonical query
        // must re-encode them (matches presign_canonical_query's uri_encode).
        let pairs: Vec<(String, String)> = parsed
            .query_pairs()
            .map(|(k, v)| (uri_encode(&k), uri_encode(&v)))
            .collect();
        let mut sigless: Vec<(String, String)> = pairs
            .iter()
            .filter(|(k, _)| k != "X-Amz-Signature")
            .cloned()
            .collect();
        sigless.sort();
        let canonical_query = sigless
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        // Recover the decoded amz_date / credential fields for the scope build.
        let raw_pairs: Vec<(String, String)> = parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let amz_date = raw_pairs
            .iter()
            .find(|(k, _)| k == "X-Amz-Date")
            .map(|(_, v)| v.clone())
            .expect("amz date");
        let credential = raw_pairs
            .iter()
            .find(|(k, _)| k == "X-Amz-Credential")
            .map(|(_, v)| v.clone())
            .expect("credential");
        // Credential = access_key/date_stamp/region/service/aws4_request
        let date_stamp = credential.split('/').nth(1).unwrap_or("").to_string();
        let region = credential.split('/').nth(2).unwrap_or("").to_string();
        let canonical_request =
            format!("GET\n{path}\n{canonical_query}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD",);
        let hashed = hex_encode(&sha256(canonical_request.as_bytes()));
        let credential_scope = format!("{date_stamp}/{region}/bedrock/aws4_request");
        let string_to_sign = format!("{ALGORITHM}\n{amz_date}\n{credential_scope}\n{hashed}");
        let signing_key = derive_signing_key(
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            &date_stamp,
            &region,
            "bedrock",
        );
        hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()))
    }

    #[test]
    fn presign_ws_url_shape_and_scheme() {
        let url = presign_bedrock_ws_url(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "amazon.nova-sonic-v1:0",
        )
        .expect("presign ok");
        assert!(url.starts_with("wss://bedrock-runtime.us-east-1.amazonaws.com/"));
        assert!(
            url.contains("/model/amazon.nova-sonic-v1%3A0/invoke-with-bidirectional-stream"),
            "model id colon must be encoded in the path: {url}"
        );
    }

    #[test]
    fn inference_profile_signature_matches_an_independent_sdk_path_vector() {
        // Fixture calculated with the SDK's quote(wire_path, safe='/~') and
        // Python's standard hashlib/hmac, independently of this implementation.
        let path = "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aapplication-inference-profile%2Fabc123/converse";
        let mut headers = vec![("x-amz-date".into(), "20150830T123600Z".into())];
        sign_request(
            "POST",
            &format!("https://bedrock-runtime.us-east-1.amazonaws.com{path}"),
            &mut headers,
            b"{}",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "us-east-1",
            "bedrock",
            None,
        )
        .unwrap();
        let authorization = headers
            .iter()
            .find(|(name, _)| name == "authorization")
            .unwrap()
            .1
            .as_str();
        assert!(authorization.ends_with(
            "Signature=ada1adeafea4ab3830a93c344e190627d20f75a622781b720642049792282182"
        ));
        assert_ne!(canonical_uri(path), canonical_uri(&percent_decode(path)));
    }

    #[test]
    fn presigned_custom_endpoint_keeps_encoded_arn_segment_port_and_query() {
        let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123";
        let signed = presign_bedrock_ws_url_for_base(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            arn,
            Some("https://runtime.example.test:8443/prefix?tenant=a%2Bb&X-Amz-Signature=obsolete"),
        )
        .unwrap();
        let url = url::Url::parse(&signed).unwrap();
        assert!(url.path().starts_with("/prefix/model/arn%3A"));
        assert!(
            url.path()
                .contains("application-inference-profile%2Fabc123/")
        );
        assert_eq!(
            url.query_pairs()
                .filter(|(key, _)| key == "X-Amz-Signature")
                .count(),
            1
        );
        assert!(
            url.query_pairs()
                .any(|(key, value)| key == "tenant" && value == "a+b")
        );
        let signature = url
            .query_pairs()
            .find(|(key, _)| key == "X-Amz-Signature")
            .unwrap()
            .1
            .into_owned();
        assert_eq!(signature, rederive_signature(&signed));
    }

    #[test]
    fn presign_ws_url_carries_required_sigv4_params() {
        let url = presign_bedrock_ws_url(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "amazon.nova-sonic-v1:0",
        )
        .expect("presign ok");
        let parsed = url::Url::parse(&url).unwrap();
        let params: std::collections::HashMap<String, String> = parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        assert_eq!(
            params.get("X-Amz-Algorithm").map(String::as_str),
            Some("AWS4-HMAC-SHA256")
        );
        assert_eq!(
            params.get("X-Amz-SignedHeaders").map(String::as_str),
            Some("host")
        );
        assert_eq!(params.get("X-Amz-Expires").map(String::as_str), Some("900"));
        assert!(params.contains_key("X-Amz-Date"));
        assert!(params.contains_key("X-Amz-Credential"));
        assert!(params.contains_key("X-Amz-Signature"));
        assert!(
            !params.contains_key("X-Amz-Security-Token"),
            "no session token supplied"
        );
    }

    #[test]
    fn presign_ws_url_embeds_session_token_when_supplied() {
        let url = presign_bedrock_ws_url(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            Some("IQoJ/cross+slash=="),
            "us-west-2",
            "amazon.nova-sonic-v1:0",
        )
        .expect("presign ok");
        let parsed = url::Url::parse(&url).unwrap();
        let token = parsed
            .query_pairs()
            .find(|(k, _)| k == "X-Amz-Security-Token")
            .map(|(_, v)| v.into_owned())
            .expect("session token present");
        assert!(
            token.contains("IQoJ") && token.contains("cross") && token.contains("slash"),
            "session token must be preserved (url-decoded): {token}"
        );
    }

    #[test]
    fn presign_ws_url_signature_is_self_consistent() {
        let url = presign_bedrock_ws_url(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "amazon.nova-sonic-v1:0",
        )
        .expect("presign ok");
        let parsed = url::Url::parse(&url).unwrap();
        let embedded = parsed
            .query_pairs()
            .find(|(k, _)| k == "X-Amz-Signature")
            .map(|(_, v)| v.into_owned())
            .unwrap();
        let rederived = rederive_signature(&url);
        assert_eq!(
            embedded, rederived,
            "embedded signature must match an independent re-derivation"
        );
    }

    #[test]
    fn presign_ws_url_changes_when_region_changes() {
        let a = presign_bedrock_ws_url(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "us-east-1",
            "amazon.nova-sonic-v1:0",
        )
        .unwrap();
        let b = presign_bedrock_ws_url(
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
            "eu-west-1",
            "amazon.nova-sonic-v1:0",
        )
        .unwrap();
        assert_ne!(a, b, "region participates in the credential scope / host");
    }

    #[test]
    fn presign_ws_url_rejects_empty_required_inputs() {
        let good_sk = "sk";
        let good_ak = "ak";
        let good_r = "us-east-1";
        let good_m = "m";
        assert!(presign_bedrock_ws_url("", good_sk, None, good_r, good_m).is_err());
        assert!(presign_bedrock_ws_url(good_ak, "", None, good_r, good_m).is_err());
        assert!(presign_bedrock_ws_url(good_ak, good_sk, None, "", good_m).is_err());
        assert!(presign_bedrock_ws_url(good_ak, good_sk, None, good_r, "").is_err());
    }
}
