//! Pure, dependency-free redaction for URLs and connection-string diagnostics.
//!
//! This source is also compiled by `oxllm-db`, which deliberately has no core
//! dependency. Keep it self-contained: no I/O, logging, URL parser, or regex.

const REDACTED: &str = "***";

/// Whether a URL query name carries credentials or signing material.
///
/// Matching ignores ASCII case and separators, and decodes percent-encoded
/// names. Invalid or excessively nested encodings fail closed. Public routing
/// parameters such as `model`, `sslmode`, and `application_name` remain visible.
pub fn is_secret_query_key(name: &str) -> bool {
    let Some(name) = normalized_key(name) else {
        return true;
    };
    matches!(
        name.as_str(),
        "key"
            | "pass"
            | "pwd"
            | "user"
            | "username"
            | "userid"
            | "uid"
            | "auth"
            | "sig"
            | "sas"
            | "sslkey"
    ) || [
        "password",
        "passwd",
        "secret",
        "token",
        "credential",
        "authorization",
        "signature",
        "apikey",
        "accesskey",
        "privatekey",
        "accountkey",
        "routingkey",
        "clientkey",
    ]
    .iter()
    .any(|part| name.contains(part))
        || name.starts_with("xamz")
        || name.starts_with("xgoog")
}

/// Whether a URL's host is a known path-bearing webhook capability endpoint.
/// The path on these hosts is a credential, rather than ordinary route context.
pub fn is_capability_url_host(host: &str) -> bool {
    let host = host.trim_end_matches('.');
    host.eq_ignore_ascii_case("hooks.slack.com") || host.eq_ignore_ascii_case("hooks.slack-gov.com")
}

/// Remove secret URL query pairs, retaining the original encoding of public
/// names and values. Both '&' and legacy ';' separators are understood so a
/// credential cannot hide after an otherwise public query option.
pub fn redact_url_query(query: &str) -> String {
    query
        .split(['&', ';'])
        .filter(|pair| {
            let key = pair.split_once('=').map_or(*pair, |(key, _)| key);
            !pair.is_empty() && !is_secret_query_key(key)
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// Redact every URL and keyword-DSN credential embedded in diagnostic text.
///
/// Unlike parsing a single URL, this handles complete error chains, `Debug`
/// renderings, multiple DSNs, malformed authorities, and libpq/ODBC-style
/// `password='quoted or escaped value'` assignments. URL userinfo becomes
/// `***@`; secret query pairs and fragments are omitted. Host/path and public
/// options remain useful, and SQLite file paths are not treated as credentials.
pub fn redact_connection_text(text: &str) -> String {
    redact_assignments(&redact_urls(text))
}

fn normalized_key(name: &str) -> Option<String> {
    let mut bytes = name.as_bytes().to_vec();
    // Each successful pass shrinks the input. Bound the work even for an
    // adversarial name containing thousands of nested `%25` encodings.
    for _ in 0..4 {
        if !bytes.contains(&b'%') {
            break;
        }
        let mut decoded = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while let Some(&byte) = bytes.get(i) {
            if byte == b'%' {
                let hi = hex_digit(*bytes.get(i + 1)?)?;
                let lo = hex_digit(*bytes.get(i + 2)?)?;
                decoded.push((hi << 4) | lo);
                i += 3;
            } else {
                decoded.push(byte);
                i += 1;
            }
        }
        bytes = decoded;
    }
    if bytes
        .iter()
        .any(|b| !b.is_ascii() || *b == b'%' || b.is_ascii_control())
    {
        return None;
    }
    Some(
        bytes
            .into_iter()
            .filter(u8::is_ascii_alphanumeric)
            .map(|b| char::from(b.to_ascii_lowercase()))
            .collect(),
    )
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn is_scheme_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.')
}

fn is_connection_scheme(scheme: &str) -> bool {
    [
        "postgres",
        "postgresql",
        "mysql",
        "mssql",
        "sqlserver",
        "sqlite",
        "redis",
        "rediss",
        "mongodb",
        "http",
        "https",
        "ws",
        "wss",
    ]
    .iter()
    .any(|candidate| scheme.eq_ignore_ascii_case(candidate))
}

// Byte offsets are sliced only at ASCII delimiters or at the end of the input,
// so Unicode diagnostics and passwords never produce a non-UTF-8 boundary.
fn next_url(text: &str, from: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut i = from;
    while let Some(&byte) = bytes.get(i) {
        if byte.is_ascii_alphabetic()
            && (i == 0
                || !bytes
                    .get(i - 1)
                    .is_some_and(|b| is_scheme_byte(*b) || *b == b'_'))
        {
            let start = i;
            while bytes.get(i).is_some_and(|b| is_scheme_byte(*b)) {
                i += 1;
            }
            if bytes.get(i) == Some(&b':')
                && (text.get(i + 1..).is_some_and(|rest| rest.starts_with("//"))
                    || is_connection_scheme(&text[start..i]))
            {
                return Some((start, i));
            }
        } else {
            i += 1;
        }
    }
    None
}

fn url_end(text: &str, start: usize, colon: usize, limit: usize) -> usize {
    let bytes = text.as_bytes();
    let quote = start
        .checked_sub(1)
        .and_then(|i| bytes.get(i))
        .copied()
        .filter(|b| matches!(b, b'\'' | b'"' | b'`'));
    let mut end = colon + 1;
    let mut escaped = false;
    while end < limit {
        let byte = bytes[end];
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if (quote.is_some_and(|q| byte == q)
            && !suspicious_authority(&text[start..end], colon - start))
            || (quote.is_none()
                && (byte.is_ascii_whitespace()
                    || matches!(byte, b'\'' | b'"' | b'`' | b'<' | b'>')))
        {
            break;
        }
        end += 1;
    }
    // An unquoted, malformed user:password authority or secret query value
    // may contain a raw space or quote. Do not leave the remaining password
    // in the diagnostic. Mask through the rest of that line (or the next URL).
    if quote.is_none()
        && (suspicious_authority(&text[start..end], colon - start)
            || trailing_secret_query(&text[start..end]))
    {
        while end < limit && !matches!(bytes[end], b'\n' | b'\r') {
            end += 1;
        }
    }
    end
}

fn trailing_secret_query(token: &str) -> bool {
    token.split_once('?').is_some_and(|(_, query)| {
        query.split(['&', ';']).next_back().is_some_and(|pair| {
            pair.split_once('=')
                .is_some_and(|(key, _)| is_secret_query_key(key))
        })
    })
}

fn suspicious_authority(token: &str, colon: usize) -> bool {
    if token[..colon].eq_ignore_ascii_case("sqlite") {
        return false;
    }
    let Some(rest) = token.get(colon + 1..).and_then(|s| s.strip_prefix("//")) else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') || authority.starts_with('[') {
        return false;
    }
    authority
        .split_once(':')
        .is_some_and(|(_, port)| port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()))
}

fn redact_urls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    while let Some((start, colon)) = next_url(text, copied) {
        let limit = next_url(text, colon + 1).map_or(text.len(), |(start, _)| start);
        let end = url_end(text, start, colon, limit);
        out.push_str(&text[copied..start]);
        out.push_str(&redact_url_token(&text[start..end], colon - start));
        copied = end;
    }
    out.push_str(&text[copied..]);
    out
}

fn redact_url_token(token: &str, colon: usize) -> String {
    let scheme = &token[..colon];
    let rest = &token[colon + 1..];
    let sqlite = scheme.eq_ignore_ascii_case("sqlite");
    if !sqlite && (!rest.starts_with("//") || suspicious_authority(token, colon)) {
        return format!("{}{}", &token[..colon + 1], REDACTED);
    }
    let without_fragment = token.split('#').next().unwrap_or(token);
    let (base, query) = without_fragment
        .split_once('?')
        .map_or((without_fragment, None), |(base, query)| {
            (base, Some(query))
        });
    let mut safe_base = base.to_string();
    if !sqlite {
        // A slash inside an unescaped password makes the DSN malformed, but
        // must not make its trailing credential fragment appear to be a path.
        if let Some(at) = base[colon + 3..].rfind('@') {
            safe_base = format!("{}***@{}", &base[..colon + 3], &base[colon + 3 + at + 1..]);
        }
        let authority_start = colon + 3;
        let authority_end = safe_base[authority_start..]
            .find('/')
            .map_or(safe_base.len(), |i| authority_start + i);
        let host = safe_base[authority_start..authority_end]
            .rsplit('@')
            .next()
            .unwrap_or("")
            .split(':')
            .next()
            .unwrap_or("");
        if is_capability_url_host(host) && authority_end < safe_base.len() {
            safe_base.truncate(authority_end);
            safe_base.push_str("/***");
        }
    }
    if let Some(query) = query {
        let kept = redact_url_query(query);
        if !kept.is_empty() {
            safe_base.push('?');
            safe_base.push_str(&kept);
        }
    }
    safe_base
}

fn is_key_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'%')
}

fn skip_space(bytes: &[u8], mut i: usize) -> usize {
    while bytes.get(i).is_some_and(u8::is_ascii_whitespace) {
        i += 1;
    }
    i
}

fn followed_by_assignment(bytes: &[u8], separator: usize) -> bool {
    let start = skip_space(bytes, separator + 1);
    let mut end = start;
    while bytes.get(end).is_some_and(|b| is_key_byte(*b)) {
        end += 1;
    }
    end > start && bytes.get(skip_space(bytes, end)) == Some(&b'=')
}

fn assignment_end(text: &str, start: usize) -> (usize, Option<u8>, bool) {
    let bytes = text.as_bytes();
    // A Debug rendering of a DSN escapes its double quotes. Treat `\"...\"`
    // as a quoted value too, so spaces in the password cannot escape masking.
    let debug_quoted = bytes.get(start) == Some(&b'\\')
        && bytes
            .get(start + 1)
            .is_some_and(|b| matches!(b, b'\'' | b'"'));
    let quote_start = start + usize::from(debug_quoted);
    let quote = bytes
        .get(quote_start)
        .copied()
        .filter(|b| matches!(b, b'\'' | b'"'));
    let mut end = quote_start + usize::from(quote.is_some());
    while let Some(&byte) = bytes.get(end) {
        if byte == b'\\' {
            if debug_quoted {
                let run_start = end;
                while bytes.get(end) == Some(&b'\\') {
                    end += 1;
                }
                if bytes.get(end).copied() == quote {
                    // Debug doubles each original backslash and escapes the
                    // quote itself. 1 mod 4 backslashes precede a closing
                    // quote; 3 mod 4 precede a quote escaped inside the DSN.
                    if (end - run_start) % 4 == 1 {
                        if bytes.get(end + 1) == Some(&b'\\')
                            && bytes.get(end + 2).copied() == quote
                        {
                            end += 3;
                            continue;
                        }
                        return (end + 1, quote, true);
                    }
                    end += 1;
                    continue;
                }
                if let Some(ch) = text.get(end..).and_then(|s| s.chars().next()) {
                    end += ch.len_utf8();
                }
                continue;
            }
            end += 1;
            if let Some(ch) = text.get(end..).and_then(|s| s.chars().next()) {
                end += ch.len_utf8();
            }
            continue;
        }
        if let Some(quote) = quote {
            if byte == quote {
                if bytes.get(end + 1) == Some(&quote) {
                    end += 2;
                    continue;
                }
                return (end + 1, Some(quote), true);
            }
        } else if byte.is_ascii_whitespace()
            || (matches!(byte, b';' | b',') && followed_by_assignment(bytes, end))
        {
            break;
        }
        end += 1;
    }
    (end, quote, false)
}

fn redact_assignments(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut i = 0;
    while let Some(&byte) = bytes.get(i) {
        if !is_key_byte(byte) || (i > 0 && is_key_byte(bytes[i - 1])) {
            i += 1;
            continue;
        }
        let start = i;
        while bytes.get(i).is_some_and(|b| is_key_byte(*b)) {
            i += 1;
        }
        if bytes.get(i) == Some(&b':')
            && (is_connection_scheme(&text[start..i])
                || text.get(i + 1..).is_some_and(|s| s.starts_with("//")))
        {
            // URL query secrets were already removed by the URL pass. Do not
            // reinterpret a SQLite filename like `password=data.db` as a DSN
            // assignment (or redact harmless '=' characters in URL paths).
            let limit = next_url(text, i + 1).map_or(text.len(), |(start, _)| start);
            i = url_end(text, start, i, limit);
            continue;
        }
        // ODBC spells this credential `User ID` (libpq uses `user`). Do not
        // mistake the second word for an unrelated, public `ID` setting.
        if text[start..i].eq_ignore_ascii_case("user") {
            let second = skip_space(bytes, i);
            let mut end = second;
            while bytes.get(end).is_some_and(|b| is_key_byte(*b)) {
                end += 1;
            }
            if text.get(second..end).is_some_and(|word| {
                word.eq_ignore_ascii_case("id") || word.eq_ignore_ascii_case("name")
            }) {
                i = end;
            }
        }
        let equals = skip_space(bytes, i);
        if bytes.get(equals) != Some(&b'=') || !is_secret_query_key(&text[start..i]) {
            continue;
        }
        let value_start = skip_space(bytes, equals + 1);
        let (end, quote, closed) = assignment_end(text, value_start);
        out.push_str(&text[copied..value_start]);
        if let Some(quote) = quote {
            out.push(char::from(quote));
        }
        out.push_str(REDACTED);
        if closed {
            if let Some(quote) = quote {
                out.push(char::from(quote));
            }
        }
        copied = end;
        i = end;
    }
    out.push_str(&text[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_names_are_case_and_encoding_insensitive() {
        for name in [
            "PaSSword",
            "api%5Fkey",
            "%41uthorization",
            "API-KEY",
            "p%2561ssword",
            "AWSAccessKeyId",
            "x-aMz-SiGnAtUrE",
            "%58-Amz-Credential",
            "sig",
            "sslpassword",
            "access_token",
            "bad%",
            "%252525252574oken",
        ] {
            assert!(is_secret_query_key(name), "must redact {name}");
        }
        for name in [
            "model",
            "sslmode",
            "application_name",
            "connect_timeout",
            "api-version",
        ] {
            assert!(!is_secret_query_key(name), "must preserve {name}");
        }
    }

    #[test]
    fn query_redaction_preserves_public_encoding_and_handles_legacy_separators() {
        assert_eq!(
            redact_url_query(
                "sslmode=require;PaSS%77ord=hidden&application_name=proxy%20app&api-version=2"
            ),
            "sslmode=require&application_name=proxy%20app&api-version=2"
        );
    }

    #[test]
    fn all_embedded_dsns_and_query_secrets_are_redacted() {
        let raw = "connect PostGreSQL://alice:one@db:5432/app?sslmode=require&PaSS%77ord=two; retry mysql://bob:three@backup/prod?api%5Fkey=four&application_name=proxy";
        let safe = redact_connection_text(raw);
        for secret in ["alice", "one", "two", "bob", "three", "four"] {
            assert!(!safe.contains(secret), "leaked {secret}: {safe}");
        }
        assert!(safe.contains("PostGreSQL://***@db:5432/app?sslmode=require"));
        assert!(safe.contains("mysql://***@backup/prod?application_name=proxy"));
        assert!(safe.contains("retry"));
    }

    #[test]
    fn keyword_passwords_support_quotes_and_escapes() {
        for raw in [
            r"host=db user=alice password='two words\' and \\slashes' dbname=app",
            r#"Host=db USER=alice Password="two words\" and \\slashes" dbname=app"#,
            r"host=db password=escaped\ space\;suffix dbname=app",
            "host=db password='doubled''quote and space' dbname=app",
            "Host=db;User ID=alice;Password=semi;colon;Database=app",
        ] {
            let safe = redact_connection_text(raw);
            for secret in [
                "alice",
                "two words",
                "slashes",
                "escaped",
                "suffix",
                "doubled",
                "semi",
                "colon",
            ] {
                assert!(!safe.contains(secret), "leaked {secret}: {safe}");
            }
            assert!(safe.contains("db"));
            assert!(safe.contains("app"));
        }
    }

    #[test]
    fn unterminated_passwords_fail_closed() {
        let safe = redact_connection_text("host=db password='broken credential still here");
        assert_eq!(safe, "host=db password='***");
    }

    #[test]
    fn malformed_authorities_do_not_leak_credential_fragments() {
        for raw in [
            "postgres://alice:fragment@ss@host/db?token=hidden",
            "postgres://alice:fragment/part@host:notaport/db?token=hidden",
            "postgres://alice:fragment@[broken/db?token=hidden",
            "postgres://alice:fragment?part@host/db",
            "postgres://alice:fragment part@host/db failed",
            "postgres://alice:fragment'part@host/db failed",
            "postgres:/alice:fragment@host/db",
            "postgres://alice:fragment",
            "'postgres://alice:fragment'part@host/db' failed",
        ] {
            let safe = redact_connection_text(raw);
            for secret in ["alice", "fragment", "part@", "hidden"] {
                assert!(!safe.contains(secret), "leaked {secret}: {safe}");
            }
        }
    }

    #[test]
    fn malformed_unquoted_query_passwords_fail_closed_without_hiding_other_dsns() {
        let raw = "postgres://db/app?password=two secret words; retry mysql://bob:backup-secret@backup/app";
        let safe = redact_connection_text(raw);
        for secret in ["two", "secret words", "bob", "backup-secret"] {
            assert!(!safe.contains(secret), "leaked {secret}: {safe}");
        }
        assert!(safe.contains("postgres://db/app"));
        assert!(safe.contains("mysql://***@backup/app"));
    }

    #[test]
    fn sqlite_paths_and_public_diagnostics_remain_visible() {
        for raw in [
            "database is locked",
            "sqlite::memory:",
            "sqlite:/var/lib/oxllm/db.sqlite",
            "sqlite:///var/lib/oxllm/alice@data.sqlite?mode=rwc&cache=shared",
            "sqlite://relative@data.sqlite",
            "sqlite:/var/password=data.sqlite",
            "host=db dbname=app sslmode=require",
        ] {
            assert_eq!(redact_connection_text(raw), raw);
        }
        assert_eq!(
            redact_connection_text("sqlite:/data.sqlite?mode=rwc&password=hidden"),
            "sqlite:/data.sqlite?mode=rwc"
        );
    }

    #[test]
    fn debug_escaped_urls_and_unicode_are_safe() {
        let raw = format!(
            "driver rejected {:?}; backup {:?}",
            "postgres://alice:p\\\"äss@host/db?ACCESS_TOKEN=hidden",
            "redis://bob:秘密@backup/0?api_key=other"
        );
        let safe = redact_connection_text(&raw);
        for secret in ["alice", "äss", "hidden", "bob", "秘密", "other"] {
            assert!(!safe.contains(secret), "leaked {secret}: {safe}");
        }
        assert!(safe.contains("host/db"));
        assert!(safe.contains("backup/0"));
    }

    #[test]
    fn debug_escaped_keyword_passwords_mask_all_words_and_escaped_quotes() {
        let dsn = r#"host=db user=alice password="two words\" and \\slashes" dbname=app"#;
        let safe = redact_connection_text(&format!("driver rejected {dsn:?}"));
        for secret in ["alice", "two", "words", "slashes"] {
            assert!(!safe.contains(secret), "leaked {secret}: {safe}");
        }
        assert!(safe.contains("host=db"));
        assert!(safe.contains("dbname=app"));
    }

    #[test]
    fn slack_capability_paths_are_never_diagnostic_context() {
        let safe = redact_connection_text(
            "POST https://hooks.slack.com/services/team/channel/capability failed",
        );
        assert_eq!(safe, "POST https://hooks.slack.com/*** failed");
    }
}
