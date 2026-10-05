//! Minimal query-value escaping (enough for key/model ids and ISO dates).

pub(crate) fn urlencoding_minimal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_unreserved_set_untouched() {
        for c in ['A', 'z', '0', '9', '-', '_', '.', '~'] {
            assert_eq!(urlencoding_minimal(&c.to_string()), c.to_string());
        }
        // A representative unreserved identifier.
        assert_eq!(urlencoding_minimal("key-123_abc.4o"), "key-123_abc.4o");
    }

    #[test]
    fn escapes_special_ascii_bytes() {
        // `/` -> %2F, ` ` -> %20, `+` -> %2B, `=` -> %3D.
        assert_eq!(urlencoding_minimal("a/b c+d=e"), "a%2Fb%20c%2Bd%3De");
    }

    #[test]
    fn escapes_multibyte_utf8_per_byte() {
        // "é" is 0xC3 0xA9; "中" is 0xE4 0xB8 0xAD. Each byte is escaped
        // individually as %XX (the proxy decodes percent-encoded UTF-8).
        assert_eq!(urlencoding_minimal("café"), "caf%C3%A9");
        assert_eq!(urlencoding_minimal("中文"), "%E4%B8%AD%E6%96%87");
    }
}
