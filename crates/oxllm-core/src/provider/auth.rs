use std::collections::HashMap;

use super::ResolvedCredentials;
use crate::Deployment;

const AUTH_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "xi-api-key",
];

/// Apply validated deployment authentication overrides after native auth.
/// Removes the native credential headers before inserting the selected header.
/// Extra configured headers are applied afterwards and retain their precedence.
/// Signed credential variants have no bearer and remain owned by the signer.
pub fn apply_auth_overrides(
    deployment: &Deployment,
    creds: &ResolvedCredentials,
    headers: &mut HashMap<String, String>,
) {
    let params = &deployment.oxllm_params;
    if params.auth_header.is_none() && params.auth_scheme.is_none() {
        return;
    }
    let Some(token) = creds.best_effort_bearer() else {
        return;
    };
    let native = AUTH_HEADERS.iter().find_map(|name| {
        headers
            .iter()
            .find(|(header, _)| header.eq_ignore_ascii_case(name))
    });
    let header = params
        .auth_header
        .as_deref()
        .or_else(|| native.map(|(name, _)| name.as_str()))
        .unwrap_or("authorization")
        .to_ascii_lowercase();
    let default_scheme = if params.auth_header.is_some() {
        if header == "authorization" {
            "Bearer"
        } else {
            ""
        }
    } else {
        native
            .and_then(|(_, value)| value.strip_suffix(token))
            .unwrap_or("")
            .trim()
    };
    let scheme = params.auth_scheme.as_deref().unwrap_or(default_scheme);
    let value = if scheme.is_empty()
        || scheme.eq_ignore_ascii_case("raw")
        || scheme.eq_ignore_ascii_case("none")
    {
        token.to_string()
    } else {
        format!("{scheme} {token}")
    };
    headers.retain(|name, _| {
        !AUTH_HEADERS
            .iter()
            .any(|auth| name.eq_ignore_ascii_case(auth))
    });
    headers.insert(header, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::OxllmParams;

    #[test]
    fn overriding_native_auth_removes_old_headers_and_preserves_other_headers() {
        for native in ["authorization", "X-API-Key", "api-key"] {
            let deployment = Deployment {
                oxllm_params: OxllmParams {
                    auth_header: Some("API-Key".into()),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut headers = HashMap::from([
                (native.into(), "Bearer secret".into()),
                ("anthropic-version".into(), "2023-06-01".into()),
            ]);
            apply_auth_overrides(
                &deployment,
                &ResolvedCredentials::from_key("secret"),
                &mut headers,
            );
            assert_eq!(headers["api-key"], "secret");
            assert_eq!(headers.len(), 2);
            assert_eq!(headers["anthropic-version"], "2023-06-01");
        }
    }

    #[test]
    fn schemes_and_repeated_application_do_not_duplicate_credential_prefixes() {
        for (header, scheme, expected) in [
            (None, Some("Token"), "Token secret"),
            (Some("authorization"), None, "Bearer secret"),
            (Some("x-auth"), Some("Bearer"), "Bearer secret"),
            (None, Some("raw"), "secret"),
            (None, Some(""), "secret"),
        ] {
            let deployment = Deployment {
                oxllm_params: OxllmParams {
                    auth_header: header.map(String::from),
                    auth_scheme: scheme.map(String::from),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut headers = HashMap::from([("authorization".into(), "Bearer secret".into())]);
            for _ in 0..2 {
                apply_auth_overrides(
                    &deployment,
                    &ResolvedCredentials::from_key("secret"),
                    &mut headers,
                );
                assert_eq!(headers[header.unwrap_or("authorization")], expected);
                assert_eq!(headers.len(), 1);
            }
        }
    }

    #[test]
    fn invalid_override_syntax_is_rejected_without_echoing_values() {
        for (header, scheme) in [
            ("bad header", "raw"),
            ("x-auth\r\nsecret", "raw"),
            ("api-key", "Bearer secret"),
        ] {
            let params = OxllmParams {
                auth_header: Some(header.into()),
                auth_scheme: Some(scheme.into()),
                ..Default::default()
            };
            let error = params.validate_auth_overrides().unwrap_err();
            assert!(!error.to_string().contains("secret"));
        }
    }
}
