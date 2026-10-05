//! Built-in browser assets. An installed binary works from any directory,
//! including an otherwise empty container, without runtime CDN dependencies.

use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

const HTMX: &[u8] = include_bytes!("../static/htmx-2.0.4.min.js");
const ALPINE: &[u8] = include_bytes!("../static/alpine-3.14.8.min.js");

pub(crate) async fn serve(Path(path): Path<String>) -> Response {
    let (content_type, bytes): (&str, &[u8]) = match path.as_str() {
        "htmx-2.0.4.min.js" => ("text/javascript; charset=utf-8", HTMX),
        "alpine-3.14.8.min.js" => ("text/javascript; charset=utf-8", ALPINE),
        "logo.png" => ("image/png", include_bytes!("../static/logo.png")),
        "htmx-2.0.4.LICENSE.txt" => (
            "text/plain; charset=utf-8",
            include_bytes!("../static/htmx-2.0.4.LICENSE.txt"),
        ),
        "alpine-3.14.8.LICENSE.txt" => (
            "text/plain; charset=utf-8",
            include_bytes!("../static/alpine-3.14.8.LICENSE.txt"),
        ),
        "vue-reactivity.LICENSE.txt" => (
            "text/plain; charset=utf-8",
            include_bytes!("../static/vue-reactivity.LICENSE.txt"),
        ),
        "PROVENANCE.txt" => (
            "text/plain; charset=utf-8",
            include_bytes!("../static/PROVENANCE.txt"),
        ),
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    let cache = if path.ends_with(".min.js") {
        "public, max-age=31536000, immutable"
    } else {
        "public, max-age=3600"
    };
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, cache),
        ],
        bytes,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn embedded_scripts_match_verified_vendor_hashes() {
        for (bytes, expected) in [
            (
                HTMX,
                "69caefd0da92269066e725d7fe175e26b9d50c962e3056459c0c477154cdb9d3",
            ),
            (
                ALPINE,
                "b600e363d99d95444db54acbfb2deffec9ae792aa99a09229bcda078e5b55643",
            ),
        ] {
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
        }
    }
}
