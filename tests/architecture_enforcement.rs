//! Workspace-wide architecture and code-principle enforcement tests.
//!
//! Enforces invariants documented in `docs/architecture.md`, `CONTRIBUTING.md`,
//! `docs/adding-a-provider.md`, and `docs/adding-a-feature.md`:
//! - Every crate and binary root forbids `unsafe_code` and zero `unsafe`
//!   blocks appear in non-test production source.
//! - Providers (`oxllm-providers`) contain no `async fn` (synchronous pure
//!   transforms).
//! - Production code avoids `todo!`, `unimplemented!`, `dbg!`, stdout/stderr
//!   macros in library crates, and unallowlisted `.unwrap()` / `.expect()` /
//!   `panic!()`.
//! - `os.environ/` resolution lives strictly in config loaders, and hot-path
//!   request/core/provider modules never call `std::env::var*`.
//! - `Coordinator` is used only for `AuthBrake` and `AppState` wiring; inline
//!   `format!("spend:...")` keys live only in `hooks/spend_keys.rs`.
//! - Hook ordering (`RateLimitHook -> CacheHook -> BudgetHook ->
//!   CostTrackingHook -> MetricsHook`) and `AppState` hot-reload preservation
//!   hold both structurally and behaviorally.
//! - Secret-bearing types redact secrets in `Debug` output and production
//!   `tracing::*!` calls do not log raw config/secret fields.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use oxllm_core::provider::ResolvedCredentials;
use oxllm_core::{CallContext, Deployment, OxllmParams};
use oxllm_server::config::types::{
    CacheConfig, GeneralSettings, LangfuseConfig, OxllmSettings, ProxyConfig, RouterSettings,
    S3LogConfig, SemanticCacheConfig, SlackAlertConfig,
};
use oxllm_server::coordination::Coordinator;
use oxllm_server::crypto::CredentialCipher;
use oxllm_server::hooks::cache_hook::cache_key_for;
use oxllm_server::hooks::{LangfuseHook, S3LogHook, run_pre_call_hooks};
use oxllm_server::security::AuthBrake;
use oxllm_server::{AppState, Cache, InMemoryCache};
use serde_json::json;
use toml::Table;

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokKind {
    Ident,
    Str,
    Char,
    Punct,
}

#[derive(Debug, Clone)]
struct Tok {
    kind: TokKind,
    text: String,
    line: usize,
}

/// Tokenize Rust source while stripping comments, strings, and char literals
/// (preserving lifetimes like `'a`).
fn tokenize_rust(src: &str) -> Vec<Tok> {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    let mut line = 1usize;
    let mut out = Vec::new();

    while i < n {
        let b = bytes[i];
        if b == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            while i < n && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
            let mut depth = 1usize;
            i += 2;
            while i < n && depth > 0 {
                if bytes[i] == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && i + 1 < n && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    if bytes[i] == b'\n' {
                        line += 1;
                    }
                    i += 1;
                }
            }
            continue;
        }
        if let Some((after_open, hashes)) = match_raw_string_prefix(bytes, i) {
            let start_line = line;
            let content_start = after_open;
            let mut j = after_open;
            let mut content_end = n;
            while j < n {
                if bytes[j] == b'\n' {
                    line += 1;
                }
                if bytes[j] == b'"'
                    && j + hashes < n
                    && bytes[j + 1..=j + hashes].iter().all(|c| *c == b'#')
                {
                    content_end = j;
                    j += 1 + hashes;
                    break;
                }
                j += 1;
            }
            i = j;
            out.push(Tok {
                kind: TokKind::Str,
                text: String::from_utf8_lossy(&bytes[content_start..content_end]).into_owned(),
                line: start_line,
            });
            continue;
        }
        if b == b'"' || ((b == b'b' || b == b'c') && i + 1 < n && bytes[i + 1] == b'"') {
            let start_line = line;
            i += if b == b'"' { 1 } else { 2 };
            let content_start = i;
            let mut content_end = n;
            while i < n {
                if bytes[i] == b'\\' {
                    if i + 1 < n && bytes[i + 1] == b'\n' {
                        line += 1;
                    }
                    i += 2;
                } else if bytes[i] == b'"' {
                    content_end = i;
                    i += 1;
                    break;
                } else {
                    if bytes[i] == b'\n' {
                        line += 1;
                    }
                    i += 1;
                }
            }
            out.push(Tok {
                kind: TokKind::Str,
                text: String::from_utf8_lossy(&bytes[content_start..content_end]).into_owned(),
                line: start_line,
            });
            continue;
        }
        if b == b'b' && i + 1 < n && bytes[i + 1] == b'\'' {
            let start_line = line;
            i += 2;
            while i < n {
                if bytes[i] == b'\\' {
                    i += 2;
                } else if bytes[i] == b'\'' {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            out.push(Tok {
                kind: TokKind::Char,
                text: String::new(),
                line: start_line,
            });
            continue;
        }
        if b == b'\'' {
            if i + 1 < n && bytes[i + 1] == b'\\' {
                let start_line = line;
                i += 2;
                while i < n && bytes[i] != b'\'' && bytes[i] != b'\n' {
                    if bytes[i] == b'\\' {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if i < n && bytes[i] == b'\'' {
                    i += 1;
                }
                out.push(Tok {
                    kind: TokKind::Char,
                    text: String::new(),
                    line: start_line,
                });
                continue;
            } else if i + 1 < n
                && let Some(ch) = src[i + 1..].chars().next()
            {
                let after_ch = i + 1 + ch.len_utf8();
                if ch != '\n' && ch != '\'' && after_ch < n && bytes[after_ch] == b'\'' {
                    out.push(Tok {
                        kind: TokKind::Char,
                        text: String::new(),
                        line,
                    });
                    i = after_ch + 1;
                    continue;
                }
            }
        }
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push(Tok {
                kind: TokKind::Ident,
                text: String::from_utf8_lossy(&bytes[start..i]).into_owned(),
                line,
            });
            continue;
        }
        if i + 1 < n && &bytes[i..i + 2] == b"::" {
            out.push(Tok {
                kind: TokKind::Punct,
                text: "::".to_string(),
                line,
            });
            i += 2;
            continue;
        }
        out.push(Tok {
            kind: TokKind::Punct,
            text: (b as char).to_string(),
            line,
        });
        i += 1;
    }

    out
}

fn match_raw_string_prefix(bytes: &[u8], i: usize) -> Option<(usize, usize)> {
    let mut k = i;
    if k < bytes.len() && (bytes[k] == b'b' || bytes[k] == b'c') {
        k += 1;
    }
    if k >= bytes.len() || bytes[k] != b'r' {
        return None;
    }
    k += 1;
    let mut hashes = 0usize;
    while k < bytes.len() && bytes[k] == b'#' {
        hashes += 1;
        k += 1;
    }
    if k < bytes.len() && bytes[k] == b'"' {
        Some((k + 1, hashes))
    } else {
        None
    }
}

/// Detect `#[cfg(test)]` or `#[cfg(all(..., test, ...))]` (returning the
/// token index immediately after `]`). Does not match `#[cfg(any(test, ...))]`
/// or `#[cfg(not(test))]`, which are compiled in production builds.
fn is_cfg_test_attr(tokens: &[Tok], idx: usize) -> Option<usize> {
    if idx + 4 >= tokens.len() || tokens[idx].text != "#" || tokens[idx + 1].text != "[" {
        return None;
    }
    let mut depth = 0usize;
    let mut j = idx + 1;
    let mut inner = Vec::new();
    while j < tokens.len() {
        let t = tokens[j].text.as_str();
        if t == "[" {
            depth += 1;
        } else if t == "]" {
            depth -= 1;
            if depth == 0 {
                break;
            }
        } else {
            inner.push(t);
        }
        j += 1;
    }
    if j >= tokens.len() {
        return None;
    }
    if inner == ["cfg", "(", "test", ")"] {
        return Some(j + 1);
    }
    if inner.len() >= 6 && inner[..4] == ["cfg", "(", "all", "("] {
        // Check if `test` is a direct argument inside `all(...)`.
        let mut p_depth = 0usize;
        for (pos, tok) in inner.iter().enumerate() {
            match *tok {
                "(" => p_depth += 1,
                ")" => p_depth = p_depth.saturating_sub(1),
                "test" if p_depth == 2 => {
                    let prev_ok = inner.get(pos - 1).is_some_and(|p| matches!(*p, "(" | ","));
                    let next_ok = inner.get(pos + 1).is_some_and(|n| matches!(*n, ")" | ","));
                    if prev_ok && next_ok {
                        return Some(j + 1);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

fn parse_following_attrs(tokens: &[Tok], mut idx: usize) -> (usize, Option<String>) {
    let mut path_attr = None;
    while idx + 1 < tokens.len() && tokens[idx].text == "#" && tokens[idx + 1].text == "[" {
        let mut depth = 0usize;
        let mut j = idx + 1;
        let mut inner = Vec::new();
        while j < tokens.len() {
            let t = tokens[j].text.as_str();
            if t == "[" {
                depth += 1;
            } else if t == "]" {
                depth -= 1;
                if depth == 0 {
                    j += 1;
                    break;
                }
            } else {
                inner.push(&tokens[j]);
            }
            j += 1;
        }
        if inner.len() == 3
            && inner[0].text == "path"
            && inner[1].text == "="
            && inner[2].kind == TokKind::Str
        {
            path_attr = Some(inner[2].text.clone());
        }
        idx = j;
    }
    (idx, path_attr)
}

fn skip_item_after_cfg_test(tokens: &[Tok], after_attr: usize) -> usize {
    let (cur, _) = parse_following_attrs(tokens, after_attr);
    if cur >= tokens.len() {
        return cur;
    }
    let is_let = tokens[cur].text == "let";
    let mut is_decl = false;
    for tok in tokens.iter().skip(cur).take(20) {
        let t = tok.text.as_str();
        if matches!(
            t,
            "fn" | "mod" | "impl" | "trait" | "enum" | "struct" | "macro_rules"
        ) {
            is_decl = true;
            break;
        }
        if matches!(t, "=" | ":" | ";" | "{" | ",") {
            break;
        }
    }
    let (mut dp, mut db, mut dc) = (0usize, 0usize, 0usize);
    let mut j = cur;
    while j < tokens.len() {
        match tokens[j].text.as_str() {
            "(" => dp += 1,
            ")" => dp = dp.saturating_sub(1),
            "[" => db += 1,
            "]" => db = db.saturating_sub(1),
            "{" => dc += 1,
            "}" => {
                if dc == 0 && dp == 0 && db == 0 {
                    return j;
                }
                dc -= 1;
                if dc == 0 && dp == 0 && db == 0 && is_decl {
                    return j + 1;
                }
            }
            ";" if dc == 0 && dp == 0 && db == 0 => return j + 1,
            "," if dc == 0 && dp == 0 && db == 0 && !is_let && !is_decl => return j + 1,
            _ => {}
        }
        j += 1;
    }
    j
}

fn collect_src_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut sorted: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    sorted.sort_by_key(|e| e.path());
    for entry in sorted {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "tests" | "benches" | "examples") {
                continue;
            }
            collect_src_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn resolve_mod_file(host_file: &Path, mod_name: &str, path_attr: Option<&str>) -> Option<PathBuf> {
    let parent = host_file.parent()?;
    let stem = host_file.file_stem()?.to_str()?;
    let search_dir = if matches!(stem, "lib" | "main" | "mod") {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    let candidates = if let Some(rel) = path_attr {
        vec![parent.join(rel), search_dir.join(rel)]
    } else {
        vec![
            search_dir.join(format!("{mod_name}.rs")),
            search_dir.join(mod_name).join("mod.rs"),
        ]
    };
    candidates
        .into_iter()
        .find(|p| p.exists())
        .and_then(|p| fs::canonicalize(p).ok())
}

fn workspace_src_files(manifest_dir: &Path) -> Vec<(PathBuf, String, Vec<Tok>)> {
    let mut src_roots = vec![manifest_dir.join("src")];
    if let Ok(entries) = fs::read_dir(manifest_dir.join("crates")) {
        let mut crate_dirs: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        crate_dirs.sort();
        for dir in crate_dirs {
            let src = dir.join("src");
            if src.is_dir() {
                src_roots.push(src);
            }
        }
    }

    let mut all_files = Vec::new();
    for root in &src_roots {
        collect_src_rs_files(root, &mut all_files);
    }

    let mut test_only = BTreeSet::new();
    for file in &all_files {
        let content = fs::read_to_string(file).expect("read rs file");
        let tokens = tokenize_rust(&content);
        let mut i = 0;
        let mut in_cfg_test_depth: Option<usize> = None;
        let mut brace_depth = 0usize;
        while i < tokens.len() {
            if let Some(after) = is_cfg_test_attr(&tokens, i) {
                let (k, path_attr) = parse_following_attrs(&tokens, after);
                if k + 2 < tokens.len()
                    && tokens[k].text == "mod"
                    && tokens[k + 1].kind == TokKind::Ident
                    && tokens[k + 2].text == ";"
                {
                    if let Some(target) =
                        resolve_mod_file(file, &tokens[k + 1].text, path_attr.as_deref())
                    {
                        test_only.insert(target);
                    }
                    i = k + 3;
                    continue;
                }
                if k + 2 < tokens.len()
                    && tokens[k].text == "mod"
                    && tokens[k + 1].kind == TokKind::Ident
                    && tokens[k + 2].text == "{"
                    && in_cfg_test_depth.is_none()
                {
                    in_cfg_test_depth = Some(brace_depth);
                    brace_depth += 1;
                    i = k + 3;
                    continue;
                }
            }
            if in_cfg_test_depth.is_some() {
                let (k, path_attr) = parse_following_attrs(&tokens, i);
                if k > i
                    && k + 2 < tokens.len()
                    && tokens[k].text == "mod"
                    && tokens[k + 1].kind == TokKind::Ident
                    && tokens[k + 2].text == ";"
                {
                    if let Some(target) =
                        resolve_mod_file(file, &tokens[k + 1].text, path_attr.as_deref())
                    {
                        test_only.insert(target);
                    }
                } else if tokens[i].text == "mod"
                    && i + 2 < tokens.len()
                    && tokens[i + 1].kind == TokKind::Ident
                    && tokens[i + 2].text == ";"
                    && let Some(target) = resolve_mod_file(file, &tokens[i + 1].text, None)
                {
                    test_only.insert(target);
                }
            }
            if tokens[i].text == "{" {
                brace_depth += 1;
            } else if tokens[i].text == "}" {
                brace_depth = brace_depth.saturating_sub(1);
                if in_cfg_test_depth == Some(brace_depth) {
                    in_cfg_test_depth = None;
                }
            }
            i += 1;
        }
    }

    let mut queue: VecDeque<PathBuf> = test_only.iter().cloned().collect();
    while let Some(file) = queue.pop_front() {
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        let tokens = tokenize_rust(&content);
        let mut i = 0;
        while i < tokens.len() {
            let (k, path_attr) = parse_following_attrs(&tokens, i);
            let check_at = if k > i { k } else { i };
            if check_at + 2 < tokens.len()
                && tokens[check_at].text == "mod"
                && tokens[check_at + 1].kind == TokKind::Ident
                && tokens[check_at + 2].text == ";"
            {
                if let Some(target) =
                    resolve_mod_file(&file, &tokens[check_at + 1].text, path_attr.as_deref())
                    && test_only.insert(target.clone())
                {
                    queue.push_back(target);
                }
                i = check_at + 3;
                continue;
            }
            i += 1;
        }
    }

    let mut prod_files = Vec::new();
    for file in all_files {
        let canon = fs::canonicalize(&file).expect("canonical path");
        if test_only.contains(&canon) {
            continue;
        }
        let content = fs::read_to_string(&file).expect("read rs file");
        let raw_tokens = tokenize_rust(&content);
        let mut filtered = Vec::new();
        let mut i = 0;
        while i < raw_tokens.len() {
            if let Some(after) = is_cfg_test_attr(&raw_tokens, i) {
                i = skip_item_after_cfg_test(&raw_tokens, after);
                continue;
            }
            filtered.push(raw_tokens[i].clone());
            i += 1;
        }
        let rel = file
            .strip_prefix(manifest_dir)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        prod_files.push((file, rel, filtered));
    }
    prod_files
}

fn has_crate_forbid_unsafe_code(tokens: &[Tok], allow_not_test_cfg: bool) -> bool {
    let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
    let unconditional = ["#", "!", "[", "forbid", "(", "unsafe_code", ")", "]"];
    let test_scoped = [
        "#",
        "!",
        "[",
        "cfg_attr",
        "(",
        "not",
        "(",
        "test",
        ")",
        ",",
        "forbid",
        "(",
        "unsafe_code",
        ")",
        ")",
        "]",
    ];
    if texts
        .windows(unconditional.len())
        .any(|w| w == unconditional)
    {
        return true;
    }
    allow_not_test_cfg && texts.windows(test_scoped.len()).any(|w| w == test_scoped)
}

#[test]
fn every_crate_and_binary_root_declares_forbid_unsafe_code() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root_toml = manifest_dir.join("Cargo.toml");
    let root_content = fs::read_to_string(&root_toml).expect("read root Cargo.toml");
    let root_manifest: Table = toml::from_str(&root_content).expect("parse root Cargo.toml");

    let mut targets: Vec<(String, PathBuf, bool)> = Vec::new();
    let mut pkg_dirs = vec![("oxllm".to_string(), manifest_dir.to_path_buf())];
    for member in root_manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .expect("workspace.members")
    {
        let rel = member.as_str().expect("member string");
        let dir = manifest_dir.join(rel);
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(rel)
            .to_string();
        pkg_dirs.push((name, dir));
    }

    for (pkg_name, dir) in pkg_dirs {
        let cargo_path = dir.join("Cargo.toml");
        let content = fs::read_to_string(&cargo_path).expect("read crate Cargo.toml");
        let manifest: Table = toml::from_str(&content).expect("parse crate Cargo.toml");
        let allow_cfg_attr = matches!(pkg_name.as_str(), "oxllm-server" | "oxllm-admin");

        if let Some(lib_path) = manifest
            .get("lib")
            .and_then(|l| l.get("path"))
            .and_then(|p| p.as_str())
        {
            targets.push((pkg_name.clone(), dir.join(lib_path), allow_cfg_attr));
        } else if dir.join("src/lib.rs").exists() {
            targets.push((pkg_name.clone(), dir.join("src/lib.rs"), allow_cfg_attr));
        }

        if let Some(bins) = manifest.get("bin").and_then(|b| b.as_array()) {
            for bin in bins {
                let name = bin
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or(pkg_name.as_str());
                let path = bin.get("path").and_then(|p| p.as_str()).expect("bin.path");
                // `oxllm` binary unit tests mutate env vars (`unsafe` in edition 2024).
                let bin_allows_cfg = name == "oxllm" || allow_cfg_attr;
                targets.push((name.to_string(), dir.join(path), bin_allows_cfg));
            }
        } else if dir.join("src/main.rs").exists() {
            targets.push((pkg_name.clone(), dir.join("src/main.rs"), allow_cfg_attr));
        }
    }

    assert!(
        targets.len() >= 12,
        "expected all 9 crate libs + root lib + 2 binaries, found {}",
        targets.len()
    );

    for (name, path, allow_cfg_attr) in targets {
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read target {name} ({}): {e}", path.display()));
        let tokens = tokenize_rust(&src);
        assert!(
            has_crate_forbid_unsafe_code(&tokens, allow_cfg_attr),
            "UNSAFE POLICY VIOLATION: target '{name}' ({}) must declare #![forbid(unsafe_code)]{}",
            path.display(),
            if allow_cfg_attr {
                " or #![cfg_attr(not(test), forbid(unsafe_code))]"
            } else {
                ""
            }
        );
    }
}

#[test]
fn no_unsafe_blocks_in_production_source() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        for tok in tokens {
            assert!(
                !(tok.kind == TokKind::Ident && tok.text == "unsafe"),
                "UNSAFE VIOLATION in production source at {rel}:{}",
                tok.line
            );
        }
    }
}

#[test]
fn oxllm_providers_has_no_async_fn() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        if !rel.starts_with("crates/oxllm-providers/") {
            continue;
        }
        for (idx, tok) in tokens.iter().enumerate() {
            assert!(
                !(tok.kind == TokKind::Ident
                    && tok.text == "async"
                    && tokens
                        .get(idx + 1)
                        .is_some_and(|n| n.kind == TokKind::Ident && n.text == "fn")),
                "ASYNC PROVIDER VIOLATION: 'async fn' in {rel}:{}",
                tok.line
            );
        }
    }
}

#[test]
fn no_todo_unimplemented_or_dbg_in_production_code() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        for (idx, tok) in tokens.iter().enumerate() {
            if tok.kind == TokKind::Ident
                && matches!(tok.text.as_str(), "todo" | "unimplemented" | "dbg")
                && tokens.get(idx + 1).is_some_and(|n| n.text == "!")
            {
                panic!(
                    "PLACEHOLDER/DEBUG MACRO VIOLATION: `{}!` in production source at {rel}:{}",
                    tok.text, tok.line
                );
            }
        }
    }
}

#[test]
fn no_stdout_or_stderr_macros_in_library_crates() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        if !rel.starts_with("crates/") {
            continue;
        }
        for (idx, tok) in tokens.iter().enumerate() {
            if tok.kind == TokKind::Ident
                && matches!(
                    tok.text.as_str(),
                    "println" | "eprintln" | "print" | "eprint"
                )
                && tokens.get(idx + 1).is_some_and(|n| n.text == "!")
            {
                panic!(
                    "STDOUT/STDERR VIOLATION: `{}!` in library crate at {rel}:{}",
                    tok.text, tok.line
                );
            }
        }
    }
}

/// Prevent any new `.unwrap()`, `.expect()`, or `panic!()` in non-test
/// production code while bounding known compile-time-invariant or audited sites
/// by per-file ceilings (never brittle line numbers).
///
/// Ceilings are a ratchet: every allowlisted file must match its ceiling
/// exactly, so fixing a site (e.g. the former `orchestration/upstream.rs`
/// `.unwrap()` or the `repos/skills.rs` poisoned-lock `.expect()`s) forces the
/// ceiling down in the same change and the fix can never silently regress.
#[test]
fn no_unallowlisted_unwrap_expect_or_panic_in_production_code() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    let unwrap_ceilings: HashMap<&str, usize> =
        HashMap::from([("crates/oxllm-server/src/guardrails/injection.rs", 1)]);

    let panic_ceilings: HashMap<&str, usize> = HashMap::from([
        ("crates/oxllm-core/src/cost.rs", 1),
        ("crates/oxllm-admin/src/templates.rs", 1),
    ]);

    let expect_ceilings: HashMap<&str, usize> = HashMap::from([
        ("crates/oxllm-core/src/cost.rs", 1),
        ("crates/oxllm-core/src/cost/service_tiers.rs", 2),
        ("crates/oxllm-core/src/router/stores.rs", 1),
        ("crates/oxllm-providers/src/azure_ai/embeddings.rs", 3),
        ("crates/oxllm-providers/src/databricks/mod.rs", 2),
        ("crates/oxllm-admin/src/proxy_client.rs", 1),
        ("crates/oxllm-server/src/app.rs", 1),
        ("crates/oxllm-server/src/auth/jwt.rs", 1),
        ("crates/oxllm-server/src/config/loader.rs", 2),
        ("crates/oxllm-server/src/guardrails/injection.rs", 1),
        ("crates/oxllm-server/src/guardrails/mod.rs", 5),
        (
            "crates/oxllm-server/src/handlers/management/models_mgmt.rs",
            1,
        ),
        ("crates/oxllm-server/src/hooks/redact.rs", 5),
        (
            "crates/oxllm-server/src/pre_call_checks/prompt_caching.rs",
            1,
        ),
        ("crates/oxllm-server/src/realtime/xai.rs", 1),
        ("crates/oxllm-server/src/routes/management.rs", 1),
    ]);

    let mut seen_files: BTreeSet<String> = BTreeSet::new();
    let mut stale: Vec<String> = Vec::new();

    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        let mut unwrap_lines = Vec::new();
        let mut expect_lines = Vec::new();
        let mut panic_lines = Vec::new();

        for (idx, tok) in tokens.iter().enumerate() {
            if tok.kind != TokKind::Ident {
                continue;
            }
            if tok.text == "unwrap"
                && idx > 0
                && tokens[idx - 1].text == "."
                && tokens.get(idx + 1).is_some_and(|n| n.text == "(")
            {
                unwrap_lines.push(tok.line);
            } else if tok.text == "expect"
                && idx > 0
                && tokens[idx - 1].text == "."
                && tokens.get(idx + 1).is_some_and(|n| n.text == "(")
            {
                expect_lines.push(tok.line);
            } else if tok.text == "panic" && tokens.get(idx + 1).is_some_and(|n| n.text == "!") {
                panic_lines.push(tok.line);
            }
        }

        for (kind, lines, ceilings) in [
            ("unwrap", &unwrap_lines, &unwrap_ceilings),
            ("expect", &expect_lines, &expect_ceilings),
            ("panic", &panic_lines, &panic_ceilings),
        ] {
            let max = ceilings.get(rel.as_str()).copied().unwrap_or(0);
            assert!(
                lines.len() <= max,
                "PANIC RISK: `{rel}` has {} `{kind}` calls (ceiling {max}) at lines {lines:?}",
                lines.len()
            );
            if ceilings.contains_key(rel.as_str()) && lines.len() < max {
                stale.push(format!(
                    "`{rel}` `{kind}` ceiling is {max} but only {} remain (lines {lines:?}); lower the ceiling",
                    lines.len()
                ));
            }
        }
        seen_files.insert(rel);
    }

    for ceilings in [&unwrap_ceilings, &expect_ceilings, &panic_ceilings] {
        for path in ceilings.keys() {
            if !seen_files.contains(*path) {
                stale.push(format!(
                    "`{path}` is allowlisted but is not a production source file"
                ));
            }
        }
    }
    assert!(
        stale.is_empty(),
        "STALE PANIC CEILINGS (ratchet must be tightened):\n{}",
        stale.join("\n")
    );
}

#[test]
fn os_environ_literal_restricted_to_config_loaders() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let allowed = [
        "crates/oxllm-server/src/config/loader.rs",
        "crates/oxllm-admin/src/config.rs",
    ];
    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        if allowed.contains(&rel.as_str()) {
            continue;
        }
        for tok in tokens {
            if tok.kind == TokKind::Str && tok.text == "os.environ/" {
                panic!(
                    "CONFIG PURITY VIOLATION: `os.environ/` resolution prefix outside config loader at {rel}:{}",
                    tok.line
                );
            }
        }
    }
}

#[test]
fn no_std_env_var_on_core_providers_or_request_hot_path() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let hot_path_prefixes = [
        "crates/oxllm-core/",
        "crates/oxllm-providers/",
        "crates/oxllm-db/",
        "crates/oxllm-cache-",
        "crates/oxllm-server/src/orchestration/",
        "crates/oxllm-server/src/pipeline/",
        "crates/oxllm-server/src/hooks/",
        "crates/oxllm-server/src/auth/",
        "crates/oxllm-server/src/routes/",
        "crates/oxllm-server/src/handlers/chat.rs",
        "crates/oxllm-server/src/handlers/completions.rs",
        "crates/oxllm-server/src/handlers/embeddings",
        "crates/oxllm-server/src/handlers/messages.rs",
        "crates/oxllm-server/src/handlers/responses.rs",
        "crates/oxllm-server/src/handlers/passthrough.rs",
        "crates/oxllm-server/src/handlers/models.rs",
        "crates/oxllm-server/src/handlers/health.rs",
        "crates/oxllm-server/src/handlers/token_counter.rs",
    ];

    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        if !hot_path_prefixes.iter().any(|p| rel.starts_with(p)) {
            continue;
        }
        for (idx, tok) in tokens.iter().enumerate() {
            if tok.kind == TokKind::Ident
                && tok.text == "env"
                && tokens.get(idx + 1).is_some_and(|n| n.text == "::")
                && tokens.get(idx + 2).is_some_and(|n| {
                    matches!(n.text.as_str(), "var" | "var_os" | "vars" | "vars_os")
                })
            {
                panic!(
                    "HOT-PATH ENV READ VIOLATION: `env::{}` at {rel}:{}",
                    tokens[idx + 2].text,
                    tok.line
                );
            }
        }
    }
}

#[test]
fn coordinator_used_only_for_auth_brake_and_state_wiring() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let allowed_files = [
        "crates/oxllm-server/src/coordination.rs",
        "crates/oxllm-server/src/app.rs",
        "crates/oxllm-server/src/security.rs",
        "crates/oxllm-server/src/state/mod.rs",
        "crates/oxllm-server/src/state/build.rs",
    ];

    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        if allowed_files.contains(&rel.as_str()) {
            continue;
        }
        for tok in tokens {
            assert!(
                !(tok.kind == TokKind::Ident && tok.text == "Coordinator"),
                "COORDINATOR SCOPE VIOLATION: `Coordinator` referenced outside AuthBrake/state wiring at {rel}:{}",
                tok.line
            );
        }
    }
}

#[test]
fn spend_counter_keys_defined_only_in_spend_keys_module() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        if rel == "crates/oxllm-server/src/hooks/spend_keys.rs" {
            continue;
        }
        for (idx, tok) in tokens.iter().enumerate() {
            if tok.kind == TokKind::Ident
                && tok.text == "format"
                && tokens.get(idx + 1).is_some_and(|n| n.text == "!")
                && tokens.get(idx + 2).is_some_and(|n| n.text == "(")
                && tokens
                    .get(idx + 3)
                    .is_some_and(|n| n.kind == TokKind::Str && n.text.starts_with("spend:"))
            {
                panic!(
                    "SPEND KEY FORMAT VIOLATION: inline `format!(\"spend:...\")` outside hooks/spend_keys.rs at {rel}:{}",
                    tok.line
                );
            }
        }
    }
}

#[test]
fn hook_chain_source_preserves_canonical_order() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let hook_builders =
        fs::read_to_string(manifest_dir.join("crates/oxllm-server/src/state/hook_builders.rs"))
            .expect("read hook_builders.rs");
    let tokens = tokenize_rust(&hook_builders);

    let build_hooks_pos = tokens
        .windows(2)
        .position(|w| w[0].text == "fn" && w[1].text == "build_hooks")
        .expect("build_hooks must exist");

    let ordered_hooks = [
        "RateLimitHook",
        "CacheHook",
        "BudgetHook",
        "CostTrackingHook",
        "MetricsHook",
    ];

    let mut search_from = build_hooks_pos;
    for hook_name in ordered_hooks {
        let found = tokens[search_from..]
            .iter()
            .position(|t| t.kind == TokKind::Ident && t.text == hook_name)
            .unwrap_or_else(|| {
                panic!(
                    "HOOK ORDERING VIOLATION: `{hook_name}` missing or out of order in build_hooks"
                )
            });
        search_from += found + 1;
    }
}

#[tokio::test]
async fn hook_chain_runtime_enforces_ratelimit_before_cache_before_budget() {
    let config = ProxyConfig {
        model_list: vec![],
        oxllm_settings: OxllmSettings {
            rpm: Some(1),
            max_budget: Some(100.0),
            ..Default::default()
        },
        general_settings: GeneralSettings::default(),
        router_settings: RouterSettings::default(),
        extra: serde_json::Map::new(),
    };
    let state = AppState::new(config).expect("AppState::new");

    let request_body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "cache": {"insert": true},
    });
    let api_key_hash = "keyhash-arch-test".to_string();
    let key = cache_key_for(&request_body, &api_key_hash);
    state
        .cache
        .set(
            &key,
            json!({"id": "cached-1", "choices": []}).to_string(),
            Some(std::time::Duration::from_secs(60)),
        )
        .await;

    let mut ctx1 = CallContext {
        model: "gpt-4o".to_string(),
        api_key_hash: api_key_hash.clone(),
        request_id: "req-1".to_string(),
        request_body: Arc::new(request_body.clone()),
        ..Default::default()
    };
    let hit = run_pre_call_hooks(&state.hooks, &mut ctx1)
        .await
        .expect("first call must short-circuit via cache");
    assert!(hit.is_some(), "CacheHook must short-circuit on cache hit");
    assert_eq!(
        state
            .cache
            .counter_value(&format!("rpm:{api_key_hash}"))
            .await,
        Some(1.0),
        "RateLimitHook must run before CacheHook so cache hits still consume RPM"
    );
    assert_eq!(
        state
            .cache
            .counter_value(&format!("spend:{api_key_hash}"))
            .await,
        None,
        "CacheHook must run before BudgetHook so cache hits never reserve budget"
    );

    let mut ctx2 = CallContext {
        model: "gpt-4o".to_string(),
        api_key_hash,
        request_id: "req-2".to_string(),
        request_body: Arc::new(request_body),
        ..Default::default()
    };
    let rejected = run_pre_call_hooks(&state.hooks, &mut ctx2).await;
    assert!(
        rejected.is_err(),
        "RateLimitHook must reject once RPM=1 is exhausted even when response is cached"
    );
}

#[test]
fn hot_reload_preserves_infrastructure_and_unchanged_components() {
    let mk_config = |jwt_secret: &str| ProxyConfig {
        model_list: vec![],
        oxllm_settings: OxllmSettings {
            slack_alerting_config: Some(SlackAlertConfig {
                webhook_url: "https://hooks.example.invalid/slack".to_string(),
                thresholds: vec![0.8],
                ..Default::default()
            }),
            cache: Some(CacheConfig {
                semantic_cache: Some(SemanticCacheConfig {
                    enabled: true,
                    embedding_model: "text-embedding-3-small".to_string(),
                    api_base: Some("https://api.openai.com/v1".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            jwt_secret: Some(jwt_secret.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: RouterSettings::default(),
        extra: serde_json::Map::new(),
    };

    let initial =
        AppState::new(mk_config("initial-jwt-secret-long-enough-32")).expect("initial AppState");

    let reloaded_same = initial
        .reload_from_config(mk_config("initial-jwt-secret-long-enough-32"))
        .expect("reload with unchanged config");

    assert!(Arc::ptr_eq(&initial.cache, &reloaded_same.cache));
    assert!(Arc::ptr_eq(&initial.shutdown, &reloaded_same.shutdown));
    assert!(Arc::ptr_eq(
        &initial.router.cooldown_arc(),
        &reloaded_same.router.cooldown_arc()
    ));
    assert!(Arc::ptr_eq(
        &initial.router.bandit_store_arc(),
        &reloaded_same.router.bandit_store_arc()
    ));
    assert!(Arc::ptr_eq(
        &initial.router.round_robin_counters_arc(),
        &reloaded_same.router.round_robin_counters_arc()
    ));
    assert!(Arc::ptr_eq(
        initial.jwt_validator.as_ref().expect("jwt"),
        reloaded_same.jwt_validator.as_ref().expect("jwt")
    ));
    assert!(Arc::ptr_eq(
        initial.auth_brake.as_ref().expect("auth_brake"),
        reloaded_same.auth_brake.as_ref().expect("auth_brake")
    ));
    assert!(Arc::ptr_eq(
        initial.slack_alert_state.as_ref().expect("slack"),
        reloaded_same.slack_alert_state.as_ref().expect("slack")
    ));
    assert!(Arc::ptr_eq(
        initial.semantic_cache_runtime.as_ref().expect("semantic"),
        reloaded_same
            .semantic_cache_runtime
            .as_ref()
            .expect("semantic")
    ));
    assert!(Arc::ptr_eq(
        &initial.credential_resolver,
        &reloaded_same.credential_resolver
    ));

    let reloaded_jwt_changed = reloaded_same
        .reload_from_config(mk_config("rotated-jwt-secret-long-enough-32"))
        .expect("reload with rotated jwt_secret");
    assert!(!Arc::ptr_eq(
        reloaded_same.jwt_validator.as_ref().expect("jwt"),
        reloaded_jwt_changed.jwt_validator.as_ref().expect("jwt")
    ));
    assert!(Arc::ptr_eq(
        &reloaded_same.cache,
        &reloaded_jwt_changed.cache
    ));
    assert!(Arc::ptr_eq(
        reloaded_same.auth_brake.as_ref().expect("auth_brake"),
        reloaded_jwt_changed
            .auth_brake
            .as_ref()
            .expect("auth_brake")
    ));
}

#[test]
fn reload_unchanged_helpers_are_wired_into_assemble() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let reload_src =
        fs::read_to_string(manifest_dir.join("crates/oxllm-server/src/state/reload.rs"))
            .expect("read reload.rs");
    let build_src = fs::read_to_string(manifest_dir.join("crates/oxllm-server/src/state/build.rs"))
        .expect("read build.rs");

    let reload_tokens = tokenize_rust(&reload_src);
    let build_tokens = tokenize_rust(&build_src);
    let build_idents: BTreeSet<&str> = build_tokens
        .iter()
        .filter(|t| t.kind == TokKind::Ident)
        .map(|t| t.text.as_str())
        .collect();

    let mut helpers = Vec::new();
    for i in 0..reload_tokens.len().saturating_sub(1) {
        if reload_tokens[i].text == "fn"
            && reload_tokens[i + 1].kind == TokKind::Ident
            && reload_tokens[i + 1].text.ends_with("_config_unchanged")
        {
            helpers.push(reload_tokens[i + 1].text.as_str());
        }
    }
    assert!(
        !helpers.is_empty(),
        "reload.rs must define *_config_unchanged preservation helpers"
    );
    for helper in helpers {
        assert!(
            build_idents.contains(helper),
            "RELOAD DRIFT: `reload::{helper}` is defined in state/reload.rs but never called in state/build.rs"
        );
    }
}

#[test]
fn handlers_do_not_bypass_orchestration_with_sqlx_or_client_construction() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        if !rel.starts_with("crates/oxllm-server/src/handlers/") {
            continue;
        }
        for (idx, tok) in tokens.iter().enumerate() {
            if tok.kind != TokKind::Ident {
                continue;
            }
            if rel != "crates/oxllm-server/src/handlers/management/mod.rs"
                && tok.text == "sqlx"
                && tokens.get(idx + 1).is_some_and(|n| n.text == "::")
            {
                panic!(
                    "HANDLER LAYERING VIOLATION: direct `sqlx::` usage in {rel}:{}",
                    tok.line
                );
            }
            if tok.text == "reqwest"
                && tokens.get(idx + 1).is_some_and(|n| n.text == "::")
                && tokens.get(idx + 2).is_some_and(|n| n.text == "Client")
                && tokens.get(idx + 3).is_some_and(|n| n.text == "::")
                && tokens
                    .get(idx + 4)
                    .is_some_and(|n| matches!(n.text.as_str(), "new" | "builder"))
            {
                panic!(
                    "HANDLER HTTP CLIENT VIOLATION: per-handler `reqwest::Client::{}` in {rel}:{}",
                    tokens[idx + 4].text,
                    tok.line
                );
            }
        }
    }
}

#[tokio::test]
async fn secret_bearing_types_redact_debug_output() {
    const SENTINEL: &str = "SUPER_SECRET_SENTINEL_VALUE_987654";

    let params = OxllmParams {
        model: "openai/gpt-4o".to_string(),
        api_key: Some(SENTINEL.to_string()),
        aws_access_key_id: Some(SENTINEL.to_string()),
        aws_secret_access_key: Some(SENTINEL.to_string()),
        aws_session_token: Some(SENTINEL.to_string()),
        api_base: Some(format!("https://{SENTINEL}.example.com")),
        extra_headers: Some(HashMap::from([(
            "x-api-key".to_string(),
            SENTINEL.to_string(),
        )])),
        extra: serde_json::Map::from_iter([("custom_token".to_string(), json!(SENTINEL))]),
        ..Default::default()
    };
    let dep = Deployment {
        model_name: "gpt-4o".to_string(),
        oxllm_params: params.clone(),
        model_info: None,
    };
    assert!(!format!("{params:?}").contains(SENTINEL));
    assert!(!format!("{dep:?}").contains(SENTINEL));

    let creds = [
        ResolvedCredentials::ApiKey {
            bearer: SENTINEL.to_string(),
        },
        ResolvedCredentials::Azure {
            api_key: SENTINEL.to_string(),
        },
        ResolvedCredentials::AzureAd {
            token: SENTINEL.to_string(),
        },
        ResolvedCredentials::AwsSigV4 {
            access_key: SENTINEL.to_string(),
            secret_key: SENTINEL.to_string(),
            session_token: Some(SENTINEL.to_string()),
            region: "us-east-1".to_string(),
        },
        ResolvedCredentials::GcpBearer {
            token: SENTINEL.to_string(),
        },
        ResolvedCredentials::PreSignedHeaders(HashMap::from([(
            "authorization".to_string(),
            SENTINEL.to_string(),
        )])),
    ];
    for c in &creds {
        assert!(!format!("{c:?}").contains(SENTINEL));
    }

    let cipher = CredentialCipher::from_raw_key([0x5a; 32]);
    assert!(!format!("{cipher:?}").contains("5a"));

    let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
    let coord = Coordinator::new(cache);
    let brake = AuthBrake::default_policy().with_coordinator(Some(coord.clone()));
    assert!(!format!("{coord:?}").contains(SENTINEL));
    let _ = format!("{brake:?}");

    let langfuse = LangfuseHook::new(
        &LangfuseConfig {
            host: "https://langfuse.example.invalid".to_string(),
            public_key: SENTINEL.to_string(),
            secret_key: SENTINEL.to_string(),
            ..Default::default()
        },
        true,
        false,
    )
    .expect("LangfuseHook::new");
    assert!(!format!("{langfuse:?}").contains(SENTINEL));

    let s3_hook = S3LogHook::new(
        &S3LogConfig {
            bucket: "audit-logs".to_string(),
            region: "us-east-1".to_string(),
            access_key_id: Some(SENTINEL.to_string()),
            secret_access_key: Some(SENTINEL.to_string()),
            ..Default::default()
        },
        true,
    )
    .expect("S3LogHook::new");
    assert!(!format!("{s3_hook:?}").contains(SENTINEL));
}

/// Locks in the secret-redacting `Debug` impls on every secret-bearing config
/// and admin form type: none may go back to `#[derive(Debug)]` (which would
/// print plaintext credentials), and each must keep a hand-written
/// `impl Debug for <Type>`. Behavioral sentinel coverage lives next to the
/// types (`config/types/tests.rs`, `pages/{auth,models}.rs`).
#[test]
fn secret_bearing_config_types_never_derive_debug() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let types_dir = "crates/oxllm-server/src/config/types";
    let required: &[(&str, &[&str])] = &[
        (
            "base.rs",
            &[
                "ProxyConfig",
                "OxllmSettings",
                "GeneralSettings",
                "PassThroughEndpoint",
            ],
        ),
        (
            "cache.rs",
            &[
                "CacheConfig",
                "S3CacheConfig",
                "GcsCacheConfig",
                "AzureBlobCacheConfig",
                "SemanticCacheConfig",
            ],
        ),
        (
            "guardrails.rs",
            &[
                "AimConfig",
                "LakeraConfig",
                "ModelArmorConfig",
                "CustomGuardrailConfig",
            ],
        ),
        (
            "telemetry.rs",
            &[
                "LangfuseConfig",
                "LangsmithConfig",
                "DatadogConfig",
                "HeliconeConfig",
                "WandbConfig",
                "S3LogConfig",
            ],
        ),
        (
            "alerting.rs",
            &[
                "WebhookAlertConfig",
                "SlackAlertConfig",
                "PagerDutyConfig",
                "EmailAlertConfig",
            ],
        ),
    ];
    let mut targets: Vec<(String, &str)> = Vec::new();
    for (file, names) in required {
        for name in *names {
            targets.push((format!("{types_dir}/{file}"), name));
        }
    }
    targets.push((
        "crates/oxllm-admin/src/pages/auth.rs".to_string(),
        "LoginForm",
    ));
    targets.push((
        "crates/oxllm-admin/src/pages/models.rs".to_string(),
        "ModelCreateForm",
    ));
    assert_eq!(targets.len(), 25, "23 config types + 2 admin forms");

    for (rel, name) in &targets {
        let src = fs::read_to_string(manifest_dir.join(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"));
        let tokens = tokenize_rust(&src);

        let struct_idx = tokens
            .windows(2)
            .position(|w| w[0].text == "struct" && w[1].text == *name)
            .unwrap_or_else(|| panic!("`struct {name}` not found in {rel}"));

        // Attributes between the previous item boundary and `struct`.
        let item_start = tokens[..struct_idx]
            .iter()
            .rposition(|t| t.kind == TokKind::Punct && (t.text == "}" || t.text == ";"))
            .map_or(0, |i| i + 1);
        let attrs = &tokens[item_start..struct_idx];
        let mut derives_debug = false;
        for (i, tok) in attrs.iter().enumerate() {
            if tok.text == "derive" && attrs.get(i + 1).is_some_and(|t| t.text == "(") {
                let close = attrs[i + 1..]
                    .iter()
                    .position(|t| t.text == ")")
                    .map_or(attrs.len(), |p| i + 1 + p);
                derives_debug |= attrs[i + 1..close].iter().any(|t| t.text == "Debug");
            }
        }
        assert!(
            !derives_debug,
            "SECRET LEAK RISK: `{name}` in {rel} derives `Debug`; keep the hand-written redacting impl"
        );

        let has_manual_impl = tokens
            .windows(3)
            .any(|w| w[0].text == "Debug" && w[1].text == "for" && w[2].text == *name);
        assert!(
            has_manual_impl,
            "SECRET LEAK RISK: `{name}` in {rel} lost its redacting `impl Debug for {name}`"
        );
    }
}

#[cfg(feature = "admin")]
#[test]
fn admin_config_redacts_secrets_in_debug() {
    const SENTINEL: &str = "ADMIN_SECRET_SENTINEL_987654";
    let cfg = oxllm_admin::AdminConfig {
        listen: "127.0.0.1:8080".to_string(),
        proxy_url: "http://127.0.0.1:4000".to_string(),
        master_key: SENTINEL.to_string(),
        session_secret: SENTINEL.to_string(),
        session_ttl_hours: 24,
        login_password: Some(SENTINEL.to_string()),
        trusted_proxies: vec![],
        cookie_secure: true,
        login_max_fails: 5,
        login_window_secs: 60,
        login_block_secs: 300,
    };
    assert!(!format!("{cfg:?}").contains(SENTINEL));
}

#[test]
fn production_tracing_calls_do_not_debug_log_config_or_raw_secrets() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let forbidden_tracing_idents = [
        "api_key",
        "master_key",
        "jwt_secret",
        "secret_key",
        "secret_access_key",
        "session_secret",
        "login_password",
        "raw_key",
        "encryption_key",
        "private_key",
    ];

    for (_, rel, tokens) in workspace_src_files(manifest_dir) {
        let mut i = 0;
        while i + 3 < tokens.len() {
            if tokens[i].text == "tracing"
                && tokens[i + 1].text == "::"
                && matches!(
                    tokens[i + 2].text.as_str(),
                    "trace" | "debug" | "info" | "warn" | "error"
                )
                && tokens[i + 3].text == "!"
                && tokens.get(i + 4).is_some_and(|t| t.text == "(")
            {
                let mut depth = 0usize;
                let mut j = i + 4;
                let mut end = tokens.len();
                while j < tokens.len() {
                    if tokens[j].text == "(" {
                        depth += 1;
                    } else if tokens[j].text == ")" {
                        depth -= 1;
                        if depth == 0 {
                            end = j;
                            break;
                        }
                    }
                    j += 1;
                }
                for (k, tok) in tokens[i + 4..end].iter().enumerate() {
                    if tok.kind == TokKind::Ident
                        && forbidden_tracing_idents.contains(&tok.text.as_str())
                    {
                        panic!(
                            "SECRET LOGGING RISK: tracing call references `{}` at {rel}:{}",
                            tok.text, tok.line
                        );
                    }
                    if tok.text == "?"
                        && tokens[i + 4..end].get(k + 1).is_some_and(|n| {
                            matches!(
                                n.text.as_str(),
                                "config" | "general_settings" | "oxllm_settings"
                            )
                        })
                        && tokens[i + 4..end].get(k + 2).is_none_or(|n| n.text != ".")
                    {
                        panic!(
                            "SECRET LOGGING RISK: tracing call debug-logs `{}` at {rel}:{}",
                            tokens[i + 4 + k + 1].text,
                            tok.line
                        );
                    }
                }
                i = end + 1;
                continue;
            }
            i += 1;
        }
    }
}
