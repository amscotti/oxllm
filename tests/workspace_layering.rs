//! Workspace crate dependency layering (DAG) and purity checks.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use toml::Table;

/// The allowed internal dependencies for each crate.
///
/// Rules:
/// - oxllm-core: no internal deps
/// - oxllm-providers: only oxllm-core
/// - oxllm-cache-redis: only oxllm-core
/// - oxllm-cache-s3: only oxllm-core
/// - oxllm-cache-gcs: only oxllm-core
/// - oxllm-cache-azure: only oxllm-core
/// - oxllm-db: no internal deps (persistence is self-contained)
/// - oxllm-server: oxllm-core, oxllm-providers, oxllm-cache-redis, oxllm-cache-s3,
///   oxllm-cache-gcs, oxllm-cache-azure, oxllm-db
/// - oxllm-admin: no internal deps (remote HTTP client of the proxy only)
/// - oxllm (root binary): production deps allow only oxllm-server and
///   oxllm-admin (optional admin UI binary); dev-dependencies additionally allow
///   oxllm-core, oxllm-providers, and oxllm-cache-redis for workspace-level
///   integration tests (they never ship in the binary)
fn allowed_internal_deps() -> HashMap<&'static str, Vec<&'static str>> {
    let mut m = HashMap::new();
    m.insert("oxllm-core", vec![]);
    m.insert("oxllm-providers", vec!["oxllm-core"]);
    m.insert("oxllm-cache-redis", vec!["oxllm-core"]);
    m.insert("oxllm-cache-s3", vec!["oxllm-core"]);
    m.insert("oxllm-cache-gcs", vec!["oxllm-core"]);
    m.insert("oxllm-cache-azure", vec!["oxllm-core"]);
    // oxllm-db does not depend on oxllm-core (persistence is self-contained).
    m.insert("oxllm-db", vec![]);
    m.insert(
        "oxllm-server",
        vec![
            "oxllm-core",
            "oxllm-providers",
            "oxllm-cache-redis",
            "oxllm-cache-s3",
            "oxllm-cache-gcs",
            "oxllm-cache-azure",
            "oxllm-db",
        ],
    );
    m.insert("oxllm-admin", vec![]);
    m.insert(
        "oxllm",
        vec![
            "oxllm-server",
            "oxllm-admin",
            "oxllm-core",
            "oxllm-providers",
            "oxllm-cache-redis",
        ],
    );
    m
}

/// Extract dependency keys from selected TOML sections (including matching
/// `[target.*.section]` tables).
fn extract_deps_from_sections(
    toml_content: &str,
    sections: &[&str],
    internal_only: bool,
) -> Vec<String> {
    let manifest: Table = match toml::from_str(toml_content) {
        Ok(m) => m,
        Err(e) => panic!("failed to parse Cargo.toml as TOML: {e}"),
    };

    let mut deps = Vec::new();

    for key in sections {
        if let Some(toml::Value::Table(table)) = manifest.get(*key) {
            collect_dep_keys(table, &mut deps, internal_only);
        }
    }

    if let Some(toml::Value::Table(targets)) = manifest.get("target") {
        for value in targets.values() {
            if let toml::Value::Table(target_table) = value {
                for key in sections {
                    if let Some(toml::Value::Table(dep_table)) = target_table.get(*key) {
                        collect_dep_keys(dep_table, &mut deps, internal_only);
                    }
                }
            }
        }
    }

    deps.sort();
    deps.dedup();
    deps
}

/// Extract every `oxllm*` dependency declared anywhere in the manifest,
/// scanning all dependency tables (including target-specific ones).
fn extract_internal_deps(toml_content: &str) -> Vec<String> {
    extract_deps_from_sections(
        toml_content,
        &["dependencies", "dev-dependencies", "build-dependencies"],
        true,
    )
}

/// Collect crate names from a dependency table, honoring optional `package = "..."`
/// aliases when present.
fn collect_dep_keys(table: &Table, deps: &mut Vec<String>, internal_only: bool) {
    for (key, value) in table {
        let pkg_name = value
            .as_table()
            .and_then(|t| t.get("package"))
            .and_then(|v| v.as_str())
            .unwrap_or(key.as_str());
        if (!internal_only || pkg_name == "oxllm" || pkg_name.starts_with("oxllm-"))
            && !deps.iter().any(|d| d == pkg_name)
        {
            deps.push(pkg_name.to_string());
        }
    }
}

#[test]
fn workspace_dag_conforms_to_allowlist() {
    let allowed = allowed_internal_deps();

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates_dir = manifest_dir.join("crates");
    let root_toml = manifest_dir.join("Cargo.toml");
    let root_content = fs::read_to_string(&root_toml).expect("root Cargo.toml must exist");
    let root_manifest: Table = toml::from_str(&root_content).expect("parse root Cargo.toml");

    // Verify [workspace.members] matches the directories under crates/ 1-to-1.
    let members: BTreeSet<String> = root_manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .expect("[workspace.members] must exist in root Cargo.toml")
        .iter()
        .map(|v| {
            v.as_str()
                .expect("workspace member must be string")
                .to_string()
        })
        .collect();

    let mut disk_members = BTreeSet::new();
    for entry in fs::read_dir(&crates_dir).expect("crates/ directory must exist") {
        let entry = entry.expect("readable dir entry");
        let crate_name = entry.file_name().to_string_lossy().to_string();
        disk_members.insert(format!("crates/{crate_name}"));
        let cargo_toml = entry.path().join("Cargo.toml");

        let content = fs::read_to_string(&cargo_toml)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", cargo_toml.display()));

        let declared = extract_internal_deps(&content);
        let allowed_list = allowed
            .get(crate_name.as_str())
            .unwrap_or_else(|| panic!("crate '{crate_name}' not in allowlist — update this test"));

        for dep in &declared {
            assert!(
                allowed_list.contains(&dep.as_str()),
                "DAG VIOLATION: crate '{crate_name}' depends on '{dep}', \
 but allowed internal deps are only: {allowed_list:?}"
            );
        }
    }

    assert_eq!(
        members, disk_members,
        "workspace.members must match crates/* directories 1-to-1"
    );

    // Root normal/build dependencies must stay strictly at the binary layer.
    let root_prod_deps =
        extract_deps_from_sections(&root_content, &["dependencies", "build-dependencies"], true);
    let allowed_root_prod = ["oxllm-server", "oxllm-admin"];
    for dep in &root_prod_deps {
        assert!(
            allowed_root_prod.contains(&dep.as_str()),
            "DAG VIOLATION: root binary 'oxllm' normal dependency '{dep}' is forbidden; \
 allowed normal deps are only: {allowed_root_prod:?}"
        );
    }

    // Root dev-dependencies may pull pure/cache crates for e2e suites.
    let declared = extract_internal_deps(&root_content);
    let allowed_list = allowed.get("oxllm").unwrap();
    for dep in &declared {
        assert!(
            allowed_list.contains(&dep.as_str()),
            "DAG VIOLATION: root binary 'oxllm' depends on '{dep}', \
 but allowed internal deps are only: {allowed_list:?}"
        );
    }
}

/// Forbidden third-party crates for pure crates (docs claim core/providers
/// stay free of network/DB I/O). Checks direct dependency keys only.
#[test]
fn pure_crates_forbid_io_dependencies() {
    let forbidden_core = ["reqwest", "sqlx", "hyper", "tokio-tungstenite", "redis"];
    let forbidden_providers = ["reqwest", "sqlx", "hyper", "redis"];

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (crate_name, forbidden) in [
        ("oxllm-core", forbidden_core.as_slice()),
        ("oxllm-providers", forbidden_providers.as_slice()),
    ] {
        let cargo_toml = manifest_dir
            .join("crates")
            .join(crate_name)
            .join("Cargo.toml");
        let content = fs::read_to_string(&cargo_toml)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", cargo_toml.display()));
        let manifest: Table = toml::from_str(&content).expect("parse Cargo.toml");
        for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
            if let Some(deps) = manifest.get(section).and_then(|v| v.as_table()) {
                for bad in forbidden {
                    assert!(
                        !deps.contains_key(*bad),
                        "PURITY VIOLATION: {crate_name} [{section}] depends on '{bad}'"
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LockPkg {
    name: String,
    version: String,
}

/// Parse `Cargo.lock` into adjacency lists keyed by `(name, version)` plus a
/// name index for resolving unambiguous `"crate"` edges alongside `"crate 1.2.3"`.
fn parse_cargo_lock(
    lock_content: &str,
) -> (
    BTreeMap<LockPkg, Vec<String>>,
    HashMap<String, Vec<LockPkg>>,
) {
    let parsed: Table = toml::from_str(lock_content).expect("parse Cargo.lock as TOML");
    let packages = parsed
        .get("package")
        .and_then(|p| p.as_array())
        .expect("Cargo.lock must contain [[package]] entries");

    let mut edges: BTreeMap<LockPkg, Vec<String>> = BTreeMap::new();
    let mut by_name: HashMap<String, Vec<LockPkg>> = HashMap::new();

    for entry in packages {
        let table = entry.as_table().expect("[[package]] entry must be table");
        let name = table
            .get("name")
            .and_then(|v| v.as_str())
            .expect("package.name")
            .to_string();
        let version = table
            .get("version")
            .and_then(|v| v.as_str())
            .expect("package.version")
            .to_string();
        let deps = table
            .get("dependencies")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let key = LockPkg {
            name: name.clone(),
            version,
        };
        by_name.entry(name).or_default().push(key.clone());
        edges.insert(key, deps);
    }

    (edges, by_name)
}

fn resolve_lock_edge(edge: &str, by_name: &HashMap<String, Vec<LockPkg>>) -> Vec<LockPkg> {
    let mut parts = edge.split_whitespace();
    let Some(name) = parts.next() else {
        return Vec::new();
    };
    if let Some(version) = parts.next() {
        return vec![LockPkg {
            name: name.to_string(),
            version: version.to_string(),
        }];
    }
    by_name.get(name).cloned().unwrap_or_default()
}

fn normal_manifest_deps(manifest_dir: &Path, crate_name: &str) -> BTreeSet<String> {
    let cargo_toml = manifest_dir
        .join("crates")
        .join(crate_name)
        .join("Cargo.toml");
    let content = fs::read_to_string(&cargo_toml)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cargo_toml.display()));
    extract_deps_from_sections(&content, &["dependencies", "build-dependencies"], false)
        .into_iter()
        .collect()
}

/// Compute the transitive `Cargo.lock` closure reachable from a workspace
/// crate's normal + build dependencies (excluding `[dev-dependencies]`, which
/// `Cargo.lock` merges into workspace `[[package]].dependencies`).
fn transitive_normal_closure(
    manifest_dir: &Path,
    edges: &BTreeMap<LockPkg, Vec<String>>,
    by_name: &HashMap<String, Vec<LockPkg>>,
    root_crate: &str,
) -> BTreeSet<LockPkg> {
    let root_pkg = by_name
        .get(root_crate)
        .and_then(|v| v.first())
        .unwrap_or_else(|| panic!("{root_crate} must exist in Cargo.lock"))
        .clone();

    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();

    let root_normal = normal_manifest_deps(manifest_dir, root_crate);
    for edge in edges.get(&root_pkg).into_iter().flatten() {
        let edge_name = edge.split_whitespace().next().unwrap_or("");
        if root_normal.contains(edge_name) {
            for target in resolve_lock_edge(edge, by_name) {
                queue.push_back(target);
            }
        }
    }

    while let Some(pkg) = queue.pop_front() {
        if !visited.insert(pkg.clone()) {
            continue;
        }
        let pkg_edges = edges.get(&pkg).into_iter().flatten();
        if pkg.name.starts_with("oxllm-") {
            let allowed_normal = normal_manifest_deps(manifest_dir, &pkg.name);
            for edge in pkg_edges {
                let edge_name = edge.split_whitespace().next().unwrap_or("");
                if allowed_normal.contains(edge_name) {
                    for target in resolve_lock_edge(edge, by_name) {
                        if !visited.contains(&target) {
                            queue.push_back(target);
                        }
                    }
                }
            }
        } else {
            for edge in pkg_edges {
                for target in resolve_lock_edge(edge, by_name) {
                    if !visited.contains(&target) {
                        queue.push_back(target);
                    }
                }
            }
        }
    }

    visited
}

fn is_forbidden_transitive_io_crate(name: &str) -> bool {
    const FORBIDDEN_EXACT: &[&str] = &[
        "reqwest",
        "hyper",
        "hyper-util",
        "h2",
        "http-body",
        "http-body-util",
        "sqlx",
        "redis",
        "tokio",
        "tokio-tungstenite",
        "tokio-stream",
        "tokio-rustls",
        "tokio-util",
        "tungstenite",
        "rustls",
        "webpki-roots",
        "rustls-webpki",
        "native-tls",
        "openssl",
        "axum",
        "tower",
        "tower-http",
        "tonic",
        "lettre",
        "moka",
        "notify",
        "jsonwebtoken",
        "ipnet",
        "minijinja",
        "aes-gcm",
        "zeroize",
    ];
    const FORBIDDEN_PREFIXES: &[&str] = &[
        "aws-sdk-",
        "aws-smithy-",
        "aws-config",
        "opentelemetry",
        "tracing-opentelemetry",
        "sqlx-",
    ];
    FORBIDDEN_EXACT.contains(&name) || FORBIDDEN_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Verify `Cargo.lock` transitive closures of `oxllm-core` and
/// `oxllm-providers` contain zero network, HTTP, DB, Redis, TLS, or async-I/O
/// runtime crates.
#[test]
fn pure_crates_forbid_transitive_io_dependencies() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lock_path = manifest_dir.join("Cargo.lock");
    let lock_content = fs::read_to_string(&lock_path).expect("Cargo.lock must exist");
    let (edges, by_name) = parse_cargo_lock(&lock_content);

    for crate_name in ["oxllm-core", "oxllm-providers"] {
        let closure = transitive_normal_closure(manifest_dir, &edges, &by_name, crate_name);
        assert!(
            !closure.is_empty(),
            "transitive closure for {crate_name} must not be empty"
        );
        let violations: Vec<String> = closure
            .iter()
            .filter(|pkg| is_forbidden_transitive_io_crate(&pkg.name))
            .map(|pkg| format!("{} {}", pkg.name, pkg.version))
            .collect();
        assert!(
            violations.is_empty(),
            "TRANSITIVE PURITY VIOLATION in {crate_name}: forbidden crates in Cargo.lock closure: {violations:?}"
        );
    }
}

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

/// Minimal Rust lexer that strips comments, normal/raw/byte/C strings, and
/// char literals (preserving `'a` lifetimes) so source checks never trigger on
/// docs, comments, or string payloads.
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
        // Line comment
        if b == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            while i < n && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Nested block comment
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
        // Raw strings: r"...", r#"..."#, br#"..."#, cr#"..."#
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
        // Normal / byte / c strings: "...", b"...", c"..."
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
        // Byte/char literals vs lifetimes
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
            } else if i + 1 < n {
                // Decode one UTF-8 scalar after `'` to see if a closing `'` follows.
                let rest = &src[i + 1..];
                if let Some(ch) = rest.chars().next() {
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
        }
        // Identifiers / keywords
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
        // Multi-char punct
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
    if inner == ["cfg", "(", "test", ")"]
        || (inner.len() >= 6 && inner[..5] == ["cfg", "(", "all", "(", "test"])
    {
        Some(j + 1)
    } else {
        None
    }
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
    for tok in tokens.iter().skip(cur).take(10) {
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

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut sorted: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    sorted.sort_by_key(|e| e.path());
    for entry in sorted {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
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

fn discover_test_only_files(src_dirs: &[PathBuf]) -> BTreeSet<PathBuf> {
    let mut all_files = Vec::new();
    for dir in src_dirs {
        collect_rs_files(dir, &mut all_files);
    }
    let mut test_only = BTreeSet::new();

    // Seed from #[cfg(test)] / #[cfg(all(test, ...))] external `mod` items
    // anywhere inside production files (including nested inside inline #[cfg(test)] blocks).
    for file in &all_files {
        let content = fs::read_to_string(file).expect("readable rs file");
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
                {
                    if let Some(target) = resolve_mod_file(file, &tokens[i + 1].text, None) {
                        test_only.insert(target);
                    }
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

    // Transitively mark submodules declared inside test-only files.
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

    test_only
}

fn non_test_tokens(tokens: &[Tok]) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if let Some(after) = is_cfg_test_attr(tokens, i) {
            i = skip_item_after_cfg_test(tokens, after);
            continue;
        }
        out.push(tokens[i].clone());
        i += 1;
    }
    out
}

/// Verify non-test production code in `oxllm-core` and `oxllm-providers`
/// never imports or calls I/O, filesystem, env, process, or async runtime APIs,
/// and that `oxllm-providers` contains no `async fn`.
#[test]
fn pure_crates_forbid_io_apis_in_production_source() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let core_src = manifest_dir.join("crates/oxllm-core/src");
    let providers_src = manifest_dir.join("crates/oxllm-providers/src");
    let test_only = discover_test_only_files(&[core_src.clone(), providers_src.clone()]);

    let forbidden_two_seg = [
        ("std", "fs"),
        ("std", "net"),
        ("std", "env"),
        ("std", "process"),
    ];
    let forbidden_three_seg = [("std", "thread", "spawn"), ("std", "thread", "sleep")];
    let forbidden_prefixes = [
        "tokio", "reqwest", "hyper", "sqlx", "redis", "axum", "tower",
    ];
    let forbidden_idents = ["TcpStream", "TcpListener", "UdpSocket", "oxllm_server"];

    for (crate_name, src_dir) in [("oxllm-core", core_src), ("oxllm-providers", providers_src)] {
        let mut files = Vec::new();
        collect_rs_files(&src_dir, &mut files);
        for file in files {
            let canon = fs::canonicalize(&file).expect("canonical path");
            if test_only.contains(&canon) {
                continue;
            }
            let content = fs::read_to_string(&file).expect("read source");
            let tokens = non_test_tokens(&tokenize_rust(&content));
            let rel = file.strip_prefix(manifest_dir).unwrap_or(&file);

            for (idx, tok) in tokens.iter().enumerate() {
                if tok.kind != TokKind::Ident {
                    continue;
                }
                if forbidden_idents.contains(&tok.text.as_str()) {
                    panic!(
                        "SOURCE PURITY VIOLATION in {crate_name}: forbidden identifier '{}' at {}:{}",
                        tok.text,
                        rel.display(),
                        tok.line
                    );
                }
                if crate_name == "oxllm-core" && tok.text == "oxllm_providers" {
                    panic!(
                        "SOURCE PURITY VIOLATION in oxllm-core: upward reference to 'oxllm_providers' at {}:{}",
                        rel.display(),
                        tok.line
                    );
                }
                if crate_name == "oxllm-providers"
                    && tok.text == "async"
                    && tokens.get(idx + 1).is_some_and(|n| n.text == "fn")
                {
                    panic!(
                        "SOURCE PURITY VIOLATION in oxllm-providers: 'async fn' forbidden (providers must be synchronous pure transforms) at {}:{}",
                        rel.display(),
                        tok.line
                    );
                }
                if forbidden_prefixes.contains(&tok.text.as_str())
                    && tokens.get(idx + 1).is_some_and(|n| n.text == "::")
                {
                    panic!(
                        "SOURCE PURITY VIOLATION in {crate_name}: forbidden path '{}::' at {}:{}",
                        tok.text,
                        rel.display(),
                        tok.line
                    );
                }
                if tokens.get(idx + 1).is_some_and(|n| n.text == "::")
                    && let Some(second) = tokens.get(idx + 2)
                {
                    if forbidden_two_seg.contains(&(tok.text.as_str(), second.text.as_str())) {
                        panic!(
                            "SOURCE PURITY VIOLATION in {crate_name}: forbidden path '{}::{}' at {}:{}",
                            tok.text,
                            second.text,
                            rel.display(),
                            tok.line
                        );
                    }
                    if tokens.get(idx + 3).is_some_and(|n| n.text == "::")
                        && let Some(third) = tokens.get(idx + 4)
                        && forbidden_three_seg.contains(&(
                            tok.text.as_str(),
                            second.text.as_str(),
                            third.text.as_str(),
                        ))
                    {
                        panic!(
                            "SOURCE PURITY VIOLATION in {crate_name}: forbidden path '{}::{}::{}' at {}:{}",
                            tok.text,
                            second.text,
                            third.text,
                            rel.display(),
                            tok.line
                        );
                    }
                }
            }
        }
    }
}
