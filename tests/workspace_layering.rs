//! Workspace crate dependency layering (DAG) checks.

use std::collections::HashMap;
use std::fs;

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
/// - oxllm-db: only oxllm-core
/// - oxllm-server: oxllm-core, oxllm-providers, oxllm-cache-redis, oxllm-cache-s3,
///   oxllm-cache-gcs, oxllm-cache-azure, oxllm-db
/// - oxllm-admin: no internal deps (remote HTTP client of the proxy only)
/// - oxllm (root binary): oxllm-server, oxllm-admin (optional admin UI binary);
///   dev-dependencies additionally allow oxllm-core, oxllm-providers, and
///   oxllm-cache-redis for the workspace-level integration tests (they never
///   ship in the binary)
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

/// Extract every `oxllm-*` dependency declared anywhere in the manifest,
/// scanning all dependency tables (including target-specific ones).
///
/// Handles all three TOML declaration forms uniformly because the value type
/// is irrelevant — only the key (the crate name) matters for the layering
/// check:
/// - `oxllm-core.workspace = true`
/// - `oxllm-core = { path = "..." }`
/// - `[dependencies.oxllm-core]` (table form)
fn extract_internal_deps(toml_content: &str) -> Vec<String> {
    let manifest: Table = match toml::from_str(toml_content) {
        Ok(m) => m,
        Err(e) => panic!("failed to parse Cargo.toml as TOML: {e}"),
    };

    let mut deps: Vec<String> = Vec::new();

    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(toml::Value::Table(table)) = manifest.get(key) {
            collect_oxllm_deps(table, &mut deps);
        }
    }

    if let Some(toml::Value::Table(targets)) = manifest.get("target") {
        for value in targets.values() {
            if let toml::Value::Table(target_table) = value {
                for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
                    if let Some(toml::Value::Table(dep_table)) = target_table.get(key) {
                        collect_oxllm_deps(dep_table, &mut deps);
                    }
                }
            }
        }
    }

    deps.sort();
    deps.dedup();
    deps
}

/// Scan a dependency table for any key beginning with `oxllm-` or equal to
/// `oxllm`.
fn collect_oxllm_deps(table: &Table, deps: &mut Vec<String>) {
    for key in table.keys() {
        if (key == "oxllm" || key.starts_with("oxllm-")) && !deps.contains(key) {
            deps.push(key.clone());
        }
    }
}

#[test]
fn workspace_dag_conforms_to_allowlist() {
    let allowed = allowed_internal_deps();

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates_dir = manifest_dir.join("crates");

    for entry in fs::read_dir(&crates_dir).expect("crates/ directory must exist") {
        let entry = entry.expect("readable dir entry");
        let crate_name = entry.file_name().to_string_lossy().to_string();
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

    let root_toml = manifest_dir.join("Cargo.toml");
    let content = fs::read_to_string(&root_toml).expect("root Cargo.toml must exist");
    let declared = extract_internal_deps(&content);
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

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
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
