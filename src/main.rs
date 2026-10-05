use clap::{Parser, Subcommand};
#[cfg(feature = "db")]
use oxllm::canonical_sqlite_url;
use oxllm::{generate_random_key, random_id, sha256_hex};

#[derive(Parser)]
#[command(
    name = "oxllm",
    version,
    about = "A fast Rust proxy gateway for LLM APIs"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the proxy server.
    Serve {
        #[arg(short, long, env = "OXLLM_CONFIG_PATH", default_value = "config.yaml")]
        config: String,
        // `Option<u16>` (no `default_value_t`) so an explicit `--port 4000`
        // is distinguishable from "not given"; `oxllm_server::run` resolves
        // the effective port (CLI flag > env > built-in default of 4000).
        #[arg(short, long, env = "OXLLM_PORT")]
        port: Option<u16>,
    },
    /// Validate a config file without starting the server.
    Validate {
        #[arg(short, long, env = "OXLLM_CONFIG_PATH", default_value = "config.yaml")]
        config: String,
    },
    /// Generate a new virtual API key (`sk-ox-…`), print it, and optionally
    /// insert it into the database. Metadata flags (--alias/--team-id/
    /// --user-id/--max-budget/--budget-duration) only apply when
    /// --database-url is given and are rejected otherwise.
    Keygen {
        /// Optional alias for the generated key (requires --database-url).
        #[arg(long)]
        alias: Option<String>,
        /// Optional team id to attach to the key (requires --database-url).
        #[arg(long)]
        team_id: Option<String>,
        /// Optional user id to attach to the key (requires --database-url).
        #[arg(long)]
        user_id: Option<String>,
        /// Optional per-key budget in USD; must be finite and non-negative
        /// (requires --database-url).
        #[arg(long)]
        max_budget: Option<f64>,
        /// Optional budget window (e.g. `30d`, `1mo`; requires
        /// --database-url).
        #[arg(long)]
        budget_duration: Option<String>,
        /// When set, persist the key to the database at this URL
        /// (e.g. `sqlite://oxllm.db`).
        #[arg(long)]
        database_url: Option<String>,
    },
    /// Run database migrations against the configured database.
    #[cfg(feature = "db")]
    Migrate {
        /// Database URL to migrate (e.g. `sqlite://oxllm.db`,
        /// `postgres://user:pass@host/db`).
        #[arg(long, env = "OXLLM_DATABASE_URL")]
        database_url: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Install the rustls crypto provider for tokio-tungstenite (WebSocket TLS).
    // reqwest manages its own TLS, but the realtime WebSocket proxy uses
    // tokio-tungstenite which needs a process-level CryptoProvider.
    // This MUST be unconditional: `oxllm-server`'s `realtime` module is
    // always compiled and opens `wss://` upstreams; without a provider every
    // `connect_async` to a `wss://` URL returns `NoCryptoProvider`. The call
    // is idempotent (returns Err if already installed, which we discard).
    install_tls_provider();

    // Defer the global tracing subscriber install: `serve` may want to attach
    // an OpenTelemetry layer, and `tracing::subscriber::set_global_default`
    // can only be called once. `serve` installs its own subscriber; every
    // other subcommand falls back to a plain fmt subscriber here.
    let cli = Cli::parse();

    let needs_default_subscriber = !matches!(cli.command, Commands::Serve { .. });
    if needs_default_subscriber {
        // Honour RUST_LOG like the serve path does (`install_fmt_subscriber`
        // in oxllm-server's app.rs): docs/troubleshooting.md recommends
        // `RUST_LOG=oxllm=debug`, which a bare fmt subscriber (defaults to
        // INFO, no env parsing) would silently ignore for validate/keygen/
        // migrate.
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .or_else(|_| tracing_subscriber::EnvFilter::try_new("info"))
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .init();
    }

    match cli.command {
        Commands::Serve { config, port } => {
            // The global tracing subscriber is installed inside
            // `oxllm_server::run` so it can honour the `log_format` setting
            // from the config file (JSON vs text). Do NOT install one here —
            // `set_global_default` can only succeed once.
            oxllm_server::run(&config, port).await?;
        }
        Commands::Validate { config } => {
            tracing::info!(%config, "validating config");
            match oxllm_server::load_config(&config) {
                Ok(cfg) => {
                    let model_count = cfg.model_list.len();
                    let auth_enabled = oxllm_server::app::auth_is_configured(
                        &cfg.general_settings.master_key,
                        &cfg.general_settings.jwt_secret,
                        &cfg.general_settings.database_url,
                    );
                    // Pre-flight must fail exactly like `serve` would: run the
                    // same OXLLM_REQUIRE_AUTH guard `oxllm_server::run`
                    // enforces, so a config the server would refuse to start
                    // is never reported as "config is valid".
                    if let Err(e) = oxllm_server::app::enforce_require_auth_guard(&cfg) {
                        tracing::error!(error = %e, "config validation failed");
                        return Err(e);
                    }
                    if model_count == 0 {
                        tracing::warn!(
                            "config is valid but model_list is empty: no models will be \
                             served (check that the model section loaded correctly)"
                        );
                    }
                    tracing::info!(models = model_count, auth_enabled, "config is valid");
                    for model in &cfg.model_list {
                        tracing::info!(
                        model_name = %model.model_name,
                        provider_model = %model.oxllm_params.model,
                        "configured model"
                        );
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "config validation failed");
                    return Err(e);
                }
            }
        }
        Commands::Keygen {
            alias,
            team_id,
            user_id,
            max_budget,
            budget_duration,
            database_url,
        } => {
            run_keygen(
                alias,
                team_id,
                user_id,
                max_budget,
                budget_duration,
                database_url,
            )
            .await?;
        }
        #[cfg(feature = "db")]
        Commands::Migrate { database_url } => {
            run_migrate(&database_url).await?;
        }
    }

    Ok(())
}

/// Install the process-level rustls `CryptoProvider` (aws-lc-rs) required by
/// `tokio-tungstenite` for `wss://` connections in the realtime proxy path.
/// `reqwest` manages its own TLS and is unaffected.
///
/// Idempotent: `install_default` returns `Err` if a provider is already
/// installed (e.g. by another crate), which is discarded. Extracted from
/// `main()` so a test can assert the provider ends up installed even under
/// `--no-default-features` (the original bug gated this on `feature = "db"`,
/// so a default build shipped a binary whose realtime TLS upstreams failed at
/// runtime with `NoCryptoProvider`).
fn install_tls_provider() {
    let _ = ::rustls::crypto::aws_lc_rs::default_provider().install_default();
}

/// Generate `sk-ox-…` (40 hex chars of entropy), compute its SHA-256 hash, and
/// optionally insert the key into the database. The raw key is printed exactly
/// once — only the hash is persisted.
async fn run_keygen(
    alias: Option<String>,
    team_id: Option<String>,
    user_id: Option<String>,
    max_budget: Option<f64>,
    budget_duration: Option<String>,
    database_url: Option<String>,
) -> anyhow::Result<()> {
    let raw_key = generate_random_key();
    let key_hash = sha256_hex(raw_key.as_bytes());
    let key_id = format!("key-{}", random_id());

    // Validate the budget window up front: a malformed value (e.g. `30 days`,
    // `1.5h`) would otherwise be stored verbatim and only fail — or silently
    // fall back to the 30d default — at request time. Uses the exact same
    // parser the enforcement path uses (`scheduler::parse_budget_duration`).
    if let Some(d) = budget_duration.as_deref() {
        if !oxllm_server::scheduler::budget_duration_is_valid(d) {
            anyhow::bail!(
                "invalid --budget-duration {d:?}: expected a suffixed duration like \
                 `30d`, `1mo`, `24h`, `5m`, `500ms`, or a word alias \
                 (hourly/daily/weekly/monthly)"
            );
        }
    }

    // `--max-budget` must pass the same finiteness / non-negativity check the
    // `/key/generate` handler enforces (handlers/management/mod.rs): a NaN
    // silently disables the limit at enforcement (every comparison with NaN
    // is false → unlimited spend in hooks/budget.rs) and a negative value
    // makes every request fail `BudgetExceeded` (`limit <= 0.0`). `0.0`
    // remains valid (reject-all key).
    if let Some(b) = max_budget {
        if !b.is_finite() {
            anyhow::bail!("invalid --max-budget {b}: must be a finite number");
        }
        if b < 0.0 {
            anyhow::bail!("invalid --max-budget {b}: must be non-negative");
        }
    }

    // Metadata flags only have an effect when the key is persisted: without
    // `--database-url` nothing is inserted, so the key can never authenticate
    // (virtual-key auth checks the SHA-256 hash in the DB) and the flags
    // would be silently discarded — bail instead of printing a no-op key.
    if database_url.is_none()
        && (alias.is_some()
            || team_id.is_some()
            || user_id.is_some()
            || max_budget.is_some()
            || budget_duration.is_some())
    {
        anyhow::bail!(
            "--alias/--team-id/--user-id/--max-budget/--budget-duration require \
             --database-url: without a database the key is not persisted (so it \
             can never authenticate) and the metadata would be discarded"
        );
    }

    // Without the `db` feature persistence is impossible: bail BEFORE the
    // key is printed, so a failed keygen never emits a key that looks valid.
    #[cfg(not(feature = "db"))]
    {
        if database_url.is_some() {
            anyhow::bail!(
                "database persistence requires the `db` feature (rebuild with `--features db`)"
            );
        }
    }

    #[cfg(feature = "db")]
    {
        if let Some(url) = database_url {
            let url = canonical_sqlite_url(url)?;
            let db = match oxllm_server::Database::connect(&url).await {
                Ok(db) => db,
                Err(err) => {
                    eprintln!("key NOT persisted (database connect failed): {err}");
                    return Err(err);
                }
            };
            if let Err(err) = db.migrate().await {
                eprintln!("key NOT persisted (migration failed): {err}");
                return Err(err);
            }
            let now = oxllm_server::now_iso();
            let record = oxllm_server::KeyRecord {
                key_id: key_id.clone(),
                key_hash: key_hash.clone(),
                key_alias: alias.clone(),
                team_id: team_id.clone(),
                user_id: user_id.clone(),
                org_id: None,
                project_id: None,
                max_budget,
                budget_duration,
                rpm: None,
                tpm: None,
                created_at: now.clone(),
                updated_at: now,
                expires_at: None,
                is_active: true,
                models: None,
                allowed_cache_controls: None,
            };
            if let Err(err) = db.create_key(&record).await {
                eprintln!("key NOT persisted (create_key failed): {err}");
                return Err(err);
            }
            tracing::info!(
            key_id = %key_id,
            url = %oxllm_server::security::sanitize_url(&url),
            "key persisted"
            );
        }
    }

    // Print only after persistence succeeded (or when no persistence was
    // requested): a key printed before a failed insert would look valid.
    println!("Key: {raw_key}");
    println!("Key ID: {key_id}");
    println!("Key Hash: {key_hash}");

    Ok(())
}

/// Apply database migrations to the configured database.
#[cfg(feature = "db")]
async fn run_migrate(database_url: &str) -> anyhow::Result<()> {
    let url = canonical_sqlite_url(database_url.to_string())?;
    let db = oxllm_server::Database::connect(&url).await?;
    db.migrate().await?;
    let version = db.get_schema_version().await?;
    tracing::info!(
        version = version,
        url = %oxllm_server::security::sanitize_url(&url),
        "migrations applied"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tls_crypto_provider_is_installed_by_preamble() {
        // Mirrors the `main()` preamble. Must leave a process-level
        // CryptoProvider installed so the realtime `wss://` path does not
        // fail with `NoCryptoProvider` at runtime.
        install_tls_provider();
        assert!(
            ::rustls::crypto::CryptoProvider::get_default().is_some(),
            "rustls CryptoProvider must be installed after the main() preamble; \
             without it every wss:// connect_async in the realtime module fails"
        );
    }

    /// C60 MEDIUM: `validate` must report the same auth status `run()` would
    /// enforce. An operator who configures only JWT (or only the virtual-key
    /// DB) — a common LiteLLM deployment shape — must see `auth_enabled=true`,
    /// not be misled into thinking the proxy will start in open mode. The
    /// subcommand now uses the shared `auth_is_configured` helper; this pins
    /// that contract end-to-end from a parsed config.
    #[test]
    fn validate_reports_auth_enabled_for_jwt_only_config() {
        let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o

general_settings:
  jwt_secret: a-long-secret-value
";
        let cfg = oxllm_server::parse_config_str(yaml).expect("jwt-only config should parse");
        let auth_enabled = oxllm_server::app::auth_is_configured(
            &cfg.general_settings.master_key,
            &cfg.general_settings.jwt_secret,
            &cfg.general_settings.database_url,
        );
        assert!(
            cfg.general_settings.master_key.is_none(),
            "test precondition: this is a jwt-only config"
        );
        assert!(
            auth_enabled,
            "jwt-only config must report auth_enabled=true, matching run()"
        );
    }

    /// C60 MEDIUM: the db-only deployment shape (virtual keys via database_url,
    /// no master_key / jwt_secret) must likewise report auth enabled.
    #[test]
    fn validate_reports_auth_enabled_for_db_only_config() {
        let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o

general_settings:
  database_url: sqlite://oxllm.db
";
        let cfg = oxllm_server::parse_config_str(yaml).expect("db-only config should parse");
        let auth_enabled = oxllm_server::app::auth_is_configured(
            &cfg.general_settings.master_key,
            &cfg.general_settings.jwt_secret,
            &cfg.general_settings.database_url,
        );
        assert!(
            cfg.general_settings.master_key.is_none(),
            "test precondition: this is a db-only config"
        );
        assert!(
            auth_enabled,
            "db-only config must report auth_enabled=true, matching run()"
        );
    }

    /// Review finding 1: `oxllm validate` must run the same
    /// `OXLLM_REQUIRE_AUTH` guard `oxllm_server::run` enforces, so a config
    /// the server would refuse to start is never reported as valid. Pins the
    /// shared guard end-to-end from a parsed, secret-less config.
    #[serial_test::serial]
    #[test]
    fn validate_rejects_secret_less_config_when_require_auth_set() {
        let yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
";
        let cfg = oxllm_server::parse_config_str(yaml).expect("open-mode config should parse");
        let prev = std::env::var("OXLLM_REQUIRE_AUTH").ok();
        // SAFETY: process-local test env var; restored before return and the
        // test is #[serial] so it cannot race other env-mutating tests.
        unsafe {
            std::env::set_var("OXLLM_REQUIRE_AUTH", "true");
        }
        let result = oxllm_server::app::enforce_require_auth_guard(&cfg);
        // Restore before asserting so a failure does not leak the env var.
        match prev {
            Some(v) => unsafe { std::env::set_var("OXLLM_REQUIRE_AUTH", v) },
            None => unsafe { std::env::remove_var("OXLLM_REQUIRE_AUTH") },
        }
        let err = result.expect_err("guard must reject a secret-less config");
        let msg = format!("{err}");
        assert!(
            msg.contains("OXLLM_REQUIRE_AUTH"),
            "error should mention OXLLM_REQUIRE_AUTH: {msg}"
        );
    }

    /// Review finding 2: `--max-budget` must be validated in `run_keygen`
    /// with the same finiteness / non-negativity check `/key/generate`
    /// enforces. NaN would silently disable the limit at enforcement and a
    /// negative value would fail every request with `BudgetExceeded`.
    #[tokio::test]
    async fn keygen_rejects_non_finite_max_budget() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = run_keygen(None, None, None, Some(bad), None, None)
                .await
                .expect_err("non-finite --max-budget must bail");
            assert!(
                format!("{err}").contains("finite"),
                "error should cite finiteness for {bad}: {err}"
            );
        }
    }

    #[tokio::test]
    async fn keygen_rejects_negative_max_budget() {
        let err = run_keygen(None, None, None, Some(-1.0), None, None)
            .await
            .expect_err("negative --max-budget must bail");
        assert!(
            format!("{err}").contains("non-negative"),
            "error should cite non-negativity: {err}"
        );
    }

    /// Review finding 3: metadata flags without `--database-url` used to be
    /// silently discarded while an unpersisted (unusable) key was printed as
    /// if created; they must now bail before any key is emitted.
    #[tokio::test]
    async fn keygen_rejects_metadata_flags_without_database_url() {
        let err = run_keygen(Some("prod".to_string()), None, None, Some(50.0), None, None)
            .await
            .expect_err("metadata flags without --database-url must bail");
        let msg = format!("{err}");
        assert!(
            msg.contains("--database-url"),
            "error should tell the user to pass --database-url: {msg}"
        );
        // Same for each individual flag.
        for case in [
            (
                Some("a".to_string()),
                None::<String>,
                None::<String>,
                None::<f64>,
                None::<String>,
            ),
            (
                None,
                Some("t".to_string()),
                None,
                None::<f64>,
                None::<String>,
            ),
            (
                None::<String>,
                None,
                Some("u".to_string()),
                None::<f64>,
                None::<String>,
            ),
            (
                None::<String>,
                None::<String>,
                None,
                Some(5.0),
                None::<String>,
            ),
            (
                None::<String>,
                None::<String>,
                None::<String>,
                None::<f64>,
                Some("30d".to_string()),
            ),
        ] {
            let (alias, team_id, user_id, max_budget, budget_duration) = case;
            assert!(
                run_keygen(alias, team_id, user_id, max_budget, budget_duration, None)
                    .await
                    .is_err(),
                "each metadata flag alone must bail without --database-url"
            );
        }
    }

    /// `--max-budget 0` stays valid (reject-all key) and, with a database,
    /// round-trips through persistence exactly like the API path.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn keygen_accepts_zero_max_budget_with_database() {
        run_keygen(
            None,
            None,
            None,
            Some(0.0),
            None,
            Some("sqlite::memory:".to_string()),
        )
        .await
        .expect("0.0 budget with --database-url must succeed");
    }

    /// The serve `--port` must be an `Option` so "explicitly 4000" is
    /// distinguishable from "not given" (a `default_value_t` would erase the
    /// distinction, and `oxllm_server::run` resolves the effective port).
    #[test]
    fn serve_port_defaults_to_none_and_accepts_explicit_value() {
        match Cli::try_parse_from(["oxllm", "serve"]).unwrap() {
            Cli {
                command: Commands::Serve { port, .. },
            } => assert_eq!(port, None, "no --port flag must yield None, not a default"),
            _ => panic!("expected Serve"),
        }
        match Cli::try_parse_from(["oxllm", "serve", "--port", "4000"]).unwrap() {
            Cli {
                command: Commands::Serve { port, .. },
            } => assert_eq!(port, Some(4000)),
            _ => panic!("expected Serve"),
        }
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn migrate_runs_against_in_memory_sqlite() {
        let db = oxllm_server::Database::connect("sqlite::memory:")
            .await
            .unwrap();
        db.migrate().await.unwrap();
        assert!(db.ping().await);
    }

    /// C68 LOW: `run_migrate` must report the resulting schema version in
    /// its success log so an operator grepping init-container logs during a
    /// failed rolling deploy can confirm the DB is at the expected version.
    #[cfg(feature = "db")]
    #[tokio::test]
    async fn run_migrate_reports_schema_version() {
        // run_migrate uses tracing::info! for its output. We can't easily
        // capture tracing output in a unit test, so instead verify the
        // underlying contract: after migrate, get_schema_version returns
        // the current version, which is what run_migrate logs.
        let url = "sqlite::memory:";
        let db = oxllm_server::Database::connect(url).await.unwrap();
        db.migrate().await.unwrap();
        let version = db.get_schema_version().await.unwrap();
        assert!(
            version > 0,
            "schema version must be > 0 after migrate (got {version})"
        );
    }

    /// C68 LOW: without the `db` feature the `migrate` subcommand must not
    /// appear in clap's parsed subcommands — the user gets a clap error at
    /// parse time, not a runtime bail.
    #[cfg(not(feature = "db"))]
    #[test]
    fn migrate_subcommand_absent_without_db_feature() {
        let result = Cli::try_parse_from(["oxllm", "migrate", "--database-url", "x"]);
        let err_msg = match result {
            Ok(_) => panic!("clap should reject `migrate` when the `db` feature is disabled"),
            Err(e) => e.to_string(),
        };
        assert!(
            err_msg.contains("unrecognized") || err_msg.contains("invalid"),
            "error should indicate the subcommand is not recognized: {err_msg}"
        );
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn keygen_inserts_into_db_when_url_given() {
        let url = "sqlite::memory:".to_string();
        let db = oxllm_server::Database::connect(&url).await.unwrap();
        db.migrate().await.unwrap();

        let raw_key = generate_random_key();
        let key_hash = sha256_hex(raw_key.as_bytes());
        let key_id = format!("key-{}", random_id());
        let now = oxllm_server::now_iso();
        let record = oxllm_server::KeyRecord {
            key_id: key_id.clone(),
            key_hash: key_hash.clone(),
            key_alias: Some("alias-x".to_string()),
            team_id: None,
            user_id: None,
            org_id: None,
            project_id: None,
            max_budget: Some(5.0),
            budget_duration: None,
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now,
            expires_at: None,
            is_active: true,
            models: None,
            allowed_cache_controls: None,
        };
        db.create_key(&record).await.unwrap();

        let fetched = db
            .get_key_by_hash(&key_hash)
            .await
            .unwrap()
            .expect("key should exist");
        assert_eq!(fetched.key_id, key_id);
        assert_eq!(fetched.key_alias.as_deref(), Some("alias-x"));
        assert!(fetched.is_active);
    }
}
