//! oxllm-admin: the admin UI server for the oxllm proxy gateway.
//!
//! A separate Axum process acting as a backend-for-frontend: it renders
//! MiniJinja + HTMX HTML pages and talks to the proxy's master-key management
//! API via [`proxy_client::ProxyClient`]. The browser never holds the master
//! key. This crate has **no** dependency on any internal `oxllm-*` crate.

// Production code is `forbid(unsafe_code)`. The crate-level attribute is
// test-scoped off because `config.rs` unit tests mutate process environment
// variables via `std::env::set_var`/`remove_var`, which are `unsafe` in
// edition 2024. Shipped (non-test) code contains no `unsafe`.
#![cfg_attr(not(test), forbid(unsafe_code))]

pub mod app;
mod assets;
pub mod auth;
pub mod config;
mod csv;
mod dates;
pub mod error;
pub mod middleware;
mod pages;
pub mod proxy_client;
pub mod templates;
mod urlencode;

use clap::{Parser, Subcommand};

pub use app::AppState;
pub use config::AdminConfig;
pub use error::AdminError;
pub use proxy_client::ProxyClient;
pub use templates::Templates;

/// Admin-server CLI.
#[derive(Parser, Debug)]
#[command(
    name = "oxllm-admin",
    version,
    about = "Admin UI for the oxllm proxy gateway"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the admin UI server.
    Serve {
        /// Path to an admin YAML config file.
        #[arg(short, long, env = "OXLLM_ADMIN_CONFIG")]
        config: Option<String>,
        /// Bind address (overrides config / `OXLLM_ADMIN_LISTEN`).
        #[arg(long, env = "OXLLM_ADMIN_LISTEN")]
        listen: Option<String>,
        /// Proxy base URL (overrides config / `OXLLM_PROXY_URL`).
        #[arg(long, env = "OXLLM_PROXY_URL")]
        proxy_url: Option<String>,
    },
}

/// Entry point used by the `oxllm-admin` binary.
pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Install a plain fmt subscriber only if no global subscriber has been
    // installed yet (`try_init`): the caller may have composed its own
    // (e.g. with an OTel layer), and `set_global_default` is one-shot —
    // unconditionally calling `init()` in the binary preamble would burn it.
    // Done after `Cli::parse()` so a bad invocation fails before any
    // subscriber side effects, mirroring `oxllm`'s main.
    let _ = tracing_subscriber::fmt().with_target(false).try_init();

    match cli.command {
        Commands::Serve {
            config,
            listen,
            proxy_url,
        } => serve(config, listen, proxy_url).await,
    }
}

async fn serve(
    config_path: Option<String>,
    listen_override: Option<String>,
    proxy_override: Option<String>,
) -> anyhow::Result<()> {
    let mut config = config::load(config_path.as_deref())?;
    if let Some(listen) = listen_override {
        config.listen = listen;
    }
    if let Some(proxy_url) = proxy_override {
        config.proxy_url = proxy_url;
    }
    // The `--listen` override lands after `config::load` has validated, so
    // re-run the bind/password guard on the final config. Without this,
    // `Dockerfile.admin`'s `--listen 0.0.0.0:8080` would bypass the
    // non-loopback password check and serve an open admin UI.
    config::enforce_bind_password_guard(&config)?;
    // Validate the final URL too: --proxy-url is applied after file/env loading.
    let proxy_url = config::validate_proxy_url(&config.proxy_url)?;
    config.proxy_url = proxy_url.to_string();
    config::log_transport_diagnostics(&config, &proxy_url);

    tracing::info!(listen = %config.listen, "starting oxllm-admin");

    let login_brake = std::sync::Arc::new(auth::LoginBrake::new(
        config.login_max_fails,
        config.login_window_secs,
        config.login_block_secs,
    ));
    let state = AppState {
        config: std::sync::Arc::new(config.clone()),
        proxy: std::sync::Arc::new(ProxyClient::new(&config.proxy_url, &config.master_key)),
        templates: std::sync::Arc::new(Templates::new()),
        login_brake,
    };

    let app = app::build(state);
    let listener = tokio::net::TcpListener::bind(&config.listen).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;
    Ok(())
}
