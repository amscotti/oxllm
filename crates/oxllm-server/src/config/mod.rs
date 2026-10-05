//! Config loading module.
//!
//! Types live in [`types`]; the YAML loader and `os.environ/` resolver live
//! in [`loader`]. Hot-reload lives in [`watcher`].

pub mod loader;
pub mod types;
pub mod watcher;

pub use loader::{
    load_config, load_config_async, parse_config_str, resolve_env_vars, validate_api_base,
    validate_host_port,
};
pub use types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings, RoutingKnobs,
};
