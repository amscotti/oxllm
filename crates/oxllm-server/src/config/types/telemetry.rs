//! Observability backend config: Langfuse, Langsmith, Datadog, Helicone,
//! Mlflow, Weights & Biases, S3 log storage.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LangfuseConfig {
    pub host: String,
    pub public_key: String,
    pub secret_key: String,
    #[serde(default)]
    pub flush_interval_secs: Option<u64>,
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    #[serde(default)]
    pub log_inputs: Option<bool>,
    #[serde(default)]
    pub log_outputs: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LangsmithConfig {
    pub api_key: String,
    pub project: String,
    #[serde(default)]
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatadogConfig {
    pub api_key: String,
    #[serde(default)]
    pub service: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeliconeConfig {
    pub api_key: String,
    #[serde(default)]
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlflowConfig {
    pub tracking_uri: String,
    #[serde(default)]
    pub experiment_name: Option<String>,
    /// Pre-existing MLflow experiment id. When set, the hook skips the
    /// `get-by-name` / `create` experiment resolution and uses this id
    /// directly when creating runs.
    #[serde(default)]
    pub experiment_id: Option<String>,
    /// Pre-existing MLflow run id. When set, the hook logs every request's
    /// metrics and params against this run (no run creation is attempted).
    /// When unset, the hook lazily resolves (or creates) a single long-lived
    /// run per experiment on first use.
    #[serde(default)]
    pub run_id: Option<String>,
}

/// Configuration for the Weights & Biases spend-logging hook.
///
/// ```yaml
/// oxllm_settings:
///   wandb_config:
///     api_key: wandb-key
///     project: oxllm
///     entity: my-team
///     tags: ["prod"]
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WandbConfig {
    /// W&B API key (sent as `Authorization: Bearer {api_key}`).
    pub api_key: String,
    /// Target W&B project name.
    pub project: String,
    /// W&B entity (team/user) under which the project lives.
    #[serde(default)]
    pub entity: Option<String>,
    /// Free-form tags attached to every logged spend record.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Override the base endpoint (defaults to `https://api.wandb.ai`).
    #[serde(default)]
    pub endpoint: Option<String>,
}

/// Configuration for the S3 log-storage hook.
///
/// ```yaml
/// oxllm_settings:
///   s3_log_config:
///     bucket: my-llm-logs
///     region: us-east-1
///     prefix: oxllm
///     access_key_id: AKIA...
///     secret_access_key: ...
/// ```
///
/// When `access_key_id` / `secret_access_key` are omitted, the hook relies on
/// the ambient AWS credentials (IRSA / instance role), but since this hook
/// signs requests with SigV4 directly, explicit credentials are required to be
/// useful outside of an AWS environment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S3LogConfig {
    /// Target S3 bucket name.
    pub bucket: String,
    /// AWS region the bucket lives in (e.g. `"us-east-1"`).
    pub region: String,
    /// Key prefix prepended to every object key. Defaults to `"oxllm"`.
    #[serde(default)]
    pub prefix: Option<String>,
    /// AWS access key id. Required for SigV4 signing (no ambient provider).
    #[serde(default)]
    pub access_key_id: Option<String>,
    /// AWS secret access key.
    #[serde(default)]
    pub secret_access_key: Option<String>,
    /// Custom S3-compatible endpoint (MinIO, R2, etc.). When set, path-style
    /// addressing is used against this base URL.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// When `true`, include a redacted copy of the response body alongside
    /// the request body in each S3 log object. Defaults to `true` (matching
    /// LiteLLM's S3 logging, which persists both halves of the exchange).
    #[serde(default)]
    pub log_response_body: Option<bool>,
}
