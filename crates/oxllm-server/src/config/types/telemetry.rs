//! Observability backend config: Langfuse, Langsmith, Datadog, Helicone,
//! Mlflow, Weights & Biases, S3 log storage.

use serde::{Deserialize, Serialize};

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
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

impl std::fmt::Debug for LangfuseConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangfuseConfig")
            .field("host", &self.host)
            .field("public_key", &super::redact_str(&self.public_key))
            .field("secret_key", &super::redact_str(&self.secret_key))
            .field("flush_interval_secs", &self.flush_interval_secs)
            .field("max_batch_size", &self.max_batch_size)
            .field("log_inputs", &self.log_inputs)
            .field("log_outputs", &self.log_outputs)
            .finish()
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LangsmithConfig {
    pub api_key: String,
    pub project: String,
    #[serde(default)]
    pub endpoint: Option<String>,
}

impl std::fmt::Debug for LangsmithConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangsmithConfig")
            .field("api_key", &super::redact_str(&self.api_key))
            .field("project", &self.project)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatadogConfig {
    pub api_key: String,
    #[serde(default)]
    pub service: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
}

impl std::fmt::Debug for DatadogConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatadogConfig")
            .field("api_key", &super::redact_str(&self.api_key))
            .field("service", &self.service)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeliconeConfig {
    pub api_key: String,
    #[serde(default)]
    pub endpoint: Option<String>,
}

impl std::fmt::Debug for HeliconeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeliconeConfig")
            .field("api_key", &super::redact_str(&self.api_key))
            .field("endpoint", &self.endpoint)
            .finish()
    }
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
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
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

impl std::fmt::Debug for WandbConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WandbConfig")
            .field("api_key", &super::redact_str(&self.api_key))
            .field("project", &self.project)
            .field("entity", &self.entity)
            .field("tags", &self.tags)
            .field("endpoint", &self.endpoint)
            .finish()
    }
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
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
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
    /// the reference S3 logging, which persists both halves of the exchange).
    #[serde(default)]
    pub log_response_body: Option<bool>,
}

impl std::fmt::Debug for S3LogConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3LogConfig")
            .field("bucket", &self.bucket)
            .field("region", &self.region)
            .field("prefix", &self.prefix)
            .field("access_key_id", &super::redact_opt(&self.access_key_id))
            .field(
                "secret_access_key",
                &super::redact_opt(&self.secret_access_key),
            )
            .field("endpoint", &self.endpoint)
            .field("log_response_body", &self.log_response_body)
            .finish()
    }
}
