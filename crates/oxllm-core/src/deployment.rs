//! Deployment types — one upstream LLM endpoint and its config.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Represents one upstream LLM endpoint (one entry in `model_list`).
///
/// ```rust
/// use oxllm_core::deployment::Deployment;
///
/// let json = r#"{"model_name":"gpt-4o","oxllm_params":{"model":"openai/gpt-4o"}}"#;
/// let dep: Deployment = serde_json::from_str(json).expect("parse");
/// assert_eq!(dep.model_name, "gpt-4o");
/// assert_eq!(dep.provider_slug(), "openai");
/// ```
///
/// `Debug` is hand-written so the secret-bearing fields nested in
/// [`OxllmParams`] (credentials, base URL, and header values) are masked; `Serialize`
/// remains verbatim for config round-trip and `stable_key` hashing.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Deployment {
    /// The name clients send (e.g. `"gpt-4o"`).
    pub model_name: String,
    /// Provider config for this deployment.
    #[serde(alias = "litellm_params")]
    pub oxllm_params: OxllmParams,
    /// Optional metadata.
    #[serde(default)]
    pub model_info: Option<ModelInfo>,
}

impl Deployment {
    /// Extracts the provider slug from `oxllm_params.model`.
    ///
    /// Returns the part before the first `/`, or `"openai"` if there is no
    /// `/` or the segment before it is empty (e.g. a leading-slash model id
    /// like `"/gpt-4o"` or an empty model string). A leading slash is treated
    /// as "no provider prefix" so it defaults to `"openai"` rather than
    /// producing an empty slug (which would fail registry lookup with no hint
    /// that the config has a malformed model prefix).
    ///
    /// ```rust
    /// use oxllm_core::deployment::Deployment;
    ///
    /// let anthropic: Deployment = serde_json::from_str(
    ///     r#"{"model_name":"claude","oxllm_params":{"model":"anthropic/claude-3"}}"#,
    /// ).expect("parse");
    /// assert_eq!(anthropic.provider_slug(), "anthropic");
    ///
    /// // A model with no provider prefix defaults to "openai".
    /// let bare: Deployment = serde_json::from_str(
    ///     r#"{"model_name":"gpt-4o","oxllm_params":{"model":"gpt-4o"}}"#,
    /// ).expect("parse");
    /// assert_eq!(bare.provider_slug(), "openai");
    ///
    /// // A leading slash also defaults to "openai" (empty prefix segment).
    /// let leading: Deployment = serde_json::from_str(
    ///     r#"{"model_name":"gpt-4o","oxllm_params":{"model":"/gpt-4o"}}"#,
    /// ).expect("parse");
    /// assert_eq!(leading.provider_slug(), "openai");
    /// ```
    pub fn provider_slug(&self) -> &str {
        match self.oxllm_params.model.split_once('/') {
            Some((provider, _)) if !provider.is_empty() => provider,
            _ => "openai",
        }
    }

    /// Warn when a deployment carries AWS or Vertex credentials but
    /// resolves to a provider that ignores them. Typical cause: missing
    /// `bedrock/` / `vertex_ai/` prefix so `claude-sonnet` is routed to
    /// first-party Anthropic and 401s.
    pub fn credential_mismatch_warning(&self) -> Option<String> {
        let slug = self.provider_slug();
        let p = &self.oxllm_params;
        let aws_set = p.aws_access_key_id.is_some()
            || p.aws_secret_access_key.is_some()
            || p.aws_session_token.is_some()
            || p.extra.get("aws_region_name").is_some()
            || p.extra.get("aws_region").is_some()
            || p.extra.get("aws_role_name").is_some()
            || p.extra.get("aws_profile_name").is_some();
        let vertex_set = p.extra.get("vertex_project").is_some()
            || p.extra.get("vertex_credentials").is_some()
            || p.extra.get("vertex_location").is_some();
        if aws_set && !(slug.starts_with("bedrock") || slug.starts_with("sagemaker")) {
            return Some(format!(
                "Deployment '{}' sets AWS credentials but model='{}' resolves to \
                 provider '{slug}', which ignores them. Prefix the model with \
                 'bedrock/' (or 'sagemaker/') so the request uses those creds.",
                self.model_name, p.model
            ));
        }
        if vertex_set && !slug.starts_with("vertex_ai") {
            return Some(format!(
                "Deployment '{}' sets Vertex credentials but model='{}' resolves to \
                 provider '{slug}', which ignores them. Prefix the model with \
                 'vertex_ai/' so the request uses those creds.",
                self.model_name, p.model
            ));
        }
        None
    }
}

impl std::fmt::Debug for Deployment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Deployment")
            .field("model_name", &self.model_name)
            .field("oxllm_params", &self.oxllm_params)
            .field("model_info", &self.model_info)
            .finish()
    }
}

/// Placeholder emitted for any secret-bearing field under `Debug`.
const REDACTED: &str = "<redacted>";

/// Helper that maps `Some(secret)` → `Some(REDACTED)` and `None` → `None`,
/// so `Debug` preserves presence/absence without leaking the secret value.
fn redact(opt: &Option<String>) -> Option<&'static str> {
    opt.as_ref().map(|_| REDACTED)
}

/// Header names remain useful for debugging, but any header value can carry
/// credentials (including custom names). Borrow the names without copying
/// the secret-bearing map or inspecting its values.
struct RedactedHeaders<'a>(&'a HashMap<String, String>);

impl std::fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for name in self.0.keys() {
            map.entry(name, &REDACTED);
        }
        map.finish()
    }
}

/// `Debug` adapter over the forward-compat `extra` map that masks every
/// string leaf as [`REDACTED`] while keeping keys and non-string values
/// visible. `extra` captures every unrecognized deployment-params entry, so
/// credential-shaped params (`azure_ad_token`, `google_application_credentials`,
/// …) land here with `os.environ/` values already resolved — default-deny on
/// strings is the only listing that cannot rot as new credential params
/// appear (same stance as `PreSignedHeaders`' `SAFE_HEADER_NAMES` allowlist).
/// Non-string leaves stay visible: credentials are always strings, so
/// numeric/bool config (e.g. a custom pricing knob) remains debuggable.
struct RedactedExtra<'a>(&'a serde_json::Map<String, serde_json::Value>);

impl std::fmt::Debug for RedactedExtra<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (key, value) in self.0 {
            map.entry(key, &RedactedValue(value));
        }
        map.finish()
    }
}

/// `Debug` adapter over a single `extra` value: string leaves (at any nesting
/// depth) are masked; objects/arrays recurse; everything else prints verbatim.
struct RedactedValue<'a>(&'a serde_json::Value);

impl std::fmt::Debug for RedactedValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            serde_json::Value::String(_) => f.write_str(REDACTED),
            serde_json::Value::Object(m) => {
                let mut map = f.debug_map();
                for (key, value) in m {
                    map.entry(key, &RedactedValue(value));
                }
                map.finish()
            }
            serde_json::Value::Array(items) => {
                let mut list = f.debug_list();
                for item in items {
                    list.entry(&RedactedValue(item));
                }
                list.finish()
            }
            other => std::fmt::Debug::fmt(other, f),
        }
    }
}

/// One Anthropic prompt-cache breakpoint.
///
/// `location` is `message`. Set `role` (`system`) or a message `index`
/// (`-1` is the last message). The Anthropic transform writes
/// `cache_control: {type: ephemeral}` there when the client did not.
/// Existing caller controls take precedence and new marks stay within the
/// four-block provider limit. Missing request targets are skipped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheControlInjectionPoint {
    /// `message`. Other locations are a transform error.
    pub location: String,
    /// Message role to mark. `system` targets the split-out system block.
    #[serde(default)]
    pub role: Option<String>,
    /// Index into the source message list. Negative indexes count from the
    /// end. Missing/out-of-range request targets are skipped.
    #[serde(default)]
    pub index: Option<i64>,
}

impl CacheControlInjectionPoint {
    /// Validate the supported message selector before serving requests.
    pub fn validate(&self) -> crate::Result<()> {
        if self.location != "message" {
            return Err(crate::Error::Config(
                "cache_control_injection_points location must be message".into(),
            ));
        }
        if self.role.is_some() == self.index.is_some() {
            return Err(crate::Error::Config(
                "cache_control_injection_points entry needs exactly one of role or index".into(),
            ));
        }
        if self
            .role
            .as_deref()
            .is_some_and(|role| !matches!(role, "system" | "user" | "assistant" | "tool"))
        {
            return Err(crate::Error::Config(
                "cache_control_injection_points role must be system, user, assistant or tool"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Provider config for a deployment (the `oxllm_params` block).
///
/// `Debug` is hand-written so the secret-bearing fields (`api_key`, AWS
/// credentials, `api_base`, and `extra_headers` values) are masked as
/// `<redacted>` and every **string** value inside
/// the forward-compat `extra` map is masked the same way (unknown params can
/// be credential-shaped, e.g. `azure_ad_token`); `Serialize` remains verbatim
/// for config round-trip and `stable_key` hashing. A future
/// `tracing::debug!(?p)` or state-snapshot dump therefore cannot leak live
/// keys. The entire base URL is masked because credentials can appear in
/// userinfo, paths, or query parameters; header names remain visible.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct OxllmParams {
    /// Provider/model string like `"openai/gpt-4o"`.
    pub model: String,
    /// API key.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Replace the provider's authentication header (for example `api-key`).
    /// Non-Authorization headers default to a raw credential value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<String>,
    /// Authentication value prefix, such as `Bearer` or `Token`. `raw`,
    /// `none`, or an empty string sends the credential without a prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_scheme: Option<String>,
    /// AWS access key id for `bedrock` and `bedrock_mantle` deployments. When
    /// omitted the host falls back to the `AWS_ACCESS_KEY_ID` environment
    /// variable at request time.
    #[serde(default)]
    pub aws_access_key_id: Option<String>,
    /// AWS secret access key for `bedrock` and `bedrock_mantle` deployments.
    /// Falls back to `AWS_SECRET_ACCESS_KEY` when omitted.
    #[serde(default)]
    pub aws_secret_access_key: Option<String>,
    /// Optional AWS STS session token for short-lived `bedrock` /
    /// `bedrock_mantle` credentials. Falls back to `AWS_SESSION_TOKEN`.
    #[serde(default)]
    pub aws_session_token: Option<String>,
    /// Base URL override.
    #[serde(default)]
    pub api_base: Option<String>,
    /// Request timeout in seconds. Stored as `f64` so fractional timeouts
    /// like `timeout: 0.5` parse (Bug #49).
    #[serde(default)]
    pub timeout: Option<f64>,
    /// Stream idle timeout in seconds.
    #[serde(default)]
    pub stream_timeout: Option<u64>,
    /// Requests per minute limit.
    #[serde(default)]
    pub rpm: Option<u32>,
    /// Tokens per minute limit.
    #[serde(default)]
    pub tpm: Option<u32>,
    /// API version (for Azure/Vertex).
    #[serde(default)]
    pub api_version: Option<String>,
    /// Optional routing weight for `simple_shuffle`. Deployments
    /// with a weight are chosen proportionally; deployments without one are
    /// treated as weight `1`. When no deployment carries a weight, behavior
    /// is uniform random (the default).
    #[serde(default)]
    pub weight: Option<u32>,
    /// Optional priority-order tier: lower `order` =
    /// higher priority. When at least one candidate deployment sets
    /// `order`, the router only considers the lowest-order tier of
    /// healthy deployments; unset counts as the lowest-priority tier.
    /// See [`crate::router`] for details.
    #[serde(default)]
    pub order: Option<u32>,
    /// Optional per-deployment input cost override (USD per token). When
    /// any pricing field is set, a pricing override takes effect; missing
    /// per-token rates default from the embedded price table (per-field
    /// merge against the model's rate card).
    #[serde(default)]
    pub input_cost_per_token: Option<f64>,
    /// Optional per-deployment output cost override (USD per token).
    #[serde(default)]
    pub output_cost_per_token: Option<f64>,
    /// Optional per-deployment reasoning (thinking) output token cost
    /// override (USD per token). Falls back to `output_cost_per_token` when
    /// `None` and a pricing override is in effect — mirrors the price-table's
    /// `output_cost_per_reasoning_token` field so a deployment override can
    /// set a distinct reasoning rate.
    #[serde(default)]
    pub output_cost_per_reasoning_token: Option<f64>,
    /// Optional per-deployment cache-read input token cost override.
    /// Falls back to the price-table cache-read rate (inherited, matching
    /// built-in cache-pricing inheritance) when `None` and the
    /// model resolves in the table; otherwise the override's input rate.
    #[serde(default)]
    pub cache_read_input_token_cost: Option<f64>,
    /// Optional per-deployment cache-creation input token cost override.
    /// Falls back to the price-table cache-creation rate (inherited) when
    /// `None` and the model resolves in the table; otherwise the override's
    /// input rate.
    #[serde(default)]
    pub cache_creation_input_token_cost: Option<f64>,
    /// Optional per-deployment cost override for cache-creation input tokens
    /// that persist for 1 hour (Anthropic's 1h-ephemeral cache tier,
    /// `cache_creation_input_token_cost_above_1hr`). Falls back to the
    /// price-table 1h rate (inherited) when `None` and the model resolves in
    /// the table.
    #[serde(default)]
    pub cache_creation_input_token_cost_above_1hr: Option<f64>,
    /// Apply deployment context-tier prices at the exact threshold when true.
    /// Unset uses an exclusive boundary.
    #[serde(default)]
    pub tier_threshold_is_inclusive: Option<bool>,
    /// Optional duration price (USD per second of elapsed request time).
    /// Takes precedence over legacy input/output duration rates. Per-token
    /// overrides take precedence over every duration rate.
    #[serde(default)]
    pub cost_per_second: Option<f64>,
    /// Legacy elapsed-time rate used when `cost_per_second` is unset.
    #[serde(default)]
    pub input_cost_per_second: Option<f64>,
    /// Legacy elapsed-time rate used when canonical/input rates are unset.
    /// Duration is billed once, even when both legacy fields are set.
    #[serde(default)]
    pub output_cost_per_second: Option<f64>,
    /// Extra HTTP headers to attach to every upstream request for this
    /// deployment (e.g. `anthropic-beta`, custom tracking headers). Header
    /// names are inserted as-is and override any provider-applied header of
    /// the same name.
    #[serde(default)]
    pub extra_headers: Option<HashMap<String, String>>,
    /// Minimum estimated prompt-prefix tokens before the prompt-caching
    /// deployment check engages for this deployment. Overrides the global
    /// `oxllm_settings.prompt_caching_min_tokens`; defaults to 1024 when
    /// neither is set.
    #[serde(default)]
    pub prompt_caching_min_tokens: Option<u32>,
    /// Per-deployment spend cap (USD) over the `budget_duration` window.
    /// Enforced in the routing retry loop against the
    /// `spend:dep:{deployment_key}` counter; a deployment at or over its cap
    /// is skipped (cooled) until the window resets. Unset means no
    /// deployment-level cap.
    #[serde(default)]
    pub max_budget: Option<f64>,
    /// Budget window for `max_budget` (e.g. `"30s"`, `"1h"`, `"1d"`).
    /// Defaults to 86400 seconds (1 day) when unset.
    #[serde(default)]
    pub budget_duration: Option<String>,
    /// Optional silent (shadow-traffic) model for traffic mirroring. When
    /// set (e.g. `"openai/gpt-4o"`), the orchestration spawns a
    /// fire-and-forget task after a successful primary call to send the
    /// same request to this model. The silent model's response is logged
    /// but never returned to the client, and its spend is not charged to
    /// the caller's budget. Failures are logged at `warn` level and do not
    /// affect the primary response.
    #[serde(default)]
    pub silent_model: Option<String>,
    /// Per-deployment override of the global `drop_params` flag. `Some`
    /// wins over the global setting; `None` inherits it.
    #[serde(default)]
    pub drop_params: Option<bool>,
    /// Repair incomplete Bedrock tool/thinking history. Adjacent Converse
    /// roles are always normalized; this override inherits the global setting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modify_params: Option<bool>,
    /// Top-level keys removed before and after the provider transform, so
    /// renamed parameters are also dropped. Applied even when a key is
    /// supported or listed in `allowed_openai_params`.
    #[serde(default)]
    pub additional_drop_params: Option<Vec<String>>,
    /// Keys preserved by the supported-param strip and restored if the
    /// provider's allow-list omitted them. Additional drops still win.
    #[serde(default)]
    pub allowed_openai_params: Option<Vec<String>>,
    /// Shallow object merged over the outbound body after the provider
    /// transform. Configured values win over client/provider values; the
    /// resolved `model` is protected and additional drops still win.
    #[serde(default)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
    /// Anthropic cache breakpoints. See [`CacheControlInjectionPoint`].
    #[serde(default)]
    pub cache_control_injection_points: Option<Vec<CacheControlInjectionPoint>>,
    /// Unknown params captured for forward compatibility.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl std::fmt::Debug for OxllmParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OxllmParams")
            .field("model", &self.model)
            .field("api_key", &redact(&self.api_key))
            .field("auth_header", &self.auth_header)
            .field("auth_scheme", &self.auth_scheme)
            .field("aws_access_key_id", &redact(&self.aws_access_key_id))
            .field(
                "aws_secret_access_key",
                &redact(&self.aws_secret_access_key),
            )
            .field("aws_session_token", &redact(&self.aws_session_token))
            .field("api_base", &redact(&self.api_base))
            .field("timeout", &self.timeout)
            .field("stream_timeout", &self.stream_timeout)
            .field("rpm", &self.rpm)
            .field("tpm", &self.tpm)
            .field("api_version", &self.api_version)
            .field("weight", &self.weight)
            .field("order", &self.order)
            .field("input_cost_per_token", &self.input_cost_per_token)
            .field("output_cost_per_token", &self.output_cost_per_token)
            .field(
                "output_cost_per_reasoning_token",
                &self.output_cost_per_reasoning_token,
            )
            .field(
                "cache_read_input_token_cost",
                &self.cache_read_input_token_cost,
            )
            .field(
                "cache_creation_input_token_cost",
                &self.cache_creation_input_token_cost,
            )
            .field(
                "cache_creation_input_token_cost_above_1hr",
                &self.cache_creation_input_token_cost_above_1hr,
            )
            .field("input_cost_per_second", &self.input_cost_per_second)
            .field("output_cost_per_second", &self.output_cost_per_second)
            .field("cost_per_second", &self.cost_per_second)
            .field(
                "tier_threshold_is_inclusive",
                &self.tier_threshold_is_inclusive,
            )
            .field(
                "extra_headers",
                &self.extra_headers.as_ref().map(RedactedHeaders),
            )
            .field("prompt_caching_min_tokens", &self.prompt_caching_min_tokens)
            .field("max_budget", &self.max_budget)
            .field("budget_duration", &self.budget_duration)
            .field("silent_model", &self.silent_model)
            .field("drop_params", &self.drop_params)
            .field("modify_params", &self.modify_params)
            .field("additional_drop_params", &self.additional_drop_params)
            .field("allowed_openai_params", &self.allowed_openai_params)
            .field("extra_body", &self.extra_body.as_ref().map(RedactedExtra))
            .field(
                "cache_control_injection_points",
                &self.cache_control_injection_points,
            )
            .field("extra", &RedactedExtra(&self.extra))
            .finish()
    }
}

impl OxllmParams {
    /// Validate authentication override syntax before constructing headers.
    pub fn validate_auth_overrides(&self) -> crate::Result<()> {
        let is_token = |value: &str| {
            !value.is_empty()
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
        };
        if self
            .auth_header
            .as_deref()
            .is_some_and(|header| !is_token(header))
        {
            return Err(crate::Error::Config(
                "auth_header must be a valid HTTP header name".into(),
            ));
        }
        if self
            .auth_scheme
            .as_deref()
            .is_some_and(|scheme| !scheme.is_empty() && !is_token(scheme))
        {
            return Err(crate::Error::Config(
                "auth_scheme must be an HTTP authentication token or raw".into(),
            ));
        }
        Ok(())
    }

    /// Returns the deployment-level pricing override when **any** pricing
    /// field is set, merging per field with the embedded price table.
    ///
    /// Custom pricing copies every non-None
    /// pricing field into the model's rate card independently,
    /// so a config that overrides only `input_cost_per_token` bills input
    /// at the custom rate and output at the table rate. This mirrors that:
    /// missing input/output rates default from the price table (0.0 when
    /// the model is unlisted, with a `debug` log on the partial
    /// input/output case so the fallback is observable), and unset cache
    /// rates inherit the table's cache rates (the
    /// `_inherit_builtin_cache_pricing`) instead of falling back to the
    /// custom input rate — user-specified fields always win. A satellite-
    /// only override (reasoning/cache rates set, both base rates unset) on
    /// an unlisted model also logs at `debug`: there is nothing to
    /// inherit, so base tokens bill at 0.0.
    ///
    /// Per-second fields (`input_cost_per_second` / `output_cost_per_second`)
    /// only take effect when no per-token field is set; when both are
    /// configured the per-token rates win and a
    /// `debug` log is emitted.
    ///
    /// All of these events log at `debug` (not `warn`): this method runs
    /// **per candidate on every pick** ([`LowestCost`](crate::router::LowestCost)'s
    /// `pick` and the adaptive router's cost blending both call it), so a
    /// persistent partial/precedence configuration must not emit a warn per
    /// request — the codebase's convention for per-pick events is debug.
    pub fn pricing_override(&self) -> Option<crate::cost::DeploymentPricing> {
        let tiers = crate::cost::cost_tiers_from_extra(&self.extra);
        let tier_set = !tiers.is_empty();
        let per_token_set = self.input_cost_per_token.is_some()
            || self.output_cost_per_token.is_some()
            || self.output_cost_per_reasoning_token.is_some()
            || self.cache_read_input_token_cost.is_some()
            || self.cache_creation_input_token_cost.is_some()
            || self.cache_creation_input_token_cost_above_1hr.is_some()
            || tier_set;
        let per_second_set = self.cost_per_second.is_some()
            || self.input_cost_per_second.is_some()
            || self.output_cost_per_second.is_some();
        if !per_token_set && !per_second_set {
            return None;
        }
        if !per_token_set {
            return Some(crate::cost::DeploymentPricing {
                cost_per_second_rate: self.cost_per_second,
                input_cost_per_second: self.input_cost_per_second,
                output_cost_per_second: self.output_cost_per_second,
                ..Default::default()
            });
        }
        if per_second_set {
            // debug, not warn: this fires per candidate on every pick (see
            // the method doc) — a persistent precedence config must not
            // emit a warn per request forever.
            tracing::debug!(
                model = %self.model,
                "deployment sets both per-token and per-second custom rates; \
                 billing uses the per-token rates (LiteLLM precedence)"
            );
        }
        if self.input_cost_per_token.is_some() != self.output_cost_per_token.is_some() {
            tracing::debug!(
                model = %self.model,
                "partial per-token pricing override: exactly one of \
                 input_cost_per_token / output_cost_per_token is set; the missing \
                 rate defaults from the price table (0.0 for an unlisted model)"
            );
        }
        if self.satellite_only_zero_base_rates() {
            tracing::debug!(
                model = %self.model,
                "satellite-only pricing override on an unlisted model: \
                 input_cost_per_token / output_cost_per_token are unset and no \
                 price-table entry exists, so base tokens bill at 0.0"
            );
        }
        let table = crate::cost::lookup_pricing(&self.model);
        let table_rates = table.map(|p| (p.input_cost_per_token, p.output_cost_per_token));
        Some(crate::cost::DeploymentPricing {
            input_cost_per_token: self
                .input_cost_per_token
                .or(table_rates.map(|r| r.0))
                .unwrap_or(0.0),
            output_cost_per_token: self
                .output_cost_per_token
                .or(table_rates.map(|r| r.1))
                .unwrap_or(0.0),
            cache_read_input_token_cost: self
                .cache_read_input_token_cost
                .or(table.and_then(|p| p.cache_read_input_token_cost)),
            cache_creation_input_token_cost: self
                .cache_creation_input_token_cost
                .or(table.and_then(|p| p.cache_creation_input_token_cost)),
            cache_creation_input_token_cost_above_1hr: self
                .cache_creation_input_token_cost_above_1hr
                .or(table.and_then(|p| p.cache_creation_input_token_cost_above_1hr)),
            output_cost_per_reasoning_token: self.output_cost_per_reasoning_token,
            tiers,
            tier_threshold_is_inclusive: self.tier_threshold_is_inclusive.unwrap_or(false),
            ..Default::default()
        })
    }

    /// Whether this override sets a satellite rate (reasoning / cache rates)
    /// while both base rates are unset **and** the model has no price-table
    /// entry — the configuration where the effective override silently bills
    /// all non-satellite tokens at 0.0. Used to emit the debug log that makes
    /// the free base tokens observable; a listed model inherits table base
    /// rates instead and does not log.
    fn satellite_only_zero_base_rates(&self) -> bool {
        self.input_cost_per_token.is_none()
            && self.output_cost_per_token.is_none()
            && (self.output_cost_per_reasoning_token.is_some()
                || self.cache_read_input_token_cost.is_some()
                || self.cache_creation_input_token_cost.is_some()
                || self.cache_creation_input_token_cost_above_1hr.is_some())
            && crate::cost::lookup_pricing(&self.model).is_none()
    }
}

/// `oxllm_params` keys that stay in the flattened `extra` map because a
/// provider or credential resolver reads them. Anything else is warned at
/// load. Parsed `*_above_*_tokens` price keys are consumed separately.
pub fn is_known_oxllm_param_extra(key: &str) -> bool {
    matches!(
        key,
        "aws_region_name"
            | "aws_region"
            | "aws_bedrock_token_expires"
            | "vertex_project"
            | "vertex_credentials"
            | "vertex_location"
            | "use_converse"
            | "use_converse_api"
            | "azure_deployment"
            | "azure_ad_token"
            | "watsonx_token"
            | "watsonx_api_key"
            | "watsonx_project"
            | "organization"
            | "project"
            | "http_referer"
            | "x_title"
            | "guardrail_id"
            | "guardrailIdentifier"
            | "guardrail_version"
            | "guardrailVersion"
            | "trace"
            | "performanceConfig"
            | "performance_config"
            | "base_model"
            | "max_input_tokens"
            | "max_output_tokens"
    )
}

/// Optional metadata for a deployment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Stable id.
    #[serde(default)]
    pub id: Option<String>,
    /// Mode (e.g. `"chat"`).
    #[serde(default)]
    pub mode: Option<String>,
    /// Base model name.
    #[serde(default)]
    pub base_model: Option<String>,
    /// Tags for tag-based routing (e.g. `["canary"]`, `["stable"]`).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Environments in which this deployment is loaded. Absent means the
    /// deployment loads in every environment. Compared with
    /// `OXLLM_ENVIRONMENT`, then the legacy environment alias, defaulting to
    /// `development`.
    #[serde(default)]
    pub supported_environments: Option<Vec<String>>,
    /// Unknown fields for forward compatibility.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_slug_with_prefix() {
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(dep.provider_slug(), "openai");
    }

    #[test]
    fn test_provider_slug_without_prefix_defaults_to_openai() {
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(dep.provider_slug(), "openai");
    }

    #[test]
    fn test_provider_slug_anthropic() {
        let dep = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(dep.provider_slug(), "anthropic");
    }

    #[test]
    fn test_provider_slug_empty_segment_defaults_to_openai() {
        // A leading slash yields an empty prefix segment, which must default
        // to "openai" rather than producing an empty slug that fails registry
        // lookup with no hint about the malformed model prefix.
        let leading_or_empty = ["/x", "", "//x", "///x"];
        for model in leading_or_empty {
            let dep = Deployment {
                model_name: model.to_string(),
                oxllm_params: OxllmParams {
                    model: model.to_string(),
                    ..Default::default()
                },
                ..Default::default()
            };
            assert_eq!(
                dep.provider_slug(),
                "openai",
                "model {model:?} should default to openai"
            );
        }
    }

    #[test]
    fn test_provider_slug_trailing_slash_keeps_provider() {
        // A trailing slash leaves the prefix intact: "openai/" → "openai",
        // "anthropic/" → "anthropic".
        for (model, expected) in [("openai/", "openai"), ("anthropic/", "anthropic")] {
            let dep = Deployment {
                model_name: model.to_string(),
                oxllm_params: OxllmParams {
                    model: model.to_string(),
                    ..Default::default()
                },
                ..Default::default()
            };
            assert_eq!(
                dep.provider_slug(),
                expected,
                "model {model:?} should keep its provider prefix"
            );
        }
    }

    #[test]
    fn credential_mismatch_warns_when_aws_creds_on_anthropic_slug() {
        let dep = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-sonnet".to_string(),
                aws_access_key_id: Some("AKIATEST".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let warning = dep
            .credential_mismatch_warning()
            .expect("AWS creds on anthropic slug must warn");
        assert!(warning.contains("bedrock/"), "{warning}");
    }

    #[test]
    fn credential_mismatch_silent_when_aws_creds_on_bedrock() {
        let dep = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "bedrock/anthropic.claude-3".to_string(),
                aws_access_key_id: Some("AKIATEST".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(dep.credential_mismatch_warning().is_none());
    }

    #[test]
    fn test_extra_fields_captured() {
        let json = r#"{
 "model": "openai/gpt-4o",
 "input_cost_per_pixel": 0.000420,
 "custom_field": true
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.model, "openai/gpt-4o");
        assert!(params.extra.contains_key("input_cost_per_pixel"));
        assert!(params.extra.contains_key("custom_field"));
    }

    #[test]
    fn test_deployment_roundtrip() {
        let json = r#"{
 "model_name": "gpt-4o",
 "oxllm_params": {
 "model": "openai/gpt-4o",
 "api_key": "sk-test",
 "rpm": 480
 },
 "model_info": {
 "id": "dep-1",
 "mode": "chat"
 }
 }"#;
        let dep: Deployment = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(dep.model_name, "gpt-4o");
        assert_eq!(dep.oxllm_params.api_key.as_deref(), Some("sk-test"));
        assert_eq!(dep.oxllm_params.rpm, Some(480));
        assert_eq!(
            dep.model_info.as_ref().unwrap().id.as_deref(),
            Some("dep-1")
        );

        let reserialized = serde_json::to_string(&dep).expect("serialize should succeed");
        let reparsed: Deployment =
            serde_json::from_str(&reserialized).expect("reparse should succeed");
        assert_eq!(reparsed.model_name, dep.model_name);
    }

    #[test]
    fn test_unknown_top_level_keys_ignored() {
        let json = r#"{
 "model_name": "gpt-4o",
 "oxllm_params": { "model": "openai/gpt-4o" },
 "unknown_future_field": "ignored"
 }"#;
        let dep: Deployment = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(dep.model_name, "gpt-4o");
        assert!(dep.model_info.is_none());
    }

    #[test]
    fn test_weight_parses_when_present() {
        let json = r#"{
 "model": "openai/gpt-4o",
 "api_key": "sk-test",
 "weight": 7
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.weight, Some(7));
    }

    #[test]
    fn test_weight_defaults_to_none_when_absent() {
        let json = r#"{
 "model": "openai/gpt-4o",
 "api_key": "sk-test"
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.weight, None);
    }

    #[test]
    fn test_order_parses_when_present() {
        let json = r#"{
 "model": "openai/gpt-4o",
 "order": 2
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.order, Some(2));
        assert!(
            !params.extra.contains_key("order"),
            "named field must not also appear in extra"
        );
    }

    #[test]
    fn test_order_defaults_to_none_when_absent() {
        let json = r#"{ "model": "openai/gpt-4o" }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.order, None);
    }

    #[test]
    fn test_pricing_overrides_parse_when_present() {
        let json = r#"{
 "model": "openai/gpt-4o",
 "api_key": "sk-test",
 "input_cost_per_token": 0.000003,
 "output_cost_per_token": 0.000015,
 "cache_read_input_token_cost": 0.0000003,
 "cache_creation_input_token_cost": 0.00000375
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.input_cost_per_token, Some(0.000003));
        assert_eq!(params.output_cost_per_token, Some(0.000015));
        assert_eq!(params.cache_read_input_token_cost, Some(0.0000003));
        assert_eq!(params.cache_creation_input_token_cost, Some(0.00000375));
    }

    #[test]
    fn test_pricing_overrides_default_to_none_when_absent() {
        let json = r#"{ "model": "openai/gpt-4o" }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.input_cost_per_token, None);
        assert_eq!(params.output_cost_per_token, None);
        assert_eq!(params.cache_read_input_token_cost, None);
        assert_eq!(params.cache_creation_input_token_cost, None);
        assert!(params.pricing_override().is_none());
    }

    #[test]
    fn test_pricing_override_returns_struct_when_input_and_output_set() {
        let params = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            input_cost_per_token: Some(0.000003),
            output_cost_per_token: Some(0.000015),
            cache_read_input_token_cost: Some(0.0000003),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("override present when input+output set");
        assert!((dp.input_cost_per_token - 0.000003).abs() < 1e-15);
        assert!((dp.output_cost_per_token - 0.000015).abs() < 1e-15);
        assert_eq!(dp.cache_read_input_token_cost, Some(0.0000003));
        assert!(dp.cache_creation_input_token_cost.is_none());
        assert!(dp.output_cost_per_reasoning_token.is_none());
    }

    #[test]
    fn test_pricing_override_partial_input_only_merges_table_rates() {
        // Merge custom pricing per field: an override with only
        // `input_cost_per_token` set bills input at the custom rate and
        // output at the price-table rate (gpt-4o output = 1e-5), instead of
        // silently dropping the whole override.
        let params = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            input_cost_per_token: Some(0.000003),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("any pricing field set must produce an override");
        assert!((dp.input_cost_per_token - 0.000003).abs() < 1e-15);
        assert!((dp.output_cost_per_token - 0.00001).abs() < 1e-15);
        // Unset cache fields inherit the table rates (gpt-4o cache-read).
        assert_eq!(dp.cache_read_input_token_cost, Some(0.00000125));

        let usage = crate::hook::UsageSummary {
            prompt_tokens: 1_000,
            completion_tokens: 500,
            total_tokens: 1_500,
            ..Default::default()
        };
        let cost = crate::cost::calculate_cost_with_override("openai/gpt-4o", &usage, Some(&dp))
            .expect("override path returns Some");
        let expected = 1_000.0 * 0.000003 + 500.0 * 0.00001;
        assert!(
            (cost - expected).abs() < 1e-12,
            "input must bill at the custom rate and output at the table rate: got {cost}, expected {expected}"
        );
    }

    #[test]
    fn test_pricing_override_partial_unknown_model_defaults_zero() {
        // An unlisted model has no table rate to merge, so the missing side
        // bills at 0.0 (the override still applies to the set side).
        let params = OxllmParams {
            model: "custom/fine-tuned-model".to_string(),
            input_cost_per_token: Some(0.000003),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("override present when input set");
        assert!((dp.input_cost_per_token - 0.000003).abs() < 1e-15);
        assert_eq!(dp.output_cost_per_token, 0.0);
    }

    #[test]
    fn test_pricing_override_inherits_table_cache_rates() {
        // Built-in cache-pricing inheritance: a custom-priced
        // Anthropic deployment with no cache fields bills cache reads at
        // the table rate (3e-7), NOT the custom input rate (3e-6) — a 10x
        // over-bill on cache-heavy traffic otherwise.
        let params = OxllmParams {
            model: "anthropic/claude-sonnet-4-20250514".to_string(),
            input_cost_per_token: Some(3e-6),
            output_cost_per_token: Some(1.5e-5),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("override present when input+output set");
        assert_eq!(dp.cache_read_input_token_cost, Some(3e-7));
        assert_eq!(dp.cache_creation_input_token_cost, Some(3.75e-6));
        assert_eq!(dp.cache_creation_input_token_cost_above_1hr, Some(6e-6));

        let usage = crate::hook::UsageSummary {
            prompt_tokens: 1_100,
            completion_tokens: 0,
            total_tokens: 1_100,
            cache_read_input_tokens: Some(1_000),
            ..Default::default()
        };
        let cost = crate::cost::calculate_cost_with_override(
            "anthropic/claude-sonnet-4-20250514",
            &usage,
            Some(&dp),
        )
        .expect("override path returns Some");
        let expected = 100.0 * 3e-6 + 1_000.0 * 3e-7;
        assert!(
            (cost - expected).abs() < 1e-12,
            "cache reads must bill at the inherited table rate: got {cost}, expected {expected}"
        );
    }

    #[test]
    fn test_pricing_override_explicit_cache_rate_wins_over_inheritance() {
        let params = OxllmParams {
            model: "anthropic/claude-sonnet-4-20250514".to_string(),
            input_cost_per_token: Some(3e-6),
            output_cost_per_token: Some(1.5e-5),
            cache_read_input_token_cost: Some(1e-7),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("override present when input+output set");
        assert_eq!(dp.cache_read_input_token_cost, Some(1e-7));
        assert_eq!(
            dp.cache_creation_input_token_cost,
            Some(3.75e-6),
            "unset siblings still inherit from the table"
        );
    }

    /// LOW (C10): `cache_creation_input_token_cost_above_1hr` is a
    /// custom-pricing field, so a migrated config
    /// setting it must take effect instead of silently landing in `extra`.
    #[test]
    fn test_pricing_override_explicit_1hr_cache_rate_wins_over_inheritance() {
        let params = OxllmParams {
            model: "anthropic/claude-sonnet-4-20250514".to_string(),
            input_cost_per_token: Some(3e-6),
            output_cost_per_token: Some(1.5e-5),
            cache_creation_input_token_cost_above_1hr: Some(1e-5),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("override present when input+output set");
        assert_eq!(dp.cache_creation_input_token_cost_above_1hr, Some(1e-5));
        assert_eq!(
            dp.cache_creation_input_token_cost,
            Some(3.75e-6),
            "unset siblings still inherit from the table"
        );
    }

    #[test]
    fn test_1hr_cache_rate_parses_and_not_in_extra() {
        let json = r#"{
          "model": "anthropic/claude-sonnet-4-20250514",
          "input_cost_per_token": 0.000003,
          "output_cost_per_token": 0.000015,
          "cache_creation_input_token_cost_above_1hr": 0.00001
          }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(
            params.cache_creation_input_token_cost_above_1hr,
            Some(0.00001)
        );
        assert!(
            !params
                .extra
                .contains_key("cache_creation_input_token_cost_above_1hr"),
            "named field must not also appear in extra"
        );
    }

    /// LOW (C10): a satellite-only override (reasoning rate set, base rates
    /// unset) on an **unlisted** model has nothing to inherit, so base tokens
    /// bill at 0.0 — the debug predicate must flag this configuration.
    #[test]
    fn test_satellite_only_override_on_unlisted_model_flags_and_bills_zero_base() {
        let params = OxllmParams {
            model: "custom/fine-tuned-model".to_string(),
            output_cost_per_reasoning_token: Some(0.000012),
            ..Default::default()
        };
        assert!(
            params.satellite_only_zero_base_rates(),
            "satellite-only override on an unlisted model must be flagged"
        );
        let dp = params
            .pricing_override()
            .expect("satellite field set must produce an override");
        assert_eq!(dp.input_cost_per_token, 0.0);
        assert_eq!(dp.output_cost_per_token, 0.0);
        assert_eq!(dp.output_cost_per_reasoning_token, Some(0.000012));
    }

    /// LOW (C10): the same satellite-only config on a **listed** model
    /// inherits table base rates and must not be flagged.
    #[test]
    fn test_satellite_only_override_on_listed_model_inherits_table_base_rates() {
        let params = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            output_cost_per_reasoning_token: Some(0.000012),
            ..Default::default()
        };
        assert!(
            !params.satellite_only_zero_base_rates(),
            "listed model inherits table base rates — no debug log"
        );
        let dp = params
            .pricing_override()
            .expect("satellite field set must produce an override");
        assert!((dp.input_cost_per_token - 2.5e-6).abs() < 1e-15);
        assert!((dp.output_cost_per_token - 1e-5).abs() < 1e-15);
    }

    /// LOW (C10): a full base-rate override on an unlisted model (both input
    /// and output set) does not trip the satellite-only debug log — the 0.0
    /// fallback only applies to rates the operator left unset.
    #[test]
    fn test_full_override_on_unlisted_model_does_not_trip_satellite_debug() {
        let params = OxllmParams {
            model: "custom/fine-tuned-model".to_string(),
            input_cost_per_token: Some(3e-6),
            output_cost_per_token: Some(1.5e-5),
            ..Default::default()
        };
        assert!(!params.satellite_only_zero_base_rates());
    }

    #[test]
    fn test_pricing_override_propagates_reasoning_rate() {
        // A deployment override can now set a distinct reasoning-token rate
        // (previously always fell back to the output rate via Default).
        let params = OxllmParams {
            model: "openai/o1".to_string(),
            input_cost_per_token: Some(0.000003),
            output_cost_per_token: Some(0.000015),
            output_cost_per_reasoning_token: Some(0.000012),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("override present when input+output set");
        assert_eq!(dp.output_cost_per_reasoning_token, Some(0.000012));
    }

    #[test]
    fn test_reasoning_override_parses_and_not_in_extra() {
        let json = r#"{
  "model": "openai/o1",
  "input_cost_per_token": 0.000003,
  "output_cost_per_token": 0.000015,
  "output_cost_per_reasoning_token": 0.000012
  }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.output_cost_per_reasoning_token, Some(0.000012));
        assert!(
            !params.extra.contains_key("output_cost_per_reasoning_token"),
            "named field must not also appear in extra"
        );
    }

    #[test]
    fn test_per_second_pricing_fields_parse_when_present() {
        let json = r#"{
 "model": "sagemaker/my-duration-model",
 "input_cost_per_second": 0.000420,
 "output_cost_per_second": 0.000180
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.input_cost_per_second, Some(0.000420));
        assert_eq!(params.output_cost_per_second, Some(0.000180));
        assert!(
            !params.extra.contains_key("input_cost_per_second"),
            "named field must not also appear in extra"
        );
        assert!(!params.extra.contains_key("output_cost_per_second"));
    }

    #[test]
    fn test_per_second_fields_default_to_none_when_absent() {
        let json = r#"{ "model": "openai/gpt-4o" }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.input_cost_per_second, None);
        assert_eq!(params.output_cost_per_second, None);
    }

    #[test]
    fn test_pricing_override_per_second_only() {
        // Custom pricing carries per-second rates; a
        // per-second-only override bills duration, not tokens.
        let params = OxllmParams {
            model: "sagemaker/duration-model".to_string(),
            input_cost_per_second: Some(0.00042),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("per-second field set must produce an override");
        assert!(dp.is_per_second_priced());
        assert_eq!(dp.input_cost_per_token, 0.0);
        assert_eq!(dp.output_cost_per_token, 0.0);
        let cost = dp
            .cost_per_second(2500.0)
            .expect("per-second override bills via duration");
        assert!((cost - 0.00042 * 2.5).abs() < 1e-15, "got {cost}");
    }

    #[test]
    fn test_pricing_override_per_token_wins_over_per_second() {
        // Pricing precedence: when both per-token and per-second custom
        // rates are configured, the per-token rates bill and the per-second
        // fields are dropped from the effective override.
        let params = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            input_cost_per_token: Some(0.000003),
            output_cost_per_token: Some(0.000015),
            input_cost_per_second: Some(0.00042),
            ..Default::default()
        };
        let dp = params
            .pricing_override()
            .expect("override present when per-token fields set");
        assert!(!dp.is_per_second_priced(), "per-token rates must win");
        assert!((dp.input_cost_per_token - 0.000003).abs() < 1e-15);
    }

    #[test]
    fn test_pricing_overrides_not_in_extra_map() {
        let json = r#"{
 "model": "openai/gpt-4o",
 "input_cost_per_token": 0.000003,
 "output_cost_per_token": 0.000015
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert!(
            !params.extra.contains_key("input_cost_per_token"),
            "named field must not also appear in extra"
        );
        assert!(
            !params.extra.contains_key("output_cost_per_token"),
            "named field must not also appear in extra"
        );
    }

    #[test]
    fn test_extra_headers_parse_when_present() {
        let json = r#"{
 "model": "anthropic/claude-3-5-sonnet",
 "extra_headers": {
 "anthropic-beta": "prompt-caching-2024-07-31,pdfs-2024-09-25",
 "x-custom": "track-me"
 }
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        let headers = params
            .extra_headers
            .as_ref()
            .expect("extra_headers should be present");
        assert_eq!(
            headers.get("anthropic-beta").map(String::as_str),
            Some("prompt-caching-2024-07-31,pdfs-2024-09-25")
        );
        assert_eq!(
            headers.get("x-custom").map(String::as_str),
            Some("track-me")
        );
    }

    #[test]
    fn test_extra_headers_default_none_when_absent() {
        let json = r#"{ "model": "openai/gpt-4o" }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert!(params.extra_headers.is_none());
    }

    #[test]
    fn test_extra_headers_not_in_extra_map() {
        let json = r#"{
 "model": "anthropic/claude-3-5-sonnet",
 "extra_headers": { "anthropic-beta": "pdfs-2024-09-25" }
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert!(
            !params.extra.contains_key("extra_headers"),
            "named field must not also appear in extra"
        );
    }

    #[test]
    fn test_max_budget_fields_parse_when_present() {
        let json = r#"{
 "model": "openai/gpt-4o",
 "max_budget": 25.5,
 "budget_duration": "1d"
 }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.max_budget, Some(25.5));
        assert_eq!(params.budget_duration.as_deref(), Some("1d"));
        assert!(
            !params.extra.contains_key("max_budget"),
            "named field must not also appear in extra"
        );
        assert!(!params.extra.contains_key("budget_duration"));
    }

    #[test]
    fn test_max_budget_fields_default_to_none_when_absent() {
        let json = r#"{ "model": "openai/gpt-4o" }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.max_budget, None);
        assert_eq!(params.budget_duration, None);
    }

    #[test]
    fn test_silent_model_parses_when_present() {
        let json = r#"{
  "model": "openai/gpt-4o",
  "silent_model": "anthropic/claude-3-5-sonnet"
  }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(
            params.silent_model.as_deref(),
            Some("anthropic/claude-3-5-sonnet")
        );
        assert!(
            !params.extra.contains_key("silent_model"),
            "named field must not also appear in extra"
        );
    }

    #[test]
    fn test_silent_model_defaults_to_none_when_absent() {
        let json = r#"{ "model": "openai/gpt-4o" }"#;
        let params: OxllmParams = serde_json::from_str(json).expect("parse should succeed");
        assert_eq!(params.silent_model, None);
    }

    #[test]
    fn test_oxllm_params_debug_redacts_secret_fields() {
        // A live key in any secret-bearing field must never appear in the
        // `Debug` output — a future `tracing::debug!(?params)` or a state
        // snapshot must not leak it.
        let params = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            api_key: Some("sk-live-key-12345".to_string()),
            aws_access_key_id: Some("AKIA-LIVE-12345".to_string()),
            aws_secret_access_key: Some("super-secret-98765".to_string()),
            aws_session_token: Some("session-tok-abcdef".to_string()),
            ..Default::default()
        };
        let s = format!("{params:?}");
        assert!(
            !s.contains("sk-live-key-12345"),
            "api_key leaked into Debug: {s}"
        );
        assert!(
            !s.contains("AKIA-LIVE-12345"),
            "aws_access_key_id leaked into Debug: {s}"
        );
        assert!(
            !s.contains("super-secret-98765"),
            "aws_secret_access_key leaked into Debug: {s}"
        );
        assert!(
            !s.contains("session-tok-abcdef"),
            "aws_session_token leaked into Debug: {s}"
        );
        // The masked marker proves the field was present (vs absent).
        assert!(
            s.contains("<redacted>"),
            "expected <redacted> marker in Debug output: {s}"
        );
        // Non-secret fields remain visible.
        assert!(
            s.contains("openai/gpt-4o"),
            "model should remain visible in Debug: {s}"
        );
    }

    #[test]
    fn test_oxllm_params_debug_distinguishes_present_absent_secrets() {
        let present = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            api_key: Some("sk-live".to_string()),
            ..Default::default()
        };
        let absent = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            ..Default::default()
        };
        let present_dbg = format!("{present:?}");
        let absent_dbg = format!("{absent:?}");
        // `Some` → Some("<redacted>"); `None` → None. The two Debug outputs
        // must differ so an operator can still tell presence from absence.
        assert!(present_dbg.contains("Some("));
        assert!(absent_dbg.contains("None"));
    }

    #[test]
    fn debug_redacts_headers_and_credential_bearing_base_urls() {
        for api_base in [
            "https://fixture-user:fixture-password@example.invalid/v1",
            "https://example.invalid/v1?key=fixture-query-token",
            "https://example.invalid/fixture-path-token/v1",
            "malformed fixture-url-token",
        ] {
            let params = OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_base: Some(api_base.to_string()),
                extra_headers: Some(HashMap::from([
                    (
                        "Authorization".to_string(),
                        "Bearer fixture-header-token".to_string(),
                    ),
                    (
                        "X-Custom-Credential".to_string(),
                        "fixture-custom-token".to_string(),
                    ),
                ])),
                ..Default::default()
            };
            let dep = Deployment {
                model_name: "chat".to_string(),
                oxllm_params: params,
                ..Default::default()
            };
            for debug in [format!("{:?}", dep.oxllm_params), format!("{dep:#?}")] {
                assert!(
                    !debug.contains("fixture-"),
                    "credential-bearing value leaked"
                );
                assert!(debug.contains("<redacted>"));
                assert!(debug.contains("Authorization"));
                assert!(debug.contains("X-Custom-Credential"));
                assert!(debug.contains("openai/gpt-4o"));
            }

            // Debug redaction must not mutate runtime config or serialization.
            let value = serde_json::to_value(&dep.oxllm_params).expect("serialize");
            assert_eq!(value["api_base"], api_base);
            assert_eq!(
                value["extra_headers"]["Authorization"],
                "Bearer fixture-header-token"
            );
            assert_eq!(
                value["extra_headers"]["X-Custom-Credential"],
                "fixture-custom-token"
            );
        }
    }

    #[test]
    fn debug_distinguishes_absent_and_empty_extra_headers() {
        let mut params = OxllmParams::default();
        assert!(format!("{params:?}").contains("extra_headers: None"));
        assert!(format!("{params:?}").contains("api_base: None"));
        params.extra_headers = Some(HashMap::new());
        params.api_base = Some(String::new());
        let debug = format!("{params:?}");
        assert!(debug.contains("extra_headers: Some({})"));
        assert!(debug.contains("api_base: Some(\"<redacted>\")"));
    }

    #[test]
    fn test_deployment_debug_redacts_nested_secrets() {
        // The Deployment Debug output delegates to OxllmParams, so a live key
        // nested under oxllm_params.api_key must not leak.
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-deployment-live".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let s = format!("{dep:?}");
        assert!(
            !s.contains("sk-deployment-live"),
            "nested api_key leaked into Deployment Debug: {s}"
        );
        assert!(s.contains("<redacted>"));
        assert!(s.contains("gpt-4o"));
    }

    #[test]
    fn test_oxllm_params_serialize_keeps_secret_verbatim() {
        // Serialize must remain verbatim (config round-trip + stable_key
        // hashing); only Debug is redacted.
        let params = OxllmParams {
            model: "openai/gpt-4o".to_string(),
            api_key: Some("sk-live-keep".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&params).expect("serialize");
        assert!(
            json.contains("sk-live-keep"),
            "Serialize must keep the secret verbatim: {json}"
        );
    }

    /// MEDIUM (C10): `extra` captures every unrecognized deployment-params
    /// entry — including credential-shaped ones like `azure_ad_token`, with
    /// `os.environ/` values already resolved at config load. The hand-written
    /// `Debug`'s documented "cannot leak live keys" guarantee must hold for
    /// those too: string values inside `extra` are masked, keys stay visible.
    #[test]
    fn test_debug_redacts_credential_shaped_values_in_extra() {
        let json = r#"{
          "model_name": "gpt-4o",
          "litellm_params": {
            "model": "azure/gpt-4o",
            "azure_ad_token": "AAD-LIVE-TOKEN-12345"
          }
        }"#;
        let dep: Deployment = serde_json::from_str(json).expect("parse should succeed");
        let s = format!("{dep:?}");
        assert!(
            !s.contains("AAD-LIVE-TOKEN-12345"),
            "azure_ad_token leaked into Debug via extra: {s}"
        );
        assert!(
            s.contains("azure_ad_token"),
            "extra keys must stay visible so structure remains debuggable: {s}"
        );
        assert!(s.contains("<redacted>"));
    }

    /// MEDIUM (C10): nested string leaves inside `extra` values (e.g. a
    /// service-account blob with an embedded private key) are masked at any
    /// depth; non-string leaves (numeric pricing knobs) stay visible so
    /// benign config remains debuggable.
    #[test]
    fn test_debug_redacts_nested_strings_in_extra_but_keeps_non_strings() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "google_application_credentials".to_string(),
            serde_json::json!({"client_email": "svc@proj.iam", "private_key": "PEM-LIVE-KEY"}),
        );
        extra.insert(
            "input_cost_per_pixel".to_string(),
            serde_json::json!(0.00042),
        );
        let params = OxllmParams {
            model: "vertex_ai/gemini-pro".to_string(),
            extra,
            ..Default::default()
        };
        let s = format!("{params:?}");
        assert!(
            !s.contains("PEM-LIVE-KEY"),
            "nested private_key leaked into Debug: {s}"
        );
        assert!(
            !s.contains("svc@proj.iam"),
            "nested credential string leaked into Debug: {s}"
        );
        assert!(
            s.contains("0.00042"),
            "numeric extra values must remain visible: {s}"
        );
    }

    /// MEDIUM (C10): `Serialize` must remain verbatim for `extra` (config
    /// round-trip), even though `Debug` masks it.
    #[test]
    fn test_serialize_keeps_extra_verbatim_while_debug_redacts() {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "azure_ad_token".to_string(),
            serde_json::json!("AAD-KEEP-VERBATIM"),
        );
        let params = OxllmParams {
            model: "azure/gpt-4o".to_string(),
            extra,
            ..Default::default()
        };
        let json = serde_json::to_string(&params).expect("serialize");
        assert!(
            json.contains("AAD-KEEP-VERBATIM"),
            "Serialize must keep extra verbatim: {json}"
        );
        assert!(!format!("{params:?}").contains("AAD-KEEP-VERBATIM"));
    }
}
