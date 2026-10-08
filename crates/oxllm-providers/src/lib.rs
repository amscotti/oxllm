//! Provider transform implementations for oxllm.
//!
//! Each provider is a module behind a Cargo feature flag. Adding a provider
//! follows the checklist in `docs/adding-a-provider.md`. Most day-to-day work
//! happens in this crate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Small cross-provider helpers. Gated on the consumers that need them:
// openai_compat hosts, bespoke embedding/rerank providers, Anthropic, and
// Bedrock (shared image/schema helpers live here so anthropic does not
// depend on the bedrock feature).
#[cfg(any(
    feature = "openai_compat",
    feature = "voyage",
    feature = "jina_ai",
    feature = "anthropic",
    feature = "bedrock",
))]
pub(crate) mod util;

// The OpenAI-compatible shape module is gated on the `openai_compat` feature,
// which every individual OpenAI-compat provider feature implies (see
// `Cargo.toml`). This replaces the prior `cfg(any(...))` list of 23 entries:
// adding a new OpenAI-compat provider is now a single `Cargo.toml` line
// (`newprov = ["openai_compat"]`) and cannot accidentally orphan the shared
// module.
#[cfg(feature = "openai_compat")]
pub mod openai_compat;

#[cfg(feature = "openai_compat")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "together_ai")]
pub mod together_ai;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "xai")]
pub mod xai;

#[cfg(feature = "mistral")]
pub mod mistral;

#[cfg(feature = "cohere")]
pub mod cohere;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "azure_ai")]
pub mod azure_ai;

#[cfg(feature = "bedrock")]
pub mod bedrock;

#[cfg(feature = "vertex_ai")]
pub mod vertex_ai;

#[cfg(feature = "databricks")]
pub mod databricks;

#[cfg(feature = "perplexity")]
pub mod perplexity;

#[cfg(feature = "sambanova")]
pub mod sambanova;

#[cfg(feature = "cloudflare")]
pub mod cloudflare;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "cerebras")]
pub mod cerebras;

#[cfg(feature = "ai21")]
pub mod ai21;

#[cfg(feature = "fireworks_ai")]
pub mod fireworks_ai;

#[cfg(feature = "deepinfra")]
pub mod deepinfra;

#[cfg(feature = "nvidia_nim")]
pub mod nvidia_nim;

#[cfg(feature = "voyage")]
pub mod voyage;

#[cfg(feature = "jina_ai")]
pub mod jina_ai;

#[cfg(feature = "watsonx")]
pub mod watsonx;

#[cfg(feature = "huggingface")]
pub mod huggingface;

#[cfg(feature = "deepgram")]
pub mod deepgram;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "dashscope")]
pub mod dashscope;

#[cfg(feature = "bedrock_mantle")]
pub mod bedrock_mantle;

#[cfg(feature = "replicate")]
pub mod replicate;

#[cfg(feature = "meta")]
pub mod meta;

#[cfg(feature = "baseten")]
pub mod baseten;

use std::sync::Arc;

use oxllm_core::ProviderRegistry;

/// Build the default [`ProviderRegistry`] from the enabled Cargo features.
///
/// The server calls this once at startup. Adding a provider to the registry
/// is one match arm here plus the `register` call — see `docs/adding-a-provider.md`.
pub fn default_registry() -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    #[cfg(feature = "openai")]
    registry.register("openai", Arc::new(openai::OpenAiProvider));
    #[cfg(feature = "anthropic")]
    registry.register("anthropic", Arc::new(anthropic::AnthropicProvider));
    #[cfg(feature = "gemini")]
    registry.register("gemini", Arc::new(gemini::PROVIDER));
    #[cfg(feature = "ollama")]
    registry.register("ollama", Arc::new(ollama::PROVIDER));
    #[cfg(feature = "groq")]
    registry.register("groq", Arc::new(groq::PROVIDER));
    #[cfg(feature = "together_ai")]
    registry.register("together_ai", Arc::new(together_ai::PROVIDER));
    #[cfg(feature = "openrouter")]
    registry.register("openrouter", Arc::new(openrouter::OpenRouterProvider));
    #[cfg(feature = "xai")]
    registry.register("xai", Arc::new(xai::PROVIDER));
    #[cfg(feature = "mistral")]
    registry.register("mistral", Arc::new(mistral::PROVIDER));
    #[cfg(feature = "cohere")]
    registry.register("cohere", Arc::new(cohere::CohereProvider));
    #[cfg(feature = "azure")]
    registry.register("azure", Arc::new(azure::AzureProvider));
    #[cfg(feature = "azure_ai")]
    registry.register("azure_ai", Arc::new(azure_ai::AzureAiProvider));
    #[cfg(feature = "bedrock")]
    registry.register("bedrock", Arc::new(bedrock::BedrockProvider));
    #[cfg(feature = "vertex_ai")]
    registry.register("vertex_ai", Arc::new(vertex_ai::VertexProvider));
    #[cfg(feature = "databricks")]
    registry.register("databricks", Arc::new(databricks::PROVIDER));
    #[cfg(feature = "perplexity")]
    registry.register("perplexity", Arc::new(perplexity::PROVIDER));
    #[cfg(feature = "sambanova")]
    registry.register("sambanova", Arc::new(sambanova::PROVIDER));
    #[cfg(feature = "cloudflare")]
    registry.register("cloudflare", Arc::new(cloudflare::PROVIDER));
    #[cfg(feature = "deepseek")]
    registry.register("deepseek", Arc::new(deepseek::PROVIDER));
    #[cfg(feature = "cerebras")]
    registry.register("cerebras", Arc::new(cerebras::PROVIDER));
    #[cfg(feature = "ai21")]
    registry.register("ai21", Arc::new(ai21::PROVIDER));
    #[cfg(feature = "fireworks_ai")]
    registry.register("fireworks_ai", Arc::new(fireworks_ai::PROVIDER));
    #[cfg(feature = "deepinfra")]
    registry.register("deepinfra", Arc::new(deepinfra::PROVIDER));
    #[cfg(feature = "nvidia_nim")]
    registry.register("nvidia_nim", Arc::new(nvidia_nim::PROVIDER));
    #[cfg(feature = "voyage")]
    registry.register("voyage", Arc::new(voyage::VoyageProvider));
    #[cfg(feature = "jina_ai")]
    registry.register("jina_ai", Arc::new(jina_ai::JinaAIProvider));
    #[cfg(feature = "watsonx")]
    registry.register("watsonx", Arc::new(watsonx::WatsonXProvider));
    #[cfg(feature = "watsonx")]
    registry.register("watsonx_text", Arc::new(watsonx::WatsonXProvider));
    #[cfg(feature = "huggingface")]
    registry.register("huggingface", Arc::new(huggingface::PROVIDER));
    #[cfg(feature = "deepgram")]
    registry.register("deepgram", Arc::new(deepgram::DeepgramProvider));
    #[cfg(feature = "elevenlabs")]
    registry.register("elevenlabs", Arc::new(elevenlabs::ElevenLabsProvider));
    #[cfg(feature = "dashscope")]
    registry.register("dashscope", Arc::new(dashscope::PROVIDER));
    #[cfg(feature = "bedrock_mantle")]
    registry.register("bedrock_mantle", Arc::new(bedrock_mantle::PROVIDER));
    #[cfg(feature = "replicate")]
    registry.register("replicate", Arc::new(replicate::PROVIDER));
    #[cfg(feature = "meta")]
    registry.register("meta", Arc::new(meta::PROVIDER));
    #[cfg(feature = "baseten")]
    registry.register("baseten", Arc::new(baseten::PROVIDER));
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_openai_registered_with_feature() {
        let registry = default_registry();
        #[cfg(feature = "openai")]
        {
            assert!(
                registry.get("openai").is_some(),
                "openai must be registered when the openai feature is enabled"
            );
        }
        #[cfg(not(feature = "openai"))]
        {
            assert!(
                registry.get("openai").is_none(),
                "openai must NOT be registered when its feature is disabled"
            );
        }
    }

    #[test]
    fn registry_replicate_presence_matches_feature_flag() {
        // Guards against the regression where the replicate registration was
        // unconditional and failed to compile when the feature was disabled.
        let registry = default_registry();
        #[cfg(feature = "replicate")]
        {
            assert!(
                registry.get("replicate").is_some(),
                "replicate must be registered when its feature is enabled"
            );
        }
        #[cfg(not(feature = "replicate"))]
        {
            assert!(
                registry.get("replicate").is_none(),
                "replicate must NOT be registered when its feature is disabled"
            );
        }
    }

    /// Guards the invariant that every provider feature flag is paired with a
    /// `register` arm in `default_registry`. A contributor who adds the module
    /// declaration + Cargo feature but forgets the `register` arm would otherwise
    /// compile cleanly and silently omit the provider. Each slug is asserted
    /// verbatim so a typo (e.g. `register("firework_ai", …)`) is also caught.
    #[test]
    fn registry_all_enabled_features_registered() {
        let registry = default_registry();
        macro_rules! check {
            ($feature:literal, $slug:literal) => {
                #[cfg(feature = $feature)]
                assert!(
                    registry.get($slug).is_some(),
                    concat!($slug, " must be registered when its feature is enabled")
                );
                #[cfg(not(feature = $feature))]
                assert!(
                    registry.get($slug).is_none(),
                    concat!(
                        $slug,
                        " must NOT be registered when its feature is disabled"
                    )
                );
            };
        }
        check!("openai", "openai");
        check!("anthropic", "anthropic");
        check!("gemini", "gemini");
        check!("ollama", "ollama");
        check!("groq", "groq");
        check!("together_ai", "together_ai");
        check!("openrouter", "openrouter");
        check!("xai", "xai");
        check!("mistral", "mistral");
        check!("cohere", "cohere");
        check!("azure", "azure");
        check!("azure_ai", "azure_ai");
        check!("bedrock", "bedrock");
        check!("vertex_ai", "vertex_ai");
        check!("databricks", "databricks");
        check!("perplexity", "perplexity");
        check!("sambanova", "sambanova");
        check!("cloudflare", "cloudflare");
        check!("deepseek", "deepseek");
        check!("cerebras", "cerebras");
        check!("ai21", "ai21");
        check!("fireworks_ai", "fireworks_ai");
        check!("deepinfra", "deepinfra");
        check!("nvidia_nim", "nvidia_nim");
        check!("voyage", "voyage");
        check!("jina_ai", "jina_ai");
        check!("watsonx", "watsonx");
        check!("watsonx", "watsonx_text");
        check!("huggingface", "huggingface");
        check!("deepgram", "deepgram");
        check!("elevenlabs", "elevenlabs");
        check!("dashscope", "dashscope");
        check!("bedrock_mantle", "bedrock_mantle");
        check!("replicate", "replicate");
        check!("meta", "meta");
        check!("baseten", "baseten");
    }

    /// Guards the invariant that the total number of registered slugs equals
    /// the number of enabled provider features plus documented aliases. The per-slug `check!` test
    /// above cannot catch a hypothetical two-features-one-slug (or
    /// duplicate-under-different-slug) collision; this does. It also pins the
    /// `default` feature array count, so dropping or adding a feature without
    /// updating both sides fails loudly.
    #[test]
    fn registry_slug_count_matches_enabled_features() {
        let registry = default_registry();
        let mut enabled = 0usize;
        macro_rules! count {
            ($feature:literal) => {
                #[cfg(feature = $feature)]
                {
                    enabled += 1;
                }
            };
        }
        count!("openai");
        count!("anthropic");
        count!("gemini");
        count!("ollama");
        count!("groq");
        count!("together_ai");
        count!("openrouter");
        count!("xai");
        count!("mistral");
        count!("cohere");
        count!("azure");
        count!("azure_ai");
        count!("bedrock");
        count!("vertex_ai");
        count!("databricks");
        count!("perplexity");
        count!("sambanova");
        count!("cloudflare");
        count!("deepseek");
        count!("cerebras");
        count!("ai21");
        count!("fireworks_ai");
        count!("deepinfra");
        count!("nvidia_nim");
        count!("voyage");
        count!("jina_ai");
        count!("watsonx");
        // Documented alias for the WatsonX chat transform.
        count!("watsonx");
        count!("huggingface");
        count!("deepgram");
        count!("elevenlabs");
        count!("dashscope");
        count!("bedrock_mantle");
        count!("replicate");
        count!("meta");
        count!("baseten");
        assert_eq!(
            registry.slugs().count(),
            enabled,
            "registered slug count must equal the number of enabled provider features and aliases \
             (a mismatch means two features share a slug, a feature is unregistered, \
             or a duplicate was registered under a second feature)"
        );
    }
}
