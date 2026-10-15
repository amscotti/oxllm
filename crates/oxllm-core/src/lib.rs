//! Core types, traits, and pure logic for the oxllm proxy gateway.
//!
//! This crate is **pure**: no network, no filesystem, no env, no DB.
//! All translation and routing logic lives here, fully unit-testable.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cache;
pub mod cost;
pub mod deployment;
pub mod endpoint;
pub mod error;
pub mod hook;
pub mod limits;
pub mod provider;
pub mod redaction;
pub mod registry;
pub mod router;
pub mod tokens;

pub use cache::{Cache, IncrResult};
pub use cost::{
    DeploymentPricing, ModelPricing, TokenRates, calculate_cost, calculate_cost_for_service_tier,
    calculate_cost_with_override, calculate_cost_with_override_for_provider,
    calculate_cost_with_provider, lookup_pricing, lookup_pricing_with_provider, price_table,
    resolve_service_tier, response_service_tier,
};
pub use deployment::{Deployment, ModelInfo, OxllmParams};
pub use endpoint::Endpoint;
pub use error::{Error, Result};
pub use hook::{
    CallContext, Hook, HookAction, RateLimitInfo, ResponseSummary, ServiceFailureEvent,
    UsageSummary,
};
pub use limits::{AllScopeLimits, ScopeLimits};
pub use provider::{
    ChatProvider, EmbeddingProvider, EmbeddingRequest, EmbeddingRequestBatch, EmbeddingRequestPlan,
    Provider, ProviderId, RealtimeProvider, RerankProvider, ResolvedCredentials,
};
pub use registry::ProviderRegistry;
pub use router::{
    AdaptiveRouter, AdaptiveRouterConfig, BanditStore, BetaCell, ComplexityRouter,
    ComplexityRouterConfig, CooldownStore, FallbackRule, LatencyBased, LatencyStore, LeastBusy,
    LowestCost, LowestLatency, PreCallCheck, PreCallData, QualityRouter, RequestContext,
    RequestType, Router, RouterConfig, RoutingStrategy, RoutingStrategyKind, SessionAffinity,
    SimpleShuffle, TagBased, Tier, UsageBased, UsageStore,
};
pub use tokens::{estimate_request_tokens, estimate_tokens, tokenizer_type_for, warm_tokenizers};
