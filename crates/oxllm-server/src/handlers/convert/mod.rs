//! API-shape conversion modules for the thin request handlers.
//!
//! Each submodule owns the pure JSON data-transformation logic for one wire
//! shape (Anthropic Messages, OpenAI Responses), keeping the handler files
//! thin: they parse the request, delegate to [`crate::handlers::chat`], and
//! convert the response back using these helpers.
//!
//! Nothing in this module touches axum, tokio, or SSE transport — it maps JSON
//! shapes only, so it can be unit-tested in isolation.

pub(crate) mod anthropic;
pub(crate) mod responses;
