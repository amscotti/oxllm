//! Stable parameter types for the pinned sqlx Any adapter.

use sqlx::any::AnyArguments;
use sqlx::query::Query;

/// Keep nullable monetary parameters FLOAT8 on PostgreSQL, including the first
/// execution of a cached statement. sqlx 0.8.6's AnyArguments::convert_to swaps
/// the Real/Double typed-NULL conversions: an ordinary None::<f64> becomes a
/// FLOAT4 parameter, while None::<f32> becomes FLOAT8. Non-NULL values must stay
/// f64; only the NULL marker uses the workaround. SQLite still receives NULL.
/// Revisit this adapter when upgrading the locked sqlx version.
pub(crate) fn bind_optional_f64<'q>(
    query: Query<'q, sqlx::Any, AnyArguments<'q>>,
    value: Option<f64>,
) -> Query<'q, sqlx::Any, AnyArguments<'q>> {
    match value {
        Some(value) => query.bind(value),
        None => query.bind(Option::<f32>::None),
    }
}
