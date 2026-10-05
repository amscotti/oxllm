//! Per-domain repositories: the query methods of [`crate::Database`], split
//! into one sibling module per entity. Rust allows `impl Database` blocks in
//! any module of the same crate, so the public API is unchanged — callers
//! still reach every method as `db.create_key()`, `db.query_spend()`, etc.

mod budgets;
mod customers;
mod keys;
mod models;
mod orgs;
mod projects;
mod scopes;
pub(crate) mod skills;
mod spend;
mod tags;
mod teams;
mod users;
