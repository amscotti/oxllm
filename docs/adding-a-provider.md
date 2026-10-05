# Adding a provider to oxllm

Checklist for adding a new LLM provider. It should take minutes, not hours, and
must **not** require changes to `oxllm-server`. (Adding a new price shard
requires one line in `oxllm-core/src/cost.rs` — see step 5.)

See [architecture.md](architecture.md) for how providers fit into the workspace.

## Steps

1. **Create the provider module**
   `crates/oxllm-providers/src/<provider>/mod.rs`

   Implement the relevant traits from `oxllm-core` (e.g. `ChatProvider`,
   `EmbeddingProvider`). For each capability you support:

   - `endpoint_url` — build the upstream URL from a base
   - `apply_auth` — add auth header(s) from `ResolvedCredentials`
   - `transform_request` — OpenAI request → provider-native request
   - `transform_response` — provider response → OpenAI response
   - `transform_stream_chunk` — one SSE chunk → OpenAI chunk

2. **Add the feature flag**  
   In `crates/oxllm-providers/Cargo.toml`, add `"<provider>" = []` under
   `[features]`. Include it in defaults only if you want it always built.

3. **Wire the module** in `crates/oxllm-providers/src/lib.rs`:

   ```rust
   #[cfg(feature = "<provider>")]
   pub mod <provider>;
   ```

   Register the provider in the default registry builder the same way existing
   providers are registered.

   > **OpenAI-compatible providers:** if your provider speaks the OpenAI wire
   > format natively (auth header + base URL differ, request/response shapes
   > don't), reuse `openai_compat`. Declare your feature as
   > `"<provider>" = ["openai_compat"]` in `Cargo.toml` `[features]` (see the
   > existing `groq`, `deepseek`, … entries). The shared module is gated on
   > that single `openai_compat` flag in `lib.rs` and is compiled
   > automatically whenever any provider implies it — there is **no
   > `cfg(any(...))` list to edit**. Each compat provider feature turns the
   > shared module on, so adding one is a one-line change and cannot orphan
   > `openai_compat`.
   >
   > Bespoke (non-compat) providers that want the `crate::util` helpers
   > (e.g. `apply_bearer_auth`) hit the one remaining hand-maintained gate:
   > add your feature to the `#[cfg(any(...))]` on `pub(crate) mod util;` in
   > `lib.rs`. Today only `voyage` and `jina_ai` need this; every other
   > bespoke consumer implies `openai_compat` already and needs nothing here.
   >
   > New providers expose a plain module-level const and register it directly:
   >
   > ```rust
   > // crates/oxllm-providers/src/<provider>/mod.rs
   > pub const PROVIDER: OpenAiCompatProvider = OpenAiCompatProvider::new("<provider>");
   >
   > // crates/oxllm-providers/src/lib.rs
   > registry.register("<provider>", Arc::new(<provider>::PROVIDER));
   > ```
   >
   > (An older convention prefixed the type with `X` and stripped the prefix
   > at registration; that is no longer used — new providers use the plain
   > `PROVIDER` const form above.)

4. **Add unit tests**  
   Prefer `crates/oxllm-providers/src/<provider>/tests.rs` and golden files under
   `goldens/` when shapes are stable. Cover: request shape, response shape,
   stream chunk, param filtering, missing fields, bad input.

5. **Price table** (if needed)  
   Add or update rows in `crates/oxllm-core/data/prices/<provider>.toml`
   (and register the shard in `PRICE_SHARDS` in `crates/oxllm-core/src/cost.rs`
   if the provider file is new). This is the one place adding a provider does
   touch `oxllm-core`: `PRICE_SHARDS` is a `&[(&str, &str)]` table of
   `(provider_slug, include_str!("../data/prices/<slug>.toml"))` rows, so a new
   provider adds exactly one `("<provider>", include_str!(...))` entry. It is a
   data-only change (no logic, no new traits) and the table is kept in
   alphabetical order.

6. **Passthrough support** (optional)  
   If the provider should be reachable via the raw passthrough proxy
   (`/{provider}/{path}`), add its default base URL to the
   `default_base_for_provider` table in
   `crates/oxllm-server/src/pipeline/passthrough.rs`. Without an entry,
   passthrough requests must carry an explicit `api_base` (config or request)
   or they will be rejected. Providers served via `openai_compat` usually
   share the OpenAI base; native-shape providers need their own entry.

7. **Done.** No edits to `oxllm-server` handlers, routes, or `oxllm-core`
   traits/router for a normal chat/embeddings provider.

## Reference implementation

See `crates/oxllm-providers/src/openai/mod.rs` — near-identity transforms because
oxllm’s client-facing format is OpenAI. OpenAI-compatible hosts often reuse
`openai_compat`.

## When you need more than a provider module

A new **endpoint kind** (e.g. a first-class `rerank` product surface with typed
traits end-to-end) is a different change: it touches `oxllm-core` types/traits
and server routes. Endpoint kinds change rarely; providers change often. That
asymmetry is intentional.

## Related

- [Architecture](architecture.md)
- Sample configs: [`examples/`](../examples/)
