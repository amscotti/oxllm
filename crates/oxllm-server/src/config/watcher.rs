//! Config hot-reload via filesystem watching.
//!
//! Uses the `notify` crate to watch the config file. On change (debounced),
//! reloads the config, rebuilds `AppState`, and swaps atomically via `ArcSwap`.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use notify::EventKind;
use notify::Watcher;

use crate::state::AppState;

/// Debounce window: wait this long after the last file event before reloading.
const DEBOUNCE_MS: u64 = 500;

/// Closure type that rebuilds `AppState` from the current one on hot-reload.
/// Receives the previous state so infrastructure connections (DB, cache,
/// spend writer, HTTP client) can be preserved across the swap.
///
/// Returns a boxed [`Future`] so the rebuild may perform async I/O (notably
/// the config file read via `tokio::fs`); doing the read synchronously inside
/// `do_reload` would block a tokio worker thread for the read duration, which
/// matters on slow NFS config mounts. The future borrows the previous state
/// for the lifetime of the rebuild (the borrow ends before the future
/// resolves, since `reload_from_config` is synchronous).
pub type Rebuild = Arc<
    dyn Fn(
            &AppState,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<AppState>> + Send + '_>,
        > + Send
        + Sync,
>;

/// Spawn a background task that watches the config file and reloads the
/// `AppState` atomically when it changes.
///
/// The `rebuild` closure receives the current `AppState` so it can preserve
/// infrastructure connections (DB, cache, spend writer, HTTP client) across
/// the swap.
pub fn spawn_watcher(config_path: PathBuf, shared: Arc<ArcSwap<AppState>>, rebuild: Rebuild) {
    tokio::spawn(async move {
        watch_loop(config_path, shared, rebuild, None).await;
    });
}

async fn watch_loop(
    config_path: PathBuf,
    shared: Arc<ArcSwap<AppState>>,
    rebuild: Rebuild,
    ready: Option<tokio::sync::oneshot::Sender<()>>,
) {
    // Canonicalize the config path so relevance checks match the path form
    // delivered by the OS watcher. On macOS in particular, `notify`'s
    // FSEvents backend resolves symlinks (e.g. `/var` → `/private/var`), so
    // a `config_path` under `std::env::temp_dir()` arrives back from the
    // watcher with a different prefix; comparing the raw path against event
    // paths would then never match. Falling back to the original path keeps
    // the watcher functional when canonicalization fails (e.g. a transient
    // ENOENT during a rename).
    let config_path = absolute_path(config_path);

    let (tx, mut rx) = tokio::sync::mpsc::channel::<notify::Result<notify::Event>>(64);

    let mut watcher =
        match notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            let _ = tx.blocking_send(res);
        }) {
            Ok(w) => w,
            Err(e) => {
                tracing::error!(error = %e, "failed to create config file watcher");
                return;
            }
        };

    let mut files = watched_config_files(&config_path).await;
    let mut watched_parents = HashSet::new();
    if !ensure_parents_watched(&mut watcher, &files, &mut watched_parents) {
        return;
    }

    tracing::info!(
        path = %config_path.display(),
        files = files.len(),
        "config hot-reload watcher started"
    );
    if let Some(ready) = ready {
        let _ = ready.send(());
    }

    let debounce = Duration::from_millis(DEBOUNCE_MS);

    loop {
        tokio::select! {
        maybe_event = rx.recv() => {
        match maybe_event {
        Some(Ok(event)) => {
        if is_relevant_to_any(&event, &files) {
        // Real debounce: drain the channel for the full window so a
        // burst of N events produces exactly one reload, not N.
        drain_during(&mut rx, debounce, &files).await;
        // Snapshot the on-disk content BEFORE the rebuild: a write landing
        // after the rebuild closure reads the file (but before we regain
        // control) is invisible to that rebuild, so we must detect it by
        // comparing content fingerprints afterwards.
        let mut last = fingerprint_all(&files).await;
        // Only drain when the reload succeeds: if the rebuild failed (e.g.
        // it read a partially-written file), the event signaling the
        // *completed* write may still be queued; dropping it would silently
        // miss the config change until the next FS event, so leave it
        // queued for the outer loop to retry.
        if do_reload(&shared, &rebuild).await.is_ok() {
            // The rebuild future awaited config I/O (and DNS validation)
            // above; FS events that arrived in the meantime are still queued
            // and would each trigger another full debounce + AppState
            // rebuild (K8s ConfigMap `..data` swaps fire several events per
            // update). Draining non-blockingly keeps exactly one reload per
            // change burst — but only for changes the rebuild already
            // consumed. A write landing after the rebuild's file read is
            // silently lost if its only trace is the drained event, so
            // re-fingerprint afterwards and reload once more when the file
            // changed. Bounded so a writer continuously rewriting the
            // config cannot spin this loop forever.
            while rx.try_recv().is_ok() {}
            files = watched_config_files(&config_path).await;
            let _ = ensure_parents_watched(&mut watcher, &files, &mut watched_parents);
            for _ in 0..3 {
                let now = fingerprint_all(&files).await;
                if now == last {
                    break;
                }
                last = now;
                if do_reload(&shared, &rebuild).await.is_err() {
                    // e.g. a torn read of another in-flight write: stop
                    // here rather than spin; a subsequent FS event (or the
                    // next retry iteration's fingerprint change) retries.
                    break;
                }
                files = watched_config_files(&config_path).await;
                let _ = ensure_parents_watched(&mut watcher, &files, &mut watched_parents);
                while rx.try_recv().is_ok() {}
            }
        }
        }
        }
        Some(Err(e)) => {
        tracing::warn!(error = %e, "config watcher event error");
        }
        None => {
        tracing::warn!("config watcher channel closed, stopping");
        return;
        }
        }
        }
        }
    }
}

/// Drain any events that arrive within `window`, returning once the window
/// has elapsed with no pending events. Each *relevant* event (per
/// [`is_relevant_event`] against `config_path`) *extends* the deadline by the
/// full window, so the wait is measured from the LAST event on the config
/// file: spaced-out writes (each within `window` of the previous) coalesce
/// into a single reload instead of triggering multiple full AppState
/// rebuilds. All subsequent events are discarded: the caller is going to
/// reload anyway, so coalescing them avoids N redundant reloads when many FS
/// events fire in quick succession (editors that write multiple temp files,
/// K8s ConfigMap swaps, etc.).
///
/// Irrelevant events (some *other* file in the watched config directory
/// churning) are consumed but do NOT extend the deadline — otherwise
/// continuous unrelated churn would push the deadline out by a full window
/// per event and starve the pending reload indefinitely.
///
/// Returns early if the channel closes mid-drain so the caller can observe
/// the closure on the next `rx.recv()`.
async fn drain_during(
    rx: &mut tokio::sync::mpsc::Receiver<notify::Result<notify::Event>>,
    window: Duration,
    config_paths: &[PathBuf],
) {
    let mut deadline = tokio::time::Instant::now() + window;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return;
        }
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => return,
            maybe_event = rx.recv() => {
                match maybe_event {
                    // Extend the deadline only from the last event on the
                    // config file itself, so the window truly elapses
                    // "after the last file event" without letting sibling
                    // churn starve the reload.
                    Some(Ok(ev)) if is_relevant_to_any(&ev, config_paths) => {
                        deadline = tokio::time::Instant::now() + window;
                    }
                    Some(_) => {}
                    None => return,
                }
            }
        }
    }
}

/// Cheap content fingerprint of the config file, used by [`watch_loop`] to
/// detect a write that landed while a reload was in flight (i.e. after the
/// rebuild closure read the file). Returns `None` when the file cannot be
/// read (transient rename/swap); callers treat `None` as "unchanged" to
/// avoid spurious extra reloads during K8s `..data` symlink swaps.
async fn fingerprint(path: &std::path::Path) -> Option<u64> {
    use std::hash::Hasher;
    let bytes = tokio::fs::read(path).await.ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write(&bytes);
    Some(hasher.finish())
}

fn is_relevant_to_any(event: &notify::Event, config_paths: &[PathBuf]) -> bool {
    config_paths
        .iter()
        .any(|path| is_relevant_event(event, path))
}

async fn watched_config_files(root: &std::path::Path) -> Vec<PathBuf> {
    // Watch logical paths as well as canonical targets: symlink swaps change
    // the logical parent, while file writes may arrive under the target path.
    let mut raw = vec![root.to_path_buf()];
    if let Some(current) = crate::config::loader::current_config_path() {
        let current = absolute_path(current);
        let same_root = current == root
            || match (
                tokio::fs::canonicalize(&current).await,
                tokio::fs::canonicalize(root).await,
            ) {
                (Ok(current), Ok(root)) => current == root,
                _ => false,
            };
        if same_root {
            raw.push(current);
            raw.extend(crate::config::loader::config_contributor_paths());
        }
    }
    let mut out = Vec::new();
    for path in raw {
        let path = absolute_path(path);
        if !out.contains(&path) {
            out.push(path.clone());
        }
        if let Ok(canon) = tokio::fs::canonicalize(&path).await {
            if !out.contains(&canon) {
                out.push(canon);
            }
        }
    }
    out
}

fn absolute_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&path))
            .unwrap_or(path)
    }
}

fn ensure_parents_watched<W: Watcher>(
    watcher: &mut W,
    files: &[PathBuf],
    watched_parents: &mut HashSet<PathBuf>,
) -> bool {
    let mut watched_any = !watched_parents.is_empty();
    for file in files {
        let parent = file.parent().unwrap_or(file.as_path()).to_path_buf();
        if !watched_parents.insert(parent.clone()) {
            watched_any = true;
            continue;
        }
        if let Err(e) = watcher.watch(&parent, notify::RecursiveMode::NonRecursive) {
            tracing::error!(error = %e, path = ?parent, "failed to watch config directory");
            watched_parents.remove(&parent);
        } else {
            watched_any = true;
        }
    }
    if !watched_any {
        tracing::error!("failed to watch any config directory");
    }
    watched_any
}

async fn fingerprint_all(paths: &[PathBuf]) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut any = false;
    for path in paths {
        if let Some(fp) = fingerprint(path).await {
            any = true;
            path.hash(&mut hasher);
            hasher.write_u64(fp);
        }
    }
    any.then_some(hasher.finish())
}

fn is_relevant_event(event: &notify::Event, config_path: &std::path::Path) -> bool {
    if !matches!(
        event.kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    ) {
        return false;
    }
    let config_parent = config_path.parent();
    event.paths.iter().any(|p| {
        // Direct match on the config file itself, or a same-basename path in
        // the *same directory* (covers editors that write to a temp file then
        // rename over the original). Constraining the basename match to the
        // same parent makes the code self-documenting and robust to a future
        // change in watcher mode or watched path; today the `NonRecursive`
        // watcher on `config_path.parent()` already limits delivered events
        // to that directory, but the explicit guard means the relevance
        // filter stays correct even if that wiring changes.
        if p == config_path {
            return true;
        }
        if p.parent() == config_parent && p.file_name() == config_path.file_name() {
            return true;
        }
        // Kubernetes ConfigMap volume mounts: the config file is a symlink
        // into `..data` (itself a symlink to a timestamped directory like
        // `..2025_01_01_00_00_00.123456789/`). On update, kubelet creates
        // a fresh timestamped dir and atomically swaps the `..data`
        // symlink, so the watcher typically delivers an event whose path
        // is `..data` (or a sibling under `..`) rather than the config
        // file itself. Treat any Modify/Create/Remove of the `..data`
        // symlink — or any other `..`-prefixed sibling inside the same
        // parent directory — as a candidate trigger.
        if let Some(parent) = config_parent {
            if p.parent() == Some(parent) {
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    // `..data` itself, or a timestamped dir like
                    // `..2025_01_01_00_00_00.123456789` — the third char is
                    // a digit. A bare `starts_with("..")` would also match
                    // editor artifacts (e.g. vim's `..config.yaml.swp`) and
                    // trigger spurious full AppState rebuilds.
                    let is_timestamped = name.as_bytes().get(2).is_some_and(|b| b.is_ascii_digit());
                    if name == "..data" || is_timestamped {
                        return true;
                    }
                }
            }
        }
        false
    })
}

async fn do_reload(shared: &Arc<ArcSwap<AppState>>, rebuild: &Rebuild) -> anyhow::Result<()> {
    let current = shared.load();
    match rebuild(&current).await {
        Ok(new_state) => {
            // Merge DB-stored models into the new state BEFORE the atomic
            // swap so there is no transient window where DB-only models are
            // absent (which would cause "model not found" errors for requests
            // arriving between the swap and a post-swap refresh). No-op when
            // no DB is wired.
            #[cfg(feature = "db")]
            let new_state = {
                let mut ns = new_state;
                if let Some(db) = ns.db.clone() {
                    match crate::state::merge_db_deployments(
                        &ns.config,
                        &db,
                        ns.credential_cipher.as_ref(),
                    )
                    .await
                    {
                        Ok(deployments) => {
                            ns.router.prune_stale_store_keys(&deployments);
                            ns.deployments = deployments;
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "failed to merge DB models before state swap; \
                                 YAML-only deployments will go live"
                            );
                        }
                    }
                }
                ns
            };
            shared.store(Arc::new(new_state));
            tracing::info!("config reloaded and state swapped successfully");
            Ok(())
        }
        Err(e) => {
            tracing::error!(error = %e, "failed to reload config, keeping previous state");
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
    use oxllm_core::deployment::OxllmParams;

    #[test]
    fn included_files_reload_and_new_include_directories_are_followed() {
        let _guard = crate::config::loader::CONFIG_PATH_TEST_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        for name in ["includes", "next"] {
            std::fs::create_dir(dir.path().join(name)).unwrap();
        }
        let root = dir.path().join("root.yaml");
        let first = dir.path().join("includes/models.yaml");
        let next = dir.path().join("next/models.yaml");
        let write_model = |path: &std::path::Path, name: &str| {
            std::fs::write(path, format!("model_list:\n  - model_name: {name}\n    oxllm_params:\n      model: openai/gpt-4o\n")).unwrap();
        };
        write_model(&first, "initial");
        write_model(&next, "new-directory");
        std::fs::write(&root, "model_list: []\ninclude: [includes/models.yaml]\n").unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let config = crate::config::load_config(root.to_str().unwrap()).unwrap();
            let shared = Arc::new(ArcSwap::from_pointee(AppState::new(config).unwrap()));
            let rebuild_path = root.clone();
            let rebuild: Rebuild = Arc::new(move |previous| {
                let path = rebuild_path.clone();
                Box::pin(async move {
                    let config = crate::config::load_config_async(path.to_str().unwrap()).await?;
                    previous.reload_from_config(config)
                })
            });
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(watch_loop(root.clone(), shared.clone(), rebuild, Some(ready_tx)));
            tokio::time::timeout(Duration::from_secs(6), ready_rx).await
                .expect("watcher must register before writes").expect("watcher started");
            let wait_for_model = |name: &'static str| {
                let shared = shared.clone();
                async move {
                    tokio::time::timeout(Duration::from_secs(6), async {
                        loop {
                            if shared
                                .load()
                                .deployments
                                .iter()
                                .any(|deployment| deployment.model_name == name)
                            {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                    })
                    .await
                    .unwrap_or_else(|_| panic!("expected include reload to {name}; live models: {:?}; contributors: {:?}",
                        shared.load().deployments.iter().map(|deployment| &deployment.model_name).collect::<Vec<_>>(),
                        crate::config::loader::config_contributor_paths()));
                }
            };
            write_model(&first, "edited-include");
            wait_for_model("edited-include").await;
            std::fs::write(&root, "model_list: []\ninclude: [next/models.yaml]\n").unwrap();
            wait_for_model("new-directory").await;
            tokio::time::sleep(Duration::from_millis(300)).await;
            write_model(&next, "edited-new-include");
            wait_for_model("edited-new-include").await;
        });
    }

    fn test_config(model_name: &str) -> ProxyConfig {
        ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: model_name.to_string(),
                oxllm_params: OxllmParams {
                    model: format!("openai/{model_name}"),
                    api_key: Some("sk-test".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    #[tokio::test]
    async fn watcher_reloads_state_on_file_change() {
        let dir = std::env::temp_dir();
        let config_path = dir.join(format!("oxllm_test_watcher_{}.yaml", uuid::Uuid::new_v4()));

        let initial = test_config("gpt-4o");
        let yaml = serde_yml::to_string(&initial).expect("serialize");
        std::fs::write(&config_path, &yaml).expect("write");

        let initial_state = AppState::new(initial).expect("build state");
        let shared = Arc::new(ArcSwap::from_pointee(initial_state));

        let rebuild: crate::config::watcher::Rebuild = Arc::new(|_prev: &AppState| {
            Box::pin(async move {
                let cfg = test_config("claude-3");
                AppState::new(cfg)
            })
        });

        spawn_watcher(config_path.clone(), shared.clone(), rebuild);

        tokio::time::sleep(Duration::from_millis(200)).await;

        let updated = test_config("claude-3");
        let new_yaml = serde_yml::to_string(&updated).expect("serialize");
        std::fs::write(&config_path, &new_yaml).expect("write update");

        tokio::time::sleep(Duration::from_secs(2)).await;

        let swapped = shared.load();
        let found = swapped
            .deployments
            .iter()
            .any(|d| d.model_name == "claude-3");

        let _ = std::fs::remove_file(&config_path);
        assert!(found, "state should have been swapped after file change");
    }

    #[cfg(feature = "db")]
    #[tokio::test]
    async fn reload_preserves_db_handle() {
        let initial = test_config("gpt-4o");
        let initial_state = AppState::new(initial).expect("build state");
        assert!(initial_state.db.is_none(), "test precondition: no db wired");

        let updated = test_config("claude-3");
        let reloaded = initial_state
            .reload_from_config(updated)
            .expect("reload should succeed");
        assert!(
            reloaded.db.is_none(),
            "db handle should be preserved (still None) across reload"
        );
    }

    #[tokio::test]
    async fn reload_preserves_http_client_and_cache() {
        let initial = test_config("gpt-4o");
        let initial_state = AppState::new(initial).expect("build state");
        let initial_cache_ptr = Arc::as_ptr(&initial_state.cache);
        let initial_router_models: Vec<String> = initial_state
            .deployments
            .iter()
            .map(|d| d.model_name.clone())
            .collect();

        let updated = test_config("claude-3");
        let reloaded = initial_state
            .reload_from_config(updated)
            .expect("reload should succeed");

        assert_eq!(
            Arc::as_ptr(&reloaded.cache),
            initial_cache_ptr,
            "cache Arc should be reused across reload"
        );

        let reloaded_models: Vec<String> = reloaded
            .deployments
            .iter()
            .map(|d| d.model_name.clone())
            .collect();
        assert_ne!(
            initial_router_models, reloaded_models,
            "deployments should reflect the new config after reload"
        );
    }

    /// `is_relevant_event` must match the Kubernetes ConfigMap update path:
    /// when the mounted config file is a symlink through `..data`, an update
    /// swaps `..data` and the watcher delivers an event with that path
    /// rather than the config file itself.
    #[test]
    fn relevant_event_matches_k8s_data_symlink() {
        use notify::EventKind;
        use std::path::Path;

        let config_path = Path::new("/mnt/config/config.yaml");
        let parent = config_path.parent().unwrap();
        let data_symlink = parent.join("..data");
        let timestamped_dir = parent.join("..2025_01_01_00_00_00.123456789");

        let mk = |path: &Path| notify::Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Any),
            paths: vec![path.to_path_buf()],
            attrs: Default::default(),
        };

        assert!(
            is_relevant_event(&mk(&data_symlink), config_path),
            "..data symlink swap in the config dir must trigger a reload"
        );
        assert!(
            is_relevant_event(&mk(&timestamped_dir), config_path),
            "K8s timestamped dir (..*) swap in the config dir must trigger a reload"
        );
        assert!(
            is_relevant_event(&mk(config_path), config_path),
            "direct match on the config path must trigger a reload"
        );

        // Editor artifacts in the config dir (vim swap files like
        // `..config.yaml.swp`) must NOT trigger a reload.
        let vim_swap = parent.join("..config.yaml.swp");
        assert!(
            !is_relevant_event(&mk(&vim_swap), config_path),
            "editor temp file (..*.swp) must not trigger a reload"
        );

        // Sibling in another dir must NOT match.
        let other_dir = Path::new("/tmp/other.yaml");
        assert!(
            !is_relevant_event(&mk(other_dir), config_path),
            "unrelated path must not trigger a reload"
        );
    }

    /// A path with the same basename as the config file but in a *different*
    /// parent directory must NOT trigger a reload. The basename check is
    /// constrained to the same parent so a future change to the watcher's
    // mode or watched path can't surface as spurious reloads for unrelated
    // files elsewhere in the filesystem tree.
    #[test]
    fn relevant_event_rejects_same_basename_in_different_directory() {
        use notify::EventKind;
        use std::path::Path;

        let config_path = Path::new("/mnt/config/config.yaml");
        let same_name_elsewhere = Path::new("/tmp/config.yaml");
        let deep_same_name = Path::new("/mnt/config/subdir/config.yaml");

        let mk = |path: &Path| notify::Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Any),
            paths: vec![path.to_path_buf()],
            attrs: Default::default(),
        };

        assert!(
            !is_relevant_event(&mk(same_name_elsewhere), config_path),
            "same basename in a different parent directory must not trigger a reload"
        );
        assert!(
            !is_relevant_event(&mk(deep_same_name), config_path),
            "same basename in a nested subdirectory must not trigger a reload"
        );

        // Sanity check: a same-basename path in the SAME parent still matches
        // (the editor temp-file-rename case the basename branch exists for).
        let same_dir_sibling = config_path.parent().unwrap().join("config.yaml");
        assert!(
            is_relevant_event(&mk(&same_dir_sibling), config_path),
            "same basename in the same parent directory must trigger a reload"
        );
    }

    /// Real debounce: a burst of events on the config file coalesced into a
    /// single reload. We verify the drain primitive discards relevant events
    /// that arrive within the window while still bounding total wall time to
    /// ~window.
    #[tokio::test]
    async fn drain_during_coalesces_burst_into_single_window() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<notify::Result<notify::Event>>(64);
        let config_path = std::path::Path::new("/mnt/config/config.yaml");

        // Pre-fill the channel from a non-runtime thread so we don't try to
        // block the async context; the drain should consume them all within
        // the window and return.
        let sender = tokio::task::spawn_blocking(move || {
            for _ in 0..5 {
                let _ = tx.blocking_send(Ok(notify::Event {
                    kind: notify::EventKind::Modify(notify::event::ModifyKind::Any),
                    paths: vec![config_path.to_path_buf()],
                    attrs: Default::default(),
                }));
            }
            // Hold tx open past the drain window so the receiver doesn't
            // observe a closed channel during the test.
            std::thread::sleep(Duration::from_millis(300));
            drop(tx);
        });

        // Give the blocking task time to enqueue the burst.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let window = Duration::from_millis(100);
        let started = tokio::time::Instant::now();
        let config_paths = [config_path.to_path_buf()];
        drain_during(&mut rx, window, &config_paths).await;
        let elapsed = started.elapsed();

        // Should have drained the queue (channel empty afterwards) and waited
        // roughly `window` (allow some scheduler slack).
        assert!(
            rx.try_recv().is_err(),
            "channel should be drained after the debounce window"
        );
        assert!(
            elapsed >= window,
            "drain_during must wait the full window even after consuming the burst, got {elapsed:?}"
        );
        assert!(
            elapsed < window + Duration::from_millis(500),
            "drain_during must not run away past the window, got {elapsed:?}"
        );

        let _ = sender.await;
    }

    /// Churn on an UNRELATED file in the watched directory must not extend
    /// the debounce deadline: the drain returns after the initial window
    /// instead of being starved by a full extra window per event.
    #[tokio::test]
    async fn drain_during_ignores_unrelated_file_churn() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<notify::Result<notify::Event>>(64);
        let config_path = std::path::Path::new("/mnt/config/config.yaml");
        let unrelated = std::path::Path::new("/mnt/config/other-file.log");

        // Emit an unrelated sibling event every 30ms for ~450ms — each well
        // within the 100ms window, so per-event deadline extension would
        // keep the drain alive for the entire churn (~550ms).
        let sender = tokio::task::spawn(async move {
            for _ in 0..15 {
                let _ = tx
                    .send(Ok(notify::Event {
                        kind: notify::EventKind::Modify(notify::event::ModifyKind::Any),
                        paths: vec![unrelated.to_path_buf()],
                        attrs: Default::default(),
                    }))
                    .await;
                tokio::time::sleep(Duration::from_millis(30)).await;
            }
            drop(tx);
        });

        let window = Duration::from_millis(100);
        let started = tokio::time::Instant::now();
        let config_paths = [config_path.to_path_buf()];
        drain_during(&mut rx, window, &config_paths).await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(400),
            "unrelated sibling churn must not extend the debounce window; \
             drain ran {elapsed:?} (window {window:?})"
        );

        let _ = sender.await;
    }

    /// A config write landing while a reload is in flight (after the rebuild
    /// closure's file read) must not be silently dropped by the post-reload
    /// event drain: the watcher re-fingerprints the file and reloads again.
    #[tokio::test]
    async fn watcher_reloads_again_if_file_changes_during_rebuild() {
        let dir = std::env::temp_dir();
        let config_path = dir.join(format!(
            "oxllm_test_watcher_mid_{}.yaml",
            uuid::Uuid::new_v4()
        ));

        let initial = test_config("gpt-4o");
        let yaml = serde_yml::to_string(&initial).expect("serialize");
        std::fs::write(&config_path, &yaml).expect("write");

        let initial_state = AppState::new(initial).expect("build state");
        let shared = Arc::new(ArcSwap::from_pointee(initial_state));

        let reload_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (started_tx, mut started_rx) = tokio::sync::mpsc::channel::<usize>(4);

        // The first rebuild "blocks" mid-flight (simulating config I/O / DNS
        // validation) long enough for the test to write the file while the
        // reload is in progress — the write lands after the rebuild's
        // would-be file read, so only the fingerprint re-check can catch it.
        let rebuild: crate::config::watcher::Rebuild = {
            let reload_count = reload_count.clone();
            Arc::new(move |_prev: &AppState| {
                let reload_count = reload_count.clone();
                let started_tx = started_tx.clone();
                Box::pin(async move {
                    let n = reload_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let _ = started_tx.send(n).await;
                    if n == 0 {
                        tokio::time::sleep(Duration::from_millis(800)).await;
                    }
                    AppState::new(test_config("claude-3"))
                })
            })
        };

        spawn_watcher(config_path.clone(), shared.clone(), rebuild);

        // Wait for the watcher to come up before touching the file.
        tokio::time::sleep(Duration::from_millis(200)).await;
        std::fs::write(&config_path, "model_list: []\n").expect("write update 1");

        // First reload fires after the debounce window and signals us when
        // it starts; then we race a second write into its in-flight window.
        let first = tokio::time::timeout(Duration::from_secs(10), started_rx.recv())
            .await
            .expect("first reload should start within 10s")
            .expect("watcher alive");
        assert_eq!(first, 0, "first reload index");

        tokio::time::sleep(Duration::from_millis(100)).await;
        std::fs::write(&config_path, "model_list: []\noxllm_settings: {}\n")
            .expect("write update 2 (mid-rebuild)");

        // The fingerprint re-check after the first reload must observe the
        // mid-rebuild write and trigger another reload.
        let second = tokio::time::timeout(Duration::from_secs(10), started_rx.recv())
            .await
            .expect("second reload should start within 10s after mid-rebuild write")
            .expect("watcher alive");

        let _ = std::fs::remove_file(&config_path);
        assert!(
            second >= 1,
            "a write landing mid-rebuild must trigger another reload"
        );
    }
}
