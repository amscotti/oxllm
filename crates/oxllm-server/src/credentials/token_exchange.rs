//! Bounded, expiry-aware token exchanges used by the Google and IBM resolvers.
//! Endpoints are fixed by the resolver, never supplied by inference requests.

use std::sync::{Arc, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use moka::future::Cache;
use oxllm_core::Error;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, OnceCell};
use tokio::time::Instant;
use zeroize::Zeroizing;

pub(super) const MAX_CREDENTIAL_BYTES: usize = 64 * 1024;
const MAX_TOKEN_BODY_BYTES: usize = 64 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const FAILURE_BACKOFF: Duration = Duration::from_secs(2);
const MAX_TOKEN_LIFETIME: Duration = Duration::from_secs(86_400);
type Form = Vec<(&'static str, String)>;
type TokenSlot = Arc<Mutex<Option<TokenEntry>>>;
type FormBuilder = Arc<dyn Fn() -> Result<Form, Error> + Send + Sync>;

struct RenewableSlot {
    entry: Mutex<Option<RenewableEntry>>,
    worker: std::sync::Mutex<Option<tokio::task::AbortHandle>>,
}

impl Drop for RenewableSlot {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.get_mut().unwrap_or_else(|err| err.into_inner()) {
            worker.abort();
        }
    }
}

enum RenewableEntry {
    Ready {
        grant: TokenGrant,
        failure: Option<TokenFailure>,
    },
    Failed {
        failure: TokenFailure,
        retry_at: Instant,
    },
}

struct TokenGrant {
    token: Zeroizing<String>,
    usable_until: Instant,
    renew_at: Instant,
}

pub(super) struct TokenExchange {
    endpoint: String,
    timeout: Duration,
    client: OnceCell<reqwest::Client>,
    // Hash the actual credential material, not the deployment/model name:
    // config reload and key rotation cannot reuse another identity's token.
    // The on-demand cache holds only hashes and short-lived token results.
    cache: Cache<[u8; 32], TokenSlot>,
    // Retain an idle identity through a one-hour token lifetime and the
    // subsequent renewal. Workers retain the renewable credential source and
    // release its zeroizing secret fields on eviction or resolver shutdown.
    renewable_cache: Cache<[u8; 32], Arc<RenewableSlot>>,
}

enum TokenEntry {
    Ready {
        token: Zeroizing<String>,
        refresh_at: Instant,
    },
    Failed {
        failure: TokenFailure,
        retry_at: Instant,
    },
}

#[derive(Clone, Copy)]
struct TokenFailure {
    status: u16,
    message: &'static str,
}

impl From<TokenFailure> for Error {
    fn from(failure: TokenFailure) -> Self {
        Error::Upstream {
            status: failure.status,
            message: failure.message.into(),
        }
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
    // IBM supplies both expires_in and an absolute Unix expiration.
    #[serde(default)]
    expiration: Option<u64>,
}

impl TokenExchange {
    pub(super) fn new(endpoint: &'static str) -> Self {
        Self {
            endpoint: endpoint.into(),
            timeout: REQUEST_TIMEOUT,
            client: OnceCell::new(),
            cache: Cache::builder()
                .max_capacity(128)
                .time_to_idle(Duration::from_secs(3600))
                .build(),
            renewable_cache: Cache::builder()
                .max_capacity(128)
                .time_to_idle(Duration::from_secs(7200))
                .build(),
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(endpoint: String, timeout: Duration) -> Self {
        let mut exchange = Self::new("https://unused.invalid/token");
        exchange.endpoint = endpoint;
        exchange.timeout = timeout;
        exchange
    }

    pub(super) async fn token<F>(&self, material: &[u8], build_form: F) -> Result<String, Error>
    where
        F: FnOnce() -> Result<Form, Error> + Send,
    {
        if material.len() > MAX_CREDENTIAL_BYTES {
            return Err(Error::Config("credential material exceeds 64 KiB".into()));
        }
        let key: [u8; 32] = Sha256::digest(material).into();
        let slot = self
            .cache
            .get_with(key, async { Arc::new(Mutex::new(None)) })
            .await;
        // Coalesce refreshes for this identity without locking other identities.
        let mut entry = slot.lock().await;
        match entry.as_ref() {
            Some(TokenEntry::Ready { token, refresh_at }) if *refresh_at > Instant::now() => {
                return Ok(token.to_string());
            }
            Some(TokenEntry::Failed { failure, retry_at }) if *retry_at > Instant::now() => {
                return Err((*failure).into());
            }
            _ => {}
        }
        let form = build_form()?;
        // If the caller cancels this future during I/O, leave a short-lived
        // failure entry instead of making every queued caller restart it.
        *entry = Some(TokenEntry::Failed {
            failure: failure(503, "credential token refresh was interrupted"),
            retry_at: Instant::now() + self.timeout + FAILURE_BACKOFF,
        });
        match self.fetch(form).await {
            Ok(grant) => {
                let result = grant.token.to_string();
                *entry = Some(TokenEntry::Ready {
                    token: grant.token,
                    refresh_at: grant.usable_until,
                });
                Ok(result)
            }
            Err(failure) => {
                *entry = Some(TokenEntry::Failed {
                    failure,
                    retry_at: Instant::now() + FAILURE_BACKOFF,
                });
                Err(failure.into())
            }
        }
    }

    /// After initial acquisition, renew at half TTL without holding the token
    /// lock during I/O. Readers keep using the unexpired token during renewal;
    /// failures are retried in the background and expired tokens are never sent.
    pub(super) async fn background_token<F>(
        self: &Arc<Self>,
        material: &[u8],
        build_form: F,
    ) -> Result<String, Error>
    where
        F: Fn() -> Result<Form, Error> + Send + Sync + 'static,
    {
        if material.len() > MAX_CREDENTIAL_BYTES {
            return Err(Error::Config("credential material exceeds 64 KiB".into()));
        }
        let key: [u8; 32] = Sha256::digest(material).into();
        let slot = self
            .renewable_cache
            .get_with(key, async {
                Arc::new(RenewableSlot {
                    entry: Mutex::new(None),
                    worker: std::sync::Mutex::new(None),
                })
            })
            .await;
        let mut entry = slot.entry.lock().await;
        match entry.as_ref() {
            Some(RenewableEntry::Ready { grant, .. }) if grant.usable_until > Instant::now() => {
                return Ok(grant.token.to_string());
            }
            Some(RenewableEntry::Ready {
                failure: last_failure,
                ..
            }) => {
                return Err(last_failure
                    .unwrap_or_else(|| {
                        failure(503, "credential background renewal has not completed")
                    })
                    .into());
            }
            Some(RenewableEntry::Failed { failure, retry_at }) if *retry_at > Instant::now() => {
                return Err((*failure).into());
            }
            _ => {}
        }
        let form = build_form()?;
        *entry = Some(RenewableEntry::Failed {
            failure: failure(503, "credential token acquisition was interrupted"),
            retry_at: Instant::now() + self.timeout + FAILURE_BACKOFF,
        });
        match self.fetch(form).await {
            Ok(grant) => {
                let token = grant.token.to_string();
                *entry = Some(RenewableEntry::Ready {
                    grant,
                    failure: None,
                });
                drop(entry);
                self.start_renewal_worker(&slot, Arc::new(build_form));
                Ok(token)
            }
            Err(failure) => {
                *entry = Some(RenewableEntry::Failed {
                    failure,
                    retry_at: Instant::now() + FAILURE_BACKOFF,
                });
                Err(failure.into())
            }
        }
    }

    fn start_renewal_worker(self: &Arc<Self>, slot: &Arc<RenewableSlot>, build_form: FormBuilder) {
        let mut worker = slot.worker.lock().unwrap_or_else(|err| err.into_inner());
        if worker.as_ref().is_some_and(|handle| !handle.is_finished()) {
            return;
        }
        let handle = tokio::spawn(Self::renew_in_background(
            Arc::downgrade(self),
            Arc::downgrade(slot),
            build_form,
        ));
        *worker = Some(handle.abort_handle());
    }

    async fn renew_in_background(
        exchange: Weak<Self>,
        slot: Weak<RenewableSlot>,
        build_form: FormBuilder,
    ) {
        let mut failures = 0_u32;
        loop {
            let Some(current) = slot.upgrade() else {
                return;
            };
            let next = {
                let entry = current.entry.lock().await;
                match entry.as_ref() {
                    Some(RenewableEntry::Ready { grant, .. }) => grant.renew_at,
                    _ => return,
                }
            };
            drop(current);
            tokio::time::sleep_until(next).await;
            let Some(exchange) = exchange.upgrade() else {
                return;
            };
            let Some(client) = exchange.client.get().cloned() else {
                return;
            };
            let endpoint = exchange.endpoint.clone();
            drop(exchange);
            // Hold no slot reference or entry lock while fetching. Eviction
            // can cancel this worker, and readers do not queue behind renewal.
            let result = match build_form() {
                Ok(form) => Self::fetch_with_client(&client, &endpoint, form).await,
                Err(_) => Err(failure(502, "cannot construct credential renewal grant")),
            };
            let Some(current) = slot.upgrade() else {
                return;
            };
            let mut entry = current.entry.lock().await;
            let Some(RenewableEntry::Ready {
                grant,
                failure: last_failure,
            }) = entry.as_mut()
            else {
                return;
            };
            match result {
                Ok(renewed) => {
                    *grant = renewed;
                    *last_failure = None;
                    failures = 0;
                    tracing::debug!("credential token renewed in background");
                }
                Err(error) => {
                    failures = failures.saturating_add(1);
                    let backoff = FAILURE_BACKOFF * (1_u32 << failures.min(4).saturating_sub(1));
                    grant.renew_at = Instant::now() + backoff;
                    *last_failure = Some(error);
                    tracing::warn!(
                        status = error.status,
                        "credential background renewal failed"
                    );
                }
            }
        }
    }

    async fn fetch(&self, form: Form) -> Result<TokenGrant, TokenFailure> {
        let client = self
            .client
            .get_or_try_init(|| async {
                reqwest::Client::builder()
                    .connect_timeout(Duration::from_secs(3))
                    .timeout(self.timeout)
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|_| failure(502, "cannot initialize credential HTTP client"))
            })
            .await?;
        Self::fetch_with_client(client, &self.endpoint, form).await
    }

    async fn fetch_with_client(
        client: &reqwest::Client,
        endpoint: &str,
        form: Form,
    ) -> Result<TokenGrant, TokenFailure> {
        let started = Instant::now();
        let unix_started = unix_now()
            .map_err(|_| failure(502, "system clock is invalid for credential exchange"))?;
        // reqwest's total timeout includes reading the body, not just headers.
        let mut response = client
            .post(endpoint)
            .header("accept", "application/json")
            .form(&form)
            .send()
            .await
            .map_err(transport_failure)?;
        if !response.status().is_success() {
            // Never echo token-endpoint error bodies: they can contain secrets
            // or identifying details. Redirects are failures, never followed.
            let status = response.status().as_u16();
            return Err(failure(
                if status >= 400 { status } else { 502 },
                "credential token endpoint rejected the exchange",
            ));
        }
        if response
            .content_length()
            .is_some_and(|len| len > MAX_TOKEN_BODY_BYTES as u64)
        {
            return Err(failure(502, "credential token response exceeds 64 KiB"));
        }
        let mut body = Zeroizing::new(Vec::new());
        while let Some(chunk) = response.chunk().await.map_err(transport_failure)? {
            if chunk.len() > MAX_TOKEN_BODY_BYTES.saturating_sub(body.len()) {
                return Err(failure(502, "credential token response exceeds 64 KiB"));
            }
            body.extend_from_slice(&chunk);
        }
        let response: TokenResponse = serde_json::from_slice(&body)
            .map_err(|_| failure(502, "invalid credential token response"))?;
        let token = Zeroizing::new(response.access_token);
        if token.trim().is_empty()
            || reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token.as_str()))
                .is_err()
        {
            return Err(failure(
                502,
                "credential endpoint returned an invalid access token",
            ));
        }
        if response
            .token_type
            .as_deref()
            .is_some_and(|kind| !kind.eq_ignore_ascii_case("bearer"))
        {
            return Err(failure(
                502,
                "credential endpoint returned a non-bearer token",
            ));
        }
        let relative = response.expires_in;
        let absolute = response
            .expiration
            .map(|exp| exp.saturating_sub(unix_started));
        let seconds = match (relative, absolute) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => {
                return Err(failure(502, "credential token response is missing expiry"));
            }
        };
        // Never round a short/expired lifetime UP. Bound huge upstream values
        // down, and account for time spent receiving the response.
        let lifetime = Duration::from_secs(seconds).min(MAX_TOKEN_LIFETIME);
        let expires_at = started + lifetime;
        if expires_at <= Instant::now() {
            return Err(failure(
                502,
                "credential endpoint returned an expired access token",
            ));
        }
        let skew = Duration::from_secs(60).min(lifetime / 10);
        Ok(TokenGrant {
            token,
            usable_until: expires_at - skew,
            renew_at: started + lifetime / 2,
        })
    }
}

fn failure(status: u16, message: &'static str) -> TokenFailure {
    TokenFailure { status, message }
}

fn transport_failure(error: reqwest::Error) -> TokenFailure {
    failure(
        if error.is_timeout() { 504 } else { 502 },
        "credential token exchange transport failed",
    )
}

pub(super) fn unix_now() -> Result<u64, Error> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|now| now.as_secs())
        .map_err(|_| Error::Config("system clock is invalid for credential exchange".into()))
}

#[cfg(test)]
#[path = "token_exchange_tests.rs"]
mod tests;
