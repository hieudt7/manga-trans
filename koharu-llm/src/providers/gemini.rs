use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde::Serialize;

use koharu_http::http::http_client;

use crate::{Language, prompt::build_system_prompt};

use super::key_pool::{ApiKeyPool, DAILY_COOLDOWN, RATE_LIMIT_COOLDOWN};
use super::{AnyProvider, ensure_provider_success, extend_story_context};

const MAX_RETRIES: usize = 4;
const RETRY_BASE_MS: u64 = 2_000;

pub struct GeminiProvider {
    /// Rotating pool: free-tier keys run out of daily quota mid-chapter, and
    /// swapping to the next one keeps the current page from failing.
    pub keys: ApiKeyPool,
    pub custom_system_prompt: Option<String>,
    pub story_context: Option<String>,
}

fn is_quota_error(err: &anyhow::Error) -> bool {
    err.to_string().starts_with("provider_quota_exceeded:")
}

/// A per-minute rate limit, as opposed to a spent daily quota. The key is still
/// good, so it only rests briefly.
fn is_rate_limit(err: &anyhow::Error) -> bool {
    err.to_string().ends_with(":rpm")
}

/// A key the provider rejected outright — mistyped, revoked, or from the wrong
/// project. Rotating past it matters as much as rotating past a spent one: a
/// single bad line in the key file would otherwise fail every page.
fn is_invalid_key(err: &anyhow::Error) -> bool {
    err.to_string().starts_with("provider_invalid_api_key:")
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
}

#[derive(Serialize)]
struct GenerateRequest {
    system_instruction: SystemInstruction,
    contents: Vec<Content>,
    generation_config: GenerationConfig,
}

fn is_retryable_gemini_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("503") || msg.contains("high demand")
}

impl AnyProvider for GeminiProvider {
    fn key_status(&self) -> Option<super::key_pool::KeyPoolStatus> {
        self.keys.status()
    }

    fn translate<'a>(
        &'a self,
        source: &'a str,
        target_language: Language,
        page_context: Option<&'a str>,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let endpoint = |key: &str| {
                format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                )
            };
            let mut api_key = match self.keys.active() {
                Some(key) => key,
                None => anyhow::bail!("provider_quota_exceeded:gemini"),
            };

            let combined = extend_story_context(self.story_context.as_deref(), page_context);
            let system_prompt_text = build_system_prompt(
                target_language,
                self.custom_system_prompt.as_deref(),
                combined.as_deref(),
            );
            tracing::info!(
                has_page_context = page_context.is_some(),
                has_story_context = self.story_context.is_some(),
                keys_in_pool = self.keys.len(),
                "Gemini translate"
            );

            let body = GenerateRequest {
                system_instruction: SystemInstruction {
                    parts: vec![Part { text: system_prompt_text }],
                },
                contents: vec![Content {
                    parts: vec![Part { text: source.to_string() }],
                }],
                generation_config: GenerationConfig { temperature: 0.0 },
            };
            let body_bytes = serde_json::to_vec(&body)?;

            let mut last_err = anyhow::anyhow!("Gemini: no attempts made");
            // Rotations get their own budget: burning a key is not a transient
            // failure, so it must not eat the backoff retries.
            let mut rotations_left = self.keys.len();
            let mut attempt = 0usize;
            while attempt <= MAX_RETRIES {
                if attempt > 0 {
                    let delay_ms = RETRY_BASE_MS * (1u64 << (attempt - 1));
                    tracing::warn!(
                        attempt,
                        delay_ms,
                        error = %last_err,
                        "Gemini retrying after error"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }

                let response = match http_client()
                    .post(endpoint(&api_key))
                    .header("content-type", "application/json")
                    .body(body_bytes.clone())
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = e.into();
                        attempt += 1;
                        continue;
                    }
                };

                let response = match ensure_provider_success("gemini", response).await {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = e;
                        // Out of quota: swap keys and resend the same blocks
                        // immediately, without counting this as a transient
                        // attempt — a spent key is not a flaky network.
                        if (is_quota_error(&last_err) || is_invalid_key(&last_err))
                            && rotations_left > 0
                        {
                            rotations_left -= 1;
                            if is_invalid_key(&last_err) {
                                tracing::warn!("gemini rejected a key as invalid, skipping it");
                            }
                            // A rejected key never becomes valid on its own, so
                            // it rests as long as a spent one.
                            let cooldown = if is_rate_limit(&last_err) {
                                RATE_LIMIT_COOLDOWN
                            } else {
                                DAILY_COOLDOWN
                            };
                            match self.keys.rotate(cooldown) {
                                Some(next) => {
                                    api_key = next;
                                    continue;
                                }
                                // Pool spent: fall through to the backoff path
                                // for a rate limit, since those keys recover in
                                // seconds; a daily exhaustion is terminal.
                                None if is_rate_limit(&last_err) => {
                                    attempt += 1;
                                    continue;
                                }
                                None => return Err(last_err),
                            }
                        }
                        if is_retryable_gemini_error(&last_err) {
                            attempt += 1;
                            continue;
                        }
                        return Err(last_err);
                    }
                };

                let resp: serde_json::Value = match response.json().await {
                    Ok(v) => v,
                    Err(e) => {
                        last_err = e.into();
                        attempt += 1;
                        continue;
                    }
                };

                let finish_reason = resp["candidates"][0]["finishReason"]
                    .as_str()
                    .unwrap_or("UNKNOWN");

                match resp["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                    Some(t) => return Ok(t.to_string()),
                    None => {
                        // No content: RECITATION, SAFETY, MAX_TOKENS, etc. — skip, don't retry.
                        tracing::warn!(finish_reason, "Gemini returned no content, skipping block");
                        return Ok(String::new());
                    }
                }
            }

            Err(last_err)
        })
    }

    /// Single-shot, non-retrying (relationship labeling is best-effort — the
    /// caller already tolerates a failed/unparseable response per pair).
    fn complete<'a>(
        &'a self,
        system_prompt: &'a str,
        user_prompt: &'a str,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            // The single key this provider used to hold became a pool; take
            // whichever key is currently in rotation.
            let Some(api_key) = self.keys.active() else {
                anyhow::bail!("provider_quota_exceeded:gemini")
            };
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            );

            let body = GenerateRequest {
                system_instruction: SystemInstruction {
                    parts: vec![Part { text: system_prompt.to_string() }],
                },
                contents: vec![Content {
                    parts: vec![Part { text: user_prompt.to_string() }],
                }],
                generation_config: GenerationConfig { temperature: 0.3 },
            };

            let response = http_client()
                .post(&url)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&body)?)
                .send()
                .await?;

            let resp: serde_json::Value = ensure_provider_success("gemini", response)
                .await?
                .json()
                .await?;

            let finish_reason = resp["candidates"][0]["finishReason"]
                .as_str()
                .unwrap_or("UNKNOWN");

            match resp["candidates"][0]["content"]["parts"][0]["text"].as_str() {
                Some(t) => Ok(t.to_string()),
                None => {
                    tracing::warn!(finish_reason, "Gemini returned no content, skipping");
                    Ok(String::new())
                }
            }
        })
    }
}
