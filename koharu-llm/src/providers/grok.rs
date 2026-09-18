use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde::Serialize;

use koharu_http::http::http_client;

use crate::{Language, prompt::build_system_prompt};

use super::{AnyProvider, ensure_provider_success};

/// xAI exposes an OpenAI-compatible chat completions endpoint.
const GROK_ENDPOINT: &str = "https://api.x.ai/v1/chat/completions";

const MAX_RETRIES: u32 = 4;
const RETRY_BASE_MS: u64 = 2_000;

pub struct GrokProvider {
    pub api_key: String,
    pub custom_system_prompt: Option<String>,
    pub story_context: Option<String>,
    /// Stable per-session id sent as `x-grok-conv-id`. xAI matches its cache
    /// from the start of the messages array, but requests can land on different
    /// servers; this header keeps a session's requests on the same cache, which
    /// xAI documents as the way to maximise the hit rate.
    pub conversation_id: String,
}

impl GrokProvider {
    pub fn new(
        api_key: String,
        custom_system_prompt: Option<String>,
        story_context: Option<String>,
    ) -> Self {
        Self {
            api_key,
            custom_system_prompt,
            story_context,
            conversation_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

#[derive(Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

fn is_retryable_grok_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("502")
        || msg.contains("503")
        || msg.contains("504")
        || msg.contains("529")
        || msg.contains("overloaded")
        || msg.contains("high demand")
}

impl AnyProvider for GrokProvider {
    /// Single-shot, no retry: the caller that uses this — relationship
    /// labelling — already tolerates a failed or unparseable answer per pair,
    /// and retrying a best-effort label is not worth the call.
    fn complete<'a>(
        &'a self,
        system_prompt: &'a str,
        user_prompt: &'a str,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let body = ChatRequest {
                model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: system_prompt.to_string(),
                    },
                    ChatMessage {
                        role: "user",
                        content: user_prompt.to_string(),
                    },
                ],
                temperature: 0.0,
            };

            let response = http_client()
                .post(GROK_ENDPOINT)
                .bearer_auth(&self.api_key)
                .header("content-type", "application/json")
                .header("x-grok-conv-id", &self.conversation_id)
                .body(serde_json::to_vec(&body)?)
                .send()
                .await?;

            let resp: serde_json::Value = ensure_provider_success("grok", response)
                .await?
                .json()
                .await?;

            resp["choices"][0]["message"]["content"]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("Grok returned no content"))
        })
    }

    fn translate<'a>(
        &'a self,
        source: &'a str,
        target_language: Language,
        page_context: Option<&'a str>,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            // Cache layout: the system message holds only what is identical for
            // every page of the session (base prompt + story context + cast), so
            // xAI can serve it from its prompt cache at a fraction of the input
            // price. Anything that changes per page — the character/pronoun
            // context — goes into the user message, after the cache boundary.
            let system_prompt_text = build_system_prompt(
                target_language,
                self.custom_system_prompt.as_deref(),
                self.story_context.as_deref(),
            );
            let user_content = match page_context.map(str::trim).filter(|c| !c.is_empty()) {
                Some(context) => format!("{context}\n\n{source}"),
                None => source.to_string(),
            };
            tracing::info!(
                has_page_context = page_context.is_some(),
                has_story_context = self.story_context.is_some(),
                cacheable_prefix_chars = system_prompt_text.len(),
                "Grok translate"
            );

            let body = ChatRequest {
                model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: system_prompt_text,
                    },
                    ChatMessage {
                        role: "user",
                        content: user_content,
                    },
                ],
                temperature: 0.0,
            };
            let body_bytes = serde_json::to_vec(&body)?;

            let mut last_err = anyhow::anyhow!("Grok: no attempts made");
            for attempt in 0..=MAX_RETRIES {
                if attempt > 0 {
                    let delay_ms = RETRY_BASE_MS * (1u64 << (attempt - 1));
                    tracing::warn!(
                        attempt,
                        delay_ms,
                        error = %last_err,
                        "Grok retrying after error"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }

                let response = match http_client()
                    .post(GROK_ENDPOINT)
                    .bearer_auth(&self.api_key)
                    .header("content-type", "application/json")
                    .header("x-grok-conv-id", &self.conversation_id)
                    .body(body_bytes.clone())
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = e.into();
                        continue;
                    }
                };

                let response = match ensure_provider_success("grok", response).await {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = e;
                        if is_retryable_grok_error(&last_err) {
                            continue;
                        }
                        return Err(last_err);
                    }
                };

                let resp: serde_json::Value = match response.json().await {
                    Ok(v) => v,
                    Err(e) => {
                        last_err = e.into();
                        continue;
                    }
                };

                let finish_reason = resp["choices"][0]["finish_reason"]
                    .as_str()
                    .unwrap_or("UNKNOWN");

                // Cached prompt tokens bill at a fraction of fresh ones; log them
                // so a broken cache prefix is visible instead of silent.
                tracing::info!(
                    prompt_tokens = resp["usage"]["prompt_tokens"].as_u64(),
                    cached_tokens =
                        resp["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64(),
                    completion_tokens = resp["usage"]["completion_tokens"].as_u64(),
                    "Grok usage"
                );

                match resp["choices"][0]["message"]["content"]
                    .as_str()
                    .filter(|text| !text.trim().is_empty())
                {
                    Some(t) => return Ok(t.to_string()),
                    None => {
                        // No content: content_filter, length, etc. — skip, don't retry.
                        tracing::warn!(finish_reason, "Grok returned no content, skipping block");
                        return Ok(String::new());
                    }
                }
            }

            Err(last_err)
        })
    }
}
