use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde::Serialize;

use koharu_http::http::http_client;

use crate::{Language, prompt::build_system_prompt};

use super::{AnyProvider, ensure_provider_success, extend_story_context};

const MAX_RETRIES: u32 = 4;
const RETRY_BASE_MS: u64 = 2_000;

pub struct GeminiProvider {
    pub api_key: String,
    pub custom_system_prompt: Option<String>,
    pub story_context: Option<String>,
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
    fn translate<'a>(
        &'a self,
        source: &'a str,
        target_language: Language,
        page_context: Option<&'a str>,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, self.api_key
            );

            let combined = extend_story_context(self.story_context.as_deref(), page_context);
            let system_prompt_text = build_system_prompt(
                target_language,
                self.custom_system_prompt.as_deref(),
                combined.as_deref(),
            );
            tracing::info!(
                has_page_context = page_context.is_some(),
                has_story_context = self.story_context.is_some(),
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
            for attempt in 0..=MAX_RETRIES {
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
                    .post(&url)
                    .header("content-type", "application/json")
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

                let response = match ensure_provider_success("gemini", response).await {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = e;
                        if is_retryable_gemini_error(&last_err) {
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
}
