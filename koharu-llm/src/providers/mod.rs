use std::future::Future;
use std::pin::Pin;

use anyhow::Context;
use keyring::Entry;

use crate::Language;

pub mod claude;
pub mod deepseek;
pub mod gemini;
pub mod grok;
pub mod key_pool;
pub mod openai;
pub mod openai_compatible;

const API_KEY_SERVICE: &str = "koharu";

fn provider_key_entry(provider: &str) -> anyhow::Result<Entry> {
    let username = format!("llm_provider_api_key_{provider}");
    Ok(Entry::new(API_KEY_SERVICE, &username)?)
}

pub fn get_saved_api_key(provider: &str) -> anyhow::Result<Option<String>> {
    let entry = provider_key_entry(provider)?;
    match entry.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(err) => Err(err.into()),
    }
}

pub fn set_saved_api_key(provider: &str, api_key: &str) -> anyhow::Result<()> {
    let entry = provider_key_entry(provider)?;
    if api_key.trim().is_empty() {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(err) => Err(err.into()),
        }
    } else {
        entry.set_password(api_key.trim())?;
        Ok(())
    }
}

pub async fn ensure_provider_success(
    provider: &str,
    response: reqwest::Response,
) -> anyhow::Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response
        .text()
        .await
        .with_context(|| format!("Failed to read {provider} error response body"))?;
    let body_lower = body.to_ascii_lowercase();
    let quota_exceeded = status.as_u16() == 429
        || body_lower.contains("insufficient_quota")
        || body_lower.contains("quota")
        || body_lower.contains("resource_exhausted")
        || body_lower.contains("rate limit exceeded")
        || body_lower.contains("credit balance is too low");

    if quota_exceeded {
        // Gemini's free tier limits requests per *minute* as well as per day,
        // and both arrive as 429. Conflating them would retire a key for hours
        // over a burst it would recover from in seconds, so the per-minute case
        // gets its own marker. The UI reads only the provider segment, so the
        // extra suffix is invisible there.
        let per_minute = body_lower.contains("perminute")
            || body_lower.contains("per minute")
            || body_lower.contains("requests per minute");
        if per_minute {
            anyhow::bail!("provider_quota_exceeded:{provider}:rpm");
        }
        anyhow::bail!("provider_quota_exceeded:{provider}");
    }

    // Providers disagree on the status code for a bad key (xAI answers 400), so
    // match on the message as well to give the UI something actionable.
    let invalid_key = matches!(status.as_u16(), 401 | 403)
        || body_lower.contains("incorrect api key")
        || body_lower.contains("invalid api key")
        || body_lower.contains("invalid_api_key")
        || body_lower.contains("api key not valid")
        || body_lower.contains("unauthorized");

    if invalid_key {
        anyhow::bail!("provider_invalid_api_key:{provider}");
    }

    anyhow::bail!("{provider} API request failed ({status}): {body}");
}

pub trait AnyProvider: Send + Sync {
    /// Live key-pool state, for providers that rotate between several keys.
    /// `None` for single-key providers.
    fn key_status(&self) -> Option<key_pool::KeyPoolStatus> {
        None
    }

    /// Translate `source` with an optional per-page character context that is
    /// appended to the provider's configured story_context for this call only.
    fn translate<'a>(
        &'a self,
        source: &'a str,
        target_language: Language,
        page_context: Option<&'a str>,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>>;

    /// Plain single-shot "system prompt + user prompt → text" call, with none of
    /// `translate`'s manga-translation shaping (no `build_system_prompt` persona,
    /// no story/page context, no SFX/quote post-processing). Used by callers that
    /// need a generic LLM completion, e.g. the Character Scanner's relationship
    /// labeling.
    fn complete<'a>(
        &'a self,
        system_prompt: &'a str,
        user_prompt: &'a str,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>>;

    /// `complete`, but with images attached. Used for reading lettering off a
    /// page: every local OCR measured against this corpus misread the outlined,
    /// artwork-backed text, while a vision model read it correctly.
    ///
    /// Takes a slice rather than one image so a whole page's balloons go in a
    /// single request — a page here carries around twenty of them, and the
    /// crops are small.
    ///
    /// Most providers cannot do this, so the default refuses rather than
    /// silently dropping the images and answering about nothing.
    fn look<'a>(
        &'a self,
        _system_prompt: &'a str,
        _user_prompt: &'a str,
        _images: &'a [Vec<u8>],
        _mime_type: &'a str,
        _model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async { anyhow::bail!("provider_no_vision") })
    }
}

/// Combine a stored story context with a per-page character context.
pub fn extend_story_context(story: Option<&str>, page: Option<&str>) -> Option<String> {
    match (
        story.filter(|s| !s.trim().is_empty()),
        page.filter(|p| !p.trim().is_empty()),
    ) {
        (Some(s), Some(p)) => Some(format!("{s}\n\n{p}")),
        (Some(s), None) => Some(s.to_string()),
        (None, Some(p)) => Some(p.to_string()),
        (None, None) => None,
    }
}

pub struct ProviderConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub custom_system_prompt: Option<String>,
    pub story_context: Option<String>,
    /// 1-based position in the key list to begin at, for pools whose earlier
    /// keys are already spent for the day.
    pub key_start_index: Option<u32>,
}

pub fn build_provider(
    provider_id: &str,
    config: ProviderConfig,
) -> anyhow::Result<Box<dyn AnyProvider>> {
    let required_api_key = |name: &str| {
        config
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| anyhow::anyhow!("api_key is required for {name}"))
    };

    let provider: Box<dyn AnyProvider> = match provider_id {
        "openai" => Box::new(openai::OpenAiProvider {
            api_key: required_api_key("openai")?,
            story_context: config.story_context,
        }),
        "gemini" => {
            // Settings key first, then KOHARU_GEMINI_API_KEYS, then a key file.
            let keys = key_pool::collect_keys("gemini", config.api_key.as_deref());
            if keys.is_empty() {
                return Err(anyhow::anyhow!("api_key is required for gemini"));
            }
            tracing::info!(
                count = keys.len(),
                start_index = config.key_start_index.unwrap_or(1),
                "gemini key pool ready"
            );
            Box::new(gemini::GeminiProvider {
                keys: key_pool::ApiKeyPool::starting_at(
                    "gemini",
                    keys,
                    config.key_start_index.unwrap_or(1),
                ),
                custom_system_prompt: config.custom_system_prompt.clone(),
                story_context: config.story_context,
            })
        }
        "grok" => Box::new(grok::GrokProvider::new(
            required_api_key("grok")?,
            config.custom_system_prompt.clone(),
            config.story_context,
        )),
        "claude" => Box::new(claude::ClaudeProvider {
            api_key: required_api_key("claude")?,
            story_context: config.story_context,
        }),
        "deepseek" => Box::new(deepseek::DeepSeekProvider {
            api_key: required_api_key("deepseek")?,
            story_context: config.story_context.clone(),
        }),
        "openai-compatible" => Box::new(openai_compatible::OpenAiCompatibleProvider {
            base_url: config
                .base_url
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!("base_url is required for the openai-compatible provider")
                })?,
            api_key: config
                .api_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            custom_system_prompt: config.custom_system_prompt,
            story_context: config.story_context,
        }),
        other => anyhow::bail!("Unknown API provider: {other}"),
    };

    Ok(provider)
}
