use std::str::FromStr;

use koharu_llm::ModelId;
use koharu_llm::api::{ALL_API_PROVIDERS, OPENAI_COMPATIBLE_ID};
use koharu_llm::facade as llm;
use koharu_llm::providers::{get_saved_api_key, openai_compatible, set_saved_api_key};
use koharu_types::commands::{
    ApiKeyGetPayload, ApiKeyResult, ApiKeySetPayload, IndexPayload, LlmGeneratePayload,
    LlmListPayload, LlmLoadPayload,
};
pub use openai_compatible::PingResult;
use strum::IntoEnumIterator;
use tracing::instrument;

use crate::{
    AppResources,
    state_tx::{self, ChangedField},
};

#[instrument(level = "debug", skip_all, fields(provider = %payload.provider))]
pub async fn get_api_key(
    _state: AppResources,
    payload: ApiKeyGetPayload,
) -> anyhow::Result<ApiKeyResult> {
    let saved = match get_saved_api_key(&payload.provider) {
        Ok(value) => value,
        Err(err) => {
            tracing::error!(%err, "keyring read failed");
            return Err(err);
        }
    };
    // Count every source, so the UI can enable a provider whose keys live only
    // in a key file rather than in Settings.
    let available_keys =
        koharu_llm::providers::key_pool::collect_keys(&payload.provider, saved.as_deref()).len();
    Ok(ApiKeyResult {
        api_key: saved,
        available_keys: available_keys as u32,
    })
}

#[instrument(level = "debug", skip_all, fields(provider = %payload.provider))]
pub async fn set_api_key(_state: AppResources, payload: ApiKeySetPayload) -> anyhow::Result<()> {
    match set_saved_api_key(&payload.provider, &payload.api_key) {
        Ok(()) => Ok(()),
        Err(err) => {
            tracing::error!(%err, "keyring write failed");
            Err(err)
        }
    }
}

pub async fn llm_list(
    state: AppResources,
    payload: LlmListPayload,
) -> anyhow::Result<Vec<llm::ModelInfo>> {
    let mut models: Vec<ModelId> = ModelId::iter().collect();
    let cpu_factor = if state.llm.is_cpu() { 10 } else { 1 };
    let lang = payload.language.as_deref().unwrap_or("en");
    let zh_locale_factor = if lang.starts_with("zh") { 10 } else { 1 };
    let non_zh_en_locale_factor = if lang.starts_with("zh") || lang.starts_with("en") {
        1
    } else {
        100
    };

    models.sort_by_key(|m| match m {
        ModelId::VntlLlama3_8Bv2 => 100,
        ModelId::Lfm2_350mEnjpMt => 200 / cpu_factor,
        ModelId::SakuraGalTransl7Bv3_7 => 300 / zh_locale_factor,
        ModelId::Sakura1_5bQwen2_5v1_0 => 400 / zh_locale_factor / cpu_factor,
        ModelId::HunyuanMT7B => 500 / non_zh_en_locale_factor,
    });

    let mut result: Vec<llm::ModelInfo> = models.into_iter().map(llm::ModelInfo::new).collect();

    for provider in ALL_API_PROVIDERS {
        for model in provider.models {
            result.push(llm::ModelInfo::api(provider.id, model.id));
        }
    }

    if let Some(base_url) = payload
        .openai_compatible_base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let api_key = match get_saved_api_key(OPENAI_COMPATIBLE_ID) {
            Ok(value) => value,
            Err(err) => {
                tracing::warn!(%err, "failed to read openai-compatible API key");
                None
            }
        };

        match openai_compatible::list_models(base_url, api_key.as_deref()).await {
            Ok(models) => {
                for model in models {
                    result.push(llm::ModelInfo::api(OPENAI_COMPATIBLE_ID, &model));
                }
            }
            Err(err) => {
                tracing::warn!(%err, "failed to list openai-compatible models");
            }
        }
    }

    Ok(result)
}

/// Join the user's story context, the translator's style profile and the
/// character roster, in that order: from what changes least to what changes
/// most, so that adding a character does not invalidate the cached prefix
/// before it.
fn merge_story_context(
    user: Option<&str>,
    style: Option<&str>,
    roster: Option<&str>,
) -> Option<String> {
    let parts: Vec<&str> = [user, style, roster]
        .into_iter()
        .flatten()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

#[instrument(level = "info", skip_all)]
pub async fn llm_load(state: AppResources, payload: LlmLoadPayload) -> anyhow::Result<()> {
    if payload.id.contains(':') {
        let (provider_id, model_id) = payload.id.split_once(':').unwrap();
        let api_key = match payload.api_key {
            Some(key) if !key.trim().is_empty() => Some(key),
            _ => get_saved_api_key(provider_id)?,
        };
        // Fold the character library into the story context, which stays byte
        // identical for the whole session and therefore lands in the provider's
        // prompt cache. Per-page context then only has to name who appears.
        // The style profile a curator chose from a reference volume, if any.
        let style = koharu_ml::bilingual::style::load_active().and_then(|p| p.to_context());
        if let Some(style) = &style {
            tracing::info!(chars = style.len(), "story context includes the active style profile");
        }
        let story_context = merge_story_context(
            payload.story_context.as_deref(),
            style.as_deref(),
            state.ml.character_roster_context().as_deref(),
        );
        state
            .llm
            .load_api(
                provider_id,
                model_id,
                koharu_llm::providers::ProviderConfig {
                    api_key,
                    base_url: payload.base_url,
                    temperature: payload.temperature,
                    max_tokens: payload.max_tokens,
                    custom_system_prompt: payload.custom_system_prompt,
                    story_context,
                    key_start_index: payload.key_start_index,
                },
            )
            .await?;
    } else {
        let id = ModelId::from_str(&payload.id)?;
        state.llm.load(id).await;
    }
    Ok(())
}

pub async fn llm_offload(state: AppResources) -> anyhow::Result<()> {
    state.llm.offload().await;
    Ok(())
}

pub async fn llm_ready(state: AppResources) -> anyhow::Result<bool> {
    Ok(state.llm.ready().await)
}

#[instrument(level = "info", skip_all)]
pub async fn llm_generate(state: AppResources, payload: LlmGeneratePayload) -> anyhow::Result<()> {
    let mut updated = state_tx::read_doc(&state.state, payload.index).await?;
    let target_language = payload.language.as_deref();

    // Scan the page for known characters and inject context into the translation prompt.
    let page_context = state.ml.scan_for_character_context(&updated.image);
    if let Some(ctx) = &page_context {
        tracing::debug!(context_len = ctx.len(), "character context injected into translate");
    }

    match payload.text_block_index {
        Some(block_index) => {
            let balloons = updated.balloons.clone();
            let text_block = updated
                .text_blocks
                .get_mut(block_index)
                .ok_or_else(|| anyhow::anyhow!("Text block not found"))?;
            super::translate::translate_block(
                &state.llm,
                text_block,
                block_index,
                &balloons,
                target_language,
                page_context.as_deref(),
            )
            .await?;
            state_tx::update_doc(
                &state.state,
                payload.index,
                updated,
                &[ChangedField::TextBlocks],
            )
            .await
        }
        None => {
            // Shared flow: SFX dictionary, punctuation pass-through, and retries
            // that re-send only the blocks still missing a translation.
            super::translate::translate_page(
                &state.llm,
                &mut updated,
                target_language,
                page_context.as_deref(),
            )
            .await?;

            state_tx::update_doc(
                &state.state,
                payload.index,
                updated,
                &[ChangedField::TextBlocks],
            )
            .await
        }
    }
}

pub async fn llm_ping(
    base_url: &str,
    api_key: Option<&str>,
) -> anyhow::Result<openai_compatible::PingResult> {
    openai_compatible::ping(base_url, api_key).await
}

pub async fn get_document_for_llm(
    state: AppResources,
    payload: IndexPayload,
) -> anyhow::Result<koharu_types::Document> {
    state_tx::read_doc(&state.state, payload.index).await
}

#[cfg(test)]
mod tests {
    use super::merge_story_context;

    /// The order is what keeps the provider's prompt cache warm: the parts that
    /// change least go first.
    #[test]
    fn parts_join_in_order_of_how_rarely_they_change() {
        assert_eq!(
            merge_story_context(Some("truyện"), Some("văn phong"), Some("nhân vật")).as_deref(),
            Some("truyện\n\nvăn phong\n\nnhân vật")
        );
    }

    #[test]
    fn missing_or_blank_parts_leave_no_empty_gap() {
        assert_eq!(
            merge_story_context(None, Some("văn phong"), Some("  ")).as_deref(),
            Some("văn phong")
        );
        assert_eq!(merge_story_context(Some(" "), None, None), None);
    }
}
