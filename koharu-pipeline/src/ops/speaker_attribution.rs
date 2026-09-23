//! Vision-based speaker attribution: before a page's text goes to the
//! (text-only) translation call, ask a vision model who speaks each balloon
//! and to whom, then pull that speaker's personality/speech-style and their
//! address for the listener out of the active character profile — instead of
//! guessing from CV face-embedding geometry (`character_lib::assign_speakers_to_blocks`,
//! `scan_pronoun_context`), which needs a bare face to embed and so is blind
//! whenever a character wears a mask or helmet. Measured on one such page
//! (Kinnikuman, a masked-wrestler series): the CV/CCIP pipeline named the
//! wrong speaker (or none) on all 4 lines; a single Gemini vision call, given
//! only the page image and each character's id/name/role, got all 4 right.
//!
//! This is enrichment, not a dependency: any failure (no active profile, no
//! Gemini key anywhere, a request error, an unparsable reply) is swallowed
//! and the caller falls back to whatever context it already had — a wrong
//! page never blocks translation.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use image::DynamicImage;
use koharu_llm::providers::{ProviderConfig, build_provider, get_saved_api_key};
use koharu_ml::bilingual::style::{CharacterProfile, load_active};
use koharu_types::TextBlock;
use serde::Deserialize;

const VISION_PROVIDER: &str = "gemini";
const VISION_MODEL: &str = "gemini-3.1-flash-lite-preview";
/// Long side a page is downscaled to before sending — plenty to read balloon
/// text and panel layout, a fraction of the tokens a full-resolution scan
/// image would cost.
const READING_SIDE: u32 = 1600;
/// Bounds one page's vision call. This provider's shared `http_client()`
/// already retries a transient error (a 503 "high demand", common on this
/// preview model) up to 3 times with exponential backoff — silently, inside
/// the single `.await` below, with no cap of its own. Measured: enough 503s
/// in a row can turn one page into 15-30+ seconds before this function's own
/// "swallow the failure" gives up, and that cost lands on every page of a
/// batch job even though the caller was told this is enrichment, not a
/// dependency. Cut losses at a bound instead of trusting the retry to be
/// quick — a page missing this context still translates, just without it.
const VISION_TIMEOUT: Duration = Duration::from_secs(12);

#[derive(Debug, Deserialize)]
struct Attribution {
    // Kept in the schema (and so in the prompt) to anchor each answer to one
    // balloon rather than a page-wide guess, even though only the aggregated
    // speaker/listener ids are used afterward.
    #[allow(dead_code)]
    #[serde(rename = "blockId")]
    block_id: String,
    speaker: Option<String>,
    listener: Option<String>,
}

/// Who speaks on this page, and how they talk — as a `page_context` string,
/// or `None` when there is nothing usable to say (no active character
/// profile, no text, no Gemini key, or the call/parse failed).
pub async fn attribute_speakers(image: &DynamicImage, text_blocks: &[TextBlock]) -> Option<String> {
    let blocks: Vec<(&str, &str)> = text_blocks
        .iter()
        .filter_map(|b| Some((b.id.as_str(), b.text.as_deref()?)))
        .filter(|(_, text)| !text.trim().is_empty())
        .collect();
    if blocks.is_empty() {
        return None;
    }

    let profile = load_active()?;
    if profile.characters.is_empty() {
        return None;
    }

    let api_key = get_saved_api_key(VISION_PROVIDER).ok().flatten();
    let provider = build_provider(
        VISION_PROVIDER,
        ProviderConfig {
            api_key,
            base_url: None,
            temperature: None,
            max_tokens: None,
            custom_system_prompt: None,
            story_context: None,
            key_start_index: None,
        },
    )
    .ok()?;

    let roster = roster_text(&profile.characters);
    let block_list = blocks
        .iter()
        .map(|(id, text)| format!("- {id}: {text}"))
        .collect::<Vec<_>>()
        .join("\n");

    let user_prompt = format!(
        "Known characters who may appear on this page (id — name — who they are):\n\
         {roster}\n\n\
         Text blocks already read off this page, by id:\n\
         {block_list}\n\n\
         For each block id, name who speaks it and who they are speaking to, using ONLY \
         the ids listed above. Use \"unknown\" for the speaker or listener when the page \
         does not make it clear (a crowd, an unnamed extra, narration, a sign, SFX) — never \
         guess a named character in without something on the page supporting it (a face you \
         recognise, a name spoken or captioned, a distinctive design).\n\n\
         Return ONLY a JSON array, one entry per block id, nothing else:\n\
         [{{\"blockId\": \"...\", \"speaker\": \"id or unknown\", \"listener\": \"id or unknown\"}}]"
    );

    let jpeg = encode_jpeg(image).ok()?;
    let raw = match tokio::time::timeout(
        VISION_TIMEOUT,
        provider.look(
            "You read manga pages accurately and answer only with the JSON asked for.",
            &user_prompt,
            &[jpeg],
            "image/jpeg",
            VISION_MODEL,
        ),
    )
    .await
    {
        Ok(result) => result
            .inspect_err(|err| tracing::warn!(%err, "speaker attribution: vision call failed"))
            .ok()?,
        Err(_) => {
            tracing::warn!(
                timeout_secs = VISION_TIMEOUT.as_secs(),
                "speaker attribution: vision call timed out, skipping this page's context"
            );
            return None;
        }
    };

    let attributions = parse_attributions(&raw).inspect_err(|err| {
        tracing::warn!(%err, reply = %raw, "speaker attribution: reply was not usable JSON")
    }).ok()?;
    if attributions.is_empty() {
        return None;
    }

    let by_id: HashMap<&str, &CharacterProfile> =
        profile.characters.iter().map(|c| (c.id.as_str(), c)).collect();

    let mut seen = HashSet::new();
    let mut lines = Vec::new();
    for a in &attributions {
        let Some(speaker_id) = a.speaker.as_deref().filter(|s| *s != "unknown") else {
            continue;
        };
        let Some(speaker) = by_id.get(speaker_id) else {
            continue;
        };
        if !seen.insert(speaker_id) {
            continue;
        }
        let listener = a
            .listener
            .as_deref()
            .filter(|l| *l != "unknown")
            .and_then(|id| by_id.get(id).copied());
        lines.push(describe_speaker(speaker, listener));
    }
    if lines.is_empty() {
        return None;
    }

    tracing::info!(
        speakers = lines.len(),
        blocks = attributions.len(),
        "speaker attribution: vision call succeeded"
    );

    Some(format!(
        "Who speaks on this page, from the character profile (personality and speech carry into \
         how their lines should read; follow the address given when they are talking to the \
         listener named):\n{}",
        lines.join("\n")
    ))
}

fn roster_text(characters: &[CharacterProfile]) -> String {
    characters
        .iter()
        .map(|c| {
            let mut names = vec![c.name.as_str()];
            if !c.name_ja.is_empty() {
                names.push(c.name_ja.as_str());
            }
            let role = if c.role.trim().is_empty() {
                String::new()
            } else {
                format!(" — {}", c.role.trim())
            };
            format!("- {}: {}{role}", c.id, names.join(" / "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn describe_speaker(speaker: &CharacterProfile, listener: Option<&CharacterProfile>) -> String {
    let mut parts = Vec::new();
    if !speaker.personality.trim().is_empty() {
        parts.push(format!("Personality: {}", speaker.personality.trim()));
    }
    if !speaker.speech.trim().is_empty() {
        parts.push(format!("Speech: {}", speaker.speech.trim()));
    }
    if let Some(listener) = listener
        && let Some(relation) = speaker.relations.iter().find(|r| r.to == listener.id)
        && !relation.address.trim().is_empty()
    {
        parts.push(format!(
            "Addressing {}: {}",
            listener.name,
            relation.address.trim()
        ));
    }
    if parts.is_empty() {
        format!("- {}", speaker.name)
    } else {
        format!("- {}. {}", speaker.name, parts.join(" "))
    }
}

fn parse_attributions(raw: &str) -> anyhow::Result<Vec<Attribution>> {
    let text = raw.trim().trim_start_matches("```json").trim_start_matches("```");
    let text = text.strip_suffix("```").unwrap_or(text).trim();
    let text = match (text.find('['), text.rfind(']')) {
        (Some(start), Some(end)) if end >= start => &text[start..=end],
        _ => text,
    };
    Ok(serde_json::from_str(text)?)
}

fn encode_jpeg(image: &DynamicImage) -> anyhow::Result<Vec<u8>> {
    let long = image.width().max(image.height());
    let resized = if long > READING_SIDE {
        image.resize(READING_SIDE, READING_SIDE, image::imageops::FilterType::Lanczos3)
    } else {
        image.clone()
    };
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90)
        .encode_image(&resized.to_rgb8())?;
    Ok(bytes)
}
