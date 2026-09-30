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

use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;
use koharu_llm::providers::{ProviderConfig, build_provider, get_saved_api_key};
use koharu_ml::bilingual::style::{CharacterProfile, load_active};
use koharu_ml::character_library::{CharacterLibrary, FaceMatch};
use koharu_types::TextBlock;
use serde::Deserialize;

const VISION_PROVIDER: &str = "gemini";
const VISION_MODEL: &str = "gemini-3.5-flash-lite";
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
///
/// `GeminiProvider::look` now also rotates through the whole key pool on a
/// spent key rather than giving up on the first one — worth a wider bound
/// than the 12s this held before that fix, since a few dead keys in a row
/// (the earlier ones in `gemini_keys.txt`, spent from an earlier run today)
/// each cost one more request before a live key is found.
const VISION_TIMEOUT: Duration = Duration::from_secs(20);

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

/// One region's answer from [`read_page`]: the text actually printed there,
/// plus who spoke it when a character roster was given to match against.
#[derive(Debug, Deserialize)]
struct BlockRead {
    // A bare number in the prompt's own example, but tolerate a quoted one
    // too ("0") since a model asked for JSON does not always follow the
    // example's exact type — either way this is an index into the block list
    // `read_page` sent, not a `TextBlock::id`; see that function's docs.
    #[serde(rename = "blockId")]
    block_id: serde_json::Value,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    speaker: Option<String>,
    #[serde(default)]
    listener: Option<String>,
    /// Fractional `[x, y, width, height]` box around the speaking character's
    /// face, drawn by the model itself — not a guessed name, a *location*.
    /// Resolved against the character library's face embeddings afterward,
    /// which is a more reliable identity check than the model naming a
    /// character off the text roster alone (see `read_page`'s use of it).
    #[serde(default, rename = "speakerFaceBox")]
    speaker_face_box: Option<Vec<f32>>,
    /// The speaker's apparent gender ("male"/"female"/"unknown"), always
    /// asked for alongside `speakerFaceBox` — the fallback signal for a line
    /// neither the face match nor the named guess resolves to anyone in the
    /// active profile.
    #[serde(default, rename = "speakerGender")]
    speaker_gender: Option<String>,
}

/// Text block indices in manga reading order — top to bottom, right to left
/// within each row of vertically-overlapping blocks — instead of whatever
/// order the detector happened to emit them in.
///
/// This is what actually fixes read_page's box-to-text mismatches, not the
/// index protocol: the model reads a busy page the way any reader does, in
/// this order, and answers in that same sequence — asking it to instead
/// track "which numbered box am I looking at" against a list given in
/// detector order (essentially arbitrary) is what was causing it to
/// transcribe correctly but mislabel which region an answer belonged to.
/// Give it a list already in the order it was always going to read in, and
/// its Nth answer simply *is* the Nth region's answer.
fn reading_order(text_blocks: &[TextBlock]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..text_blocks.len()).collect();
    indices.sort_by(|&a, &b| text_blocks[a].y.total_cmp(&text_blocks[b].y));

    let mut ordered = Vec::with_capacity(indices.len());
    let mut i = 0;
    while i < indices.len() {
        let mut row_bottom = text_blocks[indices[i]].y + text_blocks[indices[i]].height;
        let mut j = i + 1;
        while j < indices.len() && text_blocks[indices[j]].y < row_bottom {
            row_bottom = row_bottom.max(text_blocks[indices[j]].y + text_blocks[indices[j]].height);
            j += 1;
        }
        let mut row = indices[i..j].to_vec();
        row.sort_by(|&a, &b| text_blocks[b].x.total_cmp(&text_blocks[a].x));
        ordered.extend(row);
        i = j;
    }
    ordered
}

/// `BlockRead::block_id` as a list index, accepting either a JSON number or a
/// numeral quoted as a string.
fn block_index(value: &serde_json::Value) -> Option<usize> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
        .map(|n| n as usize)
}

/// What [`read_page`] got back: the OCR'd text for whichever regions the
/// model actually transcribed (never all of them is guaranteed — a region it
/// judged blank, or simply skipped, is just absent), and the same
/// "who speaks" context [`attribute_speakers`] would have built, when a
/// character profile was active to match speakers against.
pub struct PageRead {
    pub texts: HashMap<String, String>,
    pub context: Option<String>,
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
    parse_json_array(raw)
}

fn parse_json_array<T: serde::de::DeserializeOwned>(raw: &str) -> anyhow::Result<Vec<T>> {
    let text = raw.trim().trim_start_matches("```json").trim_start_matches("```");
    let text = text.strip_suffix("```").unwrap_or(text).trim();
    let text = match (text.find('['), text.rfind(']')) {
        (Some(start), Some(end)) if end >= start => &text[start..=end],
        _ => text,
    };
    Ok(serde_json::from_str(text)?)
}

/// Read every text region on a page and, when a character profile is active,
/// say who speaks each one — in one vision call instead of two.
///
/// This replaces the local OCR pass for a page being translated with
/// character context on: the same call that already has to look at the page
/// for `attribute_speakers` can transcribe the balloons itself, which is one
/// fewer model invocation per page and, going by the same measured case in
/// this module's docs (a masked-wrestler series CV/CCIP got wrong), likely a
/// more accurate reading of a stylised or unusually-set font than a
/// general-purpose OCR model trained on plainer text.
///
/// Unlike [`attribute_speakers`], this does not require an active profile —
/// with none, it still transcribes every region, it just cannot say who is
/// speaking (nothing to match a speaker id against), so `context` is `None`.
/// Callers that get `None` back entirely (no key, a request error, an
/// unparsable reply, the call timed out) should fall back to the local OCR
/// model exactly as if this function did not exist — a wrong or missing
/// reading here must never block translation.
pub async fn read_page(
    image: &DynamicImage,
    text_blocks: &[TextBlock],
    character_lib: &CharacterLibrary,
) -> Option<PageRead> {
    if text_blocks.is_empty() {
        return None;
    }

    // The model answers by index (0, 1, 2…), never by the block's own id: that
    // id is a full UUID, and asking a model to echo 36 characters back
    // verbatim for every one of a dozen-odd regions is exactly the kind of
    // task an LLM quietly gets one digit wrong on. Measured: every single
    // page in one run came back with at least one id that did not match
    // (`covers_every_block` false every time, sometimes even the anonymous
    // count didn't match either), which meant every page fell through to
    // running local OCR anyway — the whole point of this function, defeated.
    // A short index round-trips reliably (the same reasoning as
    // `BLOCK_FORMAT_INSTRUCTIONS`'s `[N]` markers in the main translate call).
    //
    // Listed in `reading_order`, not detector order — see that function.
    let (width, height) = (image.width().max(1) as f32, image.height().max(1) as f32);
    let order = reading_order(text_blocks);
    let ids: Vec<&str> = order.iter().map(|&i| text_blocks[i].id.as_str()).collect();
    let box_list = order
        .iter()
        .enumerate()
        .map(|(i, &orig)| {
            let b = &text_blocks[orig];
            format!(
                "- {i}: [{:.3}, {:.3}, {:.3}, {:.3}]",
                b.x / width,
                b.y / height,
                b.width / width,
                b.height / height
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let profile = load_active().filter(|p| !p.characters.is_empty());

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

    let user_prompt = match &profile {
        Some(profile) => format!(
            "This manga page has these text regions, numbered 0 to {last} IN READING ORDER — the \
             order you would naturally read them in, top to bottom and right to left within each \
             row. Each one is marked directly on the image: a red outline around the region and a \
             small yellow number tag at its top-left corner, in this same numbering. The tags are \
             the authoritative way to tell two regions apart — when two are close together or \
             their balloons nearly touch, read off the number burned into the pixels, not a mental \
             estimate of which fractional box a piece of text falls into. The fractional bounding \
             boxes [x, y, width, height] (top-left 0,0 to bottom-right 1,1) are given too, only as \
             a fallback for a tag hidden behind art:\n\
             {box_list}\n\n\
             Known characters who may appear on this page (id — name — who they are):\n\
             {roster}\n\n\
             For each region: (1) transcribe the exact source-language text inside it into \
             \"text\" — dialogue, narration, captions, sound effects, everything, verbatim, even \
             if partly obscured; leave \"text\" empty only if the region is truly blank; (2) if it \
             is spoken dialogue, name who speaks it and who they are speaking to, using ONLY the \
             character ids listed above. Use \"unknown\" for the speaker or listener when the page \
             does not make it clear (a crowd, an unnamed extra, narration, a sign, SFX) — never \
             guess a named character in without something on the page supporting it (a face you \
             recognise, a name spoken or captioned, a distinctive design); (3) if the speaker's \
             face is visible anywhere on the page — even if you would not otherwise recognise \
             whose it is — draw a tight fractional box around just that face into \
             \"speakerFaceBox\" as [x, y, width, height] in these same 0..1 coordinates; leave it \
             unset when no face is visible (an off-panel voice, narration, a sign, a masked or \
             helmeted character) or the region is not spoken dialogue — this is a location, not a \
             name, it will be matched against the character library separately, so draw it even \
             when you are unsure who it is; (4) always give your best guess of that speaker's \
             apparent gender in \"speakerGender\" as \"male\", \"female\", or \"unknown\", \
             regardless of whether you also named a speaker or drew a face box — it is the \
             fallback when those disagree or neither identifies anyone.\n\n\
             Return ONLY a JSON array with exactly {count} entries, reading the page in that same \
             order, one entry per region — the 1st entry is region 0 (the first thing you read on \
             the page), the 2nd is region 1, and so on. This should already be the order you are \
             reading them in, so just answer as you go; never skip or merge two regions into one \
             entry even if they sit close together. When two regions sit close together or their \
             balloons nearly touch, each region's own box coordinates are the ONLY thing that \
             decides its content — transcribe strictly what is inside that region's own box, and \
             never let a neighboring region's text leak into it or swap places with it. Also \
             include the region's number as \"blockId\", but the array's own order is what is \
             actually used to match each answer back to its region:\n\
             [{{\"blockId\": 0, \"text\": \"...\", \"speaker\": \"id or unknown\", \
             \"listener\": \"id or unknown\", \"speakerFaceBox\": [x, y, width, height] or \
             omitted, \"speakerGender\": \"male\", \"female\", or \"unknown\"}}]",
            roster = roster_text(&profile.characters),
            last = ids.len().saturating_sub(1),
            count = ids.len(),
        ),
        None => format!(
            "This manga page has these text regions, numbered 0 to {last} IN READING ORDER — the \
             order you would naturally read them in, top to bottom and right to left within each \
             row. Each one is marked directly on the image: a red outline around the region and a \
             small yellow number tag at its top-left corner, in this same numbering. The tags are \
             the authoritative way to tell two regions apart — when two are close together or \
             their balloons nearly touch, read off the number burned into the pixels, not a mental \
             estimate of which fractional box a piece of text falls into. The fractional bounding \
             boxes [x, y, width, height] (top-left 0,0 to bottom-right 1,1) are given too, only as \
             a fallback for a tag hidden behind art:\n\
             {box_list}\n\n\
             For each region, transcribe the exact source-language text inside it into \"text\" — \
             dialogue, narration, captions, sound effects, everything, verbatim, even if partly \
             obscured; leave \"text\" empty only if the region is truly blank.\n\n\
             Return ONLY a JSON array with exactly {count} entries, reading the page in that same \
             order, one entry per region — the 1st entry is region 0 (the first thing you read on \
             the page), the 2nd is region 1, and so on. This should already be the order you are \
             reading them in, so just answer as you go; never skip or merge two regions into one \
             entry even if they sit close together. When two regions sit close together or their \
             balloons nearly touch, each region's own box coordinates are the ONLY thing that \
             decides its content — transcribe strictly what is inside that region's own box, and \
             never let a neighboring region's text leak into it or swap places with it. Also \
             include the region's number as \"blockId\", but the array's own order is what is \
             actually used to match each answer back to its region:\n\
             [{{\"blockId\": 0, \"text\": \"...\"}}]",
            last = ids.len().saturating_sub(1),
            count = ids.len(),
        ),
    };

    let jpeg = encode_annotated_jpeg(image, &order, text_blocks).ok()?;
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
            .inspect_err(|err| tracing::warn!(%err, "page read: vision call failed"))
            .ok()?,
        Err(_) => {
            tracing::warn!(
                timeout_secs = VISION_TIMEOUT.as_secs(),
                "page read: vision call timed out, falling back to local OCR"
            );
            return None;
        }
    };

    let reads: Vec<BlockRead> = parse_json_array(&raw)
        .inspect_err(|err| {
            tracing::warn!(%err, reply = %raw, "page read: reply was not usable JSON")
        })
        .ok()?;
    if reads.is_empty() {
        return None;
    }

    // Every region the model answered for, keyed by the real block id —
    // including a genuinely blank one, as an empty string. A caller checking
    // coverage (did every region get an answer at all) needs that
    // distinction: an *absent* key means the model skipped the region
    // entirely and its reading cannot be trusted, where an *empty* value just
    // means it saw nothing there, same as local OCR would report.
    //
    // Matched by ARRAY POSITION, not by the model's own `blockId` claim.
    // Measured on live pages: the model transcribes every region correctly —
    // never garbled — but on a page with several regions close together it
    // regularly mislabels which number a correct answer belongs to (e.g. a
    // heartbeat SFX's text landing under a nearby dialogue balloon's number).
    // The one thing it does reliably is answer once per region without
    // skipping — so when it returns exactly as many entries as regions were
    // given, the Nth entry in the array is trusted to be the Nth region's
    // answer regardless of what `blockId` it put on it. `blockId` is kept
    // only as a diagnostic: how often it disagrees with the position is
    // logged, not acted on.
    let mut texts: HashMap<String, String> = HashMap::new();
    if reads.len() == ids.len() {
        let mut mislabeled = 0usize;
        for (i, r) in reads.iter().enumerate() {
            if block_index(&r.block_id) != Some(i) {
                mislabeled += 1;
            }
            let text = r.text.as_deref().unwrap_or("").trim().to_string();
            texts.insert(ids[i].to_string(), text);
        }
        if mislabeled > 0 {
            tracing::warn!(
                mislabeled,
                total = reads.len(),
                "page read: reply's own blockId disagreed with its array position on some \
                 entries (matched by position anyway)"
            );
        }
    } else {
        // Wrong entry count: position can't be trusted (we don't know which
        // region was skipped), so fall back to whatever blockId claims —
        // better than nothing for the entries that do parse.
        let mut bad_indices = 0usize;
        for r in &reads {
            match block_index(&r.block_id).and_then(|i| ids.get(i)) {
                Some(&id) => {
                    let text = r.text.as_deref().unwrap_or("").trim().to_string();
                    texts.insert(id.to_string(), text);
                }
                None => bad_indices += 1,
            }
        }
        tracing::warn!(
            bad_indices,
            got = reads.len(),
            expected = ids.len(),
            "page read: reply had a different entry count than regions offered, falling back \
             to blockId matching"
        );
    }

    let context = profile.map(|profile| {
        let by_id: HashMap<&str, &CharacterProfile> =
            profile.characters.iter().map(|c| (c.id.as_str(), c)).collect();
        let by_name: HashMap<String, &CharacterProfile> = profile
            .characters
            .iter()
            .map(|c| (c.name.trim().to_lowercase(), c))
            .collect();

        // A face match beats the model's own text-guessed id: `speakerFaceBox`
        // is a location the model drew, not a name it had to recall off the
        // roster, and a CCIP embedding match against the character library is
        // a much harder thing to get right by coincidence than matching a
        // short id string out of a dozen-odd candidates. Falls back to the
        // text guess when there's no box, the match isn't confident, or the
        // matched name isn't one of this profile's characters.
        //
        // One call for the whole page's hints, not one per region: the face
        // detector inside it runs once and is reused for every hint instead
        // of re-scanning the same page per region.
        let hints: Vec<Option<[f32; 4]>> = reads
            .iter()
            .map(|r| match r.speaker_face_box.as_deref() {
                Some([x, y, w, h]) => Some([*x, *y, *w, *h]),
                _ => None,
            })
            .collect();
        let face_matches = character_lib.identify_speaker_faces(image, &hints);

        let resolve_speaker = |r: &BlockRead, face_match: Option<&FaceMatch>| -> Option<&CharacterProfile> {
            if let Some(m) = face_match
                && m.is_known
                && let Some(found) = by_name.get(&m.name.trim().to_lowercase())
            {
                return Some(*found);
            }
            r.speaker
                .as_deref()
                .filter(|s| *s != "unknown")
                .and_then(|id| by_id.get(id).copied())
        };

        let mut seen = HashSet::new();
        let mut seen_genders = HashSet::new();
        let mut lines = Vec::new();
        for (r, face_match) in reads.iter().zip(face_matches.iter()) {
            match resolve_speaker(r, face_match.as_ref()) {
                Some(speaker) => {
                    if !seen.insert(speaker.id.as_str()) {
                        continue;
                    }
                    let listener = r
                        .listener
                        .as_deref()
                        .filter(|l| *l != "unknown")
                        .and_then(|id| by_id.get(id).copied());
                    lines.push(describe_speaker(speaker, listener));
                }
                // Neither the face match nor the named guess resolved to
                // anyone in this profile — still worth passing along the
                // apparent gender the model gave, so translation at least
                // picks a plausible generic address instead of a coin flip.
                None => {
                    let Some(gender) =
                        r.speaker_gender.as_deref().filter(|g| *g != "unknown")
                    else {
                        continue;
                    };
                    if !seen_genders.insert(gender) {
                        continue;
                    }
                    lines.push(format!(
                        "- An unidentified {gender} character (not in the roster above) speaks \
                         somewhere on this page — address them appropriately for that gender \
                         unless the line itself makes something else clear."
                    ));
                }
            }
        }
        (!lines.is_empty()).then(|| {
            format!(
                "Who speaks on this page, from the character profile (personality and speech \
                 carry into how their lines should read; follow the address given when they are \
                 talking to the listener named):\n{}",
                lines.join("\n")
            )
        })
    }).flatten();

    tracing::info!(
        blocks = reads.len(),
        texts = texts.len(),
        has_context = context.is_some(),
        "page read: vision OCR+attribution succeeded"
    );

    Some(PageRead { texts, context })
}

/// A tiny 3x5 bitmap digit font, one row of "on" bits per glyph row — just
/// enough to stamp a legible region number onto the page image itself (see
/// `annotate_regions`). No font file needed for nine pixels of shape.
const DIGIT_GLYPHS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111], // 0
    [0b010, 0b110, 0b010, 0b010, 0b111], // 1
    [0b111, 0b001, 0b111, 0b100, 0b111], // 2
    [0b111, 0b001, 0b111, 0b001, 0b111], // 3
    [0b101, 0b101, 0b111, 0b001, 0b001], // 4
    [0b111, 0b100, 0b111, 0b001, 0b111], // 5
    [0b111, 0b100, 0b111, 0b101, 0b111], // 6
    [0b111, 0b001, 0b010, 0b010, 0b010], // 7
    [0b111, 0b101, 0b111, 0b101, 0b111], // 8
    [0b111, 0b101, 0b111, 0b001, 0b111], // 9
];

const GLYPH_PX: i32 = 5;
const GLYPH_W: i32 = 3 * GLYPH_PX;
const GLYPH_H: i32 = 5 * GLYPH_PX;
const GLYPH_GAP: i32 = GLYPH_PX;

fn draw_digit(img: &mut RgbImage, digit: u8, x: i32, y: i32, color: Rgb<u8>) {
    for (row, bits) in DIGIT_GLYPHS[digit as usize].iter().enumerate() {
        for col in 0..3i32 {
            if bits & (1 << (2 - col)) != 0 {
                draw_filled_rect_mut(
                    img,
                    Rect::at(x + col * GLYPH_PX, y + row as i32 * GLYPH_PX)
                        .of_size(GLYPH_PX as u32, GLYPH_PX as u32),
                    color,
                );
            }
        }
    }
}

/// Stamps a small number tag at each region's top-left corner, plus an
/// outline around the region itself, both in the exact reading-order
/// numbering the prompt sends alongside — "set-of-marks" style grounding.
///
/// A vision model matching a transcription to a bare fractional coordinate
/// has to do that mapping in its head, and on a page with two balloons close
/// together (or, worse, pre-refit text-detection boxes that are only a
/// narrow glyph-column sliver of the actual balloon) that's exactly where it
/// swaps which one it describes — measured on this project: told in the
/// prompt never to do this, it still did, because there was nothing on the
/// image itself to anchor to. Burning the number into the pixels next to the
/// balloon removes that step entirely: it reads off the tag, not a coordinate
/// it has to place itself.
fn annotate_regions(image: &DynamicImage, order: &[usize], text_blocks: &[TextBlock]) -> RgbImage {
    let mut canvas = image.to_rgb8();
    let (w, h) = (canvas.width() as i32, canvas.height() as i32);
    let outline = Rgb([230, 30, 30]);
    let tag_bg = Rgb([255, 221, 0]);
    let tag_fg = Rgb([0, 0, 0]);

    for (j, &orig) in order.iter().enumerate() {
        let b = &text_blocks[orig];
        let (bx, by, bw, bh) = (
            b.x.round() as i32,
            b.y.round() as i32,
            b.width.round().max(1.0) as u32,
            b.height.round().max(1.0) as u32,
        );
        if bw > 0 && bh > 0 && bx < w && by < h {
            draw_hollow_rect_mut(&mut canvas, Rect::at(bx, by).of_size(bw, bh), outline);
        }

        let mut digits: Vec<u8> = Vec::new();
        let mut n = j;
        loop {
            digits.push((n % 10) as u8);
            n /= 10;
            if n == 0 {
                break;
            }
        }
        digits.reverse();

        let label_w = digits.len() as i32 * GLYPH_W + (digits.len() as i32 - 1) * GLYPH_GAP;
        // Anchored just above-left of the box, clamped fully on-canvas so a
        // region flush against an edge still gets a readable tag.
        let lx = (bx - 2).clamp(0, (w - label_w - 4).max(0));
        let ly = (by - GLYPH_H - 4).clamp(0, (h - GLYPH_H - 4).max(0));

        draw_filled_rect_mut(
            &mut canvas,
            Rect::at(lx - 2, ly - 2).of_size((label_w + 4) as u32, (GLYPH_H + 4) as u32),
            tag_bg,
        );
        let mut cx = lx;
        for &d in &digits {
            draw_digit(&mut canvas, d, cx, ly, tag_fg);
            cx += GLYPH_W + GLYPH_GAP;
        }
    }

    canvas
}

/// Like `encode_jpeg`, but with each region's reading-order number and
/// outline burned into the pixels first — see `annotate_regions`.
fn encode_annotated_jpeg(
    image: &DynamicImage,
    order: &[usize],
    text_blocks: &[TextBlock],
) -> anyhow::Result<Vec<u8>> {
    let annotated = DynamicImage::ImageRgb8(annotate_regions(image, order, text_blocks));
    let long = annotated.width().max(annotated.height());
    let resized = if long > READING_SIDE {
        annotated.resize(READING_SIDE, READING_SIDE, image::imageops::FilterType::Lanczos3)
    } else {
        annotated
    };
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90)
        .encode_image(&resized.to_rgb8())?;
    Ok(bytes)
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
