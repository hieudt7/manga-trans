//! Learning how a translator writes, from what they already published.
//!
//! The pipeline's own Vietnamese is serviceable and anonymous. A published
//! translation is neither: it picks particular pronouns for particular pairs of
//! characters, keeps some sound effects and rewrites others, breaks long lines
//! at particular places, and settles on names that have to stay settled. None
//! of that is guessable from the Japanese alone, and all of it is visible in a
//! volume that has already been translated.
//!
//! So the pairs from [`super::pairs_from_page`] are read once, by a model, and
//! what it notices is kept as a profile. The profile then rides in the
//! provider's story context, where it is stable across a chapter and so stays
//! in the prompt cache.
//!
//! The profile is deliberately small. It is advice given once at the top of a
//! translation, competing for the model's attention with the page in front of
//! it, and a long list of rules read as noise.

use serde::{Deserialize, Serialize};

use super::SentencePair;

/// How many pairs the model is shown.
///
/// A volume yields on the order of a thousand; sending all of them costs a lot
/// and teaches little more than a spread does, because the habits worth
/// learning repeat. The spread is taken across the whole volume rather than
/// from the front, so the opening pages — titles, credits, a contents page —
/// cannot stand in for the dialogue.
const SAMPLE_SIZE: usize = 220;

/// Pairs below this are not trusted enough to teach from. The measured corpus
/// put every true page pairing at 0.729 or better; a balloon pairing then
/// scales that down by how far the two balloons sat apart.
const MIN_CONFIDENCE: f32 = 0.45;

/// A Vietnamese line runs longer than the Japanese it came from — the measured
/// corpus had a median of 2.9 characters per Japanese character, with the
/// middle 80% between 1.8 and 4.5. Far outside that band and the two lines are
/// probably not translations of each other, whatever the geometry said.
const MIN_LENGTH_RATIO: f32 = 0.8;
const MAX_LENGTH_RATIO: f32 = 9.0;

/// What reading a translator's work taught us about how they write.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleProfile {
    /// How the translator carries the Japanese across: how register becomes
    /// Vietnamese pronouns, what is localised and what is glossed, whether
    /// honorifics are kept, how much the tone is sharpened. Kept apart from
    /// `voice`, which is how the Vietnamese itself sounds.
    #[serde(default)]
    pub approach: Vec<String>,
    /// How the prose sounds: register, sentence length, what it does with
    /// exclamations.
    #[serde(default)]
    pub voice: Vec<String>,
    /// Who calls whom what. Vietnamese forces a choice of pronoun on every
    /// line, the choice carries the relationship, and getting it wrong quietly
    /// rewrites the story — so this is the part most worth learning.
    #[serde(default)]
    pub address: Vec<String>,
    /// What the translator does with sound effects and shouting.
    #[serde(default)]
    pub sound_effects: Vec<String>,
    /// Names and recurring terms, as `[japanese, vietnamese]`. These have to
    /// stay settled across a volume, and a model left to itself will not settle
    /// them the same way twice. The Japanese side is empty when the profile was
    /// read from a Vietnamese edition alone.
    #[serde(default)]
    pub glossary: Vec<[String; 2]>,
    /// The recurring cast: who each character is, how they talk, and how they
    /// address each of the others. Pronouns in Vietnamese follow the pair of
    /// characters and their mood, so this is what keeps a translation's
    /// forms of address steady from page to page.
    #[serde(default)]
    pub characters: Vec<CharacterProfile>,
}

/// One recurring character, as the translation presents them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct CharacterProfile {
    /// Stable ASCII id, shared with the character scan and its face crops.
    pub id: String,
    /// The name the translation uses.
    pub name: String,
    /// The name as written in the original, when it was read.
    pub name_ja: String,
    /// Other names and titles the character goes by (`hoàng tử`, `Suguru`).
    pub aliases: Vec<String>,
    /// `male`, `female`, or empty when unclear.
    pub gender: String,
    /// `child`, `teen`, `young_adult`, `adult`, `middle_age`, `elder`, or empty.
    pub age_group: String,
    /// Who they are in the story.
    pub role: String,
    pub personality: String,
    /// How they talk: register, particles, verbal tics.
    pub speech: String,
    /// How they refer to themselves, with the mood when it varies.
    pub self_terms: Vec<String>,
    /// Pages they appear on.
    pub appearances: u32,
    pub relations: Vec<CharacterRelation>,
}

/// How one character stands to another and addresses them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct CharacterRelation {
    /// The other character's id.
    pub to: String,
    /// What the other character is to this one (`father`, `servant`, `rival`).
    pub relation: String,
    /// The pronouns and terms used, default first, then by mood.
    pub address: String,
}

impl CharacterProfile {
    /// One character as a few prompt lines.
    fn describe(&self, names: &std::collections::HashMap<&str, &str>) -> String {
        let mut head = format!("- {}", self.name);
        let mut also: Vec<&str> = Vec::new();
        if !self.name_ja.is_empty() {
            also.push(&self.name_ja);
        }
        also.extend(self.aliases.iter().map(String::as_str));
        if !also.is_empty() {
            head.push_str(&format!(" ({})", also.join(", ")));
        }
        let facts: Vec<&str> = [self.gender.as_str(), self.age_group.as_str()]
            .into_iter()
            .filter(|f| !f.is_empty())
            .collect();
        if !facts.is_empty() {
            head.push_str(&format!(" — {}", facts.join(", ").replace('_', " ")));
        }

        let mut lines = vec![head];
        for (label, value) in [
            ("Role", &self.role),
            ("Personality", &self.personality),
            ("Speech", &self.speech),
        ] {
            if !value.trim().is_empty() {
                lines.push(format!("  {label}: {}", value.trim()));
            }
        }
        if !self.self_terms.is_empty() {
            lines.push(format!(
                "  Refers to self as: {}",
                self.self_terms.join("; ")
            ));
        }
        for relation in &self.relations {
            let other = names
                .get(relation.to.as_str())
                .copied()
                .unwrap_or(relation.to.as_str());
            let what = if relation.relation.trim().is_empty() {
                String::new()
            } else {
                format!(" ({})", relation.relation.trim())
            };
            lines.push(format!("  → {other}{what}: {}", relation.address.trim()));
        }
        lines.join("\n")
    }
}

impl StyleProfile {
    pub fn is_empty(&self) -> bool {
        self.approach.is_empty()
            && self.voice.is_empty()
            && self.address.is_empty()
            && self.sound_effects.is_empty()
            && self.glossary.is_empty()
            && self.characters.is_empty()
    }

    /// Render the profile for a provider's story context.
    ///
    /// Phrased as how this series has been translated before, not as rules. The
    /// pronoun work already showed what happens when a heading promises the
    /// model a mandatory table: it obeys the table over the dialogue, and a line
    /// that plainly contradicts it comes out wrong.
    pub fn to_context(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }

        let mut out = vec![
            "How this series has been translated before, from its published \
             Vietnamese edition. Follow it where the page allows, and follow \
             the page where it does not."
                .to_string(),
        ];

        let section = |title: &str, items: &[String]| -> Option<String> {
            if items.is_empty() {
                return None;
            }
            let body: Vec<String> = items.iter().map(|item| format!("- {item}")).collect();
            Some(format!("{title}:\n{}", body.join("\n")))
        };

        out.extend(section("Translation approach", &self.approach));
        out.extend(section("Voice", &self.voice));
        out.extend(section("Forms of address", &self.address));
        out.extend(section("Sound effects", &self.sound_effects));

        if !self.glossary.is_empty() {
            let terms: Vec<String> = self
                .glossary
                .iter()
                .map(|[source, target]| {
                    if source.is_empty() {
                        format!("- {target}")
                    } else {
                        format!("- {source} → {target}")
                    }
                })
                .collect();
            out.push(format!(
                "Settled terms (use these spellings):\n{}",
                terms.join("\n")
            ));
        }

        if !self.characters.is_empty() {
            let names: std::collections::HashMap<&str, &str> = self
                .characters
                .iter()
                .map(|c| (c.id.as_str(), c.name.as_str()))
                .collect();
            let cast: Vec<String> = self.characters.iter().map(|c| c.describe(&names)).collect();
            out.push(format!(
                "Characters — who they are, how they speak, and how each addresses the others \
                 (default first, then by mood; follow the scene's mood):\n{}",
                cast.join("\n")
            ));
        }

        Some(out.join("\n\n"))
    }
}

/// Drop pairs that are probably not translations of each other, and spread the
/// rest across the volume.
///
/// The geometry that made a pair says how confidently two balloons were matched,
/// not whether the words inside them correspond — a page whose balloons all sit
/// in the same places scores well even where the match is wrong. The length
/// band is the cheap second opinion.
pub fn sample(pairs: &[SentencePair], limit: usize) -> Vec<&SentencePair> {
    let usable: Vec<&SentencePair> = pairs
        .iter()
        .filter(|pair| pair.confidence >= MIN_CONFIDENCE)
        .filter(|pair| {
            let source = pair.source.chars().count() as f32;
            if source == 0.0 {
                return false;
            }
            let ratio = pair.target.chars().count() as f32 / source;
            (MIN_LENGTH_RATIO..=MAX_LENGTH_RATIO).contains(&ratio)
        })
        .collect();

    if usable.len() <= limit || limit == 0 {
        return usable;
    }

    // Even steps through the volume rather than the first `limit`: the front of
    // a volume is covers and contents, and a profile learned from those would
    // describe everything except the dialogue.
    let stride = usable.len() as f64 / limit as f64;
    (0..limit)
        .map(|index| usable[((index as f64 * stride) as usize).min(usable.len() - 1)])
        .collect()
}

/// The pairs, numbered, for the model to read.
fn transcript(pairs: &[&SentencePair]) -> String {
    pairs
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            // Line breaks inside a balloon are typesetting, not sentence
            // structure, and leaving them in makes the transcript hard to read
            // as a list of pairs.
            let source = pair.source.replace('\n', " ");
            let target = pair.target.replace('\n', " ");
            format!("{}. {source}\n   → {target}", index + 1)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const SYSTEM_PROMPT: &str = "\
You study published manga translations and describe how the translator works.";

const INSTRUCTIONS: &str = "\
Above are Japanese lines from a manga and the Vietnamese a published \
translator made of them.

Describe how this translator writes, as JSON and nothing else:

{
  \"voice\": [],
  \"address\": [],
  \"sound_effects\": [],
  \"glossary\": [[\"japanese\", \"vietnamese\"]]
}

- voice: register, sentence length, particles and exclamations — at most 6 \
short observations, each one another translator could act on.
- address: which pronoun pairs this translation uses between which characters \
(tôi/cậu, ông/cháu, tao/mày, ...), and what each choice signals. This matters \
most: name the pair and say who uses it with whom. At most 8.
- sound_effects: what happens to sound effects and shouting — translated, left \
in katakana, romanised, dropped. At most 4.
- glossary: names and recurring terms whose Vietnamese spelling has settled. \
Only ones that actually recur. At most 30.

The Vietnamese was read off the page by OCR, which gets the words right but \
not the punctuation: a closing ! often comes back as 3, ' or [, and a ? as F. \
Say nothing about punctuation, and do not treat stray digits or brackets at \
the end of a line as part of the translation.

Base every line on what is in the pairs above. Leave a list empty rather than \
filling it with what is usually true of manga. Write the observations in \
English; keep Vietnamese words themselves in Vietnamese.";

/// Vietnamese personal pronouns, for checking a profile against the corpus it
/// claims to describe.
///
/// Not exhaustive, and it does not need to be: an entry is only ever dropped
/// for naming a pronoun from this list that the corpus does not contain, so a
/// pronoun missing from the list simply goes unchecked.
const PRONOUNS: &[&str] = &[
    "tôi",
    "tớ",
    "ta",
    "tao",
    "mình",
    "cậu",
    "bạn",
    "mày",
    "ngươi",
    "ông",
    "bà",
    "anh",
    "chị",
    "em",
    "cháu",
    "con",
    "chú",
    "bác",
    "cô",
    "dì",
    "cậu ấy",
    "sếp",
    "ngài",
    "thầy",
    "trò",
];

/// Drop anything the profile says that the pairs do not bear out.
///
/// A model reading a hundred and fifty lines will occasionally round them off
/// to what manga usually does rather than what this translation did. Measured
/// over repeated runs, a corpus read by a local OCR produced an invented
/// `Ông/Cháu` in roughly one run in four — and `cháu` appears nowhere in it.
/// That is the failure this catches: not a subtly wrong reading, but a word
/// that is simply not there.
///
/// Only claims that name a checkable word are checked. An observation about
/// pace or register names nothing and stands as written.
pub fn ground(profile: &mut StyleProfile, pairs: &[SentencePair]) {
    let corpus = pairs
        .iter()
        .map(|pair| pair.target.replace('\n', " "))
        .collect::<Vec<_>>()
        .join(" || ");
    ground_in(profile, &corpus);
}

/// [`ground`], against any Vietnamese text the profile was learned from.
pub fn ground_in(profile: &mut StyleProfile, corpus: &str) {
    let corpus = corpus.to_lowercase();

    profile
        .address
        .retain(|entry| pronouns_named(entry).all(|word| mentions(&corpus, word)));

    // A settled spelling that never occurs was not settled by this translator.
    profile
        .glossary
        .retain(|[_, target]| corpus.contains(&target.to_lowercase()));
}

/// The pronouns an address entry names.
fn pronouns_named(entry: &str) -> impl Iterator<Item = &'static str> + '_ {
    let lowered = entry.to_lowercase();
    PRONOUNS
        .iter()
        .copied()
        .filter(move |word| mentions(&lowered, word))
}

/// Whole-word search: `ta` must not match inside `tao`, nor `em` inside `thèm`.
fn mentions(haystack: &str, word: &str) -> bool {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric());
    let mut from = 0;
    while let Some(at) = haystack[from..].find(word) {
        let start = from + at;
        let end = start + word.len();
        if boundary(haystack[..start].chars().next_back())
            && boundary(haystack[end..].chars().next())
        {
            return true;
        }
        from = start + word.len().max(1);
    }
    false
}

/// The request that turns a corpus into a profile, as `(system, user)`.
///
/// Split from the call itself so the caller can send it through whichever
/// model the app has loaded; `None` when no pair is worth learning from.
pub fn request(pairs: &[SentencePair]) -> Option<(&'static str, String)> {
    let chosen = sample(pairs, SAMPLE_SIZE);
    if chosen.is_empty() {
        return None;
    }
    Some((
        SYSTEM_PROMPT,
        format!("{}\n\n{INSTRUCTIONS}", transcript(&chosen)),
    ))
}

/// Turn the model's reply into a profile, keeping only what the pairs bear out.
pub fn from_reply(reply: &str, pairs: &[SentencePair]) -> anyhow::Result<StyleProfile> {
    let mut profile = parse(reply)?;
    ground(&mut profile, pairs);
    Ok(profile)
}

/// Read the pairs and keep what they teach, through a provider directly.
pub async fn learn(
    provider: &dyn koharu_llm::providers::AnyProvider,
    pairs: &[SentencePair],
    model: &str,
) -> anyhow::Result<StyleProfile> {
    let (system, user) =
        request(pairs).ok_or_else(|| anyhow::anyhow!("no sentence pairs worth learning from"))?;
    let reply = provider.complete(system, &user, model).await?;
    from_reply(&reply, pairs)
}

/// Where the profile the translator should follow is kept.
///
/// One profile at a time, beside the character library: the reference volume a
/// profile was learned from is rarely the folder being translated, so the
/// profile cannot simply be looked up next to the pages.
pub fn active_profile_path() -> std::path::PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("koharu")
        .join("style_profile.json")
}

/// The profile translations currently follow, if one has been chosen.
pub fn load_active() -> Option<StyleProfile> {
    let bytes = std::fs::read(active_profile_path()).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(profile) => Some(profile),
        Err(err) => {
            tracing::warn!("ignoring unreadable style profile: {err}");
            None
        }
    }
}

/// Make `profile` the one translations follow; `None` stops following any.
pub fn set_active(profile: Option<&StyleProfile>) -> anyhow::Result<()> {
    let path = active_profile_path();
    match profile {
        Some(profile) => {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&path, serde_json::to_vec_pretty(profile)?)?;
        }
        None => match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        },
    }
    Ok(())
}

/// Pull the profile out of the model's reply.
///
/// Two things are forgiven, because neither says anything about whether the
/// model did the work: JSON fenced in markdown, and observations handed back as
/// nested arrays or objects where a sentence was asked for. Failing on either
/// would make the flow flaky for a cosmetic reason.
fn parse(reply: &str) -> anyhow::Result<StyleProfile> {
    let trimmed = reply.trim();
    let body = match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(start), Some(end)) if end > start => &trimmed[start..=end],
        _ => anyhow::bail!("no JSON object in the reply: {trimmed:.200}"),
    };
    let raw: serde_json::Value = serde_json::from_str(body)?;

    let sound_effects = if raw["soundEffects"].is_array() {
        &raw["soundEffects"]
    } else {
        &raw["sound_effects"]
    };
    Ok(StyleProfile {
        approach: observations(&raw["approach"]),
        voice: observations(&raw["voice"]),
        address: observations(&raw["address"]),
        sound_effects: observations(sound_effects),
        glossary: glossary(&raw["glossary"]),
        characters: characters(&raw["characters"]),
    })
}

/// Characters as the model wrote them. A malformed entry is dropped rather than
/// failing the whole profile; one without a name says nothing to follow.
fn characters(value: &serde_json::Value) -> Vec<CharacterProfile> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut taken = std::collections::HashSet::new();
    items
        .iter()
        .filter_map(|item| serde_json::from_value::<CharacterProfile>(item.clone()).ok())
        .filter(|c| !c.name.trim().is_empty())
        .map(|mut c| {
            if c.id.trim().is_empty() {
                c.id = slug(&c.name);
            }
            let base = c.id.clone();
            let mut n = 2;
            while !taken.insert(c.id.clone()) {
                c.id = format!("{base}-{n}");
                n += 1;
            }
            c
        })
        .collect()
}

/// An ASCII id from a name: `Tổng tư lệnh` → `tong-tu-lenh`.
pub fn slug(name: &str) -> String {
    let folded: String = name
        .chars()
        .map(|c| match c {
            'đ' | 'Đ' => 'd',
            _ => c,
        })
        .flat_map(|c| {
            // Strip Vietnamese tone and vowel marks by mapping to the base letter.
            const FROM: &str = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ";
            const TO: &str = "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyy";
            let lower = c.to_lowercase().next().unwrap_or(c);
            let base = FROM
                .chars()
                .position(|f| f == lower)
                .and_then(|i| TO.chars().nth(i))
                .unwrap_or(lower);
            Some(base)
        })
        .collect();
    let mut out = String::new();
    for c in folded.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "character".to_string()
    } else {
        out
    }
}

/// [`parse`], for other readers of the same reply shape.
pub(crate) fn parse_profile(reply: &str) -> anyhow::Result<StyleProfile> {
    parse(reply)
}

/// Flatten whatever the model put in a list into one sentence per entry.
fn observations(value: &serde_json::Value) -> Vec<String> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .map(flatten)
        .filter(|line| !line.is_empty())
        .collect()
}

/// One JSON value as the sentence it was meant to be. An object's keys are
/// dropped: they are the model's own labels, and the values already read as
/// prose.
fn flatten(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.trim().to_string(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(flatten)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" — "),
        serde_json::Value::Object(fields) => fields
            .values()
            .map(flatten)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" — "),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Glossary entries arrive as `["日本語", "tiếng Việt"]`, but a model that
/// decided on `{"japanese": ..., "vietnamese": ...}` meant the same thing.
fn glossary(value: &serde_json::Value) -> Vec<[String; 2]> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|entry| {
            let pair: Vec<String> = match entry {
                serde_json::Value::Array(parts) => parts.iter().map(flatten).collect(),
                serde_json::Value::Object(fields) => fields.values().map(flatten).collect(),
                _ => return None,
            };
            // An empty Japanese side is a term read from a Vietnamese edition
            // alone; an empty Vietnamese side says nothing to follow.
            match pair.as_slice() {
                [source, target] if !target.is_empty() => Some([source.clone(), target.clone()]),
                [target] if !target.is_empty() => Some([String::new(), target.clone()]),
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(source: &str, target: &str, confidence: f32) -> SentencePair {
        SentencePair {
            source: source.to_string(),
            target: target.to_string(),
            page: "p.jpg".to_string(),
            target_box: [0.0; 4],
            confidence,
        }
    }

    #[test]
    fn a_fenced_reply_still_parses() {
        let profile = parse("```json\n{\"voice\": [\"ngắn gọn\"]}\n```").unwrap();
        assert_eq!(profile.voice, ["ngắn gọn"]);
    }

    #[test]
    fn a_reply_with_no_json_is_an_error_rather_than_an_empty_profile() {
        assert!(parse("Tôi không chắc.").is_err());
    }

    /// Models answer the same question in different shapes from one run to the
    /// next. A profile that only parses when the shape is exactly as asked
    /// would work on Monday and not on Tuesday.
    #[test]
    fn an_observation_handed_back_as_a_nested_list_still_reads_as_a_sentence() {
        let profile = parse(r#"{"address": [["tôi/cậu", "giữa hai người bạn"]]}"#).unwrap();
        assert_eq!(profile.address, ["tôi/cậu — giữa hai người bạn"]);
    }

    #[test]
    fn an_observation_handed_back_as_an_object_still_reads_as_a_sentence() {
        let profile =
            parse(r#"{"voice": [{"note": "Câu ngắn", "why": "hợp khung thoại"}]}"#).unwrap();
        assert_eq!(profile.voice, ["Câu ngắn — hợp khung thoại"]);
    }

    #[test]
    fn a_glossary_entry_named_by_key_means_the_same_as_one_given_as_a_pair() {
        let profile =
            parse(r#"{"glossary": [{"japanese": "長官", "vietnamese": "Sếp"}]}"#).unwrap();
        assert_eq!(profile.glossary, [["長官".to_string(), "Sếp".to_string()]]);
    }

    /// A half-written entry would put an arrow with nothing on one side of it
    /// into the prompt.
    #[test]
    fn a_glossary_entry_with_nothing_to_write_is_dropped() {
        let profile = parse(r#"{"glossary": [["長官", ""], [], ["エ", "Hả"]]}"#).unwrap();
        assert_eq!(profile.glossary, [["エ".to_string(), "Hả".to_string()]]);
    }

    /// A Vietnamese edition read on its own has no Japanese to pair a name
    /// with, and the name is still worth keeping settled.
    #[test]
    fn a_glossary_entry_without_japanese_is_kept_and_rendered_alone() {
        let profile = parse(r#"{"glossary": [["", "Kinnikuman"], ["Meat"]]}"#).unwrap();
        assert_eq!(
            profile.glossary,
            [
                [String::new(), "Kinnikuman".to_string()],
                [String::new(), "Meat".to_string()]
            ]
        );
        let context = profile.to_context().unwrap();
        assert!(context.contains("\n- Kinnikuman\n"), "{context}");
        assert!(!context.contains("→ Kinnikuman"));
    }

    #[test]
    fn the_translation_approach_is_read_and_rendered_first() {
        let profile = parse(
            r#"{"approach": ["Keigo becomes ngài/tôi"], "voice": ["Lóng mạng"], "soundEffects": ["Giữ SFX vẽ tay"]}"#,
        )
        .unwrap();
        assert_eq!(profile.sound_effects, ["Giữ SFX vẽ tay"]);
        let context = profile.to_context().unwrap();
        let approach = context.find("Translation approach").unwrap();
        let voice = context.find("Voice").unwrap();
        assert!(approach < voice);
    }

    #[test]
    fn missing_sections_come_back_empty_rather_than_failing() {
        let profile = parse("{\"voice\": [\"a\"]}").unwrap();
        assert!(profile.address.is_empty());
        assert!(profile.glossary.is_empty());
    }

    #[test]
    fn a_badly_matched_pair_is_left_out() {
        let pairs = vec![
            pair("長官!!", "SẾP ƠI!!", 0.9),
            // Four Japanese characters could not have become this much
            // Vietnamese; the balloons were matched wrongly.
            pair("エ?", "TRONG CÁI TÌNH THẾ DẦU SÔI LỬA BỎNG NÀY", 0.9),
            // Matched, but too far apart to trust.
            pair("では…", "NÀO...", 0.1),
        ];

        let kept = sample(&pairs, 10);

        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].source, "長官!!");
    }

    #[test]
    fn the_sample_is_spread_across_the_volume_not_taken_from_the_front() {
        let pairs: Vec<SentencePair> = (0..100)
            .map(|i| pair("あいうえお", &format!("dòng số {i} đây"), 0.9))
            .collect();

        let kept = sample(&pairs, 10);

        assert_eq!(kept.len(), 10);
        assert_eq!(kept[0].target, "dòng số 0 đây");
        assert_eq!(kept[9].target, "dòng số 90 đây");
    }

    /// The failure this exists for: a corpus with no `cháu` in it cannot have
    /// taught anyone to say `Ông/Cháu`.
    #[test]
    fn a_pronoun_pair_the_corpus_never_uses_is_dropped() {
        let pairs = vec![pair("長官!!", "SẾP ƠI! TÔI BÁO CẬU BIẾT", 0.9)];
        let mut profile = StyleProfile {
            address: vec![
                "Sếp/Tôi — nhân viên với cấp trên".to_string(),
                "Ông/Cháu — người già với trẻ nhỏ".to_string(),
            ],
            ..Default::default()
        };

        ground(&mut profile, &pairs);

        assert_eq!(profile.address, ["Sếp/Tôi — nhân viên với cấp trên"]);
    }

    /// An observation that names no pronoun cannot be checked, and must not be
    /// dropped for it.
    #[test]
    fn an_observation_naming_no_pronoun_survives_grounding() {
        let pairs = vec![pair("あ", "MỘT HAI BA", 0.9)];
        let mut profile = StyleProfile {
            voice: vec!["Câu ngắn, nhịp nhanh".to_string()],
            address: vec!["Xưng hô thay đổi theo tình huống".to_string()],
            ..Default::default()
        };

        ground(&mut profile, &pairs);

        assert_eq!(profile.voice, ["Câu ngắn, nhịp nhanh"]);
        assert_eq!(profile.address, ["Xưng hô thay đổi theo tình huống"]);
    }

    #[test]
    fn a_settled_term_the_corpus_never_uses_is_dropped() {
        let pairs = vec![pair("キン肉マン", "KINNIKUMAN ĐÂY!", 0.9)];
        let mut profile = StyleProfile {
            glossary: vec![
                ["キン肉マン".to_string(), "Kinnikuman".to_string()],
                ["長官".to_string(), "Trưởng quan".to_string()],
            ],
            ..Default::default()
        };

        ground(&mut profile, &pairs);

        assert_eq!(
            profile.glossary,
            [["キン肉マン".to_string(), "Kinnikuman".to_string()]]
        );
    }

    /// `ta` sits inside `tao`, and `em` inside `thèm`. Matching on substrings
    /// would let a pronoun the corpus never uses pass as grounded.
    #[test]
    fn a_pronoun_hiding_inside_a_longer_word_does_not_count_as_used() {
        let pairs = vec![pair("あ", "TAO KHÔNG THÈM CHẤP", 0.9)];
        let mut profile = StyleProfile {
            address: vec![
                "Ta/Em — hai người thân thiết".to_string(),
                "Tao/Mày — lúc cáu".to_string(),
            ],
            ..Default::default()
        };

        ground(&mut profile, &pairs);

        // `tao` is there but `mày` is not, so that entry goes too.
        assert!(profile.address.is_empty());
    }

    /// On the full reference volume, 62 lines ended in a `3` that was really a
    /// `!`, and the profile described that as the translator's "teen style
    /// punctuation". The model has to be told the punctuation is not theirs.
    #[test]
    fn the_request_warns_that_ocr_punctuation_is_unreliable() {
        let pairs = vec![pair("長官!!", "SẾP ƠI3", 0.9)];
        let (_, user) = request(&pairs).unwrap();
        assert!(user.contains("Say nothing about punctuation"));
        assert!(user.contains("SẾP ƠI3"));
    }

    #[test]
    fn characters_are_read_given_ids_and_rendered_with_their_relations() {
        let profile = parse(
            r#"{"characters": [
                {"name": "Kinnikuman", "nameJa": "キン肉マン", "aliases": ["Suguru", "hoàng tử"],
                 "gender": "male", "ageGroup": "young_adult", "role": "Prince of planet Kinniku",
                 "selfTerms": ["ta (posturing)", "tớ (pleading)"],
                 "relations": [{"to": "meat", "relation": "servant", "address": "ta/ngươi; cậu when friendly"}]},
                {"id": "meat", "name": "Meat", "relations": [{"to": "kinnikuman", "relation": "master", "address": "tôi/ngài"}]},
                {"name": ""},
                "not a character"
            ]}"#,
        )
        .unwrap();

        assert_eq!(profile.characters.len(), 2);
        assert_eq!(profile.characters[0].id, "kinnikuman");
        let context = profile.to_context().unwrap();
        assert!(
            context.contains("- Kinnikuman (キン肉マン, Suguru, hoàng tử) — male, young adult"),
            "{context}"
        );
        assert!(
            context.contains("  → Meat (servant): ta/ngươi; cậu when friendly"),
            "{context}"
        );
        assert!(
            context.contains("  → Kinnikuman (master): tôi/ngài"),
            "{context}"
        );
        assert!(context.contains("Refers to self as: ta (posturing); tớ (pleading)"));
    }

    #[test]
    fn ids_are_ascii_and_do_not_collide() {
        assert_eq!(slug("Tổng tư lệnh"), "tong-tu-lenh");
        assert_eq!(slug("Đại vương Kinniku"), "dai-vuong-kinniku");
        assert_eq!(slug("キン肉マン"), "character");
        let profile = parse(r#"{"characters": [{"name": "Meat"}, {"name": "meat"}]}"#).unwrap();
        let ids: Vec<&str> = profile.characters.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["meat", "meat-2"]);
    }

    #[test]
    fn an_empty_profile_contributes_no_context() {
        assert!(StyleProfile::default().to_context().is_none());
    }

    #[test]
    fn a_profile_renders_every_section_it_has() {
        let profile = StyleProfile {
            approach: vec![],
            voice: vec!["Câu ngắn".to_string()],
            address: vec!["Meat dùng tớ/cậu với Terryman".to_string()],
            sound_effects: vec![],
            glossary: vec![["キン肉マン".to_string(), "Kinnikuman".to_string()]],
            characters: vec![],
        };

        let context = profile.to_context().unwrap();

        assert!(context.contains("- Câu ngắn"));
        assert!(context.contains("Meat dùng tớ/cậu với Terryman"));
        assert!(context.contains("キン肉マン → Kinnikuman"));
        // The empty section leaves no stray heading behind.
        assert!(!context.contains("Sound effects"));
    }
}
