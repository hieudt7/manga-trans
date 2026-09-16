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
pub struct StyleProfile {
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
    /// them the same way twice.
    #[serde(default)]
    pub glossary: Vec<[String; 2]>,
}

impl StyleProfile {
    pub fn is_empty(&self) -> bool {
        self.voice.is_empty()
            && self.address.is_empty()
            && self.sound_effects.is_empty()
            && self.glossary.is_empty()
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

        out.extend(section("Voice", &self.voice));
        out.extend(section("Forms of address", &self.address));
        out.extend(section("Sound effects", &self.sound_effects));

        if !self.glossary.is_empty() {
            let terms: Vec<String> = self
                .glossary
                .iter()
                .map(|[source, target]| format!("- {source} → {target}"))
                .collect();
            out.push(format!(
                "Settled terms (use these spellings):\n{}",
                terms.join("\n")
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
        .join(" || ")
        .to_lowercase();

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

/// Read the pairs and keep what they teach.
pub async fn learn(
    provider: &dyn koharu_llm::providers::AnyProvider,
    pairs: &[SentencePair],
    model: &str,
) -> anyhow::Result<StyleProfile> {
    let chosen = sample(pairs, SAMPLE_SIZE);
    if chosen.is_empty() {
        anyhow::bail!("no sentence pairs worth learning from");
    }

    let prompt = format!("{}\n\n{INSTRUCTIONS}", transcript(&chosen));
    let reply = provider.complete(SYSTEM_PROMPT, &prompt, model).await?;
    let mut profile = parse(&reply)?;
    ground(&mut profile, pairs);
    Ok(profile)
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

    Ok(StyleProfile {
        voice: observations(&raw["voice"]),
        address: observations(&raw["address"]),
        sound_effects: observations(&raw["sound_effects"]),
        glossary: glossary(&raw["glossary"]),
    })
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
            match pair.as_slice() {
                [source, target] if !source.is_empty() && !target.is_empty() => {
                    Some([source.clone(), target.clone()])
                }
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
    fn an_incomplete_glossary_entry_is_dropped() {
        let profile = parse(r#"{"glossary": [["長官"], ["", "Sếp"], ["エ", "Hả"]]}"#).unwrap();
        assert_eq!(profile.glossary, [["エ".to_string(), "Hả".to_string()]]);
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

    #[test]
    fn an_empty_profile_contributes_no_context() {
        assert!(StyleProfile::default().to_context().is_none());
    }

    #[test]
    fn a_profile_renders_every_section_it_has() {
        let profile = StyleProfile {
            voice: vec!["Câu ngắn".to_string()],
            address: vec!["Meat dùng tớ/cậu với Terryman".to_string()],
            sound_effects: vec![],
            glossary: vec![["キン肉マン".to_string(), "Kinnikuman".to_string()]],
        };

        let context = profile.to_context().unwrap();

        assert!(context.contains("- Câu ngắn"));
        assert!(context.contains("Meat dùng tớ/cậu với Terryman"));
        assert!(context.contains("キン肉マン → Kinnikuman"));
        // The empty section leaves no stray heading behind.
        assert!(!context.contains("Sound effects"));
    }
}
