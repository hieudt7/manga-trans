//! Learning dictionary for sound effects / onomatopoeia.
//!
//! Manga reuses the same small set of katakana SFX over and over (ドン, ガシャン,
//! ザー…). Translating them through the LLM every time costs a full system prompt
//! per block for a three-character word, and gives inconsistent results across
//! chapters. This module caches each SFX translation the first time the LLM
//! produces it, keyed by target language, and serves it from disk afterwards.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// SFX are short; anything longer is dialogue that needs real context.
const MAX_SFX_CHARS: usize = 8;
/// Fraction of "letter" characters that must be katakana to call it an SFX.
const MIN_KATAKANA_RATIO: f32 = 0.8;

fn dict_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("sfx_dict.json")
}

fn is_katakana(c: char) -> bool {
    ('\u{30A0}'..='\u{30FF}').contains(&c) || ('\u{31F0}'..='\u{31FF}').contains(&c)
}

fn is_kanji(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c) || ('\u{3400}'..='\u{4DBF}').contains(&c)
}

/// Marks that decorate an SFX without carrying meaning: elongation, repetition,
/// small tsu, and the punctuation manga uses for emphasis.
fn is_sfx_decoration(c: char) -> bool {
    matches!(
        c,
        'ー' | '〜'
            | '~'
            | 'っ'
            | 'ッ'
            | '！'
            | '!'
            | '？'
            | '?'
            | '…'
            | '・'
            | '、'
            | '。'
            | '「'
            | '」'
            | '♪'
            | '☆'
            | '★'
            | '.'
            | ','
    ) || c.is_whitespace()
}

/// Whether `text` looks like a katakana sound effect that is safe to memoise.
///
/// Deliberately stricter than the retry heuristic in the pipeline: hiragana-only
/// and kanji-bearing text is rejected, because short hiragana strings are usually
/// real dialogue (はい, そう, まだ) whose translation depends on context and must
/// not be frozen into a dictionary.
pub fn is_sfx(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    let chars: Vec<char> = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() || chars.len() > MAX_SFX_CHARS {
        return false;
    }
    if chars.iter().copied().any(is_kanji) {
        return false;
    }

    let letters: Vec<char> = chars
        .iter()
        .copied()
        .filter(|c| !is_sfx_decoration(*c))
        .collect();
    if letters.is_empty() {
        return false;
    }

    let katakana = letters.iter().copied().filter(|c| is_katakana(*c)).count();
    katakana as f32 / letters.len() as f32 >= MIN_KATAKANA_RATIO
}

/// Collapse whitespace so that line-broken SFX hit the same entry as inline ones.
fn normalize_key(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join("")
}

type Entries = HashMap<String, HashMap<String, String>>;

pub struct SfxDictionary {
    /// language tag → normalised source → translation
    entries: Mutex<Entries>,
    path: PathBuf,
}

impl SfxDictionary {
    fn load(path: PathBuf) -> Self {
        let entries = match std::fs::read_to_string(&path) {
            Ok(json) => serde_json::from_str::<Entries>(&json).unwrap_or_else(|err| {
                tracing::warn!(%err, "sfx dictionary is corrupt, starting empty");
                Entries::new()
            }),
            Err(_) => Entries::new(),
        };
        let count: usize = entries.values().map(|m| m.len()).sum();
        tracing::info!(count, path = %path.display(), "sfx dictionary loaded");
        Self {
            entries: Mutex::new(entries),
            path,
        }
    }

    /// Cached translation for `source` in `language`, if it was learned before.
    pub fn get(&self, language: &str, source: &str) -> Option<String> {
        if !is_sfx(source) {
            return None;
        }
        let key = normalize_key(source);
        let entries = self.entries.lock().ok()?;
        entries.get(language)?.get(&key).cloned()
    }

    /// Learn `translation` for `source`. Existing entries are kept, so the first
    /// translation of a given SFX becomes the canonical one for the whole series.
    pub fn learn(&self, language: &str, source: &str, translation: &str) {
        let translation = translation.trim();
        if translation.is_empty() || !is_sfx(source) {
            return;
        }
        // A parenthetical description ("(Tiếng búa đập)") is exactly what we do
        // not want to memoise as the canonical rendering.
        if crate::is_sfx_description(translation) {
            return;
        }

        let key = normalize_key(source);
        let mut changed = false;
        if let Ok(mut entries) = self.entries.lock() {
            let per_language = entries.entry(language.to_string()).or_default();
            if !per_language.contains_key(&key) {
                per_language.insert(key.clone(), translation.to_string());
                changed = true;
            }
        }

        if changed {
            tracing::info!(language, source = %key, translation, "sfx learned");
            self.save();
        }
    }

    fn save(&self) {
        let Ok(entries) = self.entries.lock() else {
            return;
        };
        let Ok(json) = serde_json::to_string_pretty(&*entries) else {
            return;
        };
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(err) = std::fs::write(&self.path, json) {
            tracing::warn!(%err, path = %self.path.display(), "failed to save sfx dictionary");
        }
    }

    /// Number of learned entries per language, for diagnostics.
    pub fn len(&self, language: &str) -> usize {
        self.entries
            .lock()
            .ok()
            .and_then(|e| e.get(language).map(|m| m.len()))
            .unwrap_or(0)
    }
}

/// Process-wide dictionary, backed by `<data_local>/koharu/sfx_dict.json`.
pub fn sfx_dictionary() -> &'static SfxDictionary {
    static DICT: OnceLock<SfxDictionary> = OnceLock::new();
    DICT.get_or_init(|| SfxDictionary::load(dict_path()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn katakana_sfx_is_detected() {
        assert!(is_sfx("ドン"));
        assert!(is_sfx("ガシャーン！"));
        assert!(is_sfx("ザーッ"));
        assert!(is_sfx("ドドドド"));
    }

    #[test]
    fn dialogue_is_not_memoised() {
        // Short hiragana is dialogue, not SFX — its translation is contextual.
        assert!(!is_sfx("はい"));
        assert!(!is_sfx("そうか"));
        // Kanji means meaning-bearing text.
        assert!(!is_sfx("行くぞ"));
        // Too long to be an SFX.
        assert!(!is_sfx("ドンドンドンドンドン"));
        // Pure punctuation has no letters at all.
        assert!(!is_sfx("……"));
        assert!(!is_sfx(""));
    }

    #[test]
    fn learn_then_get_roundtrips_and_keeps_first_translation() {
        let dir = tempfile::tempdir().unwrap();
        let dict = SfxDictionary::load(dir.path().join("sfx.json"));

        dict.learn("vi-VN", "ドン", "BÙNG!");
        assert_eq!(dict.get("vi-VN", "ドン").as_deref(), Some("BÙNG!"));

        // Second sighting must not overwrite the canonical rendering.
        dict.learn("vi-VN", "ドン", "ẦM!");
        assert_eq!(dict.get("vi-VN", "ドン").as_deref(), Some("BÙNG!"));

        // Language-scoped.
        assert!(dict.get("en-US", "ドン").is_none());

        // Whitespace and line breaks collapse to the same entry.
        assert_eq!(dict.get("vi-VN", " ド ン \n").as_deref(), Some("BÙNG!"));
    }

    #[test]
    fn parenthetical_descriptions_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let dict = SfxDictionary::load(dir.path().join("sfx.json"));
        dict.learn("vi-VN", "ガン", "(Tiếng búa đập)");
        assert!(dict.get("vi-VN", "ガン").is_none());
    }

    #[test]
    fn survives_reload_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sfx.json");
        SfxDictionary::load(path.clone()).learn("vi-VN", "ザー", "ÀO ÀO");
        let reloaded = SfxDictionary::load(path);
        assert_eq!(reloaded.get("vi-VN", "ザー").as_deref(), Some("ÀO ÀO"));
        assert_eq!(reloaded.len("vi-VN"), 1);
    }
}
