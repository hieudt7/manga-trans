//! Learning a translator's style by looking at the pages.
//!
//! The OCR route reads one balloon at a time with no picture around it, so it
//! can say what was written but not who said it to whom, or in what mood — and
//! in Vietnamese the pronoun carries exactly that. Compared on a 94-page volume
//! against a reading done by eye, the OCR profile had the direction of the main
//! character's pronouns wrong, invented an `ông/cháu` pair nobody uses, and
//! missed kept honorifics, regional words, localised jokes and the translator's
//! bracketed notes entirely, though its own corpus contained all of them.
//!
//! So this route does what the reading by eye did. Each page — the original and
//! its translation side by side, or the translation alone — goes to a model
//! that can see, which writes structured notes: every line with its speaker and
//! listener, the forms of address with their mood, and what the translator did.
//! One more call reads every page's notes and writes the profile.

use serde::{Deserialize, Serialize};

use super::style::{self, StyleProfile};

/// One line of text on the page, as the reader saw it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Line {
    pub speaker: String,
    pub listener: String,
    /// dialogue, thought, narration, sfx, sign, note
    pub kind: String,
    pub mood: String,
    /// The Vietnamese, verbatim, with its line breaks.
    pub vi: String,
    /// The Japanese it translates, when the original was shown.
    pub ja: String,
}

/// A form of address used on the page.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Address {
    pub speaker: String,
    pub listener: String,
    /// How the speaker refers to themself (`tôi`, `ta`, `tớ`…), or empty.
    pub self_term: String,
    /// How the speaker refers to or calls the listener (`ngài`, `cậu`, `hoàng tử`…).
    pub other_term: String,
    pub mood: String,
    pub quote: String,
}

/// A pair of things the translator changed, with what was made of it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Rendering {
    pub source: String,
    pub vi: String,
    pub note: String,
}

/// Everything a reader noted about one page.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PageNotes {
    pub page: String,
    /// Whether the original page was shown alongside the translation.
    pub with_raw: bool,
    pub summary: String,
    pub lines: Vec<Line>,
    pub address: Vec<Address>,
    /// How the Japanese register became Vietnamese; only with the original.
    pub register: Vec<Rendering>,
    pub voice: Vec<String>,
    pub localisation: Vec<Rendering>,
    pub translator_notes: Vec<String>,
    pub sound_effects: Vec<String>,
    /// Names and terms, Japanese (when shown) → Vietnamese.
    pub terms: Vec<Rendering>,
    pub unreadable: Vec<String>,
}

pub const PAGE_SYSTEM: &str = "\
You read manga pages the way a careful editor does: you look at the artwork to \
see who is speaking to whom and in what mood, and you read every balloon, \
caption and sign exactly as it is written. You are studying how a published \
Vietnamese translation was made.";

const PAGE_FORMAT: &str = r#"
Reply with one JSON object and nothing else:

{
  "summary": "what happens on the page, one sentence, in Vietnamese",
  "lines": [
    {"speaker": "…", "listener": "…", "kind": "dialogue|thought|narration|sfx|sign|note",
     "mood": "…", "vi": "the Vietnamese exactly as lettered, keep line breaks as \n", "ja": "…"}
  ],
  "address": [
    {"speaker": "…", "listener": "…", "selfTerm": "…", "otherTerm": "…", "mood": "…", "quote": "…"}
  ],
  "register": [{"source": "Japanese expression", "vi": "what it became", "note": "…"}],
  "voice": ["slang, regional words, idioms, particles, jokes — with the words quoted"],
  "localisation": [{"source": "…", "vi": "…", "note": "…"}],
  "translatorNotes": ["anything the translator explained in brackets, quoted"],
  "soundEffects": ["what was left in Japanese, translated, glossed or romanised"],
  "terms": [{"source": "Japanese name or term", "vi": "…", "note": ""}],
  "unreadable": ["describe any balloon you could not read"]
}

Rules:
- Pages are read right to left; a spread is two pages.
- Name characters the way the page does. If you cannot tell who a character is,
  describe them briefly ("cô gái tóc dài", "người đeo kính") and keep that
  description consistent within the page.
- "lines": every piece of text on the page, in reading order. Quote the
  Vietnamese exactly, including its punctuation and capitals. Never correct or
  complete it. If a balloon is unreadable, leave it out of "lines" and list it
  under "unreadable".
- "address": one entry for every line in which the speaker refers to themself or
  addresses someone — pronouns, kinship terms, titles, honorifics, names with
  -chan/-kun/-san. The speaker and listener come from the artwork, not from
  guessing. Record the mood of that line; the same two characters often change
  pronouns with their mood, and that change is what matters most.
- Only note what is on this page. Empty lists are fine."#;

const WITH_RAW: &str = "\
The first image is the original Japanese page; the second is the same page from \
the published Vietnamese translation.

Fill \"ja\" for each line with the Japanese it translates. In \"register\", note \
how the Japanese speech level became Vietnamese — keigo, です/ます, あなた, \
わたくし, plain だ, さま/くん/ちゃん — and where the translation is harsher, \
softer, or adds something the original does not say. In \"terms\" and \
\"localisation\", give the Japanese as written on the original page.";

const WITHOUT_RAW: &str = "\
The image is a page from a published Vietnamese translation of a Japanese \
manga. The original is not available: leave \"ja\" empty and \"register\" \
empty, and give \"terms\" with an empty \"source\".";

/// The request for one page, as `(system, user)`.
pub fn page_request(with_raw: bool) -> (&'static str, String) {
    let lead = if with_raw { WITH_RAW } else { WITHOUT_RAW };
    (PAGE_SYSTEM, format!("{lead}\n{PAGE_FORMAT}"))
}

/// Pull one page's notes out of the reader's reply.
pub fn parse_page(reply: &str, page: &str, with_raw: bool) -> anyhow::Result<PageNotes> {
    let body = json_object(reply)?;
    let mut notes: PageNotes = serde_json::from_str(body)?;
    notes.page = page.to_string();
    notes.with_raw = with_raw;
    notes.lines.retain(|line| !line.vi.trim().is_empty());
    Ok(notes)
}

fn json_object(reply: &str) -> anyhow::Result<&str> {
    let trimmed = reply.trim();
    match (trimmed.find('{'), trimmed.rfind('}')) {
        (Some(start), Some(end)) if end > start => Ok(&trimmed[start..=end]),
        _ => anyhow::bail!("no JSON object in the reply: {trimmed:.200}"),
    }
}

/// All the notes, compacted for the synthesis call.
///
/// One line of text per observation, so a whole volume fits in a request —
/// a 94-page volume comes to a few tens of thousands of tokens.
pub fn transcript(pages: &[PageNotes]) -> String {
    let mut out = String::new();
    for page in pages {
        out.push_str(&format!("## {} — {}\n", page.page, page.summary.trim()));
        for line in &page.lines {
            let who = match (line.speaker.trim(), line.listener.trim()) {
                ("", _) => String::new(),
                (s, "") => format!("{s}: "),
                (s, l) => format!("{s} → {l}: "),
            };
            let mood = if line.mood.trim().is_empty() {
                String::new()
            } else {
                format!(" [{}]", line.mood.trim())
            };
            let ja = if line.ja.trim().is_empty() {
                String::new()
            } else {
                format!("  ⟵ {}", one_line(&line.ja))
            };
            out.push_str(&format!(
                "- ({}) {who}{}{mood}{ja}\n",
                if line.kind.is_empty() {
                    "?"
                } else {
                    &line.kind
                },
                one_line(&line.vi)
            ));
        }
        for a in &page.address {
            out.push_str(&format!(
                "- ADDRESS {} → {}: self «{}» other «{}» [{}] “{}”\n",
                a.speaker,
                a.listener,
                a.self_term,
                a.other_term,
                a.mood,
                one_line(&a.quote)
            ));
        }
        let renderings = |label: &str, items: &[Rendering], out: &mut String| {
            for r in items {
                out.push_str(&format!(
                    "- {label}: {} → {}{}\n",
                    one_line(&r.source),
                    one_line(&r.vi),
                    if r.note.trim().is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", r.note.trim())
                    }
                ));
            }
        };
        renderings("REGISTER", &page.register, &mut out);
        renderings("LOCALISED", &page.localisation, &mut out);
        renderings("TERM", &page.terms, &mut out);
        for (label, items) in [
            ("VOICE", &page.voice),
            ("TRANSLATOR NOTE", &page.translator_notes),
            ("SFX", &page.sound_effects),
        ] {
            for item in items {
                out.push_str(&format!("- {label}: {}\n", one_line(item)));
            }
        }
        out.push('\n');
    }
    out
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

const SYNTHESIS_SYSTEM: &str = "\
You write translation style guides for manga. The guide you write is placed at \
the top of every request to the model that translates this series, so every \
line in it must be something that model can act on.";

const SYNTHESIS_FORMAT: &str = r#"
Write the guide as one JSON object and nothing else:

{
  "approach": [],
  "voice": [],
  "address": [],
  "soundEffects": [],
  "glossary": [["Japanese", "Vietnamese"]],
  "characters": [
    {"id": "ascii-id", "name": "…", "nameJa": "…", "aliases": [], "gender": "male|female|",
     "ageGroup": "child|teen|young_adult|adult|middle_age|elder|", "role": "…",
     "personality": "…", "speech": "…", "selfTerms": ["ta (posturing)"], "appearances": 0,
     "relations": [{"to": "other-id", "relation": "…", "address": "tôi/ngài by default; ông/tôi when fed up"}]}
  ]
}

- characters (at most 25): the recurring cast — characters who speak on at
  least three pages, or who are central. Use one name per character throughout
  and list the other names and titles the notes use under aliases. Gender and
  age come from how they are drawn and addressed; leave them empty when
  unclear. appearances is the number of pages they speak on. relations: for
  each character they speak to more than once, how they address them, default
  first and then by mood, as the ADDRESS lines show.
- approach (at most 10): how this translator carries the text across — how the
  Japanese speech level becomes Vietnamese pronouns and particles, what is
  localised and what is glossed in brackets, whether Japanese honorifics are
  kept, how far the tone is sharpened or softened, how names, attacks and
  foreign words are handled.
- voice (at most 12): what makes the Vietnamese itself recognisable — slang,
  internet slang, regional words, idioms, particles, jokes, talking to the
  reader — each with real examples quoted from the notes.
- address (at most 14): one entry per relationship that matters, written with
  its direction, for example "Meat → Kinnikuman, by default: tôi/ngài, calling
  him hoàng tử … Fed up: ông/tôi. Furious: mày." Give the default first, then
  each mood variant the notes show. Include a relationship only when ADDRESS
  lines in the notes show those two characters using those terms.
- soundEffects (at most 8): what is left in Japanese, translated, glossed,
  romanised; how laughs and long shouts are written.
- glossary (at most 60): names and recurring terms with their settled
  Vietnamese spelling. Where the translation is inconsistent, give the most
  frequent spelling here and name the variants in an approach entry.

Base every entry on the notes. Leave out what is true of any manga translation.
Write the entries in English and quote Vietnamese exactly as the translation
writes it."#;

const SYNTHESIS_WITH_RAW: &str = "\
Below are page-by-page notes on a manga volume and its published Vietnamese \
translation, read from the original and translated pages side by side.";

const SYNTHESIS_WITHOUT_RAW: &str = "\
Below are page-by-page notes on a published Vietnamese manga translation, read \
from the Vietnamese pages only — the original was not available. Describe the \
approach from what the Vietnamese shows (honorifics kept, bracketed glosses, \
localised references, how characters address each other) without guessing at \
the Japanese. Leave the Japanese side of every glossary entry empty.";

/// The synthesis request over every page's notes, as `(system, user)`.
pub fn synthesis_request(pages: &[PageNotes]) -> (&'static str, String) {
    let with_raw = pages.iter().any(|p| p.with_raw);
    let lead = if with_raw {
        SYNTHESIS_WITH_RAW
    } else {
        SYNTHESIS_WITHOUT_RAW
    };
    (
        SYNTHESIS_SYSTEM,
        format!("{lead}\n\n{}\n{SYNTHESIS_FORMAT}", transcript(pages)),
    )
}

/// Turn the synthesis reply into a profile, keeping only what the pages show.
pub fn profile_from_reply(reply: &str, pages: &[PageNotes]) -> anyhow::Result<StyleProfile> {
    let mut profile = style::parse_profile(reply)?;
    if !pages.iter().any(|p| p.with_raw) {
        for entry in &mut profile.glossary {
            entry[0].clear();
        }
    }
    let corpus: String = pages
        .iter()
        .flat_map(|p| {
            p.lines.iter().map(|l| l.vi.as_str()).chain(
                p.address
                    .iter()
                    .flat_map(|a| [a.self_term.as_str(), a.other_term.as_str()]),
            )
        })
        .collect::<Vec<_>>()
        .join(" || ");
    style::ground_in(&mut profile, &corpus);
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPLY: &str = r#"Here are the notes:
```json
{
  "summary": "Meat tìm hoàng tử",
  "lines": [
    {"speaker": "Meat", "listener": "Kinnikuman", "kind": "dialogue", "mood": "kính trọng",
     "vi": "LÀ NGÀI ĐÓ,\nHOÀNG TỬ!", "ja": "あなたですよ"},
    {"speaker": "", "listener": "", "kind": "sfx", "vi": "  "}
  ],
  "address": [
    {"speaker": "Meat", "listener": "Kinnikuman", "selfTerm": "tôi", "otherTerm": "ngài",
     "mood": "kính trọng", "quote": "LÀ NGÀI ĐÓ, HOÀNG TỬ!"}
  ],
  "register": [{"source": "あなた…です", "vi": "ngài … ạ", "note": "keigo"}],
  "terms": [{"source": "王子", "vi": "hoàng tử"}],
  "extra": "ignored"
}
```"#;

    #[test]
    fn a_fenced_page_reply_parses_and_blank_lines_are_dropped() {
        let notes = parse_page(REPLY, "0309", true).unwrap();
        assert_eq!(notes.page, "0309");
        assert!(notes.with_raw);
        assert_eq!(notes.lines.len(), 1);
        assert_eq!(notes.lines[0].vi, "LÀ NGÀI ĐÓ,\nHOÀNG TỬ!");
        assert_eq!(notes.address[0].other_term, "ngài");
        assert_eq!(notes.terms[0].source, "王子");
    }

    #[test]
    fn missing_sections_on_a_page_are_just_empty() {
        let notes = parse_page(r#"{"summary": "trang bìa"}"#, "0291", false).unwrap();
        assert!(notes.lines.is_empty() && notes.address.is_empty());
    }

    #[test]
    fn a_page_reply_with_no_json_is_an_error() {
        assert!(parse_page("Xin lỗi, tôi không đọc được trang này.", "0291", false).is_err());
    }

    #[test]
    fn the_transcript_keeps_who_said_what_to_whom_on_one_line_each() {
        let notes = parse_page(REPLY, "0309", true).unwrap();
        let text = transcript(&[notes]);
        assert!(text.contains("## 0309 — Meat tìm hoàng tử"));
        assert!(text.contains(
            "- (dialogue) Meat → Kinnikuman: LÀ NGÀI ĐÓ, HOÀNG TỬ! [kính trọng]  ⟵ あなたですよ"
        ));
        assert!(text.contains("ADDRESS Meat → Kinnikuman: self «tôi» other «ngài»"));
        assert!(text.contains("REGISTER: あなた…です → ngài … ạ (keigo)"));
    }

    #[test]
    fn the_request_changes_with_whether_the_original_was_shown() {
        let (_, with) = page_request(true);
        let (_, without) = page_request(false);
        assert!(with.contains("first image is the original"));
        assert!(without.contains("original is not available"));
        assert!(without.contains("\"address\""));
    }

    /// A pair the pages never show is dropped; so is a Japanese side the
    /// reader could not have seen.
    #[test]
    fn a_profile_from_vietnamese_pages_alone_is_grounded_and_has_no_japanese() {
        let page = parse_page(REPLY, "0309", false).unwrap();
        let reply = r#"{
            "address": ["Meat → Kinnikuman: tôi/ngài", "Ông/Cháu — old people"],
            "glossary": [["王子", "hoàng tử"], ["", "Terryman"]]
        }"#;

        let profile = profile_from_reply(reply, &[page]).unwrap();

        assert_eq!(profile.address, ["Meat → Kinnikuman: tôi/ngài"]);
        assert_eq!(profile.glossary, [[String::new(), "hoàng tử".to_string()]]);
    }

    #[test]
    fn the_synthesis_request_says_whether_the_original_was_read() {
        let with = parse_page(REPLY, "0309", true).unwrap();
        let without = parse_page(REPLY, "0309", false).unwrap();
        assert!(synthesis_request(&[with]).1.contains("side by side"));
        assert!(
            synthesis_request(&[without])
                .1
                .contains("original was not available")
        );
    }
}
