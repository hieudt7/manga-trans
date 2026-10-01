//! Canonical "speech style" taxonomy and the HL Comic size/stroke preset it
//! maps to — see `.claude/ReSizeFont.md`. A balloon's shape and a page's
//! source typography are both weak, independent signals for how a line was
//! originally "performed" (shouted, whispered, thought); this module fuses
//! them into one of a fixed set of styles, then maps that style to a fixed
//! HL Comic size so the same voice always renders at the same size instead of
//! drifting balloon to balloon.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// Dialogue size as a fraction of page height — the "chapter baseline" a
/// block's relative size is measured against. Shared with
/// `koharu-renderer`'s own dialogue sizing so the two never drift apart.
pub const DIALOGUE_FONT_HEIGHT_RATIO: f32 = 14.0 / 1200.0;

/// Never aim below this, in either crate's sizing.
pub const DIALOGUE_MIN_FONT_SIZE: f32 = 11.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[ts(export)]
pub enum SpeechStyleKind {
    Normal,
    Quiet,
    Whisper,
    Emphasis,
    Shout,
    Thought,
    Trembling,
    Narration,
}

impl SpeechStyleKind {
    /// Tolerant parse of a vision model's free-text answer — case and
    /// spacing vary even when asked for one of these exact words.
    pub fn parse_loose(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_uppercase().replace([' ', '-'], "_");
        match normalized.as_str() {
            "NORMAL" => Some(Self::Normal),
            "QUIET" => Some(Self::Quiet),
            "WHISPER" => Some(Self::Whisper),
            "EMPHASIS" => Some(Self::Emphasis),
            "SHOUT" => Some(Self::Shout),
            "THOUGHT" => Some(Self::Thought),
            "TREMBLING" => Some(Self::Trembling),
            "NARRATION" => Some(Self::Narration),
            _ => None,
        }
    }
}

/// The container's drawn shape, read by a vision model straight off the page
/// image rather than guessed from a segmentation mask — a mask cannot tell a
/// dashed outline from a solid one, a vision model looking at the pixels can.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[ts(export)]
pub enum BalloonShape {
    NormalOval,
    Jagged,
    Cloud,
    Dashed,
    Wavy,
    Rectangular,
    /// No drawn container at all — free text / SFX lettering directly on the
    /// art. Serialized as `"NONE"` to match the vision prompt's own wording.
    #[serde(rename = "NONE")]
    NoContainer,
}

impl BalloonShape {
    pub fn parse_loose(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_uppercase().replace([' ', '-'], "_");
        match normalized.as_str() {
            "NORMAL_OVAL" | "OVAL" | "NORMAL" => Some(Self::NormalOval),
            "JAGGED" | "BURST" => Some(Self::Jagged),
            "CLOUD" => Some(Self::Cloud),
            "DASHED" | "DOTTED" => Some(Self::Dashed),
            "WAVY" => Some(Self::Wavy),
            "RECTANGULAR" | "RECTANGLE" | "BOX" => Some(Self::Rectangular),
            "NONE" | "NO_CONTAINER" | "FREE_TEXT" => Some(Self::NoContainer),
            _ => None,
        }
    }

    /// The style a shape alone suggests — the balloon-shape → style table in
    /// ReSizeFont.md. A free-standing block (no drawn container) carries no
    /// shape-based vote at all.
    fn hint(self) -> Option<SpeechStyleKind> {
        match self {
            Self::NormalOval => Some(SpeechStyleKind::Normal),
            Self::Jagged => Some(SpeechStyleKind::Shout),
            Self::Cloud => Some(SpeechStyleKind::Thought),
            Self::Dashed => Some(SpeechStyleKind::Whisper),
            Self::Wavy => Some(SpeechStyleKind::Trembling),
            Self::Rectangular => Some(SpeechStyleKind::Narration),
            Self::NoContainer => None,
        }
    }
}

/// What the balloon's own drawn shape voted for.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct BalloonEvidence {
    pub shape: BalloonShape,
    pub hint: SpeechStyleKind,
}

/// What the source lettering's measured size voted for.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct TypographyEvidence {
    /// Source glyph size divided by the chapter baseline (`DIALOGUE_FONT_HEIGHT_RATIO`
    /// × page height) — 1.0 is ordinary dialogue.
    pub relative_size: f32,
    /// Detected stroke width divided by detected glyph size, kept for
    /// debugging only (not currently voted on) — a future tuning pass may
    /// use it to separate e.g. EMPHASIS from SHOUT at similar sizes.
    pub stroke_ratio: f32,
    pub hint: SpeechStyleKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ResolvedStyle {
    pub kind: SpeechStyleKind,
    /// Winning style's share of the total evidence weight that actually
    /// voted — 1.0 when every available source agreed, lower when they
    /// pulled in different directions.
    pub confidence: f32,
}

/// The HL Comic rendering this canonical style maps to. Font family is
/// always HL Comic (see `.claude/ReSizeFont.md` rule 1) — the only font on
/// this system, in one weight — so "louder" styles are expressed as a
/// thicker stroke rather than a bold face. `font_size` is a ratio of page
/// height, matching every other size constant in this pipeline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct TargetStyle {
    /// Fraction of page height.
    pub font_size_ratio: f32,
    /// Multiplier applied to the size-derived default stroke width.
    pub stroke_scale: f32,
}

/// Everything that went into a block's font-size/stroke decision, kept on
/// the block for debugging — so a line rendered at an unexpected size can be
/// traced back to whichever of balloon shape, source typography, or the
/// vision model's speech-state guess drove it there, instead of having to
/// reverse-engineer the renderer.
#[derive(Debug, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct StyleResolution {
    pub balloon: Option<BalloonEvidence>,
    pub speech_state_hint: Option<SpeechStyleKind>,
    pub source_typography: TypographyEvidence,
    pub resolved: ResolvedStyle,
    pub target: TargetStyle,
}

/// Evidence weights from ReSizeFont.md's fusion section — not claimed exact,
/// a starting point to tune once real pages are compared.
const TYPOGRAPHY_WEIGHT: f32 = 0.45;
const BALLOON_WEIGHT: f32 = 0.30;
const SPEECH_STATE_WEIGHT: f32 = 0.25;

const ALL_KINDS: [SpeechStyleKind; 8] = [
    SpeechStyleKind::Normal,
    SpeechStyleKind::Quiet,
    SpeechStyleKind::Whisper,
    SpeechStyleKind::Emphasis,
    SpeechStyleKind::Shout,
    SpeechStyleKind::Thought,
    SpeechStyleKind::Trembling,
    SpeechStyleKind::Narration,
];

/// What relative source-glyph size alone suggests, before balloon shape or
/// the vision model's read of the page nudge it — the size breakpoints from
/// ReSizeFont.md's worked example (0.98 → NORMAL, 0.81 → QUIET, 1.24 →
/// EMPHASIS, 1.60 → SHOUT).
fn typography_hint(relative_size: f32) -> SpeechStyleKind {
    if relative_size < 0.85 {
        SpeechStyleKind::Quiet
    } else if relative_size < 1.15 {
        SpeechStyleKind::Normal
    } else if relative_size < 1.45 {
        SpeechStyleKind::Emphasis
    } else {
        SpeechStyleKind::Shout
    }
}

/// Fuse balloon shape, source typography, and a vision model's own read of
/// the speech state (already reading punctuation and semantics itself, so
/// they are not scored again here) into one canonical style, then map that
/// style to its HL Comic preset.
pub fn resolve_speech_style(
    relative_size: f32,
    stroke_ratio: f32,
    balloon_shape: Option<BalloonShape>,
    speech_state_hint: Option<SpeechStyleKind>,
) -> StyleResolution {
    let typo_hint = typography_hint(relative_size);

    let mut scores: [f32; 8] = [0.0; 8];
    let index = |kind: SpeechStyleKind| ALL_KINDS.iter().position(|&k| k == kind).unwrap();

    let mut total_weight = TYPOGRAPHY_WEIGHT;
    scores[index(typo_hint)] += TYPOGRAPHY_WEIGHT;

    let balloon_hint = balloon_shape.and_then(BalloonShape::hint);
    if let Some(hint) = balloon_hint {
        scores[index(hint)] += BALLOON_WEIGHT;
        total_weight += BALLOON_WEIGHT;
    }
    if let Some(hint) = speech_state_hint {
        scores[index(hint)] += SPEECH_STATE_WEIGHT;
        total_weight += SPEECH_STATE_WEIGHT;
    }

    let (winner, top_score) = ALL_KINDS
        .iter()
        .copied()
        .zip(scores)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let confidence = if total_weight > 0.0 {
        top_score / total_weight
    } else {
        0.0
    };

    StyleResolution {
        balloon: balloon_shape.map(|shape| BalloonEvidence {
            shape,
            // `hint()` only returns `None` for `NoContainer`, and a block
            // with a shape at all always has one — falling back to `Normal`
            // here is unreachable in practice but keeps this infallible.
            hint: balloon_hint.unwrap_or(SpeechStyleKind::Normal),
        }),
        speech_state_hint,
        source_typography: TypographyEvidence {
            relative_size,
            stroke_ratio,
            hint: typo_hint,
        },
        resolved: ResolvedStyle {
            kind: winner,
            confidence,
        },
        target: hl_comic_preset(winner),
    }
}

/// HL Comic size/stroke preset per canonical style — the table in
/// ReSizeFont.md, expressed as a fraction of page height (its example sizes
/// were given at what is effectively a 1200px-tall reference page, the same
/// reference `DIALOGUE_FONT_HEIGHT_RATIO` uses).
pub fn hl_comic_preset(kind: SpeechStyleKind) -> TargetStyle {
    const REFERENCE_HEIGHT: f32 = 1200.0;
    let (size_px_at_reference, stroke_scale) = match kind {
        SpeechStyleKind::Quiet => (30.0, 0.85),
        SpeechStyleKind::Whisper => (30.0, 0.75),
        SpeechStyleKind::Thought => (32.0, 0.9),
        SpeechStyleKind::Narration => (32.0, 0.9),
        SpeechStyleKind::Normal => (34.0, 1.0),
        SpeechStyleKind::Trembling => (34.0, 1.0),
        SpeechStyleKind::Emphasis => (38.0, 1.15),
        SpeechStyleKind::Shout => (42.0, 1.35),
    };
    TargetStyle {
        font_size_ratio: size_px_at_reference / REFERENCE_HEIGHT,
        stroke_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_loose_accepts_case_and_spacing_variants() {
        assert_eq!(SpeechStyleKind::parse_loose("shout"), Some(SpeechStyleKind::Shout));
        assert_eq!(SpeechStyleKind::parse_loose(" Shout "), Some(SpeechStyleKind::Shout));
        assert_eq!(SpeechStyleKind::parse_loose("not-a-style"), None);

        assert_eq!(
            BalloonShape::parse_loose("normal_oval"),
            Some(BalloonShape::NormalOval)
        );
        assert_eq!(BalloonShape::parse_loose("none"), Some(BalloonShape::NoContainer));
        assert_eq!(BalloonShape::parse_loose("box"), Some(BalloonShape::Rectangular));
    }

    #[test]
    fn agreeing_evidence_resolves_with_full_confidence() {
        // relative_size 1.6 -> typography votes SHOUT; balloon Jagged -> SHOUT;
        // vision speech state -> SHOUT. All three agree.
        let resolution = resolve_speech_style(1.6, 0.15, Some(BalloonShape::Jagged), Some(SpeechStyleKind::Shout));
        assert_eq!(resolution.resolved.kind, SpeechStyleKind::Shout);
        assert!((resolution.resolved.confidence - 1.0).abs() < 1e-6);
        assert_eq!(resolution.target.font_size_ratio, hl_comic_preset(SpeechStyleKind::Shout).font_size_ratio);
    }

    #[test]
    fn disagreeing_evidence_still_picks_the_heaviest_voted_kind_with_lower_confidence() {
        // Typography (weight 0.45) says NORMAL; balloon (0.30) says THOUGHT;
        // no speech-state hint at all. NORMAL should win on weight alone, but
        // confidence should reflect the disagreement (well under 1.0).
        let resolution = resolve_speech_style(1.0, 0.1, Some(BalloonShape::Cloud), None);
        assert_eq!(resolution.resolved.kind, SpeechStyleKind::Normal);
        assert!(resolution.resolved.confidence < 0.7);
    }

    #[test]
    fn no_container_casts_no_balloon_vote() {
        // A free-text block still resolves from typography alone.
        let with_none = resolve_speech_style(1.0, 0.1, Some(BalloonShape::NoContainer), None);
        let without_any = resolve_speech_style(1.0, 0.1, None, None);
        assert_eq!(with_none.resolved.kind, without_any.resolved.kind);
        assert!((with_none.resolved.confidence - without_any.resolved.confidence).abs() < 1e-6);
    }

    #[test]
    fn preset_sizes_match_resizefont_doc_table() {
        let ratio = |kind| hl_comic_preset(kind).font_size_ratio * 1200.0;
        assert!((ratio(SpeechStyleKind::Quiet) - 30.0).abs() < 1e-4);
        assert!((ratio(SpeechStyleKind::Whisper) - 30.0).abs() < 1e-4);
        assert!((ratio(SpeechStyleKind::Thought) - 32.0).abs() < 1e-4);
        assert!((ratio(SpeechStyleKind::Narration) - 32.0).abs() < 1e-4);
        assert!((ratio(SpeechStyleKind::Normal) - 34.0).abs() < 1e-4);
        assert!((ratio(SpeechStyleKind::Trembling) - 34.0).abs() < 1e-4);
        assert!((ratio(SpeechStyleKind::Emphasis) - 38.0).abs() < 1e-4);
        assert!((ratio(SpeechStyleKind::Shout) - 42.0).abs() < 1e-4);
    }
}
