use std::{collections::HashMap, ops::Range};

use anyhow::Result;
use harfrust::{Direction, Feature, Tag};
use skrifa::{
    MetadataProvider,
    instance::{LocationRef, Size},
};

use crate::font::{Font, font_key};
use crate::shape::shape_segment_with_fallbacks;

pub use crate::segment::{LineBreakOpportunity, LineBreaker, LineSegment};
pub use crate::shape::{PositionedGlyph, ShapedRun, ShapingOptions, TextShaper};

/// Writing mode for text layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WritingMode {
    /// Horizontal text, left-to-right, lines flow top-to-bottom.
    #[default]
    Horizontal,
    /// Vertical text, right-to-left columns (traditional CJK).
    VerticalRl,
}

impl WritingMode {
    /// Returns true if the writing mode is vertical.
    pub fn is_vertical(&self) -> bool {
        matches!(self, WritingMode::VerticalRl)
    }
}

impl From<WritingMode> for Direction {
    fn from(mode: WritingMode) -> Self {
        match mode {
            WritingMode::Horizontal => Direction::LeftToRight,
            WritingMode::VerticalRl => Direction::TopToBottom,
        }
    }
}

/// How far a line may reach past the room its band reports.
///
/// A band is the narrowest the shape gets anywhere the line's box reaches,
/// including the leading at its foot where no ink sits. In a rounded balloon
/// that narrowest point is at the top or bottom of the line while the letters
/// themselves sit across the middle, where there is more room. Without some
/// allowance for that, names that used to sit on one line — ERAGINES!,
/// KINNIKUMAN! — get cut for the sake of a few pixels the shape does have.
const BAND_OVERFLOW_TOLERANCE: f32 = 1.08;

/// Punctuation that has to stay with the word it follows. Left to break, a
/// stacked shout ends with its exclamation mark alone on the last line and a
/// wrapped line can start with one.
fn clings_to_the_word_before(c: char) -> bool {
    matches!(
        c,
        '!' | '?' | '.' | ',' | ':' | ';' | '…' | '！' | '？' | '。' | '、'
            | ')' | ']' | '»' | '”' | '’' | '"' | '\''
    )
}

/// The page's house sizes for dialogue, largest first.
///
/// Set at the first of these the room allows, cutting a word where that is what
/// it takes. Without this a balloon with room to spare gets set at 11pt just
/// because one long name will not fit a line whole — the size stops at whatever
/// keeps the words intact and never asks what cutting would buy.
const DIALOGUE_SIZE_LADDER: [i32; 2] = [15, 12];

/// Below this a line is hard to read at all, so a word cut in half is the
/// lesser evil against shrinking further to keep it whole.
pub const COMFORTABLE_FONT_SIZE: f32 = 11.0;

/// How much room the text has, row by row, in coordinates local to the layout
/// box.
///
/// A balloon is not a rectangle, and one with a figure drawn across it is not
/// even convex. Fitting text to the largest rectangle that fits inside such a
/// shape throws away most of the room: on a real page a balloon 82px wide gave
/// a usable rectangle 41px wide, which drove an unbreakable name down to a 6px
/// font. Measuring the room line by line uses the shape as it is.
#[derive(Debug, Clone, PartialEq)]
pub struct RowSpans {
    /// `(x, width)` for each pixel row from the top of the box down. A row with
    /// no room is `(0.0, 0.0)`.
    rows: Vec<(f32, f32)>,
}

impl RowSpans {
    pub fn new(rows: Vec<(f32, f32)>) -> Self {
        Self { rows }
    }

    pub fn height(&self) -> f32 {
        self.rows.len() as f32
    }

    pub fn is_empty(&self) -> bool {
        self.rows.iter().all(|(_, width)| *width <= 0.0)
    }

    /// The room a line of `height` starting at `top` can use: the narrowest the
    /// shape gets anywhere that line would cover, since a line of text is a
    /// rectangle and has to clear the whole band.
    pub fn band(&self, top: f32, height: f32) -> Option<(f32, f32)> {
        if top < 0.0 || top + height > self.height() {
            // The line would hang off the shape. A line needs its whole height,
            // not as much of it as happens to be left.
            return None;
        }
        let first = top.floor() as usize;
        let last = ((top + height).ceil() as usize).min(self.rows.len());
        if first >= last {
            return None;
        }

        let mut left = f32::NEG_INFINITY;
        let mut right = f32::INFINITY;
        for (x, width) in &self.rows[first..last] {
            if *width <= 0.0 {
                return None;
            }
            left = left.max(*x);
            right = right.min(*x + *width);
        }

        (right > left).then_some((left, right - left))
    }

    /// The same shape with its first `rows` rows closed off, for settling the
    /// text down into the middle of the room rather than leaving it at the top.
    ///
    /// Closed rather than dropped so every coordinate still means the same
    /// place: dropping them would shift the whole shape up, and the text would
    /// have to be nudged back down afterwards — which is how it ended up
    /// hanging off the bottom of its own bitmap.
    pub fn blank_top(&self, rows: usize) -> Self {
        let mut shape = self.clone();
        for row in shape.rows.iter_mut().take(rows) {
            *row = (0.0, 0.0);
        }
        shape
    }

    /// Where each successive line of `line_height` sits, skipping past bands
    /// the shape pinches shut. `(top, x, width)` per line.
    pub fn bands(&self, line_height: f32) -> Vec<(f32, f32, f32)> {
        if line_height <= 0.0 {
            return Vec::new();
        }

        let mut bands = Vec::new();
        let mut top = 0.0f32;
        while top + line_height <= self.height() {
            match self.band(top, line_height) {
                Some((x, width)) => {
                    bands.push((top, x, width));
                    top += line_height;
                }
                // Pinched shut here — step down a row at a time looking for
                // where the shape opens up again rather than giving up.
                None => top += 1.0,
            }
        }
        bands
    }
}

/// Glyphs for one line alongside metadata required by the renderer.
#[derive(Debug, Clone, Default)]
pub struct LayoutLine<'a> {
    /// Positioned glyphs in this line.
    pub glyphs: Vec<PositionedGlyph<'a>>,
    /// Range in the original text that this line covers.
    pub range: Range<usize>,
    /// Total advance (width for horizontal, height for vertical) of this line.
    pub advance: f32,
    /// Baseline position for this line (x, y).
    pub baseline: (f32, f32),
    /// The room this line was given, when the layout followed a shape rather
    /// than a rectangle. Alignment centres the line inside this instead of
    /// inside the box.
    pub span: Option<(f32, f32)>,
}

/// A collection of laid out lines.
#[derive(Debug, Clone)]
pub struct LayoutRun<'a> {
    /// Lines in this layout run.
    pub lines: Vec<LayoutLine<'a>>,
    /// Total width of the layout.
    pub width: f32,
    /// Total height of the layout.
    pub height: f32,
    /// Font size used to generate this layout.
    pub font_size: f32,
    /// Whether every line stayed inside the room it was given. Only a
    /// shape-guided layout can come back false; a rectangle is reported through
    /// `width` and `height`.
    pub fits: bool,
    /// The most times any single word had to be cut to fit. One cut reads as
    /// hyphenation; two leaves a fragment stranded on a line of its own.
    pub max_word_cuts: usize,
}

pub struct TextLayout<'a> {
    writing_mode: WritingMode,
    center_vertical_punctuation: bool,
    font: &'a Font,
    fallback_fonts: &'a [Font],
    font_size: Option<f32>,
    /// Upper bound for the automatic search. Typesetting wants a consistent
    /// size across a page, so the fitter aims at this and only goes below when
    /// the text genuinely does not fit — rather than blowing every balloon up
    /// to its own maximum.
    preferred_font_size: Option<f32>,
    max_width: Option<f32>,
    max_height: Option<f32>,
    row_spans: Option<RowSpans>,
    stack_glyphs: bool,
}

impl<'a> TextLayout<'a> {
    pub fn new(font: &'a Font, font_size: Option<f32>) -> Self {
        Self {
            writing_mode: WritingMode::Horizontal,
            center_vertical_punctuation: true,
            font,
            fallback_fonts: &[],
            font_size,
            preferred_font_size: None,
            max_width: None,
            max_height: None,
            row_spans: None,
            stack_glyphs: false,
        }
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = Some(size);
        self
    }

    pub fn with_writing_mode(mut self, mode: WritingMode) -> Self {
        self.writing_mode = mode;
        self
    }

    pub fn with_center_vertical_punctuation(mut self, enabled: bool) -> Self {
        self.center_vertical_punctuation = enabled;
        self
    }

    pub fn with_fallback_fonts(mut self, fonts: &'a [Font]) -> Self {
        self.fallback_fonts = fonts;
        self
    }

    /// Aim for `size`, shrinking only if the text overflows its box.
    pub fn with_preferred_font_size(mut self, size: f32) -> Self {
        self.preferred_font_size = Some(size);
        self
    }

    pub fn with_max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }

    pub fn with_max_height(mut self, height: f32) -> Self {
        self.max_height = Some(height);
        self
    }

    /// Set one letter per line, the way a shout is lettered down a tall narrow
    /// balloon in Japanese. For a short word this beats hyphenating it: KHÔNG
    /// stacked reads at a glance, KHÔNGG-/G! does not.
    pub fn with_stacked_glyphs(mut self) -> Self {
        self.stack_glyphs = true;
        self
    }

    /// Lay the text into a shape measured row by row instead of a rectangle.
    /// Horizontal text only — vertical text sets columns, so its room would
    /// have to be measured the other way round.
    pub fn with_row_spans(mut self, spans: RowSpans) -> Self {
        self.row_spans = Some(spans);
        self
    }

    fn shape_bands(&self, line_height: f32) -> Option<Vec<(f32, f32, f32)>> {
        if self.writing_mode.is_vertical() {
            return None;
        }
        let spans = self.row_spans.as_ref()?;
        let bands = spans.bands(line_height);
        (!bands.is_empty()).then_some(bands)
    }

    pub fn run(&self, text: &str) -> Result<LayoutRun<'a>> {
        if let Some(font_size) = self.font_size {
            // Nothing to search: the size is fixed, so a word too wide for the
            // line has to be split or it runs off the page.
            return self.run_with_size_inner(text, font_size, true);
        }

        self.run_auto(text)
    }

    fn run_auto(&self, text: &str) -> Result<LayoutRun<'a>> {
        let max_height = self.max_height.unwrap_or(f32::INFINITY);
        let max_width = self.max_width.unwrap_or(f32::INFINITY);

        // Hitting the width limit is not a reason to shrink: the line breaker
        // wraps at `max_width`, so a larger size simply flows onto more lines.
        // Only when the wrapped block also exceeds `max_height` has the text
        // truly run out of room. The search therefore tests both, and stops at
        // `preferred_font_size` so a short line in a big balloon stays at the
        // page's normal reading size instead of being scaled up to fill it.
        let ceiling = self
            .preferred_font_size
            .map(|size| size.round().max(1.0) as i32)
            .unwrap_or(300);

        let fits = |layout: &LayoutRun<'a>| {
            // Following a shape, "does it fit" is answered line by line while
            // the text is set; a bounding box cannot express it.
            if self.row_spans.is_some() && !self.writing_mode.is_vertical() {
                layout.fits
            } else {
                layout.width <= max_width && layout.height <= max_height
            }
        };

        // Whole words: a larger size is always at least as hard to fit as a
        // smaller one, so the largest that fits can be bisected out.
        let search_whole = |ceiling: i32| -> Result<Option<LayoutRun<'a>>> {
            let (mut low, mut high) = (6, ceiling);
            let mut best: Option<LayoutRun<'a>> = None;
            while low <= high {
                let mid = (low + high) / 2;
                let layout = self.run_with_size_inner(text, mid as f32, false)?;
                if fits(&layout) {
                    best = Some(layout);
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            Ok(best)
        };

        // Once words may be cut, that stops being true: where the cut falls
        // changes with the size, so a larger size can fit where a smaller one
        // does not. Measured on one real case: 6pt overflowed, 8pt overflowed,
        // 11pt fitted. Bisecting that lands between the two overflows and
        // reports that nothing fits, so walk down from the ceiling instead. The
        // ceiling here is low, so this is a handful of tries.
        let search_split = |ceiling: i32| -> Result<Option<LayoutRun<'a>>> {
            for size in (6..=ceiling).rev() {
                let layout = self.run_with_size_inner(text, size as f32, true)?;
                if fits(&layout) {
                    return Ok(Some(layout));
                }
            }
            Ok(None)
        };

        // Cutting a word in half is not a way to keep the size up: a reader
        // would rather have KHÔNGGG! a point smaller than read it as
        // KHÔ/NGG/G!. But a long name in a narrow balloon — ABDULLAH with a
        // figure standing in front of it — can only be kept whole by shrinking
        // past the point of being readable at all, and there the cut is the
        // lesser evil. So fit the words whole where that lands at a readable
        // size, and cut only when it does not and cutting genuinely buys
        // something.
        let whole = search_whole(ceiling)?;
        let kept_whole = whole.as_ref().map_or(0.0, |layout| layout.font_size);

        // Text the artist drew large keeps its own scale — `ceiling` carries
        // that — so only reach for the house size when the words-intact answer
        // came in under it.
        for size in DIALOGUE_SIZE_LADDER.map(|size| size.min(ceiling)) {
            // Never past the ceiling: that is the page's own reading size, and
            // a short line in a big balloon still belongs at it. And never
            // below what the words already reach intact — cutting has to buy
            // something or it is not worth doing.
            if (size as f32) <= kept_whole {
                break;
            }
            let layout = self.run_with_size_inner(text, size as f32, true)?;
            // Fitting is not enough. At this size the name may need cutting
            // twice, stranding a syllable on a line of its own — KINNI/KUMA/N!.
            // A rung down it needs one cut and reads as hyphenation.
            if fits(&layout) && layout.max_word_cuts <= 1 {
                return Ok(layout);
            }
        }

        if let Some(layout) = &whole
            && layout.font_size >= COMFORTABLE_FONT_SIZE
        {
            return Ok(whole.expect("just matched"));
        }

        // Capped at the size the rescue is for. Uncapped, cutting becomes a way
        // to set text larger than the balloon is wide — KHÔNGGG! comes back as
        // three stacked fragments at twice the page's reading size, which is
        // worse than the small line it was rescuing.
        // Capped at the page's own ceiling as well: a rescue must not set text
        // larger than the page reads at.
        let split = search_split((COMFORTABLE_FONT_SIZE as i32).min(ceiling))?;
        match (whole, split) {
            (Some(whole), Some(split)) => {
                return Ok(if split.font_size > whole.font_size + 1.0 {
                    split
                } else {
                    whole
                });
            }
            (Some(only), None) | (None, Some(only)) => return Ok(only),
            (None, None) => {}
        }

        // Even the smallest size overflows; render at the floor rather than
        // dropping the text entirely.
        self.run_with_size_inner(text, 6.0, true)
    }

    pub fn run_with_size(&self, text: &str, font_size: f32) -> Result<LayoutRun<'a>> {
        self.run_with_size_inner(text, font_size, true)
    }

    /// Set at exactly this size and never cut a word. For deciding whether a
    /// size works at all, before cutting is on the table.
    pub fn run_whole_at(&self, text: &str, font_size: f32) -> Result<LayoutRun<'a>> {
        self.run_with_size_inner(text, font_size, false)
    }

    fn run_with_size_inner(
        &self,
        text: &str,
        font_size: f32,
        allow_word_split: bool,
    ) -> Result<LayoutRun<'a>> {
        let shaper = TextShaper::new();
        let line_breaker = LineBreaker::new();
        let normalized_punctuation;
        let text = if self.writing_mode.is_vertical() {
            normalized_punctuation = normalize_vertical_emphasis_punctuation(text);
            normalized_punctuation.as_str()
        } else {
            text
        };

        // Use real font metrics for consistent line sizing across modes.
        let font_ref = self.font.skrifa()?;
        let metrics = font_ref.metrics(Size::new(font_size), LocationRef::default());
        let ascent = metrics.ascent;
        let descent = -metrics.descent;
        let line_height = (ascent + descent + metrics.leading).max(font_size);

        let opts = ShapingOptions {
            direction: self.writing_mode.into(),
            font_size,
            features: if self.writing_mode.is_vertical() {
                &[
                    Feature::new(Tag::new(b"vert"), 1, ..),
                    Feature::new(Tag::new(b"vrt2"), 1, ..),
                ]
            } else {
                &[]
            },
        };

        let box_extent = if self.writing_mode.is_vertical() {
            self.max_height
        } else {
            self.max_width
        }
        .unwrap_or(f32::INFINITY);

        // Following a shape, each line gets the room the shape leaves at its own
        // height instead of one width for the whole block.
        let bands = self.shape_bands(line_height);
        let line_extent = |index: usize| -> f32 {
            match &bands {
                // Past the last band the text has overrun the shape. Keep
                // setting it at the last band's width so the run still reads
                // sensibly; `fits` reports the overrun and the size search
                // steps down.
                Some(bands) => bands
                    .get(index)
                    .or_else(|| bands.last())
                    .map_or(box_extent, |(_, _, width)| *width * BAND_OVERFLOW_TOLERANCE),
                None => box_extent,
            }
        };

        let segments = line_breaker.line_segments(text);

        let mut fonts: Vec<&Font> = Vec::with_capacity(1 + self.fallback_fonts.len());
        fonts.push(self.font);
        fonts.extend(self.fallback_fonts.iter());
        let mut lines: Vec<LayoutLine<'a>> = Vec::new();
        let mut current = LayoutLine::default();
        let mut line_offset = 0usize;
        let mut max_word_cuts = 0usize;

        for segment in segments {
            let start = segment.range.start;
            let segment_text = &text[segment.range.clone()];

            let mut shaped = if segment_text.is_empty() {
                ShapedRun {
                    glyphs: Vec::new(),
                    x_advance: 0.0,
                    y_advance: 0.0,
                }
            } else if fonts.len() == 1 {
                shaper.shape(segment_text, self.font, &opts)?
            } else {
                shape_segment_with_fallbacks(&shaper, segment_text, &fonts, &opts)?
            };
            if self.writing_mode.is_vertical() && self.center_vertical_punctuation {
                self.center_vertical_fullwidth_punctuation(
                    font_size,
                    segment_text,
                    &mut shaped.glyphs,
                );
            }
            let advance = if self.writing_mode.is_vertical() {
                shaped.y_advance
            } else {
                shaped.x_advance
            };

            let mut max_extent = line_extent(lines.len());
            let would_overflow = if self.writing_mode.is_vertical() {
                // For vertical text, advance is negative (downward), so we check absolute values
                current.advance.abs() + advance.abs() > max_extent
            } else {
                current.advance + advance > max_extent
            };
            let has_content = !current.glyphs.is_empty();

            if would_overflow && has_content {
                // Finalize current line
                current.range = line_offset..start;
                lines.push(current);

                // Start new line
                current = LayoutLine::default();
                line_offset = start;
                // The shape may leave the next line less room than the one just
                // closed, and this segment has yet to be measured against it.
                max_extent = line_extent(lines.len());
            }

            // A segment with nothing to break at can still be wider than a
            // whole line on its own — a name like KINNIKUMAN in a narrow
            // balloon. Left whole it either runs outside the balloon or drags
            // the size of the entire block down to fit it. Split it between
            // glyphs instead.
            let stack = self.stack_glyphs && !self.writing_mode.is_vertical();
            if stack
                || (allow_word_split
                    && current.glyphs.is_empty()
                    && max_extent.is_finite()
                    && advance.abs() > max_extent)
            {
                // Mark the cut. A word broken with nothing to show for it reads
                // as two words: ERAGINE / S!. The hyphen takes room of its own,
                // so the break has to be decided with that room already set
                // aside, or the line it lands on overflows.
                // Nothing to mark when the word is stacked: the reader is not
                // being asked to join two halves, they are reading down.
                let hyphen = (!stack && !self.writing_mode.is_vertical())
                    .then(|| shaper.shape("-", self.font, &opts))
                    .transpose()?;
                let hyphen_advance = hyphen.as_ref().map_or(0.0, |run| run.x_advance.abs());
                let mut cuts = 0usize;

                for mut glyph in shaped.glyphs {
                    glyph.cluster += start as u32;
                    let glyph_advance = if self.writing_mode.is_vertical() {
                        glyph.y_advance
                    } else {
                        glyph.x_advance
                    };
                    // Re-read it: each line pushed below may be given a
                    // different width by the shape.
                    let max_extent = line_extent(lines.len());
                    // Never before punctuation: it belongs to the word it
                    // follows, whether the word is being wrapped or read down a
                    // column.
                    let clings = text[glyph.cluster as usize..]
                        .chars()
                        .next()
                        .is_some_and(clings_to_the_word_before);
                    let overflows = !clings
                        && (stack
                            || current.advance.abs() + glyph_advance.abs() + hyphen_advance
                                > max_extent);
                    if overflows && !current.glyphs.is_empty() {
                        let cut = glyph.cluster as usize;
                        if let Some(hyphen) = &hyphen {
                            for mut mark in hyphen.glyphs.iter().cloned() {
                                mark.cluster = cut as u32;
                                current.glyphs.push(mark);
                            }
                            current.advance += hyphen.x_advance;
                        }
                        current.range = line_offset..cut;
                        lines.push(std::mem::take(&mut current));
                        line_offset = cut;
                        cuts += 1;
                    }
                    current.advance += glyph_advance;
                    current.glyphs.push(glyph);
                }
                if !stack {
                    max_word_cuts = max_word_cuts.max(cuts);
                }
            } else {
                // Adjust cluster indices and add glyphs to current line
                for mut glyph in shaped.glyphs {
                    glyph.cluster += start as u32;
                    current.glyphs.push(glyph);
                }
                current.advance += advance;
            }

            if segment.is_mandatory {
                current.range = line_offset..segment.range.end;
                lines.push(current);
                current = LayoutLine::default();
                line_offset = segment.next_offset;
            }
        }

        // Finalize last line
        if !current.glyphs.is_empty() {
            current.range = line_offset..text.len();
            lines.push(current);
        }

        // Baselines depend only on line index and metrics. For vertical text we compute absolute X
        // positions within the layout bounds (0..width) so the renderer can draw from the left.
        let line_count = lines.len();
        for (i, line) in lines.iter_mut().enumerate() {
            line.baseline = if self.writing_mode.is_vertical() {
                // Vertical-rl: first column is on the right, subsequent columns shift left.
                // Place the baseline at the center of each column. This avoids depending on
                // ascent/descent for X extents (which are Y metrics) and prevents right-edge clipping.
                let x = (line_count.saturating_sub(1) as f32 - i as f32) * line_height
                    + line_height * 0.5;
                (x, ascent)
            } else {
                (0.0, ascent + i as f32 * line_height)
            };
        }

        // Following a shape, each line sits where the shape leaves room for it,
        // so the positions are already the ones to draw at. Record the room
        // each line was given for alignment, and report whether the text
        // actually stayed inside it.
        let mut fits = true;
        if let Some(bands) = &bands {
            for (i, line) in lines.iter_mut().enumerate() {
                let Some((top, x, band_width)) = bands.get(i).copied() else {
                    // Ran out of shape: the text is taller than the room.
                    fits = false;
                    continue;
                };
                line.baseline = (x, top + ascent);
                line.span = Some((x, band_width));
                if line.advance.abs() > band_width * BAND_OVERFLOW_TOLERANCE + 0.5 {
                    fits = false;
                }
            }
        }

        // Compute a tight ink bounding box using per-glyph bounds from the font tables (via skrifa),
        // then translate baselines so the top-left ink origin is (0, 0). This avoids clipping without
        // having to measure Skia paths in the renderer.
        let (mut width, mut height) = self.compute_bounds(&lines, line_height, descent);
        if bands.is_some() {
            // Do not re-origin: that would slide the lines off the shape they
            // were measured against. Report the extent they occupy instead.
            let bottom = lines
                .iter()
                .map(|line| line.baseline.1 + descent)
                .fold(0.0f32, f32::max);
            let right = lines
                .iter()
                .map(|line| line.baseline.0 + line.advance.abs())
                .fold(0.0f32, f32::max);
            return Ok(LayoutRun {
                lines,
                width: right,
                height: bottom,
                font_size,
                fits,
                max_word_cuts,
            });
        }
        if let Some((mut min_x, mut min_y, mut max_x, mut max_y)) =
            self.ink_bounds(font_size, &lines)
        {
            // Keep a tiny safety pad for hinting/AA differences.
            const PAD: f32 = 1.0;
            min_x -= PAD;
            min_y -= PAD;
            max_x += PAD;
            max_y += PAD;

            for line in &mut lines {
                line.baseline.0 -= min_x;
                line.baseline.1 -= min_y;
            }
            width = (max_x - min_x).max(0.0);
            height = (max_y - min_y).max(0.0);
        }

        Ok(LayoutRun {
            lines,
            width,
            height,
            font_size,
            fits,
            max_word_cuts,
        })
    }

    fn compute_bounds(
        &self,
        lines: &[LayoutLine<'a>],
        line_height: f32,
        descent: f32,
    ) -> (f32, f32) {
        if lines.is_empty() {
            return (0.0, 0.0);
        }

        match self.writing_mode {
            WritingMode::Horizontal => {
                let w = lines.iter().map(|l| l.advance).fold(0.0f32, f32::max);
                let h = (lines.len() - 1) as f32 * line_height + lines[0].baseline.1 + descent;
                (w, h)
            }
            WritingMode::VerticalRl => {
                // Each line is a column; `line_height` is used as the column pitch (width).
                let w = lines.len() as f32 * line_height;
                // Like horizontal layout, account for the baseline offset (top padding via ascent)
                // and the descent so glyphs don't get clipped after converting to a Y-down canvas.
                let h = lines.iter().map(|l| l.advance.abs()).fold(0.0f32, f32::max)
                    + lines[0].baseline.1
                    + descent;
                (w, h)
            }
        }
    }

    fn ink_bounds(&self, font_size: f32, lines: &[LayoutLine<'a>]) -> Option<(f32, f32, f32, f32)> {
        let mut metrics_cache = HashMap::new();

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for line in lines {
            let (mut x, mut y) = line.baseline;
            for g in &line.glyphs {
                let key = font_key(g.font);
                let glyph_metrics = match metrics_cache.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        let Ok(font_ref) = g.font.skrifa() else {
                            x += g.x_advance;
                            y -= g.y_advance;
                            continue;
                        };
                        entry.insert(
                            font_ref.glyph_metrics(Size::new(font_size), LocationRef::default()),
                        )
                    }
                };

                let gid = skrifa::GlyphId::new(g.glyph_id);
                if let Some(b) = glyph_metrics.bounds(gid) {
                    let x0 = x + g.x_offset + b.x_min;
                    let x1 = x + g.x_offset + b.x_max;

                    // `b` is in a Y-up font coordinate system. Our layout coordinates are Y-down
                    // (matching the Skia canvas), so we flip by subtracting.
                    let y0 = (y - g.y_offset) - b.y_max;
                    let y1 = (y - g.y_offset) - b.y_min;

                    min_x = min_x.min(x0).min(x1);
                    max_x = max_x.max(x0).max(x1);
                    min_y = min_y.min(y0).min(y1);
                    max_y = max_y.max(y0).max(y1);
                }

                x += g.x_advance;
                y -= g.y_advance;
            }
        }

        if min_x.is_finite() {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    fn center_vertical_fullwidth_punctuation(
        &self,
        font_size: f32,
        segment: &str,
        glyphs: &mut [PositionedGlyph<'a>],
    ) {
        if segment.is_empty() || glyphs.is_empty() {
            return;
        }

        let mut metrics_cache = HashMap::new();
        for glyph in glyphs {
            let cluster = glyph.cluster as usize;
            let Some(ch) = segment.get(cluster..).and_then(|tail| tail.chars().next()) else {
                continue;
            };
            if !is_fullwidth_punctuation(ch) {
                continue;
            }

            let key = font_key(glyph.font);
            let glyph_metrics = match metrics_cache.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let Ok(font_ref) = glyph.font.skrifa() else {
                        continue;
                    };
                    entry.insert(
                        font_ref.glyph_metrics(Size::new(font_size), LocationRef::default()),
                    )
                }
            };

            let gid = skrifa::GlyphId::new(glyph.glyph_id);
            let Some(bounds) = glyph_metrics.bounds(gid) else {
                continue;
            };
            glyph.x_offset = centered_x_offset(bounds.x_min, bounds.x_max);
        }
    }
}

fn centered_x_offset(x_min: f32, x_max: f32) -> f32 {
    -((x_min + x_max) * 0.5)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmphasisMark {
    Bang,
    Question,
}

fn emphasis_mark_kind(ch: char) -> Option<EmphasisMark> {
    match ch {
        '!' | '！' => Some(EmphasisMark::Bang),
        '?' | '？' => Some(EmphasisMark::Question),
        _ => None,
    }
}

fn emphasis_pair_symbol(left: EmphasisMark, right: EmphasisMark) -> char {
    match (left, right) {
        (EmphasisMark::Bang, EmphasisMark::Bang) => '‼',
        (EmphasisMark::Question, EmphasisMark::Question) => '⁇',
        (EmphasisMark::Bang, EmphasisMark::Question) => '⁉',
        (EmphasisMark::Question, EmphasisMark::Bang) => '⁈',
    }
}

fn normalize_vertical_emphasis_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;

    while i < chars.len() {
        let Some(kind) = emphasis_mark_kind(chars[i]) else {
            out.push(chars[i]);
            i += 1;
            continue;
        };

        if i + 1 >= chars.len() {
            out.push(chars[i]);
            i += 1;
            continue;
        }

        let Some(next_kind) = emphasis_mark_kind(chars[i + 1]) else {
            out.push(chars[i]);
            i += 1;
            continue;
        };

        if kind == next_kind {
            out.push(emphasis_pair_symbol(kind, next_kind));
            i += 2;
            continue;
        }

        if i + 2 < chars.len()
            && let Some(lookahead_kind) = emphasis_mark_kind(chars[i + 2])
            && next_kind == lookahead_kind
        {
            out.push(chars[i]);
            i += 1;
            continue;
        }

        out.push(emphasis_pair_symbol(kind, next_kind));
        i += 2;
    }

    out
}

fn is_fullwidth_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '\u{3001}' // Ideographic comma
            | '\u{3002}' // Ideographic full stop
            | '\u{3008}'..='\u{3011}' // Angle/corner brackets
            | '\u{3014}'..='\u{301F}' // Tortoise shell/white brackets and marks
            | '\u{3030}' // Wavy dash
            | '\u{30FB}' // Katakana middle dot
            | '\u{FF01}'..='\u{FF0F}' // Fullwidth punctuation block 1
            | '\u{FF1A}'..='\u{FF20}' // Fullwidth punctuation block 2
            | '\u{FF3B}'..='\u{FF40}' // Fullwidth punctuation block 3
            | '\u{FF5B}'..='\u{FF65}' // Fullwidth punctuation block 4
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::{Font, FontBook};
    use skrifa::{
        MetadataProvider,
        instance::{LocationRef, Size},
    };

    fn any_system_font() -> Font {
        let mut book = FontBook::new();

        // Prefer fonts that are commonly available depending on OS/environment.
        // This is only used to construct a `TextLayout` for calling `compute_bounds`.
        let preferred = [
            "Yu Gothic",
            "MS Gothic",
            "Noto Sans CJK JP",
            "Noto Sans",
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ];

        for name in preferred {
            if let Some(post_script_name) = book
                .all_families()
                .into_iter()
                .find(|face| {
                    face.post_script_name == name
                        || face
                            .families
                            .iter()
                            .any(|(family, _)| family.as_str() == name)
                })
                .map(|face| face.post_script_name)
                .filter(|post_script_name| !post_script_name.is_empty())
                && let Ok(font) = book.query(&post_script_name)
            {
                return font;
            }
        }

        if let Some(face) = book
            .all_families()
            .into_iter()
            .find(|face| !face.post_script_name.is_empty())
        {
            return book
                .query(&face.post_script_name)
                .expect("failed to load first system font");
        }

        panic!("no system font available for tests");
    }

    fn assert_approx_eq(actual: f32, expected: f32) {
        let eps = 1e-4;
        assert!(
            (actual - expected).abs() <= eps,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn a_narrow_box_wraps_instead_of_shrinking() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG";

        // Same area, different shape: one wide and short, one narrow and tall.
        let wide = TextLayout::new(&font, None)
            .with_preferred_font_size(18.0)
            .with_max_width(600.0)
            .with_max_height(200.0)
            .run(text)?;
        let narrow = TextLayout::new(&font, None)
            .with_preferred_font_size(18.0)
            .with_max_width(200.0)
            .with_max_height(600.0)
            .run(text)?;

        // Running out of width is not a reason to shrink — the text flows onto
        // more lines and keeps the page's reading size.
        assert_eq!(wide.font_size, narrow.font_size);
        assert!(narrow.height > wide.height, "narrow box should wrap taller");
        Ok(())
    }

    #[test]
    fn a_word_wider_than_the_line_is_split_rather_than_run_outside_it() -> anyhow::Result<()> {
        let font = any_system_font();
        // One word, nothing to break at. Fitting it whole would mean either
        // spilling out of the balloon or shrinking the whole block to suit it.
        let narrow = TextLayout::new(&font, Some(16.0))
            .with_max_width(40.0)
            .run("KINNIKUMAN")?;

        assert!(
            narrow.lines.len() > 1,
            "expected the word to be split: {} line(s)",
            narrow.lines.len()
        );
        assert!(
            narrow.width <= 40.0 + 2.0,
            "no line may run past the box: width {}",
            narrow.width
        );
        // Every glyph is still there, in order.
        let mut clusters: Vec<u32> = narrow
            .lines
            .iter()
            .flat_map(|line| line.glyphs.iter().map(|glyph| glyph.cluster))
            .collect();
        let ordered = clusters.clone();
        clusters.dedup();
        assert_eq!(clusters.len(), "KINNIKUMAN".len(), "no glyph may be lost");
        assert!(ordered.windows(2).all(|w| w[0] <= w[1]), "order kept");
        Ok(())
    }

    #[test]
    fn shrinking_beats_cutting_when_it_stays_readable() -> anyhow::Result<()> {
        let font = any_system_font();
        // The word does not fit at 24pt but does a few points down, still well
        // above the size where reading gets hard. A reader would rather have it
        // a little smaller than read it as KINNI/KUMAN.
        let layout = TextLayout::new(&font, None)
            .with_preferred_font_size(24.0)
            .with_max_width(110.0)
            .with_max_height(400.0)
            .run("KINNIKUMAN")?;

        assert_eq!(
            layout.lines.len(),
            1,
            "should have shrunk instead of splitting: size {}",
            layout.font_size
        );
        assert!(layout.font_size < 24.0, "it had to shrink to get there");
        assert!(layout.font_size >= super::COMFORTABLE_FONT_SIZE);
        Ok(())
    }

    #[test]
    fn cutting_beats_shrinking_past_the_point_of_reading() -> anyhow::Result<()> {
        let font = any_system_font();
        // A long name in a balloon a figure is standing in front of. Keeping it
        // whole means shrinking it to nothing; the cut is the lesser evil.
        let whole = TextLayout::new(&font, None)
            .with_preferred_font_size(24.0)
            .with_max_width(34.0)
            .with_max_height(400.0)
            .run("ABDULLAH")?;

        assert!(
            whole.lines.len() > 1,
            "expected the name to be cut: size {}",
            whole.font_size
        );
        // Whole, ABDULLAH does not fit this box at any size the search will
        // try, so without the cut it would be dumped at the 6pt floor. The cut
        // has to leave it clearly better than that — not necessarily at
        // `COMFORTABLE_FONT_SIZE`, since the hyphen takes room of its own.
        assert!(
            whole.font_size >= 9.0,
            "cutting must buy a real improvement on the floor: got {}",
            whole.font_size
        );
        assert!(whole.width <= 36.0, "and still fit: {}", whole.width);
        Ok(())
    }

    #[test]
    fn a_shape_layout_reports_a_height_that_covers_its_last_line() -> anyhow::Result<()> {
        let font = any_system_font();
        let shape = pinched_shape();
        // Settled down into the middle of the shape, the reported height must
        // still reach the bottom of the lowest line — the bitmap is cut to it,
        // so anything below is lost.
        let layout = TextLayout::new(&font, Some(10.0))
            .with_row_spans(shape.blank_top(40))
            .run("AAA BBB CCC")?;

        let lowest = layout
            .lines
            .iter()
            .map(|line| line.baseline.1)
            .fold(0.0f32, f32::max);
        assert!(
            layout.height >= lowest,
            "height {} cuts off a line at {lowest}",
            layout.height
        );
        assert!(
            layout.lines.iter().all(|line| line.baseline.1 >= 40.0),
            "blanked rows must push the text below them"
        );
        Ok(())
    }

    #[test]
    fn a_cut_word_is_marked_with_a_hyphen() -> anyhow::Result<()> {
        let font = any_system_font();
        let cut = TextLayout::new(&font, Some(16.0))
            .with_max_width(40.0)
            .run("KINNIKUMAN")?;
        assert!(cut.lines.len() > 1, "expected a cut");

        let hyphen = font
            .skrifa()?
            .charmap()
            .map('-')
            .expect("the font has a hyphen");
        // Every line but the last one ends on the mark.
        for line in &cut.lines[..cut.lines.len() - 1] {
            assert_eq!(
                line.glyphs.last().map(|glyph| glyph.glyph_id),
                Some(hyphen.to_u32()),
                "a cut line must end in a hyphen"
            );
        }
        assert_ne!(
            cut.lines.last().unwrap().glyphs.last().map(|g| g.glyph_id),
            Some(hyphen.to_u32()),
            "the last line ends the word, not a cut"
        );
        assert!(
            cut.width <= 40.0 + 2.0,
            "the hyphen must be inside the box too: {}",
            cut.width
        );
        Ok(())
    }

    #[test]
    fn a_balloon_with_room_to_spare_reaches_the_house_size() -> anyhow::Result<()> {
        let font = any_system_font();
        // Tall and narrow, with room for many lines. KINNIKUMAN will not fit a
        // line whole above 11pt or so, but the balloon has the height to take
        // it cut — so the line should be set at the page's size, not dragged
        // down to whatever keeps the name in one piece.
        let layout = TextLayout::new(&font, None)
            .with_preferred_font_size(24.0)
            .with_max_width(74.0)
            .with_max_height(400.0)
            .run("MÀ NÀY, KINNIKUMAN!")?;

        assert!(
            layout.font_size >= super::DIALOGUE_SIZE_LADDER[1] as f32,
            "expected the house size, got {}",
            layout.font_size
        );
        Ok(())
    }

    #[test]
    fn a_stacked_shout_puts_one_letter_on_each_line_without_hyphens() -> anyhow::Result<()> {
        let font = any_system_font();
        let stacked = TextLayout::new(&font, Some(16.0))
            .with_stacked_glyphs()
            .with_max_width(400.0)
            .run("KHÔNG")?;

        assert_eq!(stacked.lines.len(), 5, "one letter to a line");
        assert!(
            stacked.lines.iter().all(|line| line.glyphs.len() == 1),
            "no line may hold two letters"
        );
        assert_eq!(
            stacked.max_word_cuts, 0,
            "reading down a column is not a cut word"
        );
        let hyphen = font.skrifa()?.charmap().map('-').expect("hyphen");
        assert!(
            !stacked
                .lines
                .iter()
                .flat_map(|line| &line.glyphs)
                .any(|glyph| glyph.glyph_id == hyphen.to_u32()),
            "a stacked shout is not hyphenated"
        );
        Ok(())
    }

    #[test]
    fn punctuation_stays_with_the_word_it_follows() -> anyhow::Result<()> {
        let font = any_system_font();
        // Down a column the mark must not end up alone on the last line.
        let stacked = TextLayout::new(&font, Some(16.0))
            .with_stacked_glyphs()
            .with_max_width(400.0)
            .run("KHÔNG!")?;
        assert_eq!(stacked.lines.len(), 5, "the mark rides with the G");
        assert_eq!(
            stacked.lines.last().unwrap().glyphs.len(),
            2,
            "last line is G plus its mark"
        );

        // Wrapping a cut word, a line must not begin with the mark either.
        let cut = TextLayout::new(&font, Some(16.0))
            .with_max_width(40.0)
            .run("KINNIKUMAN!")?;
        assert!(
            cut.lines.last().unwrap().glyphs.len() > 1,
            "the mark must not be stranded on a line of its own"
        );
        Ok(())
    }

    #[test]
    fn the_house_size_steps_down_rather_than_cut_a_word_twice() -> anyhow::Result<()> {
        let font = any_system_font();
        // A column narrow enough that KINNIKUMAN needs two cuts at the top
        // rung, stranding N! on a line by itself, and one cut at the next.
        let layout = TextLayout::new(&font, None)
            .with_preferred_font_size(24.0)
            .with_max_width(52.0)
            .with_max_height(400.0)
            .run("MÀ NÀY, KINNIKUMAN!")?;

        assert!(
            layout.max_word_cuts <= 1,
            "a word was cut {} times at size {}",
            layout.max_word_cuts,
            layout.font_size
        );
        Ok(())
    }

    #[test]
    fn the_house_size_never_beats_the_page_ceiling() -> anyhow::Result<()> {
        let font = any_system_font();
        // A ceiling below the house size wins: small lettering on a small page
        // must not be blown up to 15pt.
        let layout = TextLayout::new(&font, None)
            .with_preferred_font_size(9.0)
            .with_max_width(400.0)
            .with_max_height(400.0)
            .run("MÀ NÀY, KINNIKUMAN!")?;
        assert!(layout.font_size <= 9.0, "got {}", layout.font_size);
        Ok(())
    }

    #[test]
    fn a_word_that_fits_is_left_alone() -> anyhow::Result<()> {
        let font = any_system_font();
        let roomy = TextLayout::new(&font, Some(16.0))
            .with_max_width(4000.0)
            .run("KINNIKUMAN")?;
        assert_eq!(roomy.lines.len(), 1);
        Ok(())
    }

    /// Rows 0..40 are 100 wide; rows 40..120 are pinched to 30 by something
    /// drawn into the right of the shape. The shape of a balloon with a figure
    /// across its lower half.
    fn pinched_shape() -> RowSpans {
        RowSpans::new(
            (0..120)
                .map(|y| if y < 40 { (0.0, 100.0) } else { (0.0, 30.0) })
                .collect(),
        )
    }

    #[test]
    fn a_band_is_only_as_wide_as_the_narrowest_row_it_covers() {
        let shape = pinched_shape();
        // Wholly in the wide part.
        assert_eq!(shape.band(0.0, 20.0), Some((0.0, 100.0)));
        // Straddling the pinch: a line is a rectangle, so it gets the narrow
        // width, not the average and not the wide one.
        assert_eq!(shape.band(30.0, 20.0), Some((0.0, 30.0)));
        // Past the bottom.
        assert_eq!(shape.band(119.0, 20.0), None);
    }

    #[test]
    fn bands_step_over_a_row_the_shape_pinches_shut() {
        // Open, shut for a stretch, open again — text should resume below the
        // obstruction rather than stopping at it.
        let shape = RowSpans::new(
            (0..90)
                .map(|y| if (30..60).contains(&y) { (0.0, 0.0) } else { (0.0, 80.0) })
                .collect(),
        );

        let bands = shape.bands(20.0);
        assert!(bands.len() >= 2, "expected room above and below: {bands:?}");
        assert!(bands[0].0 < 30.0, "first band above the obstruction");
        assert!(
            bands.last().unwrap().0 >= 60.0,
            "last band below it: {bands:?}"
        );
        assert!(
            bands.iter().all(|(_, _, width)| *width > 0.0),
            "no band may be empty: {bands:?}"
        );
    }

    #[test]
    fn a_shape_gives_each_line_its_own_width() -> anyhow::Result<()> {
        let font = any_system_font();
        let layout = TextLayout::new(&font, Some(12.0))
            .with_row_spans(pinched_shape())
            .run("AAA BBB CCC DDD EEE FFF GGG HHH")?;

        let spans: Vec<(f32, f32)> = layout
            .lines
            .iter()
            .filter_map(|line| line.span)
            .collect();
        assert!(spans.len() >= 2, "expected several lines: {spans:?}");
        assert!(
            spans.iter().any(|(_, width)| *width > 90.0),
            "the wide part must be used: {spans:?}"
        );
        assert!(
            spans.iter().any(|(_, width)| *width <= 30.0),
            "the pinched part must be respected: {spans:?}"
        );
        for line in &layout.lines {
            let Some((_, room)) = line.span else { continue };
            // A line may reach a little past its band — see
            // `BAND_OVERFLOW_TOLERANCE` — but no further.
            let limit = room * super::BAND_OVERFLOW_TOLERANCE + 0.5;
            assert!(
                line.advance <= limit,
                "a line ran past its room: {} > {limit}",
                line.advance
            );
        }
        Ok(())
    }

    #[test]
    fn a_shape_beats_the_rectangle_that_fits_inside_it() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "ABDULLAH ATTACKS LIKE THIS WHAT WILL YOU DO";

        // The largest rectangle that fits the pinched shape is its narrow part
        // over the full height — which is what the balloon fitter would hand us.
        let as_rectangle = TextLayout::new(&font, None)
            .with_preferred_font_size(18.0)
            .with_max_width(30.0)
            .with_max_height(120.0)
            .run(text)?;
        let as_shape = TextLayout::new(&font, None)
            .with_preferred_font_size(18.0)
            .with_row_spans(pinched_shape())
            .run(text)?;

        assert!(
            as_shape.font_size > as_rectangle.font_size,
            "the shape should read larger: {} vs {}",
            as_shape.font_size,
            as_rectangle.font_size
        );
        Ok(())
    }

    #[test]
    fn the_preferred_size_is_a_ceiling_not_a_target_to_exceed() -> anyhow::Result<()> {
        let font = any_system_font();
        // A short line in a huge box must stay at the reading size.
        let layout = TextLayout::new(&font, None)
            .with_preferred_font_size(14.0)
            .with_max_width(4000.0)
            .with_max_height(4000.0)
            .run("OK")?;
        assert!(layout.font_size <= 14.0, "got {}", layout.font_size);
        Ok(())
    }

    #[test]
    fn text_that_cannot_fit_shrinks_rather_than_failing() -> anyhow::Result<()> {
        let font = any_system_font();
        let layout = TextLayout::new(&font, None)
            .with_preferred_font_size(18.0)
            .with_max_width(12.0)
            .with_max_height(12.0)
            .run("A VERY LONG SENTENCE THAT CANNOT POSSIBLY FIT")?;
        assert!(layout.font_size <= 18.0);
        Ok(())
    }

    #[test]
    fn compute_bounds_horizontal_uses_max_advance_and_baseline() {
        let font = any_system_font();
        let layout = TextLayout::new(&font, Some(16.0)).with_writing_mode(WritingMode::Horizontal);

        let lines = vec![
            LayoutLine {
                advance: 100.0,
                baseline: (0.0, 12.0),
                ..Default::default()
            },
            LayoutLine {
                advance: 250.0,
                baseline: (0.0, 32.0),
                ..Default::default()
            },
            LayoutLine {
                advance: 180.0,
                baseline: (0.0, 52.0),
                ..Default::default()
            },
        ];

        let line_height = 20.0;
        let descent = 5.0;
        let (w, h) = layout.compute_bounds(&lines, line_height, descent);

        assert_approx_eq(w, 250.0);
        // (len-1)*line_height + first_baseline_y + descent
        assert_approx_eq(h, 2.0 * line_height + 12.0 + descent);
    }

    #[test]
    fn compute_bounds_vertical_accounts_for_baseline_and_descent() {
        let font = any_system_font();
        let layout = TextLayout::new(&font, Some(16.0)).with_writing_mode(WritingMode::VerticalRl);

        let lines = vec![
            LayoutLine {
                // Vertical advances are typically negative in Y-up space; bounds use abs().
                advance: -100.0,
                baseline: (0.0, 12.0),
                ..Default::default()
            },
            LayoutLine {
                advance: -80.0,
                baseline: (-20.0, 12.0),
                ..Default::default()
            },
            LayoutLine {
                advance: -90.0,
                baseline: (-40.0, 12.0),
                ..Default::default()
            },
        ];

        let line_height = 20.0;
        let descent = 5.0;
        let (w, h) = layout.compute_bounds(&lines, line_height, descent);

        assert_approx_eq(w, 3.0 * line_height);
        // max(|advance|) + first_baseline_y + descent
        assert_approx_eq(h, 100.0 + 12.0 + descent);
    }

    #[test]
    fn layout_baselines_horizontal_follow_font_metrics() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 16.0;
        let layout = TextLayout::new(&font, Some(font_size))
            .with_writing_mode(WritingMode::Horizontal)
            .run("A\nB\nC")?;

        assert!(layout.lines.len() >= 2);

        let metrics = font
            .skrifa()?
            .metrics(Size::new(font_size), LocationRef::default());
        let ascent = metrics.ascent;
        let descent = -metrics.descent;
        let line_height = (ascent + descent + metrics.leading).max(font_size);

        let base_x = layout.lines[0].baseline.0;
        for line in &layout.lines {
            assert_approx_eq(line.baseline.0, base_x);
        }
        for i in 1..layout.lines.len() {
            let dy = layout.lines[i].baseline.1 - layout.lines[i - 1].baseline.1;
            assert_approx_eq(dy, line_height);
        }

        Ok(())
    }

    #[test]
    fn mandatory_newlines_are_not_shaped_as_glyphs() -> anyhow::Result<()> {
        let font = any_system_font();
        let text = "A\nB\nC";
        let layout = TextLayout::new(&font, Some(16.0))
            .with_writing_mode(WritingMode::Horizontal)
            .run(text)?;

        assert_eq!(layout.lines.len(), 3);
        for (line, expected) in layout.lines.iter().zip(["A", "B", "C"]) {
            assert_eq!(&text[line.range.clone()], expected);
            assert_eq!(line.glyphs.len(), 1);
        }

        Ok(())
    }

    #[test]
    fn layout_baselines_vertical_follow_font_metrics() -> anyhow::Result<()> {
        let font = any_system_font();
        let font_size = 16.0;
        let layout = TextLayout::new(&font, Some(font_size))
            .with_writing_mode(WritingMode::VerticalRl)
            .run("A\nB\nC")?;

        assert!(layout.lines.len() >= 2);

        let metrics = font
            .skrifa()?
            .metrics(Size::new(font_size), LocationRef::default());
        let ascent = metrics.ascent;
        let descent = -metrics.descent;
        let line_height = (ascent + descent + metrics.leading).max(font_size);
        let base_y = layout.lines[0].baseline.1;
        for line in &layout.lines {
            assert_approx_eq(line.baseline.1, base_y);
        }

        for i in 1..layout.lines.len() {
            let dx = layout.lines[i - 1].baseline.0 - layout.lines[i].baseline.0;
            assert_approx_eq(dx, line_height);
        }

        Ok(())
    }

    #[test]
    fn fullwidth_punctuation_detection_works() {
        assert!(is_fullwidth_punctuation('。'));
        assert!(is_fullwidth_punctuation('（'));
        assert!(is_fullwidth_punctuation('！'));
        assert!(!is_fullwidth_punctuation('A'));
        assert!(!is_fullwidth_punctuation('中'));
    }

    #[test]
    fn vertical_punctuation_centering_enabled_by_default() {
        let font = any_system_font();
        let layout = TextLayout::new(&font, Some(16.0));
        assert!(layout.center_vertical_punctuation);
    }

    #[test]
    fn centered_x_offset_uses_absolute_center() {
        assert_approx_eq(centered_x_offset(2.0, 6.0), -4.0);
        assert_approx_eq(centered_x_offset(-3.0, 1.0), 1.0);
    }

    #[test]
    fn normalize_vertical_emphasis_punctuation_collapses_pairs() {
        assert_eq!(normalize_vertical_emphasis_punctuation("！！"), "‼");
        assert_eq!(normalize_vertical_emphasis_punctuation("!!"), "‼");
        assert_eq!(normalize_vertical_emphasis_punctuation("!!?"), "‼?");
        assert_eq!(normalize_vertical_emphasis_punctuation("?!!"), "?‼");
        assert_eq!(normalize_vertical_emphasis_punctuation("!?!"), "⁉!");
        assert_eq!(normalize_vertical_emphasis_punctuation("！？"), "⁉");
        assert_eq!(normalize_vertical_emphasis_punctuation("？！"), "⁈");
        assert_eq!(
            normalize_vertical_emphasis_punctuation("Hello!?!"),
            "Hello⁉!"
        );
    }
}
