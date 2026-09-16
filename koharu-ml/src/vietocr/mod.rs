//! Reading Vietnamese lettering off a translated page, on this machine.
//!
//! Building a bilingual corpus means reading a whole published volume, and the
//! engines tried against this material all failed differently. Tesseract 5
//! `vie` read one balloon in six: the lettering is white glyphs with a black
//! outline sitting over artwork, not black on white, and it takes the outline
//! for strokes. PaddleOCR-VL read three in six and truncated a contents page.
//! PP-OCRv6's Vietnamese dictionary does not cover the tone combinations.
//!
//! What did work is [vietocr], a CNN-and-transformer recogniser trained on
//! Vietnamese. It is a *single-line* recogniser, though — hand it a whole
//! balloon and its decoder loops, emitting `CÓ CÓ CÓ CÓ` or a run of invented
//! digits. Measured over 151 real balloons, feeding it whole balloons through a
//! grey-level line splitter left 74.7% of characters wrong. Splitting the
//! balloon into lines properly first, with [`lines`], and blanking the artwork
//! around the lettering with [`keep_glyphs`], brought that to 6.3%, with 96% of
//! pronouns read right. A style profile learned from a corpus read this way
//! named the same pronoun pairs as one read by a vision model.
//!
//! What it still gets wrong is closing punctuation: this lettering's `!` and
//! `?` come back as `'`, `[` or `F`. That costs nothing for learning a
//! translator's voice, which is what the reader is for.
//!
//! So the line splitter is not a detail here. It is most of the accuracy.
//!
//! The models are produced by `scripts/export_vietocr_to_onnx.py`.
//!
//! [vietocr]: https://github.com/pbcquoc/vietocr

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;

/// Reserved ids at the head of the vocabulary, ahead of the characters.
const PAD: i64 = 0;
const SOS: i64 = 1;
const EOS: i64 = 2;
/// Characters start here; id `FIRST_CHAR + n` is the nth character of `vocab`.
const FIRST_CHAR: usize = 4;

/// Every strip is scaled to this height; the model was trained at it.
const IMAGE_HEIGHT: u32 = 32;
const MIN_WIDTH: u32 = 32;
const MAX_WIDTH: u32 = 512;
/// Widths are rounded up to a multiple of this, as in the original.
const WIDTH_ROUND_TO: u32 = 10;

/// A decode that has not finished by here is looping rather than reading. The
/// longest real line in the measured corpus was well under half this.
const MAX_TOKENS: usize = 128;

/// The vocabulary and the two graphs, loaded once.
pub struct VietOcr {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    /// Indexed by `id - FIRST_CHAR`.
    vocab: Vec<char>,
}

#[derive(serde::Deserialize)]
struct Meta {
    vocab: String,
}

impl VietOcr {
    /// Load from a directory holding `vietocr_encoder.onnx`,
    /// `vietocr_decoder.onnx` and `vietocr.json`.
    pub fn open(dir: &Path) -> Result<Self> {
        let meta: Meta = serde_json::from_slice(
            &std::fs::read(dir.join("vietocr.json")).context("vietocr.json")?,
        )?;

        let load = |name: &str| -> Result<Mutex<Session>> {
            let path = dir.join(name);
            let session = Session::builder()?
                .commit_from_file(&path)
                .with_context(|| format!("loading {}", path.display()))?;
            Ok(Mutex::new(session))
        };

        Ok(Self {
            encoder: load("vietocr_encoder.onnx")?,
            decoder: load("vietocr_decoder.onnx")?,
            vocab: meta.vocab.chars().collect(),
        })
    }

    /// Read one strip that holds a single line of text.
    pub fn read_line(&self, strip: &DynamicImage) -> Result<String> {
        let (width, pixels) = prepare(strip);
        let image = Tensor::from_array(Array::from_shape_vec(
            [1, 3, IMAGE_HEIGHT as usize, width as usize],
            pixels,
        )?)?;

        let memory = {
            let mut encoder = self
                .encoder
                .lock()
                .map_err(|_| anyhow::anyhow!("vietocr encoder mutex poisoned"))?;
            let outputs = encoder.run(ort::inputs! { "image" => image })?;
            outputs[0].try_extract_array::<f32>()?.view().to_owned()
        };
        let steps = memory.shape()[0];
        let width_e = memory.shape()[2];
        let memory: Vec<f32> = memory.iter().copied().collect();

        let mut decoder = self
            .decoder
            .lock()
            .map_err(|_| anyhow::anyhow!("vietocr decoder mutex poisoned"))?;

        // No key-value cache in the exported graph: each step re-runs the whole
        // prefix. The lines are short enough that this costs less than the
        // complexity of caching would.
        let mut tokens = vec![SOS];
        let mut text = String::new();
        for _ in 0..MAX_TOKENS {
            let length = tokens.len();
            let outputs = decoder.run(ort::inputs! {
                "tokens" => Tensor::from_array(Array::from_shape_vec([length, 1], tokens.clone())?)?,
                "memory" => Tensor::from_array(Array::from_shape_vec([steps, 1, width_e], memory.clone())?)?,
                "mask" => Tensor::from_array(Array::from_shape_vec([length, length], causal_mask(length))?)?,
            })?;

            let logits = outputs[0].try_extract_array::<f32>()?;
            let logits = logits.view();
            let vocab_size = logits.shape()[2];
            let last = (length - 1) * vocab_size;
            let flat: Vec<f32> = logits.iter().copied().collect();

            let next = (0..vocab_size)
                .max_by(|a, b| flat[last + a].total_cmp(&flat[last + b]))
                .unwrap_or(EOS as usize) as i64;

            if next == EOS || next == PAD {
                break;
            }
            if let Some(c) = self.character(next) {
                text.push(c);
            }
            tokens.push(next);
        }

        Ok(text)
    }

    /// Read a whole text block by splitting it into lines first.
    ///
    /// The model reads one line at a time; handed a block it loops — a five-line
    /// balloon came back as `2010`. So the block has to be cut into lines, and
    /// where the cuts go is most of the accuracy.
    ///
    /// Give it a block already run through [`keep_glyphs`] where a mask exists:
    /// blanking the artwork around the lettering took the character error over
    /// the measured corpus from 12.9% to 6.3%.
    ///
    /// Splitting on the mask itself while reading the original pixels sounds
    /// better again and was measured at 49%. The segmenter marks strokes, not
    /// lines, so a column down its output is patchier than the same column of
    /// artwork, and the row profile taken from it is noisier rather than
    /// cleaner. The mask is a good eraser and a poor ruler.
    pub fn read_block(&self, block: &DynamicImage) -> Result<String> {
        let mut out: Vec<String> = Vec::new();
        for (top, bottom, left, right) in rows_and_columns(block) {
            // A little air above and below: tone marks and descenders sit
            // outside the run of rows that carries the body of the line.
            const AIR: u32 = 3;
            let top = top.saturating_sub(AIR);
            let bottom = (bottom + AIR).min(block.height());
            if bottom.saturating_sub(top) < 8 || right <= left {
                continue;
            }
            let strip = block.crop_imm(left, top, right - left, bottom - top);
            let line = self.read_line(&strip)?;
            if !line.trim().is_empty() {
                out.push(line.trim().to_string());
            }
        }
        Ok(out.join("\n"))
    }

    fn character(&self, id: i64) -> Option<char> {
        usize::try_from(id)
            .ok()?
            .checked_sub(FIRST_CHAR)
            .and_then(|index| self.vocab.get(index))
            .copied()
    }
}

/// Upper-triangular `-inf`, so a position never attends to a later one.
fn causal_mask(length: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; length * length];
    for row in 0..length {
        for column in (row + 1)..length {
            mask[row * length + column] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Scale to the model's height, keep the aspect, and lay out as CHW in 0..1.
///
/// Matches the original's preprocessing exactly, rounding included — the model
/// is sensitive to it, and a strip scaled differently reads differently.
fn prepare(strip: &DynamicImage) -> (u32, Vec<f32>) {
    let (width, height) = strip.dimensions();
    let scaled = if height == 0 {
        MIN_WIDTH
    } else {
        (IMAGE_HEIGHT as f32 * width as f32 / height as f32) as u32
    };
    let width = (scaled.div_ceil(WIDTH_ROUND_TO) * WIDTH_ROUND_TO).clamp(MIN_WIDTH, MAX_WIDTH);

    let rgb = strip
        .resize_exact(width, IMAGE_HEIGHT, FilterType::Lanczos3)
        .to_rgb8();

    let (w, h) = (width as usize, IMAGE_HEIGHT as usize);
    let mut pixels = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                pixels[channel * h * w + y * w + x] = pixel[channel] as f32 / 255.0;
            }
        }
    }
    (width, pixels)
}

/// Blank out everything the segmenter did not call a glyph.
///
/// A detected block is a rectangle around lettering, and on a busy page that
/// rectangle also holds whatever artwork sits beside the words. The recogniser
/// has no way to tell the two apart and reads the artwork as text: a curl of
/// hair beside `GIẢI CỨU` came back as a leading `BÙ`. The pipeline already
/// separates glyphs from artwork for inpainting, so the same mask answers this.
///
/// `mask` must be the page mask cropped to the same rectangle as `crop`, and
/// white where a glyph is. A mask of a different size is ignored rather than
/// misaligned — a mask off by a few pixels erases the strokes it should keep.
pub fn keep_glyphs(crop: &DynamicImage, mask: &DynamicImage) -> DynamicImage {
    if crop.dimensions() != mask.dimensions() {
        tracing::warn!("glyph mask does not match the crop, reading the crop as it is");
        return crop.clone();
    }

    let mask = mask.to_luma8();
    let mut rgb = crop.to_rgb8();
    for (x, y, pixel) in rgb.enumerate_pixels_mut() {
        if mask.get_pixel(x, y)[0] < 128 {
            *pixel = image::Rgb([255, 255, 255]);
        }
    }
    DynamicImage::ImageRgb8(rgb)
}

/// Rows of a text block that hold a line of text, as `(top, bottom)`.
///
/// Counts how often each row crosses between light and dark rather than how far
/// it sits from the background. A row of text crosses many times — once at each
/// stroke — while blank paper and the flat inside of a balloon cross not at all.
///
/// Measuring distance-from-background instead is what the first attempt did,
/// and it fails on exactly this material: the lettering is white with a black
/// outline, so on a dark panel the white glyphs *and* the pale inside of the
/// balloon both read as "not background", and neighbouring lines merge into one
/// run. That version found the right number of lines in 33 of 151 balloons.
/// This one finds it in 117.
pub fn lines(block: &DynamicImage) -> Vec<(u32, u32)> {
    rows_and_columns(block)
        .into_iter()
        .map(|(top, bottom, _, _)| (top, bottom))
        .collect()
}

/// Each line of text as `(top, bottom, left, right)`.
///
/// The horizontal span matters as much as the vertical one. A detected block is
/// a rectangle around the lettering, and on a busy page that rectangle also
/// holds whatever artwork happens to sit beside the words. Handed a strip with
/// a lock of hair at one end, the recogniser reads the hair: on one measured
/// page `GIẢI CỨU` came back as `BÙ GIẢI CỨU`, the `BÙ` being a curl of ink.
/// Trimming each line to the columns that actually carry strokes removes it.
pub fn rows_and_columns(block: &DynamicImage) -> Vec<(u32, u32, u32, u32)> {
    let (width, height) = block.dimensions();
    let whole = vec![(0, height, 0, width)];
    if width < 8 || height < 8 {
        return whole;
    }

    let gray = block.to_luma8();
    let threshold = otsu(&gray);

    let crossings: Vec<u32> = (0..height)
        .map(|y| {
            let mut count = 0;
            let mut previous = gray.get_pixel(0, y)[0] > threshold;
            for x in 1..width {
                let current = gray.get_pixel(x, y)[0] > threshold;
                if current != previous {
                    count += 1;
                }
                previous = current;
            }
            count
        })
        .collect();

    // Scale the gate to the busiest row rather than fixing it: a caption in
    // small type crosses far less often than a shout in large type, and a fixed
    // gate would drop one or merge the other.
    let peak = crossings.iter().copied().max().unwrap_or(0);
    if peak < 4 {
        return whole;
    }
    // The gate is measured up from the floor, not down from the peak.
    //
    // Inside a balloon every single row crosses the outline twice, so the
    // profile never falls to zero — it sits on a plateau of two or three. A
    // gate set as a fraction of the peak lands on that plateau (22/12 rounds to
    // 2) and the whole block reads as one continuous line of text. The floor is
    // what the typical row does, so the median finds it whatever is drawing it.
    let floor = low_percentile(&crossings);
    let gate = floor + ((peak.saturating_sub(floor)) / 6).max(2);

    let mut cores: Vec<(u32, u32)> = Vec::new();
    let mut start: Option<u32> = None;
    for y in 0..=height {
        let busy = y < height && crossings[y as usize] >= gate;
        match (busy, start) {
            (true, None) => start = Some(y),
            (false, Some(from)) => {
                if y - from >= 4 {
                    cores.push((from, y));
                }
                start = None;
            }
            _ => {}
        }
    }

    // The gate finds the body of a line and stops short of its tone marks and
    // descenders, which cross the row fewer times. Reach back out to wherever
    // the row is still doing more than the floor, without running into the
    // neighbouring line.
    // Pitch is read off the cores, before they are widened. Once each band has
    // reached out to the gap around it they sit flush against one another, and
    // the distance from one start to the next is no longer the pitch but the
    // height of whatever band happens to precede it — on one block that turned
    // a pitch of 25 into 11 and sawed every line into thirds.
    //
    // Lines set close together run into one core whatever the threshold: the
    // tone marks of one reach into the space above the next. A block of
    // `TIẾP / THEO / LÀ / ELAGIN- / ESS!` came back as four, the second 39px
    // tall where a line is 25px, and `LÀ` was simply gone. The pitch says how
    // many lines a core holds, and unlike a threshold it does not have to suit
    // the shortest line and the tallest one at once.
    if let Some(pitch) = line_pitch(&cores) {
        cores = cores
            .into_iter()
            .flat_map(|(from, to)| split_at_pitch(from, to, pitch, &crossings))
            .collect();
    }

    // The gate finds the body of a line and stops short of its tone marks and
    // descenders, which cross the row fewer times. Reach back out to wherever
    // the row is still doing more than the floor — but each line stops halfway
    // to the next, not at it. Letting both reach the whole gap makes them
    // overlap, two bands of 8..21 and 16..29 sharing six rows, and the same
    // line is then read twice: `CHẤP HẾT / CHẤP HẾT,`.
    let mut bands: Vec<(u32, u32)> = Vec::with_capacity(cores.len());
    for (index, &(from, to)) in cores.iter().enumerate() {
        let above = match index.checked_sub(1).and_then(|i| cores.get(i)) {
            Some(&(_, end)) => end + from.saturating_sub(end) / 2,
            None => 0,
        };
        let below = match cores.get(index + 1) {
            Some(&(begin, _)) => to + begin.saturating_sub(to) / 2,
            None => height,
        };
        let mut from = from;
        let mut to = to;
        while from > above && crossings[(from - 1) as usize] > floor {
            from -= 1;
        }
        while to < below && crossings[to as usize] > floor {
            to += 1;
        }
        bands.push((from, to));
    }

    // Drop what is far thinner than its neighbours: a stray mark of artwork
    // caught inside the block is handed to the recogniser like anything else,
    // and it reads it as words — a 16px sliver of a drawn curl came back as
    // `Contransitionalists`.
    if let Some(typical) = median_height(&bands) {
        let floor = (typical * 45 / 100).max(5);
        bands.retain(|(from, to)| to - from >= floor);
    }

    let rows: Vec<_> = bands
        .into_iter()
        .map(|(from, to)| {
            let (left, right) = ink_span(&gray, threshold, from, to);
            (from, to, left, right)
        })
        .collect();

    if rows.is_empty() { whole } else { rows }
}

/// The measured pitch of a block, for diagnostics.
#[cfg(test)]
pub fn debug_pitch(block: &DynamicImage) -> Option<u32> {
    line_pitch(
        &rows_and_columns(block)
            .iter()
            .map(|&(top, bottom, _, _)| (top, bottom))
            .collect::<Vec<_>>(),
    )
}

/// The distance from one line of text to the next, if the block has more than
/// one.
///
/// Taken from how far apart the lines that *were* found start, which is a
/// steadier signal than the row profile itself: lettering is set at an even
/// pitch even when one line is two letters and the next is twelve. Two lines
/// that ran into a single band leave a gap of about twice the pitch, and the
/// median ignores it as long as most gaps are single.
///
/// An autocorrelation of the profile was tried first and gave up on exactly the
/// blocks that needed it — a five-line balloon whose lines differ in width does
/// not correlate strongly enough with itself to pass any useful threshold.
fn line_pitch(bands: &[(u32, u32)]) -> Option<u32> {
    if bands.len() < 2 {
        return None;
    }
    let mut gaps: Vec<u32> = bands
        .windows(2)
        .map(|pair| pair[1].0.saturating_sub(pair[0].0))
        .filter(|&gap| gap > 0)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable();
    Some(gaps[gaps.len() / 2])
}

/// Cut a band that spans several lines into one piece per line.
///
/// The cuts go at the quietest row near each expected boundary rather than at
/// the boundary itself: lines are evenly spaced but not exactly, and cutting
/// through a letter costs more than cutting a few rows off true.
fn split_at_pitch(from: u32, to: u32, pitch: u32, crossings: &[u32]) -> Vec<(u32, u32)> {
    let span = to - from;
    let lines = ((span as f32 / pitch as f32).round() as u32).max(1);
    if lines < 2 {
        return vec![(from, to)];
    }

    let mut cuts = vec![from];
    for index in 1..lines {
        let expected = from + span * index / lines;
        // Search a quarter of a line either side for the quietest row.
        let window = (pitch / 4).max(2);
        let low = expected.saturating_sub(window).max(from + 1);
        let high = (expected + window).min(to - 1);
        let quietest = (low..=high)
            .min_by_key(|&y| crossings[y as usize])
            .unwrap_or(expected);
        cuts.push(quietest);
    }
    cuts.push(to);

    cuts.windows(2)
        .filter(|pair| pair[1] > pair[0])
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

/// The floor the lettering rises from.
///
/// Low percentile rather than median: in a tightly set balloon the lettering
/// occupies more rows than the gaps do, and the median is then a row of text —
/// which puts the gate above every line and finds nothing at all. The tenth
/// percentile sits in the gaps while still being steadier than the minimum,
/// which one stray row would drag to zero.
fn low_percentile(values: &[u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 10]
}

fn median_height(bands: &[(u32, u32)]) -> Option<u32> {
    if bands.is_empty() {
        return None;
    }
    let mut heights: Vec<u32> = bands.iter().map(|(from, to)| to - from).collect();
    heights.sort_unstable();
    Some(heights[heights.len() / 2])
}

/// The columns of one band that carry strokes, as `(left, right)`.
fn ink_span(gray: &image::GrayImage, threshold: u8, top: u32, bottom: u32) -> (u32, u32) {
    let width = gray.width();
    let crossings: Vec<u32> = (0..width)
        .map(|x| {
            let mut count = 0;
            let mut previous = gray.get_pixel(x, top)[0] > threshold;
            for y in (top + 1)..bottom {
                let current = gray.get_pixel(x, y)[0] > threshold;
                if current != previous {
                    count += 1;
                }
                previous = current;
            }
            count
        })
        .collect();

    let first = crossings.iter().position(|&n| n > 0);
    let last = crossings.iter().rposition(|&n| n > 0);
    match (first, last) {
        // A hair of margin, so the outermost stroke is not shaved.
        (Some(first), Some(last)) if last > first => (
            (first as u32).saturating_sub(2),
            (last as u32 + 3).min(width),
        ),
        _ => (0, width),
    }
}

/// Otsu's threshold from the image's own histogram.
fn otsu(gray: &image::GrayImage) -> u8 {
    let mut histogram = [0u64; 256];
    for pixel in gray.pixels() {
        histogram[pixel[0] as usize] += 1;
    }
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return 128;
    }
    let sum: f64 = (0..256).map(|t| t as f64 * histogram[t] as f64).sum();

    let (mut weight_below, mut sum_below) = (0f64, 0f64);
    let (mut best, mut threshold) = (0f64, 128u8);
    for t in 0..256 {
        weight_below += histogram[t] as f64;
        if weight_below == 0.0 {
            continue;
        }
        let weight_above = total as f64 - weight_below;
        if weight_above == 0.0 {
            break;
        }
        sum_below += t as f64 * histogram[t] as f64;
        let mean_below = sum_below / weight_below;
        let mean_above = (sum - sum_below) / weight_above;
        let variance = weight_below * weight_above * (mean_below - mean_above).powi(2);
        if variance > best {
            best = variance;
            threshold = t as u8;
        }
    }
    threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// Bars of "text" separated by blank gaps, in a chosen polarity.
    fn page(rows: &[(u32, u32)], height: u32, dark_text: bool) -> DynamicImage {
        let (ink, paper) = if dark_text {
            (Rgb([0, 0, 0]), Rgb([255, 255, 255]))
        } else {
            (Rgb([255, 255, 255]), Rgb([20, 20, 20]))
        };
        let mut image = RgbImage::from_pixel(120, height, paper);
        for &(top, bottom) in rows {
            for y in top..bottom {
                // Strokes with gaps, so the row actually crosses.
                for x in (4..116).step_by(6) {
                    image.put_pixel(x, y, ink);
                    image.put_pixel(x + 1, y, ink);
                }
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    #[test]
    fn blank_space_between_lines_separates_them() {
        let found = lines(&page(&[(4, 20), (30, 46), (56, 72)], 80, true));
        assert_eq!(found.len(), 3);
    }

    /// The reason this splitter counts crossings at all: white lettering on a
    /// dark panel defeated the distance-from-background version, which merged
    /// every line into one run.
    #[test]
    fn light_text_on_a_dark_panel_splits_the_same_as_dark_on_light() {
        let rows = [(4, 20), (30, 46), (56, 72)];
        assert_eq!(
            lines(&page(&rows, 80, false)).len(),
            lines(&page(&rows, 80, true)).len()
        );
    }

    /// Lettering set tight leaves fewer blank rows than inked ones. Reading the
    /// floor off the median then reads it off a line of text, the gate lands
    /// above every line, and the whole block comes back as one strip.
    #[test]
    fn lines_that_fill_most_of_the_block_still_separate() {
        let rows = [(1, 15), (17, 31), (33, 47), (49, 63)];
        assert_eq!(lines(&page(&rows, 64, true)).len(), 4);
    }

    /// The balloon outline crosses every row, so the profile never reaches
    /// zero. The gate has to be measured up from that floor.
    #[test]
    fn an_outline_around_the_block_does_not_merge_its_lines() {
        let mut image = page(&[(6, 22), (30, 46)], 56, true).to_rgb8();
        for y in 0..56 {
            image.put_pixel(1, y, Rgb([0, 0, 0]));
            image.put_pixel(118, y, Rgb([0, 0, 0]));
        }
        assert_eq!(lines(&DynamicImage::ImageRgb8(image)).len(), 2);
    }

    /// Two bands that share rows hand the recogniser the same line twice, and
    /// it duly reads it twice.
    #[test]
    fn lines_never_share_rows_with_each_other() {
        let found = lines(&page(&[(4, 20), (30, 46), (56, 72)], 80, true));
        for pair in found.windows(2) {
            assert!(
                pair[0].1 <= pair[1].0,
                "dải {:?} và {:?} chồng nhau",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn a_single_line_comes_back_as_one_row() {
        assert_eq!(lines(&page(&[(6, 26)], 32, true)).len(), 1);
    }

    /// A block with nothing in it must still be readable as one strip rather
    /// than vanishing — the caller has already decided there is text here.
    #[test]
    fn a_blank_block_comes_back_whole() {
        assert_eq!(lines(&page(&[], 40, true)), vec![(0, 40)]);
    }

    #[test]
    fn a_sliver_too_small_to_hold_text_comes_back_whole() {
        assert_eq!(lines(&page(&[(1, 3)], 4, true)), vec![(0, 4)]);
    }

    #[test]
    fn the_mask_lets_each_position_see_itself_and_no_further() {
        let mask = causal_mask(3);
        assert_eq!(mask[0], 0.0);
        assert!(mask[1].is_infinite() && mask[1].is_sign_negative());
        assert_eq!(mask[3], 0.0);
        assert_eq!(mask[4], 0.0);
        assert!(mask[5].is_infinite());
        assert_eq!(&mask[6..9], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_strip_is_scaled_to_the_models_height_and_a_rounded_width() {
        let (width, pixels) = prepare(&page(&[(2, 10)], 16, true));
        assert_eq!(width % WIDTH_ROUND_TO, 0);
        assert!((MIN_WIDTH..=MAX_WIDTH).contains(&width));
        assert_eq!(pixels.len(), 3 * IMAGE_HEIGHT as usize * width as usize);
        assert!(pixels.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    /// A very wide strip must be capped rather than sent at its own width — the
    /// graph accepts it, but the model was never trained past this.
    #[test]
    fn an_extremely_wide_strip_is_capped() {
        let wide = DynamicImage::ImageRgb8(RgbImage::from_pixel(4000, 20, Rgb([255; 3])));
        assert_eq!(prepare(&wide).0, MAX_WIDTH);
    }
}

/// Read a real corpus with the local recogniser and score it against readings
/// already known to be right.
///
/// The numbers in the module docs came from this. Run it after touching
/// [`lines`] or [`prepare`]: both change what the model sees, and the model is
/// sensitive to both.
///
/// `KOHARU_VIETOCR_DIR=<onnx dir> KOHARU_CORPUS=<dir with trans/ and
/// pairs.jsonl> cargo test --release -p koharu-ml --lib measure_local_ocr --
/// --ignored --nocapture`
#[cfg(test)]
mod measure {
    use super::*;

    /// Case and line breaks are typesetting, not reading.
    fn normalise(text: &str) -> String {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_uppercase()
    }

    fn edit_distance(a: &[char], b: &[char]) -> usize {
        let mut previous: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.iter().enumerate() {
            let mut current = vec![i + 1];
            for (j, cb) in b.iter().enumerate() {
                current.push(
                    (previous[j + 1] + 1)
                        .min(current[j] + 1)
                        .min(previous[j] + usize::from(ca != cb)),
                );
            }
            previous = current;
        }
        previous[b.len()]
    }

    /// Read one page end to end — detect the text blocks, then read each with
    /// the local recogniser — and print what came out.
    ///
    /// `KOHARU_VIETOCR_DIR=<onnx dir> KOHARU_PAGE=<image> cargo test --release
    /// -p koharu-ml --lib read_one_page -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn read_one_page() -> Result<()> {
        let models = std::path::PathBuf::from(
            std::env::var_os("KOHARU_VIETOCR_DIR").expect("set KOHARU_VIETOCR_DIR"),
        );
        let page =
            std::path::PathBuf::from(std::env::var_os("KOHARU_PAGE").expect("set KOHARU_PAGE"));

        koharu_llm::sys::initialize()?;
        let backend = std::sync::Arc::new(koharu_llm::safe::llama_backend::LlamaBackend::init()?);
        let runtime = tokio::runtime::Runtime::new()?;
        let ml = runtime.block_on(crate::facade::Model::new(false, backend))?;

        let mut document = koharu_types::Document::open(page.clone())?;
        runtime.block_on(ml.detect(&mut document))?;

        let reader = VietOcr::open(&models)?;
        let sheet = image::open(&page)?;

        let out_dir = std::path::PathBuf::from(
            std::env::var_os("KOHARU_OUT").unwrap_or_else(|| "test/output".into()),
        );
        std::fs::create_dir_all(&out_dir)?;
        // ASCII only: the source pages are named 表紙0297.JPG, and a filename
        // carrying that does not open from every tool the output is read in.
        let name: String = page
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .chars()
            .filter(char::is_ascii)
            .collect();
        let stem = if name.is_empty() {
            "page".to_string()
        } else {
            name
        };

        let mut report = format!(
            "{}\n{} khung\n\n",
            page.display(),
            document.text_blocks.len()
        );
        let mut rows_html = String::new();
        let mut marked = sheet.to_rgb8();

        println!(
            "\n{}: {} khung\n",
            page.display(),
            document.text_blocks.len()
        );
        for (index, block) in document.text_blocks.iter().enumerate() {
            const PAD: f32 = 8.0;
            let left = (block.x - PAD).max(0.0) as u32;
            let top = (block.y - PAD).max(0.0) as u32;
            let right = ((block.x + block.width + PAD) as u32).min(sheet.width());
            let bottom = ((block.y + block.height + PAD) as u32).min(sheet.height());
            if right <= left || bottom <= top {
                continue;
            }
            let (w, h) = (right - left, bottom - top);

            // Both ways, side by side: the mask clears artwork out of a block
            // but can shave a tone mark off with it. Reading with the mask is
            // what the corpus build does.
            let raw = sheet.crop_imm(left, top, w, h);
            let filtered = document
                .segment
                .as_ref()
                .map(|mask| keep_glyphs(&raw, &mask.0.crop_imm(left, top, w, h)))
                .unwrap_or_else(|| raw.clone());

            let raw_text = reader.read_block(&raw)?;
            let filtered_text = reader.read_block(&filtered)?;
            let (raw_rows, filtered_rows) = (lines(&raw).len(), lines(&filtered).len());

            // The crops actually handed to the recogniser, so a wrong reading
            // can be checked against what it was looking at.
            let raw_file = format!("{stem}_khung{index:02}_goc.png");
            let filtered_file = format!("{stem}_khung{index:02}_loc.png");
            raw.save(out_dir.join(&raw_file))?;
            filtered.save(out_dir.join(&filtered_file))?;
            outline(&mut marked, left, top, right, bottom, index);

            println!(
                "[{index:2}] ({left},{top}) {w}x{h}  {filtered_rows} dòng\n     {}",
                filtered_text.replace('\n', "\n     ")
            );
            report.push_str(&format!(
                "[{index}] ({left},{top}) {w}x{h}  {filtered_rows} dòng\n{filtered_text}\n\n"
            ));
            rows_html.push_str(&format!(
                "<tr><td class=n>{index}<div class=meta>({left},{top})<br>{w}×{h}</div></td>\
                 <td><img src=\"{raw_file}\"></td>\
                 <td class=t>{}<div class=meta>{raw_rows} dòng</div></td>\
                 <td><img src=\"{filtered_file}\"></td>\
                 <td class=t>{}<div class=meta>{filtered_rows} dòng</div></td></tr>\n",
                escape(&raw_text),
                escape(&filtered_text),
            ));
        }

        let image_path = out_dir.join(format!("{stem}_khung.png"));
        let text_path = out_dir.join(format!("{stem}.txt"));
        let html_path = out_dir.join(format!("{stem}_kiemtra.html"));
        image::DynamicImage::ImageRgb8(marked).save(&image_path)?;
        std::fs::write(&text_path, &report)?;
        std::fs::write(
            &html_path,
            format!(
                "<!doctype html><html><head><meta charset=utf-8><title>{stem} — vietocr</title>\
                 <style>body{{font:14px -apple-system,system-ui,sans-serif;margin:24px;background:#fafafa;color:#111}}\
                 h1{{font-size:17px;margin:0 0 4px}}p{{color:#666;margin:0 0 18px}}\
                 table{{border-collapse:collapse;width:100%;background:#fff}}\
                 th{{text-align:left;background:#eee;padding:9px;position:sticky;top:0;font-size:12px}}\
                 td{{border-top:1px solid #e0e0e0;padding:10px;vertical-align:top}}\
                 td.n{{color:#999;width:64px}}img{{max-width:300px;border:1px solid #ccc;display:block;background:#fff}}\
                 .meta{{color:#aaa;font-size:11px;margin-top:5px;line-height:1.4}}\
                 td.t{{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:13px;min-width:170px}}</style>\
                 </head><body><h1>{stem} — ô đã cắt và chữ vietocr đọc ra</h1>\
                 <p>{} khung. Khung đánh số trên ảnh: <a href=\"{stem}_khung.png\">{stem}_khung.png</a></p>\
                 <table><tr><th>#</th><th>ô cắt nguyên bản</th><th>vietocr đọc</th>\
                 <th>ô cắt đã lọc nét chữ</th><th>vietocr đọc</th></tr>\n{rows_html}</table></body></html>",
                document.text_blocks.len()
            ),
        )?;
        println!(
            "\nghi ra:\n  {}\n  {}\n  {}",
            html_path.display(),
            text_path.display(),
            image_path.display()
        );
        Ok(())
    }

    fn escape(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    }

    /// Draw a numbered box so the picture and the text file line up.
    fn outline(
        canvas: &mut image::RgbImage,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        index: usize,
    ) {
        let red = image::Rgb([255u8, 0, 0]);
        for thickness in 0..3 {
            for x in left..right {
                for y in [top + thickness, bottom.saturating_sub(thickness + 1)] {
                    if y < canvas.height() {
                        canvas.put_pixel(x, y, red);
                    }
                }
            }
            for y in top..bottom {
                for x in [left + thickness, right.saturating_sub(thickness + 1)] {
                    if x < canvas.width() {
                        canvas.put_pixel(x, y, red);
                    }
                }
            }
        }
        // A bar of `index + 1` ticks above the box, readable without a font.
        for tick in 0..=index {
            let x0 = left + tick as u32 * 10;
            for dx in 0..7 {
                for dy in 0..7 {
                    let (x, y) = (x0 + dx, top.saturating_sub(12) + dy);
                    if x < canvas.width() && y < canvas.height() {
                        canvas.put_pixel(x, y, red);
                    }
                }
            }
        }
    }

    /// Show how one crop was split into lines, and what each line read as.
    ///
    /// The split is where the accuracy is, and a wrong reading rarely says
    /// which band caused it. This writes every strip out beside its text.
    ///
    /// `KOHARU_VIETOCR_DIR=<onnx dir> KOHARU_CROP=<png> cargo test --release
    /// -p koharu-ml --lib inspect_one_crop -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn inspect_one_crop() -> Result<()> {
        let models = std::path::PathBuf::from(
            std::env::var_os("KOHARU_VIETOCR_DIR").expect("set KOHARU_VIETOCR_DIR"),
        );
        let path =
            std::path::PathBuf::from(std::env::var_os("KOHARU_CROP").expect("set KOHARU_CROP"));
        let out_dir = std::path::PathBuf::from(
            std::env::var_os("KOHARU_OUT").unwrap_or_else(|| "test/output".into()),
        );
        std::fs::create_dir_all(&out_dir)?;

        let crop = image::open(&path)?;
        let reader = VietOcr::open(&models)?;
        let stem: String = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .chars()
            .filter(char::is_ascii)
            .collect();

        let bands = rows_and_columns(&crop);
        println!(
            "\n{} — {}x{} → {} dòng (nhịp {:?})",
            path.display(),
            crop.width(),
            crop.height(),
            bands.len(),
            debug_pitch(&crop)
        );

        for (index, (top, bottom, left, right)) in bands.iter().copied().enumerate() {
            const AIR: u32 = 3;
            let t = top.saturating_sub(AIR);
            let b = (bottom + AIR).min(crop.height());
            let strip = crop.crop_imm(left, t, right - left, b - t);
            let text = reader.read_line(&strip)?;
            let file = out_dir.join(format!("{stem}_dong{index:02}.png"));
            strip.save(&file)?;
            println!(
                "  [{index}] y {top}..{bottom} (cao {}), x {left}..{right}  →  {text:?}",
                bottom - top
            );
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn measure_local_ocr() -> Result<()> {
        let models = std::path::PathBuf::from(
            std::env::var_os("KOHARU_VIETOCR_DIR").expect("set KOHARU_VIETOCR_DIR"),
        );
        let root =
            std::path::PathBuf::from(std::env::var_os("KOHARU_CORPUS").expect("set KOHARU_CORPUS"));
        // Named separately, and never the file the corpus build writes. Scoring
        // against that file once produced a flattering 7.9% character error
        // rate that was really this reader being compared against itself.
        let answers = std::path::PathBuf::from(
            std::env::var_os("KOHARU_ANSWERS")
                .expect("set KOHARU_ANSWERS to a pairs.jsonl whose readings are known to be right"),
        );

        let reader = VietOcr::open(&models)?;
        let mut page_cache: Option<(String, DynamicImage)> = None;
        let (mut exact, mut total) = (0usize, 0usize);
        let (mut errors, mut characters) = (0usize, 0usize);

        for line in std::fs::read_to_string(&answers)?.lines() {
            let record: serde_json::Value = serde_json::from_str(line)?;
            let page = record["page"].as_str().unwrap_or_default().to_string();
            let expected = normalise(record["target"].as_str().unwrap_or_default());
            let boxed: Vec<f32> = serde_json::from_value(record["target_box"].clone())?;
            let [x, y, w, h] = boxed[..] else { continue };

            if page_cache.as_ref().map(|(name, _)| name.as_str()) != Some(page.as_str()) {
                page_cache = Some((page.clone(), image::open(root.join("trans").join(&page))?));
            }
            let (_, sheet) = page_cache.as_ref().unwrap();

            const PAD: f32 = 8.0;
            let left = (x - PAD).max(0.0) as u32;
            let top = (y - PAD).max(0.0) as u32;
            let right = ((x + w + PAD) as u32).min(sheet.width());
            let bottom = ((y + h + PAD) as u32).min(sheet.height());
            if right <= left || bottom <= top {
                continue;
            }

            let read = normalise(&reader.read_block(&sheet.crop_imm(
                left,
                top,
                right - left,
                bottom - top,
            ))?);

            total += 1;
            if read == expected {
                exact += 1;
            }
            let (a, b): (Vec<char>, Vec<char>) =
                (expected.chars().collect(), read.chars().collect());
            characters += a.len();
            errors += edit_distance(&a, &b);

            if read != expected {
                println!("  đúng  : {expected}");
                println!("  đọc ra: {read}");
            }
        }

        println!("\n{exact}/{total} khung trùng nguyên văn");
        println!(
            "sai ký tự {:.1}%",
            100.0 * errors as f32 / characters.max(1) as f32
        );
        Ok(())
    }
}
