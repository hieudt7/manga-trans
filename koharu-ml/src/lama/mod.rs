mod fft;
mod model;

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use image::{
    DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage,
    imageops::{crop_imm, replace},
};
use std::collections::HashMap;

use imageproc::{
    contours::find_contours,
    distance_transform::Norm,
    drawing::{draw_hollow_rect_mut, draw_polygon_mut},
    edges::canny,
    filter::gaussian_blur_f32,
    morphology::dilate,
    point::Point,
    rect::Rect,
    region_labelling::{Connectivity, connected_components},
};
use koharu_types::TextBlock;
use tracing::instrument;

use crate::{define_models, device, loading};

define_models! {
    Lama => ("mayocream/lama-manga", "lama-manga.safetensors"),
}

const BALLOON_CANNY_LOW: f32 = 70.0;
const BALLOON_CANNY_HIGH: f32 = 140.0;
const BALLOON_WINDOW_RATIO: f64 = 1.7;
const BALLOON_WINDOW_ASPECT_RATIO: f64 = 1.0;
const SIMPLE_BG_THRESHOLD_LOW_VARIANCE: f64 = 10.0;
const SIMPLE_BG_THRESHOLD_HIGH_VARIANCE: f64 = 7.0;
const SIMPLE_BG_CHANNEL_STD_SWITCH: f64 = 1.0;
const ALPHA_RING_RADIUS: u8 = 7;

/// Context window for text with no real container (free-standing on
/// artwork, an open region, or a box the balloon detector didn't catch) —
/// bigger than `BALLOON_WINDOW_RATIO`, since there is no balloon edge to
/// bound the crop and the model/interpolation benefits from more
/// surrounding art to read from.
const ADAPTIVE_WINDOW_RATIO: f64 = 3.0;
/// How far out from the mask's own edge to read the local background,
/// scaled by the block's own font size — see `ADAPTIVE_REFERENCE_FONT_SIZE_PX`.
const ADAPTIVE_RING_UNITS: f64 = 4.0;
const ADAPTIVE_REFERENCE_FONT_SIZE_PX: f64 = 32.0;
const ADAPTIVE_MIN_SCALE: f64 = 0.5;
const ADAPTIVE_MAX_SCALE: f64 = 3.0;
/// Below this per-channel std-dev, the ring reads as one flat colour.
const ADAPTIVE_FLAT_STD_THRESHOLD: f64 = 10.0;
/// Per-channel tolerance for `inlier_fraction`'s majority vote — generous
/// enough to absorb ordinary scan noise/halftone texture in real ink, not
/// just a synthetically flat fill.
const ADAPTIVE_FLAT_INLIER_TOLERANCE: f64 = 30.0;
/// Below this RMS residual, a planar fit explains the ring well enough to
/// call it a simple gradient rather than screentone/line art — above it,
/// only the neural model is trusted.
const ADAPTIVE_GRADIENT_RESIDUAL_THRESHOLD: f64 = 12.0;
/// A planar fit needs more points than unknowns (3) to mean anything.
const ADAPTIVE_MIN_RING_SAMPLES: usize = 12;

/// Columns vs. blocks (see `.claude/Redraw.md`'s follow-up): one logical
/// FREE_TEXT/OPEN_REGION_TEXT/BOX_TEXT block can hold several vertical
/// columns of source text — the layout detector often boxes them as one
/// block because they sit close together. Splitting that into separate
/// *translation* blocks would break reading order and narration coherence,
/// so the block itself is never split; only the *removal mask* and the
/// crops fed to `restore_region` are done per column, so redraw never
/// touches the real background between columns and never hands one huge,
/// oddly-shaped hole to the model in a single call.
/// A column's ink-density profile at a given x may be filled up to this
/// fraction of the block's own height and still count as "gap". A real
/// letterer's stroke can bridge two visually separate columns for part of
/// their height (measured on a real page: one such bridge covered ~20% of
/// the block) without the columns actually being one run of text — full
/// zero-tolerance was tried first and rejected for exactly this reason.
const COLUMN_GAP_INK_DENSITY_THRESHOLD: f32 = 0.25;
// A real gap between two hand-lettered vertical columns is often much
// narrower than the columns themselves — measured on a real page, ~15px of
// clear gap between ~50px-wide columns — so this only needs to clear noise,
// not scale up to "as wide as a column"; `strip_is_light_rgb` below is what
// actually protects an ordinary balloon paragraph from being split on its
// own (much lighter, but often narrower still) inter-line whitespace.
const COLUMN_GAP_MIN_RATIO: f32 = 0.2;
const COLUMN_GAP_MIN_PX: f32 = 10.0;
/// A gap this light and this flat is blank balloon padding, not real
/// separation between columns — never split on that alone (see
/// `.claude/Redraw.md`: "Do not split solely based on whitespace").
const COLUMN_GAP_LIGHT_MEAN_THRESHOLD: f64 = 200.0;
const COLUMN_GAP_LIGHT_STD_THRESHOLD: f64 = 25.0;
const COLUMN_PAD_PX: f32 = 2.0;
/// How far around a block's own bbox to look for its ink — small, just
/// enough to catch a component clipped at the box edge, not a context
/// window (that comes later, per column).
const COLUMN_DETECT_MARGIN_PX: u32 = 16;

type Xyxy = [u32; 4];

/// Recover bright outlines missed by the semantic segmenter, then add a small
/// anti-alias margin. The bbox only limits the search; it is never filled.
/// Return separate source and removal masks in page coordinates.
pub fn adaptive_text_masks(
    image: &RgbImage,
    segmentation: &GrayImage,
    blocks: &[TextBlock],
) -> (GrayImage, GrayImage) {
    let (w, h) = image.dimensions();
    let mut source = GrayImage::new(w, h);
    let mut removal = GrayImage::new(w, h);
    for block in blocks {
        let Some(bounds) = block_xyxy(block, w, h) else {
            continue;
        };
        // PPDocLayout's "font size" is the block's short side, which can
        // contain several CJK columns. Do not scale recovery from that width.
        let count = block.text.as_deref().unwrap_or("").chars()
            .filter(|c| !c.is_whitespace()).count();
        let measured = (count > 0).then(|| (block.width * block.height / count as f32).sqrt());
        let hint = block.detected_font_size_px.filter(|s| s.is_finite() && *s > 0.0);
        let glyph_size = match (measured, hint) {
            (Some(size), Some(hint)) => size.min(hint),
            (Some(size), None) | (None, Some(size)) => size,
            (None, None) => ADAPTIVE_REFERENCE_FONT_SIZE_PX as f32,
        };
        let reach = (glyph_size * 0.20).round().clamp(4.0, 12.0) as u8;
        let pad = u32::from(reach) + 4;
        let [x0, y0, x1, y1] = [
            bounds[0].saturating_sub(pad),
            bounds[1].saturating_sub(pad),
            (bounds[2] + pad).min(w),
            (bounds[3] + pad).min(h),
        ];
        let mut seed = crop_imm(segmentation, x0, y0, x1 - x0, y1 - y0).to_image();
        // Segmentation can omit a long em dash entirely. Recover only isolated,
        // thin punctuation aligned with the known reading direction, and only
        // when OCR actually reported it. Never threshold all ink in the bbox.
        if block
            .text
            .as_deref()
            .is_some_and(|t| t.contains('—') || t.contains('―'))
        {
            let dark = GrayImage::from_fn(x1 - x0, y1 - y0, |x, y| {
                Luma([
                    if image.get_pixel(x0 + x, y0 + y).0.iter().all(|c| *c < 150) {
                        255
                    } else {
                        0
                    },
                ])
            });
            let labels = connected_components(&dark, Connectivity::Eight, Luma([0]));
            let near_text = dilate(&seed, Norm::LInf, (glyph_size * 0.5).clamp(4.0, 24.0) as u8);
            let mut components: HashMap<u32, (Xyxy, bool)> = HashMap::new();
            for (x, y, label) in labels.enumerate_pixels() {
                if label[0] == 0 {
                    continue;
                }
                let entry = components
                    .entry(label[0])
                    .or_insert(([x, y, x + 1, y + 1], false));
                entry.0[0] = entry.0[0].min(x);
                entry.0[1] = entry.0[1].min(y);
                entry.0[2] = entry.0[2].max(x + 1);
                entry.0[3] = entry.0[3].max(y + 1);
                entry.1 |= near_text.get_pixel(x, y)[0] != 0;
            }
            let vertical = block.source_direction == Some(koharu_types::TextDirection::Vertical);
            let accepted: std::collections::HashSet<u32> = components
                .into_iter()
                .filter_map(|(id, (b, near))| {
                    let (cw, ch) = ((b[2] - b[0]) as f32, (b[3] - b[1]) as f32);
                    let (long, short) = if vertical { (ch, cw) } else { (cw, ch) };
                    (near
                        && b[0] > 0
                        && b[1] > 0
                        && b[2] < x1 - x0
                        && b[3] < y1 - y0
                        && long > short * 6.0
                        && short <= glyph_size * 0.18
                        && long >= glyph_size * 0.7
                        && long <= glyph_size * 4.0)
                        .then_some(id)
                })
                .collect();
            for (x, y, label) in labels.enumerate_pixels() {
                if accepted.contains(&label[0]) {
                    seed.put_pixel(x, y, Luma([255]));
                }
            }
        }
        let nearby = dilate(&seed, Norm::LInf, reach);
        let mut glyphs = seed.clone();
        // A white outline is high contrast against the surrounding dark ink.
        // Restrict recovery to a narrow distance from recognized text so a
        // connected speed line cannot flood the entire panel into the mask.
        for (x, y, px) in nearby.enumerate_pixels() {
            let rgb = image.get_pixel(x0 + x, y0 + y);
            if px[0] != 0 && rgb.0.iter().all(|c| *c >= 180) {
                glyphs.put_pixel(x, y, Luma([255]));
            }
        }
        // The recovered stroke may enclose dark source ink the segmenter
        // missed. Keep exterior gaps open; include only small enclosed holes.
        let holes = GrayImage::from_fn(x1 - x0, y1 - y0, |x, y| {
            Luma([255 - glyphs.get_pixel(x, y)[0]])
        });
        let labels = connected_components(&holes, Connectivity::Four, Luma([0]));
        let mut sizes = HashMap::<u32, usize>::new();
        let mut exterior = std::collections::HashSet::new();
        for (x, y, label) in labels.enumerate_pixels() {
            if label[0] == 0 {
                continue;
            }
            *sizes.entry(label[0]).or_default() += 1;
            if x == 0 || y == 0 || x + 1 == x1 - x0 || y + 1 == y1 - y0 {
                exterior.insert(label[0]);
            }
        }
        for (x, y, label) in labels.enumerate_pixels() {
            if label[0] != 0
                && !exterior.contains(&label[0])
                && sizes[&label[0]] as f32 <= glyph_size * glyph_size
            {
                glyphs.put_pixel(x, y, Luma([255]));
            }
        }
        let margin = (glyph_size * 0.06).round().clamp(1.0, 4.0) as u8;
        let expanded = dilate(&glyphs, Norm::LInf, margin);
        for (x, y, px) in glyphs.enumerate_pixels() {
            if px[0] != 0 {
                source.put_pixel(x0 + x, y0 + y, *px);
            }
        }
        for (x, y, px) in expanded.enumerate_pixels() {
            if px[0] != 0 {
                removal.put_pixel(x0 + x, y0 + y, *px);
            }
        }
    }
    (source, removal)
}

struct BalloonMasks {
    balloon_mask: GrayImage,
    non_text_mask: GrayImage,
}

pub struct Lama {
    model: model::Lama,
    device: Device,
}

/// Glyphs are separate blobs; merging at this radius groups a run of text into
/// one region instead of inpainting it a stroke at a time.
const LEFTOVER_MERGE_RADIUS: u8 = 12;
/// Smaller than this is mask noise, not writing.
const LEFTOVER_MIN_PIXELS: u32 = 24;
/// Context around a region for the model to match against.
const LEFTOVER_PAD_PX: u32 = 8;
/// A region this large is a runaway mask rather than a run of text.
const LEFTOVER_MAX_AREA_SHARE: f32 = 0.05;
/// How far past a block window a run of text may continue and still count as
/// the same run. Beyond this the mark belongs to no block at all.
const LEFTOVER_ADJACENCY_PX: u32 = 16;

/// Do the boxes touch, allowing for a run of text carrying on just past the
/// window edge?
fn near(a: Xyxy, b: Xyxy) -> bool {
    let gap = LEFTOVER_ADJACENCY_PX;
    a[0] <= b[2] + gap && b[0] <= a[2] + gap && a[1] <= b[3] + gap && b[1] <= a[3] + gap
}

/// Bounding boxes of mask content that no block window covered, restricted to
/// what continues a run of text a block window already started erasing.
///
/// The restriction is the whole point. The mask marks every glyph on the page,
/// sound effects drawn across the artwork included, and those are not text the
/// pipeline replaces — erasing one leaves the model guessing at whatever the
/// letters were drawn over. What must be finished is the run a block window
/// clipped: erasing half of a line and leaving the rest standing is the failure
/// this pass exists to fix.
fn leftover_mask_regions(mask: &GrayImage, windows: &[Xyxy]) -> Vec<[u32; 4]> {
    let (width, height) = mask.dimensions();
    if !mask.pixels().any(|pixel| pixel[0] >= 128) {
        return Vec::new();
    }

    let merged = dilate(mask, Norm::L1, LEFTOVER_MERGE_RADIUS);
    let labels = connected_components(&merged, Connectivity::Eight, Luma([0u8]));

    let mut boxes: HashMap<u32, [u32; 4]> = HashMap::new();
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for (x, y, label) in labels.enumerate_pixels() {
        let id = label[0];
        if id == 0 || mask.get_pixel(x, y)[0] < 128 {
            continue;
        }
        *counts.entry(id).or_insert(0) += 1;
        let entry = boxes.entry(id).or_insert([x, y, x + 1, y + 1]);
        entry[0] = entry[0].min(x);
        entry[1] = entry[1].min(y);
        entry[2] = entry[2].max(x + 1);
        entry[3] = entry[3].max(y + 1);
    }

    let page_area = (width as f32) * (height as f32);
    boxes
        .into_iter()
        .filter(|(id, _)| counts.get(id).copied().unwrap_or(0) >= LEFTOVER_MIN_PIXELS)
        .filter(|(_, bbox)| windows.iter().any(|window| near(*bbox, *window)))
        .map(|(_, bbox)| {
            [
                bbox[0].saturating_sub(LEFTOVER_PAD_PX),
                bbox[1].saturating_sub(LEFTOVER_PAD_PX),
                (bbox[2] + LEFTOVER_PAD_PX).min(width),
                (bbox[3] + LEFTOVER_PAD_PX).min(height),
            ]
        })
        .filter(|bbox| {
            let area = ((bbox[2] - bbox[0]) as f32) * ((bbox[3] - bbox[1]) as f32);
            bbox[2] > bbox[0] && bbox[3] > bbox[1] && area <= page_area * LEFTOVER_MAX_AREA_SHARE
        })
        .collect()
}

impl Lama {
    pub async fn load(cpu: bool) -> Result<Self> {
        let device = device(cpu)?;
        let model = loading::load_buffered_safetensors(Manifest::Lama.get(), &device, |vb| {
            model::Lama::load(&vb)
        })
        .await?;

        Ok(Self { model, device })
    }

    #[instrument(level = "debug", skip_all)]
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.model.forward(image, mask)
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference_model(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
    ) -> Result<DynamicImage> {
        let (image_tensor, mask_tensor) = self.preprocess(image, mask)?;
        let output = self.forward(&image_tensor, &mask_tensor)?;
        self.postprocess(&output)
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference(&self, image: &DynamicImage, mask: &DynamicImage) -> Result<DynamicImage> {
        self.inference_with_blocks(image, mask, None)
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_blocks(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
        text_blocks: Option<&[TextBlock]>,
    ) -> Result<DynamicImage> {
        if image.dimensions() != mask.dimensions() {
            bail!(
                "image and mask dimensions dismatch: image is {:?}, mask is {:?}",
                image.dimensions(),
                mask.dimensions()
            );
        }

        let binary_mask = binarize_mask(mask);
        let output_rgb = if let Some(blocks) = text_blocks.filter(|blocks| !blocks.is_empty()) {
            let image_rgb = image.to_rgb8();
            self.inference_blockwise(&image_rgb, &binary_mask, blocks)?
        } else {
            self.inference_crop(&image.to_rgb8(), &binary_mask)?
        };

        if image.color().has_alpha() {
            let original_alpha = image.to_rgba8();
            let alpha = extract_alpha(&original_alpha);
            let output = restore_alpha_channel(&output_rgb, &alpha, &binary_mask);
            Ok(DynamicImage::ImageRgba8(output))
        } else {
            Ok(DynamicImage::ImageRgb8(output_rgb))
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn inference_crop(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        if let Some(filled) = try_fill_balloon(image, mask) {
            return Ok(filled);
        }

        self.inference_model_rgb(image, mask)
    }

    #[instrument(level = "debug", skip_all)]
    fn inference_blockwise(
        &self,
        image: &RgbImage,
        mask: &GrayImage,
        text_blocks: &[TextBlock],
    ) -> Result<RgbImage> {
        let (im_w, im_h) = image.dimensions();
        let mut inpainted = image.clone();
        let mut working_mask = mask.clone();
        let mut windows: Vec<Xyxy> = Vec::with_capacity(text_blocks.len());

        for block in text_blocks {
            let Some(xyxy) = block_xyxy(block, im_w, im_h) else {
                continue;
            };
            let xyxy_e = enlarge_window(
                xyxy,
                im_w,
                im_h,
                BALLOON_WINDOW_RATIO,
                BALLOON_WINDOW_ASPECT_RATIO,
            );
            let crop_width = xyxy_e[2].saturating_sub(xyxy_e[0]);
            let crop_height = xyxy_e[3].saturating_sub(xyxy_e[1]);
            if crop_width == 0 || crop_height == 0 {
                continue;
            }

            let crop_image =
                crop_imm(&inpainted, xyxy_e[0], xyxy_e[1], crop_width, crop_height).to_image();
            let crop_mask =
                crop_imm(&working_mask, xyxy_e[0], xyxy_e[1], crop_width, crop_height).to_image();

            let output = if count_nonzero(&crop_mask) == 0 {
                crop_image
            } else if let Some(filled) = try_fill_balloon(&crop_image, &crop_mask) {
                filled
            } else {
                self.inference_model_rgb(&crop_image, &crop_mask)?
            };

            replace(
                &mut inpainted,
                &output,
                i64::from(xyxy_e[0]),
                i64::from(xyxy_e[1]),
            );
            // The crop was inpainted against the mask over the whole enlarged
            // window, so every mark inside it is dealt with — clearing only the
            // block's own bbox would send the rest round again below.
            clear_mask_bbox(&mut working_mask, xyxy_e);
            windows.push(xyxy_e);
        }

        // A run of text that starts inside a block window and carries on past
        // its edge comes out half-erased: the part inside cleaned, the rest
        // standing. Finish those runs, and only those — see
        // `leftover_mask_regions` for why the rest of the mask is left alone.
        for region in leftover_mask_regions(&working_mask, &windows) {
            let [x1, y1, x2, y2] = region;
            let (w, h) = (x2 - x1, y2 - y1);
            let crop_image = crop_imm(&inpainted, x1, y1, w, h).to_image();
            let crop_mask = crop_imm(&working_mask, x1, y1, w, h).to_image();
            let output = match try_fill_balloon(&crop_image, &crop_mask) {
                Some(filled) => filled,
                None => self.inference_model_rgb(&crop_image, &crop_mask)?,
            };
            replace(&mut inpainted, &output, i64::from(x1), i64::from(y1));
        }

        Ok(inpainted)
    }

    /// Redraw for text with no real container — see `.claude/Redraw.md`.
    /// Runs over the *result* of an earlier container-block pass (or the
    /// original image, if there was none), so the two compose on one shared
    /// canvas. Unlike `inference_blockwise`, this never treats a traced
    /// balloon-shaped contour as "the background": there is no balloon here,
    /// so it reads the *immediate* ring around each block's own precise
    /// glyph mask instead, and picks flat-fill / planar-gradient /
    /// neural-model per block from what that ring actually looks like.
    #[instrument(level = "debug", skip_all)]
    pub fn inference_adaptive(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
        text_blocks: &[TextBlock],
    ) -> Result<DynamicImage> {
        if text_blocks.is_empty() {
            return Ok(image.clone());
        }
        if image.dimensions() != mask.dimensions() {
            bail!(
                "image and mask dimensions dismatch: image is {:?}, mask is {:?}",
                image.dimensions(),
                mask.dimensions()
            );
        }

        let binary_mask = binarize_mask(mask);
        let image_rgb = image.to_rgb8();
        let output_rgb =
            self.inference_adaptive_blockwise(&image_rgb, &binary_mask, text_blocks)?;

        if image.color().has_alpha() {
            let original_alpha = image.to_rgba8();
            let alpha = extract_alpha(&original_alpha);
            let output = restore_alpha_channel(&output_rgb, &alpha, &binary_mask);
            Ok(DynamicImage::ImageRgba8(output))
        } else {
            Ok(DynamicImage::ImageRgb8(output_rgb))
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn inference_adaptive_blockwise(
        &self,
        image: &RgbImage,
        mask: &GrayImage,
        text_blocks: &[TextBlock],
    ) -> Result<RgbImage> {
        let (im_w, im_h) = image.dimensions();
        let mut inpainted = image.clone();
        let mut working_mask = mask.clone();
        let mut windows: Vec<Xyxy> = Vec::with_capacity(text_blocks.len());
        let debug = std::env::var("KOHARU_DEBUG_ADAPTIVE").is_ok();

        for block in text_blocks {
            let Some(xyxy) = block_xyxy(block, im_w, im_h) else {
                continue;
            };
            // Columns, not the whole block, are the unit of redraw work —
            // see `.claude/Redraw.md`: the block stays one translation
            // unit, but handing its whole (possibly multi-column, spindly)
            // mask to one inpaint call is what produced a near-identity
            // "restoration" on a page this session already hit. Each
            // column gets its own small context crop and its own precise
            // mask instead.
            let columns = cluster_columns(&inpainted, &working_mask, xyxy);
            if debug {
                tracing::info!(
                    block_id = %block.id,
                    xyxy = ?xyxy,
                    column_count = columns.len(),
                    columns = ?columns.iter().map(|c| (c.reading_order, c.bbox)).collect::<Vec<_>>(),
                    "inference_adaptive_blockwise: clustered columns"
                );
                write_column_debug(&inpainted, &block.id, xyxy, &columns);
            }

            for column in &columns {
                let xyxy_e = enlarge_window(
                    column.bbox,
                    im_w,
                    im_h,
                    ADAPTIVE_WINDOW_RATIO,
                    BALLOON_WINDOW_ASPECT_RATIO,
                );
                let crop_width = xyxy_e[2].saturating_sub(xyxy_e[0]);
                let crop_height = xyxy_e[3].saturating_sub(xyxy_e[1]);
                if crop_width == 0 || crop_height == 0 {
                    continue;
                }

                let crop_image =
                    crop_imm(&inpainted, xyxy_e[0], xyxy_e[1], crop_width, crop_height).to_image();
                // Built from this column's own mask alone — never a crop of
                // the shared page mask, so a neighbouring column's ink can
                // never leak in even though enlarged context windows for
                // adjacent columns can overlap.
                let mut crop_mask = GrayImage::new(crop_width, crop_height);
                for (x, y, pixel) in column.mask.enumerate_pixels() {
                    if pixel.0[0] == 0 {
                        continue;
                    }
                    let (px, py) = (column.mask_origin.0 + x, column.mask_origin.1 + y);
                    if px >= xyxy_e[0] && px < xyxy_e[2] && py >= xyxy_e[1] && py < xyxy_e[3] {
                        crop_mask.put_pixel(px - xyxy_e[0], py - xyxy_e[1], Luma([255]));
                    }
                }
                if count_nonzero(&crop_mask) == 0 {
                    continue;
                }

                // All recognized text in this context is excluded from
                // background sampling, including other logical blocks.
                let exclude_mask =
                    crop_imm(mask, xyxy_e[0], xyxy_e[1], crop_width, crop_height).to_image();

                // Prefer the glyph width the split itself measured over the
                // parent block's own `detected_font_size_px` — that bbox
                // can span several columns, so scaling the ring from it
                // easily makes the ring wider than the real gap the split
                // just found between them, pulling a neighbour's still
                // unprocessed ink into the read.
                let scale = (column
                    .typical_width_hint
                    .map(f64::from)
                    .or(block.detected_font_size_px.map(f64::from))
                    .unwrap_or(ADAPTIVE_REFERENCE_FONT_SIZE_PX)
                    / ADAPTIVE_REFERENCE_FONT_SIZE_PX)
                    .clamp(ADAPTIVE_MIN_SCALE, ADAPTIVE_MAX_SCALE);
                let ring_radius =
                    ((ADAPTIVE_RING_UNITS * scale).round() as u32).clamp(1, 255) as u8;

                let restored =
                    self.restore_region(&crop_image, &crop_mask, &exclude_mask, ring_radius)?;
                if debug {
                    tracing::info!(
                        block_id = %block.id,
                        reading_order = column.reading_order,
                        "inference_adaptive_blockwise: column restored"
                    );
                }
                // Composite ONLY the mask pixels back — even a model that
                // "fixes" pixels elsewhere in this window must never move
                // artwork the mask never claimed.
                composite_masked(&mut inpainted, &restored, &crop_mask, xyxy_e[0], xyxy_e[1]);
                for (x, y, pixel) in crop_mask.enumerate_pixels() {
                    if pixel[0] != 0 {
                        working_mask.put_pixel(xyxy_e[0] + x, xyxy_e[1] + y, Luma([0]));
                    }
                }
                windows.push(xyxy_e);
            }
        }

        for region in leftover_mask_regions(&working_mask, &windows) {
            let [x1, y1, x2, y2] = region;
            let (w, h) = (x2 - x1, y2 - y1);
            let crop_image = crop_imm(&inpainted, x1, y1, w, h).to_image();
            let crop_mask = crop_imm(&working_mask, x1, y1, w, h).to_image();
            let no_exclusions = GrayImage::new(w, h);
            let restored = self.restore_region(
                &crop_image,
                &crop_mask,
                &no_exclusions,
                ADAPTIVE_RING_UNITS as u8,
            )?;
            composite_masked(&mut inpainted, &restored, &crop_mask, x1, y1);
        }

        Ok(inpainted)
    }

    /// Decide, from the ring of pixels immediately around `mask`, whether
    /// the local background is flat, a simple gradient, or complex enough
    /// (screentone, line art, speed lines) to need the neural model —
    /// see `.claude/Redraw.md` §2.
    fn restore_region(
        &self,
        image: &RgbImage,
        mask: &GrayImage,
        exclude: &GrayImage,
        ring_radius: u8,
    ) -> Result<RgbImage> {
        let plan = plan_restoration(image, mask, exclude, ring_radius);
        if std::env::var("KOHARU_DEBUG_FREE_TEXT").is_ok() {
            tracing::info!(?plan, ring_radius, "restore_region: plan chosen");
        }
        match plan {
            RestorationPlan::Flat(color) => Ok(flat_fill(image, mask, color)),
            RestorationPlan::Gradient(plane) => Ok(gradient_fill(image, mask, &plane)),
            RestorationPlan::Complex => self.inference_model_rgb(image, mask),
        }
    }

    #[instrument(level = "debug", skip_all)]
    fn inference_model_rgb(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        Ok(self
            .inference_model(
                &DynamicImage::ImageRgb8(image.clone()),
                &DynamicImage::ImageLuma8(mask.clone()),
            )?
            .to_rgb8())
    }

    #[instrument(level = "debug", skip_all)]
    fn preprocess(&self, image: &DynamicImage, mask: &DynamicImage) -> Result<(Tensor, Tensor)> {
        if image.dimensions() != mask.dimensions() {
            bail!(
                "image and mask dimensions dismatch: image is {:?}, mask is {:?}",
                image.dimensions(),
                mask.dimensions()
            );
        }
        let (w, h) = (image.width() as usize, image.height() as usize);

        let rgb = image.to_rgb8().into_raw();
        let luma = mask.to_luma8().into_raw();

        let image_tensor = (Tensor::from_vec(rgb, (1, h, w, 3), &self.device)?
            .permute((0, 3, 1, 2))?
            .to_dtype(DType::F32)?
            * (1. / 255.))?;

        let mask_tensor = Tensor::from_vec(luma, (1, h, w, 1), &self.device)?
            .permute((0, 3, 1, 2))?
            .to_dtype(DType::F32)?
            .gt(1.0f32)?;

        Ok((image_tensor, mask_tensor))
    }

    #[instrument(level = "debug", skip_all)]
    fn postprocess(&self, output: &Tensor) -> Result<DynamicImage> {
        let output = output.squeeze(0)?;
        let (channels, height, width) = output.dims3()?;
        if channels != 3 {
            bail!("expected 3 channels in output, got {channels}");
        }
        let output = (output * 255.)?
            .clamp(0., 255.)?
            .permute((1, 2, 0))?
            .to_dtype(DType::U8)?;
        let raw: Vec<u8> = output.flatten_all()?.to_vec1()?;
        let image = RgbImage::from_raw(width as u32, height as u32, raw)
            .ok_or_else(|| anyhow::anyhow!("failed to create image buffer from model output"))?;
        Ok(DynamicImage::ImageRgb8(image))
    }
}

fn binarize_mask(mask: &DynamicImage) -> GrayImage {
    let mut binary = mask.to_luma8();
    for pixel in binary.pixels_mut() {
        pixel.0[0] = if pixel.0[0] > 127 { 255 } else { 0 };
    }
    binary
}

fn extract_alpha(image: &RgbaImage) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut alpha = GrayImage::new(width, height);
    for (x, y, pixel) in image.enumerate_pixels() {
        alpha.put_pixel(x, y, Luma([pixel.0[3]]));
    }
    alpha
}

fn restore_alpha_channel(
    image: &RgbImage,
    original_alpha: &GrayImage,
    mask: &GrayImage,
) -> RgbaImage {
    let mut result = RgbaImage::new(image.width(), image.height());
    let mut alpha = original_alpha.clone();

    let mask_dilated = dilate(mask, Norm::LInf, ALPHA_RING_RADIUS);
    let mut surrounding_alpha = Vec::new();
    for (x, y, pixel) in mask_dilated.enumerate_pixels() {
        if pixel.0[0] > 0 && mask.get_pixel(x, y).0[0] == 0 {
            surrounding_alpha.push(original_alpha.get_pixel(x, y).0[0]);
        }
    }

    if let Some(median_alpha) = median_u8(&surrounding_alpha)
        && median_alpha < 128
    {
        for (x, y, pixel) in mask.enumerate_pixels() {
            if pixel.0[0] > 0 {
                alpha.put_pixel(x, y, Luma([median_alpha]));
            }
        }
    }

    for (x, y, pixel) in image.enumerate_pixels() {
        result.put_pixel(
            x,
            y,
            Rgba([
                pixel.0[0],
                pixel.0[1],
                pixel.0[2],
                alpha.get_pixel(x, y).0[0],
            ]),
        );
    }

    result
}

fn block_xyxy(block: &TextBlock, width: u32, height: u32) -> Option<Xyxy> {
    let x1 = block.x.floor().max(0.0) as u32;
    let y1 = block.y.floor().max(0.0) as u32;
    let x2 = (block.x + block.width).ceil().max(block.x.floor()) as u32;
    let y2 = (block.y + block.height).ceil().max(block.y.floor()) as u32;

    let x1 = x1.min(width);
    let y1 = y1.min(height);
    let x2 = x2.min(width);
    let y2 = y2.min(height);

    if x2 <= x1 || y2 <= y1 {
        return None;
    }

    Some([x1, y1, x2, y2])
}

fn enlarge_window(rect: Xyxy, im_w: u32, im_h: u32, ratio: f64, aspect_ratio: f64) -> Xyxy {
    debug_assert!(ratio > 1.0);

    let [x1, y1, x2, y2] = rect;
    let w = f64::from(x2.saturating_sub(x1));
    let h = f64::from(y2.saturating_sub(y1));
    if w <= 0.0 || h <= 0.0 || aspect_ratio <= 0.0 {
        return [0, 0, 0, 0];
    }

    let a = aspect_ratio;
    let b = w + h * aspect_ratio;
    let c = (1.0 - ratio) * w * h;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let delta = ((-b + discriminant.sqrt()) / (2.0 * a) / 2.0).round();
    let mut delta_h = delta.max(0.0) as u32;
    let mut delta_w = (delta * aspect_ratio).round().max(0.0) as u32;

    delta_w = delta_w.min(x1).min(im_w.saturating_sub(x2));
    delta_h = delta_h.min(y1).min(im_h.saturating_sub(y2));

    [
        x1.saturating_sub(delta_w),
        y1.saturating_sub(delta_h),
        (x2 + delta_w).min(im_w),
        (y2 + delta_h).min(im_h),
    ]
}

fn try_fill_balloon(image: &RgbImage, mask: &GrayImage) -> Option<RgbImage> {
    let masks = extract_balloon_mask(image, mask)?;
    let average_bg_color = median_rgb(image, &masks.non_text_mask)?;
    let std_rgb = color_stddev(image, &masks.non_text_mask, average_bg_color);
    let inpaint_thresh = if stddev3(std_rgb) > SIMPLE_BG_CHANNEL_STD_SWITCH {
        SIMPLE_BG_THRESHOLD_HIGH_VARIANCE
    } else {
        SIMPLE_BG_THRESHOLD_LOW_VARIANCE
    };
    let std_max = std_rgb.into_iter().fold(0.0, f64::max);

    if std_max >= inpaint_thresh {
        return None;
    }

    let mut result = image.clone();
    let fill = [
        average_bg_color[0] as u8,
        average_bg_color[1] as u8,
        average_bg_color[2] as u8,
    ];
    for (x, y, pixel) in masks.balloon_mask.enumerate_pixels() {
        if pixel.0[0] > 0 {
            result.put_pixel(x, y, Rgb(fill));
        }
    }

    Some(result)
}

fn extract_balloon_mask(image: &RgbImage, mask: &GrayImage) -> Option<BalloonMasks> {
    if image.dimensions() != mask.dimensions() {
        return None;
    }

    let text_bbox = non_zero_bbox(mask)?;
    let text_sum = count_nonzero(mask);
    if text_sum == 0 {
        return None;
    }

    let gray = DynamicImage::ImageRgb8(image.clone()).to_luma8();
    let blurred = gaussian_blur_f32(&gray, 1.0);
    let mut cannyed = canny(&blurred, BALLOON_CANNY_LOW, BALLOON_CANNY_HIGH);
    cannyed = dilate(&cannyed, Norm::LInf, 1);
    draw_binary_border(&mut cannyed);
    subtract_binary_mask(&mut cannyed, mask);

    let contours = find_contours::<i32>(&cannyed);
    let (width, height) = cannyed.dimensions();
    let mut best_mask = None;
    let mut best_area = f64::INFINITY;

    for contour in contours {
        let Some(polygon) = contour_polygon(&contour.points) else {
            continue;
        };
        let bbox = polygon_bbox(&polygon)?;
        if bbox[0] > text_bbox[0]
            || bbox[1] > text_bbox[1]
            || bbox[2] < text_bbox[2]
            || bbox[3] < text_bbox[3]
        {
            continue;
        }

        let mut candidate = GrayImage::new(width, height);
        draw_polygon_mut(&mut candidate, &polygon, Luma([255u8]));
        if count_overlap(&candidate, mask) < text_sum {
            continue;
        }

        let area = polygon_area(&polygon);
        if area < best_area {
            best_area = area;
            best_mask = Some(candidate);
        }
    }

    let balloon_mask = best_mask?;
    let mut non_text_mask = balloon_mask.clone();
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] > 0 {
            non_text_mask.put_pixel(x, y, Luma([0]));
        }
    }

    Some(BalloonMasks {
        balloon_mask,
        non_text_mask,
    })
}

fn contour_polygon(points: &[Point<i32>]) -> Option<Vec<Point<i32>>> {
    let mut polygon = points.to_vec();
    if polygon.len() < 3 {
        return None;
    }
    if polygon.first() == polygon.last() {
        polygon.pop();
    }
    if polygon.len() < 3 {
        return None;
    }
    Some(polygon)
}

fn polygon_bbox(points: &[Point<i32>]) -> Option<Xyxy> {
    let first = points.first()?;
    let mut min_x = first.x;
    let mut min_y = first.y;
    let mut max_x = first.x;
    let mut max_y = first.y;
    for point in points.iter().skip(1) {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }

    Some([
        min_x.max(0) as u32,
        min_y.max(0) as u32,
        max_x.max(min_x).saturating_add(1) as u32,
        max_y.max(min_y).saturating_add(1) as u32,
    ])
}

fn polygon_area(points: &[Point<i32>]) -> f64 {
    let mut area = 0.0;
    for index in 0..points.len() {
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        area += f64::from(current.x) * f64::from(next.y) - f64::from(next.x) * f64::from(current.y);
    }
    area.abs() * 0.5
}

fn draw_binary_border(image: &mut GrayImage) {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return;
    }

    for x in 0..width {
        image.put_pixel(x, 0, Luma([255]));
        image.put_pixel(x, height - 1, Luma([255]));
    }
    for y in 0..height {
        image.put_pixel(0, y, Luma([255]));
        image.put_pixel(width - 1, y, Luma([255]));
    }
}

fn subtract_binary_mask(image: &mut GrayImage, mask: &GrayImage) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if mask.get_pixel(x, y).0[0] > 0 {
            pixel.0[0] = 0;
        }
    }
}

fn non_zero_bbox(mask: &GrayImage) -> Option<Xyxy> {
    let (width, height) = mask.dimensions();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then_some([
        min_x,
        min_y,
        max_x.saturating_add(1),
        max_y.saturating_add(1),
    ])
}

fn clear_mask_bbox(mask: &mut GrayImage, bbox: Xyxy) {
    for y in bbox[1]..bbox[3] {
        for x in bbox[0]..bbox[2] {
            mask.put_pixel(x, y, Luma([0]));
        }
    }
}

fn count_nonzero(mask: &GrayImage) -> u32 {
    mask.pixels().filter(|pixel| pixel.0[0] > 0).count() as u32
}

fn count_overlap(left: &GrayImage, right: &GrayImage) -> u32 {
    left.pixels()
        .zip(right.pixels())
        .filter(|(l, r)| l.0[0] > 0 && r.0[0] > 0)
        .count() as u32
}

fn median_rgb(image: &RgbImage, mask: &GrayImage) -> Option<[f64; 3]> {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for (pixel, mask_pixel) in image.pixels().zip(mask.pixels()) {
        if mask_pixel.0[0] == 0 {
            continue;
        }
        channels[0].push(pixel.0[0]);
        channels[1].push(pixel.0[1]);
        channels[2].push(pixel.0[2]);
    }

    Some([
        median_channel(&channels[0])?,
        median_channel(&channels[1])?,
        median_channel(&channels[2])?,
    ])
}

fn median_channel(values: &[u8]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    let mut values = values.to_vec();
    values.sort_unstable();
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((f64::from(values[mid - 1]) + f64::from(values[mid])) / 2.0)
    } else {
        Some(f64::from(values[mid]))
    }
}

fn median_u8(values: &[u8]) -> Option<u8> {
    median_channel(values).map(|value| value as u8)
}

fn color_stddev(image: &RgbImage, mask: &GrayImage, median: [f64; 3]) -> [f64; 3] {
    let mut sum_sq = [0.0; 3];
    let mut count = 0.0;

    for (pixel, mask_pixel) in image.pixels().zip(mask.pixels()) {
        if mask_pixel.0[0] == 0 {
            continue;
        }
        count += 1.0;
        for channel in 0..3 {
            let diff = f64::from(pixel.0[channel]) - median[channel];
            sum_sq[channel] += diff * diff;
        }
    }

    if count == 0.0 {
        return [f64::INFINITY; 3];
    }

    [
        (sum_sq[0] / count).sqrt(),
        (sum_sq[1] / count).sqrt(),
        (sum_sq[2] / count).sqrt(),
    ]
}

/// Fraction of `mask` pixels within `tolerance` of `color` on every
/// channel — a plain majority vote, robust to a contaminated minority of
/// the ring in a way raw std-dev is not (one bright cluster is enough to
/// dominate a variance calculation, but not a per-pixel majority count).
fn inlier_fraction(image: &RgbImage, mask: &GrayImage, color: [f64; 3], tolerance: f64) -> f64 {
    let mut inliers = 0.0;
    let mut count = 0.0;
    for (pixel, mask_pixel) in image.pixels().zip(mask.pixels()) {
        if mask_pixel.0[0] == 0 {
            continue;
        }
        count += 1.0;
        let is_inlier = (0..3).all(|c| (f64::from(pixel.0[c]) - color[c]).abs() <= tolerance);
        if is_inlier {
            inliers += 1.0;
        }
    }
    if count == 0.0 { 0.0 } else { inliers / count }
}

fn stddev3(values: [f64; 3]) -> f64 {
    let mean = values.iter().sum::<f64>() / 3.0;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / 3.0;
    variance.sqrt()
}

/// What `plan_restoration` decided from a mask's local ring — kept separate
/// from actually running the model so the classification itself (the part
/// with real logic to get wrong) can be unit-tested without a loaded model.
#[derive(Debug)]
enum RestorationPlan {
    Flat([f64; 3]),
    Gradient(Plane),
    Complex,
}

/// From the ring of pixels immediately around `mask` (radius `ring_radius`),
/// decide whether the local background is flat, a simple gradient, or
/// complex enough (screentone, line art, speed lines) to need the neural
/// model — see `.claude/Redraw.md` §2.
fn plan_restoration(
    image: &RgbImage,
    mask: &GrayImage,
    exclude: &GrayImage,
    ring_radius: u8,
) -> RestorationPlan {
    let ring = ring_mask(mask, exclude, ring_radius);
    let Some(ring_color) = median_rgb(image, &ring) else {
        // No local context at all (mask reaches the crop's own edge) —
        // nothing to read a flat/gradient decision from.
        return RestorationPlan::Complex;
    };

    let std_rgb = color_stddev(image, &ring, ring_color);
    let std_max = std_rgb.into_iter().fold(0.0, f64::max);
    let inliers = inlier_fraction(image, &ring, ring_color, ADAPTIVE_FLAT_INLIER_TOLERANCE);
    let debug = std::env::var("KOHARU_DEBUG_FREE_TEXT").is_ok();
    if debug {
        tracing::info!(
            ?ring_color,
            ?std_rgb,
            std_max,
            inliers,
            ring_pixels = count_nonzero(&ring),
            "plan_restoration: ring stats"
        );
    }
    if std_max < ADAPTIVE_FLAT_STD_THRESHOLD {
        return RestorationPlan::Flat(ring_color);
    }

    let plane_fit = fit_plane(image, &ring);
    let residual = plane_fit
        .as_ref()
        .map(|plane| plane_residual(image, &ring, plane));
    if let (Some(plane), Some(residual)) = (plane_fit, residual)
        && residual < ADAPTIVE_GRADIENT_RESIDUAL_THRESHOLD
    {
        return RestorationPlan::Gradient(plane);
    }

    RestorationPlan::Complex
}

/// One column of ink within a merged no-container text block — see the
/// `COLUMN_*` constants' doc comment. Grouped for mask/redraw purposes
/// only; the parent `TextBlock` stays the single translation unit.
struct TextColumn {
    /// Page-space bbox of this column's own ink (+ small pad).
    bbox: Xyxy,
    /// This column's own tight ink mask, cropped to `bbox` — never a
    /// neighbouring column's ink, even where enlarged context windows end
    /// up overlapping.
    mask: GrayImage,
    /// Where `mask`'s (0, 0) sits in page space.
    mask_origin: (u32, u32),
    /// 0 = read first — manga's right-to-left column order.
    reading_order: usize,
    /// Typical glyph/stroke width measured from this block's own ink when a
    /// real split happened — `None` for a single-column result. A ring
    /// radius scaled from the *parent block's* own bbox (which can span
    /// several columns) is easily wider than the real gap between the
    /// columns that bbox turned out to hold, pulling a neighbour's still
    /// unprocessed ink into the read; this is the same reference the split
    /// decision itself used, so it never overshoots the gap it just found.
    typical_width_hint: Option<f32>,
}

/// Decompose one block's own ink into columns, using evidence beyond plain
/// whitespace: components close in X (allowing for real letter/line
/// spacing) merge into one column; a gap between them only counts as a
/// real column break when the strip there is not blank/balloon-light —
/// see `.claude/Redraw.md`. A block with no such internal separation comes
/// back as a single column spanning its own ink, so callers can always
/// iterate columns uniformly.
fn cluster_columns(image: &RgbImage, mask: &GrayImage, block_bbox: Xyxy) -> Vec<TextColumn> {
    let (im_w, im_h) = mask.dimensions();
    let left = block_bbox[0].saturating_sub(COLUMN_DETECT_MARGIN_PX);
    let top = block_bbox[1].saturating_sub(COLUMN_DETECT_MARGIN_PX);
    let right = (block_bbox[2] + COLUMN_DETECT_MARGIN_PX).min(im_w);
    let bottom = (block_bbox[3] + COLUMN_DETECT_MARGIN_PX).min(im_h);
    if right <= left || bottom <= top {
        return Vec::new();
    }
    let (w, h) = (right - left, bottom - top);
    let crop_mask = crop_imm(mask, left, top, w, h).to_image();
    let crop_image = crop_imm(image, left, top, w, h).to_image();

    // A vertical ink projection — how many ink pixels sit at each x, summed
    // over the block's own full height — finds column gaps directly from
    // what the page actually looks like. `connected_components` was tried
    // first and rejected: a few pixels of the mask's own anti-alias
    // dilation is enough to bridge two visually separate columns into one
    // 8-connected blob, which is exactly the shape that broke a single
    // whole-block inpaint call in the first place.
    let mut column_ink = vec![0u32; w as usize];
    for (x, _y, pixel) in crop_mask.enumerate_pixels() {
        if pixel.0[0] > 0 {
            column_ink[x as usize] += 1;
        }
    }
    let gap_tolerance = ((h as f32) * COLUMN_GAP_INK_DENSITY_THRESHOLD).round() as u32;

    let ink_runs = runs(&column_ink, |count| count > gap_tolerance);
    if ink_runs.len() < 2 {
        return single_column(crop_mask, left, top);
    }
    let mut widths: Vec<u32> = ink_runs.iter().map(|&(a, b)| b - a).collect();
    widths.sort_unstable();
    let typical_width = widths[widths.len() / 2].max(1) as f32;

    let mut boundaries: Vec<u32> = Vec::new();
    for (gx1, gx2) in runs(&column_ink, |count| count <= gap_tolerance) {
        if gx1 == 0 || gx2 == w {
            continue; // the crop's own edge, not an interior gap
        }
        let gap_width = (gx2 - gx1) as f32;
        if gap_width <= (typical_width * COLUMN_GAP_MIN_RATIO).max(COLUMN_GAP_MIN_PX) {
            continue;
        }
        if strip_is_light_rgb(&crop_image, [gx1 as f32, 0.0, gx2 as f32, h as f32]) {
            continue; // blank balloon padding — do not split on whitespace alone
        }
        boundaries.push((gx1 + gx2) / 2);
    }

    if boundaries.is_empty() {
        return single_column(crop_mask, left, top);
    }

    let mut edges = vec![0u32];
    edges.extend(boundaries);
    edges.push(w);

    let mut columns: Vec<TextColumn> = edges
        .windows(2)
        .filter_map(|edge| {
            let (x1, x2) = (edge[0], edge[1]);
            if x2 <= x1 {
                return None;
            }
            let mut y1 = h;
            let mut y2 = 0u32;
            for x in x1..x2 {
                for y in 0..h {
                    if crop_mask.get_pixel(x, y).0[0] > 0 {
                        y1 = y1.min(y);
                        y2 = y2.max(y + 1);
                    }
                }
            }
            if y2 <= y1 {
                return None; // no ink in this x-range at all
            }
            let px1 = x1.saturating_sub(COLUMN_PAD_PX as u32);
            let py1 = y1.saturating_sub(COLUMN_PAD_PX as u32);
            let px2 = (x2 + COLUMN_PAD_PX as u32).min(w);
            let py2 = (y2 + COLUMN_PAD_PX as u32).min(h);
            let column_mask = crop_imm(&crop_mask, px1, py1, px2 - px1, py2 - py1).to_image();
            if count_nonzero(&column_mask) == 0 {
                return None;
            }
            Some(TextColumn {
                bbox: [left + px1, top + py1, left + px2, top + py2],
                mask: column_mask,
                mask_origin: (left + px1, top + py1),
                reading_order: 0, // fixed below, after sorting
                typical_width_hint: Some(typical_width),
            })
        })
        .collect();

    // Manga reads right to left.
    columns.sort_by_key(|c| std::cmp::Reverse(c.bbox[0]));
    for (order, column) in columns.iter_mut().enumerate() {
        column.reading_order = order;
    }
    columns
}

/// Contiguous index ranges `[start, end)` where `pred` holds — the runs of
/// "has ink" and "is a gap" in a column's ink-density profile.
fn runs(values: &[u32], pred: impl Fn(u32) -> bool) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    let mut start: Option<u32> = None;
    for (i, &value) in values.iter().enumerate() {
        if pred(value) {
            start.get_or_insert(i as u32);
        } else if let Some(s) = start.take() {
            result.push((s, i as u32));
        }
    }
    if let Some(s) = start {
        result.push((s, values.len() as u32));
    }
    result
}

/// The whole crop as one column — used whenever nothing in it justifies a
/// split, so callers can always iterate columns uniformly.
fn single_column(crop_mask: GrayImage, left: u32, top: u32) -> Vec<TextColumn> {
    if count_nonzero(&crop_mask) == 0 {
        return Vec::new();
    }
    let (w, h) = crop_mask.dimensions();
    vec![TextColumn {
        bbox: [left, top, left + w, top + h],
        mask: crop_mask,
        mask_origin: (left, top),
        reading_order: 0,
        typical_width_hint: None,
    }]
}

/// Is the strip between two same-block ink components blank/balloon-light
/// (uniformly light) rather than real artwork? A dark or busy strip is
/// never balloon interior, so the components either side of it are safe to
/// treat as separate columns.
fn strip_is_light_rgb(image: &RgbImage, rect: [f32; 4]) -> bool {
    let (width, height) = image.dimensions();
    let x = (rect[0].floor().max(0.0) as u32).min(width);
    let y = (rect[1].floor().max(0.0) as u32).min(height);
    let x2 = (rect[2].ceil().max(rect[0]) as u32).min(width);
    let y2 = (rect[3].ceil().max(rect[1]) as u32).min(height);
    if x2 <= x || y2 <= y {
        return true; // no real gap to sample — do not force a split on it
    }

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut n = 0.0;
    for yy in y..y2 {
        for xx in x..x2 {
            let p = image.get_pixel(xx, yy).0;
            let luma = 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
            sum += luma;
            sum_sq += luma * luma;
            n += 1.0;
        }
    }
    if n == 0.0 {
        return true;
    }
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    mean >= COLUMN_GAP_LIGHT_MEAN_THRESHOLD && variance.sqrt() <= COLUMN_GAP_LIGHT_STD_THRESHOLD
}

/// When `KOHARU_DEBUG_ADAPTIVE` is set, save `debug-columns/<block_id>.png`
/// (blue = the parent logical block's own bbox, one colour per column in
/// reading order) and a `.json` with the parent bbox, each column's bbox/
/// reading order/mask pixel count, and the union mask pixel count — see
/// `.claude/Redraw.md`'s debug requirements.
fn write_column_debug(image: &RgbImage, block_id: &str, block_bbox: Xyxy, columns: &[TextColumn]) {
    let dir = std::path::PathBuf::from("debug-columns");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("debug-columns: cannot create dir {:?}: {e}", dir);
        return;
    }

    let (im_w, im_h) = image.dimensions();
    let margin = 20u32;
    let left = block_bbox[0].saturating_sub(margin);
    let top = block_bbox[1].saturating_sub(margin);
    let right = (block_bbox[2] + margin).min(im_w);
    let bottom = (block_bbox[3] + margin).min(im_h);
    if right <= left || bottom <= top {
        return;
    }
    let mut overview = crop_imm(image, left, top, right - left, bottom - top).to_image();

    draw_hollow_rect_mut(
        &mut overview,
        Rect::at((block_bbox[0] - left) as i32, (block_bbox[1] - top) as i32).of_size(
            (block_bbox[2] - block_bbox[0]).max(1),
            (block_bbox[3] - block_bbox[1]).max(1),
        ),
        Rgb([0, 100, 255]),
    );

    const PALETTE: [Rgb<u8>; 6] = [
        Rgb([255, 60, 60]),
        Rgb([255, 165, 0]),
        Rgb([220, 220, 0]),
        Rgb([0, 200, 60]),
        Rgb([0, 180, 255]),
        Rgb([190, 0, 255]),
    ];
    for column in columns {
        let color = PALETTE[column.reading_order % PALETTE.len()];
        let (bx, by) = (column.bbox[0] - left, column.bbox[1] - top);
        let (bw, bh) = (
            column.bbox[2] - column.bbox[0],
            column.bbox[3] - column.bbox[1],
        );
        draw_hollow_rect_mut(
            &mut overview,
            Rect::at(bx as i32, by as i32).of_size(bw.max(1), bh.max(1)),
            color,
        );
    }

    let label = if block_id.trim().is_empty() {
        format!("{}_{}", block_bbox[0], block_bbox[1])
    } else {
        block_id.to_string()
    };
    let png_path = dir.join(format!("{label}.png"));
    if let Err(e) = overview.save(&png_path) {
        tracing::warn!("debug-columns: cannot save {:?}: {e}", png_path);
    }

    let union_mask_pixels: u32 = columns.iter().map(|c| count_nonzero(&c.mask)).sum();
    let json = serde_json::json!({
        "parentBbox": block_bbox,
        "columns": columns.iter().map(|c| serde_json::json!({
            "bbox": c.bbox,
            "readingOrder": c.reading_order,
            "maskPixels": count_nonzero(&c.mask),
        })).collect::<Vec<_>>(),
        "unionMaskPixels": union_mask_pixels,
    });
    let json_path = dir.join(format!("{label}.json"));
    match serde_json::to_string_pretty(&json) {
        Ok(text) => {
            if let Err(e) = std::fs::write(&json_path, text) {
                tracing::warn!("debug-columns: cannot write {:?}: {e}", json_path);
            }
        }
        Err(e) => tracing::warn!("debug-columns: cannot serialize columns: {e}"),
    }
}

/// Pixels within `radius` of `mask` but not part of it — the *immediate*
/// local background, as opposed to `extract_balloon_mask`'s Canny-traced
/// contour (which assumes a real balloon exists to trace, the wrong tool
/// for free-standing text with no container at all).
/// `exclude` marks ink that must never be sampled as background even though
/// it is not `mask` itself — a neighbouring column of the same block,
/// still showing its original bright ink because this block's own columns
/// are processed one at a time. Without this, one column's ring can reach
/// across a real but narrow gap and read a neighbour's still-unprocessed
/// ink as "background", which is exactly what turned a flat dark panel
/// into a false "Complex" read on a real page this session hit.
fn ring_mask(mask: &GrayImage, exclude: &GrayImage, radius: u8) -> GrayImage {
    let dilated = dilate(mask, Norm::LInf, radius);
    let (width, height) = mask.dimensions();
    let mut ring = GrayImage::new(width, height);
    for (x, y, pixel) in dilated.enumerate_pixels() {
        if pixel.0[0] > 0 && mask.get_pixel(x, y).0[0] == 0 && exclude.get_pixel(x, y).0[0] == 0 {
            ring.put_pixel(x, y, Luma([255]));
        }
    }
    ring
}

fn flat_fill(image: &RgbImage, mask: &GrayImage, color: [f64; 3]) -> RgbImage {
    let fill = [color[0] as u8, color[1] as u8, color[2] as u8];
    let mut result = image.clone();
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] > 0 {
            result.put_pixel(x, y, Rgb(fill));
        }
    }
    // Scanned solid ink still has slow colour variation. A single median per
    // glyph exposes the mask silhouette; harmonic interpolation follows the
    // surrounding scan while changing only authorized pixels.
    let (w, h) = image.dimensions();
    let mut values: Vec<[f32; 3]> = result.pixels().map(|p| p.0.map(f32::from)).collect();
    let points: Vec<usize> = mask
        .enumerate_pixels()
        .filter(|(x, y, p)| p[0] != 0 && *x > 0 && *y > 0 && *x + 1 < w && *y + 1 < h)
        .map(|(x, y, _)| (y * w + x) as usize)
        .collect();
    let stride = w as usize;
    for _ in 0..160 {
        for &i in &points {
            for c in 0..3 {
                values[i][c] = (values[i - 1][c]
                    + values[i + 1][c]
                    + values[i - stride][c]
                    + values[i + stride][c])
                    * 0.25;
            }
        }
    }
    for &i in &points {
        result.put_pixel(
            (i % stride) as u32,
            (i / stride) as u32,
            Rgb(values[i].map(|v| v.round() as u8)),
        );
    }
    result
}

/// `color = a·x + b·y + c`, one triple of coefficients per RGB channel —
/// enough to describe a simple lighting gradient across a small crop.
#[derive(Debug)]
struct Plane {
    coeffs: [[f64; 3]; 3],
}

/// Least-squares fit of `plane` to the pixels marked in `mask`. `None` when
/// there are too few points to trust a fit, or the point layout is
/// degenerate (all on one line, so the system has no unique solution).
fn fit_plane(image: &RgbImage, mask: &GrayImage) -> Option<Plane> {
    let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy, mut sum_yy, mut n) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut sum_v = [0.0; 3];
    let mut sum_xv = [0.0; 3];
    let mut sum_yv = [0.0; 3];

    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] == 0 {
            continue;
        }
        let (fx, fy) = (f64::from(x), f64::from(y));
        n += 1.0;
        sum_x += fx;
        sum_y += fy;
        sum_xx += fx * fx;
        sum_xy += fx * fy;
        sum_yy += fy * fy;
        let rgb = image.get_pixel(x, y).0;
        for channel in 0..3 {
            let v = f64::from(rgb[channel]);
            sum_v[channel] += v;
            sum_xv[channel] += fx * v;
            sum_yv[channel] += fy * v;
        }
    }

    if n < ADAPTIVE_MIN_RING_SAMPLES as f64 {
        return None;
    }

    let m = [
        [sum_xx, sum_xy, sum_x],
        [sum_xy, sum_yy, sum_y],
        [sum_x, sum_y, n],
    ];
    let mut coeffs = [[0.0; 3]; 3];
    for channel in 0..3 {
        let rhs = [sum_xv[channel], sum_yv[channel], sum_v[channel]];
        coeffs[channel] = solve_3x3(m, rhs)?;
    }
    Some(Plane { coeffs })
}

/// Gaussian elimination with partial pivoting for a 3×3 system — small and
/// self-contained rather than pulling in a linear-algebra crate for one
/// solve per channel.
fn solve_3x3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let mut a = matrix;
    let mut b = rhs;

    for col in 0..3 {
        let pivot_row =
            (col..3).max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())?;
        if a[pivot_row][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        for row in (col + 1)..3 {
            let factor = a[row][col] / a[col][col];
            for k in col..3 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let mut sum = b[row];
        for k in (row + 1)..3 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

fn plane_eval(plane: &Plane, x: u32, y: u32) -> [u8; 3] {
    let (fx, fy) = (f64::from(x), f64::from(y));
    let mut out = [0u8; 3];
    for channel in 0..3 {
        let [a, b, c] = plane.coeffs[channel];
        out[channel] = (a * fx + b * fy + c).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// RMS error between the fitted plane and the actual ring pixels — how well
/// "a simple gradient" explains what is really there.
fn plane_residual(image: &RgbImage, mask: &GrayImage, plane: &Plane) -> f64 {
    let mut sum_sq = 0.0;
    let mut n = 0.0;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] == 0 {
            continue;
        }
        let actual = image.get_pixel(x, y).0;
        let predicted = plane_eval(plane, x, y);
        for channel in 0..3 {
            let diff = f64::from(actual[channel]) - f64::from(predicted[channel]);
            sum_sq += diff * diff;
        }
        n += 3.0;
    }
    if n == 0.0 { 0.0 } else { (sum_sq / n).sqrt() }
}

fn gradient_fill(image: &RgbImage, mask: &GrayImage, plane: &Plane) -> RgbImage {
    let mut result = image.clone();
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] > 0 {
            result.put_pixel(x, y, Rgb(plane_eval(plane, x, y)));
        }
    }
    result
}

/// Paste `src` onto `dest` at `(offset_x, offset_y)` only where `mask` is
/// set — the discipline the user's spec calls out by name: never let a
/// restored crop overwrite artwork its own mask never claimed.
fn composite_masked(
    dest: &mut RgbImage,
    src: &RgbImage,
    mask: &GrayImage,
    offset_x: u32,
    offset_y: u32,
) {
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] > 0 {
            dest.put_pixel(offset_x + x, offset_y + y, *src.get_pixel(x, y));
        }
    }
}

/// Drop all cached MPSGraph objects on the calling thread to release compiled
/// shader temp file FDs and free their on-disk space immediately.
/// Must be called from the same thread that ran Lama inference.
pub fn clear_fft_plans_on_current_thread() {
    fft::clear_fft_plans_on_current_thread();
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_majority_white_ring_with_speed_lines_is_not_flat() {
        use super::*;
        let image = RgbImage::from_fn(64, 64, |x, _| {
            if x % 10 < 2 {
                Rgb([30; 3])
            } else {
                Rgb([255; 3])
            }
        });
        let mask = centered_mask(64, (26, 26, 38, 38));
        assert!(matches!(
            plan_restoration(&image, &mask, &GrayImage::new(64, 64), 8),
            RestorationPlan::Complex
        ));
    }

    #[test]
    fn outlined_text_recovers_stroke_without_erasing_the_block_rectangle() {
        use super::*;
        let mut image = RgbImage::from_pixel(120, 120, Rgb([50; 3]));
        let mut seed = GrayImage::new(120, 120);
        for y in 30..65 {
            for x in 30..50 {
                image.put_pixel(x, y, Rgb([255; 3]));
                if x >= 35 && x < 45 && y >= 35 && y < 60 {
                    image.put_pixel(x, y, Rgb([20; 3]));
                    seed.put_pixel(x, y, Luma([255]));
                }
            }
        }
        let block = TextBlock {
            x: 20.0,
            y: 20.0,
            width: 80.0,
            height: 80.0,
            detected_font_size_px: Some(40.0),
            ..Default::default()
        };
        let (source, removal) = adaptive_text_masks(&image, &seed, &[block]);
        assert_eq!(source.get_pixel(30, 30)[0], 255);
        assert_eq!(removal.get_pixel(29, 30)[0], 255);
        assert_eq!(
            removal.get_pixel(80, 50)[0],
            0,
            "inter-column artwork is not a removal region"
        );
        assert!(
            source
                .pixels()
                .zip(removal.pixels())
                .all(|(s, r)| s[0] == 0 || r[0] != 0)
        );
    }

    #[test]
    fn reported_em_dash_is_recovered_but_unrelated_line_is_kept() {
        use super::*;
        let mut image = RgbImage::from_pixel(160, 160, Rgb([255; 3]));
        let mut seed = GrayImage::new(160, 160);
        for y in 20..40 {
            for x in 40..55 {
                seed.put_pixel(x, y, Luma([255]));
            }
        }
        for y in 48..105 {
            for x in [47, 48, 100, 101] {
                image.put_pixel(x, y, Rgb([30; 3]));
            }
        }
        let block = TextBlock {
            x: 20.0,
            y: 15.0,
            width: 110.0,
            height: 115.0,
            detected_font_size_px: Some(30.0),
            text: Some("了——".into()),
            source_direction: Some(koharu_types::TextDirection::Vertical),
            ..Default::default()
        };
        let (source, _) = adaptive_text_masks(&image, &seed, &[block]);
        assert_eq!(source.get_pixel(47, 90)[0], 255);
        assert_eq!(source.get_pixel(100, 90)[0], 0);
    }

    #[test]
    fn local_flat_reconstruction_preserves_unmasked_scan_pixels() {
        use super::*;
        let image = RgbImage::from_fn(40, 40, |x, y| Rgb([(40 + (x + y) / 10) as u8; 3]));
        let mask = centered_mask(40, (10, 10, 30, 30));
        let filled = flat_fill(&image, &mask, [44.0; 3]);
        for (x, y, p) in image.enumerate_pixels() {
            if mask.get_pixel(x, y)[0] == 0 {
                assert_eq!(filled.get_pixel(x, y), p);
            }
        }
        assert!(filled.get_pixel(11, 11)[0] < filled.get_pixel(28, 28)[0]);
    }
    #[test]
    fn leftover_regions_group_a_run_of_glyphs_into_one_box() {
        // Four glyph blobs on one line, the shape a run of text leaves in the
        // mask after the block pass has cleared what it covered. Full page
        // size: a line is a couple of percent of a real page, which is what the
        // area guard is calibrated against.
        let mask = GrayImage::from_fn(1200, 900, |x, y| {
            let on = (40..70).contains(&y)
                && [(20, 40), (55, 75), (90, 110), (125, 145)]
                    .iter()
                    .any(|(a, b)| x >= *a && x < *b);
            Luma([if on { 255u8 } else { 0 }])
        });

        // A block window sitting over the head of the run: the tail spilling
        // out of it is exactly what this pass is for.
        let window = [0, 30, 60, 80];

        let regions = super::leftover_mask_regions(&mask, &[window]);
        assert_eq!(
            regions.len(),
            1,
            "one run should be one region: {regions:?}"
        );
        let [x1, _, x2, _] = regions[0];
        assert!(
            x1 <= 20 && x2 >= 145,
            "region must span the run: {regions:?}"
        );
    }

    #[test]
    fn leftover_regions_leave_marks_no_block_window_reaches() {
        // A sound effect drawn across the artwork, far from every balloon. The
        // mask marks it, but nothing was erased around it, so there is no
        // half-done job to finish and inpainting it would only damage the art.
        let mask = GrayImage::from_fn(1200, 900, |x, y| {
            let on = (700..760).contains(&y) && (60..140).contains(&x);
            Luma([if on { 255u8 } else { 0 }])
        });

        assert!(
            super::leftover_mask_regions(&mask, &[[400, 100, 600, 300]]).is_empty(),
            "a mark out of every window's reach must be left alone"
        );
        assert!(
            !super::leftover_mask_regions(&mask, &[[60, 600, 140, 695]]).is_empty(),
            "the same mark next to a window is a run to finish"
        );
    }

    #[test]
    fn leftover_regions_ignore_an_empty_mask_and_specks() {
        let window = [[0, 0, 100, 100]];
        assert!(super::leftover_mask_regions(&GrayImage::new(100, 100), &window).is_empty());
        let speck = GrayImage::from_fn(100, 100, |x, y| {
            Luma([if x == 50 && y == 50 { 255u8 } else { 0 }])
        });
        assert!(super::leftover_mask_regions(&speck, &window).is_empty());
    }

    use super::{
        ALPHA_RING_RADIUS, BALLOON_WINDOW_ASPECT_RATIO, BALLOON_WINDOW_RATIO, RestorationPlan,
        clear_mask_bbox, cluster_columns, composite_masked, count_nonzero, enlarge_window,
        extract_balloon_mask, gradient_fill, plan_restoration, restore_alpha_channel,
        try_fill_balloon,
    };
    use image::{GrayImage, Luma, Rgb, RgbImage};
    use imageproc::drawing::draw_hollow_rect_mut;
    use imageproc::rect::Rect;
    use koharu_types::TextBlock;

    #[test]
    fn enlarge_window_matches_ratio_1_7_reference() {
        let enlarged = enlarge_window(
            [10, 20, 50, 60],
            200,
            150,
            BALLOON_WINDOW_RATIO,
            BALLOON_WINDOW_ASPECT_RATIO,
        );

        assert_eq!(enlarged, [4, 14, 56, 66]);
    }

    #[test]
    fn extract_balloon_mask_prefers_smallest_covering_contour() {
        let mut image = RgbImage::from_pixel(80, 80, Rgb([255, 255, 255]));
        draw_hollow_rect_mut(&mut image, Rect::at(4, 4).of_size(72, 72), Rgb([0, 0, 0]));
        draw_hollow_rect_mut(&mut image, Rect::at(20, 20).of_size(28, 20), Rgb([0, 0, 0]));

        let mut mask = GrayImage::new(80, 80);
        for y in 24..36 {
            for x in 24..44 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }

        let masks = extract_balloon_mask(&image, &mask).expect("balloon should be detected");
        let balloon_pixels = count_nonzero(&masks.balloon_mask);

        assert!(
            balloon_pixels < 900,
            "expected inner contour fill, got {balloon_pixels}"
        );
        assert!(
            balloon_pixels > 250,
            "expected meaningful bubble area, got {balloon_pixels}"
        );
    }

    #[test]
    fn simple_balloon_chooses_fill_but_textured_balloon_does_not() {
        let mut flat = RgbImage::from_pixel(64, 64, Rgb([240, 240, 240]));
        draw_hollow_rect_mut(&mut flat, Rect::at(8, 8).of_size(48, 32), Rgb([0, 0, 0]));

        let mut mask = GrayImage::new(64, 64);
        for y in 18..30 {
            for x in 18..46 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }

        assert!(try_fill_balloon(&flat, &mask).is_some());

        let mut textured = flat.clone();
        for y in 9..39 {
            for x in 9..55 {
                let noise = ((x + y) % 23) as u8;
                textured.put_pixel(
                    x,
                    y,
                    Rgb([200 + noise, 210 + (noise / 2), 220 - (noise / 3)]),
                );
            }
        }

        assert!(try_fill_balloon(&textured, &mask).is_none());
    }

    fn centered_mask(size: u32, hole: (u32, u32, u32, u32)) -> GrayImage {
        let (x0, y0, x1, y1) = hole;
        GrayImage::from_fn(size, size, |x, y| {
            Luma([if x >= x0 && x < x1 && y >= y0 && y < y1 {
                255
            } else {
                0
            }])
        })
    }

    #[test]
    fn a_neighbours_unprocessed_ink_in_the_ring_is_excluded_not_sampled() {
        // A flat dark background everywhere, except a bright patch just
        // outside the hole — standing in for a neighbouring column's own
        // ink, still unrestored because this block's columns are done one
        // at a time. Without `exclude` this reads as a busy/complex ring;
        // marking that patch excluded must recover the flat read.
        // The ring for a (26,26)-(38,38) hole dilated by 4 is the frame
        // (22,22)-(42,42) minus that hole — cover all of it except a clean
        // strip along the bottom (roughly 70% of the ring, well past what
        // a majority vote can absorb) with a checkerboard, not a solid
        // colour, so it cannot be explained away as a gradient either.
        let mut image = RgbImage::from_pixel(64, 64, Rgb([22, 22, 22]));
        for y in 22..38 {
            for x in 22..42 {
                // Both values sit outside `ADAPTIVE_FLAT_INLIER_TOLERANCE`
                // of the real background (22) — a checkerboard where one
                // side happened to land within tolerance would let half of
                // "contamination" quietly count as an inlier anyway.
                let v = if (x / 2 + y / 2) % 2 == 0 {
                    100u8
                } else {
                    230u8
                };
                image.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let mask = centered_mask(64, (26, 26, 38, 38));

        let no_exclude = GrayImage::new(64, 64);
        assert!(
            matches!(
                plan_restoration(&image, &mask, &no_exclude, 4),
                RestorationPlan::Complex
            ),
            "the bright neighbour patch should contaminate an unfiltered ring"
        );

        let mut exclude = GrayImage::new(64, 64);
        for y in 22..38 {
            for x in 22..42 {
                exclude.put_pixel(x, y, Luma([255]));
            }
        }
        match plan_restoration(&image, &mask, &exclude, 4) {
            RestorationPlan::Flat(color) => {
                for channel in color {
                    assert!((channel - 22.0).abs() < 1.0, "{color:?}");
                }
            }
            other => panic!("expected Flat once the neighbour's ink is excluded, got {other:?}"),
        }
    }

    #[test]
    fn a_flat_dark_ring_is_planned_as_a_flat_fill() {
        let image = RgbImage::from_pixel(64, 64, Rgb([22, 22, 22]));
        let mask = centered_mask(64, (26, 26, 38, 38));
        let no_exclude = GrayImage::new(64, 64);

        let plan = plan_restoration(&image, &mask, &no_exclude, 4);

        match plan {
            RestorationPlan::Flat(color) => {
                for channel in color {
                    assert!((channel - 22.0).abs() < 1.0, "{color:?}");
                }
            }
            other => panic!("expected Flat, got {other:?}"),
        }
    }

    #[test]
    fn a_linear_gradient_ring_is_planned_as_a_gradient_and_interpolates() {
        // color(x, y) = 40 + 6*(x - 22) over the ring's own x-span
        // (22..42) — steep enough that roughly half the ring falls outside
        // `ADAPTIVE_FLAT_INLIER_TOLERANCE` of the median (clearing both the
        // std-dev check and the majority-vote fallback — a real gradient
        // must beat both for this to be a meaningful test), but still low
        // enough that the ring never saturates at 255 (which would break
        // the plane's linearity assumption), so a plane explains it almost
        // exactly.
        let image = RgbImage::from_fn(64, 64, |x, _y| {
            let v = (40 + 6 * (x as i32 - 22)).clamp(0, 255) as u8;
            Rgb([v, v, v])
        });
        let mask = centered_mask(64, (26, 26, 38, 38));
        let no_exclude = GrayImage::new(64, 64);

        let plan = plan_restoration(&image, &mask, &no_exclude, 4);
        let RestorationPlan::Gradient(plane) = plan else {
            panic!("expected Gradient, got {plan:?}");
        };

        let filled = gradient_fill(&image, &mask, &plane);
        let expected = (40 + 6 * (30 - 22)).clamp(0, 255); // pixel (30, 30) sits inside the hole
        let actual = filled.get_pixel(30, 30).0[0] as i32;
        assert!(
            (actual - expected).abs() <= 2,
            "expected ~{expected}, got {actual}"
        );
    }

    #[test]
    fn a_noisy_ring_falls_back_to_the_model_path() {
        // Screentone-like high-frequency noise, no flat colour and no plane
        // fits it — must not be mistaken for a flat or gradient background.
        let image = RgbImage::from_fn(64, 64, |x, y| {
            let v = if (x / 2 + y / 2) % 2 == 0 {
                30u8
            } else {
                220u8
            };
            Rgb([v, v, v])
        });
        let mask = centered_mask(64, (26, 26, 38, 38));
        let no_exclude = GrayImage::new(64, 64);

        assert!(matches!(
            plan_restoration(&image, &mask, &no_exclude, 4),
            RestorationPlan::Complex
        ));
    }

    #[test]
    fn compositing_never_touches_a_pixel_outside_the_mask() {
        let mut dest = RgbImage::from_pixel(32, 32, Rgb([1, 2, 3]));
        let src = RgbImage::from_pixel(32, 32, Rgb([9, 9, 9]));
        let mask = centered_mask(32, (10, 10, 20, 20));

        composite_masked(&mut dest, &src, &mask, 0, 0);

        assert_eq!(
            dest.get_pixel(0, 0).0,
            [1, 2, 3],
            "outside the mask must be untouched"
        );
        assert_eq!(
            dest.get_pixel(15, 15).0,
            [9, 9, 9],
            "inside the mask must be replaced"
        );
    }

    /// Four narrow ink columns spaced across a shared block bbox — the
    /// shape a layout detector's own input resolution can squeeze several
    /// separate vertical captions into one box as (see
    /// `.claude/Redraw.md`'s worked example).
    fn four_columns_page(background: Rgb<u8>) -> (RgbImage, GrayImage, [u32; 4]) {
        const SIZE: u32 = 400;
        const TOP: u32 = 60;
        const BOTTOM: u32 = 300;
        let offsets = [40u32, 100, 160, 220];
        let column_width = 20u32;

        let mut image = RgbImage::from_pixel(SIZE, SIZE, background);
        let mut mask = GrayImage::new(SIZE, SIZE);
        for &ox in &offsets {
            for y in TOP..BOTTOM {
                for x in ox..ox + column_width {
                    image.put_pixel(x, y, Rgb([250, 250, 250]));
                    mask.put_pixel(x, y, Luma([255]));
                }
            }
        }
        (image, mask, [30, TOP - 10, 250, BOTTOM + 10])
    }

    #[test]
    fn four_columns_on_dark_art_are_clustered_separately_in_reading_order() {
        let (image, mask, block_bbox) = four_columns_page(Rgb([20, 20, 20]));

        let columns = cluster_columns(&image, &mask, block_bbox);

        assert_eq!(
            columns.len(),
            4,
            "{:?}",
            columns.iter().map(|c| c.bbox).collect::<Vec<_>>()
        );
        // Manga reads right to left: reading_order 0 is the rightmost column.
        for pair in columns.windows(2) {
            assert!(
                pair[0].bbox[0] > pair[1].bbox[0],
                "columns must be in descending-x (right-to-left) order: {:?}",
                columns
                    .iter()
                    .map(|c| (c.reading_order, c.bbox))
                    .collect::<Vec<_>>()
            );
            assert!(pair[1].reading_order == pair[0].reading_order + 1);
        }
        // Each column's own mask holds only its own ink, never a
        // neighbour's — the crop can span part of the gap either side, but
        // the ink inside it must form exactly one contiguous run.
        for column in &columns {
            let (w, h) = column.mask.dimensions();
            let mut ink_columns = 0u32;
            for x in 0..w {
                if (0..h).any(|y| column.mask.get_pixel(x, y).0[0] > 0) {
                    ink_columns += 1;
                }
            }
            assert!(
                (15..=25).contains(&ink_columns),
                "expected ~20px of ink (one column), got {ink_columns} in {:?}",
                column.bbox
            );
        }
    }

    #[test]
    fn a_wide_but_light_gap_between_columns_is_not_split() {
        // Same shape as the dark-art case, but the gap between columns is
        // blank/balloon-light — must merge into one column, not four,
        // per "do not split solely based on whitespace".
        let (image, mask, block_bbox) = four_columns_page(Rgb([245, 245, 245]));

        let columns = cluster_columns(&image, &mask, block_bbox);

        assert_eq!(
            columns.len(),
            1,
            "{:?}",
            columns.iter().map(|c| c.bbox).collect::<Vec<_>>()
        );
    }

    #[test]
    fn clearing_mask_consumes_only_original_bbox() {
        let mut mask = GrayImage::from_pixel(32, 32, Luma([255]));
        clear_mask_bbox(&mut mask, [8, 10, 16, 18]);

        for y in 10..18 {
            for x in 8..16 {
                assert_eq!(mask.get_pixel(x, y).0[0], 0);
            }
        }

        assert_eq!(mask.get_pixel(7, 10).0[0], 255);
        assert_eq!(mask.get_pixel(16, 17).0[0], 255);
        assert_eq!(mask.get_pixel(8, 9).0[0], 255);
        assert_eq!(mask.get_pixel(15, 18).0[0], 255);
    }

    #[test]
    fn rgba_alpha_restore_uses_surrounding_ring() {
        let image = RgbImage::from_pixel(32, 32, Rgb([20, 30, 40]));
        let mut alpha = GrayImage::from_pixel(32, 32, Luma([255]));
        let mut mask = GrayImage::new(32, 32);

        for y in 10..22 {
            for x in 10..22 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        for y in (10 - u32::from(ALPHA_RING_RADIUS))..(22 + u32::from(ALPHA_RING_RADIUS)) {
            for x in (10 - u32::from(ALPHA_RING_RADIUS))..(22 + u32::from(ALPHA_RING_RADIUS)) {
                if x < 32 && y < 32 && mask.get_pixel(x, y).0[0] == 0 {
                    alpha.put_pixel(x, y, Luma([64]));
                }
            }
        }

        let restored = restore_alpha_channel(&image, &alpha, &mask);
        assert_eq!(restored.get_pixel(15, 15).0[3], 64);
        assert_eq!(restored.get_pixel(2, 2).0[3], 255);
    }

    #[test]
    fn block_xyxy_rounds_and_clamps_document_coords() {
        let block = TextBlock {
            x: 10.2,
            y: 20.7,
            width: 15.1,
            height: 9.4,
            ..Default::default()
        };

        let bbox = super::block_xyxy(&block, 100, 100).expect("bbox");
        assert_eq!(bbox, [10, 20, 26, 31]);
    }
}
