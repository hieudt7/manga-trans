use std::sync::{Arc, Mutex};

use anyhow::Result;
use image::{DynamicImage, GrayImage, imageops};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use koharu_types::{
    Document, FontFaceInfo, SerializableDynamicImage, TextAlign, TextBlock, TextShaderEffect,
    TextStrokeStyle, TextStyle,
};

use crate::{
    font::{FaceInfo, Font, FontBook},
    layout::{LayoutRun, RowSpans, TextLayout, WritingMode},
    renderer::{RenderOptions, RenderStrokeOptions, TinySkiaRenderer},
    text::{
        latin::{
            LayoutBox, MIN_FILL_RATIO, balloon_bounds_from_image, clear_space_rows,
            clip_box_to_nearest_owner, fill_ratio, grow_box_within, is_emphatic_lettering,
            SHAPE_GROWTH_MAX_STEPS, SHAPE_GROWTH_STEP_PX, grow_rows_into_blank,
            is_stackable_shout, shorten_elongation, source_glyph_size, source_text_rows,
            expand_latin_layout_box_relaxed, expand_latin_layout_box_strict,
            is_expanded_layout_box, latin_layout_underfilled, latin_width_overflow_factor,
            layout_box_area, layout_box_from_block, pick_better_latin_candidate,
            preferred_font_size,
        },
        script::{
            font_families_for_text, is_latin_only, normalize_translation_for_layout,
            writing_mode_for_block,
        },
    },
};

pub struct Renderer {
    fontbook: Arc<Mutex<FontBook>>,
    renderer: TinySkiaRenderer,
    symbol_fallbacks: Vec<Font>,
}

impl Renderer {
    pub fn new() -> Result<Self> {
        let mut fontbook = FontBook::new();
        let symbol_fallbacks = load_symbol_fallbacks(&mut fontbook);
        Ok(Self {
            fontbook: Arc::new(Mutex::new(fontbook)),
            renderer: TinySkiaRenderer::new()?,
            symbol_fallbacks,
        })
    }

    pub fn available_fonts(&self) -> Result<Vec<FontFaceInfo>> {
        let fontbook = self
            .fontbook
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock fontbook"))?;
        let mut fonts = fontbook
            .all_families()
            .into_iter()
            .filter(|face| !face.post_script_name.is_empty())
            .map(|face| FontFaceInfo {
                family_name: face
                    .families
                    .first()
                    .map(|(family, _)| family.clone())
                    .unwrap_or_else(|| face.post_script_name.clone()),
                post_script_name: face.post_script_name,
            })
            .collect::<Vec<_>>();
        fonts.sort();
        Ok(fonts)
    }

    pub fn render(
        &self,
        document: &mut Document,
        text_block_index: Option<usize>,
        effect: TextShaderEffect,
        stroke: Option<TextStrokeStyle>,
        font_family: Option<&str>,
    ) -> Result<()> {
        // Greyscale page used to find how much clear space a balloon actually
        // offers. The inpainted page is preferred: the source text is gone from
        // it, so the balloon interior reads as one blank region.
        let bubble_map = if let Some(inpainted) = &document.inpainted {
            inpainted.to_luma8()
        } else {
            document.image.to_luma8()
        };

        // Where the source text was before it was painted out. Outside a
        // balloon that footprint is the only room the translation has.
        let source_mask = document.segment.as_ref().map(|segment| segment.to_luma8());

        let page_height = bubble_map.height() as f32;
        // Centres of *every* block on the page, taken before the mutable borrow
        // below. Re-rendering a single block must still know where its
        // neighbours are, or a block sharing a merged balloon would trace
        // across the join again. A block's own centre is harmless in this list:
        // the clip ignores centres closer than one balloon apart.
        let block_centres: Vec<(f32, f32)> = document
            .text_blocks
            .iter()
            .map(|block| (block.x + block.width / 2.0, block.y + block.height / 2.0))
            .collect();
        // Detected balloons, so a block can tell whether it sits in a blank
        // interior it may fill or on artwork it must not overrun.
        let balloons: Vec<(f32, f32, f32, f32)> = document
            .balloons
            .iter()
            .map(|b| (b.x, b.y, b.width, b.height))
            .collect();

        let mut text_blocks = match text_block_index {
            Some(index) => document
                .text_blocks
                .get_mut(index)
                .map(|tb| vec![tb])
                .ok_or_else(|| anyhow::anyhow!("Text block index out of bounds"))?,
            None => document.text_blocks.iter_mut().collect(),
        };

        text_blocks.par_iter_mut().for_each(|text_block| {
            let _ = self.render_text_block(
                text_block,
                effect,
                stroke.clone(),
                font_family,
                Some(&bubble_map),
                source_mask.as_ref(),
                page_height,
                &block_centres,
                &balloons,
            );
        });

        if let Some(inpainted) = &document.inpainted
            && text_block_index.is_none()
        {
            let mut rendered = inpainted.to_rgba8();

            if let Some(brush_layer) = &document.brush_layer {
                let brush = brush_layer.to_rgba8();
                imageops::overlay(&mut rendered, &brush, 0, 0);
            }

            for text_block in text_blocks {
                let Some(block) = text_block.rendered.as_ref() else {
                    continue;
                };
                imageops::overlay(
                    &mut rendered,
                    &block.0,
                    text_block.x as i64,
                    text_block.y as i64,
                );
            }
            document.rendered = Some(SerializableDynamicImage(DynamicImage::ImageRgba8(rendered)));
        }
        Ok(())
    }

    fn render_text_block(
        &self,
        text_block: &mut TextBlock,
        effect: TextShaderEffect,
        global_stroke: Option<TextStrokeStyle>,
        font_family: Option<&str>,
        bubble_map: Option<&GrayImage>,
        source_mask: Option<&GrayImage>,
        page_height: f32,
        sibling_centres: &[(f32, f32)],
        balloons: &[(f32, f32, f32, f32)],
    ) -> Result<()> {
        let Some(translation) = text_block.translation.as_ref().cloned() else {
            return Ok(());
        };
        if translation.is_empty() {
            return Ok(());
        };
        let normalized_translation = normalize_translation_for_layout(&translation);
        let (seed_x, seed_y, seed_width, seed_height) = text_block.seed_layout_box();
        let layout_source_block = TextBlock {
            x: seed_x,
            y: seed_y,
            width: seed_width,
            height: seed_height,
            translation: Some(translation.clone()),
            source_direction: text_block.source_direction,
            rendered_direction: text_block.rendered_direction,
            ..Default::default()
        };

        let mut style = text_block.style.clone().unwrap_or_else(|| TextStyle {
            font_families: Vec::new(),
            font_size: None,
            color: [0, 0, 0, 255],
            effect: None,
            stroke: None,
            text_align: None,
        });

        apply_global_font_family(&mut style.font_families, font_family);
        apply_default_font_families(&mut style.font_families, &normalized_translation);
        let font = self.select_font(&style)?;
        let block_effect = style.effect.unwrap_or(effect);
        let color = text_block
            .style
            .as_ref()
            .map(|style| style.color)
            .unwrap_or([0, 0, 0, 255]);
        let writing_mode = writing_mode_for_block(&layout_source_block);
        let is_horizontal_latin =
            writing_mode == WritingMode::Horizontal && is_latin_only(&normalized_translation);
        let text_align = style.text_align.unwrap_or(if is_horizontal_latin {
            TextAlign::Center
        } else {
            TextAlign::Left
        });

        let centre = (
            text_block.x + text_block.width / 2.0,
            text_block.y + text_block.height / 2.0,
        );
        // Inside a balloon there is blank interior to fill; on artwork there is
        // not, and the original lettering size is the one that fits the gap.
        let containing_balloon = balloons.iter().copied().find(|(bx, by, bw, bh)| {
            centre.0 >= *bx && centre.0 <= bx + bw && centre.1 >= *by && centre.1 <= by + bh
        });
        let in_balloon = containing_balloon.is_some();

        let english_layout =
            english_layout_behavior(text_block, &normalized_translation, writing_mode);
        let english_horizontal_layout = english_layout != EnglishLayoutBehavior::Disabled;
        let auto_expand_english_layout = english_layout == EnglishLayoutBehavior::AutoExpand;
        let original_layout_box = layout_box_from_block(&layout_source_block);

        // The detected box only covers the source text. A Vietnamese line needs
        // far more room than the Japanese it replaces, so fitting it into that
        // box alone drives the font down to an unreadable size while the
        // balloon around it sits empty. Find the balloon instead, and fall back
        // to growing into adjacent clear space when it cannot be traced.
        // Outside a balloon there is no blank interior to grow into. The artist
        // set this lettering to the gap it sits in, so the gap is the box: keep
        // the detected one exactly where it is and hold the translation inside
        // it, rather than tracing clear space that belongs to the artwork.
        let balloon_box = if auto_expand_english_layout && in_balloon {
            bubble_map
                .and_then(|map| balloon_bounds_from_image(text_block, map))
                .map(|traced| clip_box_to_nearest_owner(traced, centre, sibling_centres))
        } else {
            None
        };
        let use_balloon = balloon_box.is_some();
        let mut layout_box = balloon_box.unwrap_or_else(|| {
            if auto_expand_english_layout && in_balloon {
                bubble_map
                    .map(|map| expand_latin_layout_box_strict(&layout_source_block, map))
                    .unwrap_or(original_layout_box)
            } else {
                original_layout_box
            }
        });

        // Typeset at the page's normal size and shrink only on overflow, rather
        // than fitting each balloon to its own maximum. Text the artist drew
        // large keeps that scale.
        // Measured from the block rather than taken from its short side, which
        // is several glyphs wide once the block holds more than one column.
        let source_glyph = source_glyph_size(
            layout_source_block.width,
            layout_source_block.height,
            text_block.text.as_deref(),
            text_block.detected_font_size_px,
        );
        let target_font_size = preferred_font_size(page_height, source_glyph, in_balloon);

        // Outside a balloon there is no blank interior to grow into: the artist
        // sized this lettering to the gap it sits in, and anything larger runs
        // over the artwork. Set it at that size rather than searching for one.
        let keep_source_size = english_horizontal_layout && !in_balloon && source_glyph.is_some();

        let build_layout = |box_for_layout: LayoutBox, allow_expanded_overflow: bool| {
            let max_width = if use_balloon
                || english_layout == EnglishLayoutBehavior::LockedToManualSize
            {
                // Traced balloon bounds, and boxes already fitted to a balloon
                // upstream, are the real limit — use them exactly.
                box_for_layout.width
            } else {
                let expanded_box = is_expanded_layout_box(box_for_layout, original_layout_box);
                let overflow = if english_horizontal_layout {
                    latin_width_overflow_factor(expanded_box, allow_expanded_overflow)
                } else {
                    1.0
                };
                if box_for_layout.width.is_finite() && box_for_layout.width > 0.0 {
                    box_for_layout.width * overflow
                } else {
                    box_for_layout.width
                }
            };

            let base = || {
                TextLayout::new(&font, None)
                    .with_fallback_fonts(&self.symbol_fallbacks)
                    .with_max_height(box_for_layout.height)
                    .with_max_width(max_width)
                    .with_writing_mode(writing_mode)
            };

            if keep_source_size {
                // Matching the original size is worth having only while the
                // result stays in its panel. A Vietnamese line is far longer
                // than the Japanese it replaces, and held at the original size
                // a long one runs off the bottom of the page — so when it will
                // not sit in the room, fall back to finding a size that does.
                let at_source = base().with_font_size(target_font_size);
                if let Ok(layout) = at_source.run(&normalized_translation)
                    && layout.height <= box_for_layout.height
                {
                    return Ok(layout);
                }
            }
            base()
                .with_preferred_font_size(target_font_size)
                .run(&normalized_translation)
        };

        // A shout the artist drew large, short enough to read down a column:
        // stacked it needs no hyphen and no narrow column of syllables, and it
        // is how the Japanese was lettered in the first place.
        let stack_shout = english_horizontal_layout
            && is_emphatic_lettering(page_height, text_block.detected_font_size_px)
            && is_stackable_shout(&normalized_translation);

        // What the block was actually set with, when that is not what it came
        // in with.
        let mut set_text: Option<String> = None;

        let mut layout = build_layout(layout_box, false)?;

        // A box fitted to a balloon upstream is deliberately conservative — the
        // largest rectangle inside a rounded shape leaves the corners unused,
        // and an unbreakable word can still force the font down inside it. When
        // the result ends up lost in its balloon, reach out toward the balloon's
        // own edge and try again; keep the attempt only if it actually reads
        // larger.
        if text_block.balloon_fitted
            && english_horizontal_layout
            && fill_ratio(&layout, layout_box) < MIN_FILL_RATIO
            && let Some(balloon) = containing_balloon
        {
            let grown = grow_box_within(layout_box, balloon);
            if layout_box_area(grown) > layout_box_area(layout_box) * 1.06
                && let Ok(candidate) = build_layout(grown, true)
                && candidate.font_size > layout.font_size + 0.25
            {
                tracing::debug!(
                    from = layout.font_size,
                    to = candidate.font_size,
                    "grew a balloon-fitted box to rescue small text"
                );
                layout = candidate;
                layout_box = grown;
            }
        }

        if auto_expand_english_layout {
            if !use_balloon {
                // No balloon traced: rescue a layout that came out too small.
                if latin_layout_underfilled(&layout, layout_box.height) {
                    let relaxed_box = bubble_map
                        .map(|map| expand_latin_layout_box_relaxed(&layout_source_block, map))
                        .unwrap_or(layout_box);
                    let relaxed_candidate =
                        if layout_box_area(relaxed_box) > layout_box_area(layout_box) * 1.06 {
                            build_layout(relaxed_box, true)
                                .ok()
                                .map(|layout| (layout, relaxed_box))
                        } else {
                            None
                        };
                    let overflow_candidate = build_layout(layout_box, true)
                        .ok()
                        .map(|layout| (layout, layout_box));
                    if let Some((candidate_layout, candidate_box)) =
                        pick_better_latin_candidate(&layout, relaxed_candidate, overflow_candidate)
                    {
                        layout = candidate_layout;
                        layout_box = candidate_box;
                    }
                }
            }
        }

        // Outside a balloon the shape to fill is the one the source text left
        // behind — an L where a column stopped short of a figure, a U around
        // one, a Z down a stepped panel. A rectangle drawn round any of those
        // reaches over the drawing.
        if keep_source_size
            && let Some(mask) = source_mask
            && let Some(page) = bubble_map
            && let Some((source_box, rows)) = source_text_rows(&layout_source_block, mask, page)
        {
            let into_shape = |rows: &[(f32, f32)], ceiling: f32| {
                TextLayout::new(&font, None)
                    .with_fallback_fonts(&self.symbol_fallbacks)
                    .with_row_spans(RowSpans::new(rows.to_vec()))
                    .with_preferred_font_size(ceiling)
                    .with_writing_mode(writing_mode)
                    .run(&normalized_translation)
                    .ok()
                    .filter(|layout| layout.fits)
            };

            // The artist's own size unless that is too small to read, in which
            // case there is something to gain by growing.
            let ceiling = target_font_size.max(OUTSIDE_BALLOON_TARGET_FONT_SIZE);

            // Grow a ring at a time into whatever blank page surrounds the
            // lettering, stopping as soon as the text reads at the target or
            // the drawing closes in. Re-measured each ring so it never takes
            // more room than it needs.
            // A layout's line positions are absolute inside the shape it was
            // measured against, so the shape it was measured against is the one
            // it has to be drawn in. Kept as a pair for that reason: growing
            // the shape while holding on to an older layout draws every line
            // shifted by the difference between the two origins.
            let mut probe_box = source_box;
            let mut probe_rows = rows;
            let mut best = into_shape(&probe_rows, ceiling).map(|layout| (layout, probe_box));
            for _ in 1..=SHAPE_GROWTH_MAX_STEPS {
                if best
                    .as_ref()
                    .is_some_and(|(layout, _)| layout.font_size >= ceiling)
                {
                    break;
                }
                let Some((grown_box, grown_rows)) =
                    grow_rows_into_blank(probe_box, &probe_rows, page, SHAPE_GROWTH_STEP_PX)
                else {
                    break;
                };
                if grown_box.width <= probe_box.width && grown_box.height <= probe_box.height {
                    // Boxed in by the drawing on every side.
                    break;
                }
                if let Some(grown_fit) = into_shape(&grown_rows, ceiling)
                    && best
                        .as_ref()
                        .is_none_or(|(before, _)| grown_fit.font_size > before.font_size)
                {
                    best = Some((grown_fit, grown_box));
                }
                probe_box = grown_box;
                probe_rows = grown_rows;
            }

            if let Some((mut fitted, source_box)) = best {
                align_layout_horizontally(&mut fitted, writing_mode, source_box.width, text_align);
                return self.paint_block(PaintBlock {
                    text_block,
                    set_text,
                    layout: &fitted,
                    layout_box: source_box,
                    writing_mode,
                    style: &style,
                    color,
                    effect: block_effect,
                    global_stroke: global_stroke.as_ref(),
                    font: &font,
                });
            }
        }

        // Follow the shape of the clear space rather than the largest rectangle
        // that fits inside it. A balloon with a figure drawn across it loses
        // more than half its room to a rectangle; measured row by row it keeps
        // it. The block's share of a merged balloon bounds the search, so two
        // blocks in one balloon still do not reach across the join.
        //
        // `lock_layout_box` is set by the balloon refit upstream, not only by a
        // reader dragging a box, so `AutoExpand` alone would skip every block in
        // a balloon — which is exactly the case the rectangle is too small for.
        // Take those as well, the way the fill-ratio rescue above does.
        let shaped = if english_horizontal_layout
            && (auto_expand_english_layout || text_block.balloon_fitted)
            && let Some(map) = bubble_map
            && let Some(balloon) = containing_balloon
        {
            let share = clip_box_to_nearest_owner(
                LayoutBox {
                    x: balloon.0,
                    y: balloon.1,
                    width: balloon.2,
                    height: balloon.3,
                },
                centre,
                sibling_centres,
            );
            clear_space_rows(
                &layout_source_block,
                map,
                (share.x, share.y, share.width, share.height),
            )
        } else {
            None
        };

        if let Some((shape_box, rows)) = shaped {
            let spans = RowSpans::new(rows);
            let shortened = shorten_elongation(&normalized_translation);
            let set = |spans: &RowSpans, stacked: bool, text: &str, size: Option<f32>| {
                let candidate = TextLayout::new(&font, None)
                    .with_fallback_fonts(&self.symbol_fallbacks)
                    .with_preferred_font_size(target_font_size)
                    .with_row_spans(spans.clone())
                    .with_writing_mode(writing_mode);
                let candidate = if stacked {
                    candidate.with_stacked_glyphs()
                } else {
                    candidate
                };
                match size {
                    Some(size) => candidate.run_whole_at(text, size),
                    None => candidate.run(text),
                }
            };
            // Re-set a shout that will not sit: base size as a line, then as a
            // column, then a point down, and only then with the drawn-out
            // letters dropped. Nothing is cut along the way.
            let reset_shout = |spans: &RowSpans| {
                shout_plan(&normalized_translation, shortened.as_deref())
                    .into_iter()
                    .find_map(|(text, size, column)| {
                        set(spans, column, text, Some(size))
                            .ok()
                            .filter(|run| run.fits && run.max_word_cuts == 0)
                            .map(|run| (run, text.to_string()))
                    })
            };

            // Keep it only if it actually reads larger. Traced clear space can
            // come out narrower than the detected box — text lettered straight
            // onto artwork has no balloon interior to find — and there the
            // rectangle was the better answer all along.
            if let Ok(mut shaped_layout) = set(&spans, false, &normalized_translation, None)
                && shaped_layout.fits
                && shaped_layout.font_size > layout.font_size
            {
                // Settle the text into the middle of the room. Re-set it from
                // the lower start rather than sliding the finished lines down,
                // which would carry a wide line into a narrow part of the shape.
                let mut settled = spans.clone();
                let spare = (shape_box.height - shaped_layout.height).max(0.0);
                if spare > 2.0 {
                    let lower = spans.blank_top((spare / 2.0) as usize);
                    if let Ok(centred) = set(&lower, false, &normalized_translation, None)
                        && centred.fits
                    {
                        shaped_layout = centred;
                        settled = lower;
                    }
                }

                if stack_shout
                    && (shaped_layout.max_word_cuts > 0
                        || shaped_layout.font_size < SHOUT_BASE_FONT_SIZE)
                    && let Some((reset, text)) = reset_shout(&settled)
                {
                    shaped_layout = reset;
                    set_text = Some(text);
                }

                tracing::debug!(
                    from = layout.font_size,
                    to = shaped_layout.font_size,
                    "followed the shape of the clear space"
                );
                layout = shaped_layout;
                layout_box = shape_box;
                align_layout_horizontally(&mut layout, writing_mode, layout_box.width, text_align);
                return self.paint_block(PaintBlock {
                    text_block,
                    set_text,
                    layout: &layout,
                    layout_box,
                    writing_mode,
                    style: &style,
                    color,
                    effect: block_effect,
                    global_stroke: global_stroke.as_ref(),
                    font: &font,
                });
            }
        }

        // The same for a block the rectangle won.
        if stack_shout
            && (layout.max_word_cuts > 0 || layout.font_size < SHOUT_BASE_FONT_SIZE)
        {
            let shortened = shorten_elongation(&normalized_translation);
            let reset = shout_plan(&normalized_translation, shortened.as_deref())
                .into_iter()
                .find_map(|(text, size, column)| {
                    let candidate = TextLayout::new(&font, None)
                        .with_fallback_fonts(&self.symbol_fallbacks)
                        .with_max_width(layout_box.width)
                        .with_max_height(layout_box.height)
                        .with_writing_mode(writing_mode);
                    let candidate = if column {
                        candidate.with_stacked_glyphs()
                    } else {
                        candidate
                    };
                    candidate
                        .run_whole_at(text, size)
                        .ok()
                        .filter(|run| {
                            run.max_word_cuts == 0
                                && run.width <= layout_box.width
                                && run.height <= layout_box.height
                        })
                        .map(|run| (run, text.to_string()))
                });
            if let Some((reset, text)) = reset {
                layout = reset;
                set_text = Some(text);
            }
        }

        if is_horizontal_latin {
            center_layout_vertically(&mut layout, layout_box.height);
        }
        align_layout_horizontally(&mut layout, writing_mode, layout_box.width, text_align);

        self.paint_block(PaintBlock {
            text_block,
            set_text,
            layout: &layout,
            layout_box,
            writing_mode,
            style: &style,
            color,
            effect: block_effect,
            global_stroke: global_stroke.as_ref(),
            font: &font,
        })
    }

    fn paint_block(&self, paint: PaintBlock<'_, '_>) -> Result<()> {
        let PaintBlock {
            text_block,
            set_text,
            layout,
            layout_box,
            writing_mode,
            style,
            color,
            effect,
            global_stroke,
            font,
        } = paint;

        let resolved_stroke = resolve_stroke_style(
            text_block,
            style.stroke.as_ref(),
            global_stroke,
            layout.font_size,
        );
        let rendered = self.renderer.render(
            layout,
            writing_mode,
            &RenderOptions {
                font_size: layout.font_size,
                color,
                effect,
                stroke: resolved_stroke,
                ..Default::default()
            },
        )?;

        // A shout whose drawn-out letters were dropped is drawn as KHÔNG! but
        // would still be stored as KHÔNGGG!. A PSD or TIFF export writes its
        // editable text layer from the stored translation while the pixels come
        // from here, so the two would disagree.
        if let Some(text) = set_text {
            text_block.translation = Some(text);
        }
        text_block.x = layout_box.x;
        text_block.y = layout_box.y;
        text_block.width = layout_box.width;
        text_block.height = layout_box.height;
        text_block.rendered_direction = Some(match writing_mode {
            WritingMode::Horizontal => koharu_types::TextDirection::Horizontal,
            WritingMode::VerticalRl => koharu_types::TextDirection::Vertical,
        });
        text_block.rendered = Some(SerializableDynamicImage(DynamicImage::ImageRgba8(rendered)));
        let persisted_style = text_block.style.get_or_insert_with(|| TextStyle {
            font_families: Vec::new(),
            font_size: None,
            color,
            effect: None,
            stroke: None,
            text_align: None,
        });
        persisted_style.font_families = vec![font.post_script_name().to_string()];
        // The size the text was actually laid out at. PSD/TIFF export reads
        // this; without it those layers fall back to a guess from font
        // detection rather than what was rendered.
        persisted_style.font_size = Some(layout.font_size);
        Ok(())
    }

    fn select_font(&self, style: &TextStyle) -> Result<Font> {
        let mut fontbook = self
            .fontbook
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock fontbook"))?;
        let faces = fontbook.all_families();
        let post_script_name = style
            .font_families
            .iter()
            .find_map(|candidate| face_post_script_name(&faces, candidate))
            .ok_or_else(|| {
                anyhow::anyhow!("no font found for candidates: {:?}", style.font_families)
            })?;
        fontbook.query(&post_script_name)
    }
}

/// Whether a block's layout box may be grown to fit the translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnglishLayoutBehavior {
    /// Vertical or CJK output — the detected box is authoritative.
    Disabled,
    /// The user dragged this block to a size of their own; respect it.
    LockedToManualSize,
    AutoExpand,
}

fn english_layout_behavior(
    text_block: &TextBlock,
    normalized_translation: &str,
    writing_mode: WritingMode,
) -> EnglishLayoutBehavior {
    let is_english_horizontal =
        writing_mode == WritingMode::Horizontal && is_latin_only(normalized_translation);
    if !is_english_horizontal {
        return EnglishLayoutBehavior::Disabled;
    }

    if text_block.lock_layout_box {
        EnglishLayoutBehavior::LockedToManualSize
    } else {
        EnglishLayoutBehavior::AutoExpand
    }
}

fn default_stroke_width(font_size: f32) -> f32 {
    (font_size * 0.10).clamp(1.2, 8.0)
}

fn apply_global_font_family(font_families: &mut Vec<String>, font_family: Option<&str>) {
    if font_families.is_empty()
        && let Some(font_family) = font_family
    {
        font_families.push(font_family.to_string());
    }
}

fn apply_default_font_families(font_families: &mut Vec<String>, text: &str) {
    if font_families.is_empty() {
        *font_families = font_families_for_text(text);
    }
}

/// What text outside a balloon is grown toward. Its own footprint is usually
/// too small for the translation, and past this there is nothing to gain:
/// the line is readable and reaching further only walks it away from where the
/// artist set it.
const OUTSIDE_BALLOON_TARGET_FONT_SIZE: f32 = 14.0;

/// A shout that will not sit in its balloon is re-set at this size, and the
/// choice between a line and a column is made there — before anything gets cut.
const SHOUT_BASE_FONT_SIZE: f32 = 12.0;
/// Two words cannot go down a column one letter at a time, so when they will
/// not take the base size they come down a point instead of being cut.
const SHOUT_TWO_WORD_FONT_SIZE: f32 = 11.0;

/// What to try, in order, for a shout that has to be re-set: the base size as a
/// line, then as a column, then a point down when a column is not an option,
/// and only then with the drawn-out letters dropped. Each entry is
/// `(text, size, as a column)`. Nothing here cuts a word.
fn shout_plan<'a>(full: &'a str, shortened: Option<&'a str>) -> Vec<(&'a str, f32, bool)> {
    let single_word = is_stackable_shout(full);
    let mut plan = vec![(full, SHOUT_BASE_FONT_SIZE, false)];
    if single_word {
        plan.push((full, SHOUT_BASE_FONT_SIZE, true));
    } else {
        plan.push((full, SHOUT_TWO_WORD_FONT_SIZE, false));
    }
    if let Some(shortened) = shortened {
        plan.push((shortened, SHOUT_BASE_FONT_SIZE, false));
        if single_word {
            plan.push((shortened, SHOUT_BASE_FONT_SIZE, true));
        }
    }
    plan
}

/// Everything `paint_block` needs, gathered so the two ways of arriving there —
/// a shape-guided layout and a rectangular one — hand over the same thing.
struct PaintBlock<'block, 'layout> {
    text_block: &'block mut TextBlock,
    /// Set when the text drawn is not the text the block came in with.
    set_text: Option<String>,
    layout: &'block LayoutRun<'layout>,
    layout_box: LayoutBox,
    writing_mode: WritingMode,
    style: &'block TextStyle,
    color: [u8; 4],
    effect: TextShaderEffect,
    global_stroke: Option<&'block TextStrokeStyle>,
    font: &'block Font,
}

fn resolve_stroke_style(
    block: &TextBlock,
    block_stroke: Option<&TextStrokeStyle>,
    global_stroke: Option<&TextStrokeStyle>,
    font_size: f32,
) -> Option<RenderStrokeOptions> {
    if let Some(stroke) = block_stroke {
        if !stroke.enabled {
            return None;
        }
        return Some(RenderStrokeOptions {
            color: stroke.color,
            width_px: stroke
                .width_px
                .unwrap_or_else(|| default_stroke_width(font_size)),
        });
    }

    if let Some(stroke) = global_stroke {
        if !stroke.enabled {
            return None;
        }
        return Some(RenderStrokeOptions {
            color: stroke.color,
            width_px: stroke
                .width_px
                .unwrap_or_else(|| default_stroke_width(font_size)),
        });
    }

    if let Some(pred) = &block.font_prediction
        && pred.stroke_width_px > 0.0
    {
        return Some(RenderStrokeOptions {
            color: [
                pred.stroke_color[0],
                pred.stroke_color[1],
                pred.stroke_color[2],
                255,
            ],
            width_px: pred.stroke_width_px,
        });
    }

    Some(RenderStrokeOptions {
        color: [255, 255, 255, 255],
        width_px: default_stroke_width(font_size),
    })
}

fn align_layout_horizontally(
    layout: &mut LayoutRun<'_>,
    writing_mode: WritingMode,
    container_width: f32,
    text_align: TextAlign,
) {
    if !container_width.is_finite() || container_width <= 0.0 {
        return;
    }

    let target_width = layout.width.max(container_width);
    if writing_mode.is_vertical() {
        let remaining = (container_width - layout.width).max(0.0);
        let offset = match text_align {
            TextAlign::Left => 0.0,
            TextAlign::Center => remaining * 0.5,
            TextAlign::Right => remaining,
        };
        if offset > 0.0 {
            for line in &mut layout.lines {
                line.baseline.0 += offset;
            }
        }
        layout.width = target_width;
        return;
    }

    for line in &mut layout.lines {
        if line.advance <= 0.0 {
            continue;
        }
        // A line set into a shape was measured against the room at its own
        // height, and its baseline already starts there. Centring it in the
        // whole box would push it back over whatever the shape was avoiding.
        let room = line.span.map_or(container_width, |(_, room)| room);
        let remaining = (room - line.advance).max(0.0);
        let offset = match text_align {
            TextAlign::Left => 0.0,
            TextAlign::Center => remaining * 0.5,
            TextAlign::Right => remaining,
        };
        match line.span {
            Some((start, _)) => line.baseline.0 = start + offset,
            None if offset > 0.0 => line.baseline.0 += offset,
            None => {}
        }
    }
    layout.width = target_width;
}

fn center_layout_vertically(layout: &mut LayoutRun<'_>, container_height: f32) {
    if !container_height.is_finite() || container_height <= 0.0 || layout.lines.is_empty() {
        return;
    }
    let offset = ((container_height - layout.height) * 0.5).max(0.0);
    if offset <= 0.0 {
        return;
    }

    for line in &mut layout.lines {
        line.baseline.1 += offset;
    }
    layout.height = layout.height.max(container_height);
}

fn load_symbol_fallbacks(fontbook: &mut FontBook) -> Vec<Font> {
    let candidates = [
        "Segoe UI Symbol",
        "Segoe UI Emoji",
        "Noto Sans Symbols",
        "Noto Sans Symbols2",
        "Noto Color Emoji",
        "Apple Color Emoji",
        "Apple Symbols",
        "Symbola",
        "Arial Unicode MS",
    ];
    let faces = fontbook.all_families();
    candidates
        .iter()
        .filter_map(|candidate| face_post_script_name(&faces, candidate))
        .filter_map(|post_script_name| fontbook.query(&post_script_name).ok())
        .collect()
}

fn face_post_script_name(faces: &[FaceInfo], candidate: &str) -> Option<String> {
    faces
        .iter()
        .find(|face| {
            face.post_script_name == candidate
                || face
                    .families
                    .iter()
                    .any(|(family, _)| family.as_str() == candidate)
        })
        .map(|face| face.post_script_name.clone())
        .filter(|post_script_name| !post_script_name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{
        EnglishLayoutBehavior, align_layout_horizontally, apply_default_font_families,
        apply_global_font_family, center_layout_vertically, english_layout_behavior,
    };
    use crate::layout::{LayoutLine, LayoutRun, WritingMode};
    use koharu_types::{TextAlign, TextBlock};

    #[test]
    fn a_shout_is_re_set_line_first_then_column_then_shorter() {
        // A single word: the base size as a line, then as a column, then the
        // same two with the drawn-out letters gone. Nothing is cut.
        let plan = super::shout_plan("KHÔNGGG!", Some("KHÔNG!"));
        assert_eq!(
            plan,
            vec![
                ("KHÔNGGG!", super::SHOUT_BASE_FONT_SIZE, false),
                ("KHÔNGGG!", super::SHOUT_BASE_FONT_SIZE, true),
                ("KHÔNG!", super::SHOUT_BASE_FONT_SIZE, false),
                ("KHÔNG!", super::SHOUT_BASE_FONT_SIZE, true),
            ]
        );
    }

    #[test]
    fn two_words_come_down_a_point_instead_of_going_down_a_column() {
        // Letters of two words cannot be read down one column, so the step
        // after the base size is a point smaller, not a column.
        let plan = super::shout_plan("ĐỒ NGỐC!", None);
        assert_eq!(
            plan,
            vec![
                ("ĐỒ NGỐC!", super::SHOUT_BASE_FONT_SIZE, false),
                ("ĐỒ NGỐC!", super::SHOUT_TWO_WORD_FONT_SIZE, false),
            ]
        );
    }

    #[test]
    fn horizontal_alignment_offsets_each_line() {
        let mut layout = LayoutRun {
            lines: vec![
                LayoutLine {
                    advance: 40.0,
                    baseline: (0.0, 10.0),
                    ..Default::default()
                },
                LayoutLine {
                    advance: 80.0,
                    baseline: (0.0, 30.0),
                    ..Default::default()
                },
            ],
            width: 80.0,
            height: 40.0,
            font_size: 16.0,
            fits: true,
            max_word_cuts: 0,
        };

        align_layout_horizontally(
            &mut layout,
            WritingMode::Horizontal,
            100.0,
            TextAlign::Center,
        );

        assert_eq!(layout.lines[0].baseline.0, 30.0);
        assert_eq!(layout.lines[1].baseline.0, 10.0);
        assert_eq!(layout.width, 100.0);
    }

    #[test]
    fn right_alignment_uses_full_remaining_width() {
        let mut layout = LayoutRun {
            lines: vec![LayoutLine {
                advance: 40.0,
                baseline: (0.0, 10.0),
                ..Default::default()
            }],
            width: 40.0,
            height: 20.0,
            font_size: 16.0,
            fits: true,
            max_word_cuts: 0,
        };

        align_layout_horizontally(
            &mut layout,
            WritingMode::Horizontal,
            100.0,
            TextAlign::Right,
        );

        assert_eq!(layout.lines[0].baseline.0, 60.0);
    }

    #[test]
    fn vertical_alignment_offsets_all_columns_as_a_group() {
        let mut layout = LayoutRun {
            lines: vec![
                LayoutLine {
                    baseline: (10.0, 12.0),
                    ..Default::default()
                },
                LayoutLine {
                    baseline: (30.0, 12.0),
                    ..Default::default()
                },
            ],
            width: 40.0,
            height: 80.0,
            font_size: 16.0,
            fits: true,
            max_word_cuts: 0,
        };

        align_layout_horizontally(
            &mut layout,
            WritingMode::VerticalRl,
            100.0,
            TextAlign::Center,
        );

        assert_eq!(layout.lines[0].baseline.0, 40.0);
        assert_eq!(layout.lines[1].baseline.0, 60.0);
        assert_eq!(layout.width, 100.0);
    }

    #[test]
    fn vertical_centering_preserves_existing_behavior() {
        let mut layout = LayoutRun {
            lines: vec![LayoutLine {
                advance: 40.0,
                baseline: (0.0, 12.0),
                ..Default::default()
            }],
            width: 40.0,
            height: 20.0,
            font_size: 16.0,
            fits: true,
            max_word_cuts: 0,
        };

        center_layout_vertically(&mut layout, 60.0);

        assert_eq!(layout.lines[0].baseline.1, 32.0);
        assert_eq!(layout.height, 60.0);
    }

    #[test]
    fn explicit_block_font_should_not_be_overridden_by_global_font() {
        let mut font_families = vec!["Block Font".to_string()];
        apply_global_font_family(&mut font_families, Some("Global Font"));

        assert_eq!(font_families, vec!["Block Font".to_string()]);
    }

    #[test]
    fn global_font_should_fill_empty_block_font_list() {
        let mut font_families = Vec::new();
        apply_global_font_family(&mut font_families, Some("Global Font"));
        assert_eq!(font_families, vec!["Global Font".to_string()]);
    }

    #[test]
    fn default_font_families_should_fill_empty_list() {
        let mut font_families = Vec::new();
        apply_default_font_families(&mut font_families, "hello");
        assert!(!font_families.is_empty());
    }

    #[test]
    fn global_font_should_be_applied_before_default_script_fonts() {
        let mut font_families = Vec::new();
        apply_global_font_family(&mut font_families, Some("Global Font"));
        apply_default_font_families(&mut font_families, "hello");

        assert_eq!(font_families, vec!["Global Font".to_string()]);
    }

    #[test]
    fn english_layout_auto_expands_by_default() {
        let block = TextBlock::default();
        let behavior = english_layout_behavior(&block, "HELLO WORLD", WritingMode::Horizontal);
        assert_eq!(behavior, EnglishLayoutBehavior::AutoExpand);
    }

    #[test]
    fn english_layout_stops_auto_expand_after_manual_resize() {
        let block = TextBlock {
            lock_layout_box: true,
            ..Default::default()
        };
        let behavior = english_layout_behavior(&block, "HELLO WORLD", WritingMode::Horizontal);
        assert_eq!(behavior, EnglishLayoutBehavior::LockedToManualSize);
    }

    #[test]
    fn non_english_layout_never_uses_english_expansion_logic() {
        let block = TextBlock::default();
        let behavior = english_layout_behavior(&block, "こんにちは", WritingMode::Horizontal);
        assert_eq!(behavior, EnglishLayoutBehavior::Disabled);
    }
}
