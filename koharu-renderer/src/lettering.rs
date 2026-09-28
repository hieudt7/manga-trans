//! Post-layout spacing for outlined FREE_TEXT. Never shapes or wraps text.
use crate::{
    layout::{LayoutLine, LayoutRun, WritingMode},
    renderer::{RenderOptions, TinySkiaRenderer},
};
use anyhow::Result;
use image::RgbaImage;
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Bounds {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}
impl Bounds {
    fn shifted(self, x: i32, y: i32) -> Self {
        Self {
            left: self.left + x,
            right: self.right + x,
            top: self.top + y,
            bottom: self.bottom + y,
        }
    }
    fn width(self) -> i32 {
        self.right - self.left
    }
    fn height(self) -> i32 {
        self.bottom - self.top
    }
}
fn bounds(image: &RgbaImage) -> Option<Bounds> {
    let mut b = Bounds {
        left: image.width() as i32,
        top: image.height() as i32,
        right: 0,
        bottom: 0,
    };
    for (x, y, p) in image.enumerate_pixels() {
        if p[3] > 0 {
            b.left = b.left.min(x as i32);
            b.top = b.top.min(y as i32);
            b.right = b.right.max(x as i32 + 1);
            b.bottom = b.bottom.max(y as i32 + 1);
        }
    }
    (b.right > b.left).then_some(b)
}
fn probe(
    renderer: &TinySkiaRenderer,
    line: &LayoutLine<'_>,
    size: f32,
    opts: &RenderOptions,
) -> Result<Option<Bounds>> {
    let pad = (size * 3.0).ceil();
    let mut line = line.clone();
    line.baseline = (pad, pad);
    let mut pen = 0.0f32;
    let mut extent = 0.0f32;
    for g in &line.glyphs {
        extent = extent.max(pen + g.x_offset + size * 2.0);
        pen += g.x_advance;
    }
    let run = LayoutRun {
        lines: vec![line],
        width: extent + pad * 2.0,
        height: pad * 2.0,
        font_size: size,
        fits: true,
        max_word_cuts: 0,
    };
    Ok(
        bounds(&renderer.render(&run, WritingMode::Horizontal, opts)?)
            .map(|b| b.shifted(-(pad as i32), -(pad as i32))),
    )
}

/// Positions existing shaped clusters; combining marks stay with their base.
/// Bounds are measured with the actual rasterizer, including every AA pixel.
pub(crate) fn apply(
    renderer: &TinySkiaRenderer,
    layout: &mut LayoutRun<'_>,
    opts: &RenderOptions,
    text: &str,
    block_id: &str,
    region_width: f32,
    region_height: f32,
) -> Result<f32> {
    let Some(stroke) = opts.stroke.filter(|s| s.width_px > 0.0 && s.color[3] > 0) else {
        return Ok(0.0);
    };
    let mut fill_opts = opts.clone();
    fill_opts.stroke = None;
    let before = layout.clone();
    let mut measurements = Vec::new();
    let mut report = String::new();
    for (index, line) in layout.lines.iter_mut().enumerate() {
        let mut desired_gap = 1;
        let mut total_shift = 0;
        let mut pair_report = String::new();
        // First reserve a clear fill pixel; then use only the existing line's
        // spare horizontal room to reduce excessive outline overlap.
        for pass in 0..2 {
            pair_report.clear();
            let mut start = 0;
            let mut pen = 0.0f32;
            let mut shift = 0i32;
            let mut previous: Option<Bounds> = None;
            let mut previous_stroke: Option<Bounds> = None;
            while start < line.glyphs.len() {
                let mut end = start + 1;
                while end < line.glyphs.len()
                    && line.glyphs[end].cluster == line.glyphs[start].cluster
                {
                    end += 1;
                }
                let mut cluster = line.clone();
                cluster.glyphs = line.glyphs[start..end].to_vec();
                for g in &mut cluster.glyphs {
                    g.x_offset += pen;
                }
                let original = probe(renderer, &cluster, layout.font_size, &fill_opts)?;
                let outer = probe(renderer, &cluster, layout.font_size, opts)?;
                if let Some(original) = original {
                    if let Some(prev) = previous {
                        let gap = original.left + shift - prev.right;
                        // Keep word spaces (empty clusters) and existing close pairs.
                        // Tighten open pairs only; keep at least one clear fill pixel.
                        let desired = desired_gap;
                        let adjustment = desired - gap;
                        shift += adjustment;
                        let actual_gap = original.left + shift - prev.right;
                        let stroke_gap = outer
                            .zip(previous_stroke)
                            .map(|(a, b)| a.left + shift - b.right);
                        pair_report.push_str(&format!("pair line={index} cluster={} tracking_delta={adjustment} fill_gap={actual_gap} stroke_gap={stroke_gap:?}\n",line.glyphs[start].cluster));
                    }
                    previous = Some(original.shifted(shift, 0));
                    previous_stroke = outer.map(|b| b.shifted(shift, 0));
                } else {
                    previous = None;
                    previous_stroke = None;
                }
                for g in &mut line.glyphs[start..end] {
                    g.x_offset += shift as f32;
                    pen += g.x_advance;
                }
                start = end;
            }
            total_shift += shift;
            if pass == 0 {
                let outer = probe(renderer, line, layout.font_size, opts)?;
                let pairs = pair_report.lines().count().max(1) as f32;
                let spare = outer
                    .map(|b| (region_width - b.width() as f32).max(0.0))
                    .unwrap_or(0.0);
                desired_gap = (1.0 + (spare / pairs).floor())
                    .min((layout.font_size * 0.10).round().max(1.0))
                    as i32;
            }
        }
        report.push_str(&pair_report);
        let fill = probe(renderer, line, layout.font_size, &fill_opts)?;
        let outer = probe(renderer, line, layout.font_size, opts)?;
        measurements.push((fill, outer));
        report.push_str(&format!(
            "line={index} text={:?} font_size={} stroke_width={} tracking_total={}\n",
            text.get(line.range.clone()).unwrap_or(""),
            layout.font_size,
            stroke.width_px,
            total_shift
        ));
    }
    let heights: i32 = measurements
        .iter()
        .filter_map(|(_, b)| b.map(Bounds::height))
        .sum();
    let gaps = layout.lines.len().saturating_sub(1) as i32;
    // Keep fractional baselines: the requested subpixel leading is
    // rasterized with antialiasing, not rounded to whole pixel rows.
    let gap = 0.3f32;
    // Give overflowing outlines room on both sides without changing the page axis.
    let widest = measurements.iter().filter_map(|(_, b)| b.map(Bounds::width))
        .max().unwrap_or(0) as f32;
    let horizontal_pad = ((widest - region_width).max(0.0) / 2.0).ceil();
    let canvas_width = region_width + horizontal_pad * 2.0;
    let total = heights as f32 + gaps as f32 * gap;
    let mut top = ((region_height - total) / 2.0).floor().max(0.0);
    for (i, (line, (_, outer))) in layout.lines.iter_mut().zip(&measurements).enumerate() {
        if let Some(b) = outer {
            line.baseline = (
                ((canvas_width - b.width() as f32) / 2.0).floor() - b.left as f32,
                top - b.top as f32,
            );
            let (fill, outer) = measurements[i];
            let x = line.baseline.0 as i32;
            let y = line.baseline.1 as i32;
            report.push_str(&format!(
                "bounds line={i} fill={:?} stroke={:?} visual_gap_next={}\n",
                fill.map(|b| b.shifted(x, y)),
                outer.map(|b| b.shifted(x, y)),
                if i + 1 < measurements.len() { gap } else { 0.0 }
            ));
            top += b.height() as f32 + gap;
        }
    }
    // Preserve available region and font selection even when constraints cannot
    // all fit; report overflow rather than silently rewrap or reduce font size.
    if total as f32 > region_height {
        tracing::warn!(
            block_id,
            total,
            region_height,
            "outlined lines exceed unchanged region"
        );
    }
    layout.width = layout.width.max(canvas_width);
    layout.height = layout.height.max(region_height).max(top as f32);
    tracing::info!(block_id, report=%report, "connected lettering geometry");
    if let Ok(dir) = std::env::var("KOHARU_DEBUG_LETTERING") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir)?;
        renderer
            .render(&before, WritingMode::Horizontal, opts)?
            .save(dir.join(format!("{block_id}-before.png")))?;
        let mut debug = renderer.render(layout, WritingMode::Horizontal, opts)?;
        for (line, (fill, stroke)) in layout.lines.iter().zip(measurements) {
            for (b, color) in [
                (stroke, image::Rgba([0, 80, 255, 255])),
                (fill, image::Rgba([255, 0, 0, 255])),
            ] {
                if let Some(b) = b {
                    let b = b.shifted(line.baseline.0 as i32, line.baseline.1 as i32);
                    draw_hollow_rect_mut(
                        &mut debug,
                        Rect::at(b.left, b.top).of_size(b.width() as u32, b.height() as u32),
                        color,
                    );
                }
            }
        }
        debug.save(dir.join(format!("{block_id}-bounds.png")))?;
        std::fs::write(dir.join(format!("{block_id}.log")), report)?;
    }
    Ok(horizontal_pad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{font::FontBook, layout::TextLayout, renderer::RenderStrokeOptions};

    #[test]
    #[ignore = "requires the installed ChalkboardSE-Bold regression font"]
    fn uppercase_overflow_keeps_first_letter_and_outline() -> Result<()> {
        let font = FontBook::new().query("ChalkboardSE-Bold")?;
        let renderer = TinySkiaRenderer::new()?;
        let text = "CỦA TÔI RỒI...";
        let mut layout = TextLayout::new(&font, Some(30.0))
            .with_max_width(600.0).with_max_height(100.0).run(text)?;
        assert_eq!(layout.lines.len(), 1);
        let opts = RenderOptions {
            font_size: layout.font_size,
            stroke: Some(RenderStrokeOptions { color: [255; 4], width_px: 6.9 }),
            ..Default::default()
        };
        let pad = apply(&renderer, &mut layout, &opts, text, "overflow", 80.0, 100.0)?;
        assert!(pad > 0.0);
        let line = &layout.lines[0];
        let expected = probe(&renderer, line, layout.font_size, &opts)?.unwrap();
        let actual = bounds(&renderer.render(&layout, WritingMode::Horizontal, &opts)?).unwrap();
        assert_eq!(actual.width(), expected.width(), "first or last glyph clipped");
        assert_eq!(actual.height(), expected.height(), "outline clipped");
        let page_center = (actual.left + actual.right) as f32 / 2.0 - pad;
        assert!((page_center - 40.0).abs() <= 0.5, "page axis moved");
        Ok(())
    }

    #[test]
    #[ignore = "requires the installed HL-Comic1unicode-Normal regression font"]
    fn vietnamese_pixel_bounds_preserve_shaping_and_separate_lines() -> Result<()> {
        let font = FontBook::new().query("HL-Comic1unicode-Normal")?;
        let renderer = TinySkiaRenderer::new()?;
        for text in [
            "Ắ Ồ Ễ Ự\nTÔI KHÔNG\nTHỂ CHỜ LÂU",
            "A\u{306}\u{301} O\u{302}\u{300}\nCẢNH SÁT\nKẾT TỘI",
        ] {
            let mut layout = TextLayout::new(&font, Some(30.0))
                .with_max_width(240.0)
                .with_max_height(300.0)
                .run(text)?;
            let before = layout.clone();
            let opts = RenderOptions {
                font_size: layout.font_size,
                stroke: Some(RenderStrokeOptions {
                    color: [255; 4],
                    width_px: 3.0,
                }),
                ..Default::default()
            };
            apply(&renderer, &mut layout, &opts, text, "unit", 240.0, 300.0)?;
            assert_eq!(layout.font_size, before.font_size);
            assert_eq!(layout.lines.len(), before.lines.len());
            let mut bottom: Option<f32> = None;
            for (line, original) in layout.lines.iter().zip(&before.lines) {
                assert_eq!(line.range, original.range);
                assert_eq!(line.glyphs.len(), original.glyphs.len());
                for (a, b) in line.glyphs.iter().zip(&original.glyphs) {
                    assert_eq!(
                        (a.glyph_id, a.cluster, a.x_advance, a.y_advance, a.y_offset),
                        (b.glyph_id, b.cluster, b.x_advance, b.y_advance, b.y_offset)
                    );
                }
                let local = probe(&renderer, line, layout.font_size, &opts)?.unwrap();
                let visual_top = local.top as f32 + line.baseline.1;
                let visual_bottom = local.bottom as f32 + line.baseline.1;
                let b = local.shifted(line.baseline.0 as i32, line.baseline.1 as i32);
                if let Some(previous) = bottom {
                    assert!(
                        (visual_top - previous - 0.3f32).abs() < 0.001,
                        "subpixel leading lost"
                    );
                }
                assert!(
                    (b.left + b.right - 240).abs() <= 1,
                    "not centred by rendered bounds"
                );
                assert!(b.left >= 0 && b.right <= 240 && b.top >= 0 && b.bottom <= 300);
                bottom = Some(visual_bottom);
            }
        }
        Ok(())
    }
}

/// Re-shape exactly the already selected lines in the original translation's
/// case. Uppercasing changes UTF-8 byte lengths, so map boundaries explicitly.
pub(crate) fn restyle<'a>(
    layout: &mut LayoutRun<'a>,
    translation: &str,
    font: &'a crate::font::Font,
) -> Result<String> {
    let mut boundaries = std::collections::BTreeMap::new();
    let mut upper = 0;
    boundaries.insert(0, 0);
    for (offset, ch) in translation.char_indices() {
        upper += ch.to_uppercase().map(char::len_utf8).sum::<usize>();
        boundaries.insert(upper, offset + ch.len_utf8());
    }
    let shaper = crate::shape::TextShaper::new();
    for line in &mut layout.lines {
        let start = *boundaries
            .get(&line.range.start)
            .ok_or_else(|| anyhow::anyhow!("line starts inside case expansion"))?;
        let end = *boundaries
            .get(&line.range.end)
            .ok_or_else(|| anyhow::anyhow!("line ends inside case expansion"))?;
        let shaped = shaper.shape(
            &translation[start..end],
            font,
            &crate::shape::ShapingOptions {
                direction: harfrust::Direction::LeftToRight,
                font_size: layout.font_size,
                features: &[],
            },
        )?;
        line.glyphs = shaped.glyphs;
        for glyph in &mut line.glyphs {
            glyph.cluster += start as u32;
        }
        line.advance = shaped.x_advance;
        line.range = start..end;
    }
    Ok(translation.to_owned())
}

#[cfg(test)]
mod face_tests {
    use super::*;
    #[test]
    #[ignore = "requires installed Vietnamese comic fonts"]
    fn new_face_preserves_existing_line_breaks_and_translation() -> Result<()> {
        let mut book = crate::font::FontBook::new();
        let old = book.query("HL-Comic1unicode-Normal")?;
        let new = book.query("ChalkboardSE-Bold")?;
        let text = "Tôi không thể chờ lâu đến thế nữa. Ắ Ồ Ễ Ự ắ ồ ễ ự";
        assert!(
            text.chars()
                .filter(|c| !c.is_whitespace())
                .all(|c| new.has_glyph(c))
        );
        let upper = text.to_uppercase();
        let mut layout = crate::layout::TextLayout::new(&old, Some(30.0))
            .with_max_width(205.0)
            .with_max_height(500.0)
            .run(&upper)?;
        let lines: Vec<_> = layout
            .lines
            .iter()
            .map(|l| upper[l.range.clone()].to_owned())
            .collect();
        let size = layout.font_size;
        assert_eq!(restyle(&mut layout, text, &new)?, text);
        assert_eq!(size, layout.font_size);
        assert_eq!(lines.len(), layout.lines.len());
        for (line, expected) in layout.lines.iter().zip(lines) {
            assert_eq!(text[line.range.clone()].to_uppercase(), expected);
        }
        Ok(())
    }
}
