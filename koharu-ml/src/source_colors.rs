//! Read lettering colours from the original pixels, before inpainting.
use image::{GrayImage, RgbImage};
use koharu_types::{FontPrediction, TextBlock};

fn distance(a: [u8; 3], b: [u8; 3]) -> u32 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| (a as i32 - b as i32).pow(2) as u32)
        .sum()
}

fn median(samples: &[[u8; 3]]) -> [u8; 3] {
    std::array::from_fn(|channel| {
        let mut values: Vec<_> = samples.iter().map(|p| p[channel]).collect();
        let middle = values.len() / 2;
        *values.select_nth_unstable(middle).1
    })
}

/// Measure bounded runs of outline colour starting at a fill/outline edge.
/// Unbounded runs (e.g. an outline merging into white artwork) supply no
/// thickness evidence. Take the shortest direction at each edge so diagonal
/// crossings and gaps between neighbouring glyphs do not inflate the result.
fn outline_width(
    image: &RgbImage,
    mask: &GrayImage,
    bounds: [u32; 4],
    fill: [u8; 3],
    stroke: [u8; 3],
) -> Option<f32> {
    let [x0, y0, x1, y1] = bounds;
    let limit = ((x1 - x0).min(y1 - y0) / 4).clamp(3, 32) as i32;
    let is_stroke = |x: i32, y: i32| {
        if x < 0 || y < 0 || x >= image.width() as i32 || y >= image.height() as i32 {
            return None;
        }
        let p = image.get_pixel(x as u32, y as u32).0;
        Some(distance(p, stroke) < distance(p, fill))
    };
    let mut widths = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            if mask.get_pixel(x, y)[0] < 128 || is_stroke(x as i32, y as i32) != Some(false) {
                continue;
            }
            let mut shortest = None;
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let mut run = 0;
                for step in 1..=limit {
                    let xx = x as i32 + dx * step;
                    let yy = y as i32 + dy * step;
                    let colour = is_stroke(xx, yy);
                    // Segmentation also bounds the outline when the artwork
                    // happens to have the same colour as the stroke.
                    let colour =
                        if colour.is_some() && mask.get_pixel(xx as u32, yy as u32)[0] < 128 {
                            Some(false)
                        } else {
                            colour
                        };
                    match colour {
                        Some(true) => run += 1,
                        Some(false) => {
                            if run > 0 {
                                shortest = Some(shortest.map_or(run, |v: i32| v.min(run)));
                            }
                            break;
                        }
                        None => break,
                    }
                }
            }
            if let Some(width) = shortest {
                widths.push(width);
            }
        }
    }
    if widths.len() < 32 {
        return None;
    }
    let mid = widths.len() / 2;
    Some(*widths.select_nth_unstable(mid).1 as f32)
}

/// Segmentation supplies ownership; its inner pixels supply the palette. The
/// colour enriched at the mask's boundary is the outline, not the fill. Never
/// sample the removal mask: dilation deliberately includes background pixels.
/// Keep the model prediction when there is too little evidence to separate ink.
pub(crate) fn refine(
    image: &RgbImage,
    mask: &GrayImage,
    block: &TextBlock,
    prediction: &mut FontPrediction,
) {
    if image.dimensions() != mask.dimensions() {
        return;
    }
    let (w, h) = image.dimensions();
    let x0 = (block.x.max(1.0) as u32).min(w.saturating_sub(1));
    let y0 = (block.y.max(1.0) as u32).min(h.saturating_sub(1));
    let x1 = ((block.x + block.width).max(0.0) as u32).min(w.saturating_sub(1));
    let y1 = ((block.y + block.height).max(0.0) as u32).min(h.saturating_sub(1));
    let mut core = Vec::new();
    let mut edge = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            if mask.get_pixel(x, y)[0] < 128 {
                continue;
            }
            let inside = (y - 1..=y + 1)
                .all(|yy| (x - 1..=x + 1).all(|xx| mask.get_pixel(xx, yy)[0] >= 128));
            if inside {
                core.push(image.get_pixel(x, y).0);
            } else {
                edge.push(image.get_pixel(x, y).0);
            }
        }
    }
    if core.len() < 32 || edge.len() < 16 {
        return;
    }
    // Two robust palette centres, initialised by the most distant colour from
    // the median. Median updates resist JPEG noise and antialiased edges.
    let first = median(&core);
    let second = *core.iter().max_by_key(|&&p| distance(p, first)).unwrap();
    let mut palette = [first, second];
    let mut groups = [Vec::new(), Vec::new()];
    for _ in 0..12 {
        groups.iter_mut().for_each(Vec::clear);
        for &p in &core {
            let i = usize::from(distance(p, palette[1]) < distance(p, palette[0]));
            groups[i].push(p);
        }
        if groups.iter().any(Vec::is_empty) {
            return;
        }
        let next = [median(&groups[0]), median(&groups[1])];
        if next == palette {
            break;
        }
        palette = next;
    }
    // A second substantial, distinct ink colour is needed to identify an
    // outline. Ambiguous/solid lettering keeps the existing model estimate.
    if distance(palette[0], palette[1]) < 3 * 70 * 70
        || groups.iter().any(|g| g.len() * 100 < core.len() * 15)
    {
        return;
    }
    let edge_first = edge
        .iter()
        .filter(|&&p| distance(p, palette[0]) <= distance(p, palette[1]))
        .count();
    let enrichment =
        edge_first as f32 / edge.len() as f32 - groups[0].len() as f32 / core.len() as f32;
    if enrichment.abs() < 0.10 {
        return;
    }
    let stroke = usize::from(enrichment < 0.0);
    prediction.text_color = palette[1 - stroke];
    prediction.stroke_color = palette[stroke];
    if let Some(width) = outline_width(
        image,
        mask,
        [x0, y0, x1, y1],
        prediction.text_color,
        prediction.stroke_color,
    ) {
        prediction.stroke_width_px = width;
    }
    tracing::info!(block_id = %block.id, text_color = ?prediction.text_color,
        stroke_color = ?prediction.stroke_color, stroke_width_px = prediction.stroke_width_px,
        "colours and outline sampled from original lettering");
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    #[test]
    fn reads_coloured_fill_and_outline_without_sampling_artwork() {
        for (fill, stroke) in [
            ([35, 38, 33], [250, 250, 249]),
            ([190, 30, 45], [20, 35, 100]),
            ([245, 240, 235], [25, 30, 35]),
        ] {
            let mut image = RgbImage::from_pixel(80, 80, Rgb([80, 180, 70]));
            let mut mask = GrayImage::new(80, 80);
            for y in 10..70 {
                for x in 30..50 {
                    mask.put_pixel(x, y, Luma([255]));
                    image.put_pixel(
                        x,
                        y,
                        Rgb(if (34..46).contains(&x) && (14..66).contains(&y) {
                            fill
                        } else {
                            stroke
                        }),
                    );
                }
            }
            let block = TextBlock {
                width: 80.0,
                height: 80.0,
                ..Default::default()
            };
            let mut pred = FontPrediction {
                text_color: [100, 110, 80],
                stroke_color: [160, 170, 90],
                ..Default::default()
            };
            refine(&image, &mask, &block, &mut pred);
            assert_eq!(pred.text_color, fill);
            assert_eq!(pred.stroke_color, stroke);
            assert_eq!(pred.stroke_width_px, 4.0);
        }
    }

    #[test]
    fn measures_different_outline_widths_on_matching_background() {
        for width in [1, 3, 6] {
            let mut image = RgbImage::from_pixel(100, 100, Rgb([250; 3]));
            let mut mask = GrayImage::new(100, 100);
            for y in 20 - width..80 + width {
                for x in 40 - width..60 + width {
                    mask.put_pixel(x, y, Luma([255]));
                    if (40..60).contains(&x) && (20..80).contains(&y) {
                        image.put_pixel(x, y, Rgb([30; 3]));
                    }
                }
            }
            assert_eq!(
                outline_width(&image, &mask, [1, 1, 99, 99], [30; 3], [250; 3]),
                Some(width as f32)
            );
        }
    }

    #[test]
    fn empty_mask_keeps_prediction() {
        let mut pred = FontPrediction {
            text_color: [180, 20, 50],
            ..Default::default()
        };
        refine(
            &RgbImage::new(20, 20),
            &GrayImage::new(20, 20),
            &TextBlock {
                width: 20.0,
                height: 20.0,
                ..Default::default()
            },
            &mut pred,
        );
        assert_eq!(pred.text_color, [180, 20, 50]);
    }
}
