use std::{path::PathBuf, sync::Mutex};

use anyhow::Result;
use image::{DynamicImage, GrayImage, Luma, imageops};
use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;

const INPUT_SIZE: u32 = 640;
const PROTO_SIZE: usize = 160;
const CONF_THRESHOLD: f32 = 0.5;
const IOU_THRESHOLD: f32 = 0.45;
const MASK_THRESHOLD: f32 = 0.5;

fn local_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("models")
        .join("bubble_detector.onnx")
}

pub fn is_available() -> bool {
    local_model_path().exists()
}

#[derive(Clone)]
pub struct BubbleBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
    /// Binary segmentation mask at original image resolution (255 = inside balloon).
    /// None if mask decoding failed — callers must handle this gracefully.
    pub mask: Option<GrayImage>,
}

struct Candidate {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    score: f32,
    anchor_idx: usize,
}

pub struct ComicBubbleDetector {
    session: Mutex<Session>,
}

impl ComicBubbleDetector {
    pub async fn load() -> Result<Self> {
        let path = local_model_path();
        if !path.exists() {
            anyhow::bail!(
                "Bubble detector model not found at {}. Run tools/export_bubble_detector.py first.",
                path.display()
            );
        }
        let session = tokio::task::spawn_blocking(move || -> Result<Session> {
            Ok(Session::builder()?.commit_from_file(&path)?)
        })
        .await??;
        Ok(Self { session: Mutex::new(session) })
    }

    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<BubbleBox>> {
        let orig_w = image.width();
        let orig_h = image.height();
        let scale_x = orig_w as f32 / INPUT_SIZE as f32;
        let scale_y = orig_h as f32 / INPUT_SIZE as f32;

        // Resize to 640×640, normalize [0,1], NCHW
        let resized = image.resize_exact(INPUT_SIZE, INPUT_SIZE, imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let (w, h) = (INPUT_SIZE as usize, INPUT_SIZE as usize);
        let mut data = vec![0f32; 3 * h * w];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let base = y as usize * w + x as usize;
            data[base] = pixel[0] as f32 / 255.0;
            data[h * w + base] = pixel[1] as f32 / 255.0;
            data[2 * h * w + base] = pixel[2] as f32 / 255.0;
        }

        let array = Array::from_shape_vec([1usize, 3, h, w], data)?;
        let input_tensor = Tensor::from_array(array)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("ComicBubbleDetector mutex poisoned"))?;

        let input_name = session.inputs()[0].name().to_string();
        let outputs = session.run(ort::inputs! { input_name.as_str() => input_tensor })?;

        // output0: [1, 37, 8400] — cx,cy,w,h,conf,mask_coef×32
        let out0 = outputs[0].try_extract_array::<f32>()?;
        let out0 = out0.view();

        // output1: [1, 32, 160, 160] — prototype masks (only present in seg models)
        let proto_masks = outputs[1].try_extract_array::<f32>().ok();

        let num_anchors = out0.shape()[2];

        // Pass 1: filter by conf, record anchor index for mask decoding
        let mut candidates: Vec<Candidate> = Vec::new();
        for i in 0..num_anchors {
            let conf = out0[[0, 4, i]];
            if conf < CONF_THRESHOLD {
                continue;
            }
            let cx = out0[[0, 0, i]];
            let cy = out0[[0, 1, i]];
            let bw = out0[[0, 2, i]];
            let bh = out0[[0, 3, i]];
            let x = ((cx - bw / 2.0) * scale_x).max(0.0);
            let y = ((cy - bh / 2.0) * scale_y).max(0.0);
            let width = (bw * scale_x).min(orig_w as f32 - x);
            let height = (bh * scale_y).min(orig_h as f32 - y);
            candidates.push(Candidate { x, y, width, height, score: conf, anchor_idx: i });
        }

        // NMS
        let kept = nms(candidates, IOU_THRESHOLD);

        let boxes: Vec<BubbleBox> = kept
            .into_iter()
            .map(|c| {
                let mask = proto_masks.as_ref().map(|protos| {
                    decode_mask(&out0, protos.view(), c.anchor_idx, orig_w, orig_h)
                });
                BubbleBox { x: c.x, y: c.y, width: c.width, height: c.height, score: c.score, mask }
            })
            .collect();

        tracing::debug!(count = boxes.len(), "bubble detector output");
        Ok(boxes)
    }
}

/// Decode the segmentation mask for one anchor from the YOLO prototype tensors.
/// Returns a binary GrayImage (255 = inside balloon) at original image resolution.
fn decode_mask(
    out0: &ndarray::ArrayViewD<f32>,
    out1: ndarray::ArrayViewD<f32>,
    anchor_idx: usize,
    orig_w: u32,
    orig_h: u32,
) -> GrayImage {
    let n = PROTO_SIZE * PROTO_SIZE;
    let mut logits = vec![0.0f32; n];

    // mask_logit[p] = Σ_k  coef[k] * proto[k, p]
    for k in 0..32 {
        let coef = out0[[0, 5 + k, anchor_idx]];
        for py in 0..PROTO_SIZE {
            for px in 0..PROTO_SIZE {
                logits[py * PROTO_SIZE + px] += coef * out1[[0, k, py, px]];
            }
        }
    }

    let mut mask_160 = GrayImage::new(PROTO_SIZE as u32, PROTO_SIZE as u32);
    for py in 0..PROTO_SIZE {
        for px in 0..PROTO_SIZE {
            let p = sigmoid(logits[py * PROTO_SIZE + px]);
            mask_160.put_pixel(px as u32, py as u32, Luma([if p > MASK_THRESHOLD { 255u8 } else { 0u8 }]));
        }
    }

    imageops::resize(&mask_160, orig_w, orig_h, imageops::FilterType::Nearest)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn nms(mut candidates: Vec<Candidate>, iou_threshold: f32) -> Vec<Candidate> {
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut suppressed = vec![false; candidates.len()];
    for i in 0..candidates.len() {
        if suppressed[i] { continue; }
        for j in (i + 1)..candidates.len() {
            if suppressed[j] { continue; }
            if iou_cand(&candidates[i], &candidates[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    candidates.into_iter().enumerate().filter(|(i, _)| !suppressed[*i]).map(|(_, c)| c).collect()
}

fn iou_cand(a: &Candidate, b: &Candidate) -> f32 {
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = (a.x + a.width).min(b.x + b.width);
    let iy2 = (a.y + a.height).min(b.y + b.height);
    if ix2 <= ix1 || iy2 <= iy1 { return 0.0; }
    let inter = (ix2 - ix1) * (iy2 - iy1);
    inter / (a.width * a.height + b.width * b.height - inter)
}

// ─── Mask pipeline utilities (used by facade) ────────────────────────────────

/// Square-kernel morphological erosion (separable: horizontal then vertical pass).
pub fn erode_binary(mask: &GrayImage, radius: u32) -> GrayImage {
    if radius == 0 { return mask.clone(); }
    let (w, h) = mask.dimensions();
    let mut tmp = GrayImage::new(w, h);
    // Horizontal
    for y in 0..h {
        for x in 0..w {
            let x1 = x.saturating_sub(radius);
            let x2 = (x + radius + 1).min(w);
            let min = (x1..x2).map(|nx| mask.get_pixel(nx, y)[0]).min().unwrap_or(0);
            tmp.put_pixel(x, y, Luma([min]));
        }
    }
    let mut out = GrayImage::new(w, h);
    // Vertical
    for y in 0..h {
        for x in 0..w {
            let y1 = y.saturating_sub(radius);
            let y2 = (y + radius + 1).min(h);
            let min = (y1..y2).map(|ny| tmp.get_pixel(x, ny)[0]).min().unwrap_or(0);
            out.put_pixel(x, y, Luma([min]));
        }
    }
    out
}

/// BFS L1 distance transform: each pixel = Manhattan distance to nearest background (value < 128).
pub fn distance_transform(mask: &GrayImage) -> Vec<f32> {
    let (w, h) = mask.dimensions();
    let n = (w * h) as usize;
    let mut dist = vec![f32::MAX; n];
    let mut queue = std::collections::VecDeque::with_capacity(n / 4);

    for y in 0..h {
        for x in 0..w {
            if mask.get_pixel(x, y)[0] < 128 {
                let idx = (y * w + x) as usize;
                dist[idx] = 0.0;
                queue.push_back((x, y));
            }
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        let d = dist[(y * w + x) as usize] + 1.0;
        for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let idx = (ny as u32 * w + nx as u32) as usize;
                if d < dist[idx] {
                    dist[idx] = d;
                    queue.push_back((nx as u32, ny as u32));
                }
            }
        }
    }
    dist
}

/// Fraction of pixels in rect (tx,ty,tw,th) that are foreground in `mask`.
pub fn fill_ratio(mask: &GrayImage, tx: f32, ty: f32, tw: f32, th: f32) -> f32 {
    let (mw, mh) = mask.dimensions();
    let x1 = tx as u32;
    let y1 = ty as u32;
    let x2 = ((tx + tw) as u32).min(mw);
    let y2 = ((ty + th) as u32).min(mh);
    if x2 <= x1 || y2 <= y1 { return 0.0; }
    let total = (x2 - x1) * (y2 - y1);
    let inside: u32 = (y1..y2).flat_map(|y| (x1..x2).map(move |x| (x, y)))
        .filter(|&(x, y)| mask.get_pixel(x, y)[0] >= 128)
        .count() as u32;
    inside as f32 / total as f32
}
