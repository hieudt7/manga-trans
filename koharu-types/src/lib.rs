pub mod commands;
pub mod events;
pub mod parse;
pub mod protocol;
pub mod views;

mod effect;
mod font;
mod image;

pub use commands::*;
pub use effect::TextShaderEffect;
pub use events::*;
pub use font::{FontPrediction, NamedFontPrediction, TextDirection};
pub use image::SerializableDynamicImage;
pub use protocol::*;

use std::{path::PathBuf, sync::Arc};

use ::image::GenericImageView;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use ts_rs::TS;
use uuid::Uuid;

fn new_text_block_id() -> String {
    Uuid::new_v4().to_string()
}

fn default_resolution() -> u32 {
    72
}

/// Parse DPI from an EXIF block (the bytes after the "Exif\0\0" header).
/// Reads XResolution (tag 0x011A), YResolution (0x011B), ResolutionUnit (0x0128)
/// from the TIFF IFD embedded in the EXIF segment. Returns None if not found.
fn extract_dpi_from_exif(exif: &[u8]) -> Option<u32> {
    if exif.len() < 8 {
        return None;
    }
    // TIFF header: byte order mark ("II" = LE, "MM" = BE) + magic 42 + IFD offset
    let little_endian = match &exif[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let read_u16 = |buf: &[u8], off: usize| -> Option<u16> {
        if off + 2 > buf.len() { return None; }
        Some(if little_endian {
            u16::from_le_bytes([buf[off], buf[off + 1]])
        } else {
            u16::from_be_bytes([buf[off], buf[off + 1]])
        })
    };
    let read_u32 = |buf: &[u8], off: usize| -> Option<u32> {
        if off + 4 > buf.len() { return None; }
        Some(if little_endian {
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        } else {
            u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        })
    };

    if read_u16(exif, 2)? != 42 {
        return None; // not a valid TIFF
    }
    let ifd_offset = read_u32(exif, 4)? as usize;
    if ifd_offset + 2 > exif.len() {
        return None;
    }
    let entry_count = read_u16(exif, ifd_offset)? as usize;

    let mut x_res_num: Option<u32> = None;
    let mut x_res_den: Option<u32> = None;
    let mut res_unit: u32 = 2; // default = inch

    for e in 0..entry_count {
        let entry_off = ifd_offset + 2 + e * 12;
        if entry_off + 12 > exif.len() {
            break;
        }
        let tag = read_u16(exif, entry_off)?;
        let value_off = entry_off + 8;

        match tag {
            0x011A => {
                // XResolution: RATIONAL (numerator/denominator), both u32
                let offset = read_u32(exif, value_off)? as usize;
                x_res_num = read_u32(exif, offset);
                x_res_den = read_u32(exif, offset + 4);
            }
            0x0128 => {
                // ResolutionUnit: SHORT — 1=no unit, 2=inch, 3=cm
                res_unit = read_u16(exif, value_off)? as u32;
            }
            _ => {}
        }
    }

    let num = x_res_num?;
    let den = x_res_den.unwrap_or(1).max(1);
    let resolution = num as f64 / den as f64;
    if resolution <= 0.0 {
        return None;
    }
    let dpi = match res_unit {
        2 => resolution,               // already in inches
        3 => resolution * 2.54,        // cm → inch
        _ => return None,
    };
    Some(dpi.round() as u32)
}

/// Try to extract DPI from raw image bytes.
/// Handles JPEG (JFIF APP0, EXIF APP1), PNG (pHYs), and TIFF (IFD tags). Falls back to 72.
fn extract_dpi(bytes: &[u8]) -> u32 {
    // TIFF: header is a TIFF IFD — reuse the same EXIF parser directly.
    // "II" = little-endian (magic = 42 LE), "MM" = big-endian (magic = 42 BE).
    if bytes.len() >= 8 {
        let is_le_tiff = &bytes[0..2] == b"II" && bytes[2] == 42 && bytes[3] == 0;
        let is_be_tiff = &bytes[0..2] == b"MM" && bytes[2] == 0  && bytes[3] == 42;
        if is_le_tiff || is_be_tiff {
            if let Some(dpi) = extract_dpi_from_exif(bytes) {
                return dpi;
            }
        }
    }

    // JPEG: scan APP markers for JFIF APP0 or EXIF APP1
    if bytes.len() >= 20 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        let mut i = 2;
        while i + 3 < bytes.len() {
            if bytes[i] != 0xFF {
                break;
            }
            let marker = bytes[i + 1];
            if i + 3 >= bytes.len() {
                break;
            }
            let seg_len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            // APP0 = 0xE0 — JFIF header
            if marker == 0xE0 && i + 13 < bytes.len() && &bytes[i + 4..i + 9] == b"JFIF\0" {
                let unit = bytes[i + 11];
                let xdpi = u16::from_be_bytes([bytes[i + 12], bytes[i + 13]]) as u32;
                if xdpi > 0 {
                    return match unit {
                        1 => xdpi,                                  // pixels per inch
                        2 => (xdpi as f64 * 2.54).round() as u32, // pixels per cm → ppi
                        _ => 72,
                    };
                }
            }
            // APP1 = 0xE1 — EXIF data
            if marker == 0xE1 && seg_len >= 8 && i + 4 + 6 <= bytes.len()
                && &bytes[i + 4..i + 10] == b"Exif\0\0"
            {
                if let Some(dpi) = extract_dpi_from_exif(&bytes[i + 10..i + 2 + seg_len]) {
                    return dpi;
                }
            }
            if seg_len < 2 {
                break;
            }
            i += 2 + seg_len;
        }
    }
    // PNG: look for pHYs chunk (pixels per unit)
    if bytes.len() >= 8 && &bytes[0..8] == b"\x89PNG\r\n\x1a\n" {
        let mut i = 8;
        while i + 12 <= bytes.len() {
            let chunk_len = u32::from_be_bytes([bytes[i], bytes[i+1], bytes[i+2], bytes[i+3]]) as usize;
            let chunk_type = &bytes[i+4..i+8];
            if chunk_type == b"pHYs" && i + 12 + chunk_len <= bytes.len() && chunk_len >= 9 {
                let xpu = u32::from_be_bytes([bytes[i+8], bytes[i+9], bytes[i+10], bytes[i+11]]);
                let unit = bytes[i + 16]; // 1 = metre
                if unit == 1 && xpu > 0 {
                    return (xpu as f64 / 39.3701).round() as u32;
                }
            }
            if chunk_type == b"IDAT" || chunk_type == b"IEND" {
                break;
            }
            i += 12 + chunk_len;
        }
    }
    72
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct BalloonDetection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextBlock {
    #[serde(default = "new_text_block_id")]
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub line_polygons: Option<Vec<[[f32; 2]; 4]>>,
    pub source_direction: Option<TextDirection>,
    pub rendered_direction: Option<TextDirection>,
    pub source_language: Option<String>,
    pub rotation_deg: Option<f32>,
    pub detected_font_size_px: Option<f32>,
    pub detector: Option<String>,
    pub text: Option<String>,
    pub translation: Option<String>,
    pub style: Option<TextStyle>,
    pub font_prediction: Option<FontPrediction>,
    pub rendered: Option<SerializableDynamicImage>,
    #[serde(skip)]
    pub lock_layout_box: bool,
    #[serde(skip)]
    pub layout_seed_x: Option<f32>,
    #[serde(skip)]
    pub layout_seed_y: Option<f32>,
    #[serde(skip)]
    pub layout_seed_width: Option<f32>,
    #[serde(skip)]
    pub layout_seed_height: Option<f32>,
}

impl TextBlock {
    pub fn ensure_id(&mut self) {
        if self.id.trim().is_empty() {
            self.id = new_text_block_id();
        }
    }

    pub fn set_layout_seed(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.layout_seed_x = Some(x);
        self.layout_seed_y = Some(y);
        self.layout_seed_width = Some(width.max(1.0));
        self.layout_seed_height = Some(height.max(1.0));
    }

    pub fn seed_layout_box(&mut self) -> (f32, f32, f32, f32) {
        match (
            self.layout_seed_x,
            self.layout_seed_y,
            self.layout_seed_width,
            self.layout_seed_height,
        ) {
            (Some(x), Some(y), Some(width), Some(height))
                if width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0 =>
            {
                (x, y, width, height)
            }
            _ => {
                self.set_layout_seed(self.x, self.y, self.width, self.height);
                (self.x, self.y, self.width.max(1.0), self.height.max(1.0))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct TextStrokeStyle {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_stroke_color")]
    pub color: [u8; 4],
    #[serde(default)]
    pub width_px: Option<f32>,
}

impl Default for TextStrokeStyle {
    fn default() -> Self {
        Self {
            enabled: true,
            color: [255, 255, 255, 255],
            width_px: None,
        }
    }
}

const fn default_true() -> bool {
    true
}

const fn default_stroke_color() -> [u8; 4] {
    [255, 255, 255, 255]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct TextStyle {
    pub font_families: Vec<String>,
    pub font_size: Option<f32>,
    pub color: [u8; 4],
    pub effect: Option<TextShaderEffect>,
    pub stroke: Option<TextStrokeStyle>,
    #[serde(default)]
    pub text_align: Option<TextAlign>,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    pub id: String,
    pub path: PathBuf,
    pub name: String,
    pub image: SerializableDynamicImage,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub revision: u64,
    /// Resolution in pixels per inch (DPI). Read from the source image on load.
    #[serde(default = "default_resolution")]
    pub resolution_dpi: u32,
    pub text_blocks: Vec<TextBlock>,
    pub segment: Option<SerializableDynamicImage>,
    pub inpainted: Option<SerializableDynamicImage>,
    pub rendered: Option<SerializableDynamicImage>,
    pub brush_layer: Option<SerializableDynamicImage>,
    #[serde(default)]
    pub balloons: Vec<BalloonDetection>,
}

impl Document {
    pub fn open(path: PathBuf) -> anyhow::Result<Self> {
        let bytes = std::fs::read(&path)?;

        let documents = Self::from_bytes(path, bytes)?;
        documents
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No document found in file"))
    }

    pub fn from_bytes(path: impl Into<PathBuf>, bytes: Vec<u8>) -> anyhow::Result<Vec<Self>> {
        let path = path.into();
        Ok(vec![Self::image(path, bytes)?])
    }

    fn image(path: PathBuf, bytes: Vec<u8>) -> anyhow::Result<Self> {
        let img = ::image::load_from_memory(&bytes)?;
        let (width, height) = img.dimensions();
        let id = blake3::hash(&bytes).to_hex().to_string();
        let name = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let resolution_dpi = extract_dpi(&bytes);
        Ok(Document {
            id,
            path,
            name,
            image: SerializableDynamicImage(img),
            width,
            height,
            resolution_dpi,
            ..Default::default()
        })
    }

    pub fn ensure_text_block_ids(&mut self) {
        for block in &mut self.text_blocks {
            block.ensure_id();
        }
    }

    pub fn bump_revision(&mut self) {
        self.revision = self.revision.saturating_add(1);
    }

    pub fn prepare_for_store(&mut self) {
        self.ensure_text_block_ids();
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub documents: Vec<Document>,
    pub folder_session: Option<FolderSession>,
}

pub type AppState = Arc<RwLock<State>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderFile {
    pub path: PathBuf,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub has_result: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderSession {
    pub root: PathBuf,
    pub result_dir: PathBuf,
    pub files: Vec<FolderFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderFileInfo {
    pub index: usize,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub has_result: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderSessionInfo {
    pub root: String,
    pub result_dir: String,
    pub files: Vec<FolderFileInfo>,
}

#[cfg(test)]
mod tests {
    use super::TextBlock;

    #[test]
    fn seed_layout_box_stays_stable_until_explicit_reset() {
        let mut block = TextBlock {
            x: 10.0,
            y: 20.0,
            width: 30.0,
            height: 40.0,
            ..Default::default()
        };

        let first = block.seed_layout_box();
        assert_eq!(first, (10.0, 20.0, 30.0, 40.0));

        block.x = 100.0;
        block.y = 200.0;
        block.width = 300.0;
        block.height = 400.0;

        let second = block.seed_layout_box();
        assert_eq!(second, first);

        block.set_layout_seed(block.x, block.y, block.width, block.height);
        let third = block.seed_layout_box();
        assert_eq!(third, (100.0, 200.0, 300.0, 400.0));
    }
}
