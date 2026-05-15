use std::{io::Cursor, path::PathBuf};

use image::{ImageFormat, RgbaImage, codecs::jpeg::JpegEncoder};
use koharu_types::commands::InpaintRegion;
use koharu_types::{Document, SerializableDynamicImage};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait InpaintRegionExt {
    fn clamp(&self, width: u32, height: u32) -> Option<(u32, u32, u32, u32)>;
}

impl InpaintRegionExt for InpaintRegion {
    fn clamp(&self, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
        if width == 0 || height == 0 {
            return None;
        }
        let x0 = self.x.min(width.saturating_sub(1));
        let y0 = self.y.min(height.saturating_sub(1));
        let x1 = self.x.saturating_add(self.width).min(width).max(x0);
        let y1 = self.y.saturating_add(self.height).min(height).max(y0);
        let w = x1.saturating_sub(x0);
        let h = y1.saturating_sub(y0);
        if w == 0 || h == 0 {
            return None;
        }
        Some((x0, y0, w, h))
    }
}

pub(crate) fn encode_image(image: &SerializableDynamicImage, ext: &str) -> anyhow::Result<Vec<u8>> {
    let format = ImageFormat::from_extension(ext).unwrap_or(ImageFormat::Jpeg);
    match format {
        ImageFormat::Jpeg => {
            let mut buf = Vec::new();
            let mut cursor = Cursor::new(&mut buf);
            let encoder = JpegEncoder::new_with_quality(&mut cursor, 95);
            image.0.write_with_encoder(encoder)?;
            Ok(buf)
        }
        _ => {
            let mut buf = Vec::new();
            let mut cursor = Cursor::new(&mut buf);
            image.0.write_to(&mut cursor, format)?;
            Ok(buf)
        }
    }
}

/// Encode an image and embed DPI metadata into the output bytes.
/// Supports JPEG (JFIF APP0 injection) and PNG (pHYs chunk injection).
pub(crate) fn encode_image_with_dpi(
    image: &SerializableDynamicImage,
    ext: &str,
    dpi: u32,
) -> anyhow::Result<Vec<u8>> {
    // TIFF: use our own flat writer which embeds DPI directly in IFD tags.
    if ext == "tif" || ext == "tiff" {
        return Ok(koharu_tiff::encode_flat(image, if dpi == 0 { 72 } else { dpi }));
    }
    let bytes = encode_image(image, ext)?;
    if dpi == 0 {
        return Ok(bytes);
    }
    Ok(match ext {
        "jpg" | "jpeg" => inject_dpi_jpeg(bytes, dpi),
        "png" => inject_dpi_png(bytes, dpi),
        _ => bytes,
    })
}

/// Inject a JFIF APP0 marker with DPI right after the JPEG SOI (FF D8).
/// If the encoder already wrote a JFIF APP0, it will be superseded by our marker
/// since most parsers use the first APP0 they encounter.
fn inject_dpi_jpeg(bytes: Vec<u8>, dpi: u32) -> Vec<u8> {
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return bytes;
    }
    let dpi = dpi.min(65535) as u16;
    // JFIF APP0: FF E0, length=16, "JFIF\0", v1.01, unit=1(DPI), xdpi, ydpi, 0, 0
    let app0: [u8; 18] = [
        0xFF, 0xE0,
        0x00, 0x10, // segment length = 16 (includes length field but not marker)
        b'J', b'F', b'I', b'F', 0x00, // identifier
        0x01, 0x01, // version 1.01
        0x01,       // units = 1 (pixels per inch)
        (dpi >> 8) as u8, (dpi & 0xFF) as u8, // X density
        (dpi >> 8) as u8, (dpi & 0xFF) as u8, // Y density
        0x00, 0x00, // thumbnail size (none)
    ];
    let mut out = Vec::with_capacity(2 + app0.len() + bytes.len() - 2);
    out.extend_from_slice(&bytes[0..2]); // SOI
    out.extend_from_slice(&app0);
    out.extend_from_slice(&bytes[2..]); // rest of JPEG
    out
}

/// Inject a pHYs chunk into PNG bytes right after the IHDR chunk.
/// pHYs: pixels per unit X, pixels per unit Y, unit = 1 (metre).
fn inject_dpi_png(bytes: Vec<u8>, dpi: u32) -> Vec<u8> {
    const PNG_SIG: &[u8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 8 || &bytes[0..8] != PNG_SIG {
        return bytes;
    }
    // Pixels per metre = dpi * 39.3701 (rounded)
    let ppm = (dpi as f64 * 39.3701).round() as u32;
    // pHYs chunk data: 4 bytes x ppm + 4 bytes y ppm + 1 byte unit
    let mut phys_data = Vec::with_capacity(9);
    phys_data.extend_from_slice(&ppm.to_be_bytes());
    phys_data.extend_from_slice(&ppm.to_be_bytes());
    phys_data.push(1u8); // unit = metre

    // CRC covers chunk type + data
    let mut crc_input = Vec::with_capacity(4 + 9);
    crc_input.extend_from_slice(b"pHYs");
    crc_input.extend_from_slice(&phys_data);
    let crc = png_crc32(&crc_input);

    // Build complete pHYs chunk: length(4) + type(4) + data(9) + crc(4) = 21 bytes
    let mut phys_chunk = Vec::with_capacity(21);
    phys_chunk.extend_from_slice(&(phys_data.len() as u32).to_be_bytes()); // length = 9
    phys_chunk.extend_from_slice(b"pHYs");
    phys_chunk.extend_from_slice(&phys_data);
    phys_chunk.extend_from_slice(&crc.to_be_bytes());

    // Find end of IHDR chunk (8 byte sig + 4 len + 4 type + 13 data + 4 crc = 33 bytes)
    let ihdr_end = 8 + 4 + 4 + 13 + 4; // = 33
    if bytes.len() < ihdr_end {
        return bytes;
    }
    let mut out = Vec::with_capacity(bytes.len() + phys_chunk.len());
    out.extend_from_slice(&bytes[..ihdr_end]);
    out.extend_from_slice(&phys_chunk);
    out.extend_from_slice(&bytes[ihdr_end..]);
    out
}

fn png_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &b in data {
        let mut c = crc ^ (b as u32);
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
        crc = c;
    }
    !crc
}

pub(crate) fn mime_from_ext(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

pub(crate) fn blank_rgba(
    width: u32,
    height: u32,
    color: image::Rgba<u8>,
) -> SerializableDynamicImage {
    let blank = RgbaImage::from_pixel(width, height, color);
    image::DynamicImage::ImageRgba8(blank).into()
}

pub fn load_documents(inputs: Vec<(PathBuf, Vec<u8>)>) -> anyhow::Result<Vec<Document>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }

    let mut documents: Vec<_> = inputs
        .into_par_iter()
        .filter_map(|(path, bytes)| match Document::from_bytes(path, bytes) {
            Ok(docs) => Some(docs),
            Err(err) => {
                tracing::warn!(?err, "Failed to parse document");
                None
            }
        })
        .flatten()
        .inspect(|doc| {
            tracing::info!(
                name = %doc.name,
                resolution_dpi = doc.resolution_dpi,
                "document loaded"
            );
        })
        .collect();

    documents.sort_by_key(|doc| doc.name.clone());
    Ok(documents)
}
