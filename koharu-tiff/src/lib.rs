use std::io::Write;

use image::RgbaImage;
use koharu_types::{Document, SerializableDynamicImage};

pub use koharu_psd::{PsdExportOptions, TextLayerMode};

#[derive(Debug, thiserror::Error)]
pub enum TiffExportError {
    #[error("PSD layer error: {0}")]
    Psd(#[from] koharu_psd::PsdExportError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Encode a single image as a flat TIFF with the given DPI.
/// No Photoshop layer data — suitable for plain image export.
pub fn encode_flat(image: &SerializableDynamicImage, resolution_dpi: u32) -> Vec<u8> {
    let rgba = image.0.to_rgba8();
    let mut out = Vec::new();
    write_flat_tiff(&mut out, &rgba, resolution_dpi).expect("write_flat_tiff failed");
    out
}

/// Export a Document as a TIFF file.
///
/// The output is valid for any TIFF viewer (shows the flattened RGBA composite)
/// and carries Photoshop layer data in tag 37724 (ImageSourceData) so that
/// Photoshop opens it as a multi-layer document with editable text layers.
pub fn export_document(
    document: &Document,
    options: &PsdExportOptions,
) -> Result<Vec<u8>, TiffExportError> {
    let mut bytes = Vec::new();
    write_document(&mut bytes, document, options)?;
    Ok(bytes)
}

pub fn write_document<W: Write>(
    writer: W,
    document: &Document,
    options: &PsdExportOptions,
) -> Result<(), TiffExportError> {
    let (layer_info, composite, resolution_dpi) = koharu_psd::export_layer_data(document, options)?;
    write_tiff(writer, &composite, &layer_info, resolution_dpi)?;
    Ok(())
}

// ─── TIFF writer ─────────────────────────────────────────────────────────────

/// Write a minimal flat TIFF (13 IFD entries, no Photoshop ISD).
/// Suitable for plain image export where layer data is not needed.
fn write_flat_tiff<W: Write>(mut w: W, image: &RgbaImage, resolution_dpi: u32) -> std::io::Result<()> {
    let width = image.width();
    let height = image.height();
    let image_byte_count = width as usize * height as usize * 4;

    const N_ENTRIES: u32 = 13;
    let ifd_offset: u32 = 8;
    let ifd_size: u32 = 2 + N_ENTRIES * 12 + 4;
    let data_start: u32 = ifd_offset + ifd_size;

    let bits_offset: u32 = data_start;
    let xres_offset: u32 = bits_offset + 8;
    let yres_offset: u32 = xres_offset + 8;
    let image_offset: u32 = yres_offset + 8;

    w.write_all(b"II")?;
    w16(&mut w, 42)?;
    w32(&mut w, ifd_offset)?;

    w16(&mut w, N_ENTRIES as u16)?;
    ifd_long (&mut w, 256, 1, width)?;
    ifd_long (&mut w, 257, 1, height)?;
    ifd_off  (&mut w, 258, 3, 4, bits_offset)?;
    ifd_short(&mut w, 259, 1, 1)?;
    ifd_short(&mut w, 262, 1, 2)?;
    ifd_long (&mut w, 273, 1, image_offset)?;
    ifd_short(&mut w, 277, 1, 4)?;
    ifd_long (&mut w, 278, 1, height)?;
    ifd_long (&mut w, 279, 1, image_byte_count as u32)?;
    ifd_off  (&mut w, 282, 5, 1, xres_offset)?;
    ifd_off  (&mut w, 283, 5, 1, yres_offset)?;
    ifd_short(&mut w, 296, 1, 2)?;
    ifd_short(&mut w, 338, 1, 2)?;
    w32(&mut w, 0)?;

    for _ in 0..4 { w16(&mut w, 8)?; }
    w32(&mut w, resolution_dpi)?; w32(&mut w, 1)?;
    w32(&mut w, resolution_dpi)?; w32(&mut w, 1)?;
    w.write_all(image.as_raw())?;
    Ok(())
}

/// Write a little-endian TIFF file with:
///   - One IFD containing the flattened RGBA composite (for OS-level viewers).
///   - Tag 37724 (ImageSourceData) containing the Photoshop layer stack so that
///     Photoshop opens the file as a fully editable multi-layer document.
///
/// Tag 37724 format (from Adobe Photoshop TIFF Technical Notes + psdtags spec):
///   b"Adobe Photoshop Document Data Block\0"   ← 36-byte magic
///   [8BIM blocks with 4-char keys, big-endian size fields]
///   · "8BIM" + "Layr" + u32BE(size) + layer_info_bytes + [pad to 4]
///   · "8BIM" + "LMsk" + u32BE(4)   + [0x00 × 4]        (empty user mask)
///
/// layer_info_bytes = [i16 layer_count] + [layer records] + [channel image data]
/// (big-endian, same encoding as PSD Section 4 inner content)
fn write_tiff<W: Write>(
    mut w: W,
    image: &RgbaImage,
    layer_info: &[u8],
    resolution_dpi: u32,
) -> std::io::Result<()> {
    let width = image.width();
    let height = image.height();
    let image_byte_count = width as usize * height as usize * 4; // RGBA uncompressed

    // ── Build tag 37724 (ImageSourceData) payload ─────────────────────────────
    let isd = build_image_source_data(layer_info);

    // ── Layout ────────────────────────────────────────────────────────────────
    // [0]   TIFF header                      8 bytes
    // [8]   IFD (N_ENTRIES entries)          2 + N*12 + 4 bytes
    // data area:
    //   bits_per_sample_offset               8 bytes  (4 × u16)
    //   xres_offset                          8 bytes  (rational 72/1)
    //   yres_offset                          8 bytes  (rational 72/1)
    //   isd_offset                           isd.len() bytes
    //   image_offset                         image_byte_count bytes

    const N_ENTRIES: u32 = 14;
    let ifd_offset: u32 = 8;
    let ifd_size: u32 = 2 + N_ENTRIES * 12 + 4;
    let data_start: u32 = ifd_offset + ifd_size;

    let bits_offset: u32 = data_start;
    let xres_offset: u32 = bits_offset + 8;
    let yres_offset: u32 = xres_offset + 8;
    let isd_offset: u32  = yres_offset + 8;
    let image_offset: u32 = isd_offset + isd.len() as u32;

    // ── TIFF header ───────────────────────────────────────────────────────────
    w.write_all(b"II")?;        // little-endian byte order
    w16(&mut w, 42)?;           // TIFF magic
    w32(&mut w, ifd_offset)?;   // offset to first IFD

    // ── IFD (tags must be sorted ascending by tag number) ─────────────────────
    w16(&mut w, N_ENTRIES as u16)?;

    ifd_long (&mut w, 256, 1, width)?;                                  // ImageWidth
    ifd_long (&mut w, 257, 1, height)?;                                 // ImageLength
    ifd_off  (&mut w, 258, 3 /*SHORT*/, 4, bits_offset)?;              // BitsPerSample
    ifd_short(&mut w, 259, 1, 1)?;                                      // Compression: none
    ifd_short(&mut w, 262, 1, 2)?;                                      // PhotometricInterp: RGB
    ifd_long (&mut w, 273, 1, image_offset)?;                           // StripOffsets
    ifd_short(&mut w, 277, 1, 4)?;                                      // SamplesPerPixel
    ifd_long (&mut w, 278, 1, height)?;                                 // RowsPerStrip
    ifd_long (&mut w, 279, 1, image_byte_count as u32)?;                // StripByteCounts
    ifd_off  (&mut w, 282, 5 /*RATIONAL*/, 1, xres_offset)?;           // XResolution
    ifd_off  (&mut w, 283, 5 /*RATIONAL*/, 1, yres_offset)?;           // YResolution
    ifd_short(&mut w, 296, 1, 2)?;                                      // ResolutionUnit: inch
    ifd_short(&mut w, 338, 1, 2)?;                                      // ExtraSamples: unassoc. alpha
    // Tag 37724 = 0x935C — ImageSourceData (Photoshop layers)
    ifd_off  (&mut w, 37724, 7 /*UNDEFINED*/, isd.len() as u32, isd_offset)?;

    w32(&mut w, 0)?; // next IFD offset = 0 (no more IFDs)

    // ── Data area ─────────────────────────────────────────────────────────────
    // BitsPerSample [8, 8, 8, 8]
    for _ in 0..4 { w16(&mut w, 8)?; }
    // XResolution: resolution_dpi / 1
    w32(&mut w, resolution_dpi)?; w32(&mut w, 1)?;
    // YResolution: resolution_dpi / 1
    w32(&mut w, resolution_dpi)?; w32(&mut w, 1)?;
    // ImageSourceData
    w.write_all(&isd)?;
    // Image data: raw RGBA bytes, row-major, uncompressed
    w.write_all(image.as_raw())?;

    Ok(())
}

// ─── ImageSourceData builder (tag 37724) ─────────────────────────────────────

/// Build the value for TIFF tag 37724 (ImageSourceData).
///
/// Structure:
///   SIGNATURE (36 bytes, null-terminated)
///   "8BIM" + "Layr" + u32BE(len) + layer_info + [pad]
///   "8BIM" + "LMsk" + u32BE(4)   + [0 × 4]    (empty user mask)
fn build_image_source_data(layer_info: &[u8]) -> Vec<u8> {
    const SIGNATURE: &[u8] = b"Adobe Photoshop Document Data Block\0";

    let layr_block = isd_block(b"Layr", layer_info);
    // empty LMsk (user mask) block — 4 zero bytes
    let lmsk_block = isd_block(b"LMsk", &[0u8; 4]);

    let mut data = Vec::with_capacity(SIGNATURE.len() + layr_block.len() + lmsk_block.len());
    data.extend_from_slice(SIGNATURE);
    data.extend_from_slice(&layr_block);
    data.extend_from_slice(&lmsk_block);
    data
}

/// Build one ImageSourceData block:
///   "8BIM" | key (4 bytes) | length (4 bytes, big-endian) | data | [pad to 4]
fn isd_block(key: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let len = data.len() as u32;
    let pad = (4 - (len as usize % 4)) % 4;
    let mut block = Vec::with_capacity(12 + data.len() + pad);
    block.extend_from_slice(b"8BIM");
    block.extend_from_slice(key);
    // length is big-endian (tag 37724 endianness follows the container
    // endianness, but the internal "psdformat" for 8-bit PSD is always BE32BIT
    // = "8BIM", so sizes within the ISD blocks are big-endian)
    block.extend_from_slice(&len.to_be_bytes());
    block.extend_from_slice(data);
    for _ in 0..pad { block.push(0x00); }
    block
}

// ─── Photoshop Image Resources helper (tag 34377, for future use) ─────────────

#[allow(dead_code)]
/// Build a Photoshop Image Resource block (numeric ID format) for tag 34377:
///   "8BIM" | id (2 bytes, big-endian) | empty pascal name (2 bytes) |
///   data length (4 bytes, big-endian) | data | optional padding byte
fn photoshop_image_resource(resource_id: u16, data: &[u8]) -> Vec<u8> {
    let mut block = Vec::with_capacity(12 + data.len());
    block.extend_from_slice(b"8BIM");
    block.push((resource_id >> 8) as u8);
    block.push((resource_id & 0xFF) as u8);
    block.push(0x00); // pascal name length = 0
    block.push(0x00); // pad to even
    let len = data.len() as u32;
    block.extend_from_slice(&len.to_be_bytes());
    block.extend_from_slice(data);
    if data.len() % 2 != 0 {
        block.push(0x00);
    }
    block
}

// ─── Binary helpers (little-endian, for TIFF) ────────────────────────────────

fn w16<W: Write>(w: &mut W, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn w32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// IFD entry whose value fits inline (SHORT type, count=1).
fn ifd_short<W: Write>(w: &mut W, tag: u16, count: u32, value: u16) -> std::io::Result<()> {
    w16(w, tag)?;
    w16(w, 3)?; // type = SHORT
    w32(w, count)?;
    w16(w, value)?;
    w16(w, 0)?; // pad value field to 4 bytes
    Ok(())
}

/// IFD entry whose value fits inline (LONG type, count=1).
fn ifd_long<W: Write>(w: &mut W, tag: u16, count: u32, value: u32) -> std::io::Result<()> {
    w16(w, tag)?;
    w16(w, 4)?; // type = LONG
    w32(w, count)?;
    w32(w, value)?;
    Ok(())
}

/// IFD entry whose value is stored at an offset (data too large for inline).
fn ifd_off<W: Write>(w: &mut W, tag: u16, tiff_type: u16, count: u32, offset: u32) -> std::io::Result<()> {
    w16(w, tag)?;
    w16(w, tiff_type)?;
    w32(w, count)?;
    w32(w, offset)?;
    Ok(())
}
