//! What a style scan reads and writes, outside the orchestration.
//!
//! A reference folder holds the artist's pages in `raw/` and the published
//! translation's in `trans/`. The scan pairs them page by page, reads both sides
//! of each matched page, and keeps the sentence pairs; a model then turns the
//! pairs into a [`StyleProfile`].

use std::path::{Path, PathBuf};

use anyhow::Result;
use image::DynamicImage;
use koharu_types::TextBlock;
use serde::{Deserialize, Serialize};

use super::style::StyleProfile;
use crate::vietocr::{VietOcr, keep_glyphs};

pub const RAW_DIR: &str = "raw";
pub const TRANSLATED_DIR: &str = "trans";

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff"];

/// The images directly inside `dir`, in name order.
///
/// Name order is page order in every volume seen so far, and the alignment
/// relies on it: it tolerates pages missing from one side, not pages shuffled.
pub fn list_pages(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        })
        .collect();
    files.sort();
    files
}

/// Where the Vietnamese recogniser's models live.
///
/// `KOHARU_VIETOCR_DIR` when set; otherwise the directory
/// `scripts/export_vietocr_to_onnx.py` is documented to write to.
pub fn vietocr_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("KOHARU_VIETOCR_DIR") {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("koharu")
        .join("vietocr")
}

/// Open the recogniser, saying how to get it when it is not there.
pub fn open_reader() -> Result<VietOcr> {
    let dir = vietocr_dir();
    if !dir.join("vietocr_encoder.onnx").is_file() {
        anyhow::bail!(
            "Vietnamese OCR models not found in {}. Create them with \
             `python scripts/export_vietocr_to_onnx.py {}`, or point \
             KOHARU_VIETOCR_DIR at a directory that has them.",
            dir.display(),
            dir.display()
        );
    }
    VietOcr::open(&dir)
}

/// Read one detected block off a translated page.
///
/// `mask` is the page's glyph mask. Blanking the artwork around the lettering
/// before reading it took the character error over the measured corpus from
/// 12.9% to 6.3%.
pub fn read_translated(
    reader: &VietOcr,
    sheet: &DynamicImage,
    mask: Option<&DynamicImage>,
    block: &TextBlock,
) -> Result<String> {
    // Lettering leans past the detected edge, and a clipped glyph is guessed
    // at rather than read.
    const PAD: f32 = 8.0;
    let left = (block.x - PAD).max(0.0) as u32;
    let top = (block.y - PAD).max(0.0) as u32;
    let right = ((block.x + block.width + PAD) as u32).min(sheet.width());
    let bottom = ((block.y + block.height + PAD) as u32).min(sheet.height());
    if right <= left || bottom <= top {
        return Ok(String::new());
    }
    let (width, height) = (right - left, bottom - top);

    let mut crop = sheet.crop_imm(left, top, width, height);
    if let Some(mask) = mask {
        crop = keep_glyphs(&crop, &mask.crop_imm(left, top, width, height));
    }
    reader.read_block(&crop)
}

/// A finished style scan, as the curator sees and edits it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleScanResult {
    pub profile: StyleProfile,
    pub raw_pages: usize,
    pub translated_pages: usize,
    /// Pages the alignment matched and whose balloons were read.
    pub paired_pages: usize,
    /// Sentence pairs the profile was learned from.
    pub pair_count: usize,
    /// Set once a person has looked the profile over and saved it.
    pub is_verified_by_human: bool,
}

/// Where a scan is resumed from after an interruption.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleScanCheckpoint {
    /// How many alignment steps have been read and written to `pairs.jsonl`.
    pub steps_done: usize,
    pub paired_pages: usize,
    /// Length of `pairs.jsonl` when this checkpoint was written. A scan cut off
    /// mid-page — a battery running out — can leave pairs after it, or half a
    /// line; resuming cuts the file back to here before reading on.
    ///
    /// `None` in checkpoints written before this was recorded: their pairs are
    /// kept as they are, since cutting back to zero would throw a scan away.
    #[serde(default)]
    pub pairs_bytes: Option<u64>,
}

pub const OUT_DIR_NAME: &str = "style_scan";

pub fn out_dir(root: &Path) -> PathBuf {
    root.join(OUT_DIR_NAME)
}

pub fn pairs_path(out_dir: &Path) -> PathBuf {
    out_dir.join("pairs.jsonl")
}

fn checkpoint_path(out_dir: &Path) -> PathBuf {
    out_dir.join("checkpoint.json")
}

fn result_path(out_dir: &Path) -> PathBuf {
    out_dir.join("style_profile_v1.json")
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(value) => Some(value),
        Err(err) => {
            tracing::warn!(path = %path.display(), "ignoring unreadable file: {err}");
            None
        }
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Write beside and rename, so an interrupted write never leaves a half
    // file where the last good one was.
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

pub fn read_checkpoint(out_dir: &Path) -> Option<StyleScanCheckpoint> {
    read_json(&checkpoint_path(out_dir))
}

pub fn write_checkpoint(out_dir: &Path, checkpoint: &StyleScanCheckpoint) -> Result<()> {
    write_json(&checkpoint_path(out_dir), checkpoint)
}

pub fn read_result(out_dir: &Path) -> Option<StyleScanResult> {
    read_json(&result_path(out_dir))
}

pub fn write_result(out_dir: &Path, result: &StyleScanResult) -> Result<()> {
    write_json(&result_path(out_dir), result)
}

/// How long the corpus file is now, for [`StyleScanCheckpoint::pairs_bytes`].
pub fn pairs_len(out_dir: &Path) -> u64 {
    std::fs::metadata(pairs_path(out_dir)).map_or(0, |m| m.len())
}

/// Cut the corpus file back to what a checkpoint recorded, dropping whatever an
/// interrupted page left behind.
pub fn truncate_pairs(out_dir: &Path, len: u64) -> Result<()> {
    let path = pairs_path(out_dir);
    match std::fs::OpenOptions::new().write(true).open(&path) {
        Ok(file) => {
            if file.metadata()?.len() > len {
                file.set_len(len)?;
            }
            Ok(())
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

/// Add one page's pairs to the corpus file.
pub fn append_pairs(out_dir: &Path, pairs: &[super::SentencePair]) -> Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(out_dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(pairs_path(out_dir))?;
    for pair in pairs {
        writeln!(file, "{}", serde_json::to_string(pair)?)?;
    }
    Ok(())
}

pub fn read_pairs(out_dir: &Path) -> Result<Vec<super::SentencePair>> {
    let Ok(text) = std::fs::read_to_string(pairs_path(out_dir)) else {
        return Ok(Vec::new());
    };
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bilingual::SentencePair;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("koharu-corpus-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn pages_come_back_in_name_order_and_only_images() {
        let dir = scratch("pages");
        for name in ["表紙0292.JPG", "表紙0291.jpg", "notes.txt", "表紙0293.png"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        std::fs::create_dir(dir.join("sub.jpg")).unwrap();

        let names: Vec<String> = list_pages(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        assert_eq!(names, ["表紙0291.jpg", "表紙0292.JPG", "表紙0293.png"]);
    }

    /// Pairs are appended page by page so an interrupted scan keeps what it
    /// already read; reading back must see every page's pairs in order.
    #[test]
    fn pairs_appended_across_pages_read_back_in_order() {
        let dir = scratch("pairs");
        let pair = |source: &str| SentencePair {
            source: source.to_string(),
            target: "đích".to_string(),
            page: "p.jpg".to_string(),
            target_box: [0.0; 4],
            confidence: 0.9,
        };

        append_pairs(&dir, &[pair("一"), pair("二")]).unwrap();
        append_pairs(&dir, &[pair("三")]).unwrap();

        let sources: Vec<String> = read_pairs(&dir)
            .unwrap()
            .into_iter()
            .map(|p| p.source)
            .collect();
        assert_eq!(sources, ["一", "二", "三"]);
    }

    /// A checkpoint from before the length was recorded must not read as
    /// "the file was empty".
    #[test]
    fn an_older_checkpoint_without_a_length_reads_as_unknown() {
        let checkpoint: StyleScanCheckpoint =
            serde_json::from_str(r#"{"stepsDone":71,"pairedPages":71}"#).unwrap();
        assert_eq!(checkpoint.steps_done, 71);
        assert_eq!(checkpoint.pairs_bytes, None);
    }

    /// A scan stopped mid-page leaves more in the file than its checkpoint
    /// knows about — sometimes half a line. Resuming must not read that page
    /// twice, and must not choke on the half line.
    #[test]
    fn resuming_drops_what_an_interrupted_page_left_behind() {
        let dir = scratch("resume");
        let pair = |source: &str| SentencePair {
            source: source.to_string(),
            target: "đích".to_string(),
            page: "p.jpg".to_string(),
            target_box: [0.0; 4],
            confidence: 0.9,
        };
        append_pairs(&dir, &[pair("一")]).unwrap();
        let checkpoint = pairs_len(&dir);

        append_pairs(&dir, &[pair("二")]).unwrap();
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(pairs_path(&dir))
                .unwrap();
            write!(file, "{{\"source\":\"三\",\"tar").unwrap();
        }
        assert!(read_pairs(&dir).is_err());

        truncate_pairs(&dir, checkpoint).unwrap();

        let sources: Vec<String> = read_pairs(&dir)
            .unwrap()
            .into_iter()
            .map(|p| p.source)
            .collect();
        assert_eq!(sources, ["一"]);
    }

    #[test]
    fn a_folder_never_scanned_has_no_pairs_and_no_result() {
        let dir = scratch("empty");
        assert!(read_pairs(&dir).unwrap().is_empty());
        assert!(read_result(&dir).is_none());
        assert!(read_checkpoint(&dir).is_none());
    }

    #[test]
    fn a_result_survives_being_written_and_read() {
        let dir = scratch("result");
        let result = StyleScanResult {
            profile: StyleProfile {
                address: vec!["Ta/Cậu — Tổng tư lệnh với Kinnikuman".to_string()],
                ..Default::default()
            },
            pair_count: 151,
            ..Default::default()
        };

        write_result(&dir, &result).unwrap();
        let back = read_result(&dir).unwrap();

        assert_eq!(back.pair_count, 151);
        assert_eq!(back.profile.address, result.profile.address);
        assert!(!dir.join("style_profile_v1.json.tmp").exists());
    }
}
