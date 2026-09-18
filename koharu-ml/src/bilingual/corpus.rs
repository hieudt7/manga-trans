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
    /// Who read the volume: `"vietocr"` for the app's own OCR scan,
    /// `"claude-api"` for the app reading the pages through the Claude API,
    /// `"claude"` for the `/style-read` command in Claude Code. `None` in
    /// results written before this was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// The model that read the pages, when a model did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Whether the original pages were read alongside the translation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub with_raw: Option<bool>,
    /// The name it is kept under in the profile library.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Seconds since the Unix epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
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

// ─── Reading pages through a vision model ─────────────────────────────────────

/// The long side a page is reduced to before a model reads it. Small print in
/// a balloon stays legible at this size, and a model downsamples anything much
/// larger anyway.
pub const READING_SIDE: u32 = 1500;

/// A page as the reader is shown it: reduced to [`READING_SIDE`], as JPEG.
pub fn reading_jpeg(path: &Path) -> Result<Vec<u8>> {
    let image = image::open(path)?;
    let long = image.width().max(image.height());
    let image = if long > READING_SIDE {
        image.resize(
            READING_SIDE,
            READING_SIDE,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        image
    };
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90)
        .encode_image(&image.to_rgb8())?;
    Ok(bytes)
}

/// A short ASCII name for a page. The source pages are called 表紙0297.JPG,
/// and a name carrying that does not open from every tool.
pub fn page_id(path: &Path, index: usize, taken: &mut std::collections::HashSet<String>) -> String {
    let ident: String = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    let ident = if ident.is_empty() || taken.contains(&ident) {
        format!("p{index:04}")
    } else {
        ident
    };
    taken.insert(ident.clone());
    ident
}

fn page_notes_dir(out_dir: &Path) -> PathBuf {
    out_dir.join("claude_api").join("pages")
}

pub fn read_page_notes(out_dir: &Path, page: &str) -> Option<super::vision::PageNotes> {
    read_json(&page_notes_dir(out_dir).join(format!("{page}.json")))
}

pub fn write_page_notes(out_dir: &Path, notes: &super::vision::PageNotes) -> Result<()> {
    write_json(
        &page_notes_dir(out_dir).join(format!("{}.json", notes.page)),
        notes,
    )
}

// ─── The profile library ──────────────────────────────────────────────────────

/// Where saved profiles live, to be picked from when translating.
///
/// `KOHARU_STYLE_PROFILES_DIR` when set; otherwise `style_profiles/` in the
/// source tree the app was built from, when that tree is still there (a
/// development build); otherwise the app's data directory.
pub fn library_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("KOHARU_STYLE_PROFILES_DIR") {
        return PathBuf::from(dir);
    }
    let in_tree = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|root| root.join("style_profiles"));
    if let Some(dir) = in_tree.filter(|dir| dir.is_dir()) {
        return dir;
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("style_profiles")
}

/// A file name for a library entry, from whatever name it was given.
pub fn library_file_name(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let cleaned = cleaned.trim_matches('-');
    if cleaned.is_empty() {
        "profile".to_string()
    } else {
        cleaned.to_string()
    }
}

/// The default library name for a reference folder: its parent and itself, so
/// `…/kinnikuman/tap1` does not collide with another series' `tap1`.
pub fn default_library_name(root: &Path) -> String {
    let part = |p: Option<&Path>| {
        p.and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
    };
    let (parent, own) = (part(root.parent()), part(Some(root)));
    library_file_name(&if parent.is_empty() {
        own
    } else {
        format!("{parent}-{own}")
    })
}

/// Keep `result` in the library under its name.
pub fn save_to_library(dir: &Path, result: &StyleScanResult) -> Result<PathBuf> {
    let name = result
        .name
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("a library entry needs a name"))?;
    let path = dir.join(format!("{}.json", library_file_name(name)));
    write_json(&path, result)?;
    Ok(path)
}

/// Every profile in the library, newest first.
pub fn list_library(dir: &Path) -> Vec<StyleScanResult> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut results: Vec<StyleScanResult> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter_map(|p| {
            let mut result: StyleScanResult = read_json(&p)?;
            if result.name.is_none() {
                result.name = p.file_stem().map(|s| s.to_string_lossy().to_string());
            }
            Some(result)
        })
        .collect();
    results.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.name.cmp(&b.name)));
    results
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

    /// `/style-read` writes this file from Python; the app has to take it as
    /// written, and still read results from before `source` existed.
    #[test]
    fn a_result_written_by_style_read_loads() {
        let claude: StyleScanResult = serde_json::from_str(
            r#"{"profile":{"voice":["v"],"address":[],"soundEffects":[],"glossary":[["長官","sếp"]]},
                "rawPages":94,"translatedPages":94,"pairedPages":94,"pairCount":0,
                "isVerifiedByHuman":false,"source":"claude"}"#,
        )
        .unwrap();
        assert_eq!(claude.source.as_deref(), Some("claude"));
        assert_eq!(
            claude.profile.glossary,
            [["長官".to_string(), "sếp".to_string()]]
        );

        let older: StyleScanResult = serde_json::from_str(
            r#"{"profile":{},"rawPages":1,"translatedPages":1,"pairedPages":1,"pairCount":3,"isVerifiedByHuman":true}"#,
        )
        .unwrap();
        assert_eq!(older.source, None);
    }

    #[test]
    fn page_ids_are_ascii_and_unique() {
        let mut taken = std::collections::HashSet::new();
        assert_eq!(
            page_id(Path::new("trans/表紙0297.JPG"), 0, &mut taken),
            "0297"
        );
        assert_eq!(
            page_id(Path::new("raw/表紙0297.png"), 1, &mut taken),
            "p0001"
        );
        assert_eq!(page_id(Path::new("表紙.jpg"), 2, &mut taken), "p0002");
    }

    #[test]
    fn a_reading_copy_is_reduced_to_the_reading_side() {
        let dir = scratch("reading");
        let page = dir.join("big.png");
        image::RgbImage::from_pixel(3000, 2000, image::Rgb([255, 255, 255]))
            .save(&page)
            .unwrap();
        let copy = image::load_from_memory(&reading_jpeg(&page).unwrap()).unwrap();
        assert_eq!((copy.width(), copy.height()), (1500, 1000));
    }

    #[test]
    fn library_names_come_from_the_folder_and_its_parent() {
        assert_eq!(
            default_library_name(Path::new("/x/kinnikuman/tap1")),
            "kinnikuman-tap1"
        );
        assert_eq!(
            library_file_name("  Kinnikuman / tập 1 "),
            "Kinnikuman---tập-1"
        );
        assert_eq!(library_file_name("///"), "profile");
    }

    #[test]
    fn saved_profiles_list_newest_first() {
        let dir = scratch("library");
        for (name, at) in [("old", 1), ("new", 2)] {
            save_to_library(
                &dir,
                &StyleScanResult {
                    name: Some(name.to_string()),
                    created_at: Some(at),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let names: Vec<String> = list_library(&dir)
            .into_iter()
            .filter_map(|r| r.name)
            .collect();
        assert_eq!(names, ["new", "old"]);
    }

    #[test]
    fn page_notes_survive_being_written_and_read() {
        let dir = scratch("notes");
        let notes = crate::bilingual::vision::PageNotes {
            page: "0309".to_string(),
            summary: "Meat tìm hoàng tử".to_string(),
            ..Default::default()
        };
        write_page_notes(&dir, &notes).unwrap();
        assert_eq!(
            read_page_notes(&dir, "0309").unwrap().summary,
            "Meat tìm hoàng tử"
        );
        assert!(read_page_notes(&dir, "0310").is_none());
    }

    /// What `/style-read` writes, cast included, has to load as written.
    #[test]
    fn a_style_read_result_with_its_cast_loads() {
        let result: StyleScanResult = serde_json::from_str(
            r#"{"profile": {"approach": ["Keeps -chan"], "voice": [], "address": [], "soundEffects": [],
                "glossary": [["", "Kinnikuman"]],
                "characters": [{"id": "kinnikuman", "name": "Kinnikuman", "nameJa": "", "aliases": [],
                  "gender": "male", "ageGroup": "young_adult", "role": "Hoàng tử", "personality": "",
                  "speech": "", "selfTerms": [], "appearances": 2,
                  "relations": [{"to": "meat", "relation": "servant", "address": "ta/ngươi"}]}]},
              "rawPages": 0, "translatedPages": 2, "pairedPages": 2, "pairCount": 0,
              "isVerifiedByHuman": false, "source": "claude", "withRaw": false,
              "name": "tmp-vi-only", "createdAt": 1}"#,
        )
        .unwrap();
        let cast = &result.profile.characters;
        assert_eq!(cast[0].relations[0].address, "ta/ngươi");
        assert_eq!(result.with_raw, Some(false));
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
