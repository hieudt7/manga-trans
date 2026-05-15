use std::{path::PathBuf, sync::Mutex};

use anyhow::Result;
use image::{DynamicImage, Rgb, imageops};
use imageproc::drawing::{draw_hollow_rect_mut, draw_line_segment_mut};
use imageproc::rect::Rect;
use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};

// ─── CCIP constants ───────────────────────────────────────────────────────────

const CCIP_SIZE: u32 = 384;
const CCIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CCIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
/// Minimum absolute cosine similarity to consider a face a known character.
const MATCH_THRESHOLD: f32 = 0.80;
/// When ≥2 characters are in the library the winning character must also be at
/// least this much better than the runner-up, preventing ambiguous faces from
/// being assigned to any character.
const MATCH_MARGIN: f32 = 0.04;
/// When `best_sim` is at or above this value the margin check is bypassed: a
/// very high cosine similarity already indicates a reliable match.
const HIGH_CONF_THRESHOLD: f32 = 0.88;

// ─── Face detector constants ──────────────────────────────────────────────────

const FACE_SIZE: u32 = 640;
const FACE_CONF_THRESHOLD: f32 = 0.35;
const FACE_IOU_THRESHOLD: f32 = 0.45;
/// Margin added around each detected face crop (as a fraction of face size).
const FACE_CROP_MARGIN: f32 = 0.2;

// ─── Panel detector constants ─────────────────────────────────────────────────

const PANEL_SIZE: u32 = 640;
const PANEL_CONF_THRESHOLD: f32 = 0.25; // recommended by leoxs22/manga-panel-detector-yolo26n

// ─── WD Tagger constants ──────────────────────────────────────────────────────

/// Input size for WD Tagger (SmilingWolf/wd-v1-4-convnext-tagger-v2 and v3).
const WD_TAGGER_SIZE: u32 = 448;
/// Minimum tag probability to consider a tag "active".
const WD_TAGGER_THRESHOLD: f32 = 0.35;

/// Tags that indicate male gender (danbooru tag names).
const MALE_TAGS: &[&str] = &[
    "1boy", "male", "male_focus", "multiple_boys", "2boys", "3boys", "4boys", "5boys",
    "bishounen", "boy", "old_man", "middle-aged_man",
];
/// Tags that indicate female gender.
const FEMALE_TAGS: &[&str] = &[
    "1girl", "female", "female_focus", "multiple_girls", "2girls", "3girls", "4girls",
    "girl", "woman", "bishoujo", "old_woman", "middle-aged_woman",
];
/// Tags that indicate child age group.
const CHILD_AGE_TAGS: &[&str] = &["loli", "shota", "child", "young_boy", "young_girl", "little_boy", "little_girl"];
/// Tags that indicate teenage/youth age group.
const TEEN_AGE_TAGS: &[&str] = &[
    "teenage", "teen", "high_school_girl", "high_school_boy",
    "school_uniform", "sailor_uniform", "student",
];
/// Tags that indicate middle-aged or elder age group.
const ELDER_AGE_TAGS: &[&str] = &[
    "old_man", "old_woman", "elderly", "wrinkles", "mature_male", "mature_female",
    "middle-aged", "middle-aged_man", "middle-aged_woman",
];

struct WdTaggerModel {
    session: Mutex<Session>,
    male_indices: Vec<usize>,
    female_indices: Vec<usize>,
    child_indices: Vec<usize>,
    teen_indices: Vec<usize>,
    elder_indices: Vec<usize>,
}

// ─── File paths ───────────────────────────────────────────────────────────────

fn ccip_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("models")
        .join("ccip.onnx")
}

fn face_det_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("models")
        .join("face_detector_anime.onnx")
}

fn panel_det_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("models")
        .join("manga_panel_detector.onnx")
}

fn wd_tagger_model_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("models")
        .join("wd_tagger.onnx")
}

fn wd_tagger_tags_path() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("models")
        .join("wd_tagger_tags.csv")
}

fn character_lib_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("koharu")
        .join("character_lib.json")
}

// ─── Types ────────────────────────────────────────────────────────────────────

/// A single known character stored in the library.
/// The `embedding` field is persisted to disk but omitted from API responses
/// (use `CharacterInfo` for API output).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CharacterEntry {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub traits: Vec<String>,
    #[serde(default)]
    pub relations: Vec<String>,
    /// 512-dim L2-normalised CCIP embedding.
    pub embedding: Vec<f32>,
}

/// API-safe view of a `CharacterEntry` — excludes the raw embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterInfo {
    pub id: String,
    pub name: String,
    pub traits: Vec<String>,
    pub relations: Vec<String>,
}

impl From<&CharacterEntry> for CharacterInfo {
    fn from(e: &CharacterEntry) -> Self {
        Self {
            id: e.id.clone(),
            name: e.name.clone(),
            traits: e.traits.clone(),
            relations: e.relations.clone(),
        }
    }
}

/// Result of matching one detected face against the character library.
#[derive(Debug, Clone, Default)]
pub struct FaceMatch {
    pub name: String,
    pub traits: Vec<String>,
    pub relations: Vec<String>,
    pub confidence: f32,
    pub is_known: bool,
}

#[derive(Clone)]
struct FaceBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    score: f32,
}

// ─── CharacterLibrary ─────────────────────────────────────────────────────────

pub struct CharacterLibrary {
    ccip: Option<Mutex<Session>>,
    face_det: Option<Mutex<Session>>,
    panel_det: Option<Mutex<Session>>,
    wd_tagger: Option<WdTaggerModel>,
    entries: Mutex<Vec<CharacterEntry>>,
    lib_path: PathBuf,
}

impl CharacterLibrary {
    /// Create an empty library with no entries and no ML models loaded.
    /// Used as a graceful fallback when the normal `load` path fails.
    pub fn empty() -> Self {
        Self {
            ccip: None,
            face_det: None,
            panel_det: None,
            wd_tagger: None,
            entries: Mutex::new(Vec::new()),
            lib_path: character_lib_path(),
        }
    }

    /// Load the library from disk and initialise ONNX sessions if models are available.
    pub fn load() -> Result<Self> {
        let lib_path = character_lib_path();
        let entries: Vec<CharacterEntry> = if lib_path.exists() {
            let json = std::fs::read_to_string(&lib_path)?;
            serde_json::from_str(&json).unwrap_or_default()
        } else {
            Vec::new()
        };

        tracing::info!(count = entries.len(), "character library loaded");

        let ccip = load_session_opt(ccip_model_path());
        if ccip.is_some() {
            tracing::info!("CCIP model loaded");
        } else {
            tracing::warn!("CCIP model not found — character identification disabled. Run tools/export_ccip.py first.");
        }

        let face_det = load_session_opt(face_det_model_path());
        if face_det.is_some() {
            tracing::info!("Anime face detector loaded");
        } else {
            tracing::info!("Anime face detector not found — per-page face scanning disabled; using full-library context");
        }

        let panel_det = load_session_opt(panel_det_model_path());
        if panel_det.is_some() {
            tracing::info!("Manga panel detector loaded");
        } else {
            tracing::info!("Manga panel detector not found — using heuristic panel detection. Run tools/export_ccip.py --panel-only to download.");
        }

        let wd_tagger = load_wd_tagger();

        Ok(Self {
            ccip: ccip.map(Mutex::new),
            face_det: face_det.map(Mutex::new),
            panel_det: panel_det.map(Mutex::new),
            wd_tagger,
            entries: Mutex::new(entries),
            lib_path,
        })
    }

    pub fn has_ccip(&self) -> bool {
        self.ccip.is_some()
    }

    pub fn has_face_detector(&self) -> bool {
        self.face_det.is_some()
    }

    pub fn has_panel_detector(&self) -> bool {
        self.panel_det.is_some()
    }

    pub fn list(&self) -> Vec<CharacterInfo> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .map(CharacterInfo::from)
            .collect()
    }

    /// Add a character from one or more face image crops.
    /// When multiple images are provided their embeddings are averaged and
    /// L2-re-normalised, giving a more robust identity centroid.
    /// Returns the new character's `id`.
    pub fn add_character(
        &self,
        name: String,
        traits: Vec<String>,
        relations: Vec<String>,
        face_images: &[DynamicImage],
    ) -> Result<String> {
        if face_images.is_empty() {
            anyhow::bail!("at least one face image is required");
        }

        // Compute embedding. If CCIP is not available, store an empty vector —
        // the character will still be useful via the full-library fallback (all
        // characters are injected as context when face detection is unavailable).
        let embedding = if let Some(ccip) = &self.ccip {
            let embeddings: Result<Vec<Vec<f32>>> =
                face_images.iter().map(|img| embed_face(ccip, img)).collect();
            let embeddings = embeddings?;

            let dim = embeddings[0].len();
            let mut mean = vec![0f32; dim];
            for emb in &embeddings {
                for (m, v) in mean.iter_mut().zip(emb.iter()) {
                    *m += v;
                }
            }
            let n = embeddings.len() as f32;
            for m in mean.iter_mut() {
                *m /= n;
            }
            let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            mean.into_iter().map(|x| x / norm).collect()
        } else {
            tracing::warn!("CCIP not loaded — storing character without face embedding; face recognition disabled");
            Vec::new()
        };

        let id = uuid::Uuid::new_v4().to_string();

        let entry = CharacterEntry {
            id: id.clone(),
            name,
            traits,
            relations,
            embedding,
        };

        {
            let mut entries = self.entries.lock().unwrap();
            entries.push(entry);
            self.save_locked(&entries)?;
        }

        Ok(id)
    }

    pub fn remove_character(&self, id: &str) -> Result<()> {
        let mut entries = self.entries.lock().unwrap();
        let len_before = entries.len();
        entries.retain(|e| e.id != id);
        if entries.len() == len_before {
            anyhow::bail!("Character not found: {id}");
        }
        self.save_locked(&entries)
    }

    /// Scan a page image for known characters.
    ///
    /// - If a face detector model is available: detect faces, embed each, and match
    ///   against the library.
    /// - Otherwise: return all characters from the library directly (full-library mode).
    ///
    /// Returns an empty Vec if the library is empty.
    pub fn scan_page(&self, image: &DynamicImage) -> Vec<FaceMatch> {
        let entries = self.entries.lock().unwrap().clone();
        if entries.is_empty() {
            return Vec::new();
        }

        // Smart mode: face detector + CCIP
        if let (Some(face_det), Some(ccip)) = (&self.face_det, &self.ccip) {
            match detect_faces(face_det, image) {
                Ok(face_boxes) if !face_boxes.is_empty() => {
                    let mut seen_names = std::collections::HashSet::new();
                    let mut matches: Vec<FaceMatch> = Vec::new();
                    for bbox in &face_boxes {
                        let crop = crop_face(image, bbox);
                        if let Ok(emb) = embed_face(ccip, &crop) {
                            let m = find_best_match(&entries, &emb);
                            if m.is_known && seen_names.insert(m.name.clone()) {
                                matches.push(m);
                            } else if !m.is_known {
                                matches.push(m);
                            }
                        }
                    }
                    return matches;
                }
                Ok(_) => {
                    tracing::debug!("face detector found no faces on page — falling back to full-library context");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "face detection failed — falling back to full-library context");
                }
            }
        }

        // Full-library fallback: include all characters
        entries
            .iter()
            .map(|e| FaceMatch {
                name: e.name.clone(),
                traits: e.traits.clone(),
                relations: e.relations.clone(),
                confidence: 1.0,
                is_known: true,
            })
            .collect()
    }

    /// Format a list of face matches into a context string suitable for the LLM system prompt.
    /// Returns `None` if there are no known characters.
    pub fn build_context(&self, matches: &[FaceMatch]) -> Option<String> {
        let known: Vec<&FaceMatch> = matches.iter().filter(|m| m.is_known).collect();
        if known.is_empty() {
            return None;
        }

        let mut lines = vec!["Characters in this scene (for context only — do NOT prefix translated lines with character names):".to_string()];
        for m in &known {
            let mut desc = format!("- {}", m.name);
            if !m.traits.is_empty() {
                desc.push_str(&format!(" ({})", m.traits.join(", ")));
            }
            if !m.relations.is_empty() {
                desc.push_str(&format!(". Relationships: {}", m.relations.join("; ")));
            }
            lines.push(desc);
        }
        Some(lines.join("\n"))
    }

    /// For each text block (id, x, y, w, h — in pixels), detect which character
    /// is speaking, using a three-step spatial pipeline:
    ///
    /// 1. **Balloon filter** — text blocks not inside any detected balloon are
    ///    narration/title/SFX and receive `None`.
    /// 2. **Panel isolation** — manga panels are detected via gutter-line analysis.
    ///    Only faces within the *same panel* as the balloon are considered.
    ///    This prevents a face from an adjacent panel from being matched to a
    ///    balloon in a different panel.
    /// 3. **Nearest-face match** — within the panel, the face closest to the
    ///    balloon centre is embedded and matched against the library.
    ///
    /// When `balloons` is empty (detection not yet run), the balloon filter is
    /// skipped and all faces on the page are candidates for every block.
    ///
    /// Falls back to full-library mode when face detection or CCIP is unavailable.
    pub fn assign_speakers_to_blocks(
        &self,
        image: &DynamicImage,
        blocks: &[(String, f32, f32, f32, f32)], // (id, x, y, w, h)
        balloons: &[(f32, f32, f32, f32)],       // (x, y, w, h)
        panels: &[(f32, f32, f32, f32)],         // pre-computed panels
    ) -> Vec<(String, Option<FaceMatch>)> {
        if blocks.is_empty() {
            return Vec::new();
        }

        let have_balloons = !balloons.is_empty();

        let entries = self.entries.lock().unwrap().clone();

        // Smart mode: face detector + CCIP
        tracing::info!(
            have_face_det = self.face_det.is_some(),
            have_ccip = self.ccip.is_some(),
            "speaker assign: model availability"
        );
        if let (Some(face_det), Some(ccip)) = (&self.face_det, &self.ccip) {
            // Run face detection per panel so each panel fills the 640×640 detector
            // input.  On the full-page pass small faces in large panels are scaled down
            // too much and missed; cropping brings them to a detectable size.
            let face_boxes: Vec<FaceBox> = {
                let mut all: Vec<FaceBox> = Vec::new();
                for (px, py, pw, ph) in panels.iter() {
                    let cx = px.max(0.0) as u32;
                    let cy = py.max(0.0) as u32;
                    let cw = (pw.max(0.0) as u32).min(image.width().saturating_sub(cx));
                    let ch = (ph.max(0.0) as u32).min(image.height().saturating_sub(cy));
                    if cw < 16 || ch < 16 {
                        continue;
                    }
                    let crop = image.crop_imm(cx, cy, cw, ch);
                    if let Ok(high_faces) = detect_faces(face_det, &crop) {
                        // Always run a second pass at low threshold to catch faces the
                        // normal threshold misses (unusual angles, art style, elderly
                        // characters, etc.).  Merge the two lists, keeping only new
                        // detections that don't overlap (IOU < 0.3) with an existing one.
                        let mut faces = high_faces;
                        if let Ok(low_faces) = detect_faces_threshold(face_det, &crop, 0.15) {
                            for lf in low_faces {
                                let overlaps = faces.iter().any(|hf| {
                                    let ix = (lf.x + lf.width).min(hf.x + hf.width)
                                        - lf.x.max(hf.x);
                                    let iy = (lf.y + lf.height).min(hf.y + hf.height)
                                        - lf.y.max(hf.y);
                                    if ix <= 0.0 || iy <= 0.0 {
                                        return false;
                                    }
                                    let inter = ix * iy;
                                    let union = lf.width * lf.height
                                        + hf.width * hf.height
                                        - inter;
                                    inter / union > 0.3
                                });
                                if !overlaps {
                                    faces.push(lf);
                                }
                            }
                        }
                        if faces.len() > 0 {
                            tracing::info!(
                                panel_x = cx, panel_y = cy,
                                count = faces.len(),
                                "panel faces detected (merged high+low threshold)"
                            );
                        }
                        for fb in &mut faces {
                            fb.x += cx as f32;
                            fb.y += cy as f32;
                        }
                        all.extend(faces);
                    }
                }
                all
            };
            tracing::info!(
                result = format!("ok faces={}", face_boxes.len()),
                "face detection result (per-panel)"
            );
            if !face_boxes.is_empty() {
                    // Embed all faces sequentially; store box + absolute pixel centre + embedding.
                    // (FaceBox is kept so we can re-crop for gender classification.)
                    let face_data: Vec<(FaceBox, f32, f32, Vec<f32>)> = face_boxes
                        .iter()
                        .filter_map(|b| {
                            let crop = crop_face(image, b);
                            let emb = embed_face(ccip, &crop).ok()?;
                            let cx = b.x + b.width / 2.0;
                            let cy = b.y + b.height / 2.0;
                            Some((b.clone(), cx, cy, emb))
                        })
                        .collect();

                    // Pre-group faces by panel index for fast per-panel lookup.
                    // face_panel_idx[i] = Some(panel_index) or None if face is outside all panels.
                    let face_panel_idx: Vec<Option<usize>> = face_data
                        .iter()
                        .map(|(_, fcx, fcy, _)| {
                            panels.iter().position(|(px, py, pw, ph)| {
                                *fcx >= *px
                                    && *fcx <= px + pw
                                    && *fcy >= *py
                                    && *fcy <= py + ph
                            })
                        })
                        .collect();

                    tracing::info!(
                        total_faces = face_data.len(),
                        faces_per_panel = ?face_panel_idx.iter().enumerate()
                            .map(|(i, p)| format!("face{}→panel{:?}", i, p))
                            .collect::<Vec<_>>(),
                        "speaker assign: face-panel grouping"
                    );
                    // Keep a ref to face_data for borrow in the block closure below.
                    let face_data_ref = &face_data;

                    // Pre-convert image to grayscale once; used for balloon tail detection
                    // inside the per-block closure below.
                    let page_gray = image.to_luma8();

                    let debug_mode = std::env::var("KOHARU_DEBUG_SCAN").is_ok();
                    let debug_cell = std::cell::RefCell::new(Vec::<DebugScanEntry>::new());

                    // ─── Phase 1: per-block geometry (balloon → panel → nearest face) ─────
                    //
                    // Compute which face each block would choose, plus the match type and
                    // score, WITHOUT running CCIP yet.  This lets Phase 2 resolve conflicts
                    // before we pay for any embedding comparisons.

                    struct BlockGeom {
                        id: String,
                        block_idx: usize,
                        /// None  = narration/SFX (not inside any balloon).
                        balloon_box: Option<(f32, f32, f32, f32)>,
                        /// None  = balloon not inside any detected panel.
                        panel_idx: Option<usize>,
                        /// Index into face_data.  None = no face candidate in panel.
                        face_idx: Option<usize>,
                        used_tail_ray: bool,
                        /// Geometry score for the face claim (lower = stronger claim).
                        score: f32,
                        tail_dir: Option<(f32, f32)>,
                    }

                    let mut geoms: Vec<BlockGeom> = blocks
                        .iter()
                        .enumerate()
                        .map(|(block_idx, (id, x, y, w, h))| {
                            let block_cx = x + w / 2.0;
                            let block_cy = y + h / 2.0;

                            // Find the containing balloon; skip narration/SFX blocks.
                            let balloon_box = balloons.iter().find_map(|(bx, by, bw, bh)| {
                                if block_cx >= *bx
                                    && block_cx <= bx + bw
                                    && block_cy >= *by
                                    && block_cy <= by + bh
                                {
                                    Some((*bx, *by, *bw, *bh))
                                } else {
                                    None
                                }
                            });

                            let (bbal_x, bbal_y, bbal_w, bbal_h) = match balloon_box {
                                Some(b) => b,
                                None => {
                                    tracing::info!(block_id = %id, "no balloon → skip");
                                    return BlockGeom {
                                        id: id.clone(), block_idx,
                                        balloon_box: None, panel_idx: None,
                                        face_idx: None, used_tail_ray: false,
                                        score: f32::MAX, tail_dir: None,
                                    };
                                }
                            };
                            let balloon_center = (bbal_x + bbal_w / 2.0, bbal_y + bbal_h / 2.0);

                            // Find which panel the balloon belongs to.
                            let panel_idx_opt = panels.iter().position(|(px, py, pw, ph)| {
                                balloon_center.0 >= *px
                                    && balloon_center.0 <= px + pw
                                    && balloon_center.1 >= *py
                                    && balloon_center.1 <= py + ph
                            });

                            tracing::info!(
                                block_id = %id,
                                balloon_cx = balloon_center.0,
                                balloon_cy = balloon_center.1,
                                panel_idx = ?panel_idx_opt,
                                "balloon → panel lookup"
                            );

                            let pidx = match panel_idx_opt {
                                Some(idx) => idx,
                                None => {
                                    tracing::info!(block_id = %id, "balloon not in any panel → skip");
                                    return BlockGeom {
                                        id: id.clone(), block_idx,
                                        balloon_box: Some((bbal_x, bbal_y, bbal_w, bbal_h)),
                                        panel_idx: None, face_idx: None,
                                        used_tail_ray: false, score: f32::MAX, tail_dir: None,
                                    };
                                }
                            };

                            // Collect faces in this panel.
                            let faces_in_panel: Vec<(usize, &(FaceBox, f32, f32, Vec<f32>))> =
                                face_data_ref
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| face_panel_idx[*i] == Some(pidx))
                                    .collect();

                            tracing::info!(
                                block_id = %id,
                                panel_idx = pidx,
                                faces_in_panel = faces_in_panel.len(),
                                "faces available in panel"
                            );

                            // Prefer faces outside the balloon (speakers are outside bubbles).
                            let faces_outside: Vec<_> = faces_in_panel
                                .iter()
                                .copied()
                                .filter(|(_, (fb, fcx, fcy, _))| {
                                    let inside = *fcx >= bbal_x
                                        && *fcx <= bbal_x + bbal_w
                                        && *fcy >= bbal_y
                                        && *fcy <= bbal_y + bbal_h;
                                    let too_small = fb.width < 20.0 || fb.height < 20.0;
                                    !inside && !too_small
                                })
                                .collect();

                            let candidates = if faces_outside.is_empty() {
                                &faces_in_panel
                            } else {
                                &faces_outside
                            };

                            tracing::info!(
                                block_id = %id,
                                candidates = candidates.len(),
                                faces_in_panel = faces_in_panel.len(),
                                "speaker assign: candidate selection"
                            );

                            // When a panel has no detected faces (face detector may have missed
                            // one), fall back to faces whose bounding box overlaps with this
                            // panel's rectangle — still constrained to the same panel area.
                            let panel_overlap_candidates: Vec<(usize, &(FaceBox, f32, f32, Vec<f32>))>;
                            let candidates = if candidates.is_empty() {
                                let (ppx, ppy, ppw, pph) = panels[pidx];
                                panel_overlap_candidates = face_data_ref
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, (fb, _, _, _))| {
                                        // Face box overlaps panel box.
                                        fb.x < ppx + ppw
                                            && fb.x + fb.width > ppx
                                            && fb.y < ppy + pph
                                            && fb.y + fb.height > ppy
                                    })
                                    .collect();
                                tracing::info!(
                                    block_id = %id,
                                    panel_idx = pidx,
                                    overlap_count = panel_overlap_candidates.len(),
                                    "no faces in panel — panel-overlap face fallback"
                                );
                                &panel_overlap_candidates
                            } else {
                                candidates
                            };

                            if candidates.is_empty() {
                                let tail_dir = detect_balloon_tail_direction(
                                    &page_gray, bbal_x, bbal_y, bbal_w, bbal_h,
                                );
                                return BlockGeom {
                                    id: id.clone(), block_idx,
                                    balloon_box: Some((bbal_x, bbal_y, bbal_w, bbal_h)),
                                    panel_idx: Some(pidx), face_idx: None,
                                    used_tail_ray: false, score: f32::MAX, tail_dir,
                                };
                            }

                            let tail_dir = detect_balloon_tail_direction(
                                &page_gray, bbal_x, bbal_y, bbal_w, bbal_h,
                            );

                            tracing::info!(
                                block_id = %id,
                                tail = ?tail_dir,
                                "balloon tail direction"
                            );

                            let bcx = bbal_x + bbal_w / 2.0;
                            let bcy = bbal_y + bbal_h / 2.0;

                            // ── Step 1: Tail-ray → Body-box collision ──────────────────────
                            // Cast a ray from balloon center in tail direction; the first
                            // candidate whose body box is hit (nearest by center distance)
                            // wins.  Body box = face box expanded to roughly full-body size.
                            let tail_ray_hit: Option<(usize, f32)> =
                                if let Some((tdx, tdy)) = tail_dir {
                                    let mag = (tdx * tdx + tdy * tdy).sqrt();
                                    let (udx, udy) = if mag > 1e-9 { (tdx / mag, tdy / mag) } else { (tdx, tdy) };
                                    let mut best_hit: Option<(usize, f32)> = None;
                                    for (fi, (face_box, fcx, fcy, _)) in candidates.iter() {
                                        // Expand face box to body box.
                                        let body_x = face_box.x - face_box.width * 0.5;
                                        let body_y = face_box.y - face_box.height * 0.2;
                                        let body_w = face_box.width * 2.0;
                                        let body_h = face_box.height * 5.0;
                                        if ray_intersects_rect(bcx, bcy, udx, udy, body_x, body_y, body_w, body_h) {
                                            let dist = ((*fcx - bcx).powi(2) + (*fcy - bcy).powi(2)).sqrt();
                                            if best_hit.map_or(true, |(_, d)| dist < d) {
                                                best_hit = Some((*fi, dist));
                                            }
                                        }
                                    }
                                    best_hit
                                } else {
                                    None
                                };

                            let (selected_face_idx, used_tail_ray, score) =
                                if let Some((hit_fi, hit_dist)) = tail_ray_hit {
                                    tracing::info!(
                                        block_id = %id,
                                        face_idx = hit_fi,
                                        dist = hit_dist,
                                        "Step 1: tail-ray body-box hit"
                                    );
                                    (hit_fi, true, hit_dist)
                                } else {
                                    // ── Step 2: Nearest neighbor (center-to-center) ─────────
                                    if tail_dir.is_some() {
                                        tracing::info!(block_id = %id, "Step 1 missed — Step 2: nearest neighbor");
                                    }
                                    // Log all candidates with distances for debugging.
                                    for (fi, (_, fcx, fcy, _)) in candidates.iter() {
                                        let d = ((*fcx - bcx).powi(2) + (*fcy - bcy).powi(2)).sqrt();
                                        tracing::info!(
                                            block_id = %id,
                                            face_idx = fi,
                                            face_cx = fcx,
                                            face_cy = fcy,
                                            balloon_cx = bcx,
                                            balloon_cy = bcy,
                                            dist = d,
                                            "Step 2: candidate"
                                        );
                                    }
                                    let best = candidates
                                        .iter()
                                        .min_by(|(_, (_, fcx_a, fcy_a, _)), (_, (_, fcx_b, fcy_b, _))| {
                                            let da = ((*fcx_a - bcx).powi(2) + (*fcy_a - bcy).powi(2)).sqrt();
                                            let db = ((*fcx_b - bcx).powi(2) + (*fcy_b - bcy).powi(2)).sqrt();
                                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                        })
                                        .unwrap();
                                    let s = ((best.1.1 - bcx).powi(2) + (best.1.2 - bcy).powi(2)).sqrt();
                                    tracing::info!(block_id = %id, face_idx = best.0, dist = s, "Step 2: selected nearest");
                                    (best.0, false, s)
                                };

                            // ── Panel-half guard ─────────────────────────────────────────
                            // The speaker must be in the same half of the panel as the
                            // balloon — both horizontally and vertically.  Using the panel
                            // dimensions directly as the natural coordinate unit avoids
                            // arbitrary magic numbers.
                            // Skip for tail-ray hits (Step 1) — those have directional proof.
                            let face_idx = if !used_tail_ray {
                                let (_, _, ppw, pph) = panels[pidx];
                                let (_, sel_fcx, sel_fcy, _) = &face_data_ref[selected_face_idx];
                                let dx = (sel_fcx - bcx).abs();
                                let dy = (sel_fcy - bcy).abs();
                                if dx > ppw * 0.5 || dy > pph * 0.5 {
                                    tracing::info!(
                                        block_id = %id,
                                        face_cx = sel_fcx, face_cy = sel_fcy,
                                        balloon_cx = bcx, balloon_cy = bcy,
                                        dx, dy,
                                        half_w = ppw * 0.5, half_h = pph * 0.5,
                                        "Step 2: face outside balloon's panel half — WD Tagger fallback"
                                    );
                                    None
                                } else {
                                    Some(selected_face_idx)
                                }
                            } else {
                                Some(selected_face_idx)
                            };

                            BlockGeom {
                                id: id.clone(), block_idx,
                                balloon_box: Some((bbal_x, bbal_y, bbal_w, bbal_h)),
                                panel_idx: Some(pidx),
                                face_idx,
                                used_tail_ray, score, tail_dir,
                            }
                        })
                        .collect();

                    // ─── Phase 2: face deduplication per panel ───────────────────────────────
                    //
                    // Each detected face may be claimed by at most ONE balloon per panel.
                    // When more balloons compete for the same face (panel has fewer detected
                    // faces than speakers), the winner is chosen by:
                    //   1. tail-ray match (stronger geometric evidence)
                    //   2. lowest geometry score (closer / better aligned)
                    // Losers have their face_idx cleared → they fall through to gender in P3.
                    {
                        let mut contested: std::collections::HashMap<(usize, usize), Vec<usize>> =
                            std::collections::HashMap::new();
                        for (gi, g) in geoms.iter().enumerate() {
                            if let (Some(pidx), Some(fidx)) = (g.panel_idx, g.face_idx) {
                                contested.entry((pidx, fidx)).or_default().push(gi);
                            }
                        }
                        for ((pidx, _fidx), claimants) in &contested {
                            if claimants.len() <= 1 {
                                continue;
                            }
                            // ── Step 3: single-character panel → all balloons share that face ──
                            // Count how many distinct faces are in this panel.
                            let faces_in_this_panel = face_panel_idx
                                .iter()
                                .filter(|&&p| p == Some(*pidx))
                                .count();
                            if faces_in_this_panel <= 1 {
                                tracing::info!(
                                    panel_idx = pidx,
                                    "Step 3: single face in panel — all balloons keep same face"
                                );
                                continue;
                            }
                            // Winner = highest priority (tail-ray bit, then lowest score).
                            let winner = *claimants
                                .iter()
                                .max_by(|&&a, &&b| {
                                    let ga = &geoms[a];
                                    let gb = &geoms[b];
                                    let ra = u8::from(ga.used_tail_ray);
                                    let rb = u8::from(gb.used_tail_ray);
                                    ra.cmp(&rb).then_with(|| {
                                        // Lower score wins; reverse comparison for max_by.
                                        gb.score
                                            .partial_cmp(&ga.score)
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                })
                                .unwrap();
                            // Collect all faces in this panel sorted by index.
                            let panel_faces_sorted: Vec<usize> = face_panel_idx
                                .iter()
                                .enumerate()
                                .filter(|(_, p)| **p == Some(*pidx))
                                .map(|(i, _)| i)
                                .collect();

                            // Track which face is claimed by the winner so losers
                            // can be redirected to the next-best unclaimed face.
                            let winner_face = geoms[winner].face_idx;

                            for &gi in claimants {
                                if gi != winner {
                                    // Try to assign the loser to the closest unclaimed face.
                                    let g = &geoms[gi];
                                    let (bbal_x, bbal_y, bbal_w, bbal_h) =
                                        g.balloon_box.unwrap_or((0.0, 0.0, 0.0, 0.0));
                                    let bcx = bbal_x + bbal_w / 2.0;
                                    let bcy = bbal_y + bbal_h / 2.0;
                                    let alt_face = panel_faces_sorted.iter()
                                        .filter(|&&fi| Some(fi) != winner_face)
                                        .min_by(|&&a, &&b| {
                                            let (_, fcx_a, fcy_a, _) = &face_data_ref[a];
                                            let (_, fcx_b, fcy_b, _) = &face_data_ref[b];
                                            let da = ((*fcx_a - bcx).powi(2) + (*fcy_a - bcy).powi(2)).sqrt();
                                            let db = ((*fcx_b - bcx).powi(2) + (*fcy_b - bcy).powi(2)).sqrt();
                                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                        })
                                        .copied();
                                    if let Some(alt) = alt_face {
                                        tracing::info!(
                                            block_id = %geoms[gi].id,
                                            alt_face_idx = alt,
                                            "face deduped — reassigned to next-best face"
                                        );
                                        geoms[gi].face_idx = Some(alt);
                                    } else {
                                        tracing::info!(
                                            block_id = %geoms[gi].id,
                                            "face deduped — no alt face, falling back to gender"
                                        );
                                        geoms[gi].face_idx = None;
                                    }
                                }
                            }
                        }
                    }

                    // ─── Phase 3: CCIP match + gender fallback ───────────────────────────────

                    let results: Vec<(String, Option<FaceMatch>)> = geoms
                        .iter()
                        .map(|g| {
                            // Narration/SFX or balloon outside all panels.
                            if g.balloon_box.is_none() || g.panel_idx.is_none() {
                                return (g.id.clone(), None);
                            }
                            let (bbal_x, bbal_y, bbal_w, bbal_h) = g.balloon_box.unwrap();

                            let (speaker, debug_face_box) = match g.face_idx {
                                Some(fi) => {
                                    let (face_box, _, _, emb) = &face_data_ref[fi];
                                    let m = find_best_match(&entries, emb);
                                    tracing::info!(
                                        block_id = %g.id,
                                        character = %m.name,
                                        confidence = m.confidence,
                                        is_known = m.is_known,
                                        "speaker matched"
                                    );
                                    let fb_clone = if debug_mode { Some(face_box.clone()) } else { None };
                                    // Unknown CCIP → WD Tagger classification.
                                    let m = if !m.is_known {
                                        if let Some(wd_tagger) = &self.wd_tagger {
                                            let crop = crop_face_expanded(image, face_box);
                                            if let Ok(label) = classify_with_wd_tagger(wd_tagger, &crop) {
                                                tracing::info!(
                                                    block_id = %g.id,
                                                    label = %label,
                                                    "unknown face classified by WD Tagger"
                                                );
                                                FaceMatch {
                                                    name: label,
                                                    confidence: m.confidence,
                                                    is_known: false,
                                                    ..Default::default()
                                                }
                                            } else {
                                                m
                                            }
                                        } else {
                                            m
                                        }
                                    } else {
                                        m
                                    };
                                    (Some(m), fb_clone)
                                }
                                None => {
                                    // face_idx was deduped away (or no face in panel).
                                    // Try gender classification using the nearest face in the
                                    // panel — even if that face was claimed by another balloon.
                                    let fallback =
                                        if let (Some(pidx), Some(wd_tagger)) =
                                            (g.panel_idx, &self.wd_tagger)
                                        {
                                            let nearest = face_data_ref
                                                .iter()
                                                .enumerate()
                                                .filter(|(i, _)| face_panel_idx[*i] == Some(pidx))
                                                .min_by(|(_, (_, fcx_a, fcy_a, _)), (_, (_, fcx_b, fcy_b, _))| {
                                                    point_to_rect_dist(*fcx_a, *fcy_a, bbal_x, bbal_y, bbal_w, bbal_h)
                                                        .partial_cmp(&point_to_rect_dist(*fcx_b, *fcy_b, bbal_x, bbal_y, bbal_w, bbal_h))
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                });
                                            if let Some((_, (face_box, _, _, _))) = nearest {
                                                let crop = crop_face_expanded(image, face_box);
                                                classify_with_wd_tagger(wd_tagger, &crop).ok().map(|label| {
                                                    tracing::info!(
                                                        block_id = %g.id,
                                                        label = %label,
                                                        "deduped block classified by WD Tagger"
                                                    );
                                                    FaceMatch {
                                                        name: label,
                                                        confidence: 0.0,
                                                        is_known: false,
                                                        ..Default::default()
                                                    }
                                                })
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        };
                                    (fallback, None)
                                }
                            };

                            if debug_mode {
                                debug_cell.borrow_mut().push(DebugScanEntry {
                                    block_idx: g.block_idx,
                                    char_name: speaker
                                        .as_ref()
                                        .map(|s| s.name.clone())
                                        .unwrap_or_else(|| "none".to_string()),
                                    balloon_box: (bbal_x, bbal_y, bbal_w, bbal_h),
                                    tail_dir: g.tail_dir,
                                    face_box: debug_face_box,
                                });
                            }

                            (g.id.clone(), speaker)
                        })
                        .collect();

                    if debug_mode {
                        let debug_entries = debug_cell.into_inner();
                        save_debug_scan(image, &face_data, &debug_entries, panels);
                    }

                    return results;
            }
        }

        // Full-library fallback: speech-balloon blocks get first character, others get None.
        let all: Vec<FaceMatch> = entries
            .iter()
            .map(|e| FaceMatch {
                name: e.name.clone(),
                traits: e.traits.clone(),
                relations: e.relations.clone(),
                confidence: 1.0,
                is_known: true,
            })
            .collect();

        blocks
            .iter()
            .map(|(id, x, y, w, h)| {
                if have_balloons {
                    let block_cx = x + w / 2.0;
                    let block_cy = y + h / 2.0;
                    let in_balloon = balloons.iter().any(|(bx, by, bw, bh)| {
                        block_cx >= *bx
                            && block_cx <= bx + bw
                            && block_cy >= *by
                            && block_cy <= by + bh
                    });
                    if !in_balloon {
                        return (id.clone(), None);
                    }
                }
                (id.clone(), all.first().cloned())
            })
            .collect()
    }

    /// Detect manga panel bounding boxes for the given page image.
    /// Uses the ML model (`manga_panel_detector.onnx`) when available;
    /// falls back to gutter-line heuristic otherwise.
    /// Returns a list of `(x, y, width, height)` in pixels.
    pub fn detect_panels(&self, image: &DynamicImage) -> Vec<(f32, f32, f32, f32)> {
        if let Some(panel_det) = &self.panel_det {
            match detect_panels_ml(panel_det, image) {
                Ok(panels) if !panels.is_empty() => {
                    tracing::info!(count = panels.len(), "panel detection: ML succeeded");
                    return sort_manga_reading_order(panels);
                }
                Ok(_) => {
                    tracing::warn!("panel detection: ML returned 0 panels — falling back to heuristic");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "panel detection: ML failed — falling back to heuristic");
                }
            }
        } else {
            tracing::info!("panel detection: no ML model loaded, using heuristic");
        }
        let panels = detect_manga_panels(image);
        tracing::info!(count = panels.len(), "panel detection: heuristic result");
        sort_manga_reading_order(panels)
    }

    /// Detect all characters in each panel and return them grouped by panel index.
    /// Each character has a name (if known) and a WD Tagger label (age + gender).
    pub fn detect_panel_characters(
        &self,
        image: &DynamicImage,
        panels: &[(f32, f32, f32, f32)],
    ) -> Vec<Vec<FaceMatch>> {
        let mut result: Vec<Vec<FaceMatch>> = vec![Vec::new(); panels.len()];

        let (Some(face_det), Some(ccip)) = (&self.face_det, &self.ccip) else {
            return result;
        };

        let entries = self.entries.lock().unwrap().clone();

        for (pidx, (px, py, pw, ph)) in panels.iter().enumerate() {
            let cx = px.max(0.0) as u32;
            let cy = py.max(0.0) as u32;
            let cw = (pw.max(0.0) as u32).min(image.width().saturating_sub(cx));
            let ch = (ph.max(0.0) as u32).min(image.height().saturating_sub(cy));
            if cw < 16 || ch < 16 {
                continue;
            }
            let crop = image.crop_imm(cx, cy, cw, ch);

            let mut face_boxes = match detect_faces(face_det, &crop) {
                Ok(f) => f,
                Err(_) => continue,
            };
            // Low-threshold pass to catch small/unusual faces.
            if let Ok(low) = detect_faces_threshold(face_det, &crop, 0.15) {
                for fb in low {
                    let already = face_boxes.iter().any(|existing| {
                        let dx = (existing.x - fb.x).abs();
                        let dy = (existing.y - fb.y).abs();
                        dx < existing.width * 0.5 && dy < existing.height * 0.5
                    });
                    if !already {
                        face_boxes.push(fb);
                    }
                }
            }

            for fb in &face_boxes {
                // Map face box back to full-image coordinates.
                let full_fb = FaceBox {
                    x: fb.x + cx as f32,
                    y: fb.y + cy as f32,
                    width: fb.width,
                    height: fb.height,
                    score: fb.score,
                };
                let face_crop = crop_face(image, &full_fb);
                let m = if let Ok(emb) = embed_face(ccip, &face_crop) {
                    let mut m = find_best_match(&entries, &emb);
                    // Unknown → WD Tagger for age/gender label.
                    if !m.is_known {
                        if let Some(wd) = &self.wd_tagger {
                            let body_crop = crop_face_expanded(image, &full_fb);
                            if let Ok(label) = classify_with_wd_tagger(wd, &body_crop) {
                                m.name = label;
                            }
                        }
                    }
                    m
                } else {
                    continue;
                };
                result[pidx].push(m);
            }
        }

        result
    }

    /// Convenience: scan + build context in one call.
    /// Returns `None` if no characters are recognised or the library is empty.
    pub fn scan_and_build_context(&self, image: &DynamicImage) -> Option<String> {
        let matches = self.scan_page(image);
        self.build_context(&matches)
    }

    /// Classify the age/gender demographics in an image crop using WD Tagger.
    /// Returns a label like "Young Male", "Adult Female", or `None` if the model is not loaded.
    pub fn classify_region(&self, image: &DynamicImage) -> Option<String> {
        let wd = self.wd_tagger.as_ref()?;
        classify_with_wd_tagger(wd, image).ok()
    }

    // ─── Private ──────────────────────────────────────────────────────────────

    fn save_locked(&self, entries: &[CharacterEntry]) -> Result<()> {
        if let Some(parent) = self.lib_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(entries)?;
        std::fs::write(&self.lib_path, json)?;
        Ok(())
    }
}

// ─── ONNX helpers ─────────────────────────────────────────────────────────────

fn load_session_opt(path: PathBuf) -> Option<Session> {
    if !path.exists() {
        return None;
    }
    match Session::builder().and_then(|mut b| b.commit_from_file(&path)) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "failed to load ONNX session");
            None
        }
    }
}

// ─── CCIP embedding ───────────────────────────────────────────────────────────

/// Embed a batch of face crops in a single ONNX inference call.
///
/// Stacks all `crops` into a `[N, 3, 384, 384]` tensor, runs one inference,
/// and returns N individually L2-normalised 512-dim embedding vectors.
///
/// Returns an error if the session rejects batch size > 1, allowing the caller
/// to fall back to sequential `embed_face` calls.
fn embed_faces_batch(ccip: &Mutex<Session>, crops: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
    if crops.is_empty() {
        return Ok(Vec::new());
    }

    let n = crops.len();
    let (h, w) = (CCIP_SIZE as usize, CCIP_SIZE as usize);
    let mut data = vec![0f32; n * 3 * h * w];

    for (idx, image) in crops.iter().enumerate() {
        let resized = image.resize_exact(CCIP_SIZE, CCIP_SIZE, imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let offset = idx * 3 * h * w;
        for y in 0..h {
            for x in 0..w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    data[offset + c * h * w + y * w + x] = (val - CCIP_MEAN[c]) / CCIP_STD[c];
                }
            }
        }
    }

    let array = Array::from_shape_vec([n, 3, h, w], data)?;
    let input_tensor = Tensor::from_array(array)?;

    let mut session = ccip
        .lock()
        .map_err(|_| anyhow::anyhow!("CCIP mutex poisoned"))?;
    let input_name = session.inputs()[0].name().to_string();
    let outputs = session.run(ort::inputs! { input_name.as_str() => input_tensor })?;

    let out = outputs[0].try_extract_array::<f32>()?;
    let view = out.view();

    // Output shape should be [N, 512]. Split into N rows, each L2-normalised.
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let raw: Vec<f32> = (0..512).map(|j| view[[i, j]]).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        result.push(raw.into_iter().map(|x| x / norm).collect());
    }

    Ok(result)
}

fn embed_face(ccip: &Mutex<Session>, image: &DynamicImage) -> Result<Vec<f32>> {
    let resized = image.resize_exact(CCIP_SIZE, CCIP_SIZE, imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let (w, h) = (CCIP_SIZE as usize, CCIP_SIZE as usize);
    let mut data = vec![0f32; 3 * h * w];

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                data[c * h * w + y * w + x] = (val - CCIP_MEAN[c]) / CCIP_STD[c];
            }
        }
    }

    let array = Array::from_shape_vec([1usize, 3, h, w], data)?;
    let input_tensor = Tensor::from_array(array)?;

    let mut session = ccip
        .lock()
        .map_err(|_| anyhow::anyhow!("CCIP mutex poisoned"))?;
    let input_name = session.inputs()[0].name().to_string();
    let outputs = session.run(ort::inputs! { input_name.as_str() => input_tensor })?;

    let out = outputs[0].try_extract_array::<f32>()?;
    let raw: Vec<f32> = out.view().iter().cloned().collect();

    // L2 normalise so cosine similarity = dot product
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    Ok(raw.into_iter().map(|x| x / norm).collect())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn find_best_match(entries: &[CharacterEntry], embedding: &[f32]) -> FaceMatch {
    let mut best_sim = -1.0f32;
    let mut second_sim = -1.0f32;
    let mut best_entry: Option<&CharacterEntry> = None;

    for entry in entries {
        if entry.embedding.len() == embedding.len() {
            let sim = cosine_similarity(&entry.embedding, embedding);
            if sim > best_sim {
                second_sim = best_sim;
                best_sim = sim;
                best_entry = Some(entry);
            } else if sim > second_sim {
                second_sim = sim;
            }
        }
    }

    if let Some(entry) = best_entry {
        // Bypass margin check when the best score is very high: at ≥ 0.88 the
        // similarity alone is strong evidence of identity, even if the 2nd-best
        // is close.  Below that threshold still require a clear margin gap.
        let margin_ok = entries.len() < 2
            || best_sim >= HIGH_CONF_THRESHOLD
            || (best_sim - second_sim) >= MATCH_MARGIN;
        tracing::info!(
            best = best_sim,
            second = second_sim,
            margin = best_sim - second_sim,
            margin_ok,
            "CCIP match scores"
        );
        if best_sim >= MATCH_THRESHOLD && margin_ok {
            return FaceMatch {
                name: entry.name.clone(),
                traits: entry.traits.clone(),
                relations: entry.relations.clone(),
                confidence: best_sim,
                is_known: true,
            };
        }
    }

    FaceMatch {
        name: "Unknown character".to_string(),
        traits: Vec::new(),
        relations: Vec::new(),
        confidence: best_sim.max(0.0),
        is_known: false,
    }
}

// ─── Face detection ───────────────────────────────────────────────────────────

fn detect_faces(face_det: &Mutex<Session>, image: &DynamicImage) -> Result<Vec<FaceBox>> {
    detect_faces_threshold(face_det, image, FACE_CONF_THRESHOLD)
}

fn detect_faces_threshold(face_det: &Mutex<Session>, image: &DynamicImage, threshold: f32) -> Result<Vec<FaceBox>> {
    let orig_w = image.width();
    let orig_h = image.height();
    let scale_x = orig_w as f32 / FACE_SIZE as f32;
    let scale_y = orig_h as f32 / FACE_SIZE as f32;

    let resized = image.resize_exact(FACE_SIZE, FACE_SIZE, imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let (w, h) = (FACE_SIZE as usize, FACE_SIZE as usize);

    let mut data = vec![0f32; 3 * h * w];
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let base = y as usize * w + x as usize;
        data[base] = pixel[0] as f32 / 255.0;
        data[h * w + base] = pixel[1] as f32 / 255.0;
        data[2 * h * w + base] = pixel[2] as f32 / 255.0;
    }

    let array = Array::from_shape_vec([1usize, 3, h, w], data)?;
    let input_tensor = Tensor::from_array(array)?;

    let mut session = face_det
        .lock()
        .map_err(|_| anyhow::anyhow!("face detector mutex poisoned"))?;
    let input_name = session.inputs()[0].name().to_string();
    let outputs = session.run(ort::inputs! { input_name.as_str() => input_tensor })?;

    // Support both common YOLOv8 output shapes:
    //   [1, 5, N] — default YOLOv8 export (cx,cy,w,h,conf)
    //   [1, N, 5] — transposed variant
    let out = outputs[0].try_extract_array::<f32>()?;
    let view = out.view();
    let shape = view.shape().to_vec();

    let mut candidates: Vec<FaceBox> = Vec::new();

    if shape.len() == 3 {
        let (dim1, dim2) = (shape[1], shape[2]);

        if dim1 >= 5 {
            // [1, >=5, N]
            let n = dim2;
            for i in 0..n {
                let conf = view[[0, 4, i]];
                if conf < threshold {
                    continue;
                }
                let (cx, cy, bw, bh) = (view[[0, 0, i]], view[[0, 1, i]], view[[0, 2, i]], view[[0, 3, i]]);
                candidates.push(scale_box(cx, cy, bw, bh, conf, scale_x, scale_y, orig_w, orig_h));
            }
        } else if dim2 >= 5 {
            // [1, N, >=5]
            let n = dim1;
            for i in 0..n {
                let conf = view[[0, i, 4]];
                if conf < threshold {
                    continue;
                }
                let (cx, cy, bw, bh) = (view[[0, i, 0]], view[[0, i, 1]], view[[0, i, 2]], view[[0, i, 3]]);
                candidates.push(scale_box(cx, cy, bw, bh, conf, scale_x, scale_y, orig_w, orig_h));
            }
        }
    }

    Ok(nms_faces(candidates))
}

fn scale_box(
    cx: f32, cy: f32, bw: f32, bh: f32, score: f32,
    sx: f32, sy: f32, img_w: u32, img_h: u32,
) -> FaceBox {
    FaceBox {
        x: ((cx - bw / 2.0) * sx).max(0.0),
        y: ((cy - bh / 2.0) * sy).max(0.0),
        width: (bw * sx).min(img_w as f32),
        height: (bh * sy).min(img_h as f32),
        score,
    }
}

fn nms_boxes(mut candidates: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut suppressed = vec![false; candidates.len()];
    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..candidates.len() {
            if suppressed[j] {
                continue;
            }
            if iou_face(&candidates[i], &candidates[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    candidates
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !suppressed[*i])
        .map(|(_, c)| c)
        .collect()
}

fn nms_faces(candidates: Vec<FaceBox>) -> Vec<FaceBox> {
    nms_boxes(candidates, FACE_IOU_THRESHOLD)
}

fn iou_face(a: &FaceBox, b: &FaceBox) -> f32 {
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = (a.x + a.width).min(b.x + b.width);
    let iy2 = (a.y + a.height).min(b.y + b.height);
    if ix2 <= ix1 || iy2 <= iy1 {
        return 0.0;
    }
    let inter = (ix2 - ix1) * (iy2 - iy1);
    inter / (a.width * a.height + b.width * b.height - inter)
}

fn crop_face(image: &DynamicImage, bbox: &FaceBox) -> DynamicImage {
    let mx = bbox.width * FACE_CROP_MARGIN;
    let my = bbox.height * FACE_CROP_MARGIN;
    let x = (bbox.x - mx).max(0.0) as u32;
    let y = (bbox.y - my).max(0.0) as u32;
    let w = ((bbox.width + 2.0 * mx) as u32).min(image.width().saturating_sub(x)).max(1);
    let h = ((bbox.height + 2.0 * my) as u32).min(image.height().saturating_sub(y)).max(1);
    image.crop_imm(x, y, w, h)
}

// ─── Debug scan visualisation ─────────────────────────────────────────────────

/// One entry per balloon block: the face that was selected (if any) plus
/// the geometry used for debugging.
struct DebugScanEntry {
    block_idx:    usize,
    char_name:    String,
    balloon_box:  (f32, f32, f32, f32), // (x, y, w, h) pixels
    tail_dir:     Option<(f32, f32)>,
    face_box:     Option<FaceBox>,      // None when no candidate face was found
}

/// When the environment variable `KOHARU_DEBUG_SCAN` is set, save two kinds
/// of files under `<tmp>/koharu-debug-scan/`:
///
/// * `overview.png` — the full page with all face boxes (cyan) and balloon
///   boxes (magenta) drawn, plus a coloured arrow showing the detected tail
///   direction (green = used for ray projection, red = wrong side fallback).
///   Each selected face box is also outlined in yellow.
///
/// * `{N:02}-{char_name}.png` — the individual face crop selected for block #N.
fn save_debug_scan(
    image: &DynamicImage,
    face_data: &[(FaceBox, f32, f32, Vec<f32>)],
    entries: &[DebugScanEntry],
    panels: &[(f32, f32, f32, f32)],
) {
    let dir = PathBuf::from("debug-scan");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("debug-scan: cannot create dir {:?}: {e}", dir);
        return;
    }

    // ── overview image ──────────────────────────────────────────────────────
    let mut overview = image.to_rgb8();
    let (img_w, img_h) = (overview.width() as i32, overview.height() as i32);

    // Draw all face boxes in cyan.
    let cyan    = Rgb([0u8,   220u8, 220u8]);
    let yellow  = Rgb([255u8, 220u8, 0u8]);
    let magenta = Rgb([220u8, 0u8,   220u8]);
    let green   = Rgb([0u8,   220u8, 0u8]);
    let red     = Rgb([220u8, 0u8,   0u8]);

    // Panel colour palette — 10 distinct colours, cycling for more panels.
    let panel_colors: [Rgb<u8>; 10] = [
        Rgb([255, 80,  80 ]),  // 0 red
        Rgb([255, 165, 0  ]),  // 1 orange
        Rgb([255, 255, 0  ]),  // 2 yellow
        Rgb([0,   200, 60 ]),  // 3 green
        Rgb([0,   180, 255]),  // 4 sky-blue
        Rgb([0,   80,  255]),  // 5 blue
        Rgb([160, 0,   255]),  // 6 violet
        Rgb([255, 0,   180]),  // 7 pink
        Rgb([255, 255, 255]),  // 8 white
        Rgb([180, 255, 180]),  // 9 mint
    ];

    // Draw panel outlines (thick = 3px) before everything else so they appear
    // behind face/balloon boxes.
    for (pi, (px, py, pw, ph)) in panels.iter().enumerate() {
        let col = panel_colors[pi % panel_colors.len()];
        for off in 0i32..3 {
            let rx = (*px as i32 + off).max(0);
            let ry = (*py as i32 + off).max(0);
            let rw = ((*pw as i32 - 2 * off) as u32)
                .min(overview.width().saturating_sub(rx as u32));
            let rh = ((*ph as i32 - 2 * off) as u32)
                .min(overview.height().saturating_sub(ry as u32));
            if rw > 0 && rh > 0 {
                draw_hollow_rect_mut(&mut overview, Rect::at(rx, ry).of_size(rw, rh), col);
            }
        }
        // Small filled 12×12 square at the top-left corner of the panel so
        // the panel index colour is easy to spot.
        let sq_x = (*px as i32).max(0) as u32;
        let sq_y = (*py as i32).max(0) as u32;
        let sq_w = 12u32.min(overview.width().saturating_sub(sq_x));
        let sq_h = 12u32.min(overview.height().saturating_sub(sq_y));
        if sq_w > 0 && sq_h > 0 {
            for dy in 0..sq_h {
                for dx in 0..sq_w {
                    overview.put_pixel(sq_x + dx, sq_y + dy, col);
                }
            }
        }
    }

    // Log colour → panel index mapping so user can read the numbers.
    tracing::info!(
        mapping = ?panels.iter().enumerate()
            .map(|(i, (x,y,w,h))| format!("P{i}@({x:.0},{y:.0} {w:.0}×{h:.0})"))
            .collect::<Vec<_>>(),
        "debug-scan: panel colour index (P0=red P1=orange P2=yellow P3=green P4=sky-blue P5=blue P6=violet P7=pink)"
    );

    for (fb, _, _, _) in face_data {
        let r = Rect::at(fb.x as i32, fb.y as i32)
            .of_size(fb.width as u32, fb.height as u32);
        draw_hollow_rect_mut(&mut overview, r, cyan);
    }

    // Collect selected face indices.
    let selected_face_boxes: Vec<Option<&FaceBox>> =
        entries.iter().map(|e| e.face_box.as_ref()).collect();

    // Draw selected faces in yellow (on top of cyan).
    for e in entries {
        if let Some(fb) = &e.face_box {
            // Draw twice to make the yellow box slightly thicker.
            for off in 0i32..=1 {
                let rx = (fb.x as i32 - off).max(0);
                let ry = (fb.y as i32 - off).max(0);
                let rw = ((fb.width as i32 + 2 * off) as u32).min(overview.width().saturating_sub(rx as u32));
                let rh = ((fb.height as i32 + 2 * off) as u32).min(overview.height().saturating_sub(ry as u32));
                if rw > 0 && rh > 0 {
                    draw_hollow_rect_mut(
                        &mut overview,
                        Rect::at(rx, ry).of_size(rw, rh),
                        yellow,
                    );
                }
            }
        }
    }

    // Draw balloon boxes in magenta and tail arrows.
    for e in entries {
        let (bx, by, bw, bh) = e.balloon_box;
        let bw_u = bw as u32;
        let bh_u = bh as u32;
        if bw_u > 0 && bh_u > 0 {
            draw_hollow_rect_mut(
                &mut overview,
                Rect::at(bx as i32, by as i32).of_size(bw_u, bh_u),
                magenta,
            );
        }

        // Tail arrow from balloon centre in tail direction.
        if let Some((dx, dy)) = e.tail_dir {
            let cx = bx + bw / 2.0;
            let cy = by + bh / 2.0;
            let arrow_len = (bw.min(bh) * 0.4).max(20.0);
            let ex = (cx + dx * arrow_len).clamp(0.0, img_w as f32 - 1.0);
            let ey = (cy + dy * arrow_len).clamp(0.0, img_h as f32 - 1.0);

            // Green if the face-in-tail-direction check passed, red if fallback.
            let used_ray = e.face_box.as_ref().map(|fb| {
                let fcx = fb.x + fb.width / 2.0;
                let fcy = fb.y + fb.height / 2.0;
                (fcx - cx) * dx + (fcy - cy) * dy > 0.0
            }).unwrap_or(false);
            let arrow_col = if used_ray { green } else { red };

            for i in -1i32..=1 {
                let ox = if dy.abs() > 0.5 { i as f32 } else { 0.0 };
                let oy = if dx.abs() > 0.5 { i as f32 } else { 0.0 };
                draw_line_segment_mut(
                    &mut overview,
                    (cx + ox, cy + oy),
                    (ex + ox, ey + oy),
                    arrow_col,
                );
            }
        }
    }

    let overview_path = dir.join("overview.png");
    if let Err(e) = overview.save(&overview_path) {
        tracing::warn!("debug-scan: cannot save overview: {e}");
    } else {
        tracing::info!("debug-scan: saved {:?}", overview_path);
    }

    // ── individual face crops ───────────────────────────────────────────────
    let _ = selected_face_boxes; // suppress unused warning
    for e in entries {
        if let Some(fb) = &e.face_box {
            let crop = crop_face(image, fb);
            let safe_name = e.char_name.replace(['/', '\\', ' '], "_");
            let fname = format!("{:02}-{}.png", e.block_idx + 1, safe_name);
            let path = dir.join(&fname);
            if let Err(err) = crop.save(&path) {
                tracing::warn!("debug-scan: cannot save {fname}: {err}");
            }
        }
    }

    tracing::info!("debug-scan: {} files saved to {:?}", entries.len() + 1, dir);
}

// ─── Gender classification ────────────────────────────────────────────────────

/// Classify the gender of an anime face crop using DOFOFFICIAL/animeGender-dvgg-0.8-alpha.
///
// ─── WD Tagger (gender + age from danbooru tags) ─────────────────────────────

/// Load WD Tagger model + parse tags CSV to build pre-computed tag index lists.
/// Returns `None` if either file is missing.
fn load_wd_tagger() -> Option<WdTaggerModel> {
    let model_path = wd_tagger_model_path();
    let tags_path = wd_tagger_tags_path();

    let session = match load_session_opt(model_path) {
        Some(s) => s,
        None => {
            tracing::info!("WD Tagger model not found — unknown characters shown as 'Unknown character'. Download wd_tagger.onnx + wd_tagger_tags.csv to ~/.cache/koharu/models/");
            return None;
        }
    };

    let csv = match std::fs::read_to_string(&tags_path) {
        Ok(c) => c,
        Err(_) => {
            tracing::info!("WD Tagger tags CSV not found ({}), tagger disabled.", tags_path.display());
            return None;
        }
    };

    // Parse CSV: skip header row, column 1 (0-based) is the tag name.
    // Row index (0-based after header) = output tensor index.
    let mut male_indices = Vec::new();
    let mut female_indices = Vec::new();
    let mut child_indices = Vec::new();
    let mut teen_indices = Vec::new();
    let mut elder_indices = Vec::new();

    for (row_idx, line) in csv.lines().skip(1).enumerate() {
        let parts: Vec<&str> = line.splitn(4, ',').collect();
        if parts.len() < 2 {
            continue;
        }
        // Support both "tag_id,name,..." and "name,category,..." formats.
        // Try index 1 first (most common: tag_id,name,...); if it looks numeric, use index 0.
        let name = if parts[0].trim().parse::<u64>().is_ok() {
            parts[1].trim()
        } else {
            parts[0].trim()
        };

        if MALE_TAGS.contains(&name) {
            male_indices.push(row_idx);
        }
        if FEMALE_TAGS.contains(&name) {
            female_indices.push(row_idx);
        }
        if CHILD_AGE_TAGS.contains(&name) {
            child_indices.push(row_idx);
        }
        if TEEN_AGE_TAGS.contains(&name) {
            teen_indices.push(row_idx);
        }
        if ELDER_AGE_TAGS.contains(&name) {
            elder_indices.push(row_idx);
        }
    }

    tracing::info!(
        male_tags = male_indices.len(),
        female_tags = female_indices.len(),
        child_tags = child_indices.len(),
        teen_tags = teen_indices.len(),
        elder_tags = elder_indices.len(),
        "WD Tagger loaded"
    );

    Some(WdTaggerModel {
        session: Mutex::new(session),
        male_indices,
        female_indices,
        child_indices,
        teen_indices,
        elder_indices,
    })
}

/// Classify gender (and optionally age group) using WD Tagger.
///
/// Input: character crop (face + body context preferred).
/// Output: label like "Male character", "Female character",
///         "Young Male", "Mature Female", etc.
///
/// WD Tagger (convnext v2/v3) expects:
///   - Input shape: [1, 448, 448, 3] NHWC, dtype float32
///   - Pixel values: BGR, range [0, 255]
///   - Output: [1, N_TAGS] probabilities (sigmoid)
fn classify_with_wd_tagger(model: &WdTaggerModel, crop: &DynamicImage) -> Result<String> {
    let s = WD_TAGGER_SIZE as usize;

    // Resize to square, pad with white if needed (WD Tagger convention).
    let resized = {
        let (w, h) = (crop.width(), crop.height());
        let max_side = w.max(h);
        let mut canvas = image::RgbImage::from_pixel(max_side, max_side, image::Rgb([255u8, 255, 255]));
        let offset_x = (max_side - w) / 2;
        let offset_y = (max_side - h) / 2;
        image::imageops::overlay(&mut canvas, &crop.to_rgb8(), offset_x as i64, offset_y as i64);
        image::DynamicImage::ImageRgb8(canvas)
            .resize_exact(WD_TAGGER_SIZE, WD_TAGGER_SIZE, imageops::FilterType::Lanczos3)
            .to_rgb8()
    };

    // Build NHWC float32 tensor: [1, H, W, 3] with BGR values in [0, 255].
    let mut data = vec![0f32; s * s * 3];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let base = (y as usize * s + x as usize) * 3;
        data[base]     = pixel[2] as f32; // B
        data[base + 1] = pixel[1] as f32; // G
        data[base + 2] = pixel[0] as f32; // R
    }

    let array = ndarray::Array::from_shape_vec([1usize, s, s, 3], data)?;
    let input_tensor = ort::value::Tensor::from_array(array)?;

    let mut session = model.session.lock()
        .map_err(|_| anyhow::anyhow!("WD Tagger mutex poisoned"))?;
    let input_name = session.inputs()[0].name().to_string();
    let outputs = session.run(ort::inputs! { input_name.as_str() => input_tensor })?;

    let out = outputs[0].try_extract_array::<f32>()?;
    let probs: Vec<f32> = out.view().iter().cloned().collect();

    // Score for each category = max probability among matching tag indices.
    let score = |indices: &[usize]| -> f32 {
        indices.iter().map(|&i| probs.get(i).copied().unwrap_or(0.0)).fold(0.0_f32, f32::max)
    };

    let male_score   = score(&model.male_indices);
    let female_score = score(&model.female_indices);
    let child_score  = score(&model.child_indices);
    let teen_score   = score(&model.teen_indices);
    let elder_score  = score(&model.elder_indices);

    tracing::info!(
        male = male_score, female = female_score,
        child = child_score, teen = teen_score, elder = elder_score,
        "WD Tagger scores"
    );

    let gender = if male_score >= female_score { "Male" } else { "Female" };

    // Age threshold is intentionally lower than gender — age tags are sparser
    // in danbooru so they get lower confidence scores overall.
    const AGE_THRESHOLD: f32 = 0.15;

    // Determine age group from the highest-scoring age tag (above threshold).
    let age_label = if child_score >= AGE_THRESHOLD && child_score >= teen_score && child_score >= elder_score {
        Some("Young")
    } else if teen_score >= AGE_THRESHOLD && teen_score >= elder_score {
        Some("Teenage")
    } else if elder_score >= AGE_THRESHOLD {
        Some("Mature")
    } else {
        None
    };

    let label = match age_label {
        Some(age) => format!("{age} {gender}"),
        None => format!("{gender} character"),
    };

    Ok(label)
}

/// Crop with modest body context below and around the face box for WD Tagger.
/// Expands: 0.5× width each side, 1.5× height downward, 0.3× upward.
/// Kept intentionally tight to avoid capturing neighbouring characters in the
/// same panel, which would confuse gender/age classification.
fn crop_face_expanded(image: &DynamicImage, fb: &FaceBox) -> DynamicImage {
    let img_w = image.width() as f32;
    let img_h = image.height() as f32;
    let margin_x = fb.width * 0.5;
    let margin_y_up = fb.height * 0.3;
    let margin_y_down = fb.height * 1.5;
    let x = (fb.x - margin_x).max(0.0) as u32;
    let y = (fb.y - margin_y_up).max(0.0) as u32;
    let x2 = ((fb.x + fb.width + margin_x).min(img_w)) as u32;
    let y2 = ((fb.y + fb.height + margin_y_down).min(img_h)) as u32;
    let w = (x2 - x).max(1);
    let h = (y2 - y).max(1);
    image.crop_imm(x, y, w, h)
}

// ─── Manga panel detection (ML) ───────────────────────────────────────────────

/// Detect manga panels using the `leoxs22/manga-panel-detector-yolo26n` model.
///
/// The model has **2 classes**: class 0 = panel, class 1 = text bubble.
/// This model outputs post-NMS results in shape [1, N, 6]:
///   - indices 0-3: (x1, y1, x2, y2) corner coordinates in the 640×640 input space
///   - index 4:     confidence score
///   - index 5:     class_id (0 = panel, 1 = text/speech-bubble)
/// Since NMS is already applied by the ONNX graph, no second NMS pass is needed.
/// Sort panels in manga reading order: top-to-bottom rows, right-to-left within each row.
/// Two panels are in the same row when their vertical ranges overlap.
fn sort_manga_reading_order(mut panels: Vec<(f32, f32, f32, f32)>) -> Vec<(f32, f32, f32, f32)> {
    if panels.len() <= 1 {
        return panels;
    }

    // Sort by top-y first so we can group rows in one pass.
    panels.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Group into rows: a panel joins the current row if it overlaps vertically
    // with the row's accumulated y-range.
    let mut rows: Vec<Vec<(f32, f32, f32, f32)>> = Vec::new();
    let mut row_y_min = f32::MAX;
    let mut row_y_max = f32::MIN;

    for panel in panels {
        let (_, y, _, h) = panel;
        let p_top = y;
        let p_bot = y + h;

        let overlaps = p_top < row_y_max && p_bot > row_y_min;
        if overlaps {
            // Extend current row's y-range.
            row_y_min = row_y_min.min(p_top);
            row_y_max = row_y_max.max(p_bot);
            rows.last_mut().unwrap().push(panel);
        } else {
            row_y_min = p_top;
            row_y_max = p_bot;
            rows.push(vec![panel]);
        }
    }

    // Within each row, sort right to left (center_x descending).
    for row in &mut rows {
        row.sort_by(|a, b| {
            let cx_a = a.0 + a.2 / 2.0;
            let cx_b = b.0 + b.2 / 2.0;
            cx_b.partial_cmp(&cx_a).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    rows.into_iter().flatten().collect()
}

fn detect_panels_ml(
    panel_det: &Mutex<Session>,
    image: &DynamicImage,
) -> Result<Vec<(f32, f32, f32, f32)>> {
    let orig_w = image.width();
    let orig_h = image.height();
    let scale_x = orig_w as f32 / PANEL_SIZE as f32;
    let scale_y = orig_h as f32 / PANEL_SIZE as f32;

    let resized = image.resize_exact(PANEL_SIZE, PANEL_SIZE, imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let (w, h) = (PANEL_SIZE as usize, PANEL_SIZE as usize);

    let mut data = vec![0f32; 3 * h * w];
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let base = y as usize * w + x as usize;
        data[base] = pixel[0] as f32 / 255.0;
        data[h * w + base] = pixel[1] as f32 / 255.0;
        data[2 * h * w + base] = pixel[2] as f32 / 255.0;
    }

    let array = Array::from_shape_vec([1usize, 3, h, w], data)?;
    let input_tensor = Tensor::from_array(array)?;

    let mut session = panel_det
        .lock()
        .map_err(|_| anyhow::anyhow!("panel detector mutex poisoned"))?;
    let input_name = session.inputs()[0].name().to_string();
    let outputs = session.run(ort::inputs! { input_name.as_str() => input_tensor })?;

    let out = outputs[0].try_extract_array::<f32>()?;
    let view = out.view();
    let shape = view.shape().to_vec();

    tracing::info!(shape = ?shape, "panel detector ONNX output shape");

    let mut candidates: Vec<FaceBox> = Vec::new();

    if shape.len() == 3 {
        let (dim1, dim2) = (shape[1], shape[2]);

        // Determine layout: [1, attrs, N] vs [1, N, attrs]
        let (num_attrs, num_dets, transposed) = if dim1 >= 5 && dim2 > dim1 {
            (dim1, dim2, false)
        } else if dim2 >= 5 && dim1 > dim2 {
            (dim2, dim1, true)
        } else if dim1 >= 5 {
            (dim1, dim2, false)
        } else {
            (dim2, dim1, true)
        };

        tracing::info!(num_attrs, num_dets, transposed, "panel detector layout parsed");

        let get = |attr: usize, i: usize| -> f32 {
            if transposed {
                view[[0, i, attr]]
            } else {
                view[[0, attr, i]]
            }
        };

        for i in 0..num_dets {
            // Index 4 = confidence score (post-NMS format).
            let conf = get(4, i);
            if conf < PANEL_CONF_THRESHOLD {
                continue;
            }

            // Index 5 = class_id: 0 = panel, 1 = text/speech-bubble. Skip non-panels.
            let class_id = get(5, i).round() as i32;
            if class_id != 0 {
                continue;
            }

            // Coordinates are corner format (x1, y1, x2, y2) scaled to [0, PANEL_SIZE].
            let (x1, y1, x2, y2) = (get(0, i), get(1, i), get(2, i), get(3, i));
            let x = (x1 * scale_x).max(0.0);
            let y = (y1 * scale_y).max(0.0);
            let w = ((x2 - x1) * scale_x).min(orig_w as f32);
            let h = ((y2 - y1) * scale_y).min(orig_h as f32);
            if w > 0.0 && h > 0.0 {
                candidates.push(FaceBox { x, y, width: w, height: h, score: conf });
            }
        }
    }

    tracing::info!(
        panel_count = candidates.len(),
        "panel detector final result (post-NMS from model)"
    );

    // Model already applied NMS internally — no second pass needed.
    Ok(candidates.into_iter().map(|b| (b.x, b.y, b.width, b.height)).collect())
}

// ─── Manga panel detection (heuristic fallback) ───────────────────────────────

/// Detect manga panel bounding boxes via gutter-line analysis.
///
/// Scans every row and every column for lines where most pixels are light
/// (the white gutters that separate manga panels).  Returns a list of
/// `(x, y, width, height)` rectangles in pixels.
///
/// Works well for standard black-and-white manga.  For pages with heavy
/// dark backgrounds, the heuristic may fall back to treating the whole page
/// as a single panel (acceptable — panel isolation simply has no effect).
fn detect_manga_panels(image: &DynamicImage) -> Vec<(f32, f32, f32, f32)> {
    let gray = image.to_luma8();
    let (iw, ih) = (gray.width() as usize, gray.height() as usize);

    const LIGHT: u8 = 200;
    const GUTTER_FRAC: f32 = 0.80; // slightly relaxed for color manga
    const MIN_PANEL_FRAC: f32 = 0.08;

    let min_h = ((ih as f32) * MIN_PANEL_FRAC) as usize;
    let min_w = ((iw as f32) * MIN_PANEL_FRAC) as usize;

    // Step 1: find horizontal splits across the full page width.
    let h_is_gutter: Vec<bool> = (0..ih)
        .map(|y| {
            let light = (0..iw)
                .filter(|&x| gray.get_pixel(x as u32, y as u32).0[0] >= LIGHT)
                .count();
            light as f32 / iw as f32 >= GUTTER_FRAC
        })
        .collect();
    let h_splits = gutter_splits(&h_is_gutter, ih);

    let mut panels = Vec::new();

    // Step 2: for each horizontal band find vertical splits WITHIN THAT BAND.
    // This is the critical fix: manga pages have irregular layouts where different
    // rows have different numbers of panels (e.g. top row has 2, bottom row has 1).
    // A global vertical split would incorrectly slice the full-width panels.
    for row in h_splits.windows(2) {
        let (y1, y2) = (row[0], row[1]);
        if y2 - y1 < min_h {
            continue;
        }

        let band_h = y2 - y1;
        let v_is_gutter: Vec<bool> = (0..iw)
            .map(|x| {
                let light = (y1..y2)
                    .filter(|&y| gray.get_pixel(x as u32, y as u32).0[0] >= LIGHT)
                    .count();
                light as f32 / band_h as f32 >= GUTTER_FRAC
            })
            .collect();
        let v_splits = gutter_splits(&v_is_gutter, iw);

        for col in v_splits.windows(2) {
            let (x1, x2) = (col[0], col[1]);
            if x2 - x1 < min_w {
                continue;
            }
            panels.push((x1 as f32, y1 as f32, (x2 - x1) as f32, (y2 - y1) as f32));
        }
    }

    panels
}

/// Score for selecting a face based on its proximity to a ray (balloon tail direction).
///
/// The ray starts at the balloon centre (ox, oy) and extends in direction (dx, dy).
/// Returns the perpendicular distance from the face centre to the ray, plus a small
/// forward-distance weight so that closer faces are preferred when perpendicular
/// distances are similar.
/// Faces *behind* the ray origin (wrong direction) are penalised by +100 000.
/// Ray-AABB intersection using the slab method.
/// Ray origin (ox, oy), direction (dx, dy) — must be unit-length.
/// Rectangle (rx, ry, rw, rh) as top-left + size.
/// Returns true when the ray hits the rectangle (including starting inside it).
fn ray_intersects_rect(ox: f32, oy: f32, dx: f32, dy: f32, rx: f32, ry: f32, rw: f32, rh: f32) -> bool {
    // If origin is already inside the rect, count as a hit.
    if ox >= rx && ox <= rx + rw && oy >= ry && oy <= ry + rh {
        return true;
    }
    let inv_dx = if dx.abs() > 1e-9 { 1.0 / dx } else { f32::INFINITY };
    let inv_dy = if dy.abs() > 1e-9 { 1.0 / dy } else { f32::INFINITY };
    let tx1 = (rx - ox) * inv_dx;
    let tx2 = (rx + rw - ox) * inv_dx;
    let ty1 = (ry - oy) * inv_dy;
    let ty2 = (ry + rh - oy) * inv_dy;
    let t_min = tx1.min(tx2).max(ty1.min(ty2));
    let t_max = tx1.max(tx2).min(ty1.max(ty2));
    t_max >= 0.0 && t_min <= t_max
}

#[allow(dead_code)]
fn point_to_ray_score(px: f32, py: f32, ox: f32, oy: f32, dx: f32, dy: f32) -> f32 {
    let rel_x = px - ox;
    let rel_y = py - oy;
    let along = rel_x * dx + rel_y * dy; // signed distance along ray
    if along < 0.0 {
        // Wrong side of the balloon — penalise heavily
        return (rel_x * rel_x + rel_y * rel_y).sqrt() + 100_000.0;
    }
    let perp_x = rel_x - along * dx;
    let perp_y = rel_y - along * dy;
    let perp = (perp_x * perp_x + perp_y * perp_y).sqrt();
    // Weight distance more heavily: nearby faces beat far-away on-axis faces.
    // perp keeps angular alignment but along * 0.4 ensures a face 100px away
    // on the ray loses to a face 10px off-axis at 30px distance.
    perp + along * 0.4
}

/// Distance from a point (px, py) to the nearest edge of rectangle (rx,ry,rw,rh).
/// Returns 0.0 when the point is inside the rectangle.
/// Unlike rect_rect_dist this never produces spurious ties from overlapping boxes.
fn point_to_rect_dist(px: f32, py: f32, rx: f32, ry: f32, rw: f32, rh: f32) -> f32 {
    let dx = (rx - px).max(0.0) + (px - (rx + rw)).max(0.0);
    let dy = (ry - py).max(0.0) + (py - (ry + rh)).max(0.0);
    (dx * dx + dy * dy).sqrt()
}

/// Minimum distance between two axis-aligned rectangles (given as x,y,w,h).
/// Returns 0.0 when the rectangles overlap or touch.
#[allow(dead_code)]
fn rect_rect_dist(ax: f32, ay: f32, aw: f32, ah: f32, bx: f32, by: f32, bw: f32, bh: f32) -> f32 {
    let dx = (ax.max(bx) - (ax + aw).min(bx + bw)).max(0.0);
    let dy = (ay.max(by) - (ay + ah).min(by + bh)).max(0.0);
    (dx * dx + dy * dy).sqrt()
}

// ─── Balloon tail detection ────────────────────────────────────────────────────

/// Detect the exit direction of a speech-bubble tail from its bounding box.
///
/// # How it works
///
/// A manga speech-bubble tail is a narrow triangular protrusion.  The tip of
/// the tail sits at (or very near) one edge of the YOLO bounding box, because
/// the detector fits a tight box around the whole balloon including its tail.
///
/// **Algorithm**: for each of the 4 edges (top/bottom/left/right) scan rows
/// (or columns) from the edge *inward*, one at a time.  For the first few rows
/// that contain any dark border pixels (luminance ≤ DARK), record the *span*
/// (rightmost − leftmost dark pixel in that row).  Take the minimum span seen.
///
/// * **Tail side** — the very first occupied row from the outside is the tail
///   tip: only 1–5 dark pixels wide → minimum span ≈ 1–5 px / box_w ≈ 0.005–0.02
/// * **Body sides** — the first occupied row from the outside is the body
///   border: spans 50–90 % of the balloon dimension → minimum span ≈ 0.5–0.9
///
/// The edge with the smallest minimum span (< `TAIL_WIDTH_RATIO`) that is also
/// much narrower than all other edges (< `TAIL_VS_BODY_RATIO` × widest) is the
/// tail direction.
///
/// Returns `None` when no clear tail is found (symmetric thought-bubbles,
/// narration boxes, very small balloons, low-contrast scans).
fn detect_balloon_tail_direction(
    gray: &image::GrayImage,
    bx: f32,
    by: f32,
    bw: f32,
    bh: f32,
) -> Option<(f32, f32)> {
    /// Luminance threshold — pixel is "dark border" when luma ≤ this.
    const DARK: u8 = 80;
    /// Number of outermost occupied rows/cols to sample per edge.
    const SCAN_ROWS: i32 = 5;
    /// Tail must be WIDER than this fraction of the balloon dimension.
    /// Filters out oval-crown artefacts (1–2 dark pixels = the very tip of the
    /// rounded balloon body) which would otherwise be mistaken for a tail tip.
    const MIN_TAIL_SPAN: f32 = 0.015; // ≈ 3–5 px on a typical 200–300 px balloon
    /// Tail must be narrower than this fraction of the balloon dimension.
    const TAIL_WIDTH_RATIO: f32 = 0.35;
    /// Tail span must be less than this fraction of the widest-edge span.
    const TAIL_VS_BODY_RATIO: f32 = 0.50;

    let img_w = gray.width();
    let img_h = gray.height();

    let x0 = bx.max(0.0) as u32;
    let y0 = by.max(0.0) as u32;
    let x1 = ((bx + bw) as u32 + 1).min(img_w);
    let y1 = ((by + bh) as u32 + 1).min(img_h);

    if x1 <= x0 + 8 || y1 <= y0 + 8 {
        return None;
    }

    let box_w = (x1 - x0) as f32;
    let box_h = (y1 - y0) as f32;

    // ── helpers ────────────────────────────────────────────────────────────────

    // Scan rows starting at `row_start`, stepping by `step` (+1 = inward from
    // top, −1 = inward from bottom).  For the first SCAN_ROWS rows that have
    // any dark pixels, record the x-span.  Returns the MINIMUM span seen,
    // normalised by box_w.  Returns 0.0 if no dark pixels found at all.
    let min_h_span = |row_start: i32, step: i32| -> f32 {
        let mut found = 0i32;
        let mut min_span = f32::MAX;
        let mut r = row_start;
        while found < SCAN_ROWS && r >= y0 as i32 && r < y1 as i32 {
            let row = r as u32;
            let first = (x0..x1).find(|&c| gray.get_pixel(c, row).0[0] <= DARK);
            let last  = (x0..x1).rev().find(|&c| gray.get_pixel(c, row).0[0] <= DARK);
            if let (Some(f), Some(l)) = (first, last) {
                let span = (l - f + 1) as f32 / box_w;
                if span < min_span {
                    min_span = span;
                }
                found += 1;
            }
            r += step;
        }
        if min_span == f32::MAX { 0.0 } else { min_span }
    };

    // Same but scans columns left→right (+1) or right→left (−1).
    // Spans are normalised by box_h.
    let min_v_span = |col_start: i32, step: i32| -> f32 {
        let mut found = 0i32;
        let mut min_span = f32::MAX;
        let mut c = col_start;
        while found < SCAN_ROWS && c >= x0 as i32 && c < x1 as i32 {
            let col = c as u32;
            let first = (y0..y1).find(|&r| gray.get_pixel(col, r).0[0] <= DARK);
            let last  = (y0..y1).rev().find(|&r| gray.get_pixel(col, r).0[0] <= DARK);
            if let (Some(f), Some(l)) = (first, last) {
                let span = (l - f + 1) as f32 / box_h;
                if span < min_span {
                    min_span = span;
                }
                found += 1;
            }
            c += step;
        }
        if min_span == f32::MAX { 0.0 } else { min_span }
    };

    // ── scan all four edges ────────────────────────────────────────────────────

    let top   = min_h_span(y0 as i32,      1);  // inward from top
    let bot   = min_h_span(y1 as i32 - 1, -1);  // inward from bottom
    let left  = min_v_span(x0 as i32,      1);  // inward from left
    let right = min_v_span(x1 as i32 - 1, -1);  // inward from right

    tracing::info!(top, bot, left, right, "balloon outermost dark-span (normalised)");

    let sides = [
        (top,   0.0f32, -1.0f32),  // tail points up
        (bot,   0.0,     1.0),     // tail points down
        (left, -1.0,     0.0),     // tail points left
        (right, 1.0,     0.0),     // tail points right
    ];

    let max_span = sides.iter().map(|(s, _, _)| *s).fold(0.0f32, f32::max);
    if max_span == 0.0 {
        return None;
    }

    let &(min_span, dx, dy) = sides
        .iter()
        .filter(|(s, _, _)| *s > 0.0)
        .min_by(|(a, _, _), (b, _, _)| a.partial_cmp(b).unwrap())?;

    // Require the tail to have a meaningful minimum width (not just 1–2 px) to
    // avoid treating the oval crown of the balloon body as a tail.
    if min_span > MIN_TAIL_SPAN && min_span < TAIL_WIDTH_RATIO && min_span / max_span < TAIL_VS_BODY_RATIO {
        tracing::info!(min_span, max_span, dx, dy, "balloon tail direction detected");
        Some((dx, dy))
    } else {
        None
    }
}

/// Convert a boolean gutter mask into a sorted list of split positions.
/// Each split is the midpoint of a consecutive run of gutter lines.
fn gutter_splits(is_gutter: &[bool], total: usize) -> Vec<usize> {
    let mut splits = vec![0usize];
    let mut in_gutter = false;
    let mut run_start = 0usize;

    for (i, &g) in is_gutter.iter().enumerate() {
        match (g, in_gutter) {
            (true, false) => {
                in_gutter = true;
                run_start = i;
            }
            (false, true) => {
                in_gutter = false;
                splits.push((run_start + i) / 2);
            }
            _ => {}
        }
    }
    splits.push(total);
    splits.dedup();
    splits
}
