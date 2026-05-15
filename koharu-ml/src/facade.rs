use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Result;
use image::DynamicImage;
use koharu_llm::paddleocr_vl::{self as paddleocr_vl_llm, PaddleOcrVl, PaddleOcrVlTask};
use koharu_llm::safe::llama_backend::LlamaBackend;
use koharu_types::{Document, FontPrediction, SerializableDynamicImage, TextBlock, TextDirection};

use crate::character_library::{self as character_library, CharacterLibrary};
use crate::comic_bubble_detector::{self as bubble_det, ComicBubbleDetector};
use crate::comic_text_detector::{self, ComicTextDetector, crop_text_block_bbox};
use crate::font_detector::{self, FontDetector};
use crate::lama::{self, Lama};
use crate::pp_doclayout_v3::{self, LayoutRegion, PPDocLayoutV3};

const NEAR_BLACK_THRESHOLD: u8 = 12;
const GRAY_NEAR_BLACK_THRESHOLD: u8 = 60;
const NEAR_WHITE_THRESHOLD: u8 = 12;
const GRAY_NEAR_WHITE_THRESHOLD: u8 = 60;
const GRAY_TOLERANCE: u8 = 10;
const SIMILAR_COLOR_MAX_DIFF: u8 = 16;
const PP_DOCLAYOUT_THRESHOLD: f32 = 0.25;
const VERTICAL_ASPECT_RATIO_THRESHOLD: f32 = 1.15;
const BLOCK_OVERLAP_DEDUPE_THRESHOLD: f32 = 0.9;
const OCR_MAX_NEW_TOKENS: usize = 128;

fn clamp_near_black(color: [u8; 3]) -> [u8; 3] {
    let max_channel = *color.iter().max().unwrap_or(&0);
    let min_channel = *color.iter().min().unwrap_or(&0);
    let is_grayish = max_channel.saturating_sub(min_channel) <= GRAY_TOLERANCE;
    let threshold = if is_grayish {
        GRAY_NEAR_BLACK_THRESHOLD
    } else {
        NEAR_BLACK_THRESHOLD
    };

    if color[0] <= threshold && color[1] <= threshold && color[2] <= threshold {
        [0, 0, 0]
    } else {
        color
    }
}

fn clamp_near_white(color: [u8; 3]) -> [u8; 3] {
    let max_channel = *color.iter().max().unwrap_or(&0);
    let min_channel = *color.iter().min().unwrap_or(&0);
    let is_grayish = max_channel.saturating_sub(min_channel) <= GRAY_TOLERANCE;
    let threshold = if is_grayish {
        GRAY_NEAR_WHITE_THRESHOLD
    } else {
        NEAR_WHITE_THRESHOLD
    };

    let min_white = 255u8.saturating_sub(threshold);
    if color[0] >= min_white && color[1] >= min_white && color[2] >= min_white {
        [255, 255, 255]
    } else {
        color
    }
}

fn colors_similar(a: [u8; 3], b: [u8; 3]) -> bool {
    a[0].abs_diff(b[0]) <= SIMILAR_COLOR_MAX_DIFF
        && a[1].abs_diff(b[1]) <= SIMILAR_COLOR_MAX_DIFF
        && a[2].abs_diff(b[2]) <= SIMILAR_COLOR_MAX_DIFF
}

fn normalize_font_prediction(prediction: &mut FontPrediction) {
    prediction.text_color = clamp_near_white(clamp_near_black(prediction.text_color));
    prediction.stroke_color = clamp_near_white(clamp_near_black(prediction.stroke_color));

    if prediction.stroke_width_px > 0.0
        && colors_similar(prediction.text_color, prediction.stroke_color)
    {
        prediction.stroke_width_px = 0.0;
        prediction.stroke_color = prediction.text_color;
    }
}

pub struct Model {
    layout_detector: PPDocLayoutV3,
    segmenter: ComicTextDetector,
    ocr: Mutex<PaddleOcrVl>,
    lama: Lama,
    font_detector: FontDetector,
    bubble_detector: Option<ComicBubbleDetector>,
    pub character_lib: CharacterLibrary,
}

impl Model {
    pub async fn new(cpu: bool, backend: Arc<LlamaBackend>) -> Result<Self> {
        let bubble_detector = match ComicBubbleDetector::load().await {
            Ok(d) => {
                tracing::info!("ComicBubbleDetector loaded");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(error = %e, "ComicBubbleDetector not available");
                None
            }
        };

        let character_lib = CharacterLibrary::load().unwrap_or_else(|e| {
            tracing::warn!(error = %e, "CharacterLibrary load failed, starting empty");
            CharacterLibrary::empty()
        });

        Ok(Self {
            layout_detector: PPDocLayoutV3::load(cpu).await?,
            segmenter: ComicTextDetector::load_segmentation_only(cpu).await?,
            ocr: Mutex::new(PaddleOcrVl::load(cpu, backend).await?),
            lama: Lama::load(cpu).await?,
            font_detector: FontDetector::load(cpu).await?,
            bubble_detector,
            character_lib,
        })
    }

    /// Detect text blocks and fonts in a document.
    /// Sets `doc.text_blocks` (with font predictions/styles) and `doc.segment`.
    pub async fn detect(&self, doc: &mut Document) -> Result<()> {
        let detect_started = Instant::now();

        let layout_started = Instant::now();
        let layout = self
            .layout_detector
            .inference_one_fast(&doc.image, PP_DOCLAYOUT_THRESHOLD)?;
        doc.text_blocks = build_text_blocks(&layout.regions);
        let layout_elapsed = layout_started.elapsed();

        let segmentation_started = Instant::now();
        let probability_map = self.segmenter.inference_segmentation(&doc.image)?;
        let mask = comic_text_detector::refine_segmentation_mask(
            &doc.image,
            &probability_map,
            &doc.text_blocks,
        );
        doc.segment = Some(DynamicImage::ImageLuma8(mask).into());
        let segmentation_elapsed = segmentation_started.elapsed();

        let font_started = Instant::now();
        if !doc.text_blocks.is_empty() {
            let images: Vec<DynamicImage> = doc
                .text_blocks
                .iter()
                .map(|block| {
                    doc.image.crop_imm(
                        block.x as u32,
                        block.y as u32,
                        block.width as u32,
                        block.height as u32,
                    )
                })
                .collect();

            let font_predictions = self.detect_fonts(&images, 1).await?;
            for (block, prediction) in doc.text_blocks.iter_mut().zip(font_predictions) {
                block.font_prediction = Some(prediction);
                block.style = None;
            }
        }
        let font_elapsed = font_started.elapsed();

        tracing::info!(
            text_blocks = doc.text_blocks.len(),
            layout_ms = layout_elapsed.as_millis(),
            segmentation_ms = segmentation_elapsed.as_millis(),
            font_ms = font_elapsed.as_millis(),
            total_ms = detect_started.elapsed().as_millis(),
            "detect stage timings"
        );

        Ok(())
    }

    /// Run OCR on all text blocks in the document.
    /// Updates `doc.text_blocks` with recognized text.
    pub async fn ocr(&self, doc: &mut Document) -> Result<()> {
        if doc.text_blocks.is_empty() {
            return Ok(());
        }

        let ocr_started = Instant::now();
        let crop_started = Instant::now();
        let regions = doc
            .text_blocks
            .iter()
            .map(|block| crop_text_block_bbox(&doc.image, block))
            .collect::<Vec<_>>();
        let crop_elapsed = crop_started.elapsed();

        let inference_started = Instant::now();
        let mut ocr = self
            .ocr
            .lock()
            .map_err(|_| anyhow::anyhow!("PaddleOCR-VL mutex poisoned"))?;
        let outputs = ocr.inference_images(&regions, PaddleOcrVlTask::Ocr, OCR_MAX_NEW_TOKENS)?;
        let inference_elapsed = inference_started.elapsed();

        for (block_index, output) in outputs.into_iter().enumerate() {
            if let Some(block) = doc.text_blocks.get_mut(block_index) {
                block.text = Some(output.text);
            }
        }

        tracing::info!(
            text_blocks = doc.text_blocks.len(),
            crop_ms = crop_elapsed.as_millis(),
            inference_ms = inference_elapsed.as_millis(),
            total_ms = ocr_started.elapsed().as_millis(),
            "ocr stage timings"
        );

        Ok(())
    }

    /// Inpaint text regions in the document.
    /// Uses the current `doc.segment` mask as the inpaint source, sets `doc.inpainted`.
    pub async fn inpaint(&self, doc: &mut Document) -> Result<()> {
        let mask = doc
            .segment
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Segment image not found"))?;
        let result = self
            .lama
            .inference_with_blocks(&doc.image, mask, Some(&doc.text_blocks))?;
        doc.inpainted = Some(result.into());

        Ok(())
    }

    /// Low-level inpaint: inpaint a specific image region with a mask.
    pub async fn inpaint_raw(
        &self,
        image: &SerializableDynamicImage,
        mask: &SerializableDynamicImage,
        text_blocks: Option<&[koharu_types::TextBlock]>,
    ) -> Result<SerializableDynamicImage> {
        let result = self.lama.inference_with_blocks(image, mask, text_blocks)?;
        Ok(result.into())
    }

    pub async fn detect_balloons(&self, doc: &mut Document) -> Result<()> {
        let Some(detector) = &self.bubble_detector else {
            tracing::warn!("bubble_detector not loaded, skipping balloon detection");
            return Ok(());
        };
        let started = Instant::now();
        let detections = detector.detect(&doc.image)?;

        doc.balloons = detections
            .iter()
            .map(|d| koharu_types::BalloonDetection {
                x: d.x,
                y: d.y,
                width: d.width,
                height: d.height,
                score: d.score,
            })
            .collect();

        // Refit each text block to the Maximum Inscribed Rectangle of its balloon mask.
        refit_text_blocks_to_balloons(&mut doc.text_blocks, &detections);

        tracing::info!(
            count = doc.balloons.len(),
            elapsed_ms = started.elapsed().as_millis(),
            "detect_balloons done"
        );
        Ok(())
    }

    /// Scan `image` for known characters and return a context string suitable for
    /// injection into the LLM system prompt, or `None` if the library is empty.
    pub fn scan_for_character_context(&self, image: &DynamicImage) -> Option<String> {
        self.character_lib.scan_and_build_context(image)
    }

    /// Per-block Vietnamese pronoun assignment using the existing tested speaker detection pipeline.
    /// For each block: detect panels → find the face nearest to each balloon (face det + WD Tagger)
    /// → derive pronouns from speaker/listener demographics.
    pub fn scan_pronoun_context(&self, doc: &Document, custom_system_prompt: Option<&str>) -> Option<String> {
        if doc.text_blocks.is_empty() {
            return None;
        }

        let image = &doc.image.0;

        let panels = self.character_lib.detect_panels(image);

        let blocks: Vec<(String, f32, f32, f32, f32)> = doc.text_blocks.iter()
            .map(|b| (b.id.clone(), b.x, b.y, b.width, b.height))
            .collect();
        let balloons: Vec<(f32, f32, f32, f32)> = doc.balloons.iter()
            .map(|b| (b.x, b.y, b.width, b.height))
            .collect();

        // Reuse the existing tested pipeline: face detection per panel → nearest face to balloon
        // → WD Tagger for age/gender on unknown characters.
        let speaker_assignments = self.character_lib.assign_speakers_to_blocks(
            image, &blocks, &balloons, &panels,
        );

        if speaker_assignments.iter().all(|(_, m)| m.is_none()) {
            return None;
        }

        // Build id → speaker label map.
        let speaker_labels: std::collections::HashMap<&str, String> = speaker_assignments.iter()
            .filter_map(|(id, m)| m.as_ref().map(|f| (id.as_str(), face_label(f))))
            .collect();

        let mut lines: Vec<String> = Vec::new();

        for (i, block) in doc.text_blocks.iter().enumerate() {
            let speaker_label = match speaker_labels.get(block.id.as_str()) {
                Some(l) => l.as_str(),
                None => continue,
            };

            // Listener = nearest block (by index) whose speaker has different demographics.
            let listener_label_owned: String = speaker_assignments.iter()
                .enumerate()
                .filter(|(j, (_, m))| {
                    *j != i && m.as_ref()
                        .map(|f| face_label(f) != speaker_label)
                        .unwrap_or(false)
                })
                .min_by_key(|(j, _)| j.abs_diff(i))
                .and_then(|(_, (_, m))| m.as_ref())
                .map(face_label)
                .unwrap_or_else(|| "Unknown".to_string());
            let listener_label = listener_label_owned.as_str();

            let (self_pron, other_pron) = suggest_vn_pronoun_pair(speaker_label, listener_label);
            let speaker_desc = demographic_desc(speaker_label);
            let listener_desc = demographic_desc(listener_label);

            tracing::info!(
                block = i,
                speaker = speaker_label,
                listener = listener_label,
                self_pronoun = self_pron,
                other_pronoun = other_pron,
                "pronoun assignment"
            );

            lines.push(format!(
                "<block id=\"{i}\"> speaker={speaker_desc}, listener={listener_desc} → use \"{self_pron}\" for I/me, \"{other_pron}\" for you"
            ));
        }

        if lines.is_empty() {
            return None;
        }

        let header = if let Some(prompt) = custom_system_prompt.filter(|p| !p.trim().is_empty()) {
            format!(
                "MANDATORY pronoun rules — you MUST use exactly these Vietnamese pronouns for each block (story context: \"{}\"). Do NOT include block tags or any metadata in translation output:",
                prompt.trim()
            )
        } else {
            "MANDATORY pronoun rules — you MUST use exactly these Vietnamese pronouns for each block. Do NOT include block tags or any metadata in translation output:".to_string()
        };
        lines.insert(0, header);
        Some(lines.join("\n"))
    }

    pub async fn detect_font(&self, image: &DynamicImage, top_k: usize) -> Result<FontPrediction> {
        let mut results = self
            .detect_fonts(std::slice::from_ref(image), top_k)
            .await?;
        Ok(results.pop().unwrap_or_default())
    }

    pub async fn detect_fonts(
        &self,
        images: &[DynamicImage],
        top_k: usize,
    ) -> Result<Vec<FontPrediction>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut predictions = self.font_detector.inference(images, top_k)?;
        for prediction in &mut predictions {
            normalize_font_prediction(prediction);
        }
        Ok(predictions)
    }
}

fn face_label(f: &character_library::FaceMatch) -> String {
    if f.traits.is_empty() {
        f.name.clone()
    } else {
        format!("{} {}", f.name, f.traits.join(" "))
    }
}

/// Return a demographic description (age + gender) from a face label.
fn demographic_desc(label: &str) -> String {
    let (age, gender) = parse_age_gender(label);
    let age_str = match age {
        Age::Young => "young",
        Age::Adult => "adult",
        Age::Old => "elderly",
        Age::Unknown => "unknown age",
    };
    let gender_str = match gender {
        Gender::Male => "male",
        Gender::Female => "female",
        Gender::Unknown => "unknown gender",
    };
    format!("{age_str} {gender_str}")
}

/// Detect family-role keywords in a label.
fn family_role(label: &str) -> Option<FamilyRole> {
    let l = label.to_lowercase();
    if l.contains("father") || l.contains("dad") || l.contains("bố") || l.contains("ba ") || l == "ba" || l.contains("papa") {
        Some(FamilyRole::Father)
    } else if l.contains("mother") || l.contains("mom") || l.contains("mẹ") || l.contains("mama") || l.contains("mum") {
        Some(FamilyRole::Mother)
    } else if l.contains("son") || l.contains("daughter") || l.contains("child") || l.contains("kid") {
        Some(FamilyRole::Child)
    } else if l.contains("grandfather") || l.contains("grandpa") || l.contains("ông nội") || l.contains("ông ngoại") {
        Some(FamilyRole::Grandfather)
    } else if l.contains("grandmother") || l.contains("grandma") || l.contains("bà nội") || l.contains("bà ngoại") {
        Some(FamilyRole::Grandmother)
    } else if l.contains("older brother") || l.contains("anh trai") {
        Some(FamilyRole::OlderBrother)
    } else if l.contains("older sister") || l.contains("chị gái") {
        Some(FamilyRole::OlderSister)
    } else {
        None
    }
}

#[derive(PartialEq, Eq)]
enum FamilyRole { Father, Mother, Child, Grandfather, Grandmother, OlderBrother, OlderSister }

/// Suggest Vietnamese first-person and second-person pronouns based on speaker/listener labels.
/// Labels come either from WD Tagger ("Young Male", "Adult Female", …) or known character traits.
fn suggest_vn_pronoun_pair(speaker_label: &str, listener_label: &str) -> (&'static str, &'static str) {
    // Family relationship takes priority over age/gender heuristic.
    if let (Some(sp_role), _) = (family_role(speaker_label), family_role(listener_label)) {
        return match sp_role {
            FamilyRole::Father => ("bố", "con"),
            FamilyRole::Mother => ("mẹ", "con"),
            FamilyRole::Grandfather => ("ông", "con/cháu"),
            FamilyRole::Grandmother => ("bà", "con/cháu"),
            FamilyRole::OlderBrother => ("anh", "em"),
            FamilyRole::OlderSister => ("chị", "em"),
            FamilyRole::Child => {
                // Child speaking — need to know who they're talking to
                match family_role(listener_label) {
                    Some(FamilyRole::Father) => ("con", "bố"),
                    Some(FamilyRole::Mother) => ("con", "mẹ"),
                    Some(FamilyRole::Grandfather) => ("cháu", "ông"),
                    Some(FamilyRole::Grandmother) => ("cháu", "bà"),
                    _ => ("con", "bố/mẹ"),
                }
            }
        };
    }
    // Listener is a parent — child is speaking
    if let Some(ls_role) = family_role(listener_label) {
        return match ls_role {
            FamilyRole::Father => ("con", "bố"),
            FamilyRole::Mother => ("con", "mẹ"),
            FamilyRole::Grandfather => ("cháu", "ông"),
            FamilyRole::Grandmother => ("cháu", "bà"),
            FamilyRole::OlderBrother => ("em", "anh"),
            FamilyRole::OlderSister => ("em", "chị"),
            FamilyRole::Child => ("bố/mẹ", "con"),
        };
    }

    let (sp_age, sp_gen) = parse_age_gender(speaker_label);
    let (ls_age, ls_gen) = parse_age_gender(listener_label);

    // Khi một trong hai bên không rõ tuổi → chỉ dùng ngôi theo giới tính, không đổi ngôi
    if sp_age == Age::Unknown || ls_age == Age::Unknown {
        return match (sp_gen, ls_gen) {
            (Gender::Male,    Gender::Female)  => ("tôi", "cô"),
            (Gender::Male,    Gender::Male)    => ("tôi", "cậu"),
            (Gender::Female,  Gender::Male)    => ("tôi", "anh"),
            (Gender::Female,  Gender::Female)  => ("tôi", "cô"),
            (Gender::Unknown, Gender::Female)  => ("tôi", "cô"),
            (Gender::Unknown, Gender::Male)    => ("tôi", "anh"),
            _                                  => ("tôi", "cậu"),
        };
    }

    // Cả hai đều biết tuổi → xét chênh lệch già/trẻ để đổi ngôi
    match (sp_age, sp_gen, ls_age, ls_gen) {
        // Người già nói chuyện
        (Age::Old, Gender::Male,   _, _) => ("ông", "cháu"),
        (Age::Old, Gender::Female, _, _) => ("bà",  "cháu"),

        // Người trẻ nói với người già
        (_, _, Age::Old, Gender::Male)   => ("cháu", "ông"),
        (_, _, Age::Old, Gender::Female) => ("cháu", "bà"),

        // Lớn hơn (Adult) nói với nhỏ hơn (Young)
        (Age::Adult, Gender::Male,   Age::Young, _) => ("anh", "em"),
        (Age::Adult, Gender::Female, Age::Young, _) => ("chị", "em"),

        // Nhỏ hơn (Young) nói với lớn hơn (Adult)
        (Age::Young, _, Age::Adult, Gender::Male)   => ("em", "anh"),
        (Age::Young, _, Age::Adult, Gender::Female) => ("em", "chị"),

        // Cùng tầm tuổi → theo giới tính
        (_, Gender::Male,    _, Gender::Female)  => ("tôi", "cô"),
        (_, Gender::Male,    _, Gender::Male)    => ("tôi", "cậu"),
        (_, Gender::Female,  _, Gender::Male)    => ("tôi", "anh"),
        (_, Gender::Female,  _, Gender::Female)  => ("tôi", "cô"),
        (_, Gender::Unknown, _, Gender::Female)  => ("tôi", "cô"),
        (_, Gender::Unknown, _, Gender::Male)    => ("tôi", "anh"),
        _                                        => ("tôi", "cậu"),
    }
}

#[derive(PartialEq, Eq)]
enum Age { Young, Adult, Old, Unknown }

#[derive(PartialEq, Eq)]
enum Gender { Male, Female, Unknown }

fn parse_age_gender(label: &str) -> (Age, Gender) {
    let l = label.to_lowercase();

    let age = if l.contains("old") || l.contains("elder") || l.contains("senior") || l.contains("grandfather") || l.contains("grandmother") {
        Age::Old
    } else if l.contains("young") || l.contains("teen") || l.contains("child") || l.contains("kid") || l.contains("boy") || l.contains("girl") {
        Age::Young
    } else if l.contains("adult") || l.contains("man") || l.contains("woman") {
        Age::Adult
    } else {
        Age::Unknown
    };

    let gender = if l.contains("female") || l.contains("woman") || l.contains("girl") {
        Gender::Female
    } else if l.contains("male") || l.contains("man") || l.contains("boy") {
        Gender::Male
    } else {
        Gender::Unknown
    };

    (age, gender)
}

pub async fn prefetch() -> Result<()> {
    pp_doclayout_v3::prefetch().await?;
    comic_text_detector::prefetch_segmentation().await?;
    paddleocr_vl_llm::prefetch().await?;
    lama::prefetch().await?;
    font_detector::prefetch().await?;
    // bubble_detector loads from local path, no prefetch needed

    Ok(())
}

fn build_text_blocks(regions: &[LayoutRegion]) -> Vec<TextBlock> {
    let mut blocks = regions
        .iter()
        .filter(|region| is_text_layout_label(&region.label))
        .filter_map(layout_region_to_text_block)
        .collect::<Vec<_>>();
    dedupe_text_blocks(&mut blocks);
    blocks
}

fn is_text_layout_label(label: &str) -> bool {
    let label = label.to_ascii_lowercase();
    label == "content" || label.contains("text") || label.contains("title")
}

fn layout_region_to_text_block(region: &LayoutRegion) -> Option<TextBlock> {
    let x1 = region.bbox[0].min(region.bbox[2]).max(0.0);
    let y1 = region.bbox[1].min(region.bbox[3]).max(0.0);
    let x2 = region.bbox[0].max(region.bbox[2]).max(x1 + 1.0);
    let y2 = region.bbox[1].max(region.bbox[3]).max(y1 + 1.0);
    let width = (x2 - x1).max(1.0);
    let height = (y2 - y1).max(1.0);

    if width < 6.0 || height < 6.0 || width * height < 48.0 {
        return None;
    }

    let source_direction = infer_text_direction(width, height);
    Some(TextBlock {
        x: x1,
        y: y1,
        width,
        height,
        confidence: region.score,
        source_direction: Some(source_direction),
        source_language: Some("unknown".to_string()),
        rotation_deg: Some(0.0),
        detected_font_size_px: Some(width.min(height).max(1.0)),
        detector: Some("pp-doclayout-v3".to_string()),
        ..Default::default()
    })
}

fn infer_text_direction(width: f32, height: f32) -> TextDirection {
    if height >= width * VERTICAL_ASPECT_RATIO_THRESHOLD {
        TextDirection::Vertical
    } else {
        TextDirection::Horizontal
    }
}

fn dedupe_text_blocks(blocks: &mut Vec<TextBlock>) {
    if blocks.len() < 2 {
        return;
    }

    let mut deduped = Vec::with_capacity(blocks.len());
    for block in std::mem::take(blocks) {
        let area = (block.width * block.height).max(1.0);
        let overlaps_existing = deduped.iter().any(|existing: &TextBlock| {
            let existing_area = (existing.width * existing.height).max(1.0);
            let overlap = overlap_area(block_bbox(&block), block_bbox(existing));
            overlap / area >= BLOCK_OVERLAP_DEDUPE_THRESHOLD
                || overlap / existing_area >= BLOCK_OVERLAP_DEDUPE_THRESHOLD
        });
        if !overlaps_existing {
            deduped.push(block);
        }
    }
    *blocks = deduped;
}

fn block_bbox(block: &TextBlock) -> [f32; 4] {
    [
        block.x,
        block.y,
        block.x + block.width,
        block.y + block.height,
    ]
}

fn overlap_area(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    if x2 <= x1 || y2 <= y1 {
        0.0
    } else {
        (x2 - x1) * (y2 - y1)
    }
}

// ─── Balloon MIR pipeline ─────────────────────────────────────────────────────

/// Erosion radius in pixels applied before the largest-rectangle sweep.
const MIR_EROSION_RADIUS: u32 = 4;
/// Inset ratio applied to the bbox fallback on each side.
const MIR_BBOX_INSET: f32 = 0.08;
/// Additional padding (px) subtracted from each side of the MIR/bbox-inset
/// result before assigning to the text block, so rendered text stays clear of
/// the balloon border.
const MIR_TEXT_PADDING: f32 = 6.0;

/// For each balloon, find all text blocks whose center lies inside the balloon bbox,
/// then refit those text blocks to the MIR of the balloon's mask.
///
/// Workflow: Detect → OCR → Balloon (calls this) → Translate → Inpaint → Render
fn refit_text_blocks_to_balloons(
    text_blocks: &mut Vec<TextBlock>,
    balloons: &[bubble_det::BubbleBox],
) {
    // Snapshot original centers before any mutation to avoid cascade matching.
    let orig_centers: Vec<(f32, f32)> = text_blocks
        .iter()
        .map(|b| (b.x + b.width / 2.0, b.y + b.height / 2.0))
        .collect();

    let mut refit_count = 0usize;

    for (bi, balloon) in balloons.iter().enumerate() {
        // Collect all text blocks whose original center is inside this balloon bbox.
        let inside_indices: Vec<usize> = orig_centers
            .iter()
            .enumerate()
            .filter(|(_, center)| {
                let (cx, cy) = **center;
                cx >= balloon.x
                    && cx <= balloon.x + balloon.width
                    && cy >= balloon.y
                    && cy <= balloon.y + balloon.height
            })
            .map(|(i, _)| i)
            .collect();

        if inside_indices.is_empty() {
            continue;
        }

        // Adjacent/merged balloons: if multiple text blocks share this balloon, skip refitting
        // and keep their original detected positions.
        if inside_indices.len() > 1 {
            tracing::debug!(
                balloon = bi, inside_count = inside_indices.len(),
                "multiple text blocks in balloon — skipping refit"
            );
            continue;
        }

        let ti = inside_indices[0];
        let mir = match &balloon.mask {
            Some(mask) => {
                let r = mir_from_mask(mask, balloon.x, balloon.y, balloon.width, balloon.height);
                if r[2] > 2.0 && r[3] > 2.0 { r } else { bbox_inset(balloon) }
            }
            None => bbox_inset(balloon),
        };

        tracing::info!(
            balloon = bi, block = ti,
            mir_x = mir[0], mir_y = mir[1], mir_w = mir[2], mir_h = mir[3],
            "refit"
        );

        let block = &mut text_blocks[ti];
        let pad = MIR_TEXT_PADDING;
        block.x = mir[0] + pad;
        block.y = mir[1] + pad;
        block.width = (mir[2] - 2.0 * pad).max(1.0);
        block.height = (mir[3] - 2.0 * pad).max(1.0);
        // Clear seed layout so the renderer uses the new refit coordinates,
        // not the stale pre-balloon values.
        block.layout_seed_x = None;
        block.layout_seed_y = None;
        block.layout_seed_width = None;
        block.layout_seed_height = None;
        // Prevent the renderer from re-scanning the image for balloon bounds
        // or auto-expanding the layout box — the MIR coordinates are authoritative.
        block.lock_layout_box = true;
        refit_count += 1;
    }

    tracing::info!(refit = refit_count, total = text_blocks.len(), "text blocks refit to balloon MIR");
}

/// Compute the Maximum Inscribed Rectangle from a binary mask via the classical
/// "largest rectangle in histogram" algorithm applied row-by-row (O(w×h)).
/// Operates on the eroded mask to stay safely inside the balloon boundary.
/// All returned coordinates are in full-image space.
/// Falls back to zeros if no usable region found (caller uses bbox_inset).
fn mir_from_mask(mask: &image::GrayImage, bx: f32, by: f32, bw: f32, bh: f32) -> [f32; 4] {
    let (img_w, img_h) = mask.dimensions();

    // 1. Crop mask to balloon bounding box.
    let cx0 = (bx as u32).min(img_w.saturating_sub(1));
    let cy0 = (by as u32).min(img_h.saturating_sub(1));
    let cx1 = ((bx + bw) as u32).min(img_w);
    let cy1 = ((by + bh) as u32).min(img_h);
    if cx1 <= cx0 || cy1 <= cy0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let cw = cx1 - cx0;
    let ch = cy1 - cy0;
    let cropped = image::imageops::crop_imm(mask, cx0, cy0, cw, ch).to_image();

    // 2. Erode for safe margin — removes thin tails/protrusions.
    let safe = bubble_det::erode_binary(&cropped, MIR_EROSION_RADIUS);

    // 3. Largest Rectangle in Binary Image via histogram sweep.
    //    heights[x] = number of consecutive foreground rows ending at current row.
    let w = cw as usize;
    let h = ch as usize;
    let mut heights = vec![0u32; w];
    let mut best_area = 0u32;
    let mut best: (u32, u32, u32, u32) = (0, 0, 0, 0); // (lx, ly, lw, lh) local coords

    for y in 0..h {
        // Update column heights.
        for x in 0..w {
            if safe.get_pixel(x as u32, y as u32)[0] >= 128 {
                heights[x] += 1;
            } else {
                heights[x] = 0;
            }
        }

        // Largest rectangle in this histogram row (stack-based, O(w)).
        let mut stack: Vec<(usize, u32)> = Vec::new(); // (x_start, height)
        for x in 0..=w {
            let cur_h = if x < w { heights[x] } else { 0 };
            let mut x_start = x;
            while let Some(&(sx, sh)) = stack.last() {
                if sh <= cur_h {
                    break;
                }
                stack.pop();
                let rect_w = (x - sx) as u32;
                let area = sh * rect_w;
                if area > best_area {
                    best_area = area;
                    // Bottom of rect is row y (inclusive), height is sh.
                    best = (sx as u32, y as u32 + 1 - sh, rect_w, sh);
                }
                x_start = sx;
            }
            stack.push((x_start, cur_h));
        }
    }

    if best_area == 0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let (lx, ly, lw, lh) = best;

    let ltx = lx as f32;
    let lty = ly as f32;
    let tw = lw as f32;
    let th = lh as f32;

    // 5. Offset back to full-image coordinates.
    [cx0 as f32 + ltx, cy0 as f32 + lty, tw, th]
}

/// Fallback: balloon bounding box with a small inset on each side.
fn bbox_inset(balloon: &bubble_det::BubbleBox) -> [f32; 4] {
    let ix = balloon.width * MIR_BBOX_INSET;
    let iy = balloon.height * MIR_BBOX_INSET;
    [
        balloon.x + ix,
        balloon.y + iy,
        (balloon.width - 2.0 * ix).max(1.0),
        (balloon.height - 2.0 * iy).max(1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_region(order: usize, label: &str, bbox: [f32; 4]) -> LayoutRegion {
        LayoutRegion {
            order,
            label_id: 0,
            label: label.to_string(),
            score: 0.9,
            bbox,
            polygon_points: vec![],
        }
    }

    #[test]
    fn build_text_blocks_keeps_textlike_regions_and_dedupes_overlaps() {
        let blocks = build_text_blocks(&[
            test_region(0, "text", [10.0, 10.0, 40.0, 40.0]),
            test_region(1, "image", [0.0, 0.0, 128.0, 128.0]),
            test_region(2, "aside_text", [12.0, 12.0, 39.0, 39.0]),
            test_region(3, "doc_title", [60.0, 8.0, 90.0, 24.0]),
        ]);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].detector.as_deref(), Some("pp-doclayout-v3"));
        assert!(blocks[0].line_polygons.is_none());
        assert_eq!(blocks[1].source_direction, Some(TextDirection::Horizontal));
    }

    #[test]
    fn build_text_blocks_marks_tall_regions_as_vertical() {
        let blocks = build_text_blocks(&[test_region(0, "text", [5.0, 5.0, 20.0, 60.0])]);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].source_direction, Some(TextDirection::Vertical));
    }
}
