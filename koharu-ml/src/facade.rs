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
use crate::comic_text_detector::{self, crop_text_block_bbox};
use crate::font_detector::{self, FontDetector};
use crate::lama::{self, Lama};
use crate::manga_text_segmentation_2025::MangaTextSegmentation;
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
/// The square the layout detector resizes every input to.
const DETECTOR_INPUT_SIZE: f32 = 800.0;
/// How far the width may be squeezed on the way into the detector before
/// neighbouring columns of vertical text start merging into one box.
const MIN_DETECTOR_SCALE: f32 = 0.75;
/// How far past its share each tile reaches, so text on a tile boundary is
/// whole in at least one of them.
const TILE_OVERLAP_RATIO: f32 = 0.06;
/// How close to a tile's inner edge a box must sit to count as cut off by it.
const TILE_EDGE_TOLERANCE_PX: f32 = 2.0;
/// How much of a cut-off box another box must cover before the cut-off one is
/// thrown away as the same text seen whole by the neighbouring tile.
const CLIPPED_REGION_COVERED_SHARE: f32 = 0.5;
/// Probability above which the text segmentation model's output counts as text.
const TEXT_MASK_THRESHOLD: f32 = 0.5;
/// Growth applied to the text mask before inpainting.
const TEXT_MASK_DILATE_RADIUS: u8 = 3;
/// Glyphs are separate blobs; merging at this radius groups a run of text
/// into one region instead of counting it stroke by stroke.
const MISSED_TEXT_MERGE_RADIUS: u8 = 12;
/// Smaller than this is mask noise or art texture, not a missed run of text —
/// much bigger than `lama`'s equivalent (24px) since this gates a whole extra
/// model pass, not just an inpainting touch-up.
const MISSED_TEXT_MIN_PIXELS: u32 = 200;
/// How far past every `PPDocLayoutV3` block's own box a run of ink must sit
/// to count as genuinely missed rather than just outside a tight box edge.
const MISSED_TEXT_GAP_PX: f32 = 24.0;
/// A region this large is a runaway mask (e.g. a whole dark panel misread as
/// text-shaped), not a missed line of dialogue.
const MISSED_TEXT_MAX_AREA_SHARE: f32 = 0.05;

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
    text_segmenter: MangaTextSegmentation,
    ocr: Mutex<PaddleOcrVl>,
    lama: Lama,
    font_detector: FontDetector,
    bubble_detector: Option<ComicBubbleDetector>,
    /// Only ever consulted when `detect()` finds text-shaped ink that no
    /// `PPDocLayoutV3` block reaches (see `uncovered_ink_regions`) —
    /// `PPDocLayoutV3` stays the trusted source for everything it already
    /// finds. Its own full detection (YOLOv5+DBNet, quad/rotation-aware) is
    /// otherwise unused; catching rotated text is exactly the gap
    /// `PPDocLayoutV3`, a general document-layout model, cannot cover.
    comic_text_detector: Option<comic_text_detector::ComicTextDetector>,
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

        let comic_text_detector = match comic_text_detector::ComicTextDetector::load(cpu).await {
            Ok(d) => {
                tracing::info!("ComicTextDetector loaded (rescan-on-miss only)");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(error = %e, "ComicTextDetector not available — rotated text PPDocLayoutV3 misses will not be recovered");
                None
            }
        };

        let character_lib = CharacterLibrary::load().unwrap_or_else(|e| {
            tracing::warn!(error = %e, "CharacterLibrary load failed, starting empty");
            CharacterLibrary::empty()
        });
        // Restore whichever series' library was active last session — `load()`
        // above always opens the pre-split shared file first, since it does not
        // know the active profile's name on its own.
        if let Some(name) = crate::bilingual::style::load_active_name() {
            if let Err(e) = character_lib.switch_to(Some(&name)) {
                tracing::warn!(error = %e, name, "failed to restore the active character library");
            }
        }

        Ok(Self {
            layout_detector: PPDocLayoutV3::load(cpu).await?,
            text_segmenter: MangaTextSegmentation::load(cpu).await?,
            ocr: Mutex::new(PaddleOcrVl::load(cpu, backend).await?),
            lama: Lama::load(cpu).await?,
            font_detector: FontDetector::load(cpu).await?,
            bubble_detector,
            comic_text_detector,
            character_lib,
        })
    }

    /// Run the layout detector, in horizontal tiles when the page is too wide
    /// to hand it whole.
    ///
    /// The detector resizes whatever it is given to a fixed 800x800 square, so
    /// a wide scan — a two-page spread, say — arrives at about half the scale a
    /// single page does. Vertical Japanese sets its columns side by side, and
    /// at that scale the gap between two columns can fall below what the
    /// detector resolves: it reports one box covering part of the text and
    /// drops the rest, which then never reaches OCR. Tiling keeps the columns
    /// apart; the boxes are put back into page coordinates afterwards.
    fn detect_layout_regions(&self, image: &DynamicImage) -> Result<Vec<LayoutRegion>> {
        let (width, height) = (image.width(), image.height());
        let tiles = detection_tiles(width);
        if tiles.len() == 1 {
            return Ok(self
                .layout_detector
                .inference_one_fast(image, PP_DOCLAYOUT_THRESHOLD)?
                .regions);
        }

        let mut found = Vec::new();
        for (left, right) in tiles {
            let tile = image.crop_imm(left, 0, right - left, height);
            let tile_width = tile.width() as f32;
            let detected = self
                .layout_detector
                .inference_one_fast(&tile, PP_DOCLAYOUT_THRESHOLD)?;
            let offset = left as f32;
            found.extend(detected.regions.into_iter().map(|mut region| {
                // An edge the tile shares with the page is the page's own; only
                // an edge cut into the middle of the page can cut text in half.
                let bbox = region_bbox(&region);
                let clipped = (left > 0 && bbox[0] <= TILE_EDGE_TOLERANCE_PX)
                    || (right < width && bbox[2] >= tile_width - TILE_EDGE_TOLERANCE_PX);
                region.bbox[0] += offset;
                region.bbox[2] += offset;
                for point in &mut region.polygon_points {
                    point[0] += offset;
                }
                (region, clipped)
            }));
        }

        let detected = found.len();
        let regions = drop_clipped_fragments(found);

        tracing::info!(
            width,
            height,
            detected,
            regions = regions.len(),
            "detected in tiles"
        );
        Ok(regions)
    }

    /// Detect text blocks and fonts in a document.
    /// Sets `doc.text_blocks` (with font predictions/styles) and `doc.segment`.
    pub async fn detect(&self, doc: &mut Document) -> Result<()> {
        let detect_started = Instant::now();

        let layout_started = Instant::now();
        let regions = self.detect_layout_regions(&doc.image)?;
        doc.text_blocks = build_text_blocks(&regions);
        let layout_elapsed = layout_started.elapsed();

        let segmentation_started = Instant::now();
        // The text segmentation model recognises glyph shapes, so it separates
        // writing from artwork — including a figure drawn across a balloon,
        // which a brightness threshold cannot do. Its output is used for the
        // whole page rather than being clipped to the detected boxes: text the
        // box detector under-segmented was previously left unmasked and
        // survived inpainting onto the finished page.
        let probability_map = self.text_segmenter.inference(&doc.image)?;
        let mask = probability_map.threshold(TEXT_MASK_THRESHOLD)?;
        // The mask hugs the strokes, so grow it enough to take the grey
        // anti-aliased edge with it; left behind, that edge reads as a halo
        // around every erased glyph.
        let mask = imageproc::morphology::dilate(
            &mask,
            imageproc::distance_transform::Norm::L1,
            TEXT_MASK_DILATE_RADIUS,
        );

        // PPDocLayoutV3 stays the trusted source for every block it already
        // finds — this only ever *adds* a block for ink the mask marks as
        // text but that sits nowhere near any of them, and only pays for a
        // ComicTextDetector pass on the rare page that actually has such a
        // gap (see `uncovered_ink_regions`), rather than double-running
        // detection on every page.
        if let Some(ctd) = &self.comic_text_detector {
            let uncovered = uncovered_ink_regions(&mask, &doc.text_blocks);
            if !uncovered.is_empty() {
                match ctd.inference(&doc.image) {
                    Ok(detection) => {
                        let recovered: Vec<TextBlock> = detection
                            .text_blocks
                            .into_iter()
                            .filter(|block| {
                                let (cx, cy) =
                                    (block.x + block.width / 2.0, block.y + block.height / 2.0);
                                uncovered
                                    .iter()
                                    .any(|u| cx >= u[0] && cx <= u[2] && cy >= u[1] && cy <= u[3])
                            })
                            .collect();
                        tracing::info!(
                            uncovered = uncovered.len(),
                            added = recovered.len(),
                            "comic-text-detector rescan for text PPDocLayoutV3 missed"
                        );
                        doc.text_blocks.extend(recovered);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "comic-text-detector rescan failed");
                    }
                }
            }
        }

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
        // A balloon holding only marks — 「・・・」 — keeps the ones the artist
        // drew. Nothing replaces them, so erasing them would leave the pause
        // silent.
        let mask = keep_wordless_marks(mask.to_luma8(), &doc.text_blocks);
        let mask = keep_only_text_found(mask, &doc.text_blocks, &doc.balloons);
        let result = self.lama.inference_with_blocks(
            &doc.image,
            &DynamicImage::ImageLuma8(mask),
            Some(&doc.text_blocks),
        )?;
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
            // Once per process, not once per page: this used to log on every
            // page while quietly costing speaker attribution and text fitting.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "bubble_detector not loaded — balloon detection is off, so speaker \
                     attribution loses its balloon filter, text blocks are not refitted to \
                     balloons, and the SFX dictionary stays disabled. See the load error above \
                     for the paths that were tried."
                );
            });
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

    /// Like [`Self::scan_for_character_context`], but emits only the names of the
    /// characters present when `concise` — used when the full cast already rides
    /// in the provider's cached story context.
    pub fn scan_for_character_context_with(
        &self,
        image: &DynamicImage,
        concise: bool,
    ) -> Option<String> {
        self.character_lib
            .scan_and_build_context_with(image, concise)
    }

    /// Full cast description, for injection into a provider's story context.
    pub fn character_roster_context(&self) -> Option<String> {
        self.character_lib.roster_context()
    }

    /// Per-block Vietnamese pronoun assignment using the existing tested speaker detection pipeline.
    /// For each block: detect panels → find the face nearest to each balloon (face det + WD Tagger)
    /// → derive pronouns from speaker/listener demographics.
    pub fn scan_pronoun_context(
        &self,
        doc: &Document,
        custom_system_prompt: Option<&str>,
    ) -> Option<String> {
        if doc.text_blocks.is_empty() {
            return None;
        }

        let image = &doc.image.0;

        let panels = self.character_lib.detect_panels(image);

        let blocks: Vec<(String, f32, f32, f32, f32)> = doc
            .text_blocks
            .iter()
            .map(|b| (b.id.clone(), b.x, b.y, b.width, b.height))
            .collect();
        let balloons: Vec<(f32, f32, f32, f32)> = doc
            .balloons
            .iter()
            .map(|b| (b.x, b.y, b.width, b.height))
            .collect();

        // Reuse the existing tested pipeline: face detection per panel → nearest face to balloon
        // → WD Tagger for age/gender on unknown characters.
        let speaker_assignments = self
            .character_lib
            .assign_speakers_to_blocks(image, &blocks, &balloons, &panels);

        if speaker_assignments.iter().all(|(_, m)| m.is_none()) {
            return None;
        }

        // Build id → speaker label map.
        let speaker_labels: std::collections::HashMap<&str, String> = speaker_assignments
            .iter()
            .filter_map(|(id, m)| m.as_ref().map(|f| (id.as_str(), face_label(f))))
            .collect();

        let mut lines: Vec<String> = Vec::new();

        for (i, block) in doc.text_blocks.iter().enumerate() {
            let speaker_label = match speaker_labels.get(block.id.as_str()) {
                Some(l) => l.as_str(),
                None => continue,
            };

            // Listener = nearest block (by index) whose speaker has different demographics.
            let listener_face = speaker_assignments
                .iter()
                .enumerate()
                .filter(|(j, (_, m))| {
                    *j != i
                        && m.as_ref()
                            .map(|f| face_label(f) != speaker_label)
                            .unwrap_or(false)
                })
                .min_by_key(|(j, _)| j.abs_diff(i))
                .and_then(|(_, (_, m))| m.as_ref());

            // Inferred from traits alone. A block where nothing is known about
            // the speaker gets no line: the translator reads the dialogue, and
            // a guess dressed up as a fact is worse than saying nothing.
            let speaker_traits = speaker_assignments[i]
                .1
                .as_ref()
                .map(demographic_label)
                .unwrap_or_default();
            let listener_traits = listener_face
                .as_ref()
                .map(|f| demographic_label(f))
                .unwrap_or_default();
            if parse_age_gender(&speaker_traits) == (Age::Unknown, Gender::Unknown) {
                continue;
            }

            let (self_pron, other_pron) =
                suggest_vn_pronoun_pair(&speaker_traits, &listener_traits);
            let speaker_desc = demographic_desc(&speaker_traits);
            let listener_desc = demographic_desc(&listener_traits);

            tracing::info!(
                block = i,
                speaker = speaker_traits.as_str(),
                listener = listener_traits.as_str(),
                self_pronoun = self_pron,
                other_pronoun = other_pron,
                "pronoun assignment"
            );

            // Same `[N]` marker the source uses, so the model can line the rule
            // up with its block without a second naming scheme to learn.
            lines.push(format!(
                "[{i}] speaker={speaker_desc}, listener={listener_desc} — \"{self_pron}\"/\"{other_pron}\" would suit that pairing"
            ));
        }

        if lines.is_empty() {
            return None;
        }

        // Read off faces, not off the dialogue: it can tell a child from an
        // adult and little else. Ordering the translator to use these exact
        // pronouns overrode what the lines themselves say — a boy speaking to
        // an old man came out on first-name terms because a cartoon face
        // tagged as "adult male" fell through to the neutral pairing. Offer it
        // as what the picture shows and let the dialogue decide.
        const GUIDANCE: &str = "Who appears to be speaking to whom, from the \
            artwork. Vietnamese pronouns depend on the relationship, so use \
            this where the dialogue leaves it open — and follow the dialogue \
            wherever it says otherwise, since this is read off faces and \
            cannot hear the conversation. Do not put speaker names or any \
            other metadata inside the translated text.";
        let header = match custom_system_prompt.filter(|p| !p.trim().is_empty()) {
            Some(prompt) => format!("{GUIDANCE} Story context: \"{}\".", prompt.trim()),
            None => GUIDANCE.to_string(),
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

/// What is actually known about a character, without their name.
///
/// A name is not a description, but the two used to be joined into one string
/// and then searched for words like "man" or "old" — so Kinnikuman read as
/// male and anyone called Goldman as elderly. Only the traits say anything
/// about who someone is.
fn demographic_label(f: &character_library::FaceMatch) -> String {
    f.traits.join(" ")
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

/// Does the label name this word or phrase, as words rather than as letters?
///
/// The labels these read carry a character's traits, and a substring test
/// finds words inside other words: `"man"` sits in Kinnikuman, `"son"` in
/// person, `"old"` in Goldman. Each of those quietly decides someone's age or
/// gender, and from there their pronouns.
fn mentions(label: &str, phrase: &str) -> bool {
    let words: Vec<String> = label
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect();
    format!(" {} ", words.join(" ")).contains(&format!(" {phrase} "))
}

/// Detect family-role keywords in a label.
fn family_role(label: &str) -> Option<FamilyRole> {
    let says = |phrase: &str| mentions(label, phrase);
    if says("father") || says("dad") || says("bố") || says("ba") || says("papa") {
        Some(FamilyRole::Father)
    } else if says("mother") || says("mom") || says("mẹ") || says("mama") || says("mum") {
        Some(FamilyRole::Mother)
    } else if says("son") || says("daughter") || says("child") || says("kid") {
        Some(FamilyRole::Child)
    } else if says("grandfather") || says("grandpa") || says("ông nội") || says("ông ngoại") {
        Some(FamilyRole::Grandfather)
    } else if says("grandmother") || says("grandma") || says("bà nội") || says("bà ngoại") {
        Some(FamilyRole::Grandmother)
    } else if says("older brother") || says("anh trai") {
        Some(FamilyRole::OlderBrother)
    } else if says("older sister") || says("chị gái") {
        Some(FamilyRole::OlderSister)
    } else {
        None
    }
}

#[derive(PartialEq, Eq)]
enum FamilyRole {
    Father,
    Mother,
    Child,
    Grandfather,
    Grandmother,
    OlderBrother,
    OlderSister,
}

/// Suggest Vietnamese first-person and second-person pronouns based on speaker/listener labels.
/// Labels come either from WD Tagger ("Young Male", "Adult Female", …) or known character traits.
fn suggest_vn_pronoun_pair(
    speaker_label: &str,
    listener_label: &str,
) -> (&'static str, &'static str) {
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
            (Gender::Male, Gender::Female) => ("tôi", "cô"),
            (Gender::Male, Gender::Male) => ("tôi", "cậu"),
            (Gender::Female, Gender::Male) => ("tôi", "anh"),
            (Gender::Female, Gender::Female) => ("tôi", "cô"),
            (Gender::Unknown, Gender::Female) => ("tôi", "cô"),
            (Gender::Unknown, Gender::Male) => ("tôi", "anh"),
            _ => ("tôi", "cậu"),
        };
    }

    // Cả hai đều biết tuổi → xét chênh lệch già/trẻ để đổi ngôi
    match (sp_age, sp_gen, ls_age, ls_gen) {
        // Người già nói chuyện
        (Age::Old, Gender::Male, _, _) => ("ông", "cháu"),
        (Age::Old, Gender::Female, _, _) => ("bà", "cháu"),

        // Người trẻ nói với người già
        (_, _, Age::Old, Gender::Male) => ("cháu", "ông"),
        (_, _, Age::Old, Gender::Female) => ("cháu", "bà"),

        // Lớn hơn (Adult) nói với nhỏ hơn (Young)
        (Age::Adult, Gender::Male, Age::Young, _) => ("anh", "em"),
        (Age::Adult, Gender::Female, Age::Young, _) => ("chị", "em"),

        // Nhỏ hơn (Young) nói với lớn hơn (Adult)
        (Age::Young, _, Age::Adult, Gender::Male) => ("em", "anh"),
        (Age::Young, _, Age::Adult, Gender::Female) => ("em", "chị"),

        // Cùng tầm tuổi → theo giới tính
        (_, Gender::Male, _, Gender::Female) => ("tôi", "cô"),
        (_, Gender::Male, _, Gender::Male) => ("tôi", "cậu"),
        (_, Gender::Female, _, Gender::Male) => ("tôi", "anh"),
        (_, Gender::Female, _, Gender::Female) => ("tôi", "cô"),
        (_, Gender::Unknown, _, Gender::Female) => ("tôi", "cô"),
        (_, Gender::Unknown, _, Gender::Male) => ("tôi", "anh"),
        _ => ("tôi", "cậu"),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Age {
    Young,
    Adult,
    Old,
    Unknown,
}

#[derive(Debug, PartialEq, Eq)]
enum Gender {
    Male,
    Female,
    Unknown,
}

fn parse_age_gender(label: &str) -> (Age, Gender) {
    let says = |phrase: &str| mentions(label, phrase);

    let age = if says("old")
        || says("elder")
        || says("elderly")
        || says("senior")
        || says("grandfather")
        || says("grandmother")
    {
        Age::Old
    } else if says("young")
        || says("teen")
        || says("child")
        || says("kid")
        || says("boy")
        || says("girl")
    {
        Age::Young
    } else if says("adult") || says("man") || says("woman") {
        Age::Adult
    } else {
        Age::Unknown
    };

    let gender = if says("female") || says("woman") || says("girl") {
        Gender::Female
    } else if says("male") || says("man") || says("boy") {
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

/// Bounding boxes of ink the segmentation mask marks as text but that sits
/// nowhere near any already-detected block — the gate for whether it's worth
/// paying for a `ComicTextDetector` rescan pass at all (see its call site in
/// `detect`). Same core technique as `lama`'s `leftover_mask_regions`
/// (dilate to merge glyphs into runs, connected-component, size-filter), but
/// inverted: that one keeps ink *near* a block (finishing an inpainting job
/// already started); this one keeps ink *far from every* block (a run the
/// detector missed completely — measured case: text rotated for visual
/// effect, which `PPDocLayoutV3` structurally cannot represent).
fn uncovered_ink_regions(mask: &image::GrayImage, blocks: &[TextBlock]) -> Vec<[f32; 4]> {
    use imageproc::region_labelling::{Connectivity, connected_components};

    let (width, height) = mask.dimensions();
    if !mask.pixels().any(|pixel| pixel[0] >= 128) {
        return Vec::new();
    }

    let merged = imageproc::morphology::dilate(
        mask,
        imageproc::distance_transform::Norm::L1,
        MISSED_TEXT_MERGE_RADIUS,
    );
    let labels = connected_components(&merged, Connectivity::Eight, image::Luma([0u8]));

    let mut boxes: std::collections::HashMap<u32, [u32; 4]> = std::collections::HashMap::new();
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (x, y, label) in labels.enumerate_pixels() {
        let id = label[0];
        if id == 0 || mask.get_pixel(x, y)[0] < 128 {
            continue;
        }
        *counts.entry(id).or_insert(0) += 1;
        let entry = boxes.entry(id).or_insert([x, y, x + 1, y + 1]);
        entry[0] = entry[0].min(x);
        entry[1] = entry[1].min(y);
        entry[2] = entry[2].max(x + 1);
        entry[3] = entry[3].max(y + 1);
    }

    let page_area = (width as f32) * (height as f32);
    let far_from_every_block = |bbox: [u32; 4]| {
        blocks.iter().all(|b| {
            let gap = MISSED_TEXT_GAP_PX;
            let (bx0, by0, bx1, by1) = (b.x, b.y, b.x + b.width, b.y + b.height);
            let (ux0, uy0, ux1, uy1) =
                (bbox[0] as f32, bbox[1] as f32, bbox[2] as f32, bbox[3] as f32);
            // NOT "near" (mirrors `lama::near`, inverted): a real gap on at
            // least one axis wider than `gap` means this ink and that block
            // do not touch or continue one another.
            !(ux0 <= bx1 + gap && bx0 <= ux1 + gap && uy0 <= by1 + gap && by0 <= uy1 + gap)
        })
    };

    boxes
        .into_iter()
        .filter(|(id, _)| counts.get(id).copied().unwrap_or(0) >= MISSED_TEXT_MIN_PIXELS)
        .map(|(_, bbox)| bbox)
        .filter(|&bbox| far_from_every_block(bbox))
        .filter(|bbox| {
            let area = ((bbox[2] - bbox[0]) as f32) * ((bbox[3] - bbox[1]) as f32);
            area <= page_area * MISSED_TEXT_MAX_AREA_SHARE
        })
        .map(|bbox| [bbox[0] as f32, bbox[1] as f32, bbox[2] as f32, bbox[3] as f32])
        .collect()
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

/// Horizontal spans to run the layout detector over, one entry for a page that
/// fits the detector as it is.
///
/// Right to left: manga reads that way, and nothing downstream reorders the
/// blocks, so the order they are detected in is the order the translator sees
/// them.
fn detection_tiles(width: u32) -> Vec<(u32, u32)> {
    let count = ((width as f32) * MIN_DETECTOR_SCALE / DETECTOR_INPUT_SIZE).ceil() as u32;
    if count <= 1 {
        return vec![(0, width)];
    }

    let overlap = ((width as f32) * TILE_OVERLAP_RATIO).round() as u32;
    (0..count)
        .rev()
        .map(|i| {
            let left = width * i / count;
            let right = width * (i + 1) / count;
            (left.saturating_sub(overlap), (right + overlap).min(width))
        })
        .collect()
}

/// Throw away boxes a tile edge cut in half, keeping the whole box the
/// neighbouring tile found.
///
/// Left in, the fragment wins: it sits almost entirely inside the whole box,
/// which is enough for `dedupe_text_blocks` to drop the whole one in its
/// favour, and OCR then reads only the sliver. A fragment nothing else covers
/// is kept — half a box beats none.
fn drop_clipped_fragments(found: Vec<(LayoutRegion, bool)>) -> Vec<LayoutRegion> {
    let boxes: Vec<[f32; 4]> = found
        .iter()
        .map(|(region, _)| region_bbox(region))
        .collect();

    found
        .iter()
        .enumerate()
        .filter(|(i, (_, clipped))| {
            if !clipped {
                return true;
            }
            let bbox = boxes[*i];
            let area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])).max(1.0);
            !boxes.iter().enumerate().any(|(j, other)| {
                j != *i && overlap_area(bbox, *other) / area >= CLIPPED_REGION_COVERED_SHARE
            })
        })
        .map(|(_, (region, _))| region.clone())
        .collect()
}

/// Unmark everything the segmentation found that the pipeline is not replacing.
///
/// The model marks writing, and on a manga page some writing belongs to the
/// drawing: the jagged outline of a shout balloon, a sound effect lettered
/// beside the dialogue. Erasing those cuts the balloon open and leaves the art
/// a mark short, with nothing put back in their place.
///
/// Shape cannot tell them apart — measured on a real page, a balloon outline
/// and the text beside it had the same density, 0.58 against 0.59. Two things
/// can: a box the detector drew round text it found, and a balloon. A mark
/// inside a balloon is dialogue even where the box stops short of it, which is
/// how a second column of one balloon used to survive; a mark outside every
/// balloon and every box is part of the drawing.
///
/// A run only partly claimed is kept whole, so text clipped by a box edge is
/// still erased in one piece rather than half-erased.
fn keep_only_text_found(
    mask: image::GrayImage,
    blocks: &[TextBlock],
    balloons: &[koharu_types::BalloonDetection],
) -> image::GrayImage {
    use imageproc::region_labelling::{Connectivity, connected_components};

    /// How far inside a balloon a mark has to sit to count as its text. Its own
    /// outline runs the length of the balloon and reaches the edge of the box
    /// around it; the words do not.
    const BALLOON_OUTLINE_MARGIN: f32 = 3.0;

    /// How far outside a detected block's own box a mark may sit and still
    /// count as belonging to it. A dense glyph (a thick CJK character, an
    /// accent, a stroke with overshoot) can segment as ink a few pixels past
    /// a box the layout detector drew a little tight — measured on a real
    /// page, one such sliver never touched its block's exact box and was
    /// dropped as "art", surviving inpainting as a visible fragment under the
    /// translated text laid over it.
    const TEXT_BLOCK_TOUCH_MARGIN_PX: f32 = 4.0;

    let (width, height) = mask.dimensions();
    let labels = connected_components(&mask, Connectivity::Eight, image::Luma([0u8]));

    // Extent of every marked component, so a balloon can tell its own outline
    // from the words inside it.
    let mut extent: std::collections::HashMap<u32, [u32; 4]> = std::collections::HashMap::new();
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] < 128 {
            continue;
        }
        let id = labels.get_pixel(x, y)[0];
        let e = extent.entry(id).or_insert([x, y, x + 1, y + 1]);
        e[0] = e[0].min(x);
        e[1] = e[1].min(y);
        e[2] = e[2].max(x + 1);
        e[3] = e[3].max(y + 1);
    }

    let mut replaced: std::collections::HashSet<u32> = std::collections::HashSet::new();

    // A box the detector drew round text it found: anything it touches is text.
    for block in blocks {
        let left = (block.x - TEXT_BLOCK_TOUCH_MARGIN_PX).floor().max(0.0) as u32;
        let top = (block.y - TEXT_BLOCK_TOUCH_MARGIN_PX).floor().max(0.0) as u32;
        let right = ((block.x + block.width + TEXT_BLOCK_TOUCH_MARGIN_PX).ceil() as u32).min(width);
        let bottom =
            ((block.y + block.height + TEXT_BLOCK_TOUCH_MARGIN_PX).ceil() as u32).min(height);
        for y in top..bottom.max(top) {
            for x in left..right.max(left) {
                if mask.get_pixel(x, y)[0] >= 128 {
                    replaced.insert(labels.get_pixel(x, y)[0]);
                }
            }
        }
    }

    // A balloon claims the marks that sit wholly inside it. Claiming everything
    // within its box instead takes the balloon's own outline with it, which is
    // how a spiked shout balloon came back cut open and filled with white.
    for balloon in balloons {
        let left = balloon.x + BALLOON_OUTLINE_MARGIN;
        let top = balloon.y + BALLOON_OUTLINE_MARGIN;
        let right = balloon.x + balloon.width - BALLOON_OUTLINE_MARGIN;
        let bottom = balloon.y + balloon.height - BALLOON_OUTLINE_MARGIN;
        for (id, [ex1, ey1, ex2, ey2]) in &extent {
            if (*ex1 as f32) >= left
                && (*ey1 as f32) >= top
                && (*ex2 as f32) <= right
                && (*ey2 as f32) <= bottom
            {
                replaced.insert(*id);
            }
        }
    }

    let mut kept = image::GrayImage::new(width, height);
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] >= 128 && replaced.contains(&labels.get_pixel(x, y)[0]) {
            kept.put_pixel(x, y, image::Luma([255]));
        }
    }
    kept
}

/// Unmark the text mask wherever a block holds marks and no words, so the
/// inpainter leaves those pixels alone.
fn keep_wordless_marks(mut mask: image::GrayImage, blocks: &[TextBlock]) -> image::GrayImage {
    let (width, height) = (mask.width() as i64, mask.height() as i64);
    for block in blocks {
        let Some(text) = block.text.as_deref() else {
            continue;
        };
        if text.trim().is_empty() || koharu_types::carries_words(text) {
            continue;
        }
        let left = (block.x.floor() as i64).clamp(0, width);
        let top = (block.y.floor() as i64).clamp(0, height);
        let right = ((block.x + block.width).ceil() as i64).clamp(left, width);
        let bottom = ((block.y + block.height).ceil() as i64).clamp(top, height);
        for y in top..bottom {
            for x in left..right {
                mask.put_pixel(x as u32, y as u32, image::Luma([0]));
            }
        }
    }
    mask
}

fn region_bbox(region: &LayoutRegion) -> [f32; 4] {
    [
        region.bbox[0].min(region.bbox[2]),
        region.bbox[1].min(region.bbox[3]),
        region.bbox[0].max(region.bbox[2]),
        region.bbox[1].max(region.bbox[3]),
    ]
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
/// Longest run of background (non-mask) pixels, in image px, that the
/// straight line between two owners' centres may cross and still count as
/// "one balloon drawn touching another" rather than two the detector's box
/// merely happened to enclose together. Two balloons actually drawn touching
/// meet at a seam — the mask stays foreground the whole way across (a
/// concave dip at the seam at most, well under this). A real gap this long
/// means solid background — art, gutter, another character — sits between
/// them; see `owners_are_bridged`.
const MAX_BACKGROUND_BRIDGE_PX: f32 = 10.0;
/// Erosion radius (px) applied before the bridge check. A balloon detector
/// merging two unrelated balloons into one "shared" box doesn't always leave
/// open background between them on the direct line between centres — an
/// antialiased edge, a stray mark, or one balloon's tail happening to point
/// past the other can keep that exact line foreground even with real
/// background on either side. Eroding first removes anything narrower than
/// this — a stray mark or a tail (typically a few px wide) disappears, while
/// a genuine shared seam between two balloons drawn touching (wide enough to
/// actually hold two balloons' worth of text) survives.
const BRIDGE_ERODE_RADIUS: u32 = 14;
/// How much more elongated (long side / short side) a shared split's result
/// may be than the block's own originally-detected box before it's rejected
/// as a distorted sliver rather than a real balloon share. Additive, not a
/// multiplier: a block detected nearly square (aspect ~1, the common case
/// for a short line of text) still gets real room to become a properly
/// balloon-shaped box without tripping this, while a split that stretches
/// a block far past its own shape — measured: aspect 1.67 to 3.45 for a
/// block the detector merely merged with a distant balloon — gets caught.
const ASPECT_DISTORTION_MARGIN: f32 = 1.2;

/// Whether the straight line between two owners' centres stays inside
/// `mask`, eroded by `BRIDGE_ERODE_RADIUS` first, the whole way (allowing a
/// short break, see `MAX_BACKGROUND_BRIDGE_PX`) — the actual test for
/// "these two claim one balloon drawn touching another", as opposed to a
/// balloon detector's box that merely happened to enclose two unrelated
/// balloons (measured: a narration balloon and a small reaction balloon two
/// panel-rows apart, boxed as one "shared" detection). Splitting a mask on
/// the latter produces a sliver nowhere near either balloon's real shape
/// rather than a clean seam — see its caller.
fn owners_are_bridged(mask: &image::GrayImage, a: (f32, f32), b: (f32, f32)) -> bool {
    let (img_w, img_h) = mask.dimensions();
    let margin = BRIDGE_ERODE_RADIUS as f32 + 4.0;
    let x0 = (a.0.min(b.0) - margin).max(0.0) as u32;
    let y0 = (a.1.min(b.1) - margin).max(0.0) as u32;
    let x1 = (((a.0.max(b.0) + margin) as u32) + 1).min(img_w);
    let y1 = (((a.1.max(b.1) + margin) as u32) + 1).min(img_h);
    if x1 <= x0 || y1 <= y0 {
        return false;
    }
    let cropped = image::imageops::crop_imm(mask, x0, y0, x1 - x0, y1 - y0).to_image();
    let eroded = bubble_det::erode_binary(&cropped, BRIDGE_ERODE_RADIUS);
    let (w, h) = eroded.dimensions();
    let (a, b) = ((a.0 - x0 as f32, a.1 - y0 as f32), (b.0 - x0 as f32, b.1 - y0 as f32));

    let dist = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    if dist < 1.0 {
        return true;
    }
    let steps = (dist / 2.0).ceil().max(1.0) as usize;
    let step_len = dist / steps as f32;
    let mut background_run = 0.0f32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        let in_bounds = x >= 0.0 && y >= 0.0 && (x as u32) < w && (y as u32) < h;
        let is_background = !in_bounds || eroded.get_pixel(x as u32, y as u32)[0] < 128;
        if is_background {
            background_run += step_len;
            if background_run > MAX_BACKGROUND_BRIDGE_PX {
                return false;
            }
        } else {
            background_run = 0.0;
        }
    }
    true
}

/// Where a mask narrows the most between two owners along the straight line
/// connecting them — the seam of a balloon drawn as one pinched hourglass
/// shape holding two lines with a pause between them (a common convention),
/// as opposed to the perpendicular bisector `nearest_owner` uses at the
/// midpoint between the two centres, which falls inside whichever lobe is
/// smaller rather than at the pinch and leaves that lobe's share a
/// distorted sliver. See its use in `refit_text_blocks_to_balloons`.
///
/// Only the middle 60% of the line is searched: the ends sit inside each
/// owner's own text, dense enough that a gap between glyphs there can look
/// narrower than the real waist.
///
/// Returns the fraction along a->b (0..1) of the narrowest crossing, or
/// `None` when there is nothing to find (the owners are too close together,
/// or the line leaves the image).
fn waist_fraction(mask: &image::GrayImage, a: (f32, f32), b: (f32, f32)) -> Option<f32> {
    let (img_w, img_h) = mask.dimensions();
    let dist = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    if dist < 8.0 {
        return None;
    }
    let dir = ((b.0 - a.0) / dist, (b.1 - a.1) / dist);
    let perp = (-dir.1, dir.0);
    const SEARCH_LO: f32 = 0.2;
    const SEARCH_HI: f32 = 0.8;
    const MAX_HALF_SPAN: f32 = 400.0;
    let steps = ((dist * (SEARCH_HI - SEARCH_LO)) / 2.0).ceil().max(1.0) as usize;

    let width_at = |t: f32| -> Option<f32> {
        let (px, py) = (a.0 + dir.0 * dist * t, a.1 + dir.1 * dist * t);
        if px < 0.0 || py < 0.0 || px as u32 >= img_w || py as u32 >= img_h {
            return None;
        }
        if mask.get_pixel(px as u32, py as u32)[0] < 128 {
            // The axis itself left the shape here — a concave pinch can do
            // that — which makes this point a maximally narrow candidate.
            return Some(0.0);
        }
        let mut width = 0.0f32;
        for sign in [-1.0f32, 1.0f32] {
            let mut s = 1.0f32;
            loop {
                let x = px + perp.0 * s * sign;
                let y = py + perp.1 * s * sign;
                if x < 0.0 || y < 0.0 || x as u32 >= img_w || y as u32 >= img_h {
                    break;
                }
                if mask.get_pixel(x as u32, y as u32)[0] < 128 {
                    break;
                }
                s += 1.0;
                if s > MAX_HALF_SPAN {
                    break;
                }
            }
            width += s - 1.0;
        }
        Some(width)
    };

    let mut best_t = None;
    let mut best_width = f32::INFINITY;
    for i in 0..=steps {
        let t = SEARCH_LO + (SEARCH_HI - SEARCH_LO) * (i as f32 / steps as f32);
        if let Some(width) = width_at(t)
            && width < best_width
        {
            best_width = width;
            best_t = Some(t);
        }
    }
    best_t
}

/// Which side of the waist point a mask pixel falls on, projected onto the
/// a->b axis: `true` for the `a` side. Paired with `waist_fraction`'s `t`.
fn owns_by_waist(px: u32, py: u32, a: (f32, f32), b: (f32, f32), dist: f32, t: f32) -> bool {
    let dir = ((b.0 - a.0) / dist, (b.1 - a.1) / dist);
    let proj = ((px as f32 - a.0) * dir.0 + (py as f32 - a.1) * dir.1) / dist;
    proj < t
}

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

        // Balloons drawn touching each other come back as one detection holding
        // several text blocks. Skipping them (the old behaviour) left every
        // block at its small detected size; laying one out across the whole
        // merged shape puts its text over the neighbour. Instead each block
        // claims the part of the shape nearest to it — the split falls on the
        // join between the balloons — and is fitted inside its own share.
        let owners: Vec<(usize, (f32, f32))> = inside_indices
            .iter()
            .map(|&i| (i, orig_centers[i]))
            .collect();
        let shared = owners.len() > 1;

        // A pinched hourglass balloon — one shape drawn for two lines with a
        // pause between them, a common convention, not two balloons the
        // detector merged — has its own natural seam: the waist. Cutting
        // there instead of at the perpendicular bisector `nearest_owner`
        // uses (the midpoint between the two centres, which falls inside
        // whichever lobe is smaller, not at the pinch) gives each lobe
        // something shaped like the balloon the artist actually drew. Only
        // handled for the common two-owner case.
        let waist = if let [(id_a, ca), (id_b, cb)] = owners[..] {
            balloon
                .mask
                .as_ref()
                .and_then(|m| waist_fraction(m, ca, cb))
                .map(|t| {
                    let dist = ((cb.0 - ca.0).powi(2) + (cb.1 - ca.1).powi(2)).sqrt();
                    (id_a, id_b, ca, cb, dist, t)
                })
        } else {
            None
        };

        for &ti in &inside_indices {
            let mir = match &balloon.mask {
                Some(mask) => {
                    // Only trust the split when `ti` is actually bridged to
                    // at least one other owner through solid mask — otherwise
                    // the balloon detector's box just happened to span two
                    // unrelated balloons, and forcing a split between them
                    // would fit this block into whatever distorted sliver of
                    // the mask is nearest its centre. See
                    // `owners_are_bridged`.
                    let treat_as_shared = shared
                        && owners.iter().any(|&(oi, oc)| {
                            oi != ti && owners_are_bridged(mask, orig_centers[ti], oc)
                        });
                    let r = if treat_as_shared {
                        let r = match waist {
                            Some((id_a, _, ca, cb, dist, t)) => {
                                let want_a = ti == id_a;
                                mir_from_mask_owned(
                                    mask,
                                    balloon.x,
                                    balloon.y,
                                    balloon.width,
                                    balloon.height,
                                    |px, py| {
                                        owns_by_waist(px, py, ca, cb, dist, t) == want_a
                                    },
                                )
                            }
                            None => mir_from_mask_owned(
                                mask,
                                balloon.x,
                                balloon.y,
                                balloon.width,
                                balloon.height,
                                |px, py| nearest_owner(px, py, &owners) == ti,
                            ),
                        };
                        // A bridged mask can still be a detector error (a
                        // segmentation model can predict one solid blob over
                        // two unrelated balloons just as easily as a box
                        // regressor can draw one box over them — erosion
                        // only catches the latter). Splitting a genuinely
                        // shared balloon fits each owner into something
                        // roughly as proportioned as what was detected for
                        // it; splitting a wrongly-merged one instead stretches
                        // a block far past its own shape trying to fill a
                        // sliver of someone else's balloon. Reject that.
                        let block = &text_blocks[ti];
                        let orig_aspect =
                            block.width.max(block.height) / block.width.min(block.height).max(1.0);
                        let r_aspect = r[2].max(r[3]) / r[2].min(r[3]).max(1.0);
                        if r[2] > 2.0 && r[3] > 2.0 && r_aspect > orig_aspect + ASPECT_DISTORTION_MARGIN
                        {
                            tracing::debug!(
                                balloon = bi,
                                block = ti,
                                orig_aspect,
                                r_aspect,
                                "shared split too distorted from original shape, keeping detection"
                            );
                            [0.0, 0.0, 0.0, 0.0]
                        } else {
                            r
                        }
                    } else if shared {
                        // Merged balloon, but this block doesn't plausibly
                        // belong to it — fall through to the "too small to
                        // trust" branch below, which keeps the block as
                        // originally detected.
                        [0.0, 0.0, 0.0, 0.0]
                    } else {
                        mir_from_mask(mask, balloon.x, balloon.y, balloon.width, balloon.height)
                    };
                    if r[2] > 2.0 && r[3] > 2.0 {
                        r
                    } else if shared {
                        // A share too small to fit anything: leave the block as
                        // detected rather than forcing it into a sliver.
                        tracing::debug!(
                            balloon = bi,
                            block = ti,
                            "share too small, keeping detection"
                        );
                        continue;
                    } else {
                        bbox_inset(balloon)
                    }
                }
                None if shared => continue,
                None => bbox_inset(balloon),
            };

            tracing::info!(
                balloon = bi,
                block = ti,
                shared,
                mir_x = mir[0],
                mir_y = mir[1],
                mir_w = mir[2],
                mir_h = mir[3],
                "refit"
            );

            let block = &mut text_blocks[ti];
            let pad = MIR_TEXT_PADDING;
            block.x = mir[0] + pad;
            block.y = mir[1] + pad;
            block.width = (mir[2] - 2.0 * pad).max(1.0);
            block.height = (mir[3] - 2.0 * pad).max(1.0);
            // Clear seed layout so the renderer uses the new refit coordinates.
            block.layout_seed_x = None;
            block.layout_seed_y = None;
            block.layout_seed_width = None;
            block.layout_seed_height = None;
            // Prevent the renderer from re-scanning the image for balloon bounds
            // or auto-expanding the layout box — the MIR coordinates are authoritative.
            block.lock_layout_box = true;
            block.balloon_fitted = true;
            refit_count += 1;
        }
    }

    tracing::info!(
        refit = refit_count,
        total = text_blocks.len(),
        "text blocks refit to balloon MIR"
    );
}

/// Compute the Maximum Inscribed Rectangle from a binary mask via the classical
/// "largest rectangle in histogram" algorithm applied row-by-row (O(w×h)).
/// Operates on the eroded mask to stay safely inside the balloon boundary.
/// All returned coordinates are in full-image space.
/// Falls back to zeros if no usable region found (caller uses bbox_inset).
fn mir_from_mask(mask: &image::GrayImage, bx: f32, by: f32, bw: f32, bh: f32) -> [f32; 4] {
    mir_from_mask_owned(mask, bx, by, bw, bh, |_, _| true)
}

/// Like [`mir_from_mask`], but only mask pixels accepted by `owns` (given in
/// full-image coordinates) count as usable area.
///
/// This is what lets two balloons drawn touching each other end up with one box
/// apiece: each block claims the part of the shared white region nearest to it,
/// and takes the largest rectangle inside its own share.
fn mir_from_mask_owned(
    mask: &image::GrayImage,
    bx: f32,
    by: f32,
    bw: f32,
    bh: f32,
    owns: impl Fn(u32, u32) -> bool,
) -> [f32; 4] {
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
    let mut cropped = image::imageops::crop_imm(mask, cx0, cy0, cw, ch).to_image();
    for y in 0..ch {
        for x in 0..cw {
            if !owns(cx0 + x, cy0 + y) {
                cropped.put_pixel(x, y, image::Luma([0]));
            }
        }
    }

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

/// The block whose centre is closest to a pixel. The boundary between two
/// owners is their perpendicular bisector, which for two balloons drawn
/// touching runs along the join between them.
fn nearest_owner(px: u32, py: u32, owners: &[(usize, (f32, f32))]) -> usize {
    let (x, y) = (px as f32, py as f32);
    owners
        .iter()
        .min_by(|(_, a), (_, b)| {
            let da = (a.0 - x).powi(2) + (a.1 - y).powi(2);
            let db = (b.0 - x).powi(2) + (b.1 - y).powi(2);
            da.total_cmp(&db)
        })
        .map(|(index, _)| *index)
        .unwrap_or(usize::MAX)
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
    use super::{
        TextBlock, bubble_det, mir_from_mask, mir_from_mask_owned, nearest_owner,
        refit_text_blocks_to_balloons,
    };

    /// Report whether sample points fall inside a detected balloon mask.
    ///
    /// `KOHARU_TEST_PAGE=page.jpg KOHARU_TEST_POINTS="x,y;x,y" cargo test --release
    /// -p koharu-ml --lib probe_balloon_coverage -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_balloon_coverage() -> anyhow::Result<()> {
        let Some(page) = std::env::var_os("KOHARU_TEST_PAGE") else {
            anyhow::bail!("set KOHARU_TEST_PAGE");
        };
        let points: Vec<(u32, u32)> = std::env::var("KOHARU_TEST_POINTS")
            .unwrap_or_default()
            .split(';')
            .filter_map(|pair| {
                let (x, y) = pair.trim().split_once(',')?;
                Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
            })
            .collect();

        let image = image::open(std::path::PathBuf::from(&page))?;
        let runtime = tokio::runtime::Runtime::new()?;
        let detector = runtime.block_on(super::ComicBubbleDetector::load())?;
        let balloons = detector.detect(&image)?;
        let gray = image.to_luma8();

        for (x, y) in points {
            let owner = balloons
                .iter()
                .position(|b| b.mask.as_ref().is_some_and(|m| m.get_pixel(x, y)[0] >= 128));
            let in_bbox = balloons.iter().position(|b| {
                x as f32 >= b.x
                    && x as f32 <= b.x + b.width
                    && y as f32 >= b.y
                    && y as f32 <= b.y + b.height
            });
            println!(
                "({x},{y}) luma={:>3}  mask: {:?}  bbox: {:?}",
                gray.get_pixel(x, y)[0],
                owner,
                in_bbox
            );
        }
        Ok(())
    }

    /// Dump the segmentation mask the pipeline actually uses, for comparison
    /// against alternatives.
    ///
    /// Print the layout boxes the detector returns for the whole page and for
    /// each tile, so a box that goes missing can be traced to the pass that
    /// should have found it.
    ///
    /// `KOHARU_TEST_PAGE=page.jpg cargo test --release -p koharu-ml --lib
    /// dump_layout_tiles -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_layout_tiles() -> anyhow::Result<()> {
        let page = std::env::var_os("KOHARU_TEST_PAGE")
            .ok_or_else(|| anyhow::anyhow!("set KOHARU_TEST_PAGE"))?;
        let image = image::open(std::path::PathBuf::from(&page))?;
        let runtime = tokio::runtime::Runtime::new()?;
        let detector = runtime.block_on(super::PPDocLayoutV3::load(false))?;

        let mut passes = vec![("whole".to_string(), (0, image.width()))];
        for (left, right) in super::detection_tiles(image.width()) {
            passes.push((format!("tile {left}..{right}"), (left, right)));
        }

        for (name, (left, right)) in passes {
            let crop = image.crop_imm(left, 0, right - left, image.height());
            let found = detector.inference_one_fast(&crop, super::PP_DOCLAYOUT_THRESHOLD)?;
            let blocks = super::build_text_blocks(&found.regions);
            println!(
                "--- {name} ({}x{}) — {} blocks",
                crop.width(),
                crop.height(),
                blocks.len()
            );
            for block in &blocks {
                println!(
                    "    x {:>5.0}..{:<5.0} y {:>5.0}..{:<5.0}  {:>3.0}x{:<3.0}",
                    block.x + left as f32,
                    block.x + block.width + left as f32,
                    block.y,
                    block.y + block.height,
                    block.width,
                    block.height
                );
            }
        }
        Ok(())
    }

    /// `KOHARU_TEST_PAGE=page.jpg KOHARU_TEST_OUT=mask.png cargo test --release
    /// -p koharu-ml --lib dump_pipeline_segmentation_mask -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_pipeline_segmentation_mask() -> anyhow::Result<()> {
        let Some(page) = std::env::var_os("KOHARU_TEST_PAGE") else {
            anyhow::bail!("set KOHARU_TEST_PAGE");
        };
        let out = std::env::var_os("KOHARU_TEST_OUT")
            .ok_or_else(|| anyhow::anyhow!("set KOHARU_TEST_OUT"))?;

        let image = image::open(std::path::PathBuf::from(&page))?;
        let runtime = tokio::runtime::Runtime::new()?;

        let layout_detector = runtime.block_on(super::PPDocLayoutV3::load(false))?;
        let layout = layout_detector.inference_one_fast(&image, super::PP_DOCLAYOUT_THRESHOLD)?;
        let blocks = super::build_text_blocks(&layout.regions);

        // The same mask `detect` builds, so what this shows is what the
        // inpainter is handed. It used to dump the old brightness-refined mask
        // from a segmenter the pipeline no longer loads, which only reported a
        // missing model file.
        let segmenter = runtime.block_on(super::MangaTextSegmentation::load(false))?;
        let probability_map = segmenter.inference(&image)?;
        let mask = probability_map.threshold(super::TEXT_MASK_THRESHOLD)?;
        let mask = imageproc::morphology::dilate(
            &mask,
            imageproc::distance_transform::Norm::L1,
            super::TEXT_MASK_DILATE_RADIUS,
        );
        let _ = &blocks;
        let covered = mask.pixels().filter(|p| p[0] >= 128).count();
        println!(
            "mask {}x{} — {:.2}% of the page marked as text",
            mask.width(),
            mask.height(),
            100.0 * covered as f64 / (mask.width() * mask.height()) as f64
        );
        mask.save(std::path::PathBuf::from(out))?;
        Ok(())
    }

    /// Diagnostic: report, for a real page, which text blocks a balloon was
    /// found for and how much the refit grew them.
    ///
    /// Run with: `KOHARU_TEST_PAGE=test/page.JPG cargo test -p koharu-ml --lib
    /// diagnose_balloon_fit -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn diagnose_balloon_fit_on_a_real_page() -> anyhow::Result<()> {
        let Some(page) = std::env::var_os("KOHARU_TEST_PAGE") else {
            anyhow::bail!("set KOHARU_TEST_PAGE");
        };
        let image = image::open(std::path::PathBuf::from(&page))?;
        let runtime = tokio::runtime::Runtime::new()?;

        let layout_detector = runtime.block_on(super::PPDocLayoutV3::load(true))?;
        let layout = layout_detector.inference_one_fast(&image, super::PP_DOCLAYOUT_THRESHOLD)?;
        let mut blocks = super::build_text_blocks(&layout.regions);
        let before: Vec<(f32, f32, f32, f32)> = blocks
            .iter()
            .map(|b| (b.x, b.y, b.width, b.height))
            .collect();

        let detector = runtime.block_on(super::ComicBubbleDetector::load())?;
        let balloons = detector.detect(&image)?;
        println!(
            "page {}x{} — {} text blocks, {} balloons",
            image.width(),
            image.height(),
            blocks.len(),
            balloons.len()
        );

        println!(
            "\n{:>3} {:>24} {:>24} {:>8} {:>6}",
            "b", "balloon bbox", "mir", "mir/bbox", "mask"
        );
        for (i, balloon) in balloons.iter().enumerate() {
            let mir = match &balloon.mask {
                Some(mask) => {
                    super::mir_from_mask(mask, balloon.x, balloon.y, balloon.width, balloon.height)
                }
                None => [0.0; 4],
            };
            let ratio = (mir[2] * mir[3]) / (balloon.width * balloon.height).max(1.0);
            println!(
                "{i:>3} {:>24} {:>24} {ratio:>7.2} {:>6}",
                format!(
                    "{:.0},{:.0} {:.0}x{:.0}",
                    balloon.x, balloon.y, balloon.width, balloon.height
                ),
                format!("{:.0},{:.0} {:.0}x{:.0}", mir[0], mir[1], mir[2], mir[3]),
                balloon.mask.is_some()
            );
        }

        super::refit_text_blocks_to_balloons(&mut blocks, &balloons);

        println!(
            "{:>3} {:>22} {:>22} {:>7} {:>6} {:>10} {:>9}",
            "idx", "detected", "after refit", "grew", "lock", "in_balloon", "src_glyph"
        );
        for (i, block) in blocks.iter().enumerate() {
            let (ox, oy, ow, oh) = before[i];
            let grew = (block.width * block.height) / (ow * oh).max(1.0);
            // The size proxy the renderer uses, measured on the *detected* box
            // (before refit) — that is the one that reflects the raw lettering.
            let src_glyph = ow.min(oh);
            let cx = ox + ow / 2.0;
            let cy = oy + oh / 2.0;
            let in_balloon = balloons
                .iter()
                .any(|b| cx >= b.x && cx <= b.x + b.width && cy >= b.y && cy <= b.y + b.height);
            println!(
                "{i:>3} {:>22} {:>22} {grew:>6.1}x {:>6} {in_balloon:>10} {src_glyph:>9.0}",
                format!("{ox:.0},{oy:.0} {ow:.0}x{oh:.0}"),
                format!(
                    "{:.0},{:.0} {:.0}x{:.0}",
                    block.x, block.y, block.width, block.height
                ),
                block.lock_layout_box
            );
        }

        let refit = blocks.iter().filter(|b| b.lock_layout_box).count();
        println!("\nrefit {refit}/{} blocks", blocks.len());

        // Draw what the layout engine will be given, so the fit can be judged
        // by eye instead of by numbers.
        if let Some(out) = std::env::var_os("KOHARU_TEST_OUT") {
            let mut canvas = image.to_rgb8();
            let mut outline = |x: f32, y: f32, w: f32, h: f32, colour: [u8; 3]| {
                let (x0, y0) = (x.max(0.0) as u32, y.max(0.0) as u32);
                let x1 = ((x + w) as u32).min(canvas.width().saturating_sub(1));
                let y1 = ((y + h) as u32).min(canvas.height().saturating_sub(1));
                for px in x0..=x1.max(x0) {
                    for py in [y0, y1] {
                        if px < canvas.width() && py < canvas.height() {
                            canvas.put_pixel(px, py, image::Rgb(colour));
                        }
                    }
                }
                for py in y0..=y1.max(y0) {
                    for px in [x0, x1] {
                        if px < canvas.width() && py < canvas.height() {
                            canvas.put_pixel(px, py, image::Rgb(colour));
                        }
                    }
                }
            };
            for balloon in &balloons {
                outline(
                    balloon.x,
                    balloon.y,
                    balloon.width,
                    balloon.height,
                    [0, 160, 255],
                );
            }
            for block in &blocks {
                let colour = if block.lock_layout_box {
                    [255, 0, 0]
                } else {
                    [255, 160, 0]
                };
                outline(block.x, block.y, block.width, block.height, colour);
            }
            canvas.save(std::path::PathBuf::from(out))?;
        }
        Ok(())
    }

    /// Diagnostic: does `ComicTextDetector` (YOLOv5+DBNet, quad/rotation-aware
    /// — its full `inference()` path is currently unused by the live
    /// pipeline, which only uses `PPDocLayoutV3` for finding text blocks)
    /// catch text that `PPDocLayoutV3` misses, specifically rotated text?
    ///
    /// Run with: `KOHARU_TEST_PAGE=test/page.png cargo test --release
    /// -p koharu-ml --lib dump_comic_text_detector_regions -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_comic_text_detector_regions() -> anyhow::Result<()> {
        let Some(page) = std::env::var_os("KOHARU_TEST_PAGE") else {
            anyhow::bail!("set KOHARU_TEST_PAGE");
        };
        let image = image::open(std::path::PathBuf::from(&page))?;
        let runtime = tokio::runtime::Runtime::new()?;

        let layout_detector = runtime.block_on(super::PPDocLayoutV3::load(true))?;
        let layout = layout_detector.inference_one_fast(&image, super::PP_DOCLAYOUT_THRESHOLD)?;
        let pp_blocks = super::build_text_blocks(&layout.regions);
        println!(
            "PPDocLayoutV3 (currently used for detection): {} blocks",
            pp_blocks.len()
        );
        for (i, b) in pp_blocks.iter().enumerate() {
            println!(
                "  [{i:>2}] x {:>5.0}..{:<5.0} y {:>5.0}..{:<5.0}  {:>3.0}x{:<3.0}",
                b.x,
                b.x + b.width,
                b.y,
                b.y + b.height,
                b.width,
                b.height
            );
        }

        let ctd = runtime.block_on(crate::comic_text_detector::ComicTextDetector::load(true))?;
        let detection = ctd.inference(&image)?;
        println!(
            "\nComicTextDetector (YOLOv5+DBNet, NOT currently wired into the pipeline): \
             {} text_blocks, {} line_polygons",
            detection.text_blocks.len(),
            detection.line_polygons.len()
        );
        for (i, b) in detection.text_blocks.iter().enumerate() {
            println!(
                "  [{i:>2}] x {:>5.0}..{:<5.0} y {:>5.0}..{:<5.0}  {:>3.0}x{:<3.0}  vertical={:?}",
                b.x,
                b.x + b.width,
                b.y,
                b.y + b.height,
                b.width,
                b.height,
                b.rendered_direction,
            );
        }
        println!("\nline_polygons (quads, may be rotated):");
        for (i, q) in detection.line_polygons.iter().enumerate() {
            println!("  [{i:>2}] {q:?}");
        }

        // Which PPDocLayoutV3 blocks have no ComicTextDetector block whose
        // centre falls inside them (or vice versa) — the mismatch this test
        // exists to surface.
        println!("\nPPDocLayoutV3 blocks with no ComicTextDetector block centred inside them:");
        for (i, b) in pp_blocks.iter().enumerate() {
            let (bx0, by0, bx1, by1) = (b.x, b.y, b.x + b.width, b.y + b.height);
            let covered = detection.text_blocks.iter().any(|c| {
                let (cx, cy) = (c.x + c.width / 2.0, c.y + c.height / 2.0);
                cx >= bx0 && cx <= bx1 && cy >= by0 && cy <= by1
            });
            if !covered {
                println!("  [{i:>2}] uncovered: {b:?}");
            }
        }
        println!("\nComicTextDetector blocks with no PPDocLayoutV3 block centred inside them:");
        for (i, c) in detection.text_blocks.iter().enumerate() {
            let (cx, cy) = (c.x + c.width / 2.0, c.y + c.height / 2.0);
            let covered = pp_blocks.iter().any(|b| {
                cx >= b.x && cx <= b.x + b.width && cy >= b.y && cy <= b.y + b.height
            });
            if !covered {
                println!("  [{i:>2}] extra (PPDocLayoutV3 missed this): {c:?}");
            }
        }

        Ok(())
    }

    /// Two circles drawn overlapping — one white region, as merged balloons appear.
    fn merged_balloon_mask(
        width: u32,
        height: u32,
        centres: &[(f32, f32)],
        radius: f32,
    ) -> image::GrayImage {
        image::GrayImage::from_fn(width, height, |x, y| {
            let inside = centres.iter().any(|(cx, cy)| {
                ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() <= radius
            });
            image::Luma([if inside { 255u8 } else { 0u8 }])
        })
    }

    fn block_at(cx: f32, cy: f32) -> TextBlock {
        TextBlock {
            x: cx - 5.0,
            y: cy - 5.0,
            width: 10.0,
            height: 10.0,
            ..Default::default()
        }
    }

    #[test]
    fn uncovered_ink_regions_finds_a_run_far_from_every_block() {
        // A run of glyphs (four blobs on one line, big enough to clear
        // MISSED_TEXT_MIN_PIXELS) nowhere near the one detected block.
        let mask = image::GrayImage::from_fn(800, 600, |x, y| {
            let on = (500..540).contains(&y)
                && [(600, 630), (635, 665), (670, 700)]
                    .iter()
                    .any(|(a, b)| x >= *a && x < *b);
            image::Luma([if on { 255u8 } else { 0 }])
        });
        let blocks = vec![block_at(50.0, 50.0)];

        let found = uncovered_ink_regions(&mask, &blocks);
        assert_eq!(found.len(), 1, "{found:?}");
        let [x1, y1, x2, y2] = found[0];
        assert!(x1 <= 600.0 && x2 >= 700.0 && y1 <= 500.0 && y2 >= 540.0, "{found:?}");
    }

    #[test]
    fn uncovered_ink_regions_ignores_a_run_touching_an_existing_block() {
        // Same run, but right up against a block's own edge — this is text a
        // block already accounts for continuing slightly past its box, not a
        // run the detector missed outright.
        let mask = image::GrayImage::from_fn(800, 600, |x, y| {
            let on = (500..540).contains(&y) && (600..700).contains(&x);
            image::Luma([if on { 255u8 } else { 0 }])
        });
        let blocks = vec![TextBlock {
            x: 700.0,
            y: 500.0,
            width: 40.0,
            height: 40.0,
            ..Default::default()
        }];

        assert!(
            uncovered_ink_regions(&mask, &blocks).is_empty(),
            "a run touching a block's edge is not a miss"
        );
    }

    #[test]
    fn uncovered_ink_regions_ignores_specks_below_the_pixel_floor() {
        let mask = image::GrayImage::from_fn(800, 600, |x, y| {
            image::Luma([if x == 400 && y == 300 { 255u8 } else { 0 }])
        });
        assert!(uncovered_ink_regions(&mask, &[]).is_empty());
    }

    #[test]
    fn nearest_owner_splits_on_the_bisector() {
        let owners = vec![(0usize, (40.0, 50.0)), (1usize, (160.0, 50.0))];
        assert_eq!(nearest_owner(50, 50, &owners), 0);
        assert_eq!(nearest_owner(150, 50, &owners), 1);
        // The join sits midway between the two centres.
        assert_eq!(nearest_owner(99, 50, &owners), 0);
        assert_eq!(nearest_owner(101, 50, &owners), 1);
    }

    #[test]
    fn merged_balloons_get_one_box_each_that_do_not_overlap() {
        let centres = [(70.0f32, 80.0f32), (170.0, 80.0)];
        let mask = merged_balloon_mask(240, 160, &centres, 60.0);
        let balloon = bubble_det::BubbleBox {
            x: 10.0,
            y: 20.0,
            width: 220.0,
            height: 120.0,
            score: 0.9,
            mask: Some(mask),
        };

        let mut blocks = vec![
            block_at(centres[0].0, centres[0].1),
            block_at(centres[1].0, centres[1].1),
        ];
        refit_text_blocks_to_balloons(&mut blocks, std::slice::from_ref(&balloon));

        // Both grew well past the 10x10 they were detected at.
        for block in &blocks {
            assert!(block.width > 20.0 && block.height > 20.0, "{block:?}");
            assert!(block.lock_layout_box, "refit boxes are authoritative");
            assert!(block.balloon_fitted, "and are marked as balloon-derived");
        }
        // And neither reaches across the join into the other balloon.
        let left_right = blocks[0].x + blocks[0].width;
        assert!(
            left_right <= blocks[1].x,
            "boxes overlap: {left_right} > {}",
            blocks[1].x
        );
    }

    /// Two balloons nowhere near each other (a real background gap between
    /// them, not a seam) that a detector's box regression merely boxed
    /// together — measured on a real page: a narration balloon and a small
    /// reaction balloon, boxed as one "shared" detection despite sitting
    /// two panel-rows apart. Splitting that merged mask must not run; each
    /// block should keep its own detected box rather than being fitted into
    /// a distorted sliver nearest its centre.
    #[test]
    fn balloons_the_detector_merely_boxed_together_are_not_forced_into_a_split() {
        let centres = [(60.0f32, 60.0f32), (60.0, 220.0)];
        let mask = merged_balloon_mask(120, 280, &centres, 30.0);
        let balloon = bubble_det::BubbleBox {
            x: 10.0,
            y: 10.0,
            width: 100.0,
            height: 260.0,
            score: 0.9,
            mask: Some(mask),
        };

        let mut blocks = vec![
            block_at(centres[0].0, centres[0].1),
            block_at(centres[1].0, centres[1].1),
        ];
        refit_text_blocks_to_balloons(&mut blocks, std::slice::from_ref(&balloon));

        for block in &blocks {
            assert_eq!(block.width, 10.0, "{block:?}");
            assert_eq!(block.height, 10.0, "{block:?}");
            assert!(!block.lock_layout_box, "kept as detected, not balloon-derived");
            assert!(!block.balloon_fitted, "{block:?}");
        }
    }

    /// Two circles of different sizes joined by a narrow rectangular neck —
    /// a pinched hourglass balloon, the shape a real "two lines with a
    /// pause between them" balloon draws.
    fn hourglass_mask(
        width: u32,
        height: u32,
        top_centre: (f32, f32),
        top_radius: f32,
        bottom_centre: (f32, f32),
        bottom_radius: f32,
        neck_half_width: f32,
        neck_y: (f32, f32),
    ) -> image::GrayImage {
        image::GrayImage::from_fn(width, height, |x, y| {
            let (fx, fy) = (x as f32, y as f32);
            let in_top = ((fx - top_centre.0).powi(2) + (fy - top_centre.1).powi(2)).sqrt()
                <= top_radius;
            let in_bottom = ((fx - bottom_centre.0).powi(2) + (fy - bottom_centre.1).powi(2))
                .sqrt()
                <= bottom_radius;
            let in_neck = fy >= neck_y.0
                && fy <= neck_y.1
                && (fx - top_centre.0).abs() <= neck_half_width;
            image::Luma([if in_top || in_bottom || in_neck { 255u8 } else { 0u8 }])
        })
    }

    #[test]
    fn a_pinched_hourglass_balloon_is_split_at_the_waist_not_the_midpoint() {
        // Small lobe on top, much larger lobe on bottom, joined by a short
        // narrow neck well above the midpoint between the two centres —
        // exactly the case where the old perpendicular-bisector split (at
        // the midpoint) lands inside the bottom lobe instead of at the neck.
        let top_centre = (150.0f32, 70.0);
        let top_radius = 55.0;
        // Starts/ends where each circle is already at least as wide as the
        // neck, so the rectangle-meets-circle join has no corner narrower
        // than the neck itself — a real hand-drawn taper has no such corner
        // either; a mask that did would not be a realistic hourglass.
        let neck_y = (110.0, 162.0);
        let bottom_centre = (150.0f32, 305.0);
        let bottom_radius = 150.0;
        // Wide enough to survive `owners_are_bridged`'s erosion (a genuine
        // pinch, not the thin stray-mark case that check exists to reject).
        let mask = hourglass_mask(
            300,
            460,
            top_centre,
            top_radius,
            bottom_centre,
            bottom_radius,
            35.0,
            neck_y,
        );
        let balloon = bubble_det::BubbleBox {
            x: 0.0,
            y: 0.0,
            width: 300.0,
            height: 460.0,
            score: 0.9,
            mask: Some(mask),
        };

        let mut blocks = vec![block_at(top_centre.0, top_centre.1), block_at(bottom_centre.0, bottom_centre.1)];
        refit_text_blocks_to_balloons(&mut blocks, std::slice::from_ref(&balloon));

        // The midpoint between centres (187.5) is well inside the bottom
        // lobe (155..455) — a split there would hand the top owner a slice
        // of the bottom balloon. The waist split keeps each owner inside
        // its own lobe.
        let midpoint = (top_centre.1 + bottom_centre.1) / 2.0;
        assert!(
            blocks[0].y + blocks[0].height <= neck_y.1 + 2.0,
            "top owner's box reached past the neck into the bottom lobe: {:?}",
            blocks[0]
        );
        assert!(
            blocks[1].y >= neck_y.0 - 2.0,
            "bottom owner's box reached above the neck: {:?}",
            blocks[1]
        );
        assert!(
            blocks[0].y + blocks[0].height < midpoint,
            "top owner's box should end well before the midpoint bisector: {:?}",
            blocks[0]
        );
        // Both still got fitted, not left at their tiny detected size.
        for block in &blocks {
            assert!(block.width > 20.0 && block.height > 20.0, "{block:?}");
            assert!(block.balloon_fitted, "{block:?}");
        }
    }

    #[test]
    fn a_partitioned_mask_yields_a_smaller_rectangle_than_the_whole() {
        let centres = [(70.0f32, 80.0f32), (170.0, 80.0)];
        let mask = merged_balloon_mask(240, 160, &centres, 60.0);
        let owners = vec![(0usize, centres[0]), (1usize, centres[1])];

        let whole = mir_from_mask(&mask, 10.0, 20.0, 220.0, 120.0);
        let half = mir_from_mask_owned(&mask, 10.0, 20.0, 220.0, 120.0, |x, y| {
            nearest_owner(x, y, &owners) == 0
        });

        assert!(
            half[2] > 2.0 && half[3] > 2.0,
            "half should still be usable: {half:?}"
        );
        assert!(
            half[2] < whole[2],
            "half {:?} should be narrower than whole {:?}",
            half,
            whole
        );
        // The left share must stay left of the join.
        assert!(half[0] + half[2] <= 121.0, "{half:?}");
    }

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
    fn a_page_the_detector_fits_is_detected_whole() {
        // Whether an image is one page or two cannot be read off its shape:
        // 890x718 is a crop of a single page and 1489x1200 is a facing pair,
        // and the two have the same aspect ratio. What matters is how far the
        // width is squeezed on the way in.
        for width in [739, 890, 1000] {
            assert_eq!(
                super::detection_tiles(width),
                vec![(0, width)],
                "{width}px fits the detector"
            );
        }
    }

    #[test]
    fn a_page_too_wide_for_the_detector_is_tiled_right_to_left_with_overlap() {
        let tiles = super::detection_tiles(1489);
        assert_eq!(tiles.len(), 2, "a spread needs two passes: {tiles:?}");

        let (right, left) = (tiles[0], tiles[1]);
        assert!(
            right.1 == 1489 && left.0 == 0,
            "tiles cover the page: {tiles:?}"
        );
        assert!(right.0 > left.0, "right to left: {tiles:?}");
        assert!(
            left.1 > right.0,
            "tiles must overlap so text on the boundary is whole in one of \
             them: {tiles:?}"
        );
        for (l, r) in &tiles {
            let scale = 800.0 / f32::from(u16::try_from(r - l).unwrap());
            assert!(
                scale >= 0.75,
                "tile {l}..{r} is still squeezed to {scale:.2}"
            );
        }

        // Wider still, and one more pass is needed.
        assert_eq!(super::detection_tiles(2400).len(), 3);
    }

    #[test]
    fn a_fragment_cut_off_by_a_tile_edge_loses_to_the_whole_box() {
        // The camel balloon from the 1489px spread: the right tile starts at
        // 655 and cuts the text there, the left tile reaches to 833 and has it
        // whole. Without this the 24px fragment survives and OCR reads one
        // column of four.
        let fragment = test_region(0, "text", [655.0, 872.0, 679.0, 998.0]);
        let whole = test_region(1, "text", [624.0, 866.0, 680.0, 997.0]);

        let kept = super::drop_clipped_fragments(vec![(fragment, true), (whole.clone(), false)]);
        assert_eq!(kept.len(), 1, "the fragment must go: {kept:?}");
        assert_eq!(kept[0].bbox, whole.bbox);
    }

    #[test]
    fn a_fragment_no_other_tile_caught_is_kept() {
        // Half a box beats none.
        let fragment = test_region(0, "text", [655.0, 872.0, 679.0, 998.0]);
        let elsewhere = test_region(1, "text", [100.0, 100.0, 200.0, 200.0]);

        let kept = super::drop_clipped_fragments(vec![(fragment, true), (elsewhere, false)]);
        assert_eq!(kept.len(), 2, "nothing covers the fragment: {kept:?}");
    }

    #[test]
    fn only_what_the_pipeline_replaces_is_erased() {
        use super::keep_only_text_found;
        use koharu_types::BalloonDetection;

        let mut mask = image::GrayImage::new(300, 100);
        let mut mark = |x0: u32, x1: u32| {
            for y in 30..60 {
                for x in x0..x1 {
                    mask.put_pixel(x, y, image::Luma([255]));
                }
            }
        };
        mark(20, 50); // inside a detected text box
        mark(120, 150); // inside a balloon, but no box reaches it
        mark(230, 260); // neither — a sound effect on the artwork

        let block = TextBlock {
            x: 15.0,
            y: 25.0,
            width: 40.0,
            height: 40.0,
            ..Default::default()
        };
        let balloon = BalloonDetection {
            x: 110.0,
            y: 20.0,
            width: 60.0,
            height: 60.0,
            score: 0.9,
        };

        let kept = keep_only_text_found(mask, &[block], &[balloon]);
        assert_eq!(kept.get_pixel(30, 40)[0], 255, "boxed text is erased");
        assert_eq!(
            kept.get_pixel(130, 40)[0],
            255,
            "a second column of the same balloon goes too"
        );
        assert_eq!(
            kept.get_pixel(240, 40)[0],
            0,
            "a mark on the artwork is left alone"
        );
    }

    #[test]
    fn a_balloon_does_not_claim_its_own_outline() {
        use super::keep_only_text_found;
        use koharu_types::BalloonDetection;

        // A balloon drawn as a ring, with a word inside it. The ring runs the
        // length of the balloon and reaches the edge of the box around it; the
        // word does not. Claiming everything inside the box takes the outline
        // too, and the balloon comes back cut open and filled with white.
        let mut mask = image::GrayImage::new(160, 160);
        for t in 20..140 {
            for d in 0..3 {
                mask.put_pixel(t, 20 + d, image::Luma([255]));
                mask.put_pixel(t, 137 + d, image::Luma([255]));
                mask.put_pixel(20 + d, t, image::Luma([255]));
                mask.put_pixel(137 + d, t, image::Luma([255]));
            }
        }
        for y in 70..90 {
            for x in 70..90 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        let balloon = BalloonDetection {
            x: 20.0,
            y: 20.0,
            width: 120.0,
            height: 120.0,
            score: 0.9,
        };

        let kept = keep_only_text_found(mask, &[], &[balloon]);
        assert_eq!(kept.get_pixel(80, 80)[0], 255, "the words inside go");
        assert_eq!(kept.get_pixel(60, 21)[0], 0, "the outline stays");
    }

    #[test]
    fn a_run_only_partly_claimed_is_erased_whole() {
        use super::keep_only_text_found;

        // One run reaching well past the box that found it. Erasing the half
        // inside and leaving the rest is worse than either.
        let mut mask = image::GrayImage::new(200, 100);
        for y in 30..60 {
            for x in 20..140 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        let block = TextBlock {
            x: 15.0,
            y: 25.0,
            width: 40.0,
            height: 40.0,
            ..Default::default()
        };

        let kept = keep_only_text_found(mask, &[block], &[]);
        assert_eq!(kept.get_pixel(130, 40)[0], 255, "the tail goes too");
    }

    #[test]
    fn a_balloon_of_marks_keeps_the_marks_the_artist_drew() {
        use super::keep_wordless_marks;

        let mask = image::GrayImage::from_pixel(60, 60, image::Luma([255]));
        let pause = TextBlock {
            x: 10.0,
            y: 10.0,
            width: 20.0,
            height: 20.0,
            text: Some("・・・".to_string()),
            ..Default::default()
        };
        let words = TextBlock {
            x: 35.0,
            y: 10.0,
            width: 20.0,
            height: 20.0,
            text: Some("こわい".to_string()),
            ..Default::default()
        };

        let kept = keep_wordless_marks(mask, &[pause, words]);
        assert_eq!(kept.get_pixel(20, 20)[0], 0, "the pause must be left alone");
        assert_eq!(kept.get_pixel(45, 20)[0], 255, "the words are still erased");
    }

    #[test]
    fn a_word_inside_another_word_does_not_decide_who_someone_is() {
        use super::{Age, Gender, parse_age_gender};

        // The names that used to settle this by accident.
        assert_eq!(
            parse_age_gender("Kinnikuman"),
            (Age::Unknown, Gender::Unknown)
        );
        assert_eq!(parse_age_gender("Goldman"), (Age::Unknown, Gender::Unknown));
        // And what an actual description still says.
        assert_eq!(parse_age_gender("old male"), (Age::Old, Gender::Male));
        assert_eq!(parse_age_gender("young girl"), (Age::Young, Gender::Female));
    }

    #[test]
    fn person_is_not_somebodys_son() {
        use super::family_role;

        // "person" carries "son", "Barbara" carries "ba".
        assert!(family_role("a person").is_none());
        assert!(family_role("Barbara").is_none());
        // A trait that really does say it still lands.
        assert!(family_role("father").is_some());
        assert!(family_role("ông nội").is_some());
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
