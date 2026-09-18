//! Manga Character Relationship Scanner: discovers recurring characters across an
//! entire manga folder from scratch (no pre-registered library required), counts
//! how often each one actually speaks, and builds a co-occurrence relationship
//! graph with LLM-generated edge labels. Output feeds the Curator UI, which lets
//! a human clean up the result before it's trusted (`is_verified_by_human`).
//!
//! This is unsupervised discovery, unlike the rest of `character_library` which
//! matches faces against a human-curated, pre-populated library: characters here
//! start out unnamed and are created on the fly as new faces are encountered.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use super::{
    CharacterLibrary, FaceBox, HIGH_CONF_THRESHOLD, MATCH_MARGIN, MATCH_THRESHOLD,
    classify_gender_age, cosine_similarity, crop_face, crop_face_expanded, locate_speaker_faces,
};

// ─── Tuning constants ──────────────────────────────────────────────────────────

/// Rule: a discovered character must speak in MORE than this many dialogue
/// blocks (across the whole folder) to survive `finalize_scan`'s filter.
const MIN_OCCURRENCES_WITH_DIALOGUE: u32 = 10;
/// Minimum shared-panel count for two kept characters to get a relationship edge.
const MIN_CO_OCCURRENCE_FOR_EDGE: u32 = 3;
/// Maximum face crops retained per character; further matches only update the
/// identity centroid, they don't add another file (see `MAX_FACES_PER_CHARACTER`
/// and `FACE_DEDUP_SIMILARITY`).
const MAX_FACES_PER_CHARACTER: usize = 40;
/// A new face crop is skipped as a near-duplicate when its cosine similarity to
/// an already-stored crop's embedding is at or above this value.
const FACE_DEDUP_SIMILARITY: f32 = 0.97;
/// Max dialogue snippets retained per character, sampled later for LLM prompts.
const MAX_DIALOGUE_SNIPPETS: usize = 30;
/// Max snippets sent per character when asking the LLM to label a relationship.
const MAX_SNIPPETS_FOR_LLM: usize = 8;

// ─── Checkpoint / discovery-state types ────────────────────────────────────────

/// A character discovered mid-scan, before the `occurrences_with_dialogue > 10`
/// filter and relationship-labeling pass in `finalize_scan`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvisionalCharacter {
    pub id: String,
    /// Running L2-normalised mean of every matched face embedding.
    pub centroid_embedding: Vec<f32>,
    /// Per-face embeddings for the crops in `face_crop_paths` (parallel array),
    /// kept so later matches can be dedup-checked without re-running CCIP.
    pub face_embeddings: Vec<Vec<f32>>,
    pub occurrences_with_dialogue: u32,
    /// Paths relative to the scan's output directory (e.g. `faces/<id>/0.png`).
    pub face_crop_paths: Vec<PathBuf>,
    pub dialogue_snippets: Vec<String>,
    /// Other provisional character id -> number of panels they've shared.
    pub co_occurrence: HashMap<String, u32>,
    pub gender: Option<String>,
    pub age_group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ScanCheckpoint {
    /// Index into the folder's file list of the next image to process.
    pub last_processed_index: usize,
    pub characters: Vec<ProvisionalCharacter>,
}

// ─── Final export types (manga_relationship_v1.json) ──────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScannedCharacter {
    pub id: String,
    pub name: String,
    pub gender: Option<String>,
    pub age_group: Option<String>,
    pub faces: Vec<PathBuf>,
    #[serde(default)]
    pub traits: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipEdge {
    pub character_id: String,
    /// Number of panels the two characters share — always filled in by the
    /// rule-based scan, independent of whether an LLM label exists yet.
    pub co_occurrence: u32,
    /// `None` until the curator clicks "Generate by LLM" (`generate_relationship_labels`)
    /// — the scan itself never calls the LLM.
    pub label: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipNode {
    pub character_id: String,
    pub related: Vec<RelationshipEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanResult {
    pub characters: Vec<ScannedCharacter>,
    pub relationship_tree: Vec<RelationshipNode>,
    pub is_verified_by_human: bool,
}

// ─── Checkpoint I/O ─────────────────────────────────────────────────────────────

fn checkpoint_path(out_dir: &Path) -> PathBuf {
    out_dir.join("checkpoint.json")
}

fn scan_result_path(out_dir: &Path) -> PathBuf {
    out_dir.join("manga_relationship_v1.json")
}

/// Atomically write `checkpoint.json` (write to a temp file, then rename) so a
/// crash mid-write never leaves a corrupt checkpoint behind.
pub fn write_checkpoint(out_dir: &Path, checkpoint: &ScanCheckpoint) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let tmp = out_dir.join("checkpoint.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(checkpoint)?)?;
    std::fs::rename(&tmp, checkpoint_path(out_dir))?;
    Ok(())
}

/// Read `checkpoint.json` if present, to resume an interrupted scan.
pub fn read_checkpoint(out_dir: &Path) -> Option<ScanCheckpoint> {
    let json = std::fs::read_to_string(checkpoint_path(out_dir)).ok()?;
    serde_json::from_str(&json).ok()
}

/// Write the final `manga_relationship_v1.json`.
pub fn write_scan_result(out_dir: &Path, result: &ScanResult) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    std::fs::write(
        scan_result_path(out_dir),
        serde_json::to_string_pretty(result)?,
    )?;
    Ok(())
}

/// Read an existing `manga_relationship_v1.json`, e.g. to load a prior run for
/// curation without rescanning.
pub fn read_scan_result(out_dir: &Path) -> Option<ScanResult> {
    let json = std::fs::read_to_string(scan_result_path(out_dir)).ok()?;
    serde_json::from_str(&json).ok()
}

// ─── Per-image scan step ────────────────────────────────────────────────────────

/// Scan one image: detect dialogue + panels + faces, then match each
/// dialogue-linked face against `characters` (creating a new provisional
/// character when nothing matches), and record co-occurrence between characters
/// sharing a panel. Mutates `characters` in place; face crops are written under
/// `out_dir/faces/<character_id>/`.
///
/// Runs its own detect/OCR/balloon-detect pass on an ephemeral `Document` — it
/// does not require the image to have gone through the normal translate
/// pipeline, and never registers the document in app state.
pub async fn scan_one_image(
    model: &crate::facade::Model,
    path: &Path,
    out_dir: &Path,
    characters: &mut Vec<ProvisionalCharacter>,
) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let mut doc = koharu_types::Document::from_bytes(path.to_path_buf(), bytes)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Failed to parse: {}", path.display()))?;

    model.detect(&mut doc).await?;
    if !doc.text_blocks.is_empty() {
        model.ocr(&mut doc).await?;
    }
    model.detect_balloons(&mut doc).await?;
    doc.ensure_text_block_ids();

    let character_lib: &CharacterLibrary = &model.character_lib;
    let (Some(face_det), Some(ccip)) =
        (character_lib.face_det.as_ref(), character_lib.ccip.as_ref())
    else {
        tracing::warn!(path = %path.display(), "character scanner: face detector/CCIP not loaded, skipping image");
        return Ok(());
    };

    let panels = character_lib.detect_panels(&doc.image);

    // Only blocks with actual OCR'd dialogue count toward "occurrences with dialogue".
    let dialogue_by_id: HashMap<String, String> = doc
        .text_blocks
        .iter()
        .filter_map(|b| {
            b.text
                .as_ref()
                .filter(|t| !t.trim().is_empty())
                .map(|t| (b.id.clone(), t.clone()))
        })
        .collect();

    let blocks: Vec<(String, f32, f32, f32, f32)> = doc
        .text_blocks
        .iter()
        .filter(|b| dialogue_by_id.contains_key(&b.id))
        .map(|b| (b.id.clone(), b.x, b.y, b.width, b.height))
        .collect();

    if blocks.is_empty() {
        return Ok(());
    }

    let balloons: Vec<(f32, f32, f32, f32)> = doc
        .balloons
        .iter()
        .map(|b| (b.x, b.y, b.width, b.height))
        .collect();

    let (geoms, face_data, face_panel_idx) =
        locate_speaker_faces(face_det, ccip, &doc.image, &blocks, &balloons, &panels);

    if face_data.is_empty() {
        return Ok(());
    }

    // ── Step A: dialogue-linked occurrence + face harvesting ───────────────────
    for g in &geoms {
        let Some(face_idx) = g.face_idx else { continue };
        let Some(text) = dialogue_by_id.get(&g.id) else {
            continue;
        };
        let (face_box, _, _, embedding) = &face_data[face_idx];

        match match_provisional(characters, embedding) {
            Some(ci) => update_character_with_face(
                &mut characters[ci],
                embedding,
                text,
                face_box,
                &doc.image,
                out_dir,
            )?,
            None => {
                let new_character = create_provisional_character(
                    character_lib,
                    embedding,
                    text,
                    face_box,
                    &doc.image,
                    out_dir,
                )?;
                characters.push(new_character);
            }
        }
    }

    // ── Step B: co-occurrence via per-panel face roster ─────────────────────────
    // Read-only matches — a face that doesn't already match a discovered
    // character never creates one here; only dialogue does (Step A).
    let mut panel_members: HashMap<usize, std::collections::HashSet<String>> = HashMap::new();
    for (i, panel_idx) in face_panel_idx.iter().enumerate() {
        let Some(pidx) = panel_idx else { continue };
        let (_, _, _, embedding) = &face_data[i];
        if let Some(ci) = match_provisional(characters, embedding) {
            panel_members
                .entry(*pidx)
                .or_default()
                .insert(characters[ci].id.clone());
        }
    }
    for members in panel_members.values() {
        for a in members {
            for b in members {
                if a != b
                    && let Some(ca) = characters.iter_mut().find(|c| &c.id == a)
                {
                    *ca.co_occurrence.entry(b.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    Ok(())
}

/// Match `embedding` against discovered characters using the same
/// threshold/margin gating as `find_best_match` (best/second-best margin check,
/// bypassed at very high similarity). Returns `None` when nothing matches
/// closely enough — the caller should create a new character in that case.
fn match_provisional(characters: &[ProvisionalCharacter], embedding: &[f32]) -> Option<usize> {
    let mut best_idx: Option<usize> = None;
    let mut best_sim = -1.0f32;
    let mut second_sim = -1.0f32;

    for (i, c) in characters.iter().enumerate() {
        let sim = cosine_similarity(&c.centroid_embedding, embedding);
        if sim > best_sim {
            second_sim = best_sim;
            best_sim = sim;
            best_idx = Some(i);
        } else if sim > second_sim {
            second_sim = sim;
        }
    }

    let margin_ok = characters.len() < 2
        || best_sim >= HIGH_CONF_THRESHOLD
        || (best_sim - second_sim) >= MATCH_MARGIN;

    if best_sim >= MATCH_THRESHOLD && margin_ok {
        best_idx
    } else {
        None
    }
}

fn create_provisional_character(
    character_lib: &CharacterLibrary,
    embedding: &[f32],
    dialogue: &str,
    face_box: &FaceBox,
    image: &DynamicImage,
    out_dir: &Path,
) -> Result<ProvisionalCharacter> {
    let id = uuid::Uuid::new_v4().to_string();
    let (gender, age_group) = classify_face(character_lib, image, face_box);
    let crop_path = save_face_crop(out_dir, &id, 0, image, face_box)?;

    Ok(ProvisionalCharacter {
        id,
        centroid_embedding: embedding.to_vec(),
        face_embeddings: vec![embedding.to_vec()],
        occurrences_with_dialogue: 1,
        face_crop_paths: vec![crop_path],
        dialogue_snippets: vec![dialogue.to_string()],
        co_occurrence: HashMap::new(),
        gender,
        age_group,
    })
}

fn update_character_with_face(
    character: &mut ProvisionalCharacter,
    embedding: &[f32],
    dialogue: &str,
    face_box: &FaceBox,
    image: &DynamicImage,
    out_dir: &Path,
) -> Result<()> {
    character.occurrences_with_dialogue += 1;

    if character.dialogue_snippets.len() < MAX_DIALOGUE_SNIPPETS {
        character.dialogue_snippets.push(dialogue.to_string());
    }

    let is_near_duplicate = character
        .face_embeddings
        .iter()
        .any(|e| cosine_similarity(e, embedding) >= FACE_DEDUP_SIMILARITY);
    if !is_near_duplicate && character.face_crop_paths.len() < MAX_FACES_PER_CHARACTER {
        let index = character.face_crop_paths.len();
        let path = save_face_crop(out_dir, &character.id, index, image, face_box)?;
        character.face_crop_paths.push(path);
        character.face_embeddings.push(embedding.to_vec());
    }

    // Incremental running-mean centroid update, re-normalised. `n` is the new
    // (post-increment) occurrence count, since every dialogue match — saved
    // crop or not — contributes its embedding to the identity centroid.
    let n = character.occurrences_with_dialogue as f32;
    let mut mean: Vec<f32> = character
        .centroid_embedding
        .iter()
        .zip(embedding.iter())
        .map(|(c, e)| (c * (n - 1.0) + e) / n)
        .collect();
    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for m in mean.iter_mut() {
        *m /= norm;
    }
    character.centroid_embedding = mean;

    Ok(())
}

/// Classify gender/age from the face's body-context crop via WD Tagger, if
/// loaded. Returns `(None, None)` when the tagger isn't available.
fn classify_face(
    character_lib: &CharacterLibrary,
    image: &DynamicImage,
    face_box: &FaceBox,
) -> (Option<String>, Option<String>) {
    let Some(wd_tagger) = character_lib.wd_tagger.as_ref() else {
        return (None, None);
    };
    let crop = crop_face_expanded(image, face_box);
    match classify_gender_age(wd_tagger, &crop) {
        Ok((gender, age_group)) => (Some(gender), Some(age_group)),
        Err(e) => {
            tracing::warn!(error = %e, "character scanner: gender/age classification failed");
            (None, None)
        }
    }
}

/// Save a tight face crop and return its path relative to `out_dir`
/// (e.g. `faces/<character_id>/<index>.png`) for storage in JSON.
fn save_face_crop(
    out_dir: &Path,
    character_id: &str,
    index: usize,
    image: &DynamicImage,
    face_box: &FaceBox,
) -> Result<PathBuf> {
    let crop = crop_face(image, face_box);
    let dir = out_dir.join("faces").join(character_id);
    std::fs::create_dir_all(&dir)?;
    let filename = format!("{index}.png");
    crop.save(dir.join(&filename))?;
    Ok(PathBuf::from("faces").join(character_id).join(filename))
}

// ─── Finalize: rule-based filter + co-occurrence tree ──────────────────────────

/// Apply the `occurrences_with_dialogue > 10` rule and build the co-occurrence
/// relationship tree — plain code logic only, no LLM call. Edges start with
/// `label`/`description` unset; the curator opts into LLM labeling afterward via
/// `generate_relationship_labels` (the "Generate by LLM" button).
pub fn finalize_scan(characters: Vec<ProvisionalCharacter>) -> ScanResult {
    let kept: Vec<ProvisionalCharacter> = characters
        .into_iter()
        .filter(|c| c.occurrences_with_dialogue > MIN_OCCURRENCES_WITH_DIALOGUE)
        .collect();

    let names: HashMap<String, String> = kept
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id.clone(), format!("Character {}", i + 1)))
        .collect();

    let mut related: HashMap<String, Vec<RelationshipEdge>> = HashMap::new();

    for (i, a) in kept.iter().enumerate() {
        for b in kept.iter().skip(i + 1) {
            let shared = a.co_occurrence.get(&b.id).copied().unwrap_or(0);
            if shared < MIN_CO_OCCURRENCE_FOR_EDGE {
                continue;
            }

            related
                .entry(a.id.clone())
                .or_default()
                .push(RelationshipEdge {
                    character_id: b.id.clone(),
                    co_occurrence: shared,
                    label: None,
                    description: None,
                });
            related
                .entry(b.id.clone())
                .or_default()
                .push(RelationshipEdge {
                    character_id: a.id.clone(),
                    co_occurrence: shared,
                    label: None,
                    description: None,
                });
        }
    }

    let relationship_tree = kept
        .iter()
        .map(|c| RelationshipNode {
            character_id: c.id.clone(),
            related: related.remove(&c.id).unwrap_or_default(),
        })
        .collect();

    let scanned_characters = kept
        .into_iter()
        .map(|c| ScannedCharacter {
            name: names.get(&c.id).cloned().unwrap_or_else(|| c.id.clone()),
            id: c.id,
            gender: c.gender,
            age_group: c.age_group,
            faces: c.face_crop_paths,
            traits: Vec::new(),
        })
        .collect();

    ScanResult {
        characters: scanned_characters,
        relationship_tree,
        is_verified_by_human: false,
    }
}

// ─── Generate by LLM (opt-in, curator-triggered) ───────────────────────────────

/// Fill in an LLM-generated label + description for every relationship edge that
/// doesn't have one yet ("Generate by LLM" button). `dialogue_by_id` maps
/// character id -> sampled dialogue lines — since `ScannedCharacter` doesn't keep
/// dialogue text, callers should read it back from `checkpoint.json`
/// (`ProvisionalCharacter::dialogue_snippets`), which is never deleted after a
/// scan completes.
///
/// Errors (no API LLM loaded) are returned to the caller instead of silently
/// falling back — the curator clicked this button expecting real output, unlike
/// the automatic scan which never calls the LLM at all. A single bad/unparseable
/// response for one pair still only falls back for that pair, though.
pub async fn generate_relationship_labels(
    result: &mut ScanResult,
    dialogue_by_id: &HashMap<String, Vec<String>>,
    llm: &koharu_llm::facade::Model,
) -> Result<()> {
    if !llm.is_api_ready().await {
        anyhow::bail!(
            "Generating relationship labels requires an API LLM provider to be loaded (local models aren't supported yet)"
        );
    }

    let names: HashMap<String, String> = result
        .characters
        .iter()
        .map(|c| (c.id.clone(), c.name.clone()))
        .collect();
    let empty: Vec<String> = Vec::new();

    // Label each unordered pair once, then apply the result to both directions.
    let mut labeled: HashMap<(String, String), (String, String)> = HashMap::new();
    for node in &result.relationship_tree {
        for edge in &node.related {
            if edge.label.is_some() {
                continue;
            }
            let key = pair_key(&node.character_id, &edge.character_id);
            if labeled.contains_key(&key) {
                continue;
            }

            let a_snippets = dialogue_by_id.get(&node.character_id).unwrap_or(&empty);
            let b_snippets = dialogue_by_id.get(&edge.character_id).unwrap_or(&empty);
            let a_name = names
                .get(&node.character_id)
                .map(String::as_str)
                .unwrap_or("A");
            let b_name = names
                .get(&edge.character_id)
                .map(String::as_str)
                .unwrap_or("B");

            let labeled_pair =
                label_relationship(llm, a_name, a_snippets, b_name, b_snippets).await;
            labeled.insert(key, labeled_pair);
        }
    }

    for node in &mut result.relationship_tree {
        for edge in &mut node.related {
            if edge.label.is_some() {
                continue;
            }
            if let Some((label, description)) =
                labeled.get(&pair_key(&node.character_id, &edge.character_id))
            {
                edge.label = Some(label.clone());
                edge.description = Some(description.clone());
            }
        }
    }

    Ok(())
}

fn pair_key(a: &str, b: &str) -> (String, String) {
    if a < b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

async fn label_relationship(
    llm: &koharu_llm::facade::Model,
    a_name: &str,
    a_snippets: &[String],
    b_name: &str,
    b_snippets: &[String],
) -> (String, String) {
    const SYSTEM_PROMPT: &str = "You are analyzing dialogue sampled from a manga to infer the \
        relationship between two recurring characters. Respond with ONLY a JSON object of the \
        form {\"label\": \"<short relationship label, 1-4 words>\", \"description\": \"<one \
        sentence description>\"} — no other text, no markdown code fences.";

    let user_prompt = format!(
        "Character A ({a_name}) sample lines:\n{}\n\nCharacter B ({b_name}) sample lines:\n{}\n\n\
         What is the relationship between Character A and Character B?",
        sample_snippets(a_snippets),
        sample_snippets(b_snippets),
    );

    match llm.complete(SYSTEM_PROMPT, &user_prompt).await {
        Ok(raw) => {
            parse_relationship_response(&raw).unwrap_or_else(|| (fallback_label(), String::new()))
        }
        Err(e) => {
            tracing::warn!(error = %e, "character scanner: relationship labeling failed, using fallback");
            (fallback_label(), String::new())
        }
    }
}

fn fallback_label() -> String {
    "Related".to_string()
}

fn sample_snippets(snippets: &[String]) -> String {
    snippets
        .iter()
        .rev()
        .take(MAX_SNIPPETS_FOR_LLM)
        .map(|s| format!("- {s}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract the first `{...}` JSON object from `raw` (LLMs sometimes wrap the
/// object in prose or code fences despite instructions) and parse `label` +
/// `description` out of it.
fn parse_relationship_response(raw: &str) -> Option<(String, String)> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(&raw[start..=end]).ok()?;
    let label = value.get("label")?.as_str()?.trim().to_string();
    if label.is_empty() {
        return None;
    }
    let description = value
        .get("description")
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    Some((label, description))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_provisional_creates_new_when_list_empty() {
        assert_eq!(match_provisional(&[], &[1.0, 0.0]), None);
    }

    #[test]
    fn parse_relationship_response_extracts_json_from_prose() {
        let raw = "Sure! Here you go:\n{\"label\": \"Siblings\", \"description\": \"They bicker constantly.\"}\nHope that helps.";
        let (label, description) = parse_relationship_response(raw).unwrap();
        assert_eq!(label, "Siblings");
        assert_eq!(description, "They bicker constantly.");
    }

    #[test]
    fn parse_relationship_response_rejects_empty_label() {
        assert!(parse_relationship_response("{\"label\": \"\", \"description\": \"x\"}").is_none());
    }

    #[test]
    fn parse_relationship_response_rejects_non_json() {
        assert!(parse_relationship_response("no json here").is_none());
    }
}
