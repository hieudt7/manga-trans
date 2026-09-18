//! Orchestrates the Manga Character Relationship Scanner as a background job:
//! iterates every image in the currently-open folder session, running
//! `koharu_ml::character_library::scanner::scan_one_image` on each and
//! checkpointing after every image so an interrupted scan can resume. Mirrors
//! `folder.rs`'s job-lifecycle/progress conventions (`PipelineHandle`,
//! `pipeline::emit_progress`) — see that module for the reference pattern.
//!
//! Unlike `folder.rs`'s translate pipeline (which only lists images directly
//! inside the chosen folder), the scanner walks subfolders too — manga is
//! commonly organised as `<series>/<volume>/<page>.jpg`, and a folder full of
//! per-volume subfolders would otherwise scan as empty.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
};

use koharu_ml::character_library::scanner::{
    self, ProvisionalCharacter, ScanCheckpoint, ScanResult,
};
use koharu_types::{
    FolderSession,
    events::{PipelineProgress, PipelineStatus},
};
use tracing::instrument;

use crate::{AppResources, pipeline};

const JOB_KIND: &str = "character-scan-folder";
const OUT_DIR_NAME: &str = "character_scan";
/// Mirrors `folder.rs::RESULT_DIR_NAME` (private there) — skipped during the
/// recursive walk so a folder that's already been translated doesn't have its
/// rendered output re-scanned as new manga pages.
const TRANSLATE_RESULT_DIR_NAME: &str = "result";
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff"];

/// Recursively collect every image file under `root`, sorted so pages within a
/// volume stay in order and volumes stay grouped together. Skips this
/// scanner's own output dir and the translate pipeline's `result/` dir so a
/// re-scan never picks up previously-generated face crops or rendered pages.
fn collect_image_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk_images(root, &mut files);
    files.sort();
    files
}

fn walk_images(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == OUT_DIR_NAME || name == TRANSLATE_RESULT_DIR_NAME || name.starts_with('.') {
                continue;
            }
            walk_images(&path, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

/// Start a character-scan job over the currently-open folder session (walked
/// recursively — see `collect_image_files`). Returns the job id and total image
/// count immediately; progress streams via the same `PipelineProgress`/SSE
/// channel as the translate pipeline (`kind` distinguishes them — see the
/// Phase 0 fix in `koharu-types/src/events.rs`).
#[instrument(level = "info", skip_all)]
pub async fn start_character_scan_job(resources: AppResources) -> anyhow::Result<(String, usize)> {
    {
        let guard = resources.pipeline.read().await;
        if guard.is_some() {
            anyhow::bail!("A pipeline job is already running");
        }
    }

    let root = {
        let guard = resources.state.read().await;
        require_session(&guard)?.root.clone()
    };

    let files = {
        let root = root.clone();
        tokio::task::spawn_blocking(move || collect_image_files(&root)).await?
    };
    if files.is_empty() {
        anyhow::bail!(
            "No image files found in {} (including subfolders)",
            root.display()
        );
    }
    let total_files = files.len();

    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let job_id = uuid::Uuid::new_v4().to_string();

    {
        let mut guard = resources.pipeline.write().await;
        *guard = Some(pipeline::PipelineHandle {
            id: job_id.clone(),
            cancel: cancel.clone(),
        });
    }

    let res = resources.clone();
    let jid = job_id.clone();
    tokio::spawn(async move {
        run_character_scan(res, cancel, jid, root, files).await;
    });

    Ok((job_id, total_files))
}

/// Load a previously-written `manga_relationship_v1.json` from the currently-open
/// folder session, without starting a new scan.
pub async fn get_character_scan_result(
    resources: AppResources,
) -> anyhow::Result<Option<ScanResult>> {
    let root = {
        let guard = resources.state.read().await;
        require_session(&guard)?.root.clone()
    };
    Ok(scanner::read_scan_result(&out_dir(&root)))
}

/// Overwrite `manga_relationship_v1.json` with curator edits, marking it
/// human-verified.
pub async fn export_character_scan_result(
    resources: AppResources,
    mut result: ScanResult,
) -> anyhow::Result<()> {
    let root = {
        let guard = resources.state.read().await;
        require_session(&guard)?.root.clone()
    };
    result.is_verified_by_human = true;
    scanner::write_scan_result(&out_dir(&root), &result)?;
    Ok(())
}

/// Curator "Generate by LLM": label every relationship edge that doesn't have
/// one yet, using the app's currently-loaded LLM (must be an API provider — see
/// `Model::is_api_ready`). Reads dialogue snippets back from `checkpoint.json`
/// (kept on disk after the scan completes) since the clean `ScanResult` doesn't
/// carry raw dialogue text. Persists the updated result to
/// `manga_relationship_v1.json` (still `isVerifiedByHuman: false` — that flag is
/// only set by the human's own Export action) and returns it.
pub async fn generate_character_scan_relationships(
    resources: AppResources,
) -> anyhow::Result<ScanResult> {
    let root = {
        let guard = resources.state.read().await;
        require_session(&guard)?.root.clone()
    };
    let dir = out_dir(&root);

    let mut result = scanner::read_scan_result(&dir)
        .ok_or_else(|| anyhow::anyhow!("No character scan result found — run a scan first"))?;
    let checkpoint = scanner::read_checkpoint(&dir).ok_or_else(|| {
        anyhow::anyhow!("Scan checkpoint not found — dialogue data is unavailable")
    })?;

    let dialogue_by_id: std::collections::HashMap<String, Vec<String>> = checkpoint
        .characters
        .into_iter()
        .map(|c| (c.id, c.dialogue_snippets))
        .collect();

    scanner::generate_relationship_labels(&mut result, &dialogue_by_id, &resources.llm).await?;
    scanner::write_scan_result(&dir, &result)?;

    Ok(result)
}

/// Absolute path to a face crop written by the scanner, for serving over HTTP.
pub async fn get_character_scan_face_path(
    resources: AppResources,
    character_id: &str,
    file: &str,
) -> anyhow::Result<PathBuf> {
    let root = {
        let guard = resources.state.read().await;
        require_session(&guard)?.root.clone()
    };
    let path = out_dir(&root).join("faces").join(character_id).join(file);
    if !path.is_file() {
        anyhow::bail!("Face crop not found: {}", path.display());
    }
    Ok(path)
}

/// Curator "+ Add Face": save a human-supplied image into a character's face
/// folder and return its path relative to the scan's output directory (the same
/// shape as the paths the scanner itself writes), for the frontend to append to
/// `ScannedCharacter.faces`.
pub async fn add_character_scan_face(
    resources: AppResources,
    character_id: &str,
    bytes: Vec<u8>,
) -> anyhow::Result<PathBuf> {
    let root = {
        let guard = resources.state.read().await;
        require_session(&guard)?.root.clone()
    };
    let dir = out_dir(&root).join("faces").join(character_id);
    tokio::fs::create_dir_all(&dir).await?;

    let mut index = 0usize;
    while dir.join(format!("{index}.png")).exists() {
        index += 1;
    }

    let image = image::load_from_memory(&bytes)?;
    let filename = format!("{index}.png");
    image.save(dir.join(&filename))?;

    Ok(PathBuf::from("faces").join(character_id).join(filename))
}

fn out_dir(root: &std::path::Path) -> PathBuf {
    root.join(OUT_DIR_NAME)
}

fn require_session(guard: &koharu_types::State) -> anyhow::Result<&FolderSession> {
    guard
        .folder_session
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No folder session active"))
}

// ─── internal ────────────────────────────────────────────────────────────────

async fn run_character_scan(
    resources: AppResources,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    job_id: String,
    root: PathBuf,
    files: Vec<PathBuf>,
) {
    let total_files = files.len();

    match run_character_scan_inner(&resources, &cancel, &job_id, &root, &files).await {
        Ok(()) if cancel.load(Ordering::Relaxed) => {
            pipeline::emit_progress(PipelineProgress {
                job_id: job_id.clone(),
                kind: JOB_KIND.to_string(),
                status: PipelineStatus::Cancelled,
                step: None,
                current_document: total_files,
                total_documents: total_files,
                current_step_index: 0,
                total_steps: 1,
                overall_percent: 0,
            });
        }
        Ok(()) => {
            pipeline::emit_progress(PipelineProgress {
                job_id: job_id.clone(),
                kind: JOB_KIND.to_string(),
                status: PipelineStatus::Completed,
                step: None,
                current_document: total_files,
                total_documents: total_files,
                current_step_index: 1,
                total_steps: 1,
                overall_percent: 100,
            });
        }
        Err(err) => {
            tracing::error!("Character scan failed: {err:#}");
            pipeline::emit_progress(PipelineProgress {
                job_id: job_id.clone(),
                kind: JOB_KIND.to_string(),
                status: PipelineStatus::Failed(err.to_string()),
                step: None,
                current_document: 0,
                total_documents: total_files,
                current_step_index: 0,
                total_steps: 1,
                overall_percent: 0,
            });
        }
    }

    let mut guard = resources.pipeline.write().await;
    *guard = None;
}

async fn run_character_scan_inner(
    res: &AppResources,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    job_id: &str,
    root: &Path,
    files: &[PathBuf],
) -> anyhow::Result<()> {
    let total_docs = files.len();
    if total_docs == 0 {
        return Ok(());
    }

    let dir = out_dir(root);
    let checkpoint = scanner::read_checkpoint(&dir);
    let (mut characters, start_index): (Vec<ProvisionalCharacter>, usize) = match checkpoint {
        Some(cp) => {
            tracing::info!(
                resume_from = cp.last_processed_index,
                characters_so_far = cp.characters.len(),
                "character scan: resuming from checkpoint"
            );
            (cp.characters, cp.last_processed_index)
        }
        None => (Vec::new(), 0),
    };

    for (idx, path) in files.iter().enumerate().skip(start_index) {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }

        pipeline::emit_progress(PipelineProgress {
            job_id: job_id.to_string(),
            kind: JOB_KIND.to_string(),
            status: PipelineStatus::Running,
            step: None,
            current_document: idx,
            total_documents: total_docs,
            current_step_index: 0,
            total_steps: 1,
            overall_percent: percent(idx, total_docs),
        });

        tokio::task::yield_now().await;

        if let Err(err) = scanner::scan_one_image(&res.ml, path, &dir, &mut characters).await {
            tracing::warn!(file = %path.display(), "character scan: skipping page due to error: {err:#}");
        }

        scanner::write_checkpoint(
            &dir,
            &ScanCheckpoint {
                last_processed_index: idx + 1,
                characters: characters.clone(),
            },
        )?;
    }

    if cancel.load(Ordering::Relaxed) {
        return Ok(());
    }

    let result = scanner::finalize_scan(characters);
    scanner::write_scan_result(&dir, &result)?;

    Ok(())
}

fn percent(doc: usize, total_docs: usize) -> u8 {
    if total_docs == 0 {
        return 0;
    }
    ((doc as f64 / total_docs as f64) * 100.0).round() as u8
}
