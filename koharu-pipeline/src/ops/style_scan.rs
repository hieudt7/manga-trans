//! Learns how a published translation reads, as a background job.
//!
//! The open folder is a reference volume: the artist's pages in `raw/`, the
//! published Vietnamese in `trans/`. The job lines the two up page by page,
//! reads the balloons on both sides of every matched page, and asks the loaded
//! LLM what the resulting sentence pairs say about the translator. Mirrors
//! `character_scan.rs` for job lifecycle, progress and checkpointing.
//!
//! The profile is not applied to anything until the curator chooses it
//! (`set_active_style_profile`): the reference volume is rarely the folder being
//! translated, so the profile is copied to one app-wide place that `llm_load`
//! reads.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
};

use koharu_ml::bilingual::{
    self, PagePairing, PageSignature,
    corpus::{self, StyleScanCheckpoint, StyleScanResult},
    style::{self, StyleProfile},
};
use koharu_types::{
    Document, FolderSession,
    events::{PipelineProgress, PipelineStatus},
};
use tracing::instrument;

use crate::{AppResources, pipeline};

const JOB_KIND: &str = "style-scan-folder";

/// Start a style scan over the open folder session. Returns the job id and the
/// number of steps progress will count through.
#[instrument(level = "info", skip_all)]
pub async fn start_style_scan_job(resources: AppResources) -> anyhow::Result<(String, usize)> {
    {
        let guard = resources.pipeline.read().await;
        if guard.is_some() {
            anyhow::bail!("A pipeline job is already running");
        }
    }

    let root = session_root(&resources).await?;
    let raw = corpus::list_pages(&root.join(corpus::RAW_DIR));
    let translated = corpus::list_pages(&root.join(corpus::TRANSLATED_DIR));
    if raw.is_empty() || translated.is_empty() {
        anyhow::bail!(
            "A style scan needs the artist's pages in {}/ and the published translation in {}/ \
             inside {} ({} and {} images found)",
            corpus::RAW_DIR,
            corpus::TRANSLATED_DIR,
            root.display(),
            raw.len(),
            translated.len()
        );
    }

    // Both of these are needed only at the far end of a long read. Finding out
    // they are missing after an hour of OCR would waste the hour.
    corpus::open_reader()?;
    if !resources.llm.is_api_ready().await {
        anyhow::bail!(
            "Load an API LLM provider first — it reads the sentence pairs at the end of the scan"
        );
    }

    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let job_id = uuid::Uuid::new_v4().to_string();
    {
        let mut guard = resources.pipeline.write().await;
        *guard = Some(pipeline::PipelineHandle {
            id: job_id.clone(),
            cancel: cancel.clone(),
        });
    }

    // Aligning, then one step per raw page at most, then learning.
    let total = raw.len() + 2;
    let res = resources.clone();
    let jid = job_id.clone();
    tokio::spawn(async move {
        run_style_scan(res, cancel, jid, root, raw, translated).await;
    });

    Ok((job_id, total))
}

/// The last finished scan of the open folder, if any.
pub async fn get_style_scan_result(
    resources: AppResources,
) -> anyhow::Result<Option<StyleScanResult>> {
    let root = session_root(&resources).await?;
    Ok(corpus::read_result(&corpus::out_dir(&root)))
}

/// Save the curator's edits to the open folder's profile, marking it reviewed.
pub async fn export_style_scan_result(
    resources: AppResources,
    mut result: StyleScanResult,
) -> anyhow::Result<()> {
    let root = session_root(&resources).await?;
    result.is_verified_by_human = true;
    corpus::write_result(&corpus::out_dir(&root), &result)
}

/// The profile translations currently follow, if any.
pub async fn get_active_style_profile() -> anyhow::Result<Option<StyleProfile>> {
    Ok(style::load_active())
}

/// Make `profile` the one translations follow, or stop following any.
///
/// Takes effect the next time an LLM is loaded: the profile rides in the story
/// context, which is fixed for a loaded session so the provider can cache it.
pub async fn set_active_style_profile(profile: Option<StyleProfile>) -> anyhow::Result<()> {
    style::set_active(profile.as_ref())
}

async fn session_root(resources: &AppResources) -> anyhow::Result<PathBuf> {
    let guard = resources.state.read().await;
    Ok(require_session(&guard)?.root.clone())
}

fn require_session(guard: &koharu_types::State) -> anyhow::Result<&FolderSession> {
    guard
        .folder_session
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No folder session active"))
}

// ─── internal ────────────────────────────────────────────────────────────────

async fn run_style_scan(
    resources: AppResources,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    job_id: String,
    root: PathBuf,
    raw: Vec<PathBuf>,
    translated: Vec<PathBuf>,
) {
    let total = raw.len() + 2;
    let outcome =
        run_style_scan_inner(&resources, &cancel, &job_id, &root, &raw, &translated).await;

    let status = match outcome {
        Ok(()) if cancel.load(Ordering::Relaxed) => PipelineStatus::Cancelled,
        Ok(()) => PipelineStatus::Completed,
        Err(err) => {
            tracing::error!("Style scan failed: {err:#}");
            PipelineStatus::Failed(err.to_string())
        }
    };
    let done = matches!(status, PipelineStatus::Completed);
    progress(&job_id, status, if done { total } else { 0 }, total);

    let mut guard = resources.pipeline.write().await;
    *guard = None;
}

async fn run_style_scan_inner(
    res: &AppResources,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    job_id: &str,
    root: &Path,
    raw: &[PathBuf],
    translated: &[PathBuf],
) -> anyhow::Result<()> {
    let total = raw.len() + 2;
    let out = corpus::out_dir(root);
    progress(job_id, PipelineStatus::Running, 0, total);

    // Aligning costs no OCR — the artwork is the same on both sides — so it is
    // simply redone on a resume rather than checkpointed.
    let steps = {
        let (raw, translated) = (raw.to_vec(), translated.to_vec());
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<PagePairing>> {
            let sign = |files: &[PathBuf]| -> anyhow::Result<Vec<PageSignature>> {
                files
                    .iter()
                    .map(|path| Ok(PageSignature::new(&image::open(path)?)))
                    .collect()
            };
            Ok(bilingual::align_pages(&sign(&raw)?, &sign(&translated)?))
        })
        .await??
    };
    let matched = steps
        .iter()
        .filter(|step| matches!(step, PagePairing::Matched { .. }))
        .count();
    tracing::info!(
        raw = raw.len(),
        translated = translated.len(),
        matched,
        "style scan: pages aligned"
    );

    // A checkpoint belongs to the pairs file written alongside it. Without one,
    // any pairs file is left over from a scan that never recorded its progress,
    // and appending to it would count its pages twice.
    let mut checkpoint = match corpus::read_checkpoint(&out) {
        Some(checkpoint) => {
            if let Some(len) = checkpoint.pairs_bytes {
                corpus::truncate_pairs(&out, len)?;
            }
            checkpoint
        }
        None => {
            let _ = std::fs::remove_file(corpus::pairs_path(&out));
            StyleScanCheckpoint::default()
        }
    };

    let reader = corpus::open_reader()?;
    for (index, step) in steps.iter().enumerate().skip(checkpoint.steps_done) {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        progress(
            job_id,
            PipelineStatus::Running,
            1 + index.min(raw.len()),
            total,
        );
        tokio::task::yield_now().await;

        if let PagePairing::Matched {
            raw: r,
            translated: t,
            score,
        } = *step
        {
            match read_page(res, &reader, &raw[r], &translated[t], score).await {
                Ok(pairs) => {
                    corpus::append_pairs(&out, &pairs)?;
                    checkpoint.paired_pages += 1;
                }
                Err(err) => {
                    tracing::warn!(page = %raw[r].display(), "style scan: skipping page: {err:#}");
                }
            }
        }

        checkpoint.steps_done = index + 1;
        checkpoint.pairs_bytes = Some(corpus::pairs_len(&out));
        corpus::write_checkpoint(&out, &checkpoint)?;
    }

    if cancel.load(Ordering::Relaxed) {
        return Ok(());
    }

    progress(job_id, PipelineStatus::Running, total - 1, total);
    let pairs = corpus::read_pairs(&out)?;
    let (system, user) = style::request(&pairs).ok_or_else(|| {
        anyhow::anyhow!(
            "No usable sentence pairs came out of {} matched pages — are raw/ and trans/ the same volume?",
            matched
        )
    })?;
    let reply = res.llm.complete(system, &user).await?;
    let profile = style::from_reply(&reply, &pairs)?;

    corpus::write_result(
        &out,
        &StyleScanResult {
            profile,
            raw_pages: raw.len(),
            translated_pages: translated.len(),
            paired_pages: checkpoint.paired_pages,
            pair_count: pairs.len(),
            is_verified_by_human: false,
        },
    )?;
    Ok(())
}

/// Read one matched page on both sides and pair its balloons.
async fn read_page(
    res: &AppResources,
    reader: &koharu_ml::vietocr::VietOcr,
    raw_path: &Path,
    translated_path: &Path,
    score: f32,
) -> anyhow::Result<Vec<bilingual::SentencePair>> {
    let mut raw = Document::open(raw_path.to_path_buf())?;
    res.ml.detect(&mut raw).await?;
    res.ml.ocr(&mut raw).await?;

    // Only the boxes and the glyph mask are wanted from the translated page;
    // its text comes from the Vietnamese reader, not the pipeline's OCR.
    let mut translated = Document::open(translated_path.to_path_buf())?;
    res.ml.detect(&mut translated).await?;
    let sheet = translated.image.0.clone();
    let mask = translated.segment.as_ref().map(|mask| mask.0.clone());
    for block in &mut translated.text_blocks {
        let text = corpus::read_translated(reader, &sheet, mask.as_ref(), block)?;
        block.text = Some(text);
    }

    let name = raw_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    Ok(bilingual::pairs_from_page(
        &raw.text_blocks,
        &translated.text_blocks,
        &name,
        score,
        raw.width as f32,
        raw.height as f32,
    ))
}

fn progress(job_id: &str, status: PipelineStatus, done: usize, total: usize) {
    pipeline::emit_progress(PipelineProgress {
        job_id: job_id.to_string(),
        kind: JOB_KIND.to_string(),
        status,
        step: None,
        current_document: done.min(total),
        total_documents: total,
        current_step_index: 0,
        total_steps: 1,
        overall_percent: if total == 0 {
            0
        } else {
            ((done.min(total) as f64 / total as f64) * 100.0).round() as u8
        },
    });
}
