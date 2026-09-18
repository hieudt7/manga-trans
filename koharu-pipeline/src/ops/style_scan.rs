//! Learns how a published translation reads, as a background job.
//!
//! The open folder is a reference volume: the artist's pages in `raw/`, the
//! published Vietnamese in `trans/`. Two readers are offered:
//!
//! - **vietocr** — detection and OCR on this machine. Lines the two volumes up,
//!   reads the balloons on both sides of every matched page, and asks the
//!   loaded LLM what the sentence pairs say about the translator. Needs both
//!   `raw/` and `trans/`.
//! - **vision** — each page is shown to a model that can see (the Claude API),
//!   which notes every line with who says it to whom and in what mood; one more
//!   call then writes the profile from those notes. This is how a person reads
//!   a volume, and on the reference corpus it caught what the OCR route could
//!   not (see `koharu_ml::bilingual::vision`). Works with `trans/` alone.
//!
//! Output goes to `<folder>/style_scan/`, and a finished profile is also kept in
//! the profile library (`corpus::library_dir`). Nothing is applied to
//! translation until the curator chooses a profile (`set_active_style_profile`).

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use koharu_llm::providers::{AnyProvider, ProviderConfig, build_provider, get_saved_api_key};
use koharu_ml::bilingual::{
    self, PagePairing, PageSignature,
    corpus::{self, StyleScanCheckpoint, StyleScanResult},
    style::{self, StyleProfile},
    vision::{self, PageNotes},
};
use koharu_types::{
    Document, FolderSession,
    events::{PipelineProgress, PipelineStatus},
};
use serde::Deserialize;
use tracing::instrument;

use crate::{AppResources, pipeline};

const JOB_KIND: &str = "style-scan-folder";

/// Pages read at once. Each is a request of a few thousand tokens; more at once
/// mostly buys rate-limit retries.
const VISION_CONCURRENCY: usize = 3;

/// A page whose reply is not usable JSON is asked again, up to this many tries.
const VISION_ATTEMPTS: usize = 2;

const DEFAULT_VISION_PROVIDER: &str = "claude";
const DEFAULT_VISION_MODEL: &str = "claude-opus-5";

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum StyleReader {
    #[default]
    Vietocr,
    Vision,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct StyleScanOptions {
    pub reader: StyleReader,
    /// API provider for the vision reader. Only providers that accept images
    /// work; `claude` by default.
    pub provider: Option<String>,
    /// Model for the vision reader; `claude-opus-5` by default.
    pub model: Option<String>,
    /// Library name; derived from the folder when absent.
    pub name: Option<String>,
}

/// One page to be read by the vision reader.
#[derive(Debug, Clone)]
struct VisionPage {
    id: String,
    raw: Option<PathBuf>,
    translated: PathBuf,
}

enum Plan {
    Vietocr {
        raw: Vec<PathBuf>,
        translated: Vec<PathBuf>,
    },
    Vision {
        raw: Vec<PathBuf>,
        translated: Vec<PathBuf>,
        provider: Arc<dyn AnyProvider>,
        model: String,
    },
}

/// Start a style scan over the open folder session. Returns the job id and the
/// number of steps progress will count through.
#[instrument(level = "info", skip_all)]
pub async fn start_style_scan_job(
    resources: AppResources,
    options: StyleScanOptions,
) -> anyhow::Result<(String, usize)> {
    {
        let guard = resources.pipeline.read().await;
        if guard.is_some() {
            anyhow::bail!("A pipeline job is already running");
        }
    }

    let root = session_root(&resources).await?;
    let raw = corpus::list_pages(&root.join(corpus::RAW_DIR));
    let mut translated = corpus::list_pages(&root.join(corpus::TRANSLATED_DIR));

    let plan = match options.reader {
        StyleReader::Vietocr => {
            if raw.is_empty() || translated.is_empty() {
                anyhow::bail!(
                    "The OCR scan needs the artist's pages in {}/ and the published translation in {}/ \
                     inside {} ({} and {} images found). Read with Claude for a translation on its own.",
                    corpus::RAW_DIR,
                    corpus::TRANSLATED_DIR,
                    root.display(),
                    raw.len(),
                    translated.len()
                );
            }
            // Both are needed only at the far end of a long read; finding out
            // they are missing after an hour of OCR would waste the hour.
            corpus::open_reader()?;
            if !resources.llm.is_api_ready().await {
                anyhow::bail!(
                    "Load an API LLM provider first — it reads the sentence pairs at the end of the scan"
                );
            }
            Plan::Vietocr { raw, translated }
        }
        StyleReader::Vision => {
            // A translation on its own may sit in trans/ or directly in the folder.
            if translated.is_empty() && raw.is_empty() {
                translated = corpus::list_pages(&root);
            }
            if translated.is_empty() {
                anyhow::bail!(
                    "No translated pages found in {}/{}/ or in the folder itself",
                    root.display(),
                    corpus::TRANSLATED_DIR
                );
            }
            let provider_id = options
                .provider
                .clone()
                .unwrap_or_else(|| DEFAULT_VISION_PROVIDER.to_string());
            let api_key = get_saved_api_key(&provider_id)?
                .filter(|key| !key.trim().is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Add an API key for {provider_id} in Settings before reading with it"
                    )
                })?;
            let provider = build_provider(
                &provider_id,
                ProviderConfig {
                    api_key: Some(api_key),
                    base_url: None,
                    temperature: None,
                    max_tokens: None,
                    custom_system_prompt: None,
                    story_context: None,
                    key_start_index: None,
                },
            )?;
            Plan::Vision {
                raw,
                translated,
                provider: Arc::from(provider),
                model: options
                    .model
                    .clone()
                    .filter(|m| !m.trim().is_empty())
                    .unwrap_or_else(|| DEFAULT_VISION_MODEL.to_string()),
            }
        }
    };

    // Preparing, one step per page, then writing the profile.
    let total = match &plan {
        Plan::Vietocr { raw, .. } => raw.len() + 2,
        Plan::Vision { translated, .. } => translated.len() + 2,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let job_id = uuid::Uuid::new_v4().to_string();
    {
        let mut guard = resources.pipeline.write().await;
        *guard = Some(pipeline::PipelineHandle {
            id: job_id.clone(),
            cancel: cancel.clone(),
        });
    }

    let name = options
        .name
        .clone()
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| corpus::default_library_name(&root));
    let res = resources.clone();
    let jid = job_id.clone();
    tokio::spawn(async move {
        let outcome = match plan {
            Plan::Vietocr { raw, translated } => {
                run_vietocr(&res, &cancel, &jid, &root, &raw, &translated, &name).await
            }
            Plan::Vision {
                raw,
                translated,
                provider,
                model,
            } => {
                let job = VisionJob {
                    cancel: cancel.clone(),
                    job_id: jid.clone(),
                    root: root.clone(),
                    provider,
                    model,
                    name,
                    total,
                };
                job.run(raw, translated).await
            }
        };
        finish(&res, &cancel, &jid, outcome, total).await;
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

/// Save the curator's edits to the open folder's profile, marking it reviewed,
/// and to its library entry.
pub async fn export_style_scan_result(
    resources: AppResources,
    mut result: StyleScanResult,
) -> anyhow::Result<()> {
    let root = session_root(&resources).await?;
    result.is_verified_by_human = true;
    if result.name.is_none() {
        result.name = Some(corpus::default_library_name(&root));
    }
    corpus::write_result(&corpus::out_dir(&root), &result)?;
    corpus::save_to_library(&corpus::library_dir(), &result)?;
    Ok(())
}

/// Every profile kept in the library, newest first.
pub async fn list_style_profiles() -> anyhow::Result<Vec<StyleScanResult>> {
    Ok(corpus::list_library(&corpus::library_dir()))
}

/// The profile translations currently follow, if any.
pub async fn get_active_style_profile() -> anyhow::Result<Option<StyleProfile>> {
    Ok(style::load_active())
}

/// Make `profile` the one translations follow, or stop following any.
///
/// The profile rides in the story context, which is fixed for a loaded model so
/// the provider can cache it; a loaded API model is therefore loaded again at
/// once, and the next page translated follows the new profile.
pub async fn set_active_style_profile(
    resources: AppResources,
    profile: Option<StyleProfile>,
) -> anyhow::Result<()> {
    style::set_active(profile.as_ref())?;
    crate::ops::refresh_story_context(resources).await?;
    Ok(())
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

fn now() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// Write the finished result beside the pages and into the library.
fn publish(root: &Path, result: &StyleScanResult) -> anyhow::Result<()> {
    corpus::write_result(&corpus::out_dir(root), result)?;
    let saved = corpus::save_to_library(&corpus::library_dir(), result)?;
    tracing::info!(path = %saved.display(), "style profile saved to the library");
    Ok(())
}

async fn finish(
    resources: &AppResources,
    cancel: &AtomicBool,
    job_id: &str,
    outcome: anyhow::Result<()>,
    total: usize,
) {
    let status = match outcome {
        Ok(()) if cancel.load(Ordering::Relaxed) => PipelineStatus::Cancelled,
        Ok(()) => PipelineStatus::Completed,
        Err(err) => {
            tracing::error!("Style scan failed: {err:#}");
            PipelineStatus::Failed(err.to_string())
        }
    };
    let done = matches!(status, PipelineStatus::Completed);
    progress(job_id, status, if done { total } else { 0 }, total);

    let mut guard = resources.pipeline.write().await;
    *guard = None;
}

// ─── vision reader ───────────────────────────────────────────────────────────

struct VisionJob {
    cancel: Arc<AtomicBool>,
    job_id: String,
    root: PathBuf,
    provider: Arc<dyn AnyProvider>,
    model: String,
    name: String,
    total: usize,
}

impl VisionJob {
    async fn run(self, raw: Vec<PathBuf>, translated: Vec<PathBuf>) -> anyhow::Result<()> {
        let out = corpus::out_dir(&self.root);
        let (raw_count, translated_count) = (raw.len(), translated.len());
        progress(&self.job_id, PipelineStatus::Running, 0, self.total);

        let pages = plan_vision_pages(raw, translated).await?;
        let paired = pages.iter().filter(|p| p.raw.is_some()).count();
        tracing::info!(pages = pages.len(), paired, model = %self.model, "style scan: reading pages");

        // A page noted by an earlier, interrupted run is not paid for twice.
        let todo: Vec<VisionPage> = pages
            .iter()
            .filter(|p| corpus::read_page_notes(&out, &p.id).is_none())
            .cloned()
            .collect();
        let done = Arc::new(AtomicUsize::new(1 + pages.len() - todo.len()));
        progress(
            &self.job_id,
            PipelineStatus::Running,
            done.load(Ordering::Relaxed),
            self.total,
        );

        self.read_pages(todo, &out, &done).await?;
        if self.cancel.load(Ordering::Relaxed) {
            return Ok(());
        }

        let notes: Vec<PageNotes> = pages
            .iter()
            .filter_map(|p| corpus::read_page_notes(&out, &p.id))
            .collect();
        let missing = pages.len() - notes.len();
        if notes.is_empty() {
            anyhow::bail!("No page could be read");
        }
        // A handful of unreadable pages does not change a volume's style; a
        // large share of them means the profile would describe a different book.
        if missing * 10 > pages.len() {
            anyhow::bail!(
                "{missing} of {} pages could not be read — run the scan again to retry them",
                pages.len()
            );
        }

        progress(
            &self.job_id,
            PipelineStatus::Running,
            self.total - 1,
            self.total,
        );
        let (system, user) = vision::synthesis_request(&notes);
        let reply = self.provider.complete(system, &user, &self.model).await?;
        let profile = vision::profile_from_reply(&reply, &notes)?;

        publish(
            &self.root,
            &StyleScanResult {
                profile,
                raw_pages: raw_count,
                translated_pages: translated_count,
                paired_pages: notes.len(),
                pair_count: notes.iter().map(|n| n.lines.len()).sum(),
                is_verified_by_human: false,
                source: Some("claude-api".to_string()),
                model: Some(self.model.clone()),
                with_raw: Some(notes.iter().any(|n| n.with_raw)),
                name: Some(self.name.clone()),
                created_at: now(),
            },
        )
    }

    async fn read_pages(
        &self,
        todo: Vec<VisionPage>,
        out: &Path,
        done: &Arc<AtomicUsize>,
    ) -> anyhow::Result<()> {
        let semaphore = Arc::new(tokio::sync::Semaphore::new(VISION_CONCURRENCY));
        let mut tasks = tokio::task::JoinSet::new();
        for page in todo {
            let semaphore = semaphore.clone();
            let provider = self.provider.clone();
            let cancel = self.cancel.clone();
            let done = done.clone();
            let out = out.to_path_buf();
            let model = self.model.clone();
            let job_id = self.job_id.clone();
            let total = self.total;
            tasks.spawn(async move {
                let _permit = semaphore.acquire_owned().await?;
                if cancel.load(Ordering::Relaxed) {
                    return Ok::<_, anyhow::Error>(());
                }
                match read_vision_page(&*provider, &model, &page).await {
                    Ok(notes) => corpus::write_page_notes(&out, &notes)?,
                    Err(err) => {
                        // A key or quota problem fails every page the same way;
                        // stop rather than spend the rest of the volume finding out.
                        let message = err.to_string();
                        if message.starts_with("provider_invalid_api_key")
                            || message.starts_with("provider_quota_exceeded")
                        {
                            cancel.store(true, Ordering::Relaxed);
                            return Err(err);
                        }
                        tracing::warn!(page = %page.id, "style scan: could not read page: {err:#}");
                    }
                }
                let now_done = done.fetch_add(1, Ordering::Relaxed) + 1;
                progress(&job_id, PipelineStatus::Running, now_done, total);
                Ok(())
            });
        }

        let mut first_error = None;
        while let Some(joined) = tasks.join_next().await {
            if let Err(err) = joined? {
                first_error.get_or_insert(err);
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

/// Pair each translated page with its original, or with nothing.
///
/// Pages only the original has are not read: there is no translation on them
/// to learn from. Aligning costs no model call — the artwork is the same on
/// both sides.
async fn plan_vision_pages(
    raw: Vec<PathBuf>,
    translated: Vec<PathBuf>,
) -> anyhow::Result<Vec<VisionPage>> {
    tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<VisionPage>> {
        let mut taken = HashSet::new();
        if raw.is_empty() {
            return Ok(translated
                .iter()
                .enumerate()
                .map(|(i, t)| VisionPage {
                    id: corpus::page_id(t, i, &mut taken),
                    raw: None,
                    translated: t.clone(),
                })
                .collect());
        }

        let sign = |files: &[PathBuf]| -> anyhow::Result<Vec<PageSignature>> {
            files
                .iter()
                .map(|path| Ok(PageSignature::new(&image::open(path)?)))
                .collect()
        };
        let mut pages = Vec::new();
        for step in bilingual::align_pages(&sign(&raw)?, &sign(&translated)?) {
            let (r, t) = match step {
                PagePairing::Matched {
                    raw: r,
                    translated: t,
                    ..
                } => (Some(r), t),
                PagePairing::TranslatedOnly(t) => (None, t),
                PagePairing::RawOnly(_) => continue,
            };
            let index = pages.len();
            pages.push((
                t,
                VisionPage {
                    id: corpus::page_id(&translated[t], index, &mut taken),
                    raw: r.map(|r| raw[r].clone()),
                    translated: translated[t].clone(),
                },
            ));
        }
        // The alignment walks the volumes backwards; read them front to back.
        pages.sort_by_key(|(t, _)| *t);
        Ok(pages.into_iter().map(|(_, page)| page).collect())
    })
    .await?
}

/// Show one page to the model and keep what it noted.
async fn read_vision_page(
    provider: &dyn AnyProvider,
    model: &str,
    page: &VisionPage,
) -> anyhow::Result<PageNotes> {
    let with_raw = page.raw.is_some();
    let images = {
        let page = page.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Vec<u8>>> {
            let mut images = Vec::new();
            if let Some(raw) = &page.raw {
                images.push(corpus::reading_jpeg(raw)?);
            }
            images.push(corpus::reading_jpeg(&page.translated)?);
            Ok(images)
        })
        .await??
    };
    let (system, user) = vision::page_request(with_raw);

    let mut last_error = None;
    for attempt in 1..=VISION_ATTEMPTS {
        let reply = provider
            .look(system, &user, &images, "image/jpeg", model)
            .await?;
        match vision::parse_page(&reply, &page.id, with_raw) {
            Ok(notes) => return Ok(notes),
            Err(err) => {
                tracing::warn!(page = %page.id, attempt, "page notes were not usable JSON: {err:#}");
                last_error = Some(err);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no reply")))
}

// ─── vietocr reader ──────────────────────────────────────────────────────────

async fn run_vietocr(
    res: &AppResources,
    cancel: &Arc<AtomicBool>,
    job_id: &str,
    root: &Path,
    raw: &[PathBuf],
    translated: &[PathBuf],
    name: &str,
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
            match read_ocr_page(res, &reader, &raw[r], &translated[t], score).await {
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

    publish(
        root,
        &StyleScanResult {
            profile,
            raw_pages: raw.len(),
            translated_pages: translated.len(),
            paired_pages: checkpoint.paired_pages,
            pair_count: pairs.len(),
            is_verified_by_human: false,
            source: Some("vietocr".to_string()),
            model: None,
            with_raw: Some(true),
            name: Some(name.to_string()),
            created_at: now(),
        },
    )
}

/// Read one matched page on both sides with the local OCR and pair its balloons.
async fn read_ocr_page(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_to_the_ocr_reader_and_accept_the_vision_one() {
        let empty: StyleScanOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.reader, StyleReader::Vietocr);

        let vision: StyleScanOptions = serde_json::from_str(
            r#"{"reader":"vision","model":"claude-sonnet-5","name":"kinnikuman-1"}"#,
        )
        .unwrap();
        assert_eq!(vision.reader, StyleReader::Vision);
        assert_eq!(vision.model.as_deref(), Some("claude-sonnet-5"));
    }

    #[tokio::test]
    async fn a_translation_on_its_own_is_read_page_by_page_without_originals() {
        let pages = plan_vision_pages(
            Vec::new(),
            vec![
                PathBuf::from("trans/表紙0291.JPG"),
                PathBuf::from("trans/表紙0292.JPG"),
            ],
        )
        .await
        .unwrap();
        let ids: Vec<&str> = pages.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["0291", "0292"]);
        assert!(pages.iter().all(|p| p.raw.is_none()));
    }
}
