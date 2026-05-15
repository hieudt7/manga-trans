use std::{
    path::PathBuf,
    str::FromStr,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use koharu_llm::ModelId;
use koharu_types::{
    FolderFile, FolderFileInfo, FolderSession, FolderSessionInfo,
    ProcessRequest,
    events::{PipelineProgress, PipelineStatus, PipelineStep},
};
use rfd::AsyncFileDialog;
use tracing::instrument;

use crate::{AppResources, pipeline};
use super::utils::encode_image_with_dpi;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "tif", "tiff"];
const RESULT_DIR_NAME: &str = "result";

/// Open a native folder picker, scan for images, and store a lightweight
/// FolderSession (paths + dimensions only, no pixel data) in app state.
#[instrument(level = "info", skip_all)]
pub async fn open_folder_session(state: AppResources) -> anyhow::Result<FolderSessionInfo> {
    let handle = AsyncFileDialog::new()
        .pick_folder()
        .await
        .ok_or_else(|| anyhow::anyhow!("No folder selected"))?;
    let folder = handle.path().to_path_buf();

    let (root, result_dir, files) =
        tokio::task::spawn_blocking(move || scan_folder_images(&folder)).await??;

    let info = build_session_info(&root, &result_dir, &files);

    {
        let mut guard = state.state.write().await;
        guard.folder_session = Some(FolderSession { root, result_dir, files });
    }

    Ok(info)
}

/// Open a folder session from an explicit path (headless mode, no native dialog).
pub async fn open_folder_session_by_path(
    state: AppResources,
    folder: PathBuf,
) -> anyhow::Result<FolderSessionInfo> {
    if !folder.is_dir() {
        anyhow::bail!("Path is not a valid directory: {}", folder.display());
    }
    let (root, result_dir, files) =
        tokio::task::spawn_blocking(move || scan_folder_images(&folder)).await??;
    let info = build_session_info(&root, &result_dir, &files);
    {
        let mut guard = state.state.write().await;
        guard.folder_session = Some(FolderSession { root, result_dir, files });
    }
    Ok(info)
}

/// Return current folder session metadata without re-scanning.
pub async fn get_folder_session(state: AppResources) -> anyhow::Result<Option<FolderSessionInfo>> {
    let guard = state.state.read().await;
    Ok(guard.folder_session.as_ref().map(|s| build_session_info(&s.root, &s.result_dir, &s.files)))
}

/// Serve a source image by session index. Reads from disk, returns bytes, does NOT store in state.
pub async fn get_folder_image_bytes(state: AppResources, index: usize) -> anyhow::Result<Vec<u8>> {
    let path = folder_file_path(&state, index).await?;
    Ok(tokio::fs::read(&path).await?)
}

/// Serve a result image by session index from the result/ subfolder.
pub async fn get_folder_result_bytes(state: AppResources, index: usize) -> anyhow::Result<Vec<u8>> {
    let (result_dir, filename) = {
        let guard = state.state.read().await;
        let session = require_session(&guard)?;
        let file = require_file(session, index)?;
        (
            session.result_dir.clone(),
            file.path.file_name().unwrap_or_default().to_os_string(),
        )
    };
    let result_path = result_dir.join(filename);
    if !result_path.exists() {
        anyhow::bail!("Result not found for index {index}");
    }
    Ok(tokio::fs::read(&result_path).await?)
}

/// Start a background folder pipeline job that processes every image sequentially,
/// saving each result to result/ and freeing memory after each page.
pub async fn start_folder_pipeline(
    resources: AppResources,
    request: ProcessRequest,
) -> anyhow::Result<String> {
    {
        let guard = resources.pipeline.read().await;
        if guard.is_some() {
            anyhow::bail!("A pipeline job is already running");
        }
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

    let res = resources.clone();
    let jid = job_id.clone();
    tokio::spawn(async move {
        run_folder_pipeline(res, request, cancel, jid).await;
    });

    Ok(job_id)
}

// ─── internal ────────────────────────────────────────────────────────────────

async fn run_folder_pipeline(
    resources: AppResources,
    request: ProcessRequest,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    job_id: String,
) {
    let total_files = {
        let guard = resources.state.read().await;
        guard.folder_session.as_ref().map(|s| s.files.len()).unwrap_or(0)
    };
    let total_steps = PipelineStep::ALL.len();

    match run_folder_pipeline_inner(&resources, &request, &cancel, &job_id).await {
        Ok(()) if cancel.load(Ordering::Relaxed) => {
            pipeline::emit_progress(PipelineProgress {
                job_id: job_id.clone(),
                status: PipelineStatus::Cancelled,
                step: None,
                current_document: total_files,
                total_documents: total_files,
                current_step_index: 0,
                total_steps,
                overall_percent: 0,
            });
        }
        Ok(()) => {
            pipeline::emit_progress(PipelineProgress {
                job_id: job_id.clone(),
                status: PipelineStatus::Completed,
                step: None,
                current_document: total_files,
                total_documents: total_files,
                current_step_index: total_steps,
                total_steps,
                overall_percent: 100,
            });
        }
        Err(err) => {
            tracing::error!("Folder pipeline failed: {err:#}");
            pipeline::emit_progress(PipelineProgress {
                job_id: job_id.clone(),
                status: PipelineStatus::Failed(err.to_string()),
                step: None,
                current_document: 0,
                total_documents: total_files,
                current_step_index: 0,
                total_steps,
                overall_percent: 0,
            });
        }
    }

    let mut guard = resources.pipeline.write().await;
    *guard = None;
}

async fn run_folder_pipeline_inner(
    res: &AppResources,
    req: &ProcessRequest,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    job_id: &str,
) -> anyhow::Result<()> {
    let (files, result_dir) = {
        let guard = res.state.read().await;
        let session = require_session(&guard)?;
        (
            session
                .files
                .iter()
                .map(|f| (f.path.clone(), f.has_result))
                .collect::<Vec<_>>(),
            session.result_dir.clone(),
        )
    };

    let total_docs = files.len();
    if total_docs == 0 {
        return Ok(());
    }

    let pending = files.iter().filter(|(_, done)| !done).count();
    if pending == 0 {
        tracing::info!("all {} images already processed, nothing to do", total_docs);
        return Ok(());
    }
    tracing::info!("folder pipeline: {} pending / {} total", pending, total_docs);

    load_llm_if_needed(res, req, cancel).await?;

    tokio::fs::create_dir_all(&result_dir).await?;

    let total_steps = PipelineStep::ALL.len();

    for (doc_idx, (file_path, has_result)) in files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }

        if *has_result {
            tracing::debug!(file = %file_path.display(), "skipping already-rendered image");
            continue;
        }

        let bytes = tokio::fs::read(file_path).await?;
        let mut doc = koharu_types::Document::from_bytes(file_path.clone(), bytes)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Failed to parse: {}", file_path.display()))?;

        let file_ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg")
            .to_string();
        let result_filename = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let page_result = process_single_file(
            res,
            req,
            cancel,
            job_id,
            &mut doc,
            doc_idx,
            total_docs,
            total_steps,
        )
        .await;

        if let Err(ref err) = page_result {
            tracing::warn!(
                file = %file_path.display(),
                "Skipping page due to error: {err:#}"
            );
        }

        // Save to result/ and free memory immediately.
        if page_result.is_ok() {
            if let Some(rendered) = &doc.rendered {
                let bytes = encode_image_with_dpi(rendered, &file_ext, doc.resolution_dpi)?;
                let out = result_dir.join(&result_filename);
                tokio::fs::write(&out, bytes).await?;
                tracing::info!(path = %out.display(), "folder result saved");

                let mut guard = res.state.write().await;
                if let Some(session) = guard.folder_session.as_mut() {
                    if let Some(f) = session.files.get_mut(doc_idx) {
                        f.has_result = true;
                    }
                }
            }
        }
        // doc dropped here — pixel data freed
    }

    Ok(())
}

async fn load_llm_if_needed(
    res: &AppResources,
    req: &ProcessRequest,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<()> {
    let Some(model_id) = &req.llm_model_id else {
        return Ok(());
    };
    if res.llm.ready().await {
        return Ok(());
    }

    if model_id.contains(':') {
        let (provider_id, model_part) = model_id.split_once(':').unwrap();
        res.llm
            .load_api(
                provider_id,
                model_part,
                koharu_llm::providers::ProviderConfig {
                    api_key: req.llm_api_key.clone(),
                    base_url: req.llm_base_url.clone(),
                    temperature: req.llm_temperature,
                    max_tokens: req.llm_max_tokens,
                    custom_system_prompt: req.llm_custom_system_prompt.clone(),
                    story_context: None,
                },
            )
            .await?;
    } else {
        let id = ModelId::from_str(model_id)?;
        res.llm.load(id).await;
        for _ in 0..300 {
            if res.llm.ready().await { break; }
            tokio::time::sleep(Duration::from_millis(100)).await;
            if cancel.load(Ordering::Relaxed) { return Ok(()); }
        }
        if !res.llm.ready().await {
            anyhow::bail!("LLM failed to load within timeout");
        }
    }
    Ok(())
}

async fn process_single_file(
    res: &AppResources,
    req: &ProcessRequest,
    cancel: &Arc<std::sync::atomic::AtomicBool>,
    job_id: &str,
    doc: &mut koharu_types::Document,
    doc_idx: usize,
    total_docs: usize,
    total_steps: usize,
) -> anyhow::Result<()> {
    for (step_idx, step) in PipelineStep::ALL.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }

        pipeline::emit_progress(PipelineProgress {
            job_id: job_id.to_string(),
            status: PipelineStatus::Running,
            step: Some(*step),
            current_document: doc_idx,
            total_documents: total_docs,
            current_step_index: step_idx,
            total_steps,
            overall_percent: percent(doc_idx, step_idx, total_docs, total_steps),
        });

        tokio::task::yield_now().await;

        match step {
            PipelineStep::Detect => res.ml.detect(doc).await?,
            PipelineStep::Ocr => {
                if !doc.text_blocks.is_empty() {
                    res.ml.ocr(doc).await?;
                }
            }
            PipelineStep::DetectBalloon => res.ml.detect_balloons(doc).await?,
            PipelineStep::LlmGenerate => {
                if res.llm.ready().await && !doc.text_blocks.is_empty() {
                    res.llm
                        .translate_with_context(doc, req.language.as_deref(), None)
                        .await?;
                    for block in &mut doc.text_blocks {
                        let has_src = block
                            .text
                            .as_deref()
                            .map(|t| !t.trim().is_empty())
                            .unwrap_or(false);
                        let empty = block
                            .translation
                            .as_deref()
                            .map(|t| t.trim().is_empty())
                            .unwrap_or(true);
                        if has_src && empty {
                            block.translation = Some(".\n.\n.".to_string());
                        }
                    }
                }
            }
            PipelineStep::Inpaint => {
                if doc.segment.is_some() && !doc.text_blocks.is_empty() {
                    res.ml.inpaint(doc).await?;
                }
                // Clear FFT plans immediately on this thread — Lama FFT ran synchronously
                // here so FFT_PLANS is populated on this exact tokio worker thread.
                // Any subsequent .await (yield_now, fs::write, etc.) may migrate the task
                // to a different thread, making a later block_in_place ineffective.
                tokio::task::block_in_place(koharu_ml::lama::clear_fft_plans_on_current_thread);
            }
            PipelineStep::Render => {
                if !doc.text_blocks.is_empty() {
                    res.renderer.render(
                        doc,
                        None,
                        req.shader_effect.unwrap_or_default(),
                        req.shader_stroke.clone(),
                        req.font_family.as_deref(),
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn percent(doc: usize, step: usize, total_docs: usize, total_steps: usize) -> u8 {
    let total = total_docs * total_steps;
    if total == 0 { return 0; }
    (((doc * total_steps + step) as f64 / total as f64) * 100.0).round() as u8
}

async fn folder_file_path(state: &AppResources, index: usize) -> anyhow::Result<PathBuf> {
    let guard = state.state.read().await;
    let session = require_session(&guard)?;
    let file = require_file(session, index)?;
    Ok(file.path.clone())
}

fn require_session(guard: &koharu_types::State) -> anyhow::Result<&FolderSession> {
    guard
        .folder_session
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No folder session active"))
}

fn require_file(session: &FolderSession, index: usize) -> anyhow::Result<&FolderFile> {
    session
        .files
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("File index {index} out of range"))
}

fn scan_folder_images(
    folder: &std::path::Path,
) -> anyhow::Result<(PathBuf, PathBuf, Vec<FolderFile>)> {
    let result_dir = folder.join(RESULT_DIR_NAME);

    let mut entries: Vec<PathBuf> = std::fs::read_dir(folder)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
                    .unwrap_or(false)
        })
        .collect();
    entries.sort();

    let files = entries
        .into_iter()
        .map(|path| {
            let (width, height) = read_image_dimensions(&path).unwrap_or((0, 0));
            let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let has_result = result_dir
                .join(path.file_name().unwrap_or_default())
                .exists();
            FolderFile { path, name, width, height, has_result }
        })
        .collect();

    Ok((folder.to_path_buf(), result_dir, files))
}

/// Read image dimensions from the file header only — fast, no full pixel decode.
fn read_image_dimensions(path: &std::path::Path) -> anyhow::Result<(u32, u32)> {
    use std::io::BufReader;
    let file = std::fs::File::open(path)?;
    let reader = image::io::Reader::new(BufReader::new(file)).with_guessed_format()?;
    Ok(reader.into_dimensions()?)
}

fn build_session_info(root: &PathBuf, result_dir: &PathBuf, files: &[FolderFile]) -> FolderSessionInfo {
    FolderSessionInfo {
        root: root.to_string_lossy().to_string(),
        result_dir: result_dir.to_string_lossy().to_string(),
        files: files
            .iter()
            .enumerate()
            .map(|(i, f)| FolderFileInfo {
                index: i,
                name: f.name.clone(),
                width: f.width,
                height: f.height,
                has_result: f.has_result,
            })
            .collect(),
    }
}
