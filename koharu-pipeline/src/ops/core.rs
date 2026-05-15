use std::path::{Path, PathBuf};

use image::ImageFormat;
use koharu_types::commands::{
    DeviceInfo, FileResult, IndexPayload, OpenDocumentsPayload, OpenExternalPayload,
    ThumbnailResult,
};
use rfd::FileDialog;

use crate::{AppResources, state_tx::{self, ChangedField}};

use super::utils::{encode_image_with_dpi, load_documents, mime_from_ext};

fn next_available_path(output_dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let mut candidate = output_dir.join(format!("{stem}.{ext}"));
    let mut suffix = 2usize;
    while candidate.exists() {
        candidate = output_dir.join(format!("{stem}_{suffix}.{ext}"));
        suffix += 1;
    }
    candidate
}

async fn pick_output_dir() -> anyhow::Result<Option<PathBuf>> {
    Ok(tokio::task::spawn_blocking(|| FileDialog::new().pick_folder()).await?)
}

fn document_ext(document: &koharu_types::Document) -> String {
    document
        .path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("jpg")
        .to_string()
}

fn export_documents_matching(
    documents: &[koharu_types::Document],
    output_dir: &Path,
    suffix: &str,
    missing_error: &str,
    image: impl Fn(&koharu_types::Document) -> Option<&koharu_types::SerializableDynamicImage>,
) -> anyhow::Result<usize> {
    let mut exported = 0usize;

    for document in documents {
        let Some(image) = image(document) else {
            continue;
        };

        let ext = document_ext(document);
        let output_path =
            next_available_path(output_dir, &format!("{}_{}", document.name, suffix), &ext);
        let bytes = encode_image_with_dpi(image, &ext, document.resolution_dpi)?;
        std::fs::write(&output_path, bytes)?;
        exported += 1;
    }

    anyhow::ensure!(exported > 0, "{missing_error}");
    Ok(exported)
}

pub async fn app_version(state: AppResources) -> anyhow::Result<String> {
    Ok(state.version.to_string())
}

pub async fn device(state: AppResources) -> anyhow::Result<DeviceInfo> {
    Ok(DeviceInfo {
        ml_device: match state.device {
            koharu_ml::Device::Cpu => "CPU".to_string(),
            koharu_ml::Device::Cuda(_) => "CUDA".to_string(),
            koharu_ml::Device::Metal(_) => "Metal".to_string(),
        },
    })
}

pub async fn open_external(
    _state: AppResources,
    payload: OpenExternalPayload,
) -> anyhow::Result<()> {
    open::that(&payload.url)?;
    Ok(())
}

pub async fn get_documents(state: AppResources) -> anyhow::Result<usize> {
    let guard = state.state.read().await;
    Ok(guard.documents.len())
}

pub async fn get_document(
    state: AppResources,
    payload: IndexPayload,
) -> anyhow::Result<koharu_types::Document> {
    state_tx::read_doc(&state.state, payload.index).await
}

pub async fn get_thumbnail(
    state: AppResources,
    payload: IndexPayload,
) -> anyhow::Result<ThumbnailResult> {
    let doc = state_tx::read_doc(&state.state, payload.index).await?;

    let source = doc.rendered.as_ref().unwrap_or(&doc.image);
    let thumbnail = source.thumbnail(200, 200);

    let mut buf = std::io::Cursor::new(Vec::new());
    thumbnail.write_to(&mut buf, ImageFormat::WebP)?;

    Ok(ThumbnailResult {
        data: buf.into_inner(),
        content_type: "image/webp".to_string(),
    })
}

pub async fn open_documents(
    state: AppResources,
    payload: OpenDocumentsPayload,
) -> anyhow::Result<usize> {
    let inputs: Vec<(PathBuf, Vec<u8>)> = payload
        .files
        .into_iter()
        .map(|f| (PathBuf::from(f.name), f.data))
        .collect();

    if inputs.is_empty() {
        anyhow::bail!("No files uploaded");
    }

    let docs = load_documents(inputs)?;
    state_tx::replace_docs(&state.state, docs).await
}

pub async fn add_documents(
    state: AppResources,
    payload: OpenDocumentsPayload,
) -> anyhow::Result<usize> {
    let inputs: Vec<(PathBuf, Vec<u8>)> = payload
        .files
        .into_iter()
        .map(|f| (PathBuf::from(f.name), f.data))
        .collect();

    if inputs.is_empty() {
        anyhow::bail!("No files uploaded");
    }

    let docs = load_documents(inputs)?;
    state_tx::append_docs(&state.state, docs).await
}

pub async fn export_document(
    state: AppResources,
    payload: IndexPayload,
) -> anyhow::Result<FileResult> {
    let document = state_tx::read_doc(&state.state, payload.index).await?;

    let ext = document
        .path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg")
        .to_string();

    let rendered = document
        .rendered
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No rendered image found"))?;

    let bytes = encode_image_with_dpi(rendered, &ext, document.resolution_dpi)?;
    let filename = format!("{}_koharu.{}", document.name, ext);
    let content_type = mime_from_ext(&ext).to_string();

    Ok(FileResult {
        filename,
        data: bytes,
        content_type,
    })
}

pub async fn save_rendered(state: AppResources, payload: IndexPayload) -> anyhow::Result<()> {
    let document = state_tx::read_doc(&state.state, payload.index).await?;

    let rendered = document
        .rendered
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No rendered image for document '{}'", document.name))?;

    let home = std::env::var("HOME")
        .map_err(|_| anyhow::anyhow!("Cannot determine home directory"))?;
    let render_dir = std::path::Path::new(&home).join("Documents").join("AI_Trans");
    std::fs::create_dir_all(&render_dir)?;

    let ext = document_ext(&document);
    let output_path = render_dir.join(format!("{}.{}", document.name, ext));
    let bytes = encode_image_with_dpi(rendered, &ext, document.resolution_dpi)?;
    std::fs::write(&output_path, bytes)?;

    // Release heavyweight image buffers now that the page is saved to disk.
    // Keeps memory low during long Process All runs.
    release_page_resources(state, payload).await
}

/// Maximum long-edge size for the downscaled preview image kept after a page
/// is saved.  Anything larger is replaced with a thumbnail so that processed
/// pages don't accumulate full-resolution originals in memory during long
/// batch runs (e.g. 100–200 page Process All).
const THUMB_MAX_PX: u32 = 500;

/// Drop all heavyweight image buffers after a page has been saved to disk.
/// The original `image` is replaced with a small thumbnail so that processed
/// pages stay in memory at ~100 KB instead of ~10 MB each.
/// `get_thumbnail` works fine with the thumbnail; full-res is on disk.
pub async fn release_page_resources(
    state: AppResources,
    payload: IndexPayload,
) -> anyhow::Result<()> {
    state_tx::mutate_doc(
        &state.state,
        payload.index,
        &[ChangedField::Rendered],
        |doc| {
            doc.segment = None;
            doc.inpainted = None;
            doc.rendered = None;
            doc.brush_layer = None;
            doc.balloons.clear();
            doc.balloons.shrink_to_fit();
            for block in &mut doc.text_blocks {
                block.rendered = None;
            }

            // Downscale the original image to a small preview.  The source
            // file is unchanged on disk; re-open to re-process this page.
            let (w, h) = (doc.image.width(), doc.image.height());
            if w > THUMB_MAX_PX || h > THUMB_MAX_PX {
                let thumb = doc.image.thumbnail(THUMB_MAX_PX, THUMB_MAX_PX);
                doc.image = thumb.into();
            }

            tracing::debug!(
                index = payload.index,
                orig_w = w, orig_h = h,
                "page resources released"
            );
            Ok(())
        },
    )
    .await
}

pub async fn save_rendered_psd(state: AppResources, payload: IndexPayload) -> anyhow::Result<()> {
    let document = state_tx::read_doc(&state.state, payload.index).await?;

    if document.rendered.is_none() {
        anyhow::bail!("No rendered image for document '{}'", document.name);
    }

    let home = std::env::var("HOME")
        .map_err(|_| anyhow::anyhow!("Cannot determine home directory"))?;
    let render_dir = std::path::Path::new(&home).join("Documents").join("AI_Trans");
    std::fs::create_dir_all(&render_dir)?;

    let output_path = render_dir.join(format!("{}.psd", document.name));
    let options = koharu_psd::PsdExportOptions {
        text_layer_mode: koharu_psd::TextLayerMode::Editable,
        ..koharu_psd::PsdExportOptions::default()
    };
    let bytes = koharu_psd::export_document(&document, &options)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::write(&output_path, bytes)?;

    release_page_resources(state, payload).await
}

pub async fn save_rendered_tiff(state: AppResources, payload: IndexPayload) -> anyhow::Result<()> {
    let document = state_tx::read_doc(&state.state, payload.index).await?;

    if document.rendered.is_none() {
        anyhow::bail!("No rendered image for document '{}'", document.name);
    }

    let home = std::env::var("HOME")
        .map_err(|_| anyhow::anyhow!("Cannot determine home directory"))?;
    let render_dir = std::path::Path::new(&home).join("Documents").join("AI_Trans");
    std::fs::create_dir_all(&render_dir)?;

    let output_path = render_dir.join(format!("{}.tif", document.name));
    let options = koharu_psd::PsdExportOptions {
        text_layer_mode: koharu_psd::TextLayerMode::Editable,
        ..koharu_psd::PsdExportOptions::default()
    };
    let bytes = koharu_tiff::export_document(&document, &options)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    std::fs::write(&output_path, bytes)?;

    release_page_resources(state, payload).await
}

pub async fn export_all_inpainted(state: AppResources) -> anyhow::Result<usize> {
    let Some(output_dir) = pick_output_dir().await? else {
        return Ok(0);
    };

    let guard = state.state.read().await;
    export_documents_matching(
        &guard.documents,
        &output_dir,
        "inpainted",
        "No inpainted images found to export",
        |document| document.inpainted.as_ref(),
    )
}

pub async fn export_all_rendered(state: AppResources) -> anyhow::Result<usize> {
    let Some(output_dir) = pick_output_dir().await? else {
        return Ok(0);
    };

    let guard = state.state.read().await;
    export_documents_matching(
        &guard.documents,
        &output_dir,
        "rendered",
        "No rendered images found to export",
        |document| document.rendered.as_ref(),
    )
}
