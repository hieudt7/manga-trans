use std::str::FromStr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use koharu_llm::ModelId;
use koharu_types::commands::ProcessRequest;
use koharu_types::events::{PipelineProgress, PipelineStatus, PipelineStep};
use once_cell::sync::Lazy;
use tokio::sync::broadcast;

use crate::{
    AppResources,
    state_tx::{self, ChangedField},
};

pub struct PipelineHandle {
    pub id: String,
    pub cancel: Arc<AtomicBool>,
}

static PIPELINE_TX: Lazy<broadcast::Sender<PipelineProgress>> =
    Lazy::new(|| broadcast::channel(256).0);

pub fn subscribe() -> broadcast::Receiver<PipelineProgress> {
    PIPELINE_TX.subscribe()
}

fn emit(progress: PipelineProgress) {
    let _ = PIPELINE_TX.send(progress);
}

pub fn emit_progress(progress: PipelineProgress) {
    emit(progress);
}

fn compute_percent(doc: usize, step: usize, total_docs: usize, total_steps: usize) -> u8 {
    let done_units = doc * total_steps + step;
    let total_units = total_docs * total_steps;
    if total_units == 0 {
        return 0;
    }
    ((done_units as f64 / total_units as f64) * 100.0).round() as u8
}

pub async fn run_pipeline(
    resources: AppResources,
    request: ProcessRequest,
    cancel: Arc<AtomicBool>,
    job_id: String,
) {
    let result = run_pipeline_inner(&resources, &request, &cancel, &job_id).await;

    let total_docs = match request.index {
        Some(_) => 1,
        None => resources.state.read().await.documents.len(),
    };

    match result {
        Ok(()) if cancel.load(Ordering::Relaxed) => {
            emit(PipelineProgress {
                job_id: job_id.clone(),
                kind: "pipeline".to_string(),
                status: PipelineStatus::Cancelled,
                step: None,
                current_document: total_docs,
                total_documents: total_docs,
                current_step_index: 0,
                total_steps: PipelineStep::ALL.len(),
                overall_percent: 0,
            });
        }
        Ok(()) => {
            emit(PipelineProgress {
                job_id: job_id.clone(),
                kind: "pipeline".to_string(),
                status: PipelineStatus::Completed,
                step: None,
                current_document: total_docs,
                total_documents: total_docs,
                current_step_index: PipelineStep::ALL.len(),
                total_steps: PipelineStep::ALL.len(),
                overall_percent: 100,
            });
        }
        Err(err) => {
            tracing::error!("Pipeline failed: {err:#}");
            emit(PipelineProgress {
                job_id: job_id.clone(),
                kind: "pipeline".to_string(),
                status: PipelineStatus::Failed(err.to_string()),
                step: None,
                current_document: 0,
                total_documents: total_docs,
                current_step_index: 0,
                total_steps: PipelineStep::ALL.len(),
                overall_percent: 0,
            });
        }
    }

    let mut guard = resources.pipeline.write().await;
    *guard = None;
}

async fn run_pipeline_inner(
    res: &AppResources,
    req: &ProcessRequest,
    cancel: &Arc<AtomicBool>,
    job_id: &str,
) -> anyhow::Result<()> {
    let total_docs = {
        let guard = res.state.read().await;
        let len = guard.documents.len();
        match req.index {
            Some(i) if i >= len => anyhow::bail!("Document index {i} out of range (have {len})"),
            Some(_) => 1,
            None => len,
        }
    };

    if total_docs == 0 {
        return Ok(());
    }

    if let Some(model_id) = &req.llm_model_id
        && !res.llm.ready().await
    {
        if model_id.contains(':') {
            // Through llm_load, so a fallback load carries the chosen style profile and
            // the character library the same way an explicit load from the picker does.
            crate::ops::llm_load(
                res.clone(),
                koharu_types::commands::LlmLoadPayload {
                    id: model_id.clone(),
                    api_key: req.llm_api_key.clone(),
                    base_url: req.llm_base_url.clone(),
                    temperature: req.llm_temperature,
                    max_tokens: req.llm_max_tokens,
                    custom_system_prompt: req.llm_custom_system_prompt.clone(),
                    story_context: None,
                    // The explicit /llm/load from the model picker carries the start index.
                    key_start_index: None,
                },
            )
            .await?;
        } else {
            let id = ModelId::from_str(model_id)?;
            res.llm.load(id).await;
            for _ in 0..300 {
                if res.llm.ready().await {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }
            }
            if !res.llm.ready().await {
                anyhow::bail!("LLM failed to load within timeout");
            }
        }
    }

    let start_index = req.index.unwrap_or(0);
    let end_index = req.index.map(|i| i + 1).unwrap_or(total_docs);
    let total_steps = PipelineStep::ALL.len();

    for (doc_ordinal, doc_index) in (start_index..end_index).enumerate() {
        for (step_ordinal, step) in PipelineStep::ALL.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }

            let overall = compute_percent(doc_ordinal, step_ordinal, total_docs, total_steps);
            emit(PipelineProgress {
                job_id: job_id.to_string(),
                kind: "pipeline".to_string(),
                status: PipelineStatus::Running,
                step: Some(*step),
                current_document: doc_ordinal,
                total_documents: total_docs,
                current_step_index: step_ordinal,
                total_steps,
                overall_percent: overall,
            });

            tokio::task::yield_now().await;
            tokio::time::sleep(Duration::from_millis(1)).await;

            let mut snapshot = state_tx::read_doc(&res.state, doc_index).await?;

            match step {
                PipelineStep::Detect => res.ml.detect(&mut snapshot).await?,
                PipelineStep::Ocr => {
                    if !snapshot.text_blocks.is_empty() {
                        res.ml.ocr(&mut snapshot).await?;
                    }
                }
                PipelineStep::DetectBalloon => res.ml.detect_balloons(&mut snapshot).await?,
                PipelineStep::LlmGenerate => {
                    if res.llm.ready().await && !snapshot.text_blocks.is_empty() {
                        tracing::info!(
                            doc_index = doc_index,
                            process_with_character = req.process_with_character,
                            text_block_count = snapshot.text_blocks.len(),
                            "LlmGenerate step"
                        );
                        let ctx = if req.process_with_character {
                            // API providers already carry the full cast in their
                            // cached story context (see llm_load), so naming who
                            // appears is enough; local models need the details.
                            let concise = res.llm.is_api().await;
                            let page_ctx = res
                                .ml
                                .scan_for_character_context_with(&snapshot.image, concise);
                            tracing::info!(
                                has_page_ctx = page_ctx.is_some(),
                                page_ctx = ?page_ctx,
                                "scan_for_character_context result"
                            );
                            let pronoun_ctx = res.ml.scan_pronoun_context(
                                &snapshot,
                                req.llm_custom_system_prompt.as_deref(),
                            );
                            tracing::info!(
                                has_pronoun_ctx = pronoun_ctx.is_some(),
                                pronoun_ctx = ?pronoun_ctx,
                                "scan_pronoun_context result"
                            );
                            match (page_ctx.as_deref(), pronoun_ctx.as_deref()) {
                                (Some(p), Some(c)) => Some(format!("{p}\n\n{c}")),
                                (Some(p), None) => Some(p.to_string()),
                                (None, Some(c)) => Some(c.to_string()),
                                (None, None) => None,
                            }
                        } else {
                            None
                        };
                        tracing::info!(has_ctx = ctx.is_some(), "sending to LLM with context");
                        crate::ops::translate_page(
                            &res.llm,
                            &mut snapshot,
                            req.language.as_deref(),
                            ctx.as_deref(),
                        )
                        .await?;
                    } else {
                        let llm_ready = res.llm.ready().await;
                        tracing::info!(
                            llm_ready,
                            text_block_count = snapshot.text_blocks.len(),
                            "LlmGenerate skipped"
                        );
                    }
                }
                PipelineStep::Inpaint => {
                    if snapshot.segment.is_some() && !snapshot.text_blocks.is_empty() {
                        res.ml.inpaint(&mut snapshot).await?;
                    }
                    tokio::task::block_in_place(koharu_ml::lama::clear_fft_plans_on_current_thread);
                }
                PipelineStep::Render => {
                    if !snapshot.text_blocks.is_empty() {
                        res.renderer.render(
                            &mut snapshot,
                            None,
                            req.shader_effect.unwrap_or_default(),
                            req.shader_stroke.clone(),
                            req.font_family.as_deref(),
                        )?;
                    }
                }
            }

            let changed = match step {
                PipelineStep::Detect => &[ChangedField::TextBlocks, ChangedField::Segment][..],
                PipelineStep::Ocr => &[ChangedField::TextBlocks][..],
                PipelineStep::DetectBalloon => &[ChangedField::Segment][..],
                PipelineStep::LlmGenerate => &[ChangedField::TextBlocks][..],
                PipelineStep::Inpaint => &[ChangedField::Inpainted][..],
                PipelineStep::Render => &[ChangedField::TextBlocks, ChangedField::Rendered][..],
            };
            let has_rendered = *step == PipelineStep::Render && snapshot.rendered.is_some();
            state_tx::update_doc(&res.state, doc_index, snapshot, changed).await?;

            if has_rendered {
                let payload = koharu_types::commands::IndexPayload { index: doc_index };
                if req.export_tiff {
                    crate::operations::save_rendered_tiff(res.clone(), payload).await?;
                } else if req.export_psd {
                    crate::operations::save_rendered_psd(res.clone(), payload).await?;
                } else {
                    crate::operations::save_rendered(res.clone(), payload).await?;
                }
            }
        }
    }

    Ok(())
}

/// Build a single-block pronoun context for retrying block `idx`.
///
/// The full batch context contains lines like:
///   `<block id="3"> speaker=... → use "tôi" for I/me, "cô" for you`
///
/// When retrying a single TextBlock, `get_source()` always emits `<block id="0">`.
/// If we pass the original context verbatim, the LLM sees id="3" in the rules but
/// id="0" in the source and echoes back id="3" — which `parse_tagged_blocks` ignores.
///
/// This function extracts the header line plus the one rule for `idx`, remapping
/// `<block id="idx">` → `<block id="0">` so both sides agree.
#[cfg(test)]
mod tests {
    use super::compute_percent;

    #[test]
    fn compute_percent_handles_zero_units() {
        assert_eq!(compute_percent(0, 0, 0, 5), 0);
        assert_eq!(compute_percent(0, 0, 2, 0), 0);
    }

    #[test]
    fn compute_percent_progresses_monotonically() {
        let total_docs = 2;
        let total_steps = 5;
        let first = compute_percent(0, 0, total_docs, total_steps);
        let middle = compute_percent(0, 3, total_docs, total_steps);
        let last = compute_percent(1, 4, total_docs, total_steps);
        assert!(first < middle);
        assert!(middle < last);
        assert_eq!(last, 90);
    }
}
