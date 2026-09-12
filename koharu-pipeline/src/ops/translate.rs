//! Shared page-translation flow.
//!
//! Both the manual "translate this page" command and the batch pipeline funnel
//! through [`translate_page`], so the token-saving rules live in exactly one
//! place:
//!
//! 1. Sound effects already in the learned dictionary never reach the model.
//! 2. Blocks with no letters (bare punctuation) pass through untranslated.
//! 3. Retries re-send **only** the blocks still missing a translation, with the
//!    per-block context narrowed to match, instead of re-paying for the page.

use koharu_llm::facade::{BlockSelection, Model, needs_translation};
use koharu_llm::sfx_dict::{is_sfx, sfx_dictionary};
use koharu_llm::Language;
use koharu_types::{BalloonDetection, Document, TextBlock};

/// How many times a batch of missing blocks is re-sent before falling back to
/// one request per block.
const MAX_BATCH_RETRIES: usize = 3;

/// Nudge for a block the model returned empty or described instead of
/// transliterating. Phrased in terms of the target language rather than a fixed
/// one, since this path also runs for non-Vietnamese output.
fn sfx_retry_hint(language: Language) -> String {
    format!(
        "IMPORTANT: This block is a sound effect / onomatopoeia. Output ONLY a short \
         {language} sound word in capitals. Do NOT write any description, \
         explanation, or parentheses."
    )
}

/// Placeholder so the renderer always has something to lay out.
const SILENCE_MARKER: &str = ".\n.\n.";

#[derive(Debug, Default, Clone, Copy)]
pub struct TranslateStats {
    /// Blocks served from the SFX dictionary — no LLM call.
    pub sfx_cache_hits: usize,
    /// New SFX translations added to the dictionary this page.
    pub sfx_learned: usize,
    /// Punctuation-only blocks passed through verbatim.
    pub passthrough: usize,
    /// Blocks actually sent to the model.
    pub translated: usize,
    /// Blocks that stayed empty after every retry.
    pub failed: usize,
}

/// Whether a block sits inside a speech balloon — the same centre-in-bbox rule
/// the speaker detection uses.
fn is_inside_balloon(block: &TextBlock, balloons: &[BalloonDetection]) -> bool {
    let cx = block.x + block.width / 2.0;
    let cy = block.y + block.height / 2.0;
    balloons.iter().any(|balloon| {
        cx >= balloon.x
            && cx <= balloon.x + balloon.width
            && cy >= balloon.y
            && cy <= balloon.y + balloon.height
    })
}

/// Whether a block's translation may be memoised in the SFX dictionary.
///
/// Katakana alone is not enough: manga sets plenty of ordinary dialogue in
/// katakana for shouting and emphasis (ハイ!, ゴメンゴメン), and freezing that
/// would pin a contextual line to one wording forever. Real sound effects are
/// drawn onto the art, outside any balloon — so that is the discriminator.
/// When balloon detection has not run there is no way to tell them apart, and
/// nothing is memoised.
fn is_memoisable_sfx(block: &TextBlock, balloons: &[BalloonDetection], source: &str) -> bool {
    !balloons.is_empty() && !is_inside_balloon(block, balloons) && is_sfx(source)
}

fn has_source(block: &TextBlock) -> bool {
    block.text.as_deref().is_some_and(|t| !t.trim().is_empty())
}

fn has_translation(block: &TextBlock) -> bool {
    block
        .translation
        .as_deref()
        .is_some_and(|t| !t.trim().is_empty())
}

/// Rewrite a per-block context so its `<block id="N">` lines match a renumbered
/// selection. Lines for blocks outside `indices` are dropped; everything else
/// (headers, character list) is kept.
fn remap_context(full_ctx: Option<&str>, indices: &[usize]) -> Option<String> {
    let ctx = full_ctx?;
    let mut out: Vec<String> = Vec::new();

    for line in ctx.lines() {
        match block_id_of(line) {
            Some(id) => {
                if let Some(position) = indices.iter().position(|index| *index == id) {
                    out.push(line.replacen(&format!("[{id}]"), &format!("[{position}]"), 1));
                }
            }
            None => out.push(line.to_string()),
        }
    }

    let remapped = out.join("\n");
    if remapped.trim().is_empty() {
        None
    } else {
        Some(remapped)
    }
}

/// The block id a context line is keyed to, if it is a per-block line.
/// Context lines carry the `[N]` marker followed by the rule for that block.
fn block_id_of(line: &str) -> Option<usize> {
    let rest = line.trim_start().strip_prefix('[')?;
    let (digits, _) = rest.split_once(']')?;
    digits.trim().parse().ok()
}

fn pending_indices(doc: &Document) -> Vec<usize> {
    doc.text_blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| has_source(block) && !has_translation(block))
        .map(|(index, _)| index)
        .collect()
}

/// Translate every block of `doc` that still needs it, as cheaply as possible.
pub async fn translate_page(
    llm: &Model,
    doc: &mut Document,
    target_language: Option<&str>,
    page_context: Option<&str>,
) -> anyhow::Result<TranslateStats> {
    let mut stats = TranslateStats::default();
    if doc.text_blocks.is_empty() {
        return Ok(stats);
    }
    let balloons = doc.balloons.clone();
    if balloons.is_empty() {
        tracing::debug!("no balloons detected, SFX dictionary disabled for this page");
    }

    let language = target_language
        .and_then(Language::parse)
        .unwrap_or(Language::English);
    let language_key = language.to_string();
    let dictionary = sfx_dictionary();

    // ── 1. Serve known SFX from the dictionary, skip letter-free blocks ───────
    for block in &mut doc.text_blocks {
        if !has_source(block) || has_translation(block) {
            continue;
        }
        let source = block.text.as_deref().unwrap_or("").trim().to_string();

        if is_memoisable_sfx(block, &balloons, &source) {
            if let Some(cached) = dictionary.get(&language_key, &source) {
                block.translation = Some(cached);
                stats.sfx_cache_hits += 1;
                continue;
            }
        }

        if !needs_translation(&source) {
            block.translation = Some(source);
            stats.passthrough += 1;
        }
    }

    // ── 2. Batch the remainder, re-sending only what is still missing ─────────
    let mut pending = pending_indices(doc);
    let to_translate = pending.len();

    for attempt in 1..=MAX_BATCH_RETRIES {
        if pending.is_empty() {
            break;
        }

        let context = remap_context(page_context, &pending);
        let mut selection = BlockSelection::from_indices(&mut doc.text_blocks, &pending);
        llm.translate_with_context(&mut selection, target_language, context.as_deref())
            .await?;

        let remaining = pending_indices(doc);
        if remaining.len() == pending.len() && attempt > 1 {
            // No forward progress — more batch attempts will not help.
            tracing::warn!(attempt, remaining = remaining.len(), "batch retry stalled");
            pending = remaining;
            break;
        }
        tracing::info!(
            attempt,
            sent = pending.len(),
            still_missing = remaining.len(),
            "batch translate"
        );
        pending = remaining;
    }

    // ── 3. One request per straggler, with an SFX hint where it applies ───────
    for index in pending {
        let source = doc.text_blocks[index].text.as_deref().unwrap_or("").trim().to_string();
        let mut context = remap_context(page_context, &[index]);
        if is_sfx(&source) {
            let hint = sfx_retry_hint(language);
            context = Some(match context {
                Some(ctx) => format!("{ctx}\n\n{hint}"),
                None => hint,
            });
        }

        llm.translate_with_context(
            &mut doc.text_blocks[index],
            target_language,
            context.as_deref(),
        )
        .await?;
    }

    // ── 4. Learn the SFX we just paid for, then backfill hopeless blocks ─────
    for block in &doc.text_blocks {
        let (Some(source), Some(translation)) = (block.text.as_deref(), block.translation.as_deref())
        else {
            continue;
        };
        if translation.trim().is_empty() || !is_memoisable_sfx(block, &balloons, source) {
            continue;
        }
        let before = dictionary.len(&language_key);
        dictionary.learn(&language_key, source, translation);
        if dictionary.len(&language_key) > before {
            stats.sfx_learned += 1;
        }
    }

    for block in &mut doc.text_blocks {
        if has_source(block) && !has_translation(block) {
            block.translation = Some(SILENCE_MARKER.to_string());
            stats.failed += 1;
        }
    }

    stats.translated = to_translate.saturating_sub(stats.failed);
    tracing::info!(
        sfx_cache_hits = stats.sfx_cache_hits,
        sfx_learned = stats.sfx_learned,
        passthrough = stats.passthrough,
        translated = stats.translated,
        failed = stats.failed,
        "page translated"
    );

    Ok(stats)
}

/// Translate a single block, honouring the SFX dictionary and narrowing the
/// per-block context to that block. `document_index` is the block's position in
/// the page, used to pick its line out of `page_context`.
pub async fn translate_block(
    llm: &Model,
    block: &mut TextBlock,
    document_index: usize,
    balloons: &[BalloonDetection],
    target_language: Option<&str>,
    page_context: Option<&str>,
) -> anyhow::Result<()> {
    let language = target_language
        .and_then(Language::parse)
        .unwrap_or(Language::English);
    let language_key = language.to_string();
    let dictionary = sfx_dictionary();

    let source = block.text.as_deref().unwrap_or("").trim().to_string();
    if source.is_empty() {
        return Ok(());
    }

    let memoisable = is_memoisable_sfx(block, balloons, &source);
    if memoisable {
        if let Some(cached) = dictionary.get(&language_key, &source) {
            tracing::info!(source = %source, "sfx served from dictionary");
            block.translation = Some(cached);
            return Ok(());
        }
    }

    if !needs_translation(&source) {
        block.translation = Some(source);
        return Ok(());
    }

    let mut context = remap_context(page_context, &[document_index]);
    if is_sfx(&source) {
        let hint = sfx_retry_hint(language);
        context = Some(match context {
            Some(ctx) => format!("{ctx}\n\n{hint}"),
            None => hint,
        });
    }

    llm.translate_with_context(block, target_language, context.as_deref())
        .await?;

    if memoisable {
        if let Some(translation) = block.translation.as_deref() {
            dictionary.learn(&language_key, &source, translation);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_context_renumbers_selected_blocks_and_keeps_headers() {
        let ctx = "Characters in this scene: Saeba, Kaori.\n\
                   [0] speaker=Saeba\n\
                   [3] speaker=Kaori\n\
                   [5] speaker=Unknown";

        let remapped = remap_context(Some(ctx), &[3, 5]).unwrap();
        assert_eq!(
            remapped,
            "Characters in this scene: Saeba, Kaori.\n\
             [0] speaker=Kaori\n\
             [1] speaker=Unknown"
        );
    }

    #[test]
    fn remap_context_drops_unselected_block_lines() {
        let ctx = "[0] a\n[1] b";
        assert_eq!(remap_context(Some(ctx), &[1]).unwrap(), "[0] b");
        assert!(remap_context(Some(ctx), &[]).is_none());
        assert!(remap_context(None, &[0]).is_none());
    }

    #[test]
    fn block_id_parsing_ignores_ordinary_lines() {
        assert_eq!(block_id_of("[12] x"), Some(12));
        assert_eq!(block_id_of("  [7] x"), Some(7));
        assert_eq!(block_id_of("Characters in scene: A"), None);
        assert_eq!(block_id_of("[abc] x"), None);
    }

    fn balloon(x: f32, y: f32, w: f32, h: f32) -> BalloonDetection {
        BalloonDetection { x, y, width: w, height: h, score: 0.9 }
    }

    fn block_at(x: f32, y: f32, text: &str) -> TextBlock {
        TextBlock {
            x,
            y,
            width: 10.0,
            height: 10.0,
            text: Some(text.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn katakana_dialogue_inside_a_balloon_is_not_memoisable() {
        let balloons = vec![balloon(0.0, 0.0, 100.0, 100.0)];
        // Shouted dialogue set in katakana, inside a balloon: contextual, must
        // never be frozen into the dictionary.
        let inside = block_at(20.0, 20.0, "ハイ!");
        assert!(!is_memoisable_sfx(&inside, &balloons, "ハイ!"));

        // The same katakana drawn onto the art, outside every balloon, is an SFX.
        let outside = block_at(200.0, 200.0, "ドン");
        assert!(is_memoisable_sfx(&outside, &balloons, "ドン"));
    }

    #[test]
    fn nothing_is_memoisable_without_balloon_detection() {
        let outside = block_at(200.0, 200.0, "ドン");
        assert!(!is_memoisable_sfx(&outside, &[], "ドン"));
    }

    #[test]
    fn balloon_containment_uses_block_centre() {
        let balloons = vec![balloon(0.0, 0.0, 30.0, 30.0)];
        // Centre (25,25) is inside.
        assert!(is_inside_balloon(&block_at(20.0, 20.0, "x"), &balloons));
        // Centre (35,35) is outside even though the box overlaps the balloon.
        assert!(!is_inside_balloon(&block_at(30.0, 30.0, "x"), &balloons));
    }

    #[test]
    fn pending_skips_blocks_without_source_or_already_translated() {
        let doc = Document {
            text_blocks: vec![
                TextBlock { text: Some("あ".into()), ..Default::default() },
                TextBlock { text: Some("い".into()), translation: Some("b".into()), ..Default::default() },
                TextBlock { text: None, ..Default::default() },
                TextBlock { text: Some("   ".into()), ..Default::default() },
            ],
            ..Default::default()
        };
        assert_eq!(pending_indices(&doc), vec![0]);
    }
}

/// Visual sample of the layout pipeline, for judging text fit by eye.
///
/// Runs detect → balloon refit → inpaint → render on a page, substituting
/// Vietnamese lines of realistic length for the translation, so the geometry
/// and font sizing can be inspected without spending an API call.
///
/// Run with: `KOHARU_TEST_PAGE=page.jpg KOHARU_TEST_OUT=out.png
/// cargo test -p koharu-pipeline --lib render_layout_sample -- --ignored --nocapture`
#[cfg(test)]
mod render_sample {
    /// Lines of the length a Vietnamese translation actually comes out at.
    const LINES: &[&str] = &[
        "THÌ TA ĐÁP TRẢ BẰNG CHIÊU NÀY!",
        "THẾ NẾU NÓ CHƠI CHIÊU NÀY THÌ SAO?",
        "Ồ!",
        "MÀY CHƠI XẤU, DÙNG HUNG KHÍ HẢ!",
        "KHÔNG ĐỜI NÀO!",
        "BIẾT ĐÂU NÓ LẠI GIỞ TRÒ NÀY RA THÌ SAO.",
        "NHƯNG MÀ NÓ HUNG DỮ THẬT ĐẤY!",
        "TIẾP THEO LÀ NGƯỜI NGOÀI HÀNH TINH SHEIK!",
        "MÀ NÀY, KINNIKUMAN!",
        "ĐỒ NGU À!!",
        "SỢ... SỢ QUÁ ĐI MẤT...",
        "CHẮC CHẮN LÀ ĐỒ VẬT ĐẾN TỪ TRUNG ĐÔNG RỒI!",
    ];

    /// Full pipeline on a real page: OCR the Japanese, translate it with
    /// Gemini, inpaint, render. Writes the finished page next to the input.
    ///
    /// `KOHARU_TEST_PAGE=page.jpg KOHARU_TEST_OUT=out.png cargo test --release
    /// -p koharu-pipeline --lib translate_and_render_sample -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn translate_and_render_sample() -> anyhow::Result<()> {
        let Some(page) = std::env::var_os("KOHARU_TEST_PAGE") else {
            anyhow::bail!("set KOHARU_TEST_PAGE");
        };
        let out = std::env::var_os("KOHARU_TEST_OUT")
            .ok_or_else(|| anyhow::anyhow!("set KOHARU_TEST_OUT"))?;

        let runtime = tokio::runtime::Runtime::new()?;
        koharu_llm::sys::initialize()?;
        let backend = std::sync::Arc::new(koharu_llm::safe::llama_backend::LlamaBackend::init()?);
        let ml = runtime.block_on(koharu_ml::facade::Model::new(false, backend.clone()))?;

        let mut doc = koharu_types::Document::open(std::path::PathBuf::from(&page))?;
        runtime.block_on(ml.detect(&mut doc))?;
        runtime.block_on(ml.ocr(&mut doc))?;
        runtime.block_on(ml.detect_balloons(&mut doc))?;

        if let (Some(seg), Some(dir)) = (doc.segment.as_ref(), std::env::var_os("KOHARU_TEST_SEG")) {
            let mask = seg.0.to_luma8();
            println!(
                "segment tại (1162,1010) = {}  (255 nghĩa là mask đã đánh dấu chữ)",
                mask.get_pixel(1162, 1010)[0]
            );
            mask.save(std::path::PathBuf::from(dir))?;
        }

        let llm = koharu_llm::facade::Model::new(false, backend);
        runtime.block_on(llm.load_api(
            "gemini",
            "gemini-3.1-flash-lite-preview",
            koharu_llm::providers::ProviderConfig {
                api_key: None,
                base_url: None,
                temperature: None,
                max_tokens: None,
                custom_system_prompt: None,
                story_context: None,
                key_start_index: None,
            },
        ))?;
        let stats = runtime.block_on(super::translate_page(
            &llm,
            &mut doc,
            Some("vi-VN"),
            None,
        ))?;
        println!("{stats:?}");

        runtime.block_on(ml.inpaint(&mut doc))?;

        let renderer = koharu_renderer::facade::Renderer::new()?;
        renderer.render(&mut doc, None, Default::default(), None, None)?;

        for (i, block) in doc.text_blocks.iter().enumerate() {
            println!(
                "{i:>3} box {:>3.0}x{:<3.0} font {:>5.1} src {:>4.0}\n    ja | {}\n    vi | {}",
                block.width,
                block.height,
                block.style.as_ref().and_then(|s| s.font_size).unwrap_or(0.0),
                block.detected_font_size_px.unwrap_or(0.0),
                block.text.as_deref().unwrap_or("").replace('\n', " "),
                block.translation.as_deref().unwrap_or("").replace('\n', " ")
            );
        }

        doc.rendered
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("nothing rendered"))?
            .0
            .save(std::path::PathBuf::from(out))?;
        Ok(())
    }

    #[test]
    #[ignore]
    fn render_layout_sample() -> anyhow::Result<()> {
        let Some(page) = std::env::var_os("KOHARU_TEST_PAGE") else {
            anyhow::bail!("set KOHARU_TEST_PAGE");
        };
        let out = std::env::var_os("KOHARU_TEST_OUT")
            .ok_or_else(|| anyhow::anyhow!("set KOHARU_TEST_OUT"))?;

        let runtime = tokio::runtime::Runtime::new()?;
        // The OCR model inside koharu-ml is llama.cpp-backed, so the runtime
        // libraries have to be loaded before any backend is created.
        koharu_llm::sys::initialize()?;
        let backend = std::sync::Arc::new(koharu_llm::safe::llama_backend::LlamaBackend::init()?);
        let ml = runtime.block_on(koharu_ml::facade::Model::new(true, backend))?;

        let mut doc = koharu_types::Document::open(std::path::PathBuf::from(&page))?;
        runtime.block_on(ml.detect(&mut doc))?;
        runtime.block_on(ml.detect_balloons(&mut doc))?;

        for (i, block) in doc.text_blocks.iter_mut().enumerate() {
            block.text = Some("ダミー".to_string());
            block.translation = Some(LINES[i % LINES.len()].to_string());
        }

        runtime.block_on(ml.inpaint(&mut doc))?;

        let renderer = koharu_renderer::facade::Renderer::new()?;
        renderer.render(&mut doc, None, Default::default(), None, None)?;

        for (i, block) in doc.text_blocks.iter().enumerate() {
            println!(
                "{i:>3} box {:.0}x{:.0} font {:?}",
                block.width,
                block.height,
                block.style.as_ref().and_then(|s| s.font_size)
            );
        }

        let rendered = doc
            .rendered
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("nothing rendered"))?;
        rendered.0.save(std::path::PathBuf::from(out))?;
        Ok(())
    }
}
