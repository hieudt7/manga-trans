use std::sync::Arc;

use serde::Serialize;
use tokio::sync::{RwLock, broadcast};

use koharu_types::{Document, LlmState, LlmStateStatus, TextBlock};

use crate::{
    GenerateOptions, Language, Llm, ModelId, language::tags as language_tags,
    safe::llama_backend::LlamaBackend, supported_locales,
};

pub use crate::prefetch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockStartTag {
    offset: usize,
    len: usize,
    id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockEndTag {
    offset: usize,
    len: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub languages: Vec<String>,
    pub source: &'static str,
}

impl ModelInfo {
    pub fn new(id: ModelId) -> Self {
        let languages = id.languages();
        Self {
            id: id.to_string(),
            languages: language_tags(&languages),
            source: "local",
        }
    }

    pub fn api(provider_id: &'static str, model_id: &str) -> Self {
        Self {
            id: format!("{provider_id}:{model_id}"),
            languages: supported_locales(),
            source: provider_id,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(strum::Display)]
pub enum State {
    #[strum(serialize = "empty")]
    Empty,
    #[strum(serialize = "loading")]
    Loading { model_id: String, source: String },
    #[strum(serialize = "ready")]
    Ready(Llm),
    #[strum(serialize = "ready")]
    ApiReady {
        provider: Box<dyn crate::providers::AnyProvider>,
        provider_id: String,
        model: String,
    },
    #[strum(serialize = "failed")]
    Failed(String),
}

pub struct Model {
    state: Arc<RwLock<State>>,
    state_tx: broadcast::Sender<LlmState>,
    cpu: bool,
    backend: Arc<LlamaBackend>,
}

pub trait Translatable {
    fn get_source(&self) -> anyhow::Result<String>;
    fn set_translation(&mut self, translation: String) -> anyhow::Result<()>;
}

fn escape_block_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn unescape_block_text(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn strip_wrapping_quotes(text: &str) -> String {
    let mut current = text.trim();

    loop {
        let next = match current {
            _ if current.starts_with('"') && current.ends_with('"') => {
                current.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
            }
            _ if current.starts_with('\'') && current.ends_with('\'') => current
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\'')),
            _ if current.starts_with('“') && current.ends_with('”') => {
                current.strip_prefix('“').and_then(|s| s.strip_suffix('”'))
            }
            _ if current.starts_with('‘') && current.ends_with('’') => {
                current.strip_prefix('‘').and_then(|s| s.strip_suffix('’'))
            }
            _ => break,
        };
        let Some(next) = next else {
            break;
        };
        current = next.trim();
    }

    strip_incomplete_corner_quotes(current)
}

/// Strip parenthetical SFX descriptions like "(Tiếng búa đập)" that LLMs
/// sometimes produce instead of translating the onomatopoeia directly.
/// If the ENTIRE translation is a parenthetical → strip the parens, leaving
/// the inner text (e.g. "Tiếng búa đập") which is still more useful than nothing.
pub fn is_sfx_description(text: &str) -> bool {
    let t = text.trim();
    t.starts_with('(') && t.ends_with(')') && !t[1..t.len()-1].contains('(')
}

fn strip_sfx_description(text: &str) -> String {
    if is_sfx_description(text) {
        tracing::warn!(original = %text.trim(), "discarded SFX parenthetical description");
        return String::new();
    }
    text.to_string()
}

/// Strip "Speaker name: " prefix that LLMs sometimes add despite instructions.
/// Only strips if the prefix is a short name-like string (no newlines, no block tags).
fn strip_speaker_prefix(text: &str) -> String {
    // Try each line — prefix may only be on the first line.
    if let Some(colon_pos) = text.find(": ") {
        let prefix = &text[..colon_pos];
        // Valid name prefix: short, no newline, no XML/angle brackets, no digits at start.
        let looks_like_name = prefix.len() <= 50
            && !prefix.contains('\n')
            && !prefix.contains('<')
            && !prefix.contains('>')
            && !prefix.trim_start().starts_with(|c: char| c.is_ascii_digit());
        if looks_like_name {
            let stripped = text[colon_pos + 2..].trim().to_string();
            tracing::warn!(prefix = %prefix, "stripped speaker prefix from translation");
            return stripped;
        }
    }
    text.to_string()
}

fn strip_incomplete_corner_quotes(text: &str) -> String {
    let mut current = text.trim();

    loop {
        let open_count = current.chars().filter(|&c| c == '「').count();
        let close_count = current.chars().filter(|&c| c == '」').count();

        if open_count > close_count && current.starts_with('「') {
            current = current.trim_start_matches('「').trim_start();
            continue;
        }

        if close_count > open_count && current.ends_with('」') {
            current = current.trim_end_matches('」').trim_end();
            continue;
        }

        break;
    }

    current.to_string()
}

fn format_document_blocks(blocks: &[TextBlock]) -> String {
    blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| {
            let text = block.text.as_deref().unwrap_or("<empty>");
            format!(
                r#"<block id="{idx}">
{}
</block>"#,
                escape_block_text(text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_tagged_blocks(
    translation: &str,
    expected_blocks: usize,
) -> anyhow::Result<Option<Vec<String>>> {
    if find_next_block_start_tag(translation).is_none() {
        return Ok(None);
    }

    let mut blocks = vec![String::new(); expected_blocks];
    let mut cursor = translation;
    let mut parsed_count = 0usize;

    while let Some(start_tag) = find_next_block_start_tag(cursor) {
        cursor = &cursor[start_tag.offset + start_tag.len..];

        let id = start_tag.id;
        let closing_tag = find_next_block_end_tag(cursor);
        let block_end = block_boundary(cursor, closing_tag.map(|tag| tag.offset));
        let content = unescape_block_text(cursor[..block_end].trim());

        cursor = if closing_tag.map(|tag| tag.offset) == Some(block_end) {
            let closing_len = closing_tag.map(|tag| tag.len).unwrap_or(0);
            &cursor[block_end + closing_len..]
        } else {
            &cursor[block_end..]
        };

        if id >= expected_blocks {
            tracing::warn!(id, expected_blocks, "ignoring out-of-range block id");
            continue;
        }

        if blocks[id].is_empty() {
            parsed_count += 1;
        }
        blocks[id] = content;
    }

    if parsed_count < expected_blocks {
        tracing::warn!(parsed_count, expected_blocks, "Gemini returned fewer blocks than expected");
    }

    Ok(Some(blocks))
}

/// Parse numbered-list responses that some API providers return instead of block tags:
///
/// ```text
/// 0
/// Translation for block 0.
///
/// 1
/// Translation for block 1.
/// ```
///
/// Returns `None` if the text doesn't look like numbered list format.
fn parse_numbered_list_blocks(translation: &str, expected_blocks: usize) -> Option<Vec<String>> {
    let mut blocks: Vec<Option<String>> = vec![None; expected_blocks];
    let mut last_id: Option<usize> = None;
    let mut found_count = 0usize;

    for section in translation.split("\n\n") {
        let section_trimmed = section.trim();
        if section_trimmed.is_empty() {
            continue;
        }

        let mut lines = section_trimmed.lines();
        let first_line = lines.next().unwrap_or("").trim();
        let rest_content: String = lines.collect::<Vec<_>>().join("\n");

        if let Ok(id) = first_line.parse::<usize>() {
            if id < expected_blocks {
                let content = rest_content.trim().to_string();
                if blocks[id].is_none() {
                    found_count += 1;
                }
                blocks[id] = Some(content);
                last_id = Some(id);
            } else {
                // Number out of range — treat as continuation of the previous block
                if let Some(prev_id) = last_id {
                    if let Some(ref mut existing) = blocks[prev_id] {
                        existing.push_str("\n\n");
                        existing.push_str(section_trimmed);
                    }
                } else {
                    return None;
                }
            }
        } else {
            // First line is not a number
            if let Some(prev_id) = last_id {
                // Paragraph break within a translation
                if let Some(ref mut existing) = blocks[prev_id] {
                    if !existing.is_empty() {
                        existing.push_str("\n\n");
                    }
                    existing.push_str(section_trimmed);
                }
            } else {
                // No numbered block seen yet — not numbered list format
                return None;
            }
        }
    }

    if found_count == 0 {
        return None;
    }

    if found_count < expected_blocks {
        tracing::warn!(found_count, expected_blocks, "numbered list: fewer blocks than expected");
    }

    Some(blocks.into_iter().map(|b| b.unwrap_or_default()).collect())
}

fn split_legacy_lines(translation: &str, expected_blocks: usize) -> anyhow::Result<Vec<String>> {
    let mut translations = translation
        .lines()
        .map(|line| line.trim_end_matches('\r').to_string())
        .collect::<Vec<_>>();

    if translations.len() != expected_blocks {
        tracing::warn!(
            "Translated line count mismatch: expected {expected_blocks}, got {}",
            translations.len()
        );
    }

    translations.truncate(expected_blocks);
    while translations.len() < expected_blocks {
        translations.push(String::new());
    }

    Ok(translations)
}

fn block_boundary(cursor: &str, closing_tag: Option<usize>) -> usize {
    let next_block_start = find_next_block_start_tag(cursor).map(|tag| tag.offset);
    match (closing_tag, next_block_start) {
        (Some(close), Some(next)) => close.min(next),
        (Some(close), None) => close,
        (None, Some(next)) => next,
        (None, None) => cursor.len(),
    }
}

fn find_next_block_start_tag(text: &str) -> Option<BlockStartTag> {
    let mut search_from = 0usize;
    while let Some(rel_start) = text[search_from..].find('<') {
        let offset = search_from + rel_start;
        if let Some((len, id)) = parse_block_start_tag(&text[offset..]) {
            return Some(BlockStartTag { offset, len, id });
        }
        search_from = offset + 1;
    }
    None
}

fn parse_block_start_tag(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    if bytes.first().copied()? != b'<' {
        return None;
    }

    let mut index = 1usize;
    skip_ascii_whitespace(bytes, &mut index);
    if !consume_ascii_keyword(bytes, &mut index, "block") {
        return None;
    }

    let mut parsed_id = None;
    loop {
        skip_ascii_whitespace(bytes, &mut index);
        match bytes.get(index).copied()? {
            b'>' => return parsed_id.map(|id| (index + 1, id)),
            b'/' if bytes.get(index + 1).copied() == Some(b'>') => {
                return parsed_id.map(|id| (index + 2, id));
            }
            _ => {}
        }

        let name_start = index;
        while matches!(
            bytes.get(index).copied(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-')
        ) {
            index += 1;
        }
        if index == name_start {
            return None;
        }
        let attr_name = &text[name_start..index];

        skip_ascii_whitespace(bytes, &mut index);
        if bytes.get(index).copied()? != b'=' {
            return None;
        }
        index += 1;
        skip_ascii_whitespace(bytes, &mut index);

        let attr_value = match bytes.get(index).copied()? {
            b'"' | b'\'' => {
                let quote = bytes[index];
                index += 1;
                let value_start = index;
                while bytes.get(index).copied()? != quote {
                    index += 1;
                }
                let value = &text[value_start..index];
                index += 1;
                value
            }
            _ => {
                let value_start = index;
                while matches!(bytes.get(index).copied(), Some(byte) if !byte.is_ascii_whitespace() && byte != b'>')
                {
                    index += 1;
                }
                &text[value_start..index]
            }
        };

        if attr_name.eq_ignore_ascii_case("id") {
            parsed_id = attr_value.parse::<usize>().ok();
        }
    }
}

fn find_next_block_end_tag(text: &str) -> Option<BlockEndTag> {
    let mut search_from = 0usize;
    while let Some(rel_start) = text[search_from..].find('<') {
        let offset = search_from + rel_start;
        if let Some(len) = parse_block_end_tag(&text[offset..]) {
            return Some(BlockEndTag { offset, len });
        }
        search_from = offset + 1;
    }
    None
}

fn parse_block_end_tag(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.first().copied()? != b'<' {
        return None;
    }

    let mut index = 1usize;
    skip_ascii_whitespace(bytes, &mut index);
    if bytes.get(index).copied()? != b'/' {
        return None;
    }
    index += 1;
    skip_ascii_whitespace(bytes, &mut index);
    if !consume_ascii_keyword(bytes, &mut index, "block") {
        return None;
    }
    skip_ascii_whitespace(bytes, &mut index);
    if bytes.get(index).copied()? != b'>' {
        return None;
    }
    Some(index + 1)
}

fn skip_ascii_whitespace(bytes: &[u8], index: &mut usize) {
    while matches!(bytes.get(*index).copied(), Some(byte) if byte.is_ascii_whitespace()) {
        *index += 1;
    }
}

fn consume_ascii_keyword(bytes: &[u8], index: &mut usize, keyword: &str) -> bool {
    let end = *index + keyword.len();
    let Some(slice) = bytes.get(*index..end) else {
        return false;
    };
    if !slice.eq_ignore_ascii_case(keyword.as_bytes()) {
        return false;
    }
    *index = end;
    true
}

impl Translatable for Document {
    fn get_source(&self) -> anyhow::Result<String> {
        Ok(format_document_blocks(&self.text_blocks))
    }

    fn set_translation(&mut self, translation: String) -> anyhow::Result<()> {
        let expected = self.text_blocks.len();
        let translations = match parse_tagged_blocks(&translation, expected)? {
            Some(blocks) => blocks,
            None => {
                if expected == 1 {
                    split_legacy_lines(&translation, 1)?
                } else if let Some(blocks) = parse_numbered_list_blocks(&translation, expected) {
                    tracing::debug!(expected, "parsed numbered list blocks from LLM response");
                    blocks
                } else {
                    tracing::warn!(
                        expected,
                        "LLM response had no block tags, will retry each block individually"
                    );
                    return Ok(());
                }
            }
        };

        for (block, trans) in self.text_blocks.iter_mut().zip(translations) {
            let clean = strip_sfx_description(&strip_speaker_prefix(&strip_wrapping_quotes(&trans)));
            block.translation = Some(clean);
        }
        Ok(())
    }
}

impl Translatable for TextBlock {
    fn get_source(&self) -> anyhow::Result<String> {
        let source = self
            .text
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No source text found"))?;
        Ok(format!(
            r#"<block id="0">
{}
</block>"#,
            escape_block_text(&source)
        ))
    }

    fn set_translation(&mut self, translation: String) -> anyhow::Result<()> {
        let translation = match parse_tagged_blocks(&translation, 1)? {
            Some(blocks) => blocks.into_iter().next().unwrap_or_default(),
            None => translation,
        };
        self.translation = Some(strip_sfx_description(&strip_speaker_prefix(&strip_wrapping_quotes(&translation))));
        Ok(())
    }
}

impl Model {
    pub fn new(cpu: bool, backend: Arc<LlamaBackend>) -> Self {
        Self {
            state: Arc::new(RwLock::new(State::Empty)),
            state_tx: broadcast::channel(64).0,
            cpu,
            backend,
        }
    }

    pub fn is_cpu(&self) -> bool {
        self.cpu
    }

    pub async fn load_api(
        &self,
        provider_id: &str,
        model_id: &str,
        config: crate::providers::ProviderConfig,
    ) -> anyhow::Result<()> {
        let provider = crate::providers::build_provider(provider_id, config)?;
        *self.state.write().await = State::ApiReady {
            provider,
            provider_id: provider_id.to_string(),
            model: model_id.to_string(),
        };
        self.emit_state().await;
        Ok(())
    }

    pub async fn load(&self, id: ModelId) {
        {
            let mut guard = self.state.write().await;
            *guard = State::Loading {
                model_id: id.to_string(),
                source: "local".to_string(),
            };
        }
        self.emit_state().await;

        let state_cloned = self.state.clone();
        let state_tx = self.state_tx.clone();
        let cpu = self.cpu;
        let backend = self.backend.clone();
        tokio::spawn(async move {
            let res = Llm::load(id, cpu, backend).await;
            match res {
                Ok(llm) => {
                    let mut guard = state_cloned.write().await;
                    *guard = State::Ready(llm);
                }
                Err(e) => {
                    tracing::error!("LLM load join error: {e}");
                    let mut guard = state_cloned.write().await;
                    *guard = State::Failed(format!("join error: {e}"));
                }
            }
            let snapshot = {
                let guard = state_cloned.read().await;
                snapshot_from_state(&guard)
            };
            let _ = state_tx.send(snapshot);
        });
    }

    pub async fn get(&self) -> tokio::sync::RwLockReadGuard<'_, State> {
        self.state.read().await
    }

    pub async fn get_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, State> {
        self.state.write().await
    }

    pub async fn offload(&self) {
        *self.state.write().await = State::Empty;
        self.emit_state().await;
    }

    pub async fn ready(&self) -> bool {
        matches!(
            *self.state.read().await,
            State::Ready(_) | State::ApiReady { .. }
        )
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LlmState> {
        self.state_tx.subscribe()
    }

    pub async fn snapshot(&self) -> LlmState {
        let guard = self.state.read().await;
        snapshot_from_state(&guard)
    }

    async fn emit_state(&self) {
        let _ = self.state_tx.send(self.snapshot().await);
    }

    pub async fn translate(
        &self,
        doc: &mut impl Translatable,
        target_language: Option<&str>,
    ) -> anyhow::Result<()> {
        self.translate_with_context(doc, target_language, None).await
    }

    /// Like `translate`, but injects `page_context` into the system prompt for
    /// API providers. For local (llama.cpp) models the context is currently ignored.
    pub async fn translate_with_context(
        &self,
        doc: &mut impl Translatable,
        target_language: Option<&str>,
        page_context: Option<&str>,
    ) -> anyhow::Result<()> {
        let target_language = target_language
            .and_then(Language::parse)
            .unwrap_or(Language::English);
        let source = doc.get_source()?;
        block_debug_write(&format!("=== SEND ===\n{source}\n"));
        let mut guard = self.state.write().await;
        let translation = match &mut *guard {
            State::Ready(llm) => {
                llm.generate(&source, &GenerateOptions {
                    story_context: page_context.map(str::to_owned),
                    ..GenerateOptions::default()
                }, target_language)
            }
            State::ApiReady {
                provider, model, ..
            } => {
                let model = model.clone();
                provider
                    .translate(&source, target_language, page_context, &model)
                    .await
            }
            State::Loading { .. } => Err(anyhow::anyhow!("Model is still loading")),
            State::Failed(e) => Err(anyhow::anyhow!("Model failed to load: {e}")),
            State::Empty => Err(anyhow::anyhow!("No model is loaded")),
        }?;
        let trimmed = translation.trim().to_string();
        block_debug_write(&format!("=== RECV ===\n{trimmed}\n"));
        doc.set_translation(trimmed)
    }
}

fn snapshot_from_state(state: &State) -> LlmState {
    match state {
        State::Empty => LlmState {
            status: LlmStateStatus::Empty,
            model_id: None,
            source: None,
            error: None,
        },
        State::Loading { model_id, source } => LlmState {
            status: LlmStateStatus::Loading,
            model_id: Some(model_id.clone()),
            source: Some(source.clone()),
            error: None,
        },
        State::Ready(llm) => LlmState {
            status: LlmStateStatus::Ready,
            model_id: Some(llm.id().to_string()),
            source: Some("local".to_string()),
            error: None,
        },
        State::ApiReady {
            provider_id, model, ..
        } => LlmState {
            status: LlmStateStatus::Ready,
            model_id: Some(format!("{provider_id}:{model}")),
            source: Some(provider_id.clone()),
            error: None,
        },
        State::Failed(error) => LlmState {
            status: LlmStateStatus::Failed,
            model_id: None,
            source: None,
            error: Some(error.clone()),
        },
    }
}

fn block_debug_write(content: &str) {
    use std::io::Write;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| {
            let s = d.as_secs();
            format!("{:02}:{:02}:{:02}", (s / 3600) % 24, (s / 60) % 60, s % 60)
        })
        .unwrap_or_default();
    let debug_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("debug-scan/debug_block.txt");
    let _ = std::fs::create_dir_all(debug_path.parent().unwrap());
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&debug_path) {
        let _ = writeln!(f, "[{ts}] {content}");
    }
}

#[cfg(test)]
mod tests {
    use koharu_types::Document;

    use super::*;

    #[test]
    fn document_source_uses_tagged_blocks() -> anyhow::Result<()> {
        let doc = Document {
            text_blocks: vec![
                TextBlock {
                    text: Some("Hello".to_string()),
                    ..Default::default()
                },
                TextBlock {
                    text: Some("1 < 2\nA & B".to_string()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let source = doc.get_source()?;
        assert_eq!(
            source,
            "<block id=\"0\">\nHello\n</block>\n<block id=\"1\">\n1 &lt; 2\nA &amp; B\n</block>"
        );

        Ok(())
    }

    #[test]
    fn document_translation_parses_tagged_blocks_by_id() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default(), TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation(
            "<block id=\"1\">\nSecond line\nnext\n</block>\n<block id=\"0\">\nFirst &lt;done&gt;\n</block>".to_string(),
        )?;

        assert_eq!(
            doc.text_blocks[0].translation.as_deref(),
            Some("First <done>")
        );
        assert_eq!(
            doc.text_blocks[1].translation.as_deref(),
            Some("Second line\nnext")
        );

        Ok(())
    }

    #[test]
    fn document_translation_strips_wrapping_quotes() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default(), TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation(
            "<block id=\"0\">\n\"Hello\"\n</block>\n<block id=\"1\">\n“World”\n</block>"
                .to_string(),
        )?;

        assert_eq!(doc.text_blocks[0].translation.as_deref(), Some("Hello"));
        assert_eq!(doc.text_blocks[1].translation.as_deref(), Some("World"));

        Ok(())
    }

    #[test]
    fn document_translation_ignores_no_tag_response_for_multi_block() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default(), TextBlock::default()],
            ..Default::default()
        };

        // No block tags in response → both blocks stay None (retry will handle them)
        doc.set_translation("only one line".to_string())?;
        assert_eq!(doc.text_blocks[0].translation, None);
        assert_eq!(doc.text_blocks[1].translation, None);

        Ok(())
    }

    #[test]
    fn document_translation_allows_missing_closing_tags() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default(), TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation(
            "<block id=\"0\">\nFirst line\n<block id=\"1\">\nSecond line".to_string(),
        )?;

        assert_eq!(
            doc.text_blocks[0].translation.as_deref(),
            Some("First line")
        );
        assert_eq!(
            doc.text_blocks[1].translation.as_deref(),
            Some("Second line")
        );

        Ok(())
    }

    #[test]
    fn document_translation_uses_end_of_text_when_last_closing_tag_is_missing() -> anyhow::Result<()>
    {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation("<block id=\"0\">\nFinal line".to_string())?;

        assert_eq!(
            doc.text_blocks[0].translation.as_deref(),
            Some("Final line")
        );

        Ok(())
    }

    #[test]
    fn document_translation_ignores_out_of_range_tagged_blocks() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation(
            "<block id=\"0\">\nKept\n</block>\n<block id=\"1\">\nIgnored\n</block>".to_string(),
        )?;

        assert_eq!(doc.text_blocks[0].translation.as_deref(), Some("Kept"));

        Ok(())
    }

    #[test]
    fn document_translation_accepts_relaxed_block_tag_formatting() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default(), TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation(
            "<block id = '1' >\nSecond\n</ block>\n<Block id=0>\nFirst\n</BLOCK>".to_string(),
        )?;

        assert_eq!(doc.text_blocks[0].translation.as_deref(), Some("First"));
        assert_eq!(doc.text_blocks[1].translation.as_deref(), Some("Second"));

        Ok(())
    }

    #[test]
    fn document_translation_accepts_unquoted_block_ids() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation("<block id=0>\nOnly first\n</block>".to_string())?;

        assert_eq!(
            doc.text_blocks[0].translation.as_deref(),
            Some("Only first")
        );

        Ok(())
    }

    #[test]
    fn document_translation_pads_missing_tagged_blocks() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default(), TextBlock::default()],
            ..Default::default()
        };

        doc.set_translation("<block id=\"0\">\nOnly first\n</block>".to_string())?;

        assert_eq!(
            doc.text_blocks[0].translation.as_deref(),
            Some("Only first")
        );
        assert_eq!(doc.text_blocks[1].translation.as_deref(), Some(""));

        Ok(())
    }

    #[test]
    fn document_translation_parses_numbered_list_format() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![
                TextBlock { text: Some("A".to_string()), ..Default::default() },
                TextBlock { text: Some("B".to_string()), ..Default::default() },
                TextBlock { text: Some("C".to_string()), ..Default::default() },
            ],
            ..Default::default()
        };

        doc.set_translation(
            "0\nFirst translation\n\n1\nSecond translation\n\n2\nThird translation".to_string(),
        )?;

        assert_eq!(doc.text_blocks[0].translation.as_deref(), Some("First translation"));
        assert_eq!(doc.text_blocks[1].translation.as_deref(), Some("Second translation"));
        assert_eq!(doc.text_blocks[2].translation.as_deref(), Some("Third translation"));
        Ok(())
    }

    #[test]
    fn document_translation_parses_numbered_list_with_single_char_block() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![
                TextBlock { text: Some("Hello".to_string()), ..Default::default() },
                TextBlock { text: Some("?".to_string()), ..Default::default() },
                TextBlock { text: Some("END".to_string()), ..Default::default() },
            ],
            ..Default::default()
        };

        // Mirrors Gemini returning "?" as a valid 1-char translation
        doc.set_translation("0\nXin chào\n\n1\n?\n\n2\nKết thúc".to_string())?;

        assert_eq!(doc.text_blocks[0].translation.as_deref(), Some("Xin chào"));
        assert_eq!(doc.text_blocks[1].translation.as_deref(), Some("?"));
        assert_eq!(doc.text_blocks[2].translation.as_deref(), Some("Kết thúc"));
        Ok(())
    }

    #[test]
    fn text_block_translation_strips_wrapping_quotes() -> anyhow::Result<()> {
        let mut block = TextBlock::default();
        block.set_translation("“quoted”".to_string())?;
        assert_eq!(block.translation.as_deref(), Some("quoted"));
        Ok(())
    }

    #[test]
    fn text_block_source_uses_single_tagged_block() -> anyhow::Result<()> {
        let block = TextBlock {
            text: Some("1 < 2\nA & B".to_string()),
            ..Default::default()
        };

        let source = block.get_source()?;
        assert_eq!(source, "<block id=\"0\">\n1 &lt; 2\nA &amp; B\n</block>");

        Ok(())
    }

    #[test]
    fn text_block_translation_extracts_tagged_block_content() -> anyhow::Result<()> {
        let mut block = TextBlock::default();
        block.set_translation(
            "Sure.\n<block id=\"0\">\nTranslated &lt;line&gt;\n</block>\nDone.".to_string(),
        )?;
        assert_eq!(block.translation.as_deref(), Some("Translated <line>"));
        Ok(())
    }

    #[test]
    fn text_block_translation_keeps_multiline_plain_text() -> anyhow::Result<()> {
        let mut block = TextBlock::default();
        block.set_translation("First line\nSecond line".to_string())?;
        assert_eq!(
            block.translation.as_deref(),
            Some("First line\nSecond line")
        );
        Ok(())
    }

    #[test]
    fn text_block_translation_keeps_japanese_dialogue_quotes() -> anyhow::Result<()> {
        let mut block = TextBlock::default();
        block.set_translation("「quoted」".to_string())?;
        assert_eq!(block.translation.as_deref(), Some("「quoted」"));
        Ok(())
    }
}
