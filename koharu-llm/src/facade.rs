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

/// Whether a source string is worth spending an LLM call on.
///
/// Blocks that carry no letters at all — bare punctuation, ellipses, digits,
/// musical notes — render the same translated or not, so they are skipped.
pub fn needs_translation(text: &str) -> bool {
    text.chars()
        .any(|c| c.is_alphabetic() || ('\u{3040}'..='\u{30FF}').contains(&c))
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

/// Render blocks in the `[N]` wire format.
///
/// The marker is what the model has to echo back, so its cost is paid twice —
/// once in the prompt and once, at double the rate, in the completion. `[N]` is
/// about three tokens against nine for an XML block tag, and needs no entity
/// escaping, which keeps short SFX lines from being dwarfed by their delimiters.
fn format_blocks<'a>(texts: impl Iterator<Item = &'a str>) -> String {
    texts
        .enumerate()
        .map(|(idx, text)| format!("[{idx}]\n{}", text.trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_document_blocks(blocks: &[TextBlock]) -> String {
    format_blocks(blocks.iter().map(|block| block.text.as_deref().unwrap_or("")))
}

/// Parse a `[N]`-marked response into `expected` slots.
///
/// A marker line is `[N]` alone (tolerating surrounding spaces); everything up
/// to the next marker is that block's text. Mapping is by the id the model
/// wrote, not by position, so a dropped or reordered block leaves its slot empty
/// for retry instead of shifting every later translation onto the wrong bubble.
fn parse_marked_blocks(translation: &str, expected: usize) -> Option<Vec<String>> {
    let mut blocks = vec![String::new(); expected];
    let mut current: Option<usize> = None;
    let mut pending: Vec<&str> = Vec::new();
    let mut seen = 0usize;

    fn flush(blocks: &mut [String], current: Option<usize>, pending: &mut Vec<&str>, seen: &mut usize) {
        let Some(id) = current else {
            pending.clear();
            return;
        };
        let text = pending.join("\n").trim().to_string();
        pending.clear();
        if let Some(slot) = blocks.get_mut(id) {
            if slot.is_empty() && !text.is_empty() {
                *seen += 1;
            }
            *slot = text;
        }
    }

    for line in translation.lines() {
        match marker_id(line) {
            Some(id) => {
                flush(&mut blocks, current, &mut pending, &mut seen);
                current = Some(id);
            }
            None => pending.push(line),
        }
    }
    flush(&mut blocks, current, &mut pending, &mut seen);

    if seen == 0 {
        return None;
    }
    if seen < expected {
        tracing::warn!(parsed = seen, expected, "fewer marked blocks than expected");
    }
    Some(blocks)
}

/// The block id of a `[N]` marker line, or `None` for a content line.
pub fn marker_id(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    let inner = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    inner.trim().parse().ok()
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

/// Parse a model response into exactly `expected` block translations, falling
/// back to the numbered-list and single-line formats some providers emit.
/// `None` means the response was unusable and the caller should retry.
fn parse_block_translations(
    translation: &str,
    expected: usize,
) -> anyhow::Result<Option<Vec<String>>> {
    if let Some(blocks) = parse_marked_blocks(translation, expected) {
        return Ok(Some(blocks));
    }
    // Older prompts (and stale custom prompts) can still elicit XML block tags.
    if let Some(blocks) = parse_tagged_blocks(translation, expected)? {
        return Ok(Some(blocks));
    }
    if expected == 1 {
        return Ok(Some(split_legacy_lines(translation, 1)?));
    }
    if let Some(blocks) = parse_numbered_list_blocks(translation, expected) {
        tracing::debug!(expected, "parsed numbered list blocks from LLM response");
        return Ok(Some(blocks));
    }
    tracing::warn!(
        expected,
        "LLM response had no block tags, will retry each block individually"
    );
    Ok(None)
}

fn clean_translation(raw: &str) -> String {
    strip_sfx_description(&strip_speaker_prefix(&strip_wrapping_quotes(raw)))
}

/// A subset of a document's text blocks, renumbered `0..n`.
///
/// Sending only the blocks that still need work — instead of the whole page —
/// keeps retries from re-paying for blocks that already translated fine.
pub struct BlockSelection<'a> {
    blocks: Vec<&'a mut TextBlock>,
}

impl<'a> BlockSelection<'a> {
    /// Select `indices` (document order) out of `blocks`.
    pub fn from_indices(blocks: &'a mut [TextBlock], indices: &[usize]) -> Self {
        let wanted: std::collections::HashSet<usize> = indices.iter().copied().collect();
        Self {
            blocks: blocks
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| wanted.contains(index))
                .map(|(_, block)| block)
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

impl Translatable for BlockSelection<'_> {
    fn get_source(&self) -> anyhow::Result<String> {
        Ok(format_blocks(
            self.blocks
                .iter()
                .map(|block| block.text.as_deref().unwrap_or("")),
        ))
    }

    fn set_translation(&mut self, translation: String) -> anyhow::Result<()> {
        let Some(translations) = parse_block_translations(&translation, self.blocks.len())? else {
            return Ok(());
        };
        for (block, trans) in self.blocks.iter_mut().zip(translations) {
            // Leave blanks alone: an unfilled slot must stay eligible for retry
            // rather than being overwritten with an empty translation.
            if trans.trim().is_empty() {
                continue;
            }
            block.translation = Some(clean_translation(&trans));
        }
        Ok(())
    }
}

impl Translatable for Document {
    fn get_source(&self) -> anyhow::Result<String> {
        Ok(format_document_blocks(&self.text_blocks))
    }

    fn set_translation(&mut self, translation: String) -> anyhow::Result<()> {
        let expected = self.text_blocks.len();
        let Some(translations) = parse_block_translations(&translation, expected)? else {
            return Ok(());
        };

        for (block, trans) in self.text_blocks.iter_mut().zip(translations) {
            block.translation = Some(clean_translation(&trans));
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
        Ok(format_blocks(std::iter::once(source.as_str())))
    }

    fn set_translation(&mut self, translation: String) -> anyhow::Result<()> {
        // Only unwrap when the model actually delimited the block; a bare
        // multi-line reply must survive verbatim.
        let translation = match parse_marked_blocks(&translation, 1) {
            Some(blocks) => blocks.into_iter().next().unwrap_or_default(),
            None => match parse_tagged_blocks(&translation, 1)? {
                Some(blocks) => blocks.into_iter().next().unwrap_or_default(),
                None => translation,
            },
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

    /// True when an API provider is loaded. API providers receive a stable
    /// story context at load time (cacheable by the provider), so per-page
    /// context can stay minimal; local models get everything per call.
    pub async fn is_api(&self) -> bool {
        matches!(*self.state.read().await, State::ApiReady { .. })
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
            keys_total: None,
            keys_available: None,
            key_index: None,
        },
        State::Loading { model_id, source } => LlmState {
            status: LlmStateStatus::Loading,
            model_id: Some(model_id.clone()),
            source: Some(source.clone()),
            error: None,
            keys_total: None,
            keys_available: None,
            key_index: None,
        },
        State::Ready(llm) => LlmState {
            status: LlmStateStatus::Ready,
            model_id: Some(llm.id().to_string()),
            source: Some("local".to_string()),
            error: None,
            keys_total: None,
            keys_available: None,
            key_index: None,
        },
        State::ApiReady {
            provider,
            provider_id,
            model,
        } => {
            // Read live, so a UI that polls this sees rotations as they happen.
            let keys = provider.key_status();
            LlmState {
                status: LlmStateStatus::Ready,
                model_id: Some(format!("{provider_id}:{model}")),
                source: Some(provider_id.clone()),
                error: None,
                keys_total: keys.map(|k| k.total),
                keys_available: keys.map(|k| k.available),
                key_index: keys.map(|k| k.current),
            }
        }
        State::Failed(error) => LlmState {
            status: LlmStateStatus::Failed,
            model_id: None,
            source: None,
            error: Some(error.clone()),
            keys_total: None,
            keys_available: None,
            key_index: None,
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
    fn document_source_uses_marked_blocks() -> anyhow::Result<()> {
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
        // No entity escaping: the marker format is not XML.
        assert_eq!(source, "[0]\nHello\n[1]\n1 < 2\nA & B");

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
    fn text_block_source_uses_single_marked_block() -> anyhow::Result<()> {
        let block = TextBlock {
            text: Some("1 < 2\nA & B".to_string()),
            ..Default::default()
        };

        let source = block.get_source()?;
        assert_eq!(source, "[0]\n1 < 2\nA & B");

        Ok(())
    }

    #[test]
    fn marked_blocks_map_by_id_not_position() -> anyhow::Result<()> {
        let mut doc = Document {
            text_blocks: vec![TextBlock::default(), TextBlock::default(), TextBlock::default()],
            ..Default::default()
        };
        // Model answered out of order and skipped block 1.
        doc.set_translation("[2]\nthird\n[0]\nfirst".to_string())?;
        let got: Vec<_> = doc
            .text_blocks
            .iter()
            .map(|b| b.translation.as_deref().unwrap_or(""))
            .collect();
        assert_eq!(got, vec!["first", "", "third"]);
        Ok(())
    }

    #[test]
    fn marked_blocks_keep_multiline_content() {
        let parsed = parse_marked_blocks("[0]\nline one\nline two\n[1]\nsolo", 2).unwrap();
        assert_eq!(parsed, vec!["line one\nline two".to_string(), "solo".to_string()]);
    }

    #[test]
    fn marker_id_only_matches_a_bare_marker_line() {
        assert_eq!(marker_id("[3]"), Some(3));
        assert_eq!(marker_id("  [12]  "), Some(12));
        assert_eq!(marker_id("[0] text on same line"), None);
        assert_eq!(marker_id("text [0]"), None);
        assert_eq!(marker_id("[abc]"), None);
        assert_eq!(marker_id("plain line"), None);
    }

    #[test]
    fn unmarked_response_is_rejected_so_the_caller_can_retry() {
        assert!(parse_marked_blocks("just some prose", 3).is_none());
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
