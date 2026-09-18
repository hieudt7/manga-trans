use std::future::Future;
use std::pin::Pin;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Serialize;
use serde_json::json;

use koharu_http::http::http_client;

use crate::{Language, prompt::build_system_prompt};

use super::{AnyProvider, ensure_provider_success, extend_story_context};

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Room for the answer and for adaptive thinking, which current models run by
/// default and which counts against this ceiling. A tighter cap cuts replies
/// off mid-sentence.
const MAX_TOKENS: u32 = 16_000;

/// Models that take `fallbacks: "default"`: a request their safety classifiers
/// decline is re-run server-side on Anthropic's recommended substitute instead
/// of coming back as a refusal. Manga dialogue is full of fights and threats,
/// which is exactly what a classifier can misread.
const FALLBACK_MODELS: &[&str] = &["claude-opus-5", "claude-fable-5-1"];
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

pub struct ClaudeProvider {
    pub api_key: String,
    pub story_context: Option<String>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block<'a> {
    Text { text: &'a str },
    Image { source: ImageSource<'a> },
}

#[derive(Serialize)]
struct ImageSource<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    media_type: &'a str,
    data: String,
}

impl ClaudeProvider {
    /// One user turn in, the text of the reply out.
    async fn send(
        &self,
        model: &str,
        system: &str,
        content: Vec<Block<'_>>,
    ) -> anyhow::Result<String> {
        let mut body = json!({
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{ "role": "user", "content": content }],
        });

        let mut request = http_client()
            .post(MESSAGES_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");
        if FALLBACK_MODELS.contains(&model) {
            body["fallbacks"] = json!("default");
            request = request.header("anthropic-beta", FALLBACK_BETA);
        }

        let response = request.body(serde_json::to_vec(&body)?).send().await?;
        let reply: serde_json::Value = ensure_provider_success("claude", response)
            .await?
            .json()
            .await?;
        text_of(&reply)
    }
}

/// The text of a reply.
///
/// Only `text` blocks count. With thinking on, the first block is a `thinking`
/// block, and taking `content[0]` — as this provider once did — reads it as an
/// empty answer.
fn text_of(reply: &serde_json::Value) -> anyhow::Result<String> {
    if reply["stop_reason"] == "refusal" {
        let reason = reply["stop_details"]["explanation"]
            .as_str()
            .or_else(|| reply["stop_details"]["category"].as_str())
            .unwrap_or("no reason given");
        anyhow::bail!("Claude declined the request: {reason}");
    }

    let text: String = reply["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("");

    if text.trim().is_empty() {
        anyhow::bail!(
            "Claude returned no text (stop_reason: {})",
            reply["stop_reason"].as_str().unwrap_or("unknown")
        );
    }
    if reply["stop_reason"] == "max_tokens" {
        tracing::warn!("Claude reply was cut off at max_tokens");
    }
    Ok(text)
}

impl AnyProvider for ClaudeProvider {
    fn translate<'a>(
        &'a self,
        source: &'a str,
        target_language: Language,
        page_context: Option<&'a str>,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let combined = extend_story_context(self.story_context.as_deref(), page_context);
            let system = build_system_prompt(target_language, None, combined.as_deref());
            self.send(model, &system, vec![Block::Text { text: source }])
                .await
        })
    }

    fn complete<'a>(
        &'a self,
        system_prompt: &'a str,
        user_prompt: &'a str,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            self.send(
                model,
                system_prompt,
                vec![Block::Text { text: user_prompt }],
            )
            .await
        })
    }

    fn look<'a>(
        &'a self,
        system_prompt: &'a str,
        user_prompt: &'a str,
        images: &'a [Vec<u8>],
        mime_type: &'a str,
        model: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send + 'a>> {
        Box::pin(async move {
            // Images first, then the question about them.
            let mut content: Vec<Block> = images
                .iter()
                .map(|bytes| Block::Image {
                    source: ImageSource {
                        kind: "base64",
                        media_type: mime_type,
                        data: BASE64.encode(bytes),
                    },
                })
                .collect();
            content.push(Block::Text { text: user_prompt });
            self.send(model, system_prompt, content).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_taken_from_text_blocks_not_from_the_thinking_block_before_them() {
        let reply = json!({
            "stop_reason": "end_turn",
            "content": [
                { "type": "thinking", "thinking": "" },
                { "type": "text", "text": "Xin " },
                { "type": "text", "text": "chào" }
            ]
        });
        assert_eq!(text_of(&reply).unwrap(), "Xin chào");
    }

    #[test]
    fn a_refusal_is_an_error_not_an_empty_answer() {
        let reply = json!({
            "stop_reason": "refusal",
            "stop_details": { "type": "refusal", "category": "cyber", "explanation": null },
            "content": []
        });
        let err = text_of(&reply).unwrap_err().to_string();
        assert!(err.contains("declined"), "{err}");
    }

    #[test]
    fn a_reply_with_only_thinking_is_an_error() {
        let reply = json!({
            "stop_reason": "max_tokens",
            "content": [{ "type": "thinking", "thinking": "" }]
        });
        assert!(text_of(&reply).is_err());
    }

    #[test]
    fn an_image_block_serialises_as_the_messages_api_expects() {
        let block = Block::Image {
            source: ImageSource {
                kind: "base64",
                media_type: "image/jpeg",
                data: "QQ==".into(),
            },
        };
        assert_eq!(
            serde_json::to_value(&block).unwrap(),
            json!({ "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "QQ==" } })
        );
    }
}
