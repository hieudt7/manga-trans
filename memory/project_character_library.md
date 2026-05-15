---
name: Character Library feature
description: CharacterLibrary module added with CCIP embeddings, face detection, and translation context injection
type: project
---

CharacterLibrary module (`koharu-ml/src/character_library/mod.rs`) was implemented with:

- CCIP ONNX embeddings (128×128 input, 512-dim output, ImageNet normalization, cosine similarity threshold 0.7)
- Optional YOLOv8-format anime face detector (`face_detector_anime.onnx`)
- Character entries stored in `~/.local/share/koharu/character_lib.json`
- `scan_and_build_context(image)` returns LLM-ready context string
- Full-library fallback when face detector is not available

**Why:** User requested per-page character context injection into translation prompts so the LLM knows which characters appear in each scene.

**How to apply:**

- CCIP model path: `~/.cache/koharu/models/ccip.onnx` — export with `tools/export_ccip.py`
- Face detector path: `~/.cache/koharu/models/face_detector_anime.onnx` — same script with `--face-only`
- API endpoints: `GET /characters`, `POST /characters` (multipart: face+name+traits+relations), `DELETE /characters/{id}`
- `page_context: Option<&str>` was added to `AnyProvider::translate` and `Model::translate_with_context`
- Character context is injected automatically in both `pipeline.rs` and `ops/llm.rs` before each translation
- Local (llama.cpp) models do NOT receive page_context yet — only API providers (openai, claude, gemini, deepseek, openai-compatible)
