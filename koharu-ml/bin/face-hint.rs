//! Print face + balloon detections for one manga page as JSON — the CV half
//! of the combined face-box approach: a real detector draws the box, an LLM
//! (Claude/Gemini, in /style-read) only has to pick which one is the right
//! character, instead of estimating `[x, y, w, h]` from the whole page by eye.
//!
//! Needs `face_detector_anime.onnx` and `bubble_detector.onnx` in the model
//! cache (`tools/export_ccip.py --face-only`, and the bubble detector export
//! script). Exits with an error, not a crash, when a model is missing — the
//! caller (prepare.py) treats that as "no hint this run", not a hard failure.
//!
//!     face-hint --input path/to/page.jpg
//!
//! Output is exactly `character_library::PageFaceScan`, printed to stdout —
//! boxes as fractions of the page, `/style-read`'s own convention, so they
//! paste straight into a `faces` entry with no unit conversion.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::character_library::scan_page_faces;
use tokio::runtime::Builder;

#[path = "common.rs"]
mod common;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,
}

fn main() -> Result<()> {
    common::init_tracing();

    std::thread::Builder::new()
        .name("face-hint".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let runtime = Builder::new_current_thread().enable_all().build()?;
            runtime.block_on(async_main())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("face-hint thread panicked"))?
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();
    let scan = scan_page_faces(&cli.input).await?;
    println!("{}", serde_json::to_string_pretty(&scan)?);
    Ok(())
}
