//! One-off comparison tool: run the real app's speaker-detection pipeline
//! (bubble detect -> panel detect -> face detect -> CCIP identity match)
//! on a single page, using a character library built from /style-read's own
//! face crops, and print which character it thinks said each balloon.
//!
//! Temporary — not wired into Cargo.toml permanently, just for one test run.
//!
//!     speaker-test --page path/to/page.jpg --faces path/to/character_scan/faces

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::character_library::{scan_page_faces, CharacterLibrary};
use tokio::runtime::Builder;

#[path = "common.rs"]
mod common;

#[derive(Parser)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    page: PathBuf,
    #[arg(long, value_name = "DIR")]
    faces: PathBuf,
}

fn main() -> Result<()> {
    common::init_tracing();

    std::thread::Builder::new()
        .name("speaker-test".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let runtime = Builder::new_current_thread().enable_all().build()?;
            runtime.block_on(async_main())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("speaker-test thread panicked"))?
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();

    let lib = CharacterLibrary::load()?;

    let mut loaded = Vec::new();
    for entry in std::fs::read_dir(&cli.faces)? {
        let entry = entry?;
        if !entry.path().is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        let mut face_images = Vec::new();
        let mut file_list = std::fs::read_dir(entry.path())?
            .filter_map(|f| f.ok())
            .filter(|f| f.path().extension().map(|e| e == "png").unwrap_or(false))
            .map(|f| f.path())
            .collect::<Vec<_>>();
        file_list.sort();
        for path in &file_list {
            face_images.push(image::open(path)?);
        }
        if face_images.is_empty() {
            continue;
        }
        lib.add_character(id.clone(), vec![], vec![], &face_images)?;
        loaded.push((id, file_list.len()));
    }
    eprintln!("loaded {} characters into a fresh test library:", loaded.len());
    for (id, n) in &loaded {
        eprintln!("  - {id} ({n} face image(s))");
    }

    let scan = scan_page_faces(&cli.page).await?;
    eprintln!(
        "\ndetected {} face(s), {} balloon(s) on the page",
        scan.faces.len(),
        scan.balloons.len()
    );

    let bytes = std::fs::read(&cli.page)?;
    let format = image::guess_format(&bytes)?;
    let image = image::load_from_memory_with_format(&bytes, format)?;
    let (w, h) = (image.width() as f32, image.height() as f32);

    let panels = lib.detect_panels(&image);
    eprintln!("detected {} panel(s)\n", panels.len());

    let blocks: Vec<(String, f32, f32, f32, f32)> = scan
        .balloons
        .iter()
        .enumerate()
        .map(|(i, b)| {
            (
                format!("b{i}"),
                b.bbox.x * w,
                b.bbox.y * h,
                b.bbox.width * w,
                b.bbox.height * h,
            )
        })
        .collect();
    let balloons: Vec<(f32, f32, f32, f32)> = scan
        .balloons
        .iter()
        .map(|b| (b.bbox.x * w, b.bbox.y * h, b.bbox.width * w, b.bbox.height * h))
        .collect();

    let assignments = lib.assign_speakers_to_blocks(&image, &blocks, &balloons, &panels);

    println!("=== speaker assignments (balloon position as fraction of page, x,y from top-left) ===");
    for (id, m) in &assignments {
        let idx: usize = id[1..].parse().unwrap();
        let b = &scan.balloons[idx].bbox;
        match m {
            Some(fm) => println!(
                "{id} @ ({:.3}, {:.3}) size ({:.3} x {:.3}) -> {}  (confidence {:.3}, known={})",
                b.x, b.y, b.width, b.height, fm.name, fm.confidence, fm.is_known
            ),
            None => println!(
                "{id} @ ({:.3}, {:.3}) size ({:.3} x {:.3}) -> (no match)",
                b.x, b.y, b.width, b.height
            ),
        }
    }

    Ok(())
}
