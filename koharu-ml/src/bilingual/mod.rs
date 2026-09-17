//! Lining up a volume with its published translation.
//!
//! To learn how a translator writes, the pipeline needs sentence pairs: this
//! Japanese line, that Vietnamese one. What it is given instead is two folders
//! of images — the raw volume and the translated volume — and those rarely line
//! up one to one. A cover, a credits page, an advertisement or a colour insert
//! appears in one and not the other, and from the first extra page onwards
//! every naive pairing is off by one.
//!
//! Nothing here reads any text. The translated page is the same artwork with
//! different words on it, so the artwork is the only thing worth matching on,
//! and matching on it costs no OCR. Text comes later, and only for the pages
//! that survived.

pub mod corpus;
pub mod style;

use koharu_types::TextBlock;

/// How alike two pages must look before pairing them beats leaving both
/// unpaired.
///
/// Measured on a 94-page volume against its own translation, where the right
/// answer is known: the worst true pair scored 0.729 and the best wrong pair
/// 0.450. With `GAP_COST` this puts the decision at 0.59 — a little under
/// halfway, with about 0.14 of room on each side.
const MATCH_FLOOR: f32 = 0.69;

/// What skipping one page costs. Two of these are what a pair has to beat, so
/// together with `MATCH_FLOOR` they set where the decision falls.
const GAP_COST: f32 = 0.05;

/// Side of the square the page is reduced to before comparing.
///
/// Bigger separates better and costs almost nothing in accuracy: from 16px to
/// 256px the worst true pair only fell from 0.965 to 0.952 while the best wrong
/// pair fell from 0.947 to 0.450. The worry that lettering would start to count
/// against true pairs did not show up in the measurement. One volume's
/// signatures at this size come to about 12MB, and only one volume is aligned
/// at a time.
pub const THUMBNAIL_SIDE: usize = 256;

/// What a page looks like, with nothing that depends on the language it is
/// lettered in.
///
/// Panel layout was tried here first and measured actively harmful: the panel
/// detector reported a different grid for a page and its own translation in a
/// third of cases, and every weight above zero pushed the worst true pair below
/// the best wrong one. It is gone, which also means aligning a volume needs no
/// model at all.
#[derive(Debug, Clone, PartialEq)]
pub struct PageSignature {
    /// Greyscale page reduced to `THUMBNAIL_SIDE` squared.
    pub thumbnail: Vec<u8>,
}

impl PageSignature {
    pub fn new(image: &image::DynamicImage) -> Self {
        Self {
            thumbnail: image
                .resize_exact(
                    THUMBNAIL_SIDE as u32,
                    THUMBNAIL_SIDE as u32,
                    image::imageops::FilterType::Triangle,
                )
                .to_luma8()
                .into_raw(),
        }
    }
}

/// How alike two pages are, from 0 to 1.
///
/// Each thumbnail is taken against its own mean before comparing. A manga page
/// is mostly white, so comparing the pixels directly mostly compares how much
/// paper is showing — two unrelated pages already agree on all of it. Measured,
/// that told true pairs from wrong ones by 0.135; against the mean, by 0.279.
pub fn similarity(a: &PageSignature, b: &PageSignature) -> f32 {
    let (x, y) = (&a.thumbnail, &b.thumbnail);
    if x.is_empty() || x.len() != y.len() {
        return 0.0;
    }

    let n = x.len() as f32;
    let mean_x = x.iter().map(|v| f32::from(*v)).sum::<f32>() / n;
    let mean_y = y.iter().map(|v| f32::from(*v)).sum::<f32>() / n;

    let (mut covariance, mut var_x, mut var_y) = (0.0f32, 0.0f32, 0.0f32);
    for (a, b) in x.iter().zip(y) {
        let (a, b) = (f32::from(*a) - mean_x, f32::from(*b) - mean_y);
        covariance += a * b;
        var_x += a * a;
        var_y += b * b;
    }

    let spread = (var_x * var_y).sqrt();
    if spread <= 0.0 {
        // A blank page has nothing to correlate; treat it as telling us
        // nothing rather than as a match with every other blank.
        return 0.0;
    }
    (covariance / spread).clamp(0.0, 1.0)
}

/// How much two rectangles overlap, as a fraction of the area they cover
/// together. Used for balloons, which sit in the same places on a page and its
/// translation because the artwork under them is the same.
fn rect_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax2, ay2) = (a[0] + a[2], a[1] + a[3]);
    let (bx2, by2) = (b[0] + b[2], b[1] + b[3]);
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = ax2.min(bx2);
    let y2 = ay2.min(by2);
    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }
    let intersection = (x2 - x1) * (y2 - y1);
    let union = a[2] * a[3] + b[2] * b[3] - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// One step of the alignment between the two volumes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PagePairing {
    /// The same page, found in both.
    Matched {
        raw: usize,
        translated: usize,
        score: f32,
    },
    /// In the raw volume only — an insert the translation dropped.
    RawOnly(usize),
    /// In the translation only — a credits page, a note from the group.
    TranslatedOnly(usize),
}

/// Line the two volumes up, allowing either to carry pages the other does not.
///
/// Pages keep their order in both volumes, and that is the whole of what makes
/// this tractable: it turns "which page goes with which" into aligning two
/// sequences, where an extra page is a gap rather than a wrong answer. Pairing
/// each page with its own best match independently would let one bad page drag
/// the rest out of step.
pub fn align_pages(raw: &[PageSignature], translated: &[PageSignature]) -> Vec<PagePairing> {
    let (n, m) = (raw.len(), translated.len());
    if n == 0 || m == 0 {
        return (0..n)
            .map(PagePairing::RawOnly)
            .chain((0..m).map(PagePairing::TranslatedOnly))
            .collect();
    }

    // Scoring a pair against `MATCH_FLOOR` rather than against zero is what
    // lets a poor pair lose to two gaps.
    let mut score = vec![vec![0.0f32; m + 1]; n + 1];
    for i in 1..=n {
        score[i][0] = score[i - 1][0] - GAP_COST;
    }
    for j in 1..=m {
        score[0][j] = score[0][j - 1] - GAP_COST;
    }
    for i in 1..=n {
        for j in 1..=m {
            let paired =
                score[i - 1][j - 1] + similarity(&raw[i - 1], &translated[j - 1]) - MATCH_FLOOR;
            let skip_raw = score[i - 1][j] - GAP_COST;
            let skip_translated = score[i][j - 1] - GAP_COST;
            score[i][j] = paired.max(skip_raw).max(skip_translated);
        }
    }

    let mut steps = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let sim = similarity(&raw[i - 1], &translated[j - 1]);
            if (score[i][j] - (score[i - 1][j - 1] + sim - MATCH_FLOOR)).abs() < f32::EPSILON {
                steps.push(PagePairing::Matched {
                    raw: i - 1,
                    translated: j - 1,
                    score: sim,
                });
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && (score[i][j] - (score[i - 1][j] - GAP_COST)).abs() < f32::EPSILON {
            steps.push(PagePairing::RawOnly(i - 1));
            i -= 1;
        } else {
            steps.push(PagePairing::TranslatedOnly(j - 1));
            j -= 1;
        }
    }
    steps.reverse();
    steps
}

/// A balloon found in both versions of one page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockPairing {
    pub raw: usize,
    pub translated: usize,
    /// How far the two sat apart, as a fraction of the page diagonal.
    pub distance: f32,
}

/// How far apart two blocks may sit and still be the same balloon, as a
/// fraction of the page diagonal.
///
/// Measured across ten pages of a volume and its translation: where both sides
/// found the same number of balloons, the worst true pair sat 0.013 away and
/// the average 0.005. Pairs this far apart do not occur between balloons that
/// correspond; the larger distances in the measurement all came from forcing a
/// match where there was no counterpart.
const MAX_CENTRE_DISTANCE: f32 = 0.08;

/// Pair the balloons of one page with the balloons of its translation.
///
/// Matched on where they sit, not on the shape of the box. Japanese runs down
/// the page in a narrow column and Vietnamese across it in wide lines, so the
/// same balloon yields two quite different rectangles — measured, the overlap
/// between true pairs averaged 0.40 while their centres agreed to within half a
/// percent of the page. Overlap says they are different; position says they are
/// the same, and position is right.
///
/// A block with no counterpart simply goes unpaired: a translator who merged
/// two balloons into one, or a detector that missed one, costs those lines and
/// nothing else. Requiring each to be the other's nearest keeps a crowd of
/// balloons on one side from all claiming the same one on the other.
pub fn match_blocks(
    raw: &[TextBlock],
    translated: &[TextBlock],
    page_width: f32,
    page_height: f32,
) -> Vec<BlockPairing> {
    let diagonal = page_width.hypot(page_height).max(1.0);
    let limit = MAX_CENTRE_DISTANCE * diagonal;
    let centre = |b: &TextBlock| (b.x + b.width / 2.0, b.y + b.height / 2.0);
    let gap = |a: &TextBlock, b: &TextBlock| {
        let ((ax, ay), (bx, by)) = (centre(a), centre(b));
        (ax - bx).hypot(ay - by)
    };

    let nearest = |from: &TextBlock, among: &[TextBlock]| -> Option<usize> {
        among
            .iter()
            .enumerate()
            .map(|(i, other)| (i, gap(from, other)))
            .filter(|(_, d)| *d <= limit)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
    };

    raw.iter()
        .enumerate()
        .filter_map(|(i, block)| {
            let j = nearest(block, translated)?;
            // Each has to be the other's nearest, or two balloons close
            // together on one side both land on the same one opposite.
            (nearest(&translated[j], raw)? == i).then_some(BlockPairing {
                raw: i,
                translated: j,
                distance: gap(block, &translated[j]) / diagonal,
            })
        })
        .collect()
}

/// One line of dialogue in both languages, taken from the same balloon.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SentencePair {
    /// What the artist wrote.
    pub source: String,
    /// What the translator made of it.
    pub target: String,
    /// The raw page it came from, for tracing a suspect pair back.
    pub page: String,
    /// Where the translated line sat on its page, so a suspect pair can be
    /// looked at rather than argued about.
    pub target_box: [f32; 4],
    /// How far the page and the balloon agreed — the page's alignment score
    /// times the balloon's overlap. A corpus is only worth as much as its worst
    /// pairs, so this travels with the pair rather than being discarded once
    /// the pair is made.
    pub confidence: f32,
}

/// Take the dialogue of one page and its translation, balloon by balloon.
///
/// Both documents must already have been detected and read. Returns nothing
/// when the balloons do not correspond — see `match_blocks` for why a page is
/// dropped whole rather than partly trusted.
pub fn pairs_from_page(
    raw: &[TextBlock],
    translated: &[TextBlock],
    page: &str,
    page_score: f32,
    page_width: f32,
    page_height: f32,
) -> Vec<SentencePair> {
    match_blocks(raw, translated, page_width, page_height)
        .into_iter()
        .filter_map(|pairing| {
            let source = raw[pairing.raw].text.as_deref()?.trim();
            let target = translated[pairing.translated].text.as_deref()?.trim();
            // A balloon holding only marks teaches nothing about how someone
            // translates.
            if !koharu_types::carries_words(source) || !koharu_types::carries_words(target) {
                return None;
            }
            let t = &translated[pairing.translated];
            Some(SentencePair {
                source: source.to_string(),
                target: target.to_string(),
                page: page.to_string(),
                target_box: [t.x, t.y, t.width, t.height],
                // Near misses are worth less than exact ones, and the score
                // travels with the pair so a corpus can be trimmed later.
                confidence: page_score * (1.0 - pairing.distance / MAX_CENTRE_DISTANCE).max(0.0),
            })
        })
        .collect()
}

/// Check a real corpus lines up, and how far the right answers sit from the
/// wrong ones.
///
/// The thresholds in this module were set from this measurement rather than
/// guessed; run it against a new corpus before trusting them on one.
///
/// `KOHARU_CORPUS=<dir with raw/ and trans/> cargo test --release -p koharu-ml
/// --lib measure_corpus_alignment -- --ignored --nocapture`
#[cfg(test)]
mod measure {
    use super::*;

    /// The vision model reads every balloon on a page in one request, so its
    /// answers have to come back separably. A bare numbered list would be
    /// ambiguous the moment a balloon's own text starts with a digit, hence the
    /// marker line.
    const READ_PROMPT: &str = "\
Mỗi ảnh là một khung chữ cắt từ trang truyện tranh tiếng Việt, theo đúng thứ tự.
Với ảnh thứ n, in ra một dòng chỉ gồm ###n, rồi đến các dòng chữ đọc được trong ảnh đó.
Chép đúng chữ, giữ nguyên dấu thanh và chữ hoa thường; xuống dòng đúng như trong ảnh.
Bỏ qua chữ thuộc hình vẽ nền hay nhãn đè lên; chỉ đọc chữ của khung đó.
Nếu ảnh không có chữ, để trống phần sau ###n.
Không thêm bất kỳ lời giải thích nào.";

    /// The local Vietnamese recogniser, when its models are on hand.
    ///
    /// Preferred over the vision model when both are available: it costs
    /// nothing, needs no network, and measured 16.2% character error against
    /// 151 balloons whose right answers are known — close enough that a style
    /// profile learned from it named the same pronoun pairs.
    fn local_reader() -> anyhow::Result<Option<crate::vietocr::VietOcr>> {
        let Some(dir) = std::env::var_os("KOHARU_VIETOCR_DIR") else {
            return Ok(None);
        };
        Ok(Some(crate::vietocr::VietOcr::open(std::path::Path::new(
            &dir,
        ))?))
    }

    /// Off by default so the alignment measurement still runs without a key
    /// or a network.
    fn vision_ocr_wanted() -> bool {
        std::env::var("KOHARU_VISION_OCR").is_ok_and(|v| v != "0")
    }

    fn vision_provider() -> anyhow::Result<Box<dyn koharu_llm::providers::AnyProvider>> {
        koharu_llm::providers::build_provider(
            "gemini",
            koharu_llm::providers::ProviderConfig {
                api_key: None,
                base_url: None,
                temperature: None,
                max_tokens: None,
                custom_system_prompt: None,
                story_context: None,
                key_start_index: None,
            },
        )
    }

    /// Crop each box with a little air around it — lettering often leans past
    /// the detected edge, and a model shown a clipped glyph guesses the rest.
    fn crop_boxes(sheet: &image::DynamicImage, boxes: &[[f32; 4]]) -> anyhow::Result<Vec<Vec<u8>>> {
        const PAD: f32 = 8.0;
        let mut out = Vec::with_capacity(boxes.len());
        for &[x, y, w, h] in boxes {
            let left = (x - PAD).max(0.0) as u32;
            let top = (y - PAD).max(0.0) as u32;
            let right = ((x + w + PAD) as u32).min(sheet.width());
            let bottom = ((y + h + PAD) as u32).min(sheet.height());
            let mut png = std::io::Cursor::new(Vec::new());
            if right > left && bottom > top {
                sheet
                    .crop_imm(left, top, right - left, bottom - top)
                    .write_to(&mut png, image::ImageFormat::Png)?;
            }
            out.push(png.into_inner());
        }
        Ok(out)
    }

    /// Split the model's reply back into one reading per box. Anything the
    /// model failed to mark comes back empty rather than shifting every later
    /// balloon onto the wrong line.
    fn split_readings(reply: &str, count: usize) -> Vec<String> {
        let mut out = vec![String::new(); count];
        let mut current: Option<usize> = None;
        for line in reply.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("###") {
                current = rest
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .and_then(|n| n.checked_sub(1));
                continue;
            }
            if let Some(index) = current.filter(|i| *i < count) {
                if !out[index].is_empty() {
                    out[index].push('\n');
                }
                out[index].push_str(trimmed);
            }
        }
        for reading in &mut out {
            *reading = reading.trim().to_string();
        }
        out
    }

    async fn read_balloons(
        provider: &dyn koharu_llm::providers::AnyProvider,
        sheet: &image::DynamicImage,
        boxes: &[[f32; 4]],
    ) -> anyhow::Result<Vec<String>> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        let crops = crop_boxes(sheet, boxes)?;
        let reply = provider
            .look(
                "Bạn đọc chữ trên trang truyện tranh tiếng Việt.",
                READ_PROMPT,
                &crops,
                "image/png",
                "gemini-3.1-flash-lite-preview",
            )
            .await?;
        Ok(split_readings(&reply, boxes.len()))
    }

    #[test]
    fn a_reply_splits_back_into_one_reading_per_balloon() {
        let reply = "###1\nTHƯA SẾP,\nNGUY TO RỒI Ạ!\n###2\nHỬM?\n###3\n";
        assert_eq!(
            split_readings(reply, 3),
            vec!["THƯA SẾP,\nNGUY TO RỒI Ạ!", "HỬM?", ""]
        );
    }

    /// A balloon the model forgot to mark must not shift every later balloon
    /// onto the wrong text — a silent off-by-one would poison the whole corpus.
    #[test]
    fn a_skipped_marker_leaves_a_hole_rather_than_shifting_the_rest() {
        let reply = "###1\nMỘT\n###3\nBA";
        assert_eq!(split_readings(reply, 3), vec!["MỘT", "", "BA"]);
    }

    /// Balloons whose own text starts with digits are why the marker is a
    /// distinct line and not a bare number.
    #[test]
    fn digits_inside_a_balloon_are_not_mistaken_for_markers() {
        let reply = "###1\n1979\nNĂM ẤY\n###2\n60 PHÚT";
        assert_eq!(split_readings(reply, 2), vec!["1979\nNĂM ẤY", "60 PHÚT"]);
    }

    /// Distil a style profile out of a corpus already paired into
    /// `pairs.jsonl`, and print it.
    ///
    /// `KOHARU_CORPUS=<dir with pairs.jsonl> cargo test --release -p koharu-ml
    /// --lib measure_style -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn measure_style() -> anyhow::Result<()> {
        let root =
            std::path::PathBuf::from(std::env::var_os("KOHARU_CORPUS").expect("set KOHARU_CORPUS"));

        let pairs: Vec<SentencePair> = std::fs::read_to_string(root.join("pairs.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        println!("{} cặp câu", pairs.len());
        println!(
            "{} cặp dùng để học",
            super::style::sample(&pairs, 220).len()
        );

        let provider = vision_provider()?;
        let runtime = tokio::runtime::Runtime::new()?;
        let profile = runtime.block_on(super::style::learn(
            &*provider,
            &pairs,
            "gemini-3.1-flash-lite-preview",
        ))?;

        println!("\n{}", serde_json::to_string_pretty(&profile)?);
        println!(
            "\n--- đưa vào story context ---\n{}",
            profile.to_context().unwrap_or_default()
        );
        Ok(())
    }

    /// Read lettering off real balloons with a vision model, and print the
    /// answers next to whatever is already recorded for them.
    ///
    /// Every local OCR measured against this corpus (PaddleOCR-VL 1.5 and 1.6,
    /// PP-OCRv6, Tesseract 5 `vie`) misread the outlined text that sits over
    /// artwork. This is the check that a vision model does better, on the same
    /// balloons, before anything is built on top of it.
    ///
    /// A disagreement here is not a verdict either way — the recorded reading
    /// is PaddleOCR's, and on this corpus it is wrong more often than not. The
    /// point of the run is to put the two side by side and read them.
    ///
    /// `KOHARU_CORPUS=<dir with trans/ and pairs.jsonl> cargo test --release
    /// -p koharu-ml --lib measure_vision_ocr -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn measure_vision_ocr() -> anyhow::Result<()> {
        let root =
            std::path::PathBuf::from(std::env::var_os("KOHARU_CORPUS").expect("set KOHARU_CORPUS"));

        let provider = vision_provider()?;
        let runtime = tokio::runtime::Runtime::new()?;

        let mut by_page: Vec<(String, Vec<(String, [f32; 4])>)> = Vec::new();
        for line in std::fs::read_to_string(root.join("pairs.jsonl"))?.lines() {
            let record: serde_json::Value = serde_json::from_str(line)?;
            let page = record["page"].as_str().unwrap_or_default().to_string();
            let recorded = record["target"].as_str().unwrap_or_default().to_string();
            let boxed: Vec<f32> = serde_json::from_value(record["target_box"].clone())?;
            let [x, y, w, h] = boxed[..] else { continue };
            match by_page.last_mut() {
                Some((name, items)) if *name == page => items.push((recorded, [x, y, w, h])),
                _ => by_page.push((page, vec![(recorded, [x, y, w, h])])),
            }
        }

        let mut agreed = 0usize;
        let mut total = 0usize;
        for (page, items) in &by_page {
            let sheet = image::open(root.join("trans").join(page))?;
            let boxes: Vec<[f32; 4]> = items.iter().map(|(_, b)| *b).collect();
            let readings = runtime.block_on(read_balloons(&*provider, &sheet, &boxes))?;

            println!("\n{page}");
            for ((recorded, _), read) in items.iter().zip(&readings) {
                total += 1;
                let same = read.trim().eq_ignore_ascii_case(recorded.trim());
                if same {
                    agreed += 1;
                }
                println!(
                    "  đã ghi: {recorded:?}\n  nhìn ra: {read:?}{}",
                    if same { "   (trùng)" } else { "" }
                );
            }
        }

        println!("\n{agreed}/{total} khung trùng bản PaddleOCR đã ghi — chỗ lệch cần đọc bằng mắt");
        Ok(())
    }

    #[test]
    #[ignore]
    fn measure_corpus_alignment() -> anyhow::Result<()> {
        let root =
            std::path::PathBuf::from(std::env::var_os("KOHARU_CORPUS").expect("set KOHARU_CORPUS"));

        let mut sides = Vec::new();
        for side in ["raw", "trans"] {
            let files = corpus::list_pages(&root.join(side));
            println!("{side}: {} trang", files.len());
            let mut signatures = Vec::with_capacity(files.len());
            for path in &files {
                signatures.push((
                    path.file_name().unwrap().to_string_lossy().to_string(),
                    PageSignature::new(&image::open(path)?),
                ));
            }
            sides.push(signatures);
        }

        let raw: Vec<_> = sides[0].iter().map(|(_, s)| s.clone()).collect();
        let translated: Vec<_> = sides[1].iter().map(|(_, s)| s.clone()).collect();

        let steps = align_pages(&raw, &translated);
        let mut scores = Vec::new();
        let mut matched = 0usize;
        for step in &steps {
            match step {
                PagePairing::Matched {
                    raw: r,
                    translated: t,
                    score,
                } => {
                    matched += 1;
                    scores.push(*score);
                    if *score < MATCH_FLOOR {
                        println!(
                            "  khớp yếu {score:.3}: {} ↔ {}",
                            sides[0][*r].0, sides[1][*t].0
                        );
                    }
                }
                PagePairing::RawOnly(i) => println!("  chỉ có ở raw: {}", sides[0][*i].0),
                PagePairing::TranslatedOnly(j) => {
                    println!("  chỉ có ở trans: {}", sides[1][*j].0)
                }
            }
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "\nghép được {matched}/{} trang",
            raw.len().max(translated.len())
        );
        if !scores.is_empty() {
            println!(
                "điểm khớp: thấp nhất {:.3}  trung vị {:.3}  cao nhất {:.3}",
                scores[0],
                scores[scores.len() / 2],
                scores[scores.len() - 1]
            );
        }

        // When both sides name their pages alike, page i is page i and the
        // right answer is known — so the gap between right and wrong can be
        // measured directly rather than inferred from how the alignment went.
        if raw.len() == translated.len() {
            let mut right: Vec<f32> = (0..raw.len())
                .map(|i| similarity(&raw[i], &translated[i]))
                .collect();
            let mut wrong: Vec<f32> = (0..raw.len())
                .flat_map(|i| [(i + 1) % raw.len(), (i + 7) % raw.len()].map(move |j| (i, j)))
                .filter(|(i, j)| i != j)
                .map(|(i, j)| similarity(&raw[i], &translated[j]))
                .collect();
            right.sort_by(|a, b| a.partial_cmp(b).unwrap());
            wrong.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let decision = MATCH_FLOOR - 2.0 * GAP_COST;
            println!(
                "đúng thấp nhất {:.3}  |  ngưỡng {decision:.3}  |  sai cao nhất {:.3}",
                right[0],
                wrong[wrong.len() - 1]
            );
        }

        // Reading the pages is the expensive half, so it runs only over pages
        // the alignment already accepted, and only as many as asked for.
        if let Some(limit) = std::env::var("KOHARU_CORPUS_PAGES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            koharu_llm::sys::initialize()?;
            let backend =
                std::sync::Arc::new(koharu_llm::safe::llama_backend::LlamaBackend::init()?);
            let runtime = tokio::runtime::Runtime::new()?;
            let ml = runtime.block_on(crate::facade::Model::new(false, backend))?;

            // The glyph mask comes back with the blocks: the local reader uses
            // it to blank the artwork that shares a block with the lettering.
            let read = |path: &std::path::Path| -> anyhow::Result<(Vec<TextBlock>, Option<image::DynamicImage>)> {
                let mut doc = koharu_types::Document::open(path.to_path_buf())?;
                runtime.block_on(ml.detect(&mut doc))?;
                runtime.block_on(ml.ocr(&mut doc))?;
                let mask = doc.segment.as_ref().map(|m| m.0.clone());
                Ok((doc.text_blocks, mask))
            };

            let reader = local_reader()?;
            let vision: Option<Box<dyn koharu_llm::providers::AnyProvider>> =
                if reader.is_none() && vision_ocr_wanted() {
                    Some(vision_provider()?)
                } else {
                    None
                };

            let raw_files = corpus::list_pages(&root.join(corpus::RAW_DIR));
            let trans_files = corpus::list_pages(&root.join(corpus::TRANSLATED_DIR));
            let mut pairs = Vec::new();
            let (mut seen, mut dropped) = (0usize, 0usize);
            for step in &steps {
                if seen >= limit {
                    break;
                }
                let PagePairing::Matched {
                    raw: r,
                    translated: t,
                    score,
                } = step
                else {
                    continue;
                };
                seen += 1;
                let name = sides[0][*r].0.clone();
                let (a, _) = read(&raw_files[*r])?;
                let (mut b, trans_mask) = read(&trans_files[*t])?;
                let page_image = image::open(&raw_files[*r])?;

                // The pipeline's own OCR finds the Vietnamese boxes reliably
                // but misreads what is in them — outlined lettering over
                // artwork defeated every general engine measured. Detection
                // stays as it is; only the reading of the translated side is
                // redone, by whichever reader was asked for.
                if reader.is_some() || vision_ocr_wanted() {
                    let sheet = image::open(&trans_files[*t])?;
                    let seen: Vec<String> = match reader.as_ref() {
                        Some(local) => b
                            .iter()
                            .map(|block| {
                                corpus::read_translated(local, &sheet, trans_mask.as_ref(), block)
                            })
                            .collect::<anyhow::Result<_>>()?,
                        None => {
                            let boxes: Vec<[f32; 4]> =
                                b.iter().map(|t| [t.x, t.y, t.width, t.height]).collect();
                            runtime.block_on(read_balloons(
                                vision.as_deref().unwrap(),
                                &sheet,
                                &boxes,
                            ))?
                        }
                    };
                    for (block, reading) in b.iter_mut().zip(seen) {
                        block.text = Some(reading);
                    }
                }
                let found = pairs_from_page(
                    &a,
                    &b,
                    &name,
                    *score,
                    page_image.width() as f32,
                    page_image.height() as f32,
                );
                if found.is_empty() {
                    dropped += 1;
                }
                println!(
                    "  {name}: {} ↔ {} bong bóng → {} cặp",
                    a.len(),
                    b.len(),
                    found.len()
                );
                pairs.extend(found);
            }

            println!(
                "\nđọc {seen} trang, bỏ {dropped}, lấy được {} cặp câu",
                pairs.len()
            );
            if !pairs.is_empty() {
                // The filter on length is not set yet; this is the measurement
                // that would set it.
                let mut ratios: Vec<f32> = pairs
                    .iter()
                    .map(|p| {
                        p.target.chars().count() as f32 / p.source.chars().count().max(1) as f32
                    })
                    .collect();
                ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "tỉ lệ dài việt/nhật: thấp nhất {:.2}  phân vị 10 {:.2}  trung vị {:.2}  phân vị 90 {:.2}  cao nhất {:.2}",
                    ratios[0],
                    ratios[ratios.len() / 10],
                    ratios[ratios.len() / 2],
                    ratios[ratios.len() * 9 / 10],
                    ratios[ratios.len() - 1]
                );
                let out = root.join("pairs.jsonl");
                let body: String = pairs
                    .iter()
                    .map(|p| serde_json::to_string(p).unwrap() + "\n")
                    .collect();
                std::fs::write(&out, body)?;
                println!("ghi ra {}", out.display());
            }
        }

        let in_order = steps.iter().enumerate().all(|(k, step)| {
            matches!(step, PagePairing::Matched { raw, translated, .. } if *raw == k && *translated == k)
        });
        println!("khớp đúng theo thứ tự file: {in_order}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A thumbnail with structure in it, different for every `seed`. Real pages
    /// correlate because they share drawing, not because they share brightness,
    /// so a flat fill would test nothing.
    fn page(seed: u32) -> PageSignature {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let thumbnail = (0..THUMBNAIL_SIDE * THUMBNAIL_SIDE)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        PageSignature { thumbnail }
    }

    /// The same page lettered again: the drawing is there, the words are not.
    fn relettered(seed: u32) -> PageSignature {
        let mut signature = page(seed);
        for (i, value) in signature.thumbnail.iter_mut().enumerate() {
            if i % 11 == 0 {
                *value = value.wrapping_add(90);
            }
        }
        signature
    }

    fn matched(steps: &[PagePairing]) -> Vec<(usize, usize)> {
        steps
            .iter()
            .filter_map(|step| match step {
                PagePairing::Matched {
                    raw, translated, ..
                } => Some((*raw, *translated)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_page_matches_itself_relettered_and_not_its_neighbour() {
        assert!(similarity(&page(3), &relettered(3)) > MATCH_FLOOR);
        assert!(similarity(&page(3), &relettered(4)) < MATCH_FLOOR - 2.0 * GAP_COST);
    }

    #[test]
    fn a_blank_page_matches_nothing() {
        // Nothing drawn means nothing to compare — which must read as no
        // information, not as agreement with every other blank page.
        let blank = PageSignature {
            thumbnail: vec![255; THUMBNAIL_SIDE * THUMBNAIL_SIDE],
        };
        assert_eq!(similarity(&blank, &blank), 0.0);
    }

    #[test]
    fn two_volumes_that_run_in_step_pair_straight_through() {
        let raw: Vec<_> = (0..4).map(page).collect();
        let translated: Vec<_> = (0..4).map(relettered).collect();

        assert_eq!(
            matched(&align_pages(&raw, &translated)),
            [(0, 0), (1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn a_page_only_the_translation_carries_becomes_a_gap() {
        // A credits page dropped into the middle of the translated volume.
        // Everything after it would be off by one if pairing went by position.
        let raw: Vec<_> = (0..4).map(page).collect();
        let mut translated: Vec<_> = (0..4).map(relettered).collect();
        translated.insert(2, page(99));

        let steps = align_pages(&raw, &translated);
        assert_eq!(matched(&steps), [(0, 0), (1, 1), (2, 3), (3, 4)]);
        assert!(
            steps.contains(&PagePairing::TranslatedOnly(2)),
            "the extra page has to be called out, not quietly absorbed: {steps:?}"
        );
    }

    #[test]
    fn a_colour_insert_the_translation_dropped_becomes_a_gap() {
        let mut raw: Vec<_> = (0..4).map(page).collect();
        let translated: Vec<_> = (0..4).map(relettered).collect();
        raw.insert(1, page(99));

        let steps = align_pages(&raw, &translated);
        assert!(steps.contains(&PagePairing::RawOnly(1)), "{steps:?}");
        assert_eq!(matched(&steps), [(0, 0), (2, 1), (3, 2), (4, 3)]);
    }

    #[test]
    fn two_volumes_with_nothing_in_common_pair_nothing() {
        let raw: Vec<_> = (0..3).map(page).collect();
        let translated: Vec<_> = (20..23).map(page).collect();

        assert!(matched(&align_pages(&raw, &translated)).is_empty());
    }

    #[test]
    fn an_empty_side_leaves_every_page_unpaired() {
        let raw: Vec<_> = (0..2).map(page).collect();
        let steps = align_pages(&raw, &[]);
        assert_eq!(matched(&steps), []);
        assert_eq!(steps.len(), 2);
    }

    fn block(x: f32, y: f32, w: f32, h: f32) -> TextBlock {
        TextBlock {
            x,
            y,
            width: w,
            height: h,
            ..Default::default()
        }
    }

    const PAGE: (f32, f32) = (1500.0, 1200.0);

    #[test]
    fn the_same_balloon_pairs_though_its_text_changes_shape() {
        // Japanese runs down the balloon in a narrow column; the Vietnamese
        // that replaces it runs across in wide lines. Two very different
        // rectangles around the same place.
        let raw = [
            block(100.0, 100.0, 30.0, 120.0),
            block(800.0, 600.0, 28.0, 140.0),
        ];
        let translated = [
            block(795.0, 655.0, 120.0, 40.0),
            block(55.0, 145.0, 130.0, 34.0),
        ];

        let pairs = match_blocks(&raw, &translated, PAGE.0, PAGE.1);
        assert_eq!(pairs.len(), 2, "both balloons should pair: {pairs:?}");
        assert_eq!(pairs[0].translated, 1);
        assert_eq!(pairs[1].translated, 0);
    }

    #[test]
    fn a_balloon_with_no_counterpart_goes_unpaired_and_the_rest_survive() {
        // The translator merged two balloons into one. The page still yields
        // the lines it can, rather than being thrown away whole.
        let raw = [
            block(100.0, 100.0, 30.0, 120.0),
            block(200.0, 100.0, 30.0, 120.0),
            block(900.0, 700.0, 30.0, 120.0),
        ];
        let translated = [
            block(150.0, 110.0, 120.0, 40.0),
            block(905.0, 705.0, 120.0, 40.0),
        ];

        let pairs = match_blocks(&raw, &translated, PAGE.0, PAGE.1);
        assert_eq!(
            pairs.len(),
            2,
            "the far balloon and one of the pair: {pairs:?}"
        );
        assert!(pairs.iter().any(|p| p.raw == 2 && p.translated == 1));
    }

    #[test]
    fn balloons_on_opposite_sides_of_the_page_never_pair() {
        let raw = [block(50.0, 50.0, 30.0, 120.0)];
        let translated = [block(1400.0, 1100.0, 120.0, 40.0)];

        assert!(match_blocks(&raw, &translated, PAGE.0, PAGE.1).is_empty());
    }
}
