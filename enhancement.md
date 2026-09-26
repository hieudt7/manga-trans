# Enhancement log — text placement/translation accuracy (scan/kinnikuman-v1-v5 branch)

Context for a fresh session picking this up. Test data used throughout:
`test/cityhunter/v6/` (raw = `City hunter V01_0XX copy.png`, translated = `City hunter V01_0XX.png`).

## How to test a change (established workflow this session)

1. Rebuild: `cargo build --release -p koharu`.
2. Run headless (don't reuse a long-lived session across many hours — see
   "Known gotcha" below): `./target/release/koharu --headless --port=9997`.
3. Open a **fresh** folder session pointed at a temp dir containing only the
   raw file(s) you're testing (copy `... copy.png` files in) — do not reuse
   an old session/result dir that's accumulated state across restarts:
   ```
   curl -X POST http://127.0.0.1:9997/api/v1/folder/open-path \
     -H "content-type: application/json" -d '{"path": "<temp dir>"}'
   curl -X POST http://127.0.0.1:9997/api/v1/jobs/pipeline-folder \
     -H "content-type: application/json" -d '{
       "llmModelId": "gemini:gemini-3.1-flash-lite-preview",
       "llmApiKey": null, "llmBaseUrl": null, "llmTemperature": null,
       "llmMaxTokens": null, "llmCustomSystemPrompt": null,
       "language": "vi-VN", "shaderEffect": null, "shaderStroke": null,
       "fontFamily": null, "processWithCharacter": true
     }'
   ```
4. Poll `<temp dir>/result/` for the output file(s), then `pkill -f
   "target/release/koharu --headless"`.
5. **Test the single changed file first**, confirm the fix, only then run
   the full batch to check for regressions — cheaper feedback loop.
6. Crop with Python/PIL and view with the Read tool to actually look at the
   rendered bubble, rather than trusting log numbers alone — several bugs
   this session only became clear once the pixels were compared to the
   computed geometry (see the fixes below).

**`llmModelId` must be `provider:model`, e.g.
`"gemini:gemini-3.1-flash-lite-preview"`.** A bare model id
(`"gemini-3.1-flash-lite-preview"`) is parsed as a *local* `ModelId` and
fails with `Matching variant not found`; an empty/missing `llmModelId`
silently skips `LlmGenerate` entirely (the page renders untranslated, no
error) — both easy to trip over when curling the API by hand and can look
exactly like a hang if you don't check the job's error field / log for
"sending to LLM" and "Gemini translate" lines.

**Known gotcha, partially explained this session:** reusing one headless
instance / folder session across a very long series of restarts and
`rm`+retrigger cycles produced a run where the server logged `folder result
saved path=...` for a file, but the file was gone moments later and a
following job call logged `all N images already processed, nothing to do`
even though it wasn't there. Separately, several runs this session simply
froze — process alive, ~0% CPU, zero further log output for minutes — right
around the first Gemini call, with no error and no retry/rotation log line
(compare to a healthy run, which logs a key-rotation line almost
immediately). **One root cause found and fixed:** `koharu-http/src/http.rs`
built the shared `reqwest::Client` (used for every Gemini call) with **no
timeout at all** — a stalled connection blocked forever with no error, which
would produce exactly this symptom. Fixed by adding
`.timeout(60s).connect_timeout(15s)` to the client builder. **Still
unconfirmed** whether this was the *whole* explanation — a couple of hangs
during this session's testing persisted even after the fix and then
resolved on a later retry with no code change, so there may be a second,
environmental factor (this machine specifically) still unaccounted for.
**Workaround that reliably works regardless:** always open a brand-new
folder session (fresh temp input dir, fresh port) rather than reusing one
that's been through many restarts in the same conversation, and if a run
looks frozen, kill it and retry fresh rather than waiting indefinitely.

## Fixed this session (all verified against real render output, not just logs)

### 1. Gemini OCR swapped text between two close-together regions
**File:** `koharu-pipeline/src/ops/speaker_attribution.rs`
**Symptom:** two adjacent speech-bubble text blocks on one page got each
other's translated text (a short interjection landing in the big bubble, the
long line landing in the small one).
**Root cause:** `read_page`'s prompt gave Gemini fractional coordinates only;
telling it not to swap wasn't reliable for a page with two regions close
together — verified by capturing the raw Gemini reply and diffing it against
the given region order.
**Fix:** `encode_annotated_jpeg` (`speaker_attribution.rs:779`) burns a red
outline + yellow number tag onto the image itself for every region
(`annotate_regions`, `:726`) before sending it to Gemini — "set-of-marks"
style grounding — instead of relying on it to map coordinates in its head.
Prompt updated to say the tags are authoritative. `read_page` calls this
instead of the old `encode_jpeg` at `:485`.

### 2. Balloon detector merges two genuinely-separate balloons ("shared" split was naive)
**File:** `koharu-ml/src/facade.rs`, function `refit_text_blocks_to_balloons`
(`:1239`).
**Symptom:** two text blocks the ML balloon detector boxed as one "shared"
detection got a perpendicular-bisector split at the midpoint between their
centres. For balloons of very different size (a small reaction bubble far
from a big narration bubble, or a genuinely pinched hourglass balloon with
lobes of unequal size) the midpoint lands inside the smaller lobe, or the
detector's box just plain merged two unrelated balloons.
**Fix, layered:**
- `owners_are_bridged` (`:1114`, uses `BRIDGE_ERODE_RADIUS=14`,
  `MAX_BACKGROUND_BRIDGE_PX=10` at `:1094`/`:1084`) — erodes the mask along
  the line between two owners' centres; only treat them as one balloon if a
  real (not paper-thin) connection survives.
- `waist_fraction` (`:1168`) + `owns_by_waist` (`:1229`) — when bridged, cut
  at the mask's actual narrowest point ("waist") between the two centres,
  not the plain midpoint. Handles a genuinely pinched hourglass balloon
  correctly (2 lines with a pause between them, one shape).
- `ASPECT_DISTORTION_MARGIN=1.2` (`:1103`) — final sanity check: if the
  resulting split box is far more elongated than the block's own originally
  detected box, reject the split and keep the block at its original
  detection rather than a distorted sliver.
- New tests: `balloons_the_detector_merely_boxed_together_are_not_forced_into_a_split`,
  `a_pinched_hourglass_balloon_is_split_at_the_waist_not_the_midpoint`.

### 3. Renderer had its own, un-fixed copy of the same naive split
**File:** `koharu-renderer/src/text/latin.rs`, `clip_box_to_nearest_owner`
(`:352`).
**Symptom:** even after fix #2, one bubble in a pinched-hourglass pair still
came out with a big empty gap before its content, because the *renderer's*
row-based "follow the shape" system (`clear_space_rows`) clips its own
flood-filled search region using a **separate**, still-naive midpoint-based
function — fix #2 only touched the ML crate's copy.
**Fix:** `mask_waist_fraction` (`:294`) — same waist-finding technique,
reimplemented against the renderer's own `bubble_map: &GrayImage` (can't
share code across the two crates without more plumbing; documented as
intentional duplication in the doc comment). `clip_box_to_nearest_owner`
now takes `mask: Option<&GrayImage>` and cuts at the waist when given one,
falling back to the midpoint otherwise. Both call sites in
`koharu-renderer/src/facade.rs` updated to pass `bubble_map`. New test:
`a_pinched_hourglass_is_clipped_at_the_waist_not_the_midpoint`.

### 4. Vertical centring inside an irregular (shape-following) balloon region
**File:** `koharu-renderer/src/facade.rs`, inside `render_text_block`, the
`if let Ok(mut shaped_layout) = set(&spans, ...)` block (currently starting
around `:569`).
**Symptom (two distinct bugs found here):**
- A shape that narrows going down (top lobe of a pinched balloon) could
  "successfully" re-centre by re-laying out from a lower start point, but
  the re-wrap into narrower rows down there bloated the content height
  back up (78px of content came back needing 293px) — reported as
  `recentred=true` while visually still pinned near the top.
- Separately, a `reset_shout` re-layout (for oversized/loud lettering) could
  silently replace an already-correctly-centred layout with a fresh
  never-centred one, because the shout-reset ran *after* the centring step.
**Fix (redesigned per explicit user direction — do not force a rectangle,
keep following whatever shape was traced, but stop "guessing via re-layout"
as the primary strategy):**
1. Compute the plain centred offset (`(room − content) / 2`) on the
   **already correctly-wrapped** top-anchored layout.
2. Ask the *actual shape data* (`RowSpans::band`, which works for any
   traced shape — round, square, zigzag, whatever `clear_space_rows` found)
   whether the region the shift would move the content into is wide enough
   for the widest line. If yes, shift the finished layout directly — cheap,
   exact, and doesn't touch how lines wrapped.
3. Only if that band-check fails (shifting would run a line into a
   genuinely narrower part of the shape) does it fall back to the old
   "re-layout from a lower start" trick — now still guarded by the
   `RESETTLE_GROWTH_TOLERANCE_PX=24.0` (`:862`) growth-rejection check from
   the same investigation.
4. `reset_shout`'s replacement (if it fires) now correctly clears the
   "already centred" flag so the final fallback baseline-shift still runs
   against whatever layout is actually about to be painted.
5. `center_layout_vertically` (`:1008`) is the shared baseline-shift
   primitive used by both this shape-fit path and the plain-rectangle path.

### 5. First/last line of a shape-following block jammed into a corner sliver
**Files:** `koharu-renderer/src/text/latin.rs`, `source_text_rows` (`:520`,
used for the "outside a balloon, follow the clear space the source text left
behind" path) and `clear_space_rows` (`:901`, used for the in-balloon
flood-filled shape path) — both share a new helper `trim_narrow_edge_rows`.
**Symptom:** in a rounded/oval balloon, the very top (or bottom) row of the
traced shape can be a razor-thin, off-centre sliver — the curvature of the
cap intersected only a few pixels of clear space, offset hard to one side —
while every other row is nearly the shape's full width. The layout engine
(correctly, mechanically) wrapped the first word alone into that sliver,
producing a line that reads as badly mis-centred relative to the rest of the
block (reported by the user as "MỘT" — the first word of a bubble — sitting
jammed against the right edge while the rest of the text was centred fine).
Confirmed via live per-line debug logging (temporarily added at all three
`render_text_block` return paths, since a naive assumption about which path
a block used was wrong once — see "outside a balloon" is the actual path for
`keep_source_size` blocks, not the plain rectangle path): line 0's span was
`(x=105, width=76)` against a `layout_box_w=197`, while every other line's
span was `(x≈0-3, width≈183-195)`. Horizontal-centring fixes (see below)
cannot help this — the line is clamped to its own row span, which *is* the
problem.
**Fix:** `trim_narrow_edge_rows` (`:520`-ish, shared by both functions) drops
a **short** (≤ `MAX_EDGE_TRIM_ROWS = 24`) contiguous run of rows from the
start/end of the traced shape if their width is under
`EDGE_ROW_MIN_WIDTH_FRACTION = 0.5` of the shape's widest row — i.e. treats
it as balloon-cap rounding, not real room, and starts the text where the
balloon has actually opened up. Deliberately **does not** touch a long run
(a genuine narrow leg of an L/U/Z shape, which needs to stay exactly as
traced — `the_shape_turns_an_l_where_the_drawing_takes_the_corner` is the
regression test guarding this: it has a 75-row narrow leg that must survive
untouched). Verified against real render output: the previously-reported
`City hunter V01_020.png` "MỘT" bubble and `V01_017.png` "CÔ HUNG DỮ" bubble
(originally reported together as "khoảng trắng lớn phía trên" — same root
cause, different symptom) both render correctly now; results copied into
`test/cityhunter/v6/City hunter V01_017 copy.png` and `V01_020 copy.png`.

### 6. Shared HTTP client had no request timeout (see gotcha note above)
**File:** `koharu-http/src/http.rs`. Added `.timeout(Duration::from_secs(60))`
and `.connect_timeout(Duration::from_secs(15))` to the `reqwest::Client`
builder underlying every Gemini call. A stalled connection previously
blocked the whole pipeline forever with no error and no log output; now it
surfaces as a retryable error within the existing `RetryTransientMiddleware`
policy. See the gotcha note above for the caveat that this may not be the
*only* cause of hangs observed this session.

### 7. Rotated/slanted text is invisible to `PPDocLayoutV3`, so it never becomes a text block, never reaches Gemini, and never gets translated or erased
**Files:** `koharu-ml/src/facade.rs` (`Model` struct, `detect`, new
`uncovered_ink_regions`) and `koharu-ml/src/comic_text_detector/` (pre-existing,
previously only used for its segmentation-only path via `prefetch_segmentation`
and `extract_text_block_regions` — its full `inference()`, YOLOv5+DBNet,
quad/rotation-aware, was dead code for finding new text blocks until now).
**Symptom, root cause, and two disproven theories along the way:** all still
documented under "Known, not yet fixed" below — kept there rather than
duplicated, since that write-up already has the measurements
(`KOHARU_DEBUG_MASKS` row-scan, the MIR/bbox overlay, the `KOHARU_DEBUG_REGIONS`
+ `KOHARU_DEBUG_VISION` dumps) that pinned the cause on `PPDocLayoutV3` missing
the region entirely, not on anything in this codebase's OCR-matching,
balloon-mask, or MIR logic.
**Fix:** rather than trying to make `PPDocLayoutV3` (a general document-layout
model) understand rotation, or teach the balloon-mask/MIR logic to guess at
content it was never given, a second, already-in-tree detector
(`mayocream/comic-text-detector`, YOLOv5+DBNet, whose `Quad` output is
genuinely rotation-aware — confirmed by a throwaway diagnostic test,
`dump_comic_text_detector_regions`, which found the exact rotated balloon
`PPDocLayoutV3` missed, `rotation_deg: 37.27693°`, `detector: "ctd"`) now
**only ever runs as a rescan when there's a concrete sign of a miss**, so the
common case (no miss) pays nothing extra:
1. `uncovered_ink_regions` (`facade.rs`) reuses the exact technique
   `lama::leftover_mask_regions` already uses for a different purpose
   (dilate the segmentation mask to merge glyphs into runs, connected-component
   label, size-filter) but inverted: instead of keeping ink *near* an existing
   block (finishing an inpainting job), it keeps ink *far from every* block —
   a run nothing has claimed at all.
2. `detect()` calls this once per page (mask and text blocks it already has,
   no extra model cost) right after the segmentation mask is built. Empty
   result → nothing else happens, same cost as before this fix.
3. Only when it finds something does `Model.comic_text_detector` (a new
   field, loaded the same optional/graceful-degradation way as
   `bubble_detector`) run its one `inference()` pass on the page. Any of its
   detected blocks whose centre falls inside an uncovered region is appended
   to `doc.text_blocks` — `PPDocLayoutV3`'s own blocks are never touched or
   replaced, only added to.
4. Everything downstream (translation, balloon MIR refit, font detection,
   rendering) treats the recovered block exactly like any other; no
   special-casing needed there, `TextBlock::ensure_id` assigns it a real id
   the same as every other detector-sourced block already relies on.
**Verified against real render output:** re-ran the full folder pipeline on
`test/cityhunter/v6/City hunter V01_017.png`. Log:
`comic-text-detector rescan for text PPDocLayoutV3 missed uncovered=9 added=1`
— 9 ink clusters cleared the distance gate (some of that is probably noise;
the gate is deliberately generous since firing it costs one detector pass,
not several), but exactly 1 recovered block matched, and it was the right
one: the round balloon now renders with a clean erased background and
"GIỐNG HỆT MẶT CÔ VẬY." centred inside it, no raw source glyphs left behind.
Result copied into `test/cityhunter/v6/City hunter V01_017 copy.png`.
New tests: `uncovered_ink_regions_finds_a_run_far_from_every_block`,
`uncovered_ink_regions_ignores_a_run_touching_an_existing_block`,
`uncovered_ink_regions_ignores_specks_below_the_pixel_floor`.
**Loose end, not chased further this session:** the refit log for the
balloon spanning both the caption and the recovered round-balloon block no
longer prints a `refit balloon=7 …` line at all post-fix (it did,
`shared=false`, before this fix existed). The render looks correct for both
blocks either way — likely because the renderer's own shape-following
placement (fix #4/#5, operates on `bubble_map` directly at render time,
independent of whether the ML-side MIR refit locked a box) covered whatever
this refit step didn't — but the exact code path taken (shared/waist-split
vs. the "share too small, keeping detection" fallback) wasn't traced. Worth
five minutes with `tracing::debug` turned on if balloon-mask work touches
this area again.

### Also added this session (minor, not a bug fix)
`align_layout_horizontally` (`koharu-renderer/src/facade.rs:970`) got a
shared-centring-axis pass for the `TextAlign::Center` + row-span case
(widest line sets the centre, other lines clamp to it within their own row).
Harmless and covered by a new test, but turned out to be a no-op for the
actual bug reported this session (see #5 above) — the real fix was in how
the rows themselves are computed, not how lines are centred within them.
Left in since it's still correct behaviour for whatever case it does apply
to, just don't expect it to explain future corner-sliver symptoms.

All of `cargo test -p koharu-ml -p koharu-renderer --lib` and
`cargo check --workspace` were clean after every fix above (re-verify before
trusting this doc if more changes have landed since).

## Known, not yet fixed

**Empty text block silently skips the local-OCR fallback.**
`koharu-pipeline/src/ops/folder.rs:431` — `covers_every_block` only checks
`HashMap::contains_key`, not whether the value is non-empty. When Gemini's
`read_page` returns exactly as many entries as regions but leaves one
region's `text` empty (distinct from actually swapping — this is Gemini
just not transcribing a small/awkward region), `covers_every_block` is
still `true`, so the local OCR fallback (`:435`) never runs for that block,
and it renders with no translation (raw source text stays visible).
Reproduces on `test/cityhunter/v6/City hunter V01_017.png`: the small tail
bubble reading "像你這樣莽撞的戀人..." near the bottom is still untranslated.
Proposed fix (not yet implemented): treat a present-but-empty `text` as *not*
covered, so that block still gets a shot at local OCR.

**Multi-column vertical caption text on a black panel (no balloon) gets
merged/misaligned.**
Reproduces on `test/cityhunter/v6/City hunter V01_020.png`, the black
night-street panel (raw has 4 separate vertical Chinese caption columns of
different lengths directly on the black art, no balloon/white background —
"我已經等不了那麼久...癌細胞正吞噬...我的生命"). Only 6 text regions were
detected for the whole page (`vision OCR result blocks=6`), meaning this
cluster of 4 columns was folded into a single region. After translation, the
Vietnamese text lands on top of the erased-text mask (the white glyph-shaped
patches left by text removal) but doesn't track the columns: the leftmost
column ends up completely blank (no translated text placed there at all),
while another fragment gets an olive/yellow background pill that overlaps
onto the black background instead of sitting only over white. Likely because
the shape-following renderer (see fix #4/#5 above) assumes one continuous
clear-space region; here the "clear space" is several disconnected
glyph-shaped islands on black, which isn't the shape those fixes were
designed for. Not yet investigated at the code level — flagging from visual
inspection only.

**FIXED — see fix #7 above.** The upstream layout detector misses
rotated/slanted text entirely, so it never becomes a text block, never
reaches Gemini, and never gets translated or erased — this was the actual
cause of the V01_017 "small tail bubble" bug, superseding two earlier,
disproven theories from this same investigation. Kept in full below —
including the measurements behind the fix and a record of what was ruled
out and how — since fix #7's write-up points back here rather than
duplicating it.

Reported by user on `test/cityhunter/v6/City hunter V01_017.png`, bottom-left
panel: a plain caption ("你真兇...我的臉需要康復治療。", rendered fine) sits
next to a small round balloon holding a separate, visibly slanted/rotated
aside ("像你這樣莽撞的戀人..."), which stays untranslated in every render.

Two theories were tried and **disproven by direct measurement** before
finding the real cause — recorded here so a future session doesn't repeat
the same dead ends:
1. *"Single-owner balloon mask has two lobes, waist-split never runs"* —
   disproven by dumping the actual balloon mask (`KOHARU_DEBUG_MASKS`,
   dumping `BubbleBox.mask` to PNG) and scanning it row-by-row: every one of
   its 517 rows has exactly one contiguous foreground run (never two), so
   there is no thin neck anywhere for an erosion-based split to find. A fix
   built on this theory (`single_owner_lobe_owns` in
   `koharu-ml/src/facade.rs`, erosion + connected-components + BFS regrow,
   with a passing unit test on a *synthetic* thin-neck mask) was written,
   verified to compile and pass `cargo test -p koharu-ml`, then **reverted**
   once the real mask (measured, not synthetic) showed no neck to erode —
   re-running the actual page confirmed the fix changed nothing (identical
   `mir_x/mir_w` before and after). Do not re-attempt this specific approach
   without first dumping and measuring the real mask.
2. *"The MIR box itself is mis-sized/off-centre"* — disproven by overlaying
   the logged balloon bbox and MIR box on the raw page art (PIL
   `ImageDraw.rectangle`, red = MIR, yellow = balloon bbox). The MIR
   (`mir_x=587, mir_y=1816, mir_w=157, mir_h=242`) turned out to match the
   original 3-column caption's own extent almost exactly, stopping right
   where the character's hand starts. The geometry is correct; the
   "off-centre" impression was just the untranslated round balloon sitting
   in the remaining panel space next to it, making the translated block
   look mis-placed relative to the whole panel even though it is correctly
   sized for its own content.

**Confirmed root cause**, by dumping what actually gets sent to and back
from Gemini (temporary `KOHARU_DEBUG_REGIONS` logging every raw
`LayoutRegion` from `detect_layout_regions`, and `KOHARU_DEBUG_VISION`
logging `read_page`'s full prompt `box_list` and raw Gemini reply — both
since reverted, not in the tree): the page produced exactly **9** raw layout
regions in total across both detector tiles, sent as exactly 9 boxes to
Gemini, which answered with exactly 9 entries — a clean 1:1 match, no
merging on Gemini's side at all. **None of the 9 regions is anywhere near
the round balloon's lower portion** (the nearest, region/block 7, only
reaches down to y≈1879–2092, well short of the balloon's full extent down to
y≈2333). The rotated aside text is simply never proposed as a region by
`PPDocLayoutV3` (`koharu-ml/src/pp_doclayout_v3`) in the first place — a
detection miss in the pretrained layout model itself, not a bug in this
codebase's OCR-matching, balloon-mask, or MIR logic.

Consequence: because no text block ever exists for that aside, none of the
existing "did every block get translated" safeguards ever run for it — not
the empty-text bug above (that needs a block that *was* detected), and not
`leftover_mask_regions` (`koharu-ml/src/lama/mod.rs`) either, which by
design only extends inpainting to stray ink *near an existing block's
window* (see its own test,
`leftover_regions_leave_marks_no_block_window_reaches`) and explicitly
leaves ink alone that no block's window reaches.

**Fixed — see fix #7 above** for the implementation: `uncovered_ink_regions`
notices exactly this ("a run of glyphs the mask segmenter marks as text but
that no detected `TextBlock`'s window reaches at all"), gating a
`ComicTextDetector` rescan pass that synthesizes the missing block instead
of leaving it as an inpainting-only leftover region.

## Session context worth knowing

- The Rust workspace here is a fork/local build of "Koharu", a manga
  translation app (Tauri + Next.js UI + Rust ML/render pipeline). This
  session only touched the Rust backend (`koharu-ml`, `koharu-renderer`,
  `koharu-pipeline`), never the UI.
- Gemini vision (`gemini-3.1-flash-lite-preview`) is used for both the
  page-OCR+speaker-attribution call and the translation call — 2 calls per
  page, by design (see `speaker_attribution.rs` module docs).
- `character_libraries/<slug>.json` holds per-series CCIP face embeddings
  (in-tree, not under Application Support) — City Hunter's is
  `character_libraries/city-hunter-deluxe-edition-raw.json`, already
  populated, so CCIP speaker-face matching is live for this test data, not a
  no-op.
