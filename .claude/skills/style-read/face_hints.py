"""Pre-compute a CV face/balloon hint for every page, once, before reading.

    python3 .claude/skills/style-read/face_hints.py <folder> [--raw-only]

manga-page-reader estimates a face box by eye, from the whole page, and never
sees the crop it produces — that is the root cause behind most of the bad
face crops this series turned up: a box a little off does not land on a
slightly wrong part of a face, it lands on the speech bubble beside it. A
real detector (deepghs/anime_face_detection) does not have that problem — it
was measured against this same series at 9/9 correct, tight face crops on a
page a blind estimate had gotten wrong.

So this runs it first, once per page, and writes what it found next to the
page's other reading copies:

    <work>/face_hints/<page id>.json

`{"faces": [{"x","y","width","height","confidence"}, ...],
  "balloons": [{"x","y","width","height","confidence","nearestFace"}, ...]}`

— boxes as fractions of the page, the same convention a hand-written `box`
already uses, so a hint can be copied straight into a `faces` entry.
`nearestFace` is a geometry guess (closest face centre to each balloon
centre), not a verified match — manga-page-reader still has to confirm it is
the right character before using it, the same as any box it drew itself.

This is optional and additive: a page with no hint file (model not built,
not installed, or detection genuinely found nothing) is read exactly as
before — by eye. Nothing about the read depends on this having run.

Needs the `face-hint` binary (`cargo build -p koharu-ml --bin face-hint`,
`--release` for real series — debug is noticeably slower per page) and the
anime face detector + bubble detector models
(`python3 tools/export_ccip.py --face-only`; the bubble detector is exported
separately — see its own tool). Missing models are reported once and the run
continues without hints, rather than failing the whole prepare step over an
optional accuracy improvement.
"""

import json
import os
import subprocess
import sys

import progress

BINARY_NAMES = ("face-hint",)


def find_binary():
    """The release build if it exists — debug is about 7x slower per page
    (measured: 7.2s vs 1.0s), almost entirely model-load overhead paid fresh
    on every page since each is its own process. That is ~4 hours against
    ~10 minutes over a 563-page series. Fall back to debug rather than
    refuse to run, but say so, since the run is about to be slow."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for profile in ("release", "debug"):
        for name in BINARY_NAMES:
            candidate = os.path.join(root, "target", profile, name)
            if os.path.isfile(candidate):
                if profile == "debug":
                    print(
                        "using the debug build (no release build found) — about 7x slower "
                        "per page; for a full series, cargo build -p koharu-ml --bin face-hint --release",
                        file=sys.stderr,
                    )
                return candidate
    return None


def main():
    args = sys.argv[1:]
    raw_only = "--raw-only" in args
    args = [a for a in args if a != "--raw-only"]
    if len(args) != 1:
        sys.exit(__doc__)

    root = os.path.abspath(args[0])
    work = os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")
    manifest = progress.read_json(os.path.join(work, "pages.json"), None)
    if manifest is None:
        sys.exit("missing pages.json — run prepare.py first")

    binary = find_binary()
    if not binary:
        sys.exit(
            "face-hint binary not found — run: cargo build --release -p koharu-ml --bin face-hint"
            "\nSkip this script entirely to read without CV hints; nothing else depends on it."
        )

    hints_dir = os.path.join(work, "face_hints")
    os.makedirs(hints_dir, exist_ok=True)

    pages = manifest["pages"]
    done = skipped = failed = 0
    model_warning_shown = False
    for page in pages:
        source = page.get("rawSource") or page.get("transSource")
        if not source:
            continue
        out_path = os.path.join(hints_dir, f"{page['id']}.json")
        if os.path.exists(out_path):
            skipped += 1
            continue
        result = subprocess.run(
            [binary, "--input", source], capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            failed += 1
            if not model_warning_shown:
                print(f"  face-hint failed on {page['id']}: {result.stderr.strip()[-300:]}", file=sys.stderr)
                print(
                    "  (printing this once — if every page fails the same way, a model is "
                    "probably missing from the cache; the run continues without hints)",
                    file=sys.stderr,
                )
                model_warning_shown = True
            continue
        try:
            scan = json.loads(result.stdout)
        except json.JSONDecodeError:
            failed += 1
            continue
        progress.write_json(out_path, scan)
        done += 1

    print(f"face hints: {done} written, {skipped} already had one, {failed} failed")
    if done or skipped:
        total_faces = 0
        for page in pages:
            hint = progress.read_json(os.path.join(hints_dir, f"{page['id']}.json"), None)
            if hint:
                total_faces += len(hint.get("faces") or [])
        print(f"  {total_faces} faces detected across {done + skipped} hinted pages")


if __name__ == "__main__":
    main()
