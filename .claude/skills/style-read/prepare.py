"""Get a reference volume ready for /style-read.

    python3 .claude/skills/style-read/prepare.py <folder> [--raw-only] [--pages N] [--only ID,ID]
    python3 .claude/skills/style-read/prepare.py <series folder> --from 1 --to 5 [--raw-only] …

<folder> holds the published translation in trans/ (or directly inside it) and,
when available, the original in raw/. A folder with only raw/ is read from the
Japanese alone. --raw-only reads only the original even when a translation is
there (its images in raw/, or directly inside the folder). --pages reads only
the first N pages, --only reads the named page ids; both are for trying the
command out cheaply.

With --from/--to, <folder> is a series: its sub-folders are volumes, and the
volumes from --from to --to (volume numbers such as 第01巻 → 1, or part of a
folder name) are read as one run with one cast. In a volume, only the images
directly inside it (or in its raw/ and trans/) are read; other sub-folders are
left alone. Page ids get the volume in front: v01-0291.

Lines the two volumes up page by page and writes reading copies of every
matched pair to <folder>/style_scan/claude/, with a manifest listing them and
which pages already have notes.

The alignment is the one in koharu-ml/src/bilingual/mod.rs, constants and all:
greyscale thumbnails compared against their own mean, then a sequence
alignment in which a page present on one side only is a gap. Keep the two in
step if either changes.

Needs Pillow. Uses numpy when it is installed, and plain Python otherwise at a
smaller thumbnail (the measured corpus still separated true from false pairs
at 16px; 128px leaves a wide margin).
"""

import json
import os
import re
import sys

from PIL import Image

import progress

MATCH_FLOOR = 0.69
GAP_COST = 0.05
# 1568 is where a page stops getting clearer to the reader: above it the image
# is scaled back down before it is looked at, so the tokens buy nothing.
READING_SIDE = 1568
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

try:
    import numpy as np
except ImportError:  # plain Python fallback
    np = None

THUMBNAIL_SIDE = 256 if np is not None else 128


def list_pages(folder):
    if not os.path.isdir(folder):
        return []
    names = sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, name))
    )
    return [os.path.join(folder, name) for name in names]


def signature(path):
    image = Image.open(path).convert("L").resize(
        (THUMBNAIL_SIDE, THUMBNAIL_SIDE), Image.BILINEAR
    )
    values = list(image.getdata())
    mean = sum(values) / len(values)
    centred = [v - mean for v in values]
    norm = sum(v * v for v in centred) ** 0.5
    if norm == 0:
        return None  # a blank page matches nothing
    if np is not None:
        return np.asarray(centred, dtype=np.float32) / norm
    return [v / norm for v in centred]


def similarities(raw, translated):
    if np is not None:
        blank = np.zeros(THUMBNAIL_SIDE * THUMBNAIL_SIDE, dtype=np.float32)
        a = np.stack([s if s is not None else blank for s in raw])
        b = np.stack([s if s is not None else blank for s in translated])
        return np.clip(a @ b.T, 0.0, 1.0).tolist()
    import operator

    table = []
    for x in raw:
        row = []
        for y in translated:
            if x is None or y is None:
                row.append(0.0)
            else:
                row.append(min(1.0, max(0.0, sum(map(operator.mul, x, y)))))
        table.append(row)
    return table


def align(sim, n, m):
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] - GAP_COST
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] - GAP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j] = max(
                score[i - 1][j - 1] + sim[i - 1][j - 1] - MATCH_FLOOR,
                score[i - 1][j] - GAP_COST,
                score[i][j - 1] - GAP_COST,
            )

    steps, i, j = [], n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and abs(score[i][j] - (score[i - 1][j - 1] + sim[i - 1][j - 1] - MATCH_FLOOR))
            < 1e-6
        ):
            steps.append(("matched", i - 1, j - 1, sim[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and abs(score[i][j] - (score[i - 1][j] - GAP_COST)) < 1e-6:
            steps.append(("raw_only", i - 1, None, None))
            i -= 1
        else:
            steps.append(("translated_only", None, j - 1, None))
            j -= 1
    steps.reverse()
    return steps


def page_id(path, index, taken):
    """A short ASCII name: the source pages are called 表紙0297.JPG, and a
    name carrying that does not open from every tool."""
    stem = os.path.splitext(os.path.basename(path))[0]
    ident = "".join(c for c in stem if c.isascii() and (c.isalnum() or c in "-_"))
    if not ident or ident in taken:
        ident = f"p{index:04d}"
    taken.add(ident)
    return ident


def fresh(target, source):
    """A copy already made at the size we want now."""
    if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(source):
        return False
    try:
        with Image.open(target) as made:
            return max(made.size) >= READING_SIDE - 1 or max(made.size) == max(
                Image.open(source).size
            )
    except OSError:
        return False


def half_is_fresh(target, source):
    """A half is remade unless it has already been brought up to size."""
    if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(source):
        return False
    try:
        with Image.open(target) as made:
            return max(made.size) >= READING_SIDE - 1
    except OSError:
        return False


def fit(image, enlarge=False):
    """Bring an image to the size the reader is given.

    Scans of this vintage are often smaller than that, and a half-spread from
    one is very small indeed. Enlarging does not add detail, but it does spread
    the lettering over more of the picture the reader is shown, which is what
    cropping a panel used to do by hand."""
    scale = READING_SIDE / max(image.size)
    if scale < 1 or (enlarge and scale > 1):
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
        )
    return image


def reading_copy(source, target):
    """The page as the reader sees it, plus each half on its own.

    A scan is a two-page spread, so at one image a spread each page is only
    half as wide as the reader is given — small enough that readers used to
    crop panels to make out the lettering, which cost more than the page. The
    halves are the same spread at twice the size, to fall back on; the spread
    stays the one the face boxes are fractions of.
    """
    made = []
    if not fresh(target, source):
        image = Image.open(source).convert("RGB")
        fit(image).save(target, quality=90)
        made.append(target)
    stem, extension = os.path.splitext(target)
    halves = {}
    for suffix, side in (("_r", "right"), ("_l", "left")):
        halves[side] = stem + suffix + extension
    if not all(half_is_fresh(h, source) for h in halves.values()):
        image = Image.open(source).convert("RGB")
        middle = image.width // 2
        # Right half first: a spread is read right to left.
        fit(image.crop((middle, 0, image.width, image.height)), enlarge=True).save(
            halves["right"], quality=90
        )
        fit(image.crop((0, 0, middle, image.height)), enlarge=True).save(
            halves["left"], quality=90
        )
    return halves


def work_dir(root, raw_only):
    """Notes read from the Japanese alone say different things from notes read
    against a translation, so the two never share a folder: a later run of the
    other kind would take them as done."""
    return os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")


VOLUME_NUMBER = (
    re.compile(r"第\s*(\d+)\s*[巻卷話集]"),
    re.compile(r"(?:\bvol\.?|\bv|tập|tap)\s*(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)(?!.*\d)"),
)


def volume_number(name):
    """第01巻 → 1, `Kinnikuman II Sei v21-25` → 21, `tap3` → 3."""
    for pattern in VOLUME_NUMBER:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def pick_volumes(series, first, last):
    """The volume folders of a series from `first` to `last`, both included.

    Each bound is a volume number (matched against 第N巻, vN, tậpN or the last
    number in the folder name) or part of a folder name. Folders the app and
    this command write (style_scan, character_scan) are never volumes.
    """
    names = sorted(
        (
            name
            for name in os.listdir(series)
            if os.path.isdir(os.path.join(series, name))
            and not name.startswith((".", "_"))
            and name not in ("style_scan", "character_scan")
        ),
        key=natural_key,
    )

    def position(bound, default):
        if not bound:
            return default
        if bound.isdigit():
            hits = [i for i, n in enumerate(names) if volume_number(n) == int(bound)]
        else:
            hits = [i for i, n in enumerate(names) if bound.lower() in n.lower()]
        if not hits:
            sys.exit(f"No volume folder in {series} matches '{bound}'. Folders: " + ", ".join(names))
        if len(hits) > 1:
            sys.exit(f"'{bound}' matches several folders: " + ", ".join(names[i] for i in hits))
        return hits[0]

    start = position(first, 0)
    end = position(last, len(names) - 1)
    if start > end:
        sys.exit(f"'{first}' comes after '{last}'")
    return [os.path.join(series, name) for name in names[start : end + 1]]


def volume_steps(root, raw_only):
    """What to read in one folder: (raw pages, translated pages, raw_only,
    alignment steps). Only images directly in the folder, raw/ or trans/ are
    taken; any other sub-folder is ignored."""
    raw = list_pages(os.path.join(root, "raw"))
    translated = list_pages(os.path.join(root, "trans"))
    raw_only = raw_only or (bool(raw) and not translated)
    if raw_only:
        translated = []
        if not raw:
            # The original on its own may sit directly in the folder.
            raw = list_pages(root)
        if not raw:
            sys.exit(f"No original pages in {root}/raw/ or in the folder itself.")
    elif not translated:
        # A translation on its own may sit directly in the folder.
        translated = list_pages(root)
        if not translated:
            sys.exit(f"No pages in {root}/raw/, {root}/trans/ or the folder itself.")

    name = os.path.basename(root)
    if raw_only:
        print(f"{name}: {len(raw)} original pages, read without a translation", flush=True)
        steps = [("raw_only", r, None, None) for r in range(len(raw))]
    elif raw:
        print(f"{name}: aligning {len(raw)} raw with {len(translated)} translated pages…", flush=True)
        sim = similarities([signature(p) for p in raw], [signature(p) for p in translated])
        steps = align(sim, len(raw), len(translated))
    else:
        print(f"{name}: {len(translated)} translated pages, no original", flush=True)
        steps = [("translated_only", None, t, None) for t in range(len(translated))]
    return raw, translated, raw_only, steps


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--pages", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--from", dest="first", default="")
    parser.add_argument("--to", dest="last", default="")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    root = os.path.abspath(args.folder)
    series = bool(args.first or args.last)
    volumes = pick_volumes(root, args.first, args.last) if series else [root]

    plans = [(volume, *volume_steps(volume, args.raw_only)) for volume in volumes]
    raw_only = plans[0][3]
    if any(plan[3] != raw_only for plan in plans):
        sys.exit("Some volumes have a translation and some do not; read them apart, or pass --raw-only.")

    out = work_dir(root, raw_only)
    pages_dir = os.path.join(out, "pages")
    notes_dir = os.path.join(out, "notes")
    # Where each batch leaves what it found; progress.py folds it into the cast.
    updates_dir = os.path.join(out, progress.UPDATES_DIR)
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(notes_dir, exist_ok=True)
    os.makedirs(updates_dir, exist_ok=True)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    pages, skipped, taken, listed = [], [], set(), []
    raw_total = translated_total = 0
    for volume, raw, translated, _, steps in plans:
        raw_total += len(raw)
        translated_total += len(translated)
        number = volume_number(os.path.basename(volume)) if series else None
        prefix = "" if not series else (f"v{number:02d}-" if number is not None else f"v{len(listed) + 1:02d}-")
        count = 0
        for kind, r, t, score in steps:
            if kind == "raw_only" and not raw_only:
                # Nothing translated on it to learn from.
                skipped.append({"side": "raw", "file": os.path.relpath(raw[r], root)})
                continue
            if args.pages and len(pages) >= args.pages:
                break
            base = page_id(raw[r], r, set()) if raw_only else page_id(translated[t], t, set())
            ident = prefix + base
            if ident in taken:
                ident = f"{prefix}p{(r if raw_only else t):04d}"
            taken.add(ident)
            if only and ident not in only and base not in only:
                continue
            trans_copy = trans_halves = None
            if t is not None:
                trans_copy = os.path.join(pages_dir, f"{ident}_trans.jpg")
                trans_halves = reading_copy(translated[t], trans_copy)
            raw_copy = raw_halves = None
            if r is not None:
                raw_copy = os.path.join(pages_dir, f"{ident}_raw.jpg")
                raw_halves = reading_copy(raw[r], raw_copy)
            note = os.path.join(notes_dir, f"{ident}.md")
            pages.append(
                {
                    "id": ident,
                    "volume": volume,
                    "raw": raw_copy,
                    "trans": trans_copy,
                    "rawHalves": raw_halves,
                    "transHalves": trans_halves,
                    "rawSource": raw[r] if r is not None else None,
                    "transSource": translated[t] if t is not None else None,
                    "rawFile": os.path.basename(raw[r]) if r is not None else None,
                    "transFile": os.path.basename(translated[t]) if t is not None else None,
                    "score": round(float(score), 3) if score is not None else None,
                    "note": note,
                    "done": os.path.exists(note) and os.path.getsize(note) > 0,
                }
            )
            count += 1
        listed.append({"name": os.path.basename(volume), "path": volume, "number": number, "pages": count})

    manifest = {
        "root": root,
        "rawPages": raw_total,
        "translatedPages": translated_total,
        "withRaw": bool(raw_total),
        "withTranslation": not raw_only,
        "limited": bool(args.pages or only),
        "series": series,
        "args": {"from": args.first, "to": args.last, "pages": args.pages, "only": args.only},
        "volumes": listed,
        "pages": pages,
        "skipped": skipped,
        "cast": os.path.join(out, "cast.json"),
        "known": os.path.join(out, "known.md"),
        "updates": updates_dir,
        "profile": os.path.join(out, "profile.json"),
    }
    progress.record(manifest)

    done = sum(p["done"] for p in pages)
    weak = [p["id"] for p in pages if p["score"] is not None and p["score"] < 0.8]
    kind = "original" if raw_only else "matched"
    if series:
        print("volumes: " + ", ".join(f"{v['name']} ({v['pages']})" for v in listed))
    print(f"{len(pages)} {kind} pages ({done} already have notes), {len(skipped)} unmatched")
    if skipped:
        print("unmatched: " + ", ".join(f"{s['side']}/{s['file']}" for s in skipped))
    if weak:
        print("weak matches, check the pair when reading: " + ", ".join(weak))
    print(f"manifest: {os.path.join(out, 'pages.json')}")


if __name__ == "__main__":
    main()
