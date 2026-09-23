"""Record how far /style-read has got, and what it already knows.

    python3 .claude/skills/style-read/progress.py <folder> [--raw-only]

Reads the manifest prepare.py wrote, checks which pages have notes, and writes
<volume>/style_scan/style_read_progress.json into every volume of the run (the
folder itself when it is not a series), so that anyone looking at a volume can
see whether it has been read, how far, and the command that carries on. A page
counts as read once its note exists; the note is the last thing written for a
page, so a run cut off mid-page reads that page again.

It also keeps the cast up to date. A batch never opens cast.json — the file
grows past what one read can hold, and two batches writing it would collide —
so each page's findings go to updates/<page id>.json and are folded in here,
then set aside under updates/applied/. From the notes it works out which pages
each character is on, which pairs have been seen addressing each other and on
how many pages, and which characters and pairs are settled — known well enough
that later pages need not describe them again.

known.md, beside cast.json, is what the next batch reads instead: every
character in one line, with the forms of address already recorded named but not
quoted, so it stays a list the reader can hold in its head.

prepare.py runs this when it finishes, the reading loop after every batch, and
finish.py after publishing.
"""

import json
import os
import re
import shlex
import sys
import time

from PIL import Image

PROGRESS_FILE = "style_read_progress.json"
STATE_FILE = "profile_state.json"

# A character seen on this many pages, with everything below filled in, is
# settled; a pair seen addressing each other on this many pages, with a default
# form of address, is settled.
SETTLED_PAGES = 6
SETTLED_PAIR_PAGES = 3
GENDERS = ("male", "female")

# Where a batch leaves what it found, and the shape known.md keeps it in. The
# limits are what stops the file a batch reads from growing with the cast: a
# mood is a tag ("giận"), not the retelling of a scene.
UPDATES_DIR = "updates"
APPLIED_FILE = "cast_applied.json"
# The cast block a note ends with, which is how a page reports what it found
# without spending a second call on a file of its own.
CAST_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
FACES_PER_CHARACTER = 4
# A face box is text a model estimated while looking at a whole page — it never
# saw the crop it was describing, and nothing round-trips the estimate back
# through its own eyes. Measured against a real series (Kinnikuman, all six
# volumes, every recorded box actually rendered and looked at): "this box is
# the wrong shape or size" and "this box is a flat blob or blank paper, not a
# face" are catchable this cheaply, without a second model call. "This is the
# wrong character" and "this box landed on the speech bubble, not the face
# beside it" are not — those need a look, and pixel statistics genuinely
# cannot tell manga line art apart from manga hand lettering, both being thin
# black strokes on white. That gap is closed in the prompt instead (tighter
# box, a page picked once the character is already certain), not here.
#
# A rejected box does not count against FACES_PER_CHARACTER, so it costs the
# character nothing — a later, more confident batch gets the slot instead.
#
# Area: a real single-face crop ran 0.6%-5.1% of the page on this series — the
# top of that range was a giant character's deliberate close-up, not an error.
# 6.6% was confirmed by hand to be a crowd panel wrongly boxed as one face.
FACE_MIN_AREA = 0.003
FACE_MAX_AREA = 0.055
# Grayscale fraction thresholds, checked on the padded crop. A flat dark blob
# (a shadow, a solid panel, hair filling the frame with no face in it) reads as
# mostly near-black; a blank gutter or margin reads as almost nothing else but
# near-white. A real face crop — line art on paper — is neither: measured
# alongside the bad ones above, the darkest genuine face (heavy black armour)
# still sat under 0.25 near-black, and the sparsest genuine face still had
# over twice the pixels below 0.06.
FACE_MAX_DARK_FRACTION = 0.5
FACE_MAX_BLANK_FRACTION = 0.9
FACE_MIN_INK_FRACTION = 0.06
FACE_PADDING = 0.12
MIN_FACE_SIDE = 40
MOOD_KEY_MAX = 28
MOOD_VALUE_MAX = 90
LOOKS_MAX = 110
ONCE_LOOKS_MAX = 60
# A standing warning about one character — what two readings disagreed about,
# or what someone was mistaken for. It is printed to every later batch whole,
# because this is the line that stops the same person being given a second id.
NOTE_MAX = 200
MOODS_SHOWN = 8
PAIRS_SHOWN = 12
OPEN_PAIRS_SHOWN = 10
# An open character last seen long ago is unlikely to walk back on; it is kept
# in the list so it is not given a second id, but not described in full.
RECENT_PAGES = 80
# Kept by the scripts from the notes; an update file may not set them.
SCRIPT_FIELDS = ("id", "pages", "pairPages", "settled", "settledPairs", "missing")

ADDRESS_LINE = re.compile(r"ADDRESS\s+(.+?)\s*(?:→|->)\s*(.+?)\s*(?:\[|:)")


def progress_path(volume):
    return os.path.join(volume, "style_scan", PROGRESS_FILE)


def note_written(page):
    return os.path.exists(page["note"]) and os.path.getsize(page["note"]) > 0


def command_for(manifest):
    parts = ["/style-read", manifest["root"]]
    args = manifest.get("args") or {}
    if args.get("from"):
        parts += ["--from", args["from"]]
    if args.get("to"):
        parts += ["--to", args["to"]]
    if not manifest.get("withTranslation", True):
        parts.append("--raw-only")
    if args.get("pages"):
        parts += ["--pages", str(args["pages"])]
    if args.get("only"):
        parts += ["--only", args["only"]]
    return " ".join(shlex.quote(p) for p in parts)


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def volumes_of(manifest):
    return manifest.get("volumes") or [{"path": manifest["root"], "name": os.path.basename(manifest["root"])}]


def complete_volumes(manifest):
    """Volumes every page of which has a note."""
    pages = manifest["pages"]
    return [
        v
        for v in volumes_of(manifest)
        if all(note_written(p) for p in pages if p.get("volume", manifest["root"]) == v["path"])
    ]


def published_volumes(work):
    return set(read_json(os.path.join(work, STATE_FILE), {}).get("published", []))


def mark_published(work, paths):
    state = read_json(os.path.join(work, STATE_FILE), {})
    state["published"] = sorted(set(state.get("published", [])) | set(paths))
    write_json(os.path.join(work, STATE_FILE), state)


def names_to_ids(characters):
    lookup = {}
    for c in characters:
        ident = c.get("id")
        for name in [ident, c.get("name"), c.get("nameJa"), *(c.get("aliases") or [])]:
            if name:
                lookup.setdefault(str(name).strip().lower(), ident)
    return lookup


def who(text, lookup):
    """`kinnikuman (Kinnikuman)`, `Kinnikuman` or `キン肉マン` → `kinnikuman`."""
    text = re.sub(r"\s*[\(（].*?[\)）]\s*", " ", text).strip().lower()
    return lookup.get(text)


def evidence(pages, characters):
    """Pages each character is on, and pages each pair addressed each other on,
    read from the notes."""
    lookup = names_to_ids(characters)
    on_pages, pair_pages = {}, {}
    for page in pages:
        if not note_written(page):
            continue
        with open(page["note"], encoding="utf-8") as f:
            for line in f:
                if line.startswith("Nhân vật:"):
                    for name in re.split(r"[,、;]", line.split(":", 1)[1]):
                        ident = who(name, lookup)
                        if ident:
                            on_pages.setdefault(ident, set()).add(page["id"])
                match = ADDRESS_LINE.search(line)
                if match:
                    a, b = who(match.group(1), lookup), who(match.group(2), lookup)
                    if a and b:
                        for ident in (a, b):
                            on_pages.setdefault(ident, set()).add(page["id"])
                        pair_pages.setdefault((a, b), set()).add(page["id"])
    return on_pages, pair_pages


def short(text, limit):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def usable_face(page, face):
    """Reject a face box before it ever reaches cast.json, not after.

    Checks, in order: the box is well-formed; its area is in the range a real
    single face occupied on this series (not a sliver, not a crowd panel);
    then the rendered crop is not a flat dark blob or blank paper. What it
    cannot catch — the right shape and tone, wrong character; or a box that
    landed on the speech bubble beside a face rather than the face itself —
    needs an actual look, which is what the manga-page-reader brief now warns
    about explicitly instead.

    True only once a candidate has cleared every check; anything False is
    simply not folded in, and does not count against FACES_PER_CHARACTER, so
    a later, more confident batch is still free to offer a real one.
    """
    box = face.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    if not (w > 0 and h > 0 and 0 <= x and 0 <= y and x + w <= 1.05 and y + h <= 1.05):
        return False
    area = w * h
    if not (FACE_MIN_AREA <= area <= FACE_MAX_AREA):
        return False

    source = page.get("rawSource") if face.get("side") == "raw" else page.get("transSource")
    source = source or page.get("rawSource") or page.get("transSource")
    if not source:
        return True  # nothing to render against; size alone already passed

    try:
        image = Image.open(source).convert("L")
    except OSError:
        return True  # a missing/unreadable file is not this box's fault

    half = str(face.get("half", "")).lower()
    if half in ("right", "left"):
        x, w = x / 2, w / 2
        if half == "right":
            x += 0.5
    pad_w, pad_h = w * FACE_PADDING, h * FACE_PADDING
    left = max(0, int((x - pad_w) * image.width))
    top = max(0, int((y - pad_h) * image.height))
    right = min(image.width, int((x + w + pad_w) * image.width))
    bottom = min(image.height, int((y + h + pad_h) * image.height))
    if right - left < MIN_FACE_SIDE or bottom - top < MIN_FACE_SIDE:
        return False

    crop = image.crop((left, top, right, bottom))
    histogram = crop.histogram()
    total = sum(histogram) or 1
    dark = sum(histogram[:40]) / total  # near-black: a flat shadow or panel
    bright = sum(histogram[220:]) / total  # near-white: blank paper or gutter
    ink = 1 - dark - bright  # everything in between — where line art lives
    if dark > FACE_MAX_DARK_FRACTION:
        return False
    if bright > FACE_MAX_BLANK_FRACTION and ink < FACE_MIN_INK_FRACTION:
        return False
    return True


def mood_key(key):
    """A mood is a tag. Readers write whole clauses ("cãi lại, biện minh"), and
    every one of them is repeated to every later batch, so keep the first
    clause and let the note carry the rest."""
    first = re.split(r"[,;、，]", " ".join(str(key or "").split()))[0].strip()
    return short(first, MOOD_KEY_MAX)


def merge_character(target, source, pages_by_id=None):
    """Fold one update entry into a cast entry: fill what is empty, add what is
    new, and never overwrite what is already recorded — a settled character is
    not described again, and a batch working from a stale known.md must not be
    able to undo a later one."""
    for key, value in source.items():
        if key in SCRIPT_FIELDS:
            continue
        if key == "faces":
            faces = target.setdefault("faces", [])
            seen = {(f.get("page"), f.get("side")) for f in faces}
            for face in value or []:
                if len(faces) >= FACES_PER_CHARACTER:
                    break
                mark = (face.get("page"), face.get("side"))
                if mark in seen:
                    continue
                page = (pages_by_id or {}).get(face.get("page"))
                # No page to check the box against (a stale id, or this is
                # being called from somewhere that never had pages to give) —
                # do not silently drop it over a lookup miss, only over a box
                # that fails its own check.
                if page is not None and not usable_face(page, face):
                    continue
                faces.append(face)
                seen.add(mark)
        elif key == "addresses":
            addresses = target.setdefault("addresses", {})
            for other, entry in (value or {}).items():
                kept = addresses.setdefault(other, {})
                default = str((entry or {}).get("default", "")).strip()
                if default and not str(kept.get("default", "")).strip():
                    kept["default"] = short(default, MOOD_VALUE_MAX)
                for mood, how in ((entry or {}).get("moods") or {}).items():
                    tag = mood_key(mood)
                    if tag and tag not in kept.setdefault("moods", {}):
                        kept["moods"][tag] = short(how, MOOD_VALUE_MAX)
        elif key == "aliases":
            aliases = target.setdefault("aliases", [])
            for alias in value or []:
                if alias and alias not in aliases:
                    aliases.append(alias)
        elif key == "looks":
            if not str(target.get("looks", "")).strip():
                target["looks"] = short(value, LOOKS_MAX)
        elif str(value or "").strip() and not str(target.get(key, "")).strip():
            target[key] = value
    return target


def cast_block(note_path):
    """The JSON a note ends with: what that page found out about the cast.

    It rides in the note so that reading a page costs one write, not two — the
    calls a batch makes are most of what it costs."""
    try:
        with open(note_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    matches = CAST_BLOCK.findall(text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return False  # there is one, and it is broken: worth saying so


def apply_updates(cast_path, pages=()):
    """Fold what the batches found into cast.json, once each.

    Two sources: the cast block at the end of each note, and any standalone
    updates/*.json a reader wrote. Both are folded in once — notes by page id
    in cast_applied.json, files by being moved to updates/applied/ — so that a
    character retired by hand is not conjured back on the next run."""
    work = os.path.dirname(cast_path)
    state = read_json(os.path.join(work, APPLIED_FILE), {}) or {}
    done = set(state.get("pages") or [])

    fresh_notes, broken = [], []
    for page in pages:
        if page["id"] in done or not note_written(page):
            continue
        block = cast_block(page["note"])
        if block is False:
            broken.append(page["id"])
            continue
        done.add(page["id"])  # a note with no block has nothing to fold in
        if block:
            fresh_notes.append((page["id"], block))

    updates = os.path.join(work, UPDATES_DIR)
    files = (
        sorted(f for f in os.listdir(updates) if f.endswith(".json"))
        if os.path.isdir(updates)
        else []
    )
    if not fresh_notes and not files:
        if broken:
            print("  broken cast block in: " + ", ".join(broken), file=sys.stderr)
        return 0, []

    cast = read_json(cast_path, None) or {"characters": []}
    characters = cast.setdefault("characters", [])
    by_id = {c.get("id"): c for c in characters}
    pages_by_id = {p["id"]: p for p in pages}
    added = []

    def fold(update):
        for entry in (update or {}).get("characters") or []:
            ident = str(entry.get("id", "")).strip()
            if not ident:
                continue
            if ident not in by_id:
                by_id[ident] = {"id": ident}
                characters.append(by_id[ident])
                added.append(ident)
            merge_character(by_id[ident], entry, pages_by_id)

    for _, block in fresh_notes:
        fold(block)

    applied = os.path.join(updates, "applied")
    for name in files:
        path = os.path.join(updates, name)
        update = read_json(path, None)
        if update is None:
            print(f"  {name} is not valid JSON — left in {UPDATES_DIR}/ for a look", file=sys.stderr)
            continue
        fold(update)
        os.makedirs(applied, exist_ok=True)
        os.replace(path, os.path.join(applied, name))

    write_json(cast_path, cast)
    write_json(os.path.join(work, APPLIED_FILE), {"pages": sorted(done)})
    if broken:
        print("  broken cast block in: " + ", ".join(broken), file=sys.stderr)
    return len(fresh_notes) + len(files), added


def missing_for_settling(c, with_raw):
    missing = []
    if not str(c.get("name", "")).strip():
        missing.append("name")
    if with_raw and not str(c.get("nameJa", "")).strip():
        missing.append("nameJa")
    if c.get("gender") not in GENDERS:
        missing.append("gender")
    if not c.get("ageGroup"):
        missing.append("ageGroup")
    if not str(c.get("looks", "")).strip():
        missing.append("looks")
    if not c.get("faces"):
        missing.append("faces")
    if len(c.get("pages", [])) < SETTLED_PAGES:
        missing.append(f"{SETTLED_PAGES} pages")
    return missing


def settle(manifest):
    """Fold in what the last batch found, bring cast.json's page counts and
    settled flags up to date, and write known.md for the next batch."""
    cast_path = manifest["cast"]
    files, added = apply_updates(cast_path, manifest["pages"])
    if files:
        print(f"folded in {files} page(s) of cast findings" + (f", new: {', '.join(added)}" if added else ""))
    cast = read_json(cast_path, None)
    if not cast or not cast.get("characters"):
        return
    characters = cast["characters"]
    on_pages, pair_pages = evidence(manifest["pages"], characters)
    order = {p["id"]: i for i, p in enumerate(manifest["pages"])}
    with_raw = manifest.get("withRaw", True)

    for c in characters:
        ident = c.get("id")
        pages = set(c.get("pages", [])) | on_pages.get(ident, set())
        c["pages"] = sorted(pages, key=lambda p: (order.get(p, len(order)), p))
        addresses = c.get("addresses") or {}
        seen = {b: len(ps) for (a, b), ps in pair_pages.items() if a == ident}
        c["pairPages"] = seen
        c["settledPairs"] = sorted(
            b
            for b, n in seen.items()
            if n >= SETTLED_PAIR_PAGES and str((addresses.get(b) or {}).get("default", "")).strip()
        )
        missing = missing_for_settling(c, with_raw)
        c["settled"] = not missing and not c.get("hold")
        c["missing"] = missing
    write_json(cast_path, cast)

    by_id = {c["id"]: c for c in characters}
    lines = [
        "# Known characters",
        "",
        "This is the cast. Never open cast.json — it is far larger than one read",
        "holds, so what you would see is a fraction of it. Write what you find to",
        "your update file instead; a script folds it in.",
        "",
        "Settled characters are identified for good: recognise them, name them by id,",
        "do not describe them again. Under each one are the pairs already settled —",
        "the usual form of address, then in brackets the moods already recorded. Write",
        "an ADDRESS line for a settled pair only when the form is neither the default",
        "nor one of those moods.",
        "",
        "Faces are a separate matter from settling, and stay open past it: a line",
        "marked (needs a face) still has fewer than four usable ones on record. If",
        "you land on a page where that character is clearly, confidently visible, add",
        "one. Do not go looking for one on their account — this is only for when one is",
        "already in front of you — and never guess at a half-seen or ambiguous face",
        "just to fill the slot: a box is checked and silently dropped if it does not",
        "hold a real face, so a wrong or unclear one wastes the page, not the slot.",
        "",
        "## Settled",
        "",
    ]
    for c in characters:
        if not c["settled"]:
            continue
        lines.append(
            f"- **{c['id']}** — {c.get('name', '')}"
            + (f" ({c['nameJa']})" if c.get("nameJa") else "")
            + f", {c.get('gender', '')}, {c.get('ageGroup', '')}"
            + (f"; aka {', '.join(c.get('aliases') or [])}" if c.get("aliases") else "")
            + (f"; {short(c.get('looks', ''), LOOKS_MAX)}" if c.get("looks") else "")
            + (" (needs a face)" if len(c.get("faces") or []) < FACES_PER_CHARACTER else "")
            + (f"\n    ! {short(c['note'], NOTE_MAX)}" if c.get("note") else "")
        )
        for b in c["settledPairs"][:PAIRS_SHOWN]:
            entry = c.get("addresses", {}).get(b, {})
            tags = [mood_key(k) for k in (entry.get("moods") or {})]
            shown = ", ".join(tags[:MOODS_SHOWN]) + (
                f", +{len(tags) - MOODS_SHOWN}" if len(tags) > MOODS_SHOWN else ""
            )
            lines.append(
                f"    → {b}: {short(entry.get('default', ''), MOOD_VALUE_MAX)}"
                + (f" [{shown}]" if tags else "")
            )
        if len(c["settledPairs"]) > PAIRS_SHOWN:
            lines.append(f"    → …{len(c['settledPairs']) - PAIRS_SHOWN} more settled pairs")
    lines += [
        "",
        "## Still open — complete these when they appear",
        "",
        "These are what the effort goes on. `looks` is there so you can tell them",
        "apart; a character held open is one two readings disagree about — never",
        "merge it into another id yourself, say so in your report instead.",
        "",
    ]
    # Most-seen first: those are the ones likely to walk back onto a page.
    open_cast = sorted(
        (c for c in characters if not c["settled"]),
        key=lambda c: (-len(c.get("pages", [])), c["id"]),
    )
    read_so_far = [p["id"] for p in manifest["pages"] if note_written(p)]
    recent = set(read_so_far[-RECENT_PAGES:])

    def worth_describing(c):
        if c.get("hold") or len(c.get("pages", [])) >= 3:
            return True
        return bool(recent.intersection(c.get("pages", [])))

    once = []
    for c in open_cast:
        if not worth_describing(c):
            once.append(c)
            continue
        open_pairs = [b for b in c.get("pairPages", {}) if b not in c["settledPairs"] and b in by_id]
        lines.append(
            f"- **{c['id']}** — {c.get('name', '')}"
            + (f" ({c['nameJa']})" if c.get("nameJa") else "")
            + f": missing {', '.join(c['missing']) or 'nothing'}"
            + ("; HELD OPEN" if c.get("hold") else "")
            + (f"; {short(c.get('looks', ''), LOOKS_MAX)}" if c.get("looks") else "")
            + (
                f"; open pairs → {', '.join(open_pairs[:OPEN_PAIRS_SHOWN])}"
                if open_pairs
                else ""
            )
            + (f"\n    ! {short(c['note'], NOTE_MAX)}" if c.get("note") else "")
        )
    if once:
        lines += [
            "",
            "Seen once, or not for a long time — check against these before giving",
            "anyone a new id, but do not go looking for them:",
            "",
        ]
        for c in once:
            lines.append(
                f"- **{c['id']}** — {c.get('name', '')}"
                + (f"; {short(c.get('looks', ''), ONCE_LOOKS_MAX)}" if c.get("looks") else "")
            )
    with open(os.path.join(os.path.dirname(cast_path), "known.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def record(manifest):
    """Write every volume's progress file; return (read, total) for the run."""
    pages = manifest["pages"]
    for page in pages:
        page["done"] = note_written(page)
    volumes = volumes_of(manifest)
    work = os.path.dirname(manifest["cast"])
    published = published_volumes(work)
    settle(manifest)
    now = int(time.time())

    for volume in volumes:
        mine = [p for p in pages if p.get("volume", manifest["root"]) == volume["path"]]
        read = [p for p in mine if p["done"]]
        waiting = [p for p in mine if not p["done"]]
        if volume["path"] in published and not waiting:
            status = "published"
        elif not read:
            status = "not_started"
        elif waiting:
            status = "reading"
        else:
            # Every page has a note; the profile has not taken it in yet.
            status = "read"
        write_json(
            progress_path(volume["path"]),
            {
                "status": status,
                "mode": "with-translation" if manifest.get("withTranslation", True) else "raw-only",
                "pagesRead": len(read),
                "pagesTotal": len(mine),
                "lastPageRead": read[-1]["id"] if read else None,
                "nextPage": waiting[0]["id"] if waiting else None,
                "notRead": [p["id"] for p in waiting],
                "run": {
                    "root": manifest["root"],
                    "volumes": [v.get("name") for v in volumes],
                    "pagesRead": sum(p["done"] for p in pages),
                    "pagesTotal": len(pages),
                    "workDir": work,
                },
                "resume": command_for(manifest),
                "updatedAt": now,
            },
        )

    with open(os.path.join(work, "pages.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return sum(p["done"] for p in pages), len(pages)


def main():
    args = sys.argv[1:]
    raw_only = "--raw-only" in args
    args = [a for a in args if a != "--raw-only"]
    if len(args) != 1:
        sys.exit(__doc__)
    root = os.path.abspath(args[0])
    work = os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")
    try:
        with open(os.path.join(work, "pages.json"), encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        sys.exit("missing pages.json — run prepare.py first")

    read, total = record(manifest)
    print(f"{read}/{total} pages read")
    for volume in manifest.get("volumes") or []:
        with open(progress_path(volume["path"]), encoding="utf-8") as f:
            state = json.load(f)
        print(f"  {volume['name']}: {state['pagesRead']}/{state['pagesTotal']} ({state['status']})")
    waiting = [p["id"] for p in manifest["pages"] if not p["done"]]
    if waiting:
        print("next: " + ", ".join(waiting[:6]))
    unpublished = [v["name"] for v in complete_volumes(manifest) if v["path"] not in published_volumes(work)]
    if unpublished:
        print("read but not yet in the profile: " + ", ".join(unpublished))
    cast = read_json(manifest["cast"], {}) or {}
    settled = [c["id"] for c in cast.get("characters", []) if c.get("settled")]
    if cast.get("characters"):
        print(f"settled characters: {len(settled)}/{len(cast['characters'])}")


if __name__ == "__main__":
    main()
