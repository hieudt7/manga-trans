"""Record how far /style-read has got, and what it already knows.

    python3 .claude/skills/style-read/progress.py <folder> [--raw-only]

Reads the manifest prepare.py wrote, checks which pages have notes, and writes
<volume>/style_scan/style_read_progress.json into every volume of the run (the
folder itself when it is not a series), so that anyone looking at a volume can
see whether it has been read, how far, and the command that carries on. A page
counts as read once its note exists; the note is the last thing written for a
page, so a run cut off mid-page reads that page again.

It also keeps the cast up to date from the notes: which pages each character
is on, which pairs have been seen addressing each other and on how many pages,
and which characters and pairs are settled — known well enough that later
pages need not describe them again. The settled ones are listed in known.md in
the work folder, which each batch reads before it starts.

prepare.py runs this when it finishes, the reading loop after every batch, and
finish.py after publishing.
"""

import json
import os
import re
import shlex
import sys
import time

PROGRESS_FILE = "style_read_progress.json"
STATE_FILE = "profile_state.json"

# A character seen on this many pages, with everything below filled in, is
# settled; a pair seen addressing each other on this many pages, with a default
# form of address, is settled.
SETTLED_PAGES = 6
SETTLED_PAIR_PAGES = 3
GENDERS = ("male", "female")

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
    """Bring cast.json's page counts and settled flags up to date, and write
    known.md for the next batch."""
    cast_path = manifest["cast"]
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
        "Settled: recognise them and name them by id, but do not describe them again,",
        "and add no face for them. For a settled pair, write no ADDRESS line unless",
        "the form of address differs from the one given (a new mood counts).",
        "",
    ]
    for c in characters:
        if not c["settled"]:
            continue
        pairs = []
        for b in c["settledPairs"]:
            entry = c.get("addresses", {}).get(b, {})
            moods = "; ".join(f"{k}: {v}" for k, v in (entry.get("moods") or {}).items())
            pairs.append(f"{b}: {entry.get('default', '')}" + (f" ({moods})" if moods else ""))
        lines.append(
            f"- **{c['id']}** — {c.get('name', '')} ({c.get('nameJa', '')}), {c.get('gender', '')}, "
            f"{c.get('ageGroup', '')}; looks: {c.get('looks', '')}; aliases: {', '.join(c.get('aliases') or [])}"
        )
        if pairs:
            lines.append("  - settled pairs → " + " | ".join(pairs))
    lines += ["", "# Still open — complete these when they appear", ""]
    for c in characters:
        if c["settled"]:
            continue
        open_pairs = [b for b in c.get("pairPages", {}) if b not in c["settledPairs"] and b in by_id]
        lines.append(
            f"- **{c['id']}** — {c.get('name', '')}: missing {', '.join(c['missing']) or 'nothing'}"
            + ("; held open" if c.get("hold") else "")
            + (f"; open pairs → {', '.join(open_pairs)}" if open_pairs else "")
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
