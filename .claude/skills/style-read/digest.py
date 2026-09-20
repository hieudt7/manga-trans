"""Print one volume's notes, stripped to what the profile is written from.

    python3 .claude/skills/style-read/digest.py <folder> [--raw-only] [--volume NAME]

Step 3 of /style-read writes the profile from a volume's notes. Opening them
one by one costs a tool call and a page of dialogue each; a volume of 94 notes
is most of a context window, and almost all of it is the transcript, which the
profile never quotes.

So: one call, and in raw-only mode the `### Lời thoại` transcript is left out.
There the notes are evidence about the cast — who is on the page, how each pair
addresses the other, names, terms, ways of speaking — and that is what comes
out. With a translation the dialogue *is* the material (`voice` and `approach`
are learned from the Vietnamese), so nothing is dropped.

--volume picks one volume of a series by name or number; without it every
volume in the run is printed. Pass the same --raw-only as the other scripts.

--cast prints the cast instead of the notes, in the shape the profile is
written from: one line a character, appearances counted, every form of address
spelled out. cast.json itself carries page lists and face boxes that only the
scripts use and is several times too long to read whole.
"""

import json
import os
import re
import sys

import progress

# The profile keeps at most 30 characters and one line of address a relation,
# so the cast is spelled out for the ones it can reach and named for the rest.
CAST_IN_FULL = 40
MOODS_SHOWN = 8
DEFAULT_MAX = 110
MOOD_MAX = 80

# The headings a note is built from. In raw-only mode the transcript goes; the
# cast block always does — progress.py has already folded it into the cast, and
# the profile is written from the cast digest, not from JSON in a note.
TRANSCRIPT = "### Lời thoại"
CAST_SECTION = "### Cast"
SECTION = re.compile(r"^#{2,3}\s")


def volume_matches(volume, wanted):
    name = volume.get("name") or ""
    if wanted.lower() in name.lower():
        return True
    number = volume.get("number")
    return bool(wanted.isdigit() and number is not None and int(wanted) == number)


def note_lines(path, keep_transcript):
    """The note, minus the sections this mode does not need."""
    out, skipping = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if SECTION.match(line):
                # Readers qualify the heading — "### Lời thoại — nửa phải
                # (tr.6)" — so match on how it starts, not on the whole line.
                head = line.strip()
                skipping = head.startswith(CAST_SECTION) or (
                    not keep_transcript and head.startswith(TRANSCRIPT)
                )
                if skipping:
                    continue
            if skipping:
                # Keep the lines that carry evidence even inside the transcript:
                # a reader sometimes files an ADDRESS line under it.
                if "ADDRESS" not in line:
                    continue
            if line.strip():
                out.append(line)
    return out


def cast_lines(work, with_raw):
    """The cast as the profile needs it: who they are, how often, and how they
    address each person they speak to."""
    cast = progress.read_json(os.path.join(work, "cast.json"), {}) or {}
    characters = sorted(
        cast.get("characters") or [],
        key=lambda c: (-len(c.get("pages", [])), c.get("id", "")),
    )
    out = [
        f"# cast: {len(characters)} characters, most-seen first",
        "# `appearances` is the page count; write the profile from the ones a",
        "# translator meets most. Forms of address are default first, then moods.",
        f"# The first {CAST_IN_FULL} are spelled out; the rest are named at the end,",
        "# with the notes behind them if one of them belongs in the profile.",
    ]
    for c in characters[:CAST_IN_FULL]:
        pair_pages = c.get("pairPages") or {}
        out.append("")
        out.append(
            f"## {c.get('id', '')} — {c.get('name', '')}"
            + (f" ({c['nameJa']})" if with_raw and c.get("nameJa") else "")
            + f"  [appearances {len(c.get('pages', []))}"
            + (", settled" if c.get("settled") else "")
            + (", HELD OPEN" if c.get("hold") else "")
            + "]"
        )
        if c.get("aliases"):
            out.append(f"aliases: {', '.join(c['aliases'])}")
        out.append(
            f"{c.get('gender', '') or '?'}, {c.get('ageGroup', '') or '?'}; "
            f"{c.get('looks', '')}"
        )
        for other, entry in sorted(
            (c.get("addresses") or {}).items(),
            key=lambda kv: -pair_pages.get(kv[0], 0),
        ):
            if pair_pages.get(other, 0) < 1 and not (entry or {}).get("default"):
                continue
            recorded = list(((entry or {}).get("moods") or {}).items())
            moods = "; ".join(
                f"{progress.short(k, 28)}: {progress.short(v, MOOD_MAX)}"
                for k, v in recorded[:MOODS_SHOWN]
            )
            if len(recorded) > MOODS_SHOWN:
                moods += f"; +{len(recorded) - MOODS_SHOWN} more"
            out.append(
                f"→ {other} ({pair_pages.get(other, 0)}p): "
                + progress.short((entry or {}).get("default", ""), DEFAULT_MAX)
                + (f" | {moods}" if moods else "")
            )
    rest = characters[CAST_IN_FULL:]
    if rest:
        out += ["", f"# seen less often ({len(rest)}) — id, name, appearances:", ""]
        for c in rest:
            out.append(
                f"- {c.get('id', '')} — {c.get('name', '')} ({len(c.get('pages', []))}p)"
            )
    return out


def main():
    args = sys.argv[1:]
    raw_only = "--raw-only" in args
    args = [a for a in args if a != "--raw-only"]
    want_cast = "--cast" in args
    args = [a for a in args if a != "--cast"]
    wanted = ""
    if "--volume" in args:
        i = args.index("--volume")
        wanted = args[i + 1] if i + 1 < len(args) else ""
        del args[i : i + 2]
    if len(args) != 1:
        sys.exit(__doc__)

    root = os.path.abspath(args[0])
    work = os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")
    try:
        with open(os.path.join(work, "pages.json"), encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        sys.exit("missing pages.json — run prepare.py first")

    if want_cast:
        print("\n".join(cast_lines(work, manifest.get("withRaw", True))))
        return

    keep_transcript = manifest.get("withTranslation", True)
    volumes = progress.volumes_of(manifest)
    if wanted:
        volumes = [v for v in volumes if volume_matches(v, wanted)]
        if not volumes:
            sys.exit("No volume matches that. Volumes: " + ", ".join(v.get("name", "?") for v in volumes))

    paths = {v["path"] for v in volumes}
    pages = [
        p
        for p in manifest["pages"]
        if p.get("volume", manifest["root"]) in paths and progress.note_written(p)
    ]
    if not pages:
        sys.exit("No notes yet for that volume.")

    shown = ", ".join(v.get("name", "?") for v in volumes)
    print(f"# {len(pages)} notes — {shown}")
    print(
        "# The transcript is left out; these are the cast and style lines."
        if not keep_transcript
        else "# Every line of every note."
    )
    for page in pages:
        print()
        print("\n".join(note_lines(page["note"], keep_transcript)))


if __name__ == "__main__":
    main()
