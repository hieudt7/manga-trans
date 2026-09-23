"""Drop face crops that a look, not a script, found to be wrong.

    python3 .claude/skills/style-read/review_faces.py <folder> --bad id/index,id/index,... [--raw-only]

`usable_face` (progress.py) catches size and blankness — a box sized for a
crowd panel, a flat black shadow, a blank margin. Measured against this same
series after that check was already in place: it does not catch a box that
lands on a speech bubble instead of the face beside it, or the wrong
character's face. Both read as ordinary line art to a pixel histogram; only
looking at the actual crop tells them apart, and nothing in the pipeline
looks. A face-hunt subagent — whose only job was finding faces — produced
these same two failures on its own picks, not just on the original reads'.

So looking is not optional, it is the other half of the fix: after
`finish.py` runs, open a handful of the crops it just wrote
(`<volume>/character_scan/faces/<id>/*.png`) and check each one is really
that character's face — not text, not the wrong person, not a sliver of
something else. Do this yourself for a few, or dispatch one subagent (Read
tool only, no notes, no dialogue — just "is this a face of this character,
yes or no") to go through all of them; either way, feed what you found to
this script rather than editing cast.json by hand — hand edits are how a
previous pass removed the wrong one of two bad boxes and left the other.

--bad takes "id/index" pairs from what a review found, e.g.
"sheik-seijin/0,natsuko/0" for sheik-seijin's 0.png and natsuko/0.png. It
looks each one up in face_origins.json (written by the last finish.py run) to
find the EXACT cast.json entry that produced it — never re-derives this by
re-running the crop logic, which is exactly the step that went wrong doing
it by hand. Removed entries do not count against a character's cap, so a
later face-hunt (or a plain read, on a future volume) still gets to try again.

Rerun finish.py afterward to regenerate the crops without them.
"""

import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)


def main():
    args = sys.argv[1:]
    raw_only = "--raw-only" in args
    args = [a for a in args if a != "--raw-only"]
    bad = []
    if "--bad" in args:
        i = args.index("--bad")
        bad = [x.strip() for x in (args[i + 1] if i + 1 < len(args) else "").split(",") if x.strip()]
        del args[i : i + 2]
    if len(args) != 1 or not bad:
        sys.exit(__doc__)

    root = os.path.abspath(args[0])
    work = os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")

    origins = read_json(os.path.join(work, "face_origins.json"))
    if origins is None:
        sys.exit(
            f"missing {os.path.join(work, 'face_origins.json')} — run finish.py at least once first"
        )
    cast_path = os.path.join(work, "cast.json")
    cast = read_json(cast_path, {}) or {}
    by_id = {c.get("id"): c for c in cast.get("characters") or []}

    removed, missing = [], []
    for key in bad:
        face = origins.get(key)
        if face is None:
            missing.append(key)
            continue
        cid = key.rsplit("/", 1)[0]
        character = by_id.get(cid)
        if not character:
            missing.append(key)
            continue
        before = len(character.get("faces") or [])
        character["faces"] = [
            f
            for f in character.get("faces") or []
            if not (f.get("page") == face.get("page") and f.get("box") == face.get("box"))
        ]
        if len(character["faces"]) < before:
            removed.append(key)
        else:
            missing.append(key)

    if removed:
        write_json(cast_path, cast)
    print(f"removed: {', '.join(removed) or '(none)'}")
    if missing:
        print(
            f"could not find (already gone, or wrong id/index — check face_origins.json): "
            f"{', '.join(missing)}"
        )
    if removed:
        print("\nNow rerun finish.py to regenerate the crops without them.")


if __name__ == "__main__":
    main()
