"""Check profile.json against cast.json and the notes — shape and truth both.

    python3 .claude/skills/style-read/verify_profile.py <folder> [--raw-only]

finish.py's validator catches shape: the caps, the enum fields, a relation
pointing at a real id. It does not catch truth — whether a quoted line was
ever actually said. Measured on Kinnikuman: one relation carried a line that
was never in the notes at all, spliced together from two different
characters' dialogue on two different pages, and it passed finish.py cleanly
because finish.py never reads the notes.

Every character's `speechEvidence` and every relation's `evidence` are short
Japanese strings a writer is told to copy, not compose. This script is the
part of that promise that is actually enforced: it rereads every one of them
against the volume's own notes and reports any that do not appear there,
character for character. `finish.py` cannot do this — it does not load the
notes. Nothing else in the pipeline checks it either, so this is the only
thing standing between a fabricated line and the app.

Run this — not just finish.py — every time profile.json is hand-written or
subagent-written, before telling anyone the volume is done:

    python3 .claude/skills/style-read/verify_profile.py <folder> [--raw-only]

Exit code is 0 only when the shape check passes AND every citation is
grounded. A non-zero exit means go back and fix what it names — either find
the real line to copy, or drop the claim; never rewrite the citation to match
a paraphrase, that is exactly the mistake this exists to catch.
"""

import json
import os
import sys

import finish
import write_profile as w


def main():
    args = sys.argv[1:]
    raw_only = "--raw-only" in args
    args = [a for a in args if a != "--raw-only"]
    if len(args) != 1:
        sys.exit(__doc__)

    root = os.path.abspath(args[0])
    work = os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")

    manifest = finish.read_json(os.path.join(work, "pages.json"))
    if manifest is None:
        sys.exit("missing pages.json — run prepare.py first")
    profile = finish.read_json(os.path.join(work, "profile.json"))
    if profile is None:
        sys.exit("missing profile.json — nothing to verify")
    cast = (finish.read_json(os.path.join(work, "cast.json"), {}) or {}).get("characters", [])

    ok = True

    finish.problems.clear()
    finish.check_profile(
        json.loads(json.dumps(profile)),
        cast,
        manifest.get("withRaw", True),
        manifest.get("withTranslation", True),
    )
    if finish.problems:
        ok = False
        print(f"SHAPE — {len(finish.problems)} problem(s):")
        for p in finish.problems:
            print(f"  - {p}")
    else:
        print("shape: OK")

    corpus = w.notes_corpus(manifest)
    grounded, unfounded = w.check_grounding(json.loads(json.dumps(profile)), corpus)
    total = grounded + len(unfounded)
    if unfounded:
        ok = False
        print(f"\nCITATIONS — {grounded}/{total} grounded, {len(unfounded)} NOT found in the notes:")
        for ident, where, quote in unfounded:
            print(f"  ✗ {ident} {where}: {quote}")
        print(
            "\nEach line above is either quoted from the wrong page, reworded while "
            "copying, or invented. Open the character's own notes (digest.py --cast "
            "and grep the quote) and either find where it really was said and fix the "
            "citation, or remove the claim if it was never said."
        )
    elif total:
        print(f"citations: {grounded}/{total} grounded, OK")
    else:
        print("citations: none to check (no evidence/speechEvidence on any entry)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
