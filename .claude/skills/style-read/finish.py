"""Publish what /style-read learned where the app will find it.

    python3 .claude/skills/style-read/finish.py <folder> [--raw-only]

Reads <folder>/style_scan/claude/profile.json and cast.json (claude_raw/ with
--raw-only, when the volume was read from the Japanese alone), checks them,
and writes:

- <folder>/style_scan/style_profile_v1.json — the translation profile,
  including the cast, in the shape of StyleScanResult
  (koharu-ml/src/bilingual/corpus.rs). The Style Scanner page loads it.
- style_profiles/<name>.json in this project — the same file, in the library
  the Style Scanner page lists and the translator loads from.
- <folder>/character_scan/manga_relationship_v1.json and face crops — the cast
  as a character tree, in the shape the Character Scanner page loads
  (koharu-ml/src/character_library/scanner.rs, ScanResult).

For a series read with prepare.py --from/--to, <folder> is the series folder.
The profile is the series' character dictionary: it is written to
<folder>/character_dictionary.json and, with the character tree, into every
volume read to the end so far (the app opens a volume at a time). Volumes still
being read are left alone, so this runs after each volume, not only at the end.

Results the app's own scans wrote are kept beside the new ones as *.tool.json
rather than overwritten.
"""

import json
import os
import sys
import time
import unicodedata

from PIL import Image

import progress

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

LIMITS = {
    "approach": 10,
    "voice": 12,
    "address": 14,
    "soundEffects": 8,
    "glossary": 60,
    "characters": 30,
}
GENDERS = {"", "male", "female"}
AGE_GROUPS = {"", "child", "teen", "young_adult", "adult", "middle_age", "elder"}
# Without a translation there is no translator whose habits these describe.
TRANSLATION_ONLY = ("approach", "voice", "soundEffects")
FACES_PER_CHARACTER = 3
FACE_PADDING = 0.12
MIN_FACE_SIDE = 40

problems = []


def problem(message):
    problems.append(message)


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as err:
        sys.exit(f"{path}: not valid JSON: {err}")


def slug(name):
    """ASCII id from a name, as koharu-ml's style::slug does: `Tổng tư lệnh` → `tong-tu-lenh`."""
    folded = unicodedata.normalize("NFD", name.replace("đ", "d").replace("Đ", "d"))
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    out = []
    for c in folded:
        if c.isascii() and c.isalnum():
            out.append(c)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-") or "character"


def strings(profile, key):
    items = profile.get(key, [])
    if not isinstance(items, list) or not all(isinstance(i, str) and i.strip() for i in items):
        problem(f"'{key}' must be a list of non-empty strings")
        return []
    if len(items) > LIMITS[key]:
        problem(f"'{key}' has {len(items)} entries; keep it to {LIMITS[key]} — it rides in every translation prompt")
    return [i.strip() for i in items]


def check_characters(raw_characters, cast):
    if not isinstance(raw_characters, list):
        problem("'characters' must be a list")
        return []
    if len(raw_characters) > LIMITS["characters"]:
        problem(f"'characters' has {len(raw_characters)} entries; keep the recurring cast to {LIMITS['characters']}")

    cast_by_id = {c.get("id"): c for c in cast}
    characters, seen = [], set()
    for index, c in enumerate(raw_characters):
        where = f"characters[{index}]"
        if not isinstance(c, dict) or not str(c.get("name", "")).strip():
            problem(f"{where}: needs a name")
            continue
        ident = str(c.get("id") or slug(c["name"])).strip()
        if ident != slug(ident) and ident != ident.lower():
            problem(f"{where}: id '{ident}' must be lower-case ASCII with dashes")
        if ident in seen:
            problem(f"{where}: id '{ident}' is used twice")
        seen.add(ident)

        gender = str(c.get("gender") or "").strip()
        age = str(c.get("ageGroup") or "").strip()
        if gender not in GENDERS:
            problem(f"{where}: gender must be one of {sorted(GENDERS)}, not '{gender}'")
        if age not in AGE_GROUPS:
            problem(f"{where}: ageGroup must be one of {sorted(AGE_GROUPS)}, not '{age}'")

        from_cast = cast_by_id.get(ident, {})
        appearances = c.get("appearances")
        if not isinstance(appearances, int) or appearances <= 0:
            appearances = len(from_cast.get("pages", []))

        relations = []
        for r in c.get("relations", []) or []:
            if not isinstance(r, dict) or not str(r.get("to", "")).strip():
                problem(f"{where}: every relation needs 'to'")
                continue
            if not str(r.get("address", "")).strip():
                problem(f"{where} → {r.get('to')}: 'address' is empty — say how they address them")
            relations.append(
                {
                    "to": str(r["to"]).strip(),
                    "relation": str(r.get("relation", "")).strip(),
                    "address": str(r.get("address", "")).strip(),
                }
            )

        characters.append(
            {
                "id": ident,
                "name": c["name"].strip(),
                "nameJa": str(c.get("nameJa") or "").strip(),
                "aliases": [str(a).strip() for a in c.get("aliases", []) or [] if str(a).strip()],
                "gender": gender,
                "ageGroup": age,
                "role": str(c.get("role") or "").strip(),
                "personality": str(c.get("personality") or "").strip(),
                "speech": str(c.get("speech") or "").strip(),
                "selfTerms": [str(t).strip() for t in c.get("selfTerms", []) or [] if str(t).strip()],
                "appearances": appearances,
                "relations": relations,
            }
        )

    ids = {c["id"] for c in characters}
    for c in characters:
        for r in c["relations"]:
            if r["to"] not in ids:
                problem(f"characters '{c['id']}' → '{r['to']}': no character has that id")
    return characters


def check_profile(profile, cast, with_raw, with_translation):
    if not isinstance(profile, dict):
        sys.exit("profile.json: expected a JSON object")
    unknown = set(profile) - set(LIMITS)
    if unknown:
        problem(f"unexpected keys {sorted(unknown)}; allowed: {sorted(LIMITS)}")

    glossary = profile.get("glossary", [])
    if not isinstance(glossary, list) or not all(
        isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) for x in e) and e[1].strip()
        for e in glossary
    ):
        problem("'glossary' must be a list of [japanese, vietnamese] pairs with the Vietnamese filled in")
        glossary = []
    if len(glossary) > LIMITS["glossary"]:
        problem(f"'glossary' has {len(glossary)} entries; keep it to {LIMITS['glossary']}")

    if not with_translation:
        for key in TRANSLATION_ONLY:
            if profile.get(key):
                problem(f"'{key}' must be empty: there is no translation to learn it from")

    return {
        "approach": strings(profile, "approach"),
        "voice": strings(profile, "voice"),
        "address": strings(profile, "address"),
        "soundEffects": strings(profile, "soundEffects"),
        # With no original read, there is no Japanese to pair a name with.
        "glossary": [[a.strip() if with_raw else "", b.strip()] for a, b in glossary],
        "characters": check_characters(profile.get("characters", []), cast),
    }


def crop_faces(faces_dir, manifest, cast, characters):
    """Cut each character's face out of the page it was noted on.

    Boxes are fractions of the page, so they apply to the full-size original as
    well as to the reading copy they were noted on.
    """
    pages = {p["id"]: p for p in manifest["pages"]}
    cast_by_id = {c.get("id"): c for c in cast}
    crops = {}
    for character in characters:
        entries = cast_by_id.get(character["id"], {}).get("faces", []) or []
        saved = []
        for face in entries[:FACES_PER_CHARACTER]:
            page = pages.get(face.get("page"))
            box = face.get("box")
            if not page or not isinstance(box, list) or len(box) != 4:
                continue
            wants_raw = face.get("side") == "raw" or not page.get("transSource")
            source = page["rawSource"] if wants_raw and page.get("rawSource") else page["transSource"]
            try:
                image = Image.open(source).convert("RGB")
            except OSError:
                continue
            x, y, w, h = (float(v) for v in box)
            pad_w, pad_h = w * FACE_PADDING, h * FACE_PADDING
            left = max(0, int((x - pad_w) * image.width))
            top = max(0, int((y - pad_h) * image.height))
            right = min(image.width, int((x + w + pad_w) * image.width))
            bottom = min(image.height, int((y + h + pad_h) * image.height))
            if right - left < MIN_FACE_SIDE or bottom - top < MIN_FACE_SIDE:
                continue
            directory = os.path.join(faces_dir, character["id"])
            os.makedirs(directory, exist_ok=True)
            name = f"{len(saved)}.png"
            image.crop((left, top, right, bottom)).save(os.path.join(directory, name))
            saved.append(f"faces/{character['id']}/{name}")
        crops[character["id"]] = saved
    return crops


def character_tree(cast, characters, crops):
    """The cast in the Character Scanner's shape."""
    pages_of = {c.get("id"): set(c.get("pages", [])) for c in cast}
    scanned, tree = [], []
    for c in characters:
        traits = [t for t in (c["role"], c["personality"]) if t]
        scanned.append(
            {
                "id": c["id"],
                "name": c["name"],
                "gender": c["gender"] or None,
                "ageGroup": c["ageGroup"] or None,
                "faces": crops.get(c["id"], []),
                "traits": traits,
            }
        )
        related = [
            {
                "characterId": r["to"],
                "coOccurrence": len(pages_of.get(c["id"], set()) & pages_of.get(r["to"], set())),
                "label": r["relation"] or None,
                "description": r["address"] or None,
            }
            for r in c["relations"]
        ]
        tree.append({"characterId": c["id"], "related": related})
    return {"characters": scanned, "relationshipTree": tree, "isVerifiedByHuman": False}


def keep_previous(path, marker):
    """Set aside a result the app's own scan wrote, before replacing it."""
    if os.path.exists(path) and not os.path.exists(marker):
        backup = path.replace(".json", ".tool.json")
        os.replace(path, backup)
        print(f"kept the app's own result as {backup}")


def main():
    args = sys.argv[1:]
    raw_only = "--raw-only" in args
    args = [a for a in args if a != "--raw-only"]
    if len(args) != 1:
        sys.exit(__doc__)
    root = os.path.abspath(args[0])
    work = os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")

    manifest = read_json(os.path.join(work, "pages.json"))
    if manifest is None:
        sys.exit("missing pages.json — run prepare.py first")
    profile = read_json(os.path.join(work, "profile.json"))
    if profile is None:
        sys.exit("missing profile.json — write the profile first")

    # Page counts in the cast come from the notes; bring them up to date first.
    progress.settle(manifest)
    cast = (read_json(os.path.join(work, "cast.json"), {}) or {}).get("characters", [])
    series = manifest.get("series")
    if series:
        volumes = progress.complete_volumes(manifest)
        if not volumes:
            problem("no volume has been read to the end yet")
    else:
        volumes = [{"path": root, "number": None}]
        missing = [p["id"] for p in manifest["pages"] if not progress.note_written(p)]
        if missing:
            problem(f"{len(missing)} pages have no notes yet: {', '.join(missing[:20])}")
    covered = {v["path"] for v in volumes}
    pages = [p for p in manifest["pages"] if p.get("volume", root) in covered]

    with_translation = manifest.get("withTranslation", True)
    checked = check_profile(profile, cast, manifest.get("withRaw", True), with_translation)
    if problems:
        print("Fix these and run finish.py again:")
        for p in problems:
            print(f"- {p}")
        sys.exit(1)

    if series:
        # One entry for the whole series, updated as volumes are added, rather
        # than one per stage the dictionary went through.
        name = slug(os.path.basename(root))
    else:
        parent = os.path.basename(os.path.dirname(root))
        name = slug(f"{parent}-{os.path.basename(root)}" if parent else os.path.basename(root))
    if not with_translation:
        name += "-raw"
    if manifest.get("limited"):
        name += "-sample"

    result = {
        "profile": checked,
        "rawPages": sum(1 for p in pages if p.get("rawSource")),
        "translatedPages": sum(1 for p in pages if p.get("transSource")),
        "pairedPages": len(pages),
        "pairCount": 0,
        "isVerifiedByHuman": False,
        "source": "claude",
        "withRaw": manifest.get("withRaw", True),
        "name": name,
        "createdAt": int(time.time()),
    }

    library = os.environ.get("KOHARU_STYLE_PROFILES_DIR") or os.path.join(REPO, "style_profiles")
    write_json(os.path.join(library, f"{name}.json"), result)
    if series:
        dictionary = os.path.join(root, "character_dictionary.json")
        write_json(dictionary, result)
        print(f"dictionary:       {dictionary}")

    counts = ", ".join(f"{k} {len(v)}" for k, v in checked.items())
    print(f"read {len(pages)} pages; {counts}")
    print(f"library:          {os.path.join(library, name + '.json')}")
    for volume in volumes:
        publish(volume["path"], result, manifest, cast, checked["characters"], with_translation)
    progress.mark_published(work, covered)
    progress.record(manifest)
    waiting = [v["name"] for v in manifest.get("volumes") or [] if v["path"] not in covered]
    if waiting:
        print("still being read: " + ", ".join(waiting))


def publish(folder, result, manifest, cast, characters, with_translation):
    """Put the profile and the character tree where the app looks when this
    folder is opened."""
    style_dir = os.path.join(folder, "style_scan")
    target = os.path.join(style_dir, "style_profile_v1.json")
    previous = read_json(target, {}) or {}
    if os.path.exists(target) and previous.get("source") != "claude":
        backup = os.path.join(style_dir, "style_profile_v1.tool.json")
        os.replace(target, backup)
        print(f"kept the app's own style result as {backup}")
    elif previous.get("translatedPages") and not with_translation:
        # A profile learned from the translation says more than this one; keep it.
        backup = os.path.join(style_dir, "style_profile_v1.with_translation.json")
        os.replace(target, backup)
        print(f"kept the profile learned from the translation as {backup}")
    write_json(target, result)
    print(f"profile:          {target}")

    if not characters:
        return
    scan_dir = os.path.join(folder, "character_scan")
    marker = os.path.join(scan_dir, ".written-by-style-read")
    tree_path = os.path.join(scan_dir, "manga_relationship_v1.json")
    keep_previous(tree_path, marker)
    crops = crop_faces(os.path.join(scan_dir, "faces"), manifest, cast, characters)
    write_json(tree_path, character_tree(cast, characters, crops))
    with open(marker, "w") as f:
        f.write("character_scan/manga_relationship_v1.json was written by /style-read\n")
    faces = sum(len(v) for v in crops.values())
    print(f"character tree:   {tree_path} ({faces} face crops)")


if __name__ == "__main__":
    main()
