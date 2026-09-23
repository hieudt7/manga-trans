"""Write a volume into profile.json with Gemini, instead of in the session.

    python3 .claude/skills/style-read/write_profile.py <folder> --volume <name|number> [--raw-only]

Step 3 of /style-read used to be done by the session driving the reading loop:
it read the volume's digest (over 100KB) and the whole profile, and every tool
call afterwards carried both. Measured on volume 6, the three readers cost 187k
tokens and the session cost roughly 1.3M — the profile write, not the reading,
was the bill.

Nothing in step 3 needs the session's context. It is a function: (this
volume's notes, the cast, the profile so far) -> the profile with this volume
folded in. So it runs here, against Gemini, on the free keys in
gemini_keys.txt, and the session only reads the summary this prints.

**Gemini returns a patch, not the file.** Asking for all 90KB back each volume
invites a truncated response and lets it quietly reword volumes it was not
asked about. Instead it returns characters to upsert (matched by id), ids to
retire, and the short cross-cutting lists in full; this script applies them, so
every character the volume did not touch keeps exactly the bytes it had.

The result is checked with finish.py's own validator before it is written, so a
profile that lands here is one finish.py will accept. If the model returns
something that does not pass, its output is kept as profile.rejected.json and
nothing is overwritten.

But the validator checks shape, not truth. Asked to write all six Kinnikuman
volumes from scratch it produced a dictionary that passed every check and was
still wrong: `speech` with no Vietnamese in it at all, two characters' registers
stated backwards, and one form of address simply invented (Robin Mask calling
Terryman by Robin's own name). So the model is also made to CITE — a few
Japanese strings copied out of the notes behind each `speech` and each relation
— and those citations are looked up in the notes here. A fabricated line cannot
be cited. The evidence is stripped again before anything is written, so it
never reaches the published dictionary.

What it prints after writing is a list of what to go and look at: unfounded
citations first, then speech fields with no Vietnamese, then relations with a
single flat form of address.

Options:
  --volume NAME   the volume to fold in; without it, every note in the run
  --raw-only      the same flag the other scripts take
  --model NAME    default gemini-pro-latest, which has no free quota in
                  practice; use gemini-flash-latest. Do not pin a version —
                  gemini-2.5-pro and gemini-2.5-flash both went 404 for new
                  keys while this was being written.
  --dry-run       build the prompt and print its size, call nothing

Two Gemini calls, not one. A first version asked for the relationship tree
(relations, address) and personality (role, personality, speech) in the same
call, from the same notes, in one breath — and personality came out shallower
for it: the model was budgeting attention across "who addresses whom" and
"who this person is" at once, and the tree, being more mechanical, tended to
win. Now the TREE PASS writes relations/address (plus the identity fields and
the cross-cutting lists) and leaves role/personality/speech blank; only once
that tree exists does the PERSONALITY PASS get it — as settled context, not
something to invent alongside — and writes role/personality/speech grounded
in the dialogue, for exactly the characters the tree pass touched this
volume. Same shared standard (below) governs both; only the envelope around
it differs per pass.
"""

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import digest
import finish
import progress

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# The model the translation pipeline itself uses — koharu-llm/src/api.rs, the
# GEMINI provider's model list. Same model, same keys, same quota as the app,
# so what the dictionary is written by is what reads it. Keep the two in step.
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
KEY_FILE = "gemini_keys.txt"
TIMEOUT = 600
ATTEMPTS_PER_KEY = 2
# A patch is normally a few characters plus the short lists. But the first run
# on a finished series has no profile to patch, so the answer IS the whole
# dictionary — around 30k tokens for six volumes. Ask for the model's ceiling;
# an unused allowance costs nothing, a truncated dictionary costs the run.
MAX_OUTPUT_TOKENS = 65536

EMPTY = {
    "characters": [],
    "approach": [],
    "voice": [],
    "address": [],
    "soundEffects": [],
    "glossary": [],
}


def repo_root():
    """The project root, where gemini_keys.txt lives — three levels up."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def load_keys():
    path = os.path.join(repo_root(), KEY_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            keys = [line.strip() for line in f]
    except FileNotFoundError:
        sys.exit(f"missing {path} — it holds one Gemini API key a line")
    keys = [k for k in keys if k and not k.startswith("#")]
    if not keys:
        sys.exit(f"{path} has no keys in it")
    random.shuffle(keys)
    return keys


def volume_digest(manifest, work, wanted):
    """The notes for one volume, exactly as digest.py would print them."""
    keep_transcript = manifest.get("withTranslation", True)
    volumes = progress.volumes_of(manifest)
    if wanted:
        picked = [v for v in volumes if digest.volume_matches(v, wanted)]
        if not picked:
            names = ", ".join(v.get("name", "?") for v in volumes)
            sys.exit(f"No volume matches '{wanted}'. Volumes: {names}")
        volumes = picked

    paths = {v["path"] for v in volumes}
    pages = [
        p
        for p in manifest["pages"]
        if p.get("volume", manifest["root"]) in paths and progress.note_written(p)
    ]
    if not pages:
        sys.exit("No notes yet for that volume — nothing to fold in.")

    names = ", ".join(v.get("name", "?") for v in volumes)
    out = [f"# {len(pages)} notes — {names}"]
    for page in pages:
        out.append("")
        out.append("\n".join(digest.note_lines(page["note"], keep_transcript)))
    return names, "\n".join(out)


SPEC_START = "<!-- profile-spec:start -->"
SPEC_END = "<!-- profile-spec:end -->"


def spec():
    """The standard, read out of SKILL.md rather than copied into here.

    There were two statements of it for a while — one in SKILL.md for whoever
    writes the profile by hand, one in this prompt for Gemini — and they drifted
    within a day. SKILL.md's example `speech` was a 54-character fragment with
    no Vietnamese in it, and the model reproduced exactly that. One copy, and
    the file a human edits is the copy.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        body = text.split(SPEC_START, 1)[1].split(SPEC_END, 1)[0]
    except IndexError:
        sys.exit(f"{path}: cannot find {SPEC_START} … {SPEC_END} — the spec the job sends lives there")
    return body.strip()


def tree_rules(with_translation, first_volume=False):
    """The shared spec, plus the envelope for pass 1: the relationship tree."""
    opening = (
        """
You are the editor of a manga series' character dictionary.

The dictionary is EMPTY and you are starting it. Write every character this
volume establishes — not a sample of them, and not only the two or three
leads. Anyone who speaks to a named character on more than one page belongs in
it now.
"""
        if first_volume
        else """
You are the editor of a manga series' character dictionary.

You are folding ONE new volume into a dictionary that already covers the
earlier ones. Keep what is there. Correct it only where this volume shows it
was wrong. Add what is new: new characters, new relations, new mood variants,
more examples. Do not restate a volume that is already described.
"""
    )
    envelope = opening + """

THIS PASS writes the relationship tree, not the character. For every
character object you upsert, fill: id, name, nameJa, aliases, gender,
ageGroup, relations (each with "to", "relation" and "address"), selfTerms,
appearances. Leave "role", "personality" and "speech" as empty strings "" —
a second pass writes those next, once this tree is settled, so it can lean on
the relations you write here instead of guessing personality and address in
the same breath.

One thing a first pass loses, so watch for it:
- COVERAGE. Write a relation for EVERY character this one speaks to on more
  than one page, not just the obvious two or three, and give each one its mood
  variants. Under-writing here is the most common failure.

There is no cap on the cast and no judgement call to make about who is
"central enough": a character qualifies by the rule alone — more than 10
pages, OR seen in 2 or more volumes (check the digest's `volumes` column, not
just `appearances`) — and every character who qualifies goes in, however long
that makes the dictionary.

Return a PATCH as JSON, not the whole dictionary:

{
  "characters": {
    "upsert": [ <full character object, role/personality/speech left "" >, ... ],
    "remove": [ "<id>", ... ]
  },
  "address": [ ... ], "glossary": [ [ja, vi], ... ],
  "approach": [ ... ], "voice": [ ... ], "soundEffects": [ ... ]
}

- "upsert" carries the COMPLETE object for every character you are adding or
  changing, with all of its relations — it replaces that id wholesale. Leave a
  character out entirely if this volume gives you nothing to change about it.
  (Its existing role/personality/speech, if any, survive elsewhere — you are
  not responsible for them this pass, so leaving them "" here loses nothing.)
- "remove" retires an id only if it turns out not to be a real, distinct
  character after all (a duplicate, a misread) — never to make room; there is
  no room to make.
- The five lists are FULL REPLACEMENTS: give each one back complete, carrying
  forward the entries that still hold. An omitted list is left untouched.

Hard rules the result is checked against, and rejected on: gender one of "",
"male", "female"; ageGroup one of "", "child", "teen", "young_adult", "adult",
"middle_age", "elder"; ids lower-case ASCII with dashes; every relation needs a
non-empty "address" and a "to" that is still a character after the patch; at
most 14 address and 60 glossary entries; glossary entries are
[japanese, vietnamese] pairs.

Below is the standard itself, exactly as the skill states it. It describes the
FINISHED character, including role/personality/speech — read it for how the
relations you write now need to fit together with those, even though you are
not writing them yet.
"""
    mode = (
        """
This series HAS a published translation, so `approach`, `voice` and
`soundEffects` apply as the standard describes them.
"""
        if with_translation
        else """
This series has NO translation — only the Japanese original. So the "Original
only" paragraph of the standard governs: names, selfTerms and every relation's
address are your SUGGESTED Vietnamese with the Japanese they stand for, and
`approach`, `voice` and `soundEffects` MUST be returned as empty lists.
"""
    )
    return envelope + "\n" + ("=" * 70) + "\n" + spec() + "\n" + ("=" * 70) + "\n" + mode


def volumes_seen(pages):
    """Distinct volumes a character's page ids fall in — mirrors digest.py's
    `volume_spread`, kept separate since this module doesn't import digest
    for it alone."""
    return len({p.rsplit("-", 1)[0] for p in pages if "-" in p})


def qualifying_but_missing(cast, profile):
    """Characters the cast-selection rule already admits — more than 10
    pages, or seen in 2+ volumes — that the dictionary doesn't have yet.

    Computed here rather than left for the model to notice on its own: a cast
    digest can list well over 100 characters, most shown only as a one-line
    tail entry, and a rule that depends on the model reliably scanning all of
    them (rather than reasoning about the few it already recognises as
    important) is a rule that quietly stops applying past whatever length its
    attention holds up for. Measured on a real 128-character run: naming the
    rule in the spec caught some but silently missed 11 qualifying characters
    while also wrongly adding 6 that did not qualify — reverting to its own
    sense of who mattered instead of checking the two numbers. Handing back
    the exact list closes that gap; the model still writes what each one IS,
    it just cannot decide anymore who's eligible.
    """
    published = {c.get("id") for c in profile.get("characters", [])}
    out = []
    for c in cast:
        ident = c.get("id")
        if not ident or ident in published:
            continue
        pages = c.get("pages", [])
        pc, vc = len(pages), volumes_seen(pages)
        if pc > 10 or vc >= 2:
            out.append({"id": ident, "name": c.get("name", ""), "pages": pc, "volumes": vc})
    out.sort(key=lambda c: (-c["volumes"], -c["pages"]))
    return out


def build_tree_prompt(
    profile, cast_text, notes_text, names, with_translation, first_volume=False, must_add=None,
):
    must_add_block = ""
    if must_add:
        lines = "\n".join(
            f"- {c['id']} — {c['name']} ({c['pages']}p, {c['volumes']}v)" for c in must_add
        )
        must_add_block = "\n".join(
            [
                "",
                "=" * 70,
                "MUST ADD THIS PASS — already qualify by the rule above, not yet in the dictionary",
                "=" * 70,
                "Every id below already earns a place (checked mechanically against the cast "
                "digest, not left to you to notice in a long list): each is either on more than "
                "10 pages, or seen in 2+ volumes, or both. Upsert every one of them — do not "
                "reassess whether they qualify, only write who they are:",
                lines,
            ]
        )
    return "\n".join(
        [
            tree_rules(with_translation, first_volume),
            must_add_block,
            "",
            "=" * 70,
            "THE DICTIONARY SO FAR (earlier volumes; patch this)",
            "=" * 70,
            json.dumps(profile, ensure_ascii=False, indent=1),
            "",
            "=" * 70,
            "THE CAST (ids, appearance counts, who addresses whom and how)",
            "=" * 70,
            cast_text,
            "",
            "=" * 70,
            f"THE NEW VOLUME'S NOTES — {names}",
            "=" * 70,
            notes_text,
            "",
            "=" * 70,
            f"Fold '{names}' into the dictionary's relationship tree. Return only the patch JSON.",
        ]
    )


def personality_rules(with_translation):
    """The shared spec, plus the envelope for pass 2: role/personality/speech."""
    envelope = """
You are the editor of a manga series' character dictionary.

The relationship tree below — who each of these characters is to the others,
and how they address them — was just settled from this volume's dialogue. Use
it as SETTLED CONTEXT: a character who snaps at one person and stays formal
with another should read that way in `personality`, not be described as if
those relations did not already exist. You are not being asked to touch the
tree; only to write who each of these people IS.

For EVERY character listed below (by id), write "role", "personality" and
"speech" from this volume's notes. If the character already had these fields
(shown alongside the tree, when not empty), KEEP what still holds and correct
only what this volume shows was wrong — do not rewrite a settled character
from scratch.

Return a PATCH as JSON, a list of partial objects — not the full dictionary
and not the tree:

{
  "characters": [
    {"id": "<id>", "role": "...", "personality": "...", "speech": "...",
     "speechEvidence": ["<Japanese line copied from the notes>", ...]},
    ...
  ]
}

Every id you were given below must appear exactly once. "speechEvidence" is
mandatory whenever "speech" makes a claim about register or tone — copy the
Japanese line character-for-character out of the notes; it is checked against
them, and an uncited or invented line gets the whole entry rejected.

Below is the standard itself, exactly as the skill states it — it also
describes the relations you are NOT writing this pass; read past that to the
`role`/`personality`/`speech` guidance, which is what this pass is checked
against.
"""
    mode = (
        ""
        if with_translation
        else """
This series has NO translation — only the Japanese original. So the "Original
only" paragraph of the standard governs `speech`: describe the Japanese
(sentence endings, dialect, tics) and how to carry it into Vietnamese: the
translation has not been made yet.
"""
    )
    return envelope + "\n" + ("=" * 70) + "\n" + spec() + "\n" + ("=" * 70) + "\n" + mode


def characters_for_personality_pass(tree_profile, ids):
    """The settled tree, in the shape the personality pass reads it in: id,
    name, relations/address, and whatever role/personality/speech this
    character already had (so the model can keep it rather than reinvent it),
    for exactly the ids the tree pass touched this volume."""
    names = {c.get("id"): c.get("name") for c in tree_profile.get("characters", [])}
    out = []
    for c in tree_profile.get("characters", []):
        if c.get("id") not in ids:
            continue
        entry = {
            "id": c.get("id"),
            "name": c.get("name"),
            "nameJa": c.get("nameJa", ""),
            "gender": c.get("gender", ""),
            "ageGroup": c.get("ageGroup", ""),
            "relations": [
                {
                    "to": names.get(r.get("to"), r.get("to")),
                    "relation": r.get("relation", ""),
                    "address": r.get("address", ""),
                }
                for r in c.get("relations", []) or []
            ],
        }
        for field in ("role", "personality", "speech"):
            if c.get(field):
                entry[f"current_{field}"] = c[field]
        out.append(entry)
    return out


def build_personality_prompt(tree_profile, ids, notes_text, names, with_translation):
    characters = characters_for_personality_pass(tree_profile, ids)
    return "\n".join(
        [
            personality_rules(with_translation),
            "",
            "=" * 70,
            "THE SETTLED TREE (write role/personality/speech for exactly these ids)",
            "=" * 70,
            json.dumps(characters, ensure_ascii=False, indent=1),
            "",
            "=" * 70,
            f"THE VOLUME'S NOTES — {names}",
            "=" * 70,
            notes_text,
            "",
            "=" * 70,
            "Write role/personality/speech for every id above. Return only the patch JSON.",
        ]
    )


def call_gemini(prompt, model, keys, note=""):
    """Try each key in turn; rate limits move on, server errors back off."""
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
    ).encode("utf-8")
    url = ENDPOINT.format(model=model)

    last = ""
    for index, key in enumerate(keys, 1):
        for attempt in range(1, ATTEMPTS_PER_KEY + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                    payload = json.load(response)
                break_reason = (payload.get("candidates") or [{}])[0].get("finishReason")
                if break_reason not in (None, "STOP"):
                    last = f"finishReason {break_reason}"
                    print(f"  key {index}: {last}", file=sys.stderr)
                    break
                parts = (payload.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    last = "empty response"
                    print(f"  key {index}: {last}", file=sys.stderr)
                    break
                used = payload.get("usageMetadata", {})
                print(
                    f"  key {index}/{len(keys)} answered{note} — "
                    f"{used.get('promptTokenCount', '?')} in, "
                    f"{used.get('candidatesTokenCount', '?')} out",
                    file=sys.stderr,
                )
                return text
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last = f"HTTP {e.code} {detail}"
                # A bad model name or a malformed request is not the key's
                # fault — trying the other 39 only wastes a minute.
                if e.code in (400, 404):
                    sys.exit(f"{model}: HTTP {e.code}\n{detail}\n\nPick another with --model.")
                if e.code in (429, 403):
                    print(f"  key {index}: {e.code}, next key", file=sys.stderr)
                    break
                if attempt < ATTEMPTS_PER_KEY:
                    time.sleep(2 * attempt)
                    continue
                print(f"  key {index}: {last}", file=sys.stderr)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = str(e)
                if attempt < ATTEMPTS_PER_KEY:
                    time.sleep(2 * attempt)
                    continue
                print(f"  key {index}: {last}", file=sys.stderr)
    sys.exit(
        f"all {len(keys)} keys failed for {model} (last: {last}).\n"
        "If this is the free-tier quota, try --model gemini-flash-lite-latest, or wait: the free daily quota resets."
    )


def parse_patch(text):
    """The first JSON object in the answer, ignoring anything after it.

    responseMimeType asks for JSON and usually gets it, but a stray closing
    brace after the object is common enough to have broken a run: json.loads
    rejects the whole thing, and a greedy {.*} regex swallows the stray brace
    and fails too. raw_decode stops at the end of the first complete value.
    """
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def apply_patch(profile, patch, cast):
    """Upsert by id, retire the removals, replace the lists that were given.

    The tree pass is told to leave role/personality/speech "" — the
    personality pass owns them. So an upsert that comes back with one of
    those blank does not get to erase what an earlier volume already wrote;
    the old value survives here, and only a personality-pass patch (see
    `apply_personality_patch`) actually replaces it.
    """
    out = {k: list(v) for k, v in profile.items()}
    characters = patch.get("characters") or {}
    if isinstance(characters, list):  # tolerate a bare list of upserts
        characters = {"upsert": characters}

    by_id = {c.get("id"): dict(c) for c in out.get("characters", [])}
    order = [c.get("id") for c in out.get("characters", [])]
    for c in characters.get("upsert", []) or []:
        ident = str(c.get("id") or "").strip()
        if not ident:
            continue
        previous = by_id.get(ident)
        if previous:
            for field in ("role", "personality", "speech"):
                if not str(c.get(field) or "").strip() and previous.get(field):
                    c[field] = previous[field]
        if ident not in by_id:
            order.append(ident)
        by_id[ident] = c
    for ident in characters.get("remove", []) or []:
        by_id.pop(ident, None)

    counts = {c.get("id"): len(c.get("pages", [])) for c in cast}
    kept = [by_id[i] for i in order if i in by_id]
    for c in kept:
        if c.get("id") in counts:
            c["appearances"] = counts[c["id"]]
    kept.sort(key=lambda c: -(c.get("appearances") or 0))
    out["characters"] = kept

    for key in ("address", "glossary", "approach", "voice", "soundEffects"):
        if key in patch and isinstance(patch[key], list):
            out[key] = patch[key]
    return out


def apply_personality_patch(profile, patch):
    """Merge role/personality/speech into already-upserted characters by id —
    a field patch, not a wholesale replace, so it never touches relations,
    gender, or anything the tree pass just settled.

    Copies each character dict before mutating it (`apply_patch` does the
    same) — a retry calls this again against the same `tree` object, and
    without the copy the first, rejected attempt's writes would already be
    sitting on those dicts when the second one runs.
    """
    out = {k: list(v) for k, v in profile.items()}
    out["characters"] = [dict(c) for c in out.get("characters", [])]
    entries = patch.get("characters") or []
    if not isinstance(entries, list):
        entries = []
    by_id = {str(e.get("id") or "").strip(): e for e in entries if isinstance(e, dict)}
    for c in out.get("characters", []):
        entry = by_id.get(c.get("id"))
        if not entry:
            continue
        for field in ("role", "personality", "speech"):
            if field in entry:
                c[field] = entry[field]
        if "speechEvidence" in entry:
            c["speechEvidence"] = entry["speechEvidence"]
    return out


def validate(profile, cast, manifest):
    """finish.py's own checker, so anything that passes here passes there."""
    finish.problems.clear()
    finish.check_profile(
        json.loads(json.dumps(profile)),
        cast,
        manifest.get("withRaw", True),
        manifest.get("withTranslation", True),
    )
    return list(finish.problems)


JAPANESE = re.compile("[぀-ヿ一-鿿]")
# Long vowels, wave dashes and repeated small kana are written loosely in the
# notes and loosely again when quoted; a run of them is not what identifies a
# line, so both sides are flattened before comparing.
NOISE = re.compile(r"[\s ー〜~…\.。、,！!？?・「」『』\"'‘’゛゜／/‼⁉]+")


def notes_corpus(manifest):
    parts = []
    for page in manifest["pages"]:
        if progress.note_written(page):
            with open(page["note"], encoding="utf-8") as f:
                parts.append(f.read())
    return NOISE.sub("", "\n".join(parts))


def check_grounding(profile, corpus):
    """Was each quoted line actually in the notes, or was it invented?

    The validator checks shape. It cannot check whether 'Robin Mask calls
    Terryman ロビン' is true — and in the from-scratch test that exact line came
    back, fabricated, and passed. So the model is made to cite, and the
    citations are looked up here. This is the only check that catches an
    invented form of address.

    Returns (grounded, unfounded_list). Evidence is stripped as it goes: it is
    scaffolding for this check, and must not reach the published dictionary.
    """
    grounded, unfounded = 0, []
    for c in profile.get("characters", []):
        claims = [("speech", c.pop("speechEvidence", None) or [])]
        for r in c.get("relations", []):
            claims.append((f"→{r['to']}", r.pop("evidence", None) or []))
        for where, quotes in claims:
            for q in quotes if isinstance(quotes, list) else [quotes]:
                q = str(q)
                if len(JAPANESE.findall(q)) < 3:
                    continue  # too short to identify anything
                if NOISE.sub("", q) in corpus:
                    grounded += 1
                else:
                    unfounded.append((c.get("id", "?"), where, q))
    return grounded, unfounded


VIETNAMESE = re.compile(
    "[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I
)


def warn_thin(profile, grounding=None):
    """Things the validator cannot see, but that make a profile useless.

    Measured against a hand-written dictionary for the same six volumes: the
    model got the roster (29 of 30 ids) and the character tree (125 of 128
    address lines carried Vietnamese) right, but wrote `speech` as a bare
    English description with no Vietnamese at all, at a sixth of the length —
    and in two cases stated the rule backwards, telling the translator to
    coarsen a character whose politeness under cruelty IS the character. None
    of that fails validation. So it is said out loud instead.
    """
    notes = []
    if grounding is not None:
        ok, unfounded = grounding
        total = ok + len(unfounded)
        if total:
            notes.append(f"{ok}/{total} quoted lines were found in the notes")
        for ident, where, quote in unfounded[:12]:
            notes.append(f"  NOT IN THE NOTES — {ident} {where}: {quote}")
        if len(unfounded) > 12:
            notes.append(f"  …and {len(unfounded) - 12} more")
        if not total:
            notes.append("it cited nothing, so no claim here could be checked against the notes")
    speech = [c for c in profile["characters"] if c.get("speech")]
    no_viet = [c["id"] for c in speech if not VIETNAMESE.search(c["speech"])]
    if no_viet:
        notes.append(
            f"{len(no_viet)}/{len(speech)} 'speech' say nothing about Vietnamese "
            f"({', '.join(no_viet[:6])}{'…' if len(no_viet) > 6 else ''}) — "
            "that field is where a translator learns how to carry the register"
        )
    bloated = [c["id"] for c in profile["characters"] if len(c.get("role", "")) > 300]
    if bloated:
        notes.append(
            f"{len(bloated)} 'role' are over 300 characters "
            f"({', '.join(bloated[:6])}{'…' if len(bloated) > 6 else ''}) — a role says who "
            "someone is in a sentence or two; this one has turned into a plot summary, and it "
            "rides in every translation request"
        )
    thin = [c["id"] for c in speech if len(c["speech"]) < 200]
    if len(thin) > len(speech) // 3:
        notes.append(f"{len(thin)}/{len(speech)} 'speech' are under 200 characters — probably too thin to act on")
    bare = [r["to"] for c in profile["characters"] for r in c["relations"] if ";" not in r["address"]]
    if bare:
        notes.append(f"{len(bare)} relations give one form of address with no mood variants")
    if notes:
        print("\nworth a look before publishing:", file=sys.stderr)
        for n in notes:
            print(f"  - {n}", file=sys.stderr)
        print(
            "  Read the entries it added. The validator checks shape, not whether\n"
            "  an address line is true — a fabricated one passes.",
            file=sys.stderr,
        )


def touched_ids(patch):
    """The ids a tree-pass patch actually upserted this volume — exactly who
    the personality pass needs to cover next."""
    characters = patch.get("characters") or {}
    if isinstance(characters, list):
        characters = {"upsert": characters}
    return {
        str(c.get("id") or "").strip()
        for c in characters.get("upsert", []) or []
        if str(c.get("id") or "").strip()
    }


def summarise(before, after, names):
    was = {c.get("id") for c in before.get("characters", [])}
    now = {c.get("id") for c in after.get("characters", [])}
    print(f"folded in: {names}")
    print(f"  characters {len(was)} -> {len(now)}")
    if now - was:
        print(f"  added:   {', '.join(sorted(now - was))}")
    if was - now:
        print(f"  retired: {', '.join(sorted(was - now))}")
    changed = [
        c["id"]
        for c in after.get("characters", [])
        if c.get("id") in was
        and c != next(x for x in before["characters"] if x.get("id") == c["id"])
    ]
    if changed:
        print(f"  amended: {', '.join(changed)}")
    for key in ("address", "glossary", "approach", "voice", "soundEffects"):
        a, b = len(before.get(key, [])), len(after.get(key, []))
        if a != b:
            print(f"  {key} {a} -> {b}")


def main():
    args = sys.argv[1:]
    raw_only = "--raw-only" in args
    dry_run = "--dry-run" in args
    args = [a for a in args if a not in ("--raw-only", "--dry-run")]
    model = DEFAULT_MODEL
    wanted = ""
    for flag, target in (("--model", "model"), ("--volume", "volume")):
        if flag in args:
            i = args.index(flag)
            value = args[i + 1] if i + 1 < len(args) else ""
            del args[i : i + 2]
            if target == "model":
                model = value
            else:
                wanted = value
    if len(args) != 1:
        sys.exit(__doc__)

    root = os.path.abspath(args[0])
    work = os.path.join(root, "style_scan", "claude_raw" if raw_only else "claude")
    try:
        with open(os.path.join(work, "pages.json"), encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        sys.exit("missing pages.json — run prepare.py first")

    profile_path = os.path.join(work, "profile.json")
    profile = progress.read_json(profile_path, None) or dict(EMPTY)
    for key, value in EMPTY.items():
        profile.setdefault(key, list(value))
    cast = progress.read_json(os.path.join(work, "cast.json"), {}).get("characters", [])

    names, notes_text = volume_digest(manifest, work, wanted)
    cast_text = "\n".join(digest.cast_lines(work, manifest.get("withRaw", True)))
    with_translation = manifest.get("withTranslation", True)
    first_volume = not profile.get("characters")
    must_add = qualifying_but_missing(cast, profile)
    if must_add:
        print(
            f"{len(must_add)} character(s) already qualify but aren't in the dictionary yet "
            f"— telling the tree pass to add them: {', '.join(c['id'] for c in must_add)}",
            file=sys.stderr,
        )
    tree_prompt = build_tree_prompt(
        profile, cast_text, notes_text, names, with_translation,
        first_volume=first_volume, must_add=must_add,
    )

    print(
        f"tree prompt: {len(tree_prompt) / 1024:.0f} KB "
        f"(profile {len(json.dumps(profile, ensure_ascii=False)) / 1024:.0f} KB, "
        f"cast {len(cast_text) / 1024:.0f} KB, notes {len(notes_text) / 1024:.0f} KB)",
        file=sys.stderr,
    )
    if dry_run:
        print(
            "(the personality prompt is built from this pass's own output, "
            "so --dry-run cannot size it ahead of time)",
            file=sys.stderr,
        )
        return

    keys = load_keys()
    print(f"{model}, {len(keys)} keys", file=sys.stderr)
    corpus = notes_corpus(manifest)

    # Pass 1: the relationship tree.
    text = call_gemini(tree_prompt, model, keys, note=" (tree)")
    patch = parse_patch(text)
    if patch is None:
        rejected = os.path.join(work, "profile.rejected.json")
        with open(rejected, "w", encoding="utf-8") as f:
            f.write(text)
        sys.exit(f"the model did not return JSON; its answer is in {rejected}")

    tree = apply_patch(profile, patch, cast)
    tree_grounding = check_grounding(tree, corpus)
    problems = validate(tree, cast, manifest)

    if problems:
        print(f"tree pass: {len(problems)} problem(s); asking it to fix them", file=sys.stderr)
        retry = tree_prompt + "\n\n" + "\n".join(
            ["Your previous patch was rejected:"]
            + [f"- {p}" for p in problems]
            + ["Return a corrected patch, in the same shape."]
        )
        text = call_gemini(retry, model, keys, note=" (tree retry)")
        patch = parse_patch(text) or {}
        tree = apply_patch(profile, patch, cast)
        tree_grounding = check_grounding(tree, corpus)
        problems = validate(tree, cast, manifest)

    if problems:
        rejected = os.path.join(work, "profile.rejected.json")
        progress.write_json(rejected, tree)
        print(f"\ntree pass still rejected, profile.json untouched. Kept as {rejected}:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    ids = touched_ids(patch)
    merged, personality_grounding = tree, (0, [])

    # Pass 2: role/personality/speech, from the tree pass 1 just settled.
    if ids:
        personality_prompt = build_personality_prompt(tree, ids, notes_text, names, with_translation)
        text = call_gemini(personality_prompt, model, keys, note=" (personality)")
        ppatch = parse_patch(text)
        if ppatch is None:
            rejected = os.path.join(work, "profile.rejected.json")
            with open(rejected, "w", encoding="utf-8") as f:
                f.write(text)
            sys.exit(f"the model did not return JSON for the personality pass; its answer is in {rejected}")

        merged = apply_personality_patch(tree, ppatch)
        personality_grounding = check_grounding(merged, corpus)
        problems = validate(merged, cast, manifest)

        if problems:
            print(f"personality pass: {len(problems)} problem(s); asking it to fix them", file=sys.stderr)
            retry = personality_prompt + "\n\n" + "\n".join(
                ["Your previous patch was rejected:"]
                + [f"- {p}" for p in problems]
                + ["Return a corrected patch, in the same shape."]
            )
            text = call_gemini(retry, model, keys, note=" (personality retry)")
            ppatch = parse_patch(text) or {}
            merged = apply_personality_patch(tree, ppatch)
            personality_grounding = check_grounding(merged, corpus)
            problems = validate(merged, cast, manifest)

        if problems:
            rejected = os.path.join(work, "profile.rejected.json")
            progress.write_json(rejected, merged)
            print(f"\npersonality pass still rejected, profile.json untouched. Kept as {rejected}:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            sys.exit(1)

    progress.write_json(profile_path, merged)
    summarise(profile, merged, names)
    grounding = (
        tree_grounding[0] + personality_grounding[0],
        tree_grounding[1] + personality_grounding[1],
    )
    warn_thin(merged, grounding)
    print(f"\nwrote {profile_path}")
    print("Check it, then publish:")
    print(
        f"  python3 .claude/skills/style-read/finish.py '{root}'"
        + (" --raw-only" if raw_only else "")
    )


if __name__ == "__main__":
    main()
