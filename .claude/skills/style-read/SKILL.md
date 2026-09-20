---
name: style-read
description: Read a manga volume page by page by looking at the images — the Japanese original on its own, or with its published Vietnamese translation, or the translation alone — and build a profile for the app - the recurring cast with names, gender, age, personality and how each character addresses every other (a character tree), plus, when there is a translation, the translator's approach, voice, sound-effect handling and settled names. Writes a file the app's Style Scanner and Character Scanner pages load and the translator sends with every request. Use when the user runs /style-read or asks Claude to scan a volume's characters or read its translation style itself.
argument-hint: <folder> [--from VOL --to VOL] [--raw-only] [--pages N] [--only ID,ID] [--reader sonnet|opus]
---

# /style-read

You are reading a volume the way a careful editor does: looking at the artwork
to see who speaks to whom and in what mood, reading every balloon, and noting
how the translator carried the Japanese across. OCR sees one balloon at a time
and has to guess all of that. Accuracy is the point of this command.

What comes out is one profile that the app puts at the top of every
translation request: who each recurring character is, how they talk, how they
address each of the others (by mood), and how this translation handles the
text in general.

## Arguments

`$ARGUMENTS` is `<folder> [--from VOL --to VOL] [--raw-only] [--pages N] [--only ID,ID] [--reader sonnet|opus]`.
The user may say these in words ("folder Kinnikuman, tập 1 đến 5, chỉ raw"):
turn that into the flags.

- `<folder>`: the original in `raw/` and/or the published translation in
  `trans/`. Ask for it if missing.
- `--from VOL --to VOL`: `<folder>` is a whole series whose sub-folders are
  volumes; read the volumes from VOL to VOL as one run with **one cast**. VOL
  is a volume number (`第01巻`, `v01`, `tập 1` → `1`) or part of a folder name;
  one of the two may be left out to mean the first or last volume. Inside a
  volume only the images directly in it (or in its `raw/`, `trans/`) are read;
  other sub-folders such as `result/` are ignored. Page ids carry the volume:
  `v01-0291`. The work folder is in the series folder, and the result is
  written into every volume read, because the app opens one volume at a time.
- `--raw-only`: read only the Japanese original, even if a translation is
  there. Also needed when the original's images sit directly in the folder
  (images directly in the folder are otherwise taken as a translation).
- `--pages N` / `--only 0297,0309`: read only some pages — for trying the command
  out. The result is saved as a `-sample` profile.
- `--reader`: the model that reads the pages. Default `sonnet`, which is
  good at reading and uses far less of the plan's limits; `opus` for the
  hardest volumes. The profile itself is always written by you, in this session.

## Two ways to read

- **With a translation** (`raw/` + `trans/`, or `trans/` alone): the cast and
  the character tree, and how the translation was made. Work folder
  `<folder>/style_scan/claude/`.
- **Original only** (`raw/` alone, or `--raw-only`): the same cast and
  character tree, read from the Japanese. Nobody has translated it yet, so the
  Vietnamese names and forms of address are **your suggestions**, and there is
  no translator's style to describe. Work folder
  `<folder>/style_scan/claude_raw/`. Pass `--raw-only` to every script
  whenever the manifest says `"withTranslation": false`.

The steps are the same; where they differ, it is said below.

## 1. Prepare

```
python3 .claude/skills/style-read/prepare.py <folder> [--from VOL --to VOL] [--raw-only] [--pages N] [--only …]
```

This pairs each translated page with its original (in the translation mode,
pages only the original has are skipped), writes reading copies, and writes
`pages.json` in the work folder, along with the `notes/` and `updates/` folders
the batches write into.

Each page gets three images: the spread, and its right and left halves
(`rawHalves`, `transHalves` in the manifest). A reader looks at the spread and
falls back to a half only for a balloon it cannot read — which is why it needs
no shell and cannot crop. Face boxes stay fractions of the spread. If it cannot find a volume, it lists the
folders it saw — pick the right bound from them, or ask the user.

Read the manifest: `withTranslation` says which mode this is, `withRaw`
whether originals exist; each page has `raw` (or null), `trans` (or null), its
`volume`, its `note` path, and `done`. Pages already `done` are not read again —
an interrupted run continues where it stopped, in this session or a later one.

## 2. Read the pages in batches, with a shared cast

Split the pages not yet `done` into batches of **4**, in page order (volume by
volume, never mixing two volumes in a batch), and read them **one batch after
another** — never in parallel: each batch needs the cast the previous one left
behind, or the same character ends up under two names.

Four, and measured: a batch costs the calls it makes times the context each one
carries, and both grow with the batch — the pages stay in front of the reader
to the end, and it takes more turns to work through them. Batches of 8 cost
143k input tokens a page against 96k at 4; the fixed part is not what dominates.

Tell the user once, before the first batch: how many pages and batches there
are, and that the run can be stopped at any time and carried on later.

For each batch, start one subagent with the **`manga-page-reader`** type
(`model` from `--reader`). It carries the whole brief — how to read a spread,
what to write, both note formats — so the prompt is only the batch:

> Mode: original only *(or: with a translation)*.
> Pages, in this order — read the right half, then the left:
> - `v05-061` — right `<path>` left `<path>` (whole spread, if ever needed:
>   `<path>`) *(with a translation, the translated halves and the raw halves)*
> - …
> known.md: `<work>/known.md`
> Previous page's note: `<work>/notes/v05-060.md` *(or "none")*
> Notes folder: `<work>/notes/`

If `manga-page-reader` is not among the agent types this session offers, use
`general-purpose` and open the prompt with: "Read
`.claude/agents/manga-page-reader.md` and follow it." Do not paste the brief
into the prompt, and do not open the page images yourself — the subagents do,
which is what keeps this session's context small.

Wait for it, check its short report, then record the progress:

```
python3 .claude/skills/style-read/progress.py <folder> [--raw-only]
```

It does three things:

- folds the `### Cast` block at the end of each new note into `cast.json`,
  once per page (`cast_applied.json` remembers which) — a character already
  recorded is never overwritten, so a stale batch cannot undo a later one, and
  a character retired by hand is not conjured back;
- writes `style_scan/style_read_progress.json` into every volume being read
  (`status` `not_started` / `reading` / `read` / `published`, pages read, the
  last page read, the next one, and the `resume` command);
- brings `cast.json` up to date from the notes — the pages each character is
  on, the pages each pair addressed each other on — and marks what is
  **settled**, then writes `known.md` beside it for the next batch:
  - a character is settled once it is on at least 6 pages with a name,
    Japanese name (when there is a raw), gender, age group, `looks` and a face;
  - a pair is settled once it has been seen addressing each other on at least
    3 pages and the speaker's `addresses` entry for it has a `default`.

  Set `"hold": true` on a character in `cast.json` to keep it open (two people
  who look alike, a disguise, a character whose age is unclear).

`known.md` is what a batch reads, and it is written to stay small as the cast
grows: one line a character, the forms of address already recorded named but
not quoted, one-page walk-ons gathered at the end. Never hand a batch
`cast.json` instead — it outgrows a single read early in a series, so a reader
given it sees a fraction of the cast and invents ids for people already in it.

If the report says a volume just ended, update the dictionary (step 3) before
the next batch. If a batch reports a page it could not read, retry that page
once in the next batch. If it says it thinks two ids are the same person, look
at the two yourself before merging them — it was told not to.

### Stopping and carrying on

The notes, the update files, `cast.json` and `profile.json` are the whole
state; nothing is kept in this session. When the run stops — the plan's limit,
the user, a closed window — it carries on by running the same command again
(the `resume` value in any volume's `style_read_progress.json`), in this
session or a new one: prepare.py marks every page with a note as done and folds
in any update file left behind, and reading starts at the first page without a
note, with the cast and dictionary as they were. If progress.py says a volume
is "read but not yet in the profile", do step 3 for it first. Never delete the
work folder to "start clean" unless the user asks — that throws away every page
read.

When the user says to stop, or the limit is close, finish the batch in hand,
run progress.py, and tell the user how far it got and the command to carry on.

## 3. Write the profile — the series' character dictionary

`profile.json` in the work folder is the dictionary: it grows volume by volume
and never starts over.

Read the volume's notes and the cast with `digest.py`, not file by file: a
volume is around a hundred notes, and `cast.json` carries page lists and face
boxes that only the scripts use.

```
python3 .claude/skills/style-read/digest.py <folder> --volume <name or number> [--raw-only]
python3 .claude/skills/style-read/digest.py <folder> --cast [--raw-only]
```

The first prints that volume's notes in one go (reading the original alone, the
dialogue transcript is left out — there the notes are evidence about the cast,
and the profile never quotes them). The second prints the cast in the shape the
profile is written from.

- **A series:** update it each time a volume has been read to the end (the
  batch report says so; progress.py lists it as "read but not yet in the
  profile"). Read the current `profile.json` (if any), `digest.py --cast`, and
  the digest of **only that volume** — the volumes before it are already in the
  profile. Keep what is there, correct it where the new volume shows it was
  wrong, and add what is new: new characters, new relations, new mood variants,
  more examples.
- **One folder:** write it once, when every page has a note, from the digest of
  all the notes.

This text goes to the top of every translation request, so every entry must be
something a translator can act on — in English, with Vietnamese quoted as the
translation writes it.

```json
{
  "characters": [
    {"id": "kinnikuman", "name": "Kinnikuman", "nameJa": "キン肉マン",
     "aliases": ["Suguru", "hoàng tử"], "gender": "male", "ageGroup": "young_adult",
     "role": "Prince of planet Kinniku, a clumsy, self-important superhero",
     "personality": "boastful, cowardly under pressure, soft-hearted",
     "speech": "pompous when posing as a hero; whines and pleads when scared",
     "selfTerms": ["ta (posing as a hero, to almost everyone)", "tớ (pleading)", "tôi (polite, to Mari)"],
     "appearances": 60,
     "relations": [
       {"to": "meat", "relation": "his servant and friend",
        "address": "ta/ngươi by default; cậu when friendly; tao/mày when scolding; tớ when pleading"}
     ]}
  ],
  "approach": [],
  "voice": [],
  "address": [],
  "soundEffects": [],
  "glossary": [["キン肉マン", "Kinnikuman"]]
}
```

- **characters** (at most 30): the recurring cast — everyone on at least 3
  pages, plus anyone central. Take ids, names, gender and age from the cast
  digest, merging any duplicates it still shows. Write `role`, `personality`
  and `speech` from what the notes show. `relations`: for each character they
  speak to on more than one page (the `(Np)` beside each `→` line), what that
  character is to them and how they address them — the default first, then each
  mood variant, from that line and the ADDRESS lines. `appearances` is the
  digest's appearance count. Over 30 across a long series, keep
  the ones a translator meets most.
- **approach** (at most 10): how the Japanese is carried across — how speech
  levels become Vietnamese pronouns and particles, what is localised and what is
  glossed in brackets, whether honorifics like -chan/-kun are kept, how far the
  tone is sharpened, how names and attacks are handled, and every term that is
  translated inconsistently.
- **voice** (at most 12): what makes the Vietnamese recognisable — slang,
  internet slang, regional words, idioms, particles, talking to the reader —
  each with examples quoted from the notes.
- **address** (at most 14): rules that cut across characters (children and
  adults, strangers, the public), since per-character address is under
  `characters`.
- **soundEffects** (at most 8), **glossary** (at most 60: names and recurring
  terms with the most frequent spelling; Japanese empty when there is no raw).

**Original only:** `name` is the name to use in Vietnamese; `nameJa` the
Japanese. `selfTerms` and each relation's `address` are the suggested
Vietnamese with the Japanese they stand for, e.g.
`"ta (わし, to anyone younger)"`, `"tôi/ngài by default (です/ます, 長官); tao/mày when furious (俺/てめえ)"`.
`speech` describes the Japanese (sentence endings, dialect, tics) and how to
carry it. `address` may hold cross-cutting suggestions; `approach`, `voice` and
`soundEffects` stay empty; `glossary` pairs Japanese with the suggested
Vietnamese.

Leave out anything true of every manga translation. Only include a pronoun pair
between two characters when the notes or the cast show those two using it.

Then publish:

```
python3 .claude/skills/style-read/finish.py <folder> [--raw-only]
```

It checks everything and writes the profile into the project's
`style_profiles/` library and, with the character tree and face crops, into
every folder read to the end (for a series, every finished volume, plus
`<series>/character_dictionary.json`). Volumes still being read are left for a
later run. If it lists problems, fix `profile.json` (or `cast.json`) and run it
again. Then go on with the next batch.

## 4. Report

When every page has been read and the last volume published, tell the user, in
Vietnamese:

- how many pages (and which volumes) were read, and which pages could not be;
- the cast, how many are settled, and the address rules you are least sure of
  and why;
- terms translated inconsistently (with a translation), or the names and
  forms of address you were least sure how to carry into Vietnamese (original
  only — they need a look before translating);
- where to see it: in the app, open the same folder (for a series, any of the
  volumes read) on **Style Scanner** (the profile, editable, and in **Saved
  profiles**) and on **Character Scanner** (the character tree with faces).
  Choosing **Use for translation** on Style Scanner applies it to the loaded
  model at once — the saved profile works for volumes not read yet, too.
