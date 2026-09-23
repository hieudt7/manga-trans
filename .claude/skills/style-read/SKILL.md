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
falls back to a half only for a balloon it cannot read. Face boxes stay
fractions of the spread. If it cannot find a volume, it lists the folders it
saw — pick the right bound from them, or ask the user.

Read the manifest: `withTranslation` says which mode this is, `withRaw`
whether originals exist; each page has `raw` (or null), `trans` (or null), its
`volume`, its `note` path, and `done`. Pages already `done` are not read again —
an interrupted run continues where it stopped, in this session or a later one.

**Then, before reading, pre-compute a face hint for every page — optional, but
worth doing when the models are installed:**

```
python3 .claude/skills/style-read/face_hints.py <folder> [--raw-only]
```

A reader estimating a face box by eye never sees the crop it produces; a real
detector does not have that problem. Measured on a real six-volume series: 9/9
tight, correct face crops from the detector on a page a blind estimate had
gotten wrong. This writes `<work>/face_hints/<page id>.json` for every page —
563 pages in under 2.5 minutes, no tokens spent — and a reading batch is told
where to find them (below); a reader given a hint copies its box instead of
guessing one. Needs the `face-hint` binary and its two models
(`cargo build --release -p koharu-ml --bin face-hint`;
`python3 tools/export_ccip.py --face-only` for the face detector, plus the
bubble detector's own export step). Skip this script entirely if those are not
set up — every reading batch still works without it, just back to estimating
by eye.

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

### Read in a session that does nothing else

This matters more than anything the reader does. You pay for every batch twice:
once for the reader, and once for your own context, which is re-sent on every
call you make. The reader's context is thrown away when the batch ends; yours
is not — it grows with every report you read and every file you open, and the
loop makes two calls a batch.

Measured over one run of 33 batches: the readers cost 10.1M input tokens and
this session cost 62.6M — **86% of the bill was the session driving the loop,
not the reading**. Its context had grown from 38k to 511k, so the last batches
cost about 450k a call before a single page was opened.

So, while the loop is running:

- do nothing else in this session — no analysis, no measuring, no commits;
- do step 3 by **running the job** (`write_profile.py`) and reading the few
  lines it prints. Never open the volume digest to write the profile by hand
  here: that is 170KB of context which then rides on every batch after it, and
  it is what made volume 6 cost 1.3M tokens against the readers' 187k;
- do not open notes, `cast.json`, `profile.json` or the page images yourself;
- keep your own replies to a line. The batch report is already in your context
  whether you summarise it or not.

If the loop has been running long enough that your context is large, stop and
carry on in a new session. Nothing is lost: the notes are the state, and
`resume` in any volume's `style_read_progress.json` is the command.

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
> Face hints folder: `<work>/face_hints/` *(omit this line entirely if
> face_hints.py was not run — a reader with no hints folder just estimates
> boxes by eye, as before)*

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

**Do not write it yourself.** Run the job:

```
python3 .claude/skills/style-read/write_profile.py <folder> --volume <name or number> [--raw-only]
```

It calls Gemini on the free keys in `gemini_keys.txt`, **twice, in sequence,
not once:**

1. **The tree pass.** Sends the volume's notes, the cast and the profile so
   far, and asks for the relationship tree — id, name, gender, ageGroup,
   `relations` (who each character addresses, and how, by mood) — for every
   character this volume touches. `role`, `personality` and `speech` are left
   `""` here on purpose.
2. **The personality pass.** Once the tree pass's patch is applied and
   validated, it is sent back — as *settled* context, not something to
   redecide — together with the same notes, and Gemini is asked for `role`,
   `personality` and `speech` for exactly the characters the tree pass just
   touched. Personality is written from the dialogue with the character's
   relations already in front of the model, not guessed in the same breath as
   who-addresses-whom — a character who snaps at one person and stays formal
   with another can be described that way, instead of the tree and the
   personality being two independent guesses that happen to agree.

Both passes are checked with finish.py's own validator and retried once on
failure, same as a single pass would be; a rejected personality pass leaves
`profile.json` exactly as the tree pass left it (nothing half-written). Read
only the summary printed at the end — a few lines naming the characters added,
retired and amended.

Nothing in this step needs your context: it is a function of three files on
disk. Doing it in the session costs far more than the reading does. Measured on
Kinnikuman volume 6, the three readers cost 187k tokens and the session that
drove them cost roughly 1.3M — almost all of it the volume digest (93KB) and
the profile (75KB) riding on every call after they were opened. The job sends
the same digest and profile once to each pass, to a key that costs nothing.

Each pass asks for a patch, not the whole file — the tree pass upserts full
character objects by id (ids to retire, the short lists in full), the
personality pass a smaller list of `{id, role, personality, speech}` — so a
character this volume did not touch keeps exactly the bytes it had, and a
truncated answer cannot quietly eat five volumes of work. If a pass's result
fails validation it is retried once with the problems fed back; if it still
fails, `profile.json` is left alone and that pass's answer is kept as
`profile.rejected.json`.

`--model` picks the model; the default is the alias `gemini-pro-latest`.
**Pinned versions get retired** — `gemini-2.5-pro` and `gemini-2.5-flash` both
went 404 for new keys. Free quota for pro models is effectively zero, so in
practice use `--model gemini-flash-latest`, which answers on the first key.

Then read the result before publishing. You are still the editor: check the new
characters, spot-check a few `address` lines against the notes, and fix
`profile.json` by hand where it is thin. `digest.py` is how you look at the
evidence for one thing without reading the whole volume:

```
python3 .claude/skills/style-read/digest.py <folder> --volume <name or number> [--raw-only]
python3 .claude/skills/style-read/digest.py <folder> --cast [--raw-only]
```

The first prints that volume's notes in one go (reading the original alone, the
dialogue transcript is left out — there the notes are evidence about the cast,
and the profile never quotes them). The second prints the cast in the shape the
profile is written from. Opening either one costs you the context the job
exists to avoid, so reach for them to check something, not by default.

What the job is told to produce, and what you are checking it against:

- **A series:** run the job each time a volume has been read to the end (the
  batch report says so; progress.py lists it as "read but not yet in the
  profile"), with `--volume` naming **only that volume** — the ones before it
  are already in the profile. It keeps what is there, corrects it where the new
  volume shows it was wrong, and adds what is new: new characters, new
  relations, new mood variants, more examples.
- **One folder:** run it once, when every page has a note, with no `--volume`.

Everything between the two markers below is the **one** statement of the
standard. `write_profile.py` reads it straight out of this file and puts it in
the prompt, so what Gemini is held to and what you are holding it to cannot
drift apart. Edit it here and the job follows; do not keep a second copy.

<!-- profile-spec:start -->

This text goes to the top of every translation request, so every entry must be
something a translator can act on.

**Write the whole dictionary in Vietnamese.** Every prose field — `role`,
`personality`, `speech`, `relation`, `address`, `selfTerms`, and the
cross-cutting lists. Quote Japanese in brackets where it is the evidence for a
claim. Do not mix languages between volumes: one Vietnamese file, always.

**What this is for, and what it is not.** It is a character tree: who each
recurring character is, their age and gender, how they talk, and how they
address each of the others. Keep every field to that job.

- `role` is **one or two sentences** saying who the character is — not a
  summary of the plot. A new volume can correct it; it does not get a new
  paragraph each time. If it is over about 200 characters, it has turned into a
  plot summary and should be cut back.
- `personality` and `speech` describe the person and the voice, not events.
- `relations` is the point of the file. Spend the length there.

**Choosing the cast.** Include every character on **more than 10 pages** —
`appearances` in the cast digest — **or seen in 2 or more volumes** —
`volumes` in the same digest, computed from the page ids' volume prefixes.
Either one alone qualifies; do not weigh them against `headline` or `speech`
or any other judgement call. The volumes count catches who the page-count rule
alone would miss: a character thin in any single volume but who keeps coming
back, series-wide — 2 pages in volume 2, 3 in volume 4, 5 in volume 6 is 10
pages total (short of "more than 10") across three separate returns, and that
recurrence is itself the signal a one-off with the same 10 pages in one volume
does not carry. Both rules stay deliberately blunt: simple enough that two
different runs pick the same cast from the same notes, which a "central but
thin" exception never was.

There is no cap — every character either rule reaches goes in, however long
that makes the cast. On a six-volume, 563-page, 128-character run, the >10
rule alone landed on 29; the volumes rule adds whoever recurs series-wide on
fewer pages than that (see `digest.py`'s `volumes` column).

This does still cost real central characters who are simply seen less, and
never come back — a one-off retired champion who sets a whole arc's plot in
motion on seven pages, in one volume, say. That loss is accepted on purpose:
both rules stay about being seen enough, on either axis, not about narrative
weight, which no two runs would judge alike.

A character object. This is the standard to match, not a sketch of one; every
field here does work a shorter version would not do.

```json
{
  "characters": [
    {"id": "robin-mask", "name": "Robin Mask", "nameJa": "ロビン・マスク",
     "aliases": ["ロビン"], "gender": "male", "ageGroup": "adult",
     "role": "Võ sĩ giáp sắt của nước Anh, nhà vô địch kỳ trước và là đối thủ lớn nhất của nhân vật chính.",
     "personality": "Kiêu ngạo, trịch thượng nhưng trọng danh dự — từ chối cả chiến thắng mình thấy không xứng đáng. Bị làm nhục thật sự thì mất kiểm soát và trở nên tàn độc.",
     "speech": "Nền là わたし/キミ trang trọng, điềm đạm. Căng lên thì rơi xuống lối nói cộc — だぜ khi đang khoá đòn, 許さん khi điên tiết. Nhưng KHÔNG phải cứ giận là bỏ lịch sự: câu lạnh nhất của anh ta, tuyên bố sẽ giết đối thủ, vẫn giữ thể ます —「死んでもらいます!」. Tiếng Việt: 'ta/cậu' quý tộc trịch thượng làm nền, gằn xuống 'tao/mày' khi đang khoá đòn — nhưng câu doạ giết phải giữ vẻ lịch sự lạnh gáy ('mời cậu chết đi' chứ không phải 'tao giết mày'), vì chính chỗ lịch sự đó mới rợn.",
     "speechEvidence": ["死んでもらいます", "許さん"],
     "selfTerms": ["ta (わたし, trịch thượng, mặc định)", "tao (khi khoá đòn)"],
     "appearances": 121,
     "relations": [
       {"to": "gania-mask", "relation": "huấn luyện viên riêng kiêm người phụ tá của anh ta",
        "address": "ta/Gania (gọi tên rút gọn) — thừa nhận đã nương tay; quát 'どけガニア!!' xua ông ra khi đang điên",
        "evidence": ["どけガニア"]}
     ]}
  ],
  "approach": [],
  "voice": [],
  "address": [],
  "soundEffects": [],
  "glossary": [["ロビン・マスク", "Robin Mask"]]
}
```

Three things that example is showing, and that are checked:

1. Every `speech` ends with a sentence beginning `Tiếng Việt:` saying how to
   carry the register into Vietnamese. Describing the Japanese alone is
   useless — whoever reads this dictionary writes Vietnamese.
2. Every `speech` carries `speechEvidence`, and every relation carries
   `evidence`: short Japanese strings **copied character for character** out of
   the notes. They are looked up in the notes; anything not found there is
   reported. Copy, never reconstruct, never normalise, never translate. If
   there is no line to copy, the claim does not go in. (Both fields are
   scaffolding for that check — they are stripped before anything is written
   and never reach the published dictionary.)
3. `address` gives the default **first**, then each mood variant, separated by
   semicolons — not one flat form.

Getting a register **backwards** is the most damaging mistake available here,
and the most tempting, because the obvious summary is usually wrong in the same
direction. Politeness is not a measure of calm: a character can plan a murder,
threaten a child or lock on a hold while keeping です/ます, and for several of
them that held politeness *is* the character. Before writing a `speech`, find
the line where the character is at their worst and check what register it is
actually in. Where politeness survives cruelty, say so, and say plainly that
the translator must **not** coarsen them.

- **characters** (no cap): the recurring cast, per "Choosing the cast" above —
  everyone on more than 10 pages, or seen in 2 or more volumes, whichever
  either. Take ids, names, gender and age from the cast digest, merging any
  duplicates it still shows. Write `role`, `personality` and `speech` from
  what the notes show. `relations`: for each character they speak to on more
  than one page (the `(Np)` beside each `→` line), what that character is to
  them and how they address them — the default first, then each mood variant,
  from that line and the ADDRESS lines. `appearances` is the digest's
  appearance count.
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
Japanese. **Keep the name, don't invent one.** For a personal name written in
katakana — especially a foreign-derived one (a wrestler's ring name, an
English loanword like ミート/Meat) — the Vietnamese `name` is that name kept as
spelled or romanized (`Meat`, `Terryman`, `Robin Mask`), never a substituted
Vietnamese word chosen for sound or whimsy (`Mít` for ミート is wrong: it
replaces the name with an unrelated Vietnamese word instead of carrying it
over). Only translate when there is no personal name to carry over and the
"name" is actually a title or epithet (`委員長` → `Chủ tịch Ban tổ chức`,
`キン肉大王` → `Đại Vương`) — there the Vietnamese is a translation of what the
title means, not a name substitution. `selfTerms` and each relation's
`address` are the suggested Vietnamese with the Japanese they stand for, e.g.
`"ta (わし, to anyone younger)"`, `"tôi/ngài by default (です/ます, 長官); tao/mày when furious (俺/てめえ)"`.
`speech` describes the Japanese (sentence endings, dialect, tics) and how to
carry it. `address` may hold cross-cutting suggestions; `approach`, `voice` and
`soundEffects` stay empty; `glossary` pairs Japanese with the suggested
Vietnamese.

Leave out anything true of every manga translation. Only include a pronoun pair
between two characters when the notes or the cast show those two using it.

<!-- profile-spec:end -->

One thing the spec asks for that is easy to lose, and that was measured going
wrong: **coverage**. A first pass tends to write one relation a character and
stop — scored against the ADDRESS lines in the notes, a careful hand pass
reached 91% of the pairs seen on two or more pages while a first machine pass
reached 76%. Check this before publishing: for a character with several
`(Np)` co-appearances in the cast digest, does `relations` actually have an
entry for each?

**Before publishing, verify — do not just write and trust:**

```
python3 .claude/skills/style-read/verify_profile.py <folder> [--raw-only]
```

This is not optional and not the same check `finish.py` does. `finish.py`
checks shape: caps, enum fields, a relation pointing at a real id — it never
opens the notes, so it cannot tell a fabricated line from a real one.
`verify_profile.py` rereads every `speechEvidence` and every relation's
`evidence` against the volume's own notes and fails (exit code 1) on anything
that is not there character-for-character. Measured on a real six-volume
series: one relation carried a line spliced together from two different
characters' dialogue on two different pages, invented rather than copied, and
it passed `finish.py` cleanly — `finish.py` has no way to catch it.
**Run `verify_profile.py` after writing or updating profile.json, every
time, before telling anyone a volume is done.** If it fails, open the
character's own notes (`digest.py --cast`, then grep the quote) and either
find the real line and fix the citation, or drop the claim — never reword the
citation to match a paraphrase instead of fixing it.

Once `verify_profile.py` passes clean, publish:

```
python3 .claude/skills/style-read/finish.py <folder> [--raw-only]
```

It checks everything and writes the profile into the project's
`style_profiles/` library and, with the character tree and face crops, into
every folder read to the end (for a series, every finished volume, plus
`<series>/character_dictionary.json`). Volumes still being read are left for a
later run. If it lists problems, fix `profile.json` (or `cast.json`) and run it
again.

**Then close out faces before moving on — in two passes, not one.**

*Pass 1 — fill the gaps.* `finish.py` ends by printing how many published
characters are still short a usable face, and writes `face_gaps.json` — for
each one, 8 candidate pages already known (from `pages`, never a re-read) to
hold that character, skipping their very first appearance since that is when
a reader is least sure who anyone is. **If `face_gaps.json` is non-empty,
dispatch one subagent (`manga-page-reader` type) covering everyone in it**,
give it each character's `looks` and candidate pages, ask it to pick the
clearest CONFIRMED shot from those pages only and say so plainly when none
qualify rather than force one, fold the result in with `progress.py` (it runs
every box through the same check a reading batch's box gets), and run
`finish.py` once more.

*Pass 2 — actually look at what got kept, including the old ones.* This is
not optional, and skipping it is not a shortcut — it is how bad faces stay in
the app. A face box, whether from the original read or from the pass just
above, is a reader estimating pixel fractions from a whole page it can see,
describing a crop it never sees — nothing round-trips the guess back through
an eye. `usable_face` (progress.py) only catches what pixel statistics can
tell: a box sized for a crowd panel, a flat black shadow, a blank margin. It
cannot tell a face from a speech bubble — both are just dark strokes on white
paper to a histogram — and it cannot tell the right character from the wrong
one. Measured on a real six-volume series, after both checks above already
existed: roughly a third of published crops were actually a text bubble, a
title-page logo, or the wrong person, including some the face-hunt subagent
in pass 1 had just picked itself, confident in its own reasoning, from boxes
that turned out to land one panel off.

So: open a sample of `<volume>/character_scan/faces/<id>/*.png` yourself, or
dispatch one review subagent (Read tool only — no notes, no dialogue, just
"is this really this character's face, yes or no" for each file) to go
through all of them. For anything reported wrong:

```
python3 .claude/skills/style-read/review_faces.py <folder> --bad id/index,id/index,... [--raw-only]
```

`id/index` is read off the filename, e.g. `sheik-seijin/0` for
`faces/sheik-seijin/0.png`. This looks the crop up in `face_origins.json`
(finish.py writes it fresh every run) to remove the *exact* cast.json entry
that produced it — never guess at this by hand and edit cast.json directly;
that is exactly how a hand fix once removed the wrong one of two bad boxes and
left the actual bad one standing. Removed entries do not count against the
character's cap, so pass 1 gets another shot at them next time it runs.
Re-run `finish.py` after any removal to regenerate the crops without them.

A character who still comes up empty after a couple of rounds is left open,
not blocked on — do not re-hunt the same candidate pages hoping for a
different answer, and do not lower the bar to force a bad box through.

Then go on with the next batch.

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
