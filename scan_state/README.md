# Scan state

What a `/style-read` run leaves behind, kept here so a scan started on one
machine can be continued — or just looked at in the app — on another.

One folder per series, mirroring the layout inside the series folder itself:

```
scan_state/<series>/
  character_dictionary.json            the series' cast, what the app reads
  style_scan/claude_raw/               the run's state
    notes/<page id>.md                 one note per page read
    cast.json                          the cast as the readers left it
    known.md                           what the next batch is told
    profile.json                       the profile, rewritten each volume
    profile_state.json                 which volumes are already in it
  <volume folder>/
    character_scan/                    the character tree and face crops
    style_scan/                        the published profile for that volume
```

## Restoring on another machine

Copy the contents of `scan_state/<series>/` over the series folder, keeping the
volume folder names exactly as they are:

```
rsync -a scan_state/kinnikuman/ "/path/to/Kinnikuman/"
```

Then the app finds the character tree when you open a volume, and `/style-read`
carries on from the first page without a note.

## What is not here

`style_scan/claude_raw/pages/` — the reading copies of the pages themselves.
They are the manga, they run to hundreds of megabytes, and `prepare.py` writes
them again from the series folder on the next run. `pages.json` is left out for
the same reason: it holds absolute paths from the machine that made it, and is
rebuilt on every run.
