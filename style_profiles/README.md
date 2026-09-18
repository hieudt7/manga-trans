# Style profiles

Translation style profiles, one JSON file per reference volume. The app lists
them on the Style Scanner page; choosing one there (**Use for translation**)
puts it at the top of every translation request.

A profile is written here when a style scan finishes:

- **Read with Claude** — every page is shown to the Claude API, which notes who
  says what to whom, then writes the profile. Works with the original and the
  published translation (`raw/` + `trans/`), or with the translation alone.
- **Local OCR** — vietocr on this machine; needs `raw/` and `trans/`.

Saving edits on the Style Scanner page updates the file here too.

Each file has the shape of `StyleScanResult` in
`koharu-ml/src/bilingual/corpus.rs`; the part used as the prompt is `profile`
(`approach`, `voice`, `address`, `soundEffects`, `glossary`).

The app uses this folder when it runs from this source tree. Set
`KOHARU_STYLE_PROFILES_DIR` to use another one; a build without the source tree
keeps profiles in the app's data directory.
