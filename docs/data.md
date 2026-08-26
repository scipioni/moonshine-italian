# Data Preparation

Datasets (configured in `config.yaml` under `datasets:`), all used in the
`final` training mix (`training.profiles.final.datasets`):

| dataset         | source                                       | use                    | auth          |
|-----------------|-----------------------------------------------|------------------------|---------------|
| FLEURS it_it    | `google/fleurs` (HF)                          | smoke train + all eval; final mix (in-domain, modern news) | public |
| MLS italian     | `facebook/multilingual_librispeech` (HF)      | final training (audiobook prose) | public |
| Common Voice it | local CC0 archive, see below                  | final training (crowd-read, 7k+ speakers) | `CV_ARCHIVE_PATH` |

## Download

```bash
task download-model                      # base checkpoint (checksummed, idempotent)
task download-data DATASET=fleurs        # smoke + eval set
task download-data DATASET=mls           # final training corpus
task download-data DATASET=common_voice  # extracts from CV_ARCHIVE_PATH, see below
```

Missing-token / missing-archive failures point at `.env.example` before any
download starts.

### Common Voice: local archive

`mozilla-foundation/common_voice_*` on Hugging Face returns 404 (the repo
was removed there). This pipeline instead uses a local, CC0-licensed archive:

1. Download the Italian corpus from
   [mozilladatacollective.com](https://mozilladatacollective.com/datasets/cmqini14100vmnq07309ocknr)
   — a `cv-corpus-<version>-<date>-it.tar.gz`, ~10 GB.
2. Save it anywhere on disk (e.g. `/tmp/cv-corpus-it.tar.gz` — it does **not**
   need to live inside the repo).
3. Point `.env` at it:
   ```bash
   # .env
   CV_ARCHIVE_PATH=/tmp/cv-corpus-it.tar.gz
   ```
4. `task download-data DATASET=common_voice` extracts it into
   `data/raw/common_voice_it/`.

Extraction is a single sequential pass over the archive: the per-language
metadata TSVs (`train.tsv`/`dev.tsv`/`test.tsv`) sort before `clips/` inside
these archives, so by the time the first audio clip is reached the full
train+dev+test clip-name set is already known, and every later member is
filtered against it in the same pass — no full unpack, no second read of the
compressed stream. Only the ~204k clips actually referenced by those three
splits are written (~72% of the ~284k clips in the full archive; `other.tsv`/
`invalidated.tsv` clips are skipped). Expect a multi-GB, several-minute
operation; it's idempotent (a `.extracted.json` marker skips re-extraction —
pass `--force` via `task download-data DATASET=common_voice -- --force` to
redo it, e.g. after a newer corpus release).

If `mozilladatacollective.com` changes the archive's internal directory name
(currently `cv-corpus-26.0-2026-06-12/it/`), update
`datasets.common_voice.local.inner_prefix` in `config.yaml` to match — run
`tar -tzf <archive> | head` to check.

The original tarball can be deleted after extraction; only
`data/raw/common_voice_it/` (gitignored, like the rest of `data/`) is needed
downstream.

## Prepare

16 kHz mono resample, VAD-aware segmentation into 1–10 s chunks with
per-chunk aligned transcripts, Italian text normalization (accents
preserved, lowercase convention, number expansion rules in
`src/moonshine_it/normalize_it.py`):

```bash
task prepare DATASET=fleurs -- --force   # re-segment even if manifests exist
task prepare DATASET=mls
task prepare DATASET=common_voice        # ~200k clips; several hours on CPU VAD
```

Splits are skipped when their manifest already exists (use `--force` to
re-prepare). Manifest stats land next to each split as `<split>_stats.json`.
Common Voice's `dev.tsv`/`test.tsv` map to the pipeline's `validation`/`test`
split names (`datasets.common_voice.local.splits` in `config.yaml`); its
`train`/`dev`/`test` are Common Voice's own speaker-balanced, validated
splits, so no extra vote filtering is applied on top.

## Smoke slice

Deterministic subset of prepared FLEURS (fixed seed, counts from
`config.yaml` `smoke:`):

```bash
task slice-smoke
```

Two preparations produce byte-identical manifests (checksums recorded in
`results/smoke/slice.json`).

## Verification

- Unit tests: `uv run python -m pytest tests/test_normalize_it.py`
- Manifest durations within 1–10 s bounds (checked during preparation)
- Final-run usable-hours stats: see `docs/results.md`
