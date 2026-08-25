# Data Preparation

Datasets (configured in `config.yaml` under `datasets:`):

| dataset         | HF repo                              | use                    | auth     |
|-----------------|--------------------------------------|------------------------|----------|
| FLEURS it_it    | `google/fleurs`                      | smoke train + all eval | public   |
| MLS italian     | `facebook/multilingual_librispeech`  | final training         | public   |
| Common Voice it | `mozilla-foundation/common_voice_21_0` | optional mix-in      | `HF_TOKEN` |

## Download

```bash
task download-model                      # base checkpoint (checksummed, idempotent)
task download-data DATASET=fleurs        # smoke + eval set
task download-data DATASET=mls           # final training corpus
task download-data DATASET=common_voice  # needs HF_TOKEN in .env
```

Missing-token failures point at `.env.example` before any download starts.

## Prepare

16 kHz mono resample, VAD-aware segmentation into 1–10 s chunks with
per-chunk aligned transcripts, Italian text normalization (accents
preserved, lowercase convention, number expansion rules in
`src/moonshine_it/normalize_it.py`):

```bash
task prepare DATASET=fleurs -- --force   # re-segment even if manifests exist
task prepare DATASET=mls
```

Splits are skipped when their manifest already exists (use `--force` to
re-prepare). Manifest stats land next to each split as `<split>_stats.json`.

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
