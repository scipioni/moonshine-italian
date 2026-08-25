# moonshine-italian

Fine-tune `moonshine-ai/moonshine-streaming-small` (123M, English) into
**`moonshine-streaming-small-it`** — an Italian streaming speech-to-text
model — and ship it as INT8 `.ort` artifacts for an Arduino UNO Q 4 GB
voice agent. The full process (data → training → dual-mode eval → ONNX/INT8
export → validated release → board deployment) is task-driven and proven
end-to-end on a small smoke slice before any multi-day run.

## Quickstart (smoke = the whole pipeline on a small slice)

```bash
# prerequisites: Arch + GPU packages + uv (see docs/environment.md)
uv venv --system-site-packages && uv sync
cp .env.example .env          # set HF_TOKEN only if using Common Voice

task smoke PROFILE=rocm12g    # download → spike → train → eval → export → ort → validate
```

The chain prints a per-phase summary, exits non-zero at the first failing
phase, and records the pass in `results/smoke/record.json` — which
`task final-train` requires before the multi-day run will start.

## Task map

| target          | produces                                                        |
|-----------------|-----------------------------------------------------------------|
| `env-check`     | GPU/ORT sanity + profile validation                             |
| `download-model`, `download-data` | base checkpoint, FLEURS/MLS/CV-it (checksummed)   |
| `prepare`, `slice-smoke` | 16 kHz VAD-chunked manifests + deterministic smoke slice |
| `spike`         | gradient/tokenizer/baseline spikes + fallback-latch verdict     |
| `train` / `final-train` | fine-tuning (smoke / final profiles; latch-enforced)     |
| `eval`          | full + streaming WER/CER with gates                             |
| `export`        | gate-checked ONNX graphs + parity verification                  |
| `ort`           | INT8 quantization + `.ort` serialization + size report          |
| `ort-eval`      | `.ort` streaming eval + post-quantization gate                  |
| `validate` / `release` | checksummed, smoke-loaded release bundle                |
| `board-deploy`  | checksum-gated scp to the UNO Q                                 |

Every target needs `PROFILE=rocm12g|strix|cuda` (unset/invalid fails with
the valid list) and fails fast pointing at the target that produces a
missing input.

## Layout

```
config.yaml            # all profiles, gates, budgets (single source of truth)
Taskfile.yml           # orchestration (this table)
train.py               # training entry (smoke/final share one code path)
src/moonshine_it/      # prep, normalization, training, eval, export, release
scripts/board/         # deploy.sh (checksum-gated), measure.py (budget report)
docs/                  | environment · data · training · evaluation · export · board · results
artifacts/release/     # validated .ort bundles (git-ignored)
```

## Board notes

The UNO Q Linux side (QRB2210, Debian) consumes `.ort` only; deployment is
checksum-gated and measured against the configured latency/RAM budget, with
a streaming-tiny fallback if small misses it — see `docs/board.md`.

## Status

Smoke pipeline: **PASS** (parity, gates, release bundle). Final training on
Strix Halo + board deployment pending — current numbers in
`docs/results.md`.
