# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fine-tunes `moonshine-ai/moonshine-streaming-small` (123M, English) into an
Italian streaming speech-to-text model and ships it as INT8 `.ort`
artifacts for an Arduino UNO Q 4 GB voice agent. The pipeline is
task-driven (`Taskfile.yml`) and every phase is proven end-to-end on a small
"smoke" slice before any multi-day final run is allowed to start.

## Commands

All targets are run through `task` (go-task) and require
`PROFILE=rocm12g|strix|cuda` — unset/invalid fails fast with the valid
list, there are no silent defaults.

```bash
uv venv --system-site-packages && uv sync   # env setup (see docs/environment.md)
task env-check PROFILE=rocm12g              # GPU/ORT sanity check

task smoke PROFILE=rocm12g                  # whole pipeline on a small slice:
                                             # download -> spike -> train -> eval -> export -> ort -> validate
```

Individual phases (each has a precondition that names the target to run if
its input is missing — see `Taskfile.yml`):

```bash
task download-model
task download-data DATASET=fleurs|mls|common_voice
task prepare DATASET=fleurs [-- --force]
task slice-smoke
task spike PROFILE=rocm12g
task train PROFILE=rocm12g [TRAIN_PROFILE=smoke|final]
task eval PROFILE=rocm12g [TRAIN_PROFILE=smoke GATE=smoke]
task export PROFILE=rocm12g
task ort PROFILE=rocm12g
task ort-eval PROFILE=rocm12g
task validate PROFILE=rocm12g
task release PROFILE=rocm12g                # export -> ort -> ort-eval -> validate
task final-train PROFILE=strix              # requires results/smoke/record.json
task board-deploy PROFILE=rocm12g           # requires BOARD_HOST
```

Every task target ultimately calls `uv run --no-sync python -m moonshine_it.<module>`
(or `train.py` directly) — same entry points are also reachable via the
`moonshine-it` console script (`src/moonshine_it/cli.py`) for ad-hoc use
outside `task`.

Tests:

```bash
uv run python -m pytest                          # full suite (tests/)
uv run python -m pytest tests/test_normalize_it.py
uv run python -m pytest tests/test_gates.py -k spike_latch
```

## Architecture

**Single source of truth:** `config.yaml` holds every profile, gate
threshold, dataset spec, and budget. `src/moonshine_it/config.py` loads
and validates it strictly (`ConfigError`/`SystemExit` on missing/invalid
keys — this is intentional, not a bug to relax) and resolves a hardware
profile + training profile into one `ResolvedProfile`. Secrets
(`HF_TOKEN`) live only in `.env`, never in `config.yaml`.

**Orchestration vs. logic split:** `Taskfile.yml` encodes the phase graph
and preconditions (what must exist before a target runs); the actual work
lives in `src/moonshine_it/*.py` modules, each independently invocable via
`moonshine_it.cli` or `python -m moonshine_it.<module>`. When changing
what a phase produces or requires, update both the module and the
corresponding `preconditions:` block in `Taskfile.yml`.

**Enforced latches (not advisory — they raise `SystemExit`, see
`gates.py`):**
- *Spike latch*: no training run starts without a passing
  `results/spike/verdict.json`. A failed verdict requires an explicit
  fallback in `config.yaml` (`base_model.selected_base`:
  `non_streaming_base` or `lora`).
- *Smoke latch*: `task final-train` refuses to run without a recorded
  `results/smoke/record.json` (written only by a full, chained
  `task smoke` pass).
- *Gate latches*: `task export` refuses without a passing gate record in
  `results/gates/<name>.json`; release promotion (`task validate`)
  refuses without a passing `post_quant` gate.

**Gates** (`config.yaml` `evaluation.gates`, enforced by `gates.py`):
`smoke` (full-mode WER ≤ baseline × 1.10 — process validation, not model
quality), `final` (streaming WER ≤ 15% absolute), `post_quant` (.ort
streaming WER ≤ pre-quant checkpoint WER × 1.25). A failed gate exits
non-zero with measured-vs-allowed values and blocks all downstream
targets.

**Pipeline data flow:**
```
download (model, fleurs/mls/common_voice)
  -> prepare (16 kHz mono, Silero VAD segmentation, manifests, IT text normalization)
  -> slice-smoke (deterministic subset, fixed seed)
  -> spike (grad/tokenizer/baseline spikes -> verdict; latches training)
  -> train (train.py -> train_loop.py; one code path for smoke/final profiles,
     schedule-free AdamW, curriculum staging on `final`, chunked-audio augmentation)
  -> eval (evaluate_cli.py -> evaluate.py; dual-mode: full-utterance + chunked
     streaming simulation with speculative-decoding verification; jiwer WER/CER)
  -> export (PyTorch -> 4 ONNX graphs: encoder, adapter, cross_kv, decoder_kv;
     gate-checked, then parity.py verifies ONNX-vs-PyTorch logits + cached-path decode)
  -> quantize/ort (onnx-shrink-ray INT8: integer_weights for encoder,
     integer_activations for adapter/cross_kv/decoder_kv; per-channel scaling
     is load-bearing for accuracy; then convert to .ort FlatBuffers)
  -> ort-eval (streaming eval of the .ort bundle; post_quant gate)
  -> validate/release (sha256 manifest + ORT smoke-load per graph; nothing
     unvalidated reaches board-deploy)
  -> board-deploy (checksum-gated scp; re-verifies sha256 on-board before use)
```

**Runtime contract:** the on-device/runtime consumption path accepts only
`.ort` FlatBuffers, never `.onnx` — every downstream loader rejects `.onnx`
intermediates with an explicit error naming the expected `.ort` artifact
(`moonshine_it.release.require_ort_file`). `src/moonshine_it/ort_runtime.py`
is a standalone module: it's copied directly onto the board venv with no
repo checkout and no libsndfile dependency (uses stdlib `wave` for PCM_16),
so avoid adding repo-relative imports or non-stdlib deps to it beyond
numpy/onnxruntime/tokenizers.

**Board deployment target:** Arduino UNO Q 4 GB (Qualcomm QRB2210, Debian
arm64, Linux side only — the STM32U585 side is out of scope). Runs `.ort`
graphs via onnxruntime Python wheels, CPUExecutionProvider. Budget
(`config.yaml` `board.budget`): re-decode latency ≤ 1500 ms, pause→final
latency ≤ 3000 ms, peak RSS ≤ 2048 MB. If the small model misses budget,
the documented fallback is a `streaming-tiny-it` fine-tune through the
same pipeline — `scripts/board/measure.py` names this fallback explicitly
in its report rather than failing silently.

**Results are the record, not a summary of it** (see `docs/results.md`):
numbers in docs are copied verbatim from JSON artifacts under `results/`
and `artifacts/release/`. When updating docs after a real run, copy from
the JSON rather than re-deriving numbers.

## Key module map (`src/moonshine_it/`)

- `config.py` — config.yaml/`.env` loading + validation, profile resolution
- `gates.py` — latches (spike/smoke) and gate pass/fail records
- `download.py`, `prepare.py`, `slice_smoke.py`, `normalize_it.py` — data pipeline (VAD segmentation, IT text normalization: lowercase, number expansion, accents preserved)
- `spike.py` — pre-training gradient/tokenizer/baseline spikes + verdict
- `train_loop.py` (invoked via `train.py`) — shared smoke/final training loop
- `evaluate.py` / `evaluate_cli.py` — dual-mode (full + streaming) WER/CER
- `export.py` / `parity.py` / `model_io.py` — ONNX export + PyTorch-vs-ONNX parity checks
- `quantize.py` — INT8 quantization + `.ort` serialization
- `ort_eval.py` / `ort_runtime.py` — `.ort` bundle streaming eval / standalone board runtime
- `release.py` — checksum manifest + validated release bundle promotion
- `env_check.py` — GPU/ORT/profile sanity check
- `cli.py` — subcommand dispatch (`moonshine-it <command>`)

## Conventions

- Do not add a `Co-Authored-By` line to git commits.
- The Strix Halo box ("max", `PROFILE=strix`) is reachable via `ssh scipio@max`.
- Hardware profiles (`rocm12g`, `strix`, `cuda`) and training profiles
  (`smoke`, `final`) are validated against `config.py`'s
  `VALID_HW_PROFILES`/`VALID_TRAIN_PROFILES` — extend these plus
  `config.yaml` together, never just one.
- Tests isolate `gates.py`'s filesystem effects by monkeypatching
  `REPO_ROOT` to a `tmp_path` (see `tests/test_gates.py`) rather than
  touching the real `results/` tree.
- Torch and GPU-accelerated onnxruntime come from Arch pacman packages
  (`python-pytorch-opt-rocm`/`-cuda`, `python-onnxruntime-rocm`/`-cuda`)
  via `--system-site-packages`, never from PyPI — `pyproject.toml`
  explicitly overrides `torch`/`torchaudio` to an impossible marker so
  `uv sync` can't pull them in transitively.
