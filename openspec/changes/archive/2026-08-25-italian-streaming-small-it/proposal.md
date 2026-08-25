# Proposal: italian-streaming-small-it

## Why

Moonshine Voice ships streaming speech-to-text models in English only; the catalog has no Italian model of any kind (verified against the available-models table, 2026-08). We are building an Italian voice agent on an Arduino UNO Q (4 GB), whose latency budget requires a streaming model — so we must create `moonshine-streaming-small-it` ourselves: fine-tune `moonshine-ai/moonshine-streaming-small` (123M params, English) on Italian corpora, export to INT8 `.ort`, and validate the full pipeline before committing to a long final training run. No public recipe exists for fine-tuning a Moonshine *streaming* checkpoint — the only community toolkit (`pierre-cheneau/finetune-moonshine-asr`) handles non-streaming architectures — so the process must be proven end-to-end on a small dataset first (smoke profile), then scaled.

## What Changes

- New fine-tuning pipeline (PyTorch, `uv`-managed `.venv` on Arch Linux) that loads `moonshine-ai/moonshine-streaming-small` safetensors and trains on Italian audio (MLS Italian primary, Common Voice Italian optional mix, FLEURS Italian for smoke + eval).
- New streaming evaluation harness: full-utterance WER/CER plus chunked streaming simulation (32–100 ms hops) with speculative decoding enabled, using jiwer and the FLEURS-it test split.
- New export/quantization chain: PyTorch checkpoint → ONNX (encoder + KV-cached decoder) → INT8 → `.ort` FlatBuffers via `scripts/convert-models-to-ort.py` equivalents from upstream moonshine.
- New Taskfile orchestration: `download / prepare / train / eval / export / ort / smoke / e2e` targets with hardware profiles (`rocm12g`, `strix`, `cuda`), each phase gated on the previous outputs.
- Configuration via a single committed `config.yaml` (all profiles, smoke + final) plus `.env` for private values (`HF_TOKEN`), with `.env.example` committed.
- Complete smoke procedure that exercises every phase (download → train on a small FLEURS-it slice → eval → export → `.ort` → on-device-ready artifact) in one command, to validate the process before final training.
- Documentation: detailed `docs/` guides (environment, data, training, eval, export, board deployment) and a concise top-level README.
- Explicitly deferred: upstream PR to `moonshine-ai/moonshine` (catalog entry, `ModelArch` enum, bindings) and any STM32U585-side work. This repo produces the model and artifacts; integration comes later.

## Capabilities

### New Capabilities
- `training-pipeline`: Environment setup (Arch + uv + ROCm/CUDA profiles), dataset download/preparation (MLS-it, CV-it, FLEURS-it), text normalization for Italian, and fine-tuning of the streaming-small checkpoint with smoke and final profiles.
- `evaluation`: WER/CER evaluation on full utterances and in chunked streaming simulation, with pass/fail gates relative to baseline, tracked per phase.
- `export-pipeline`: Conversion of fine-tuned checkpoints to ONNX with KV-cache I/O, INT8 quantization, `.ort` serialization, and artifact validation (checksums, sizes, runtime smoke load).
- `task-orchestration`: Taskfile targets and hardware profiles that sequence every phase, with gates and a single-command smoke run.
- `board-deployment`: Deployment and validation of `.ort` artifacts on the Arduino UNO Q 4 GB Linux side (QRB2210), including latency and RAM measurement criteria.

### Modified Capabilities

(none — no existing specs in this repo)

## Impact

- **Code**: new repo content — `train.py`-style pipeline code, `scripts/`, `configs/` (via `config.yaml`), `Taskfile.yml`, `docs/`, `.env.example`; `.venv` managed by `uv`.
- **Dependencies**: Arch native packages (`python-pytorch-rocm`/`-cuda`, `python-onnxruntime-rocm`/`-cuda`, `rocm-hip-sdk` or CUDA); uv-managed pure-Python deps (transformers, datasets, jiwer, schedulefree, etc.). Hugging Face: `moonshine-ai/moonshine-streaming-small` (weights), `facebook/multilingual_librispeech`, `google/fleurs`, Mozilla Common Voice (auth via `HF_TOKEN`).
- **Hardware**: AMD ROCm 12 GB PC (smoke/testing), Strix Halo ROCm (final training), NVIDIA CUDA (supported profile). Arduino UNO Q 4 GB as deployment target.
- **Risks**: (1) No public recipe for fine-tuning streaming checkpoints — Spike 1 (gradients through the streaming checkpoint) gates the design; fallback ladder: non-streaming `moonshine-base` bring-up → `moonshine-voice[lora]` `fit_adapter` path. (2) Tokenizer coverage of Italian accents/subwords unknown until the checkpoint is inspected (Spike 2). (3) QRB2210 CPU may be slower than RPi5 (~527 ms/re-decode for small-streaming); speculative decoding masks this, but `streaming-tiny-it` export is the documented fallback.
