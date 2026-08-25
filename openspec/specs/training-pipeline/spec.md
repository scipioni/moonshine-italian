# training-pipeline Specification

## Purpose

Covers environment setup, Italian dataset acquisition and preparation, and fine-tuning of the English streaming-small checkpoint into an Italian streaming model, with distinct smoke and final training profiles.

## Requirements

### Requirement: Environment reproducibility via uv and native packages
The pipeline SHALL run inside a `.venv` managed by `uv`, with GPU-accelerated PyTorch and ONNX Runtime provided by native Arch Linux packages (`python-pytorch-rocm` or `python-pytorch-cuda`, `python-onnxruntime-rocm` or `python-onnxruntime-cuda`), and all remaining Python dependencies managed by `uv`. A single environment-check command SHALL verify that the virtualenv sees the native GPU packages and report the detected accelerator kind (CUDA or ROCm) and VRAM.

#### Scenario: Environment check on a ROCm machine
- **WHEN** the environment check runs on a machine with `python-pytorch-rocm` installed and an AMD GPU present
- **THEN** it reports ROCm as the accelerator, a non-zero VRAM figure, and exits successfully

#### Scenario: Missing native GPU package
- **WHEN** the environment check runs and the virtualenv cannot import a GPU-enabled PyTorch build
- **THEN** it fails with a message naming the pacman package(s) to install for the detected hardware

### Requirement: Configuration from config.yaml and .env
All pipeline behavior SHALL be driven by a single committed `config.yaml` containing the smoke and final training profiles, dataset selections, training hyperparameters, and hardware profiles. Private values (`HF_TOKEN` and similar) SHALL come from a git-ignored `.env` file whose keys are documented in a committed `.env.example`. The pipeline SHALL NOT embed secrets in `config.yaml`, logs, or artifacts.

#### Scenario: Missing .env when a gated dataset is enabled
- **WHEN** Common Voice Italian is enabled in `config.yaml` and `.env` contains no `HF_TOKEN`
- **THEN** the affected dataset download fails with a message pointing to `.env.example`, before any training starts

### Requirement: Dataset download and preparation
The pipeline SHALL download and prepare Multilingual LibriSpeech Italian (primary training), Mozilla Common Voice Italian (optional mix-in, gated on `HF_TOKEN`), and FLEURS Italian (smoke training + held-out evaluation). Preparation SHALL resample all audio to 16 kHz mono, segment long recordings into utterance-length chunks bounded by configured min/max durations, normalize Italian transcripts (accented characters preserved, casing and number expansion rules applied), and emit a manifest with per-split file lists and checksums.

#### Scenario: Segmentation bounds
- **WHEN** preparation runs over MLS Italian with min-duration 1.0 s and max-duration 10.0 s
- **THEN** every emitted training chunk's duration falls within those bounds and each chunk carries its aligned normalized transcript

#### Scenario: Reproducible preparation
- **WHEN** preparation runs twice with the same `config.yaml` inputs
- **THEN** the resulting manifests list identical files with identical checksums

### Requirement: Streaming checkpoint fine-tuning
The pipeline SHALL load `moonshine-ai/moonshine-streaming-small` safetensors, fine-tune on prepared Italian audio-transcript pairs, and save resumable checkpoints plus TensorBoard logs under a profile-specific output directory. The smoke profile SHALL complete the identical code path as the final profile, differing only in dataset slice size and step counts, so that a successful smoke run validates the full training mechanics.

#### Scenario: Smoke training run
- **WHEN** training runs with the smoke profile on the FLEURS Italian slice
- **THEN** it completes without error, saves at least one checkpoint that can be reloaded, and records loss curves viewable in TensorBoard

#### Scenario: Checkpoint resume
- **WHEN** training is interrupted and restarted from an existing checkpoint
- **THEN** it resumes from the recorded step without replaying already-completed optimizer steps

### Requirement: Gradient-path spike gate
Before any long training run, the pipeline SHALL provide a spike verification that the streaming-small checkpoint accepts forward and backward passes in the training framework, and that its tokenizer can encode and decode Italian evaluation text (including à, è, é, ì, ò, ù and apostrophes) without unrecoverable loss. The spike result SHALL be recorded, and if either check fails, the documented fallback ladder (non-streaming `moonshine-base` bring-up, then LoRA adapter path) SHALL be selected explicitly in `config.yaml` before training proceeds.

#### Scenario: Tokenizer round-trip
- **WHEN** the spike encodes and decodes a sample Italian sentence containing accented characters
- **THEN** the decoded text matches the input after normalization, otherwise the spike records a failure

#### Scenario: Spike failure triggers fallback selection
- **WHEN** the spike fails for the streaming checkpoint
- **THEN** the pipeline refuses to start a final training run until a fallback base is selected in `config.yaml`
