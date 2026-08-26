# Environment Setup

This pipeline targets Arch Linux with native GPU packages and a `uv`-managed
virtualenv. All commands assume the repo root.

## 1. Native (pacman) packages

ROCm (AMD) — used for the 12 GB smoke PC and the Strix Halo final-training box:

```bash
sudo pacman -S python-pytorch-opt-rocm python-onnxruntime-rocm rocm-hip-sdk
```

CUDA (NVIDIA), alternative profile:

```bash
sudo pacman -S python-pytorch-opt-cuda python-onnxruntime-cuda cuda
```

Notes:

- The `rocm`/`cuda` package families conflict with each other and with stock
  `python-pytorch`/`python-onnxruntime` — install exactly one family.
- `task` (go-task) is the orchestrator: `sudo pacman -S go-task`.

## 2. Python environment (uv)

The venv is created with system-site-packages so the pacman GPU builds are
visible, while all pure-Python deps are locked by `uv`:

```bash
uv venv --system-site-packages
uv sync
```

Verify the GPU stack resolves:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Secrets

Copy `.env.example` to `.env` and fill in:

- `HF_TOKEN` (https://huggingface.co/settings/tokens) — only needed if you
  hit a genuinely gated HF dataset; FLEURS and MLS are public.
- `CV_ARCHIVE_PATH` — path to a locally-downloaded Common Voice Italian
  archive (`mozilla-foundation/common_voice_*` is 404 on HF). Download it
  from
  [mozilladatacollective.com](https://mozilladatacollective.com/datasets/cmqini14100vmnq07309ocknr),
  save it anywhere (e.g. `/tmp`), and point `CV_ARCHIVE_PATH` at it. See
  `docs/data.md` for the full download/extract flow.

## 4. Hardware profiles

Everything profile-dependent is selected with `PROFILE=`:

| profile  | hardware                | batch | precision | workers |
|----------|-------------------------|-------|-----------|---------|
| `rocm12g`| AMD ROCm 12 GB (smoke)  | 8     | bf16      | 4       |
| `strix`  | Strix Halo (final)      | 32    | bf16      | 8       |
| `cuda`   | NVIDIA CUDA             | 16    | bf16      | 8       |

An unset or invalid profile fails with the list of valid choices — there are
no silent defaults.

## 5. Sanity check

```bash
task env-check PROFILE=rocm12g
```

Reports accelerator kind, VRAM, GPU torch + ORT imports, and validates the
profile. Fails with pacman guidance when the GPU packages are absent.
