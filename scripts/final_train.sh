#!/bin/bash
# Task 10.2: final training with the strix profile on Strix Halo.
# Runs detached; multi-day. Auto-resumes on restart.
#
# Numerics note: the training loop disables the SDPA mem-efficient and flash
# backends (math fallback) — see train_loop.py for the measured rationale
# (inf LayerNorm gradients on torch-2.12 ROCm mem-efficient; AOTriton correct
# but pathologically slow on this iGPU).
set -e
cd ~/moonshine-italian
export TOKENIZERS_PARALLELISM=false
uv run --no-sync python train.py --profile final --hardware strix
echo "=== TRAINING DONE ==="
