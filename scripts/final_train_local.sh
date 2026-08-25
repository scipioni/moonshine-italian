#!/bin/bash
# Final training on the local 12 GB PC (discrete GPU).
# Strix Halo (max) was benched: its iGPU shows bimodal per-batch performance
# (1s vs 130s for identical work, backend-independent) AND torch-2.12
# mem-efficient SDPA produces inf LayerNorm grads there. The local box +
# math SDPA is uniform 0.5s/batch, zero non-finite grads (measured).
set -e
cd /lab/moonshine-italian
export TOKENIZERS_PARALLELISM=false
exec uv run --no-sync python train.py --profile final --hardware rocm12g
