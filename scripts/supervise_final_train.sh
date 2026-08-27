#!/bin/bash
# Supervisor: restart final training on GPU faults/crashes until "done".
# The Navi GPU hits an amdgpu gfxhub page fault roughly once per ~8k steps
# under sustained SDPA load (kernel-confirmed). Checkpoints make restarts
# cheap; auto-resume is built into the trainer. Exits when train.py reports
# completion.
#
# Each attempt varies MOONSHINE_SHUFFLE_SEED: resuming with an unchanged seed
# deterministically re-walks the loader into the same faulting batch (observed
# to recur even after one reseed, at a different but nearby step). See the
# shuffle_seed comment in config.yaml and train_loop.py's loader construction.
set -u
cd /lab/moonshine-italian || exit 1
export TOKENIZERS_PARALLELISM=false

LOG=results/logs/final-train.log
BASE_SEED=$(grep -E '^  shuffle_seed:' config.yaml | awk '{print $2}')
if ! [[ "$BASE_SEED" =~ ^[0-9]+$ ]]; then
  echo "supervisor: could not read training.shuffle_seed from config.yaml" >&2
  exit 1
fi
mkdir -p results/logs

attempt=0
while true; do
  attempt=$((attempt + 1))
  seed=$((BASE_SEED + attempt - 1))
  echo "=== supervisor: attempt $attempt seed=$seed $(date -Is) ===" >> "$LOG"
  if MOONSHINE_SHUFFLE_SEED=$seed uv run --no-sync python train.py \
      --profile final --hardware rocm12g >> "$LOG" 2>&1; then
    if tail -50 "$LOG" | grep -aq "done at step"; then
      echo "=== supervisor: TRAINING COMPLETE $(date -Is) ===" >> "$LOG"
      exit 0
    fi
  fi
  echo "=== supervisor: crash detected, restarting in 30s ===" >> "$LOG"
  pkill -9 -f "train\.py --profile final" 2>/dev/null
  sleep 30
done
