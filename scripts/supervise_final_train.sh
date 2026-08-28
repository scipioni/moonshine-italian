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
  MOONSHINE_SHUFFLE_SEED=$seed uv run --no-sync python train.py \
      --profile final --hardware rocm12g >> "$LOG" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]] && tail -50 "$LOG" | grep -aq "done at step"; then
    echo "=== supervisor: TRAINING COMPLETE $(date -Is) ===" >> "$LOG"
    exit 0
  fi
  # A latch or gate refusing to continue is a verdict, not a fault. Restarting
  # it just re-runs the same failing evals: on 2026-08-28 the regression latch
  # fired at step 14,000 and this loop restarted it ~127 times over eight
  # hours. Exit and let a human read the verdict. Keep in sync with
  # POLICY_STOP_EXIT_CODE in src/moonshine_it/train_loop.py.
  if [[ $rc -eq 3 ]]; then
    echo "=== supervisor: POLICY STOP (exit 3) — not a crash, not restarting; \
see the message above $(date -Is) ===" >> "$LOG"
    exit 3
  fi
  echo "=== supervisor: crash detected, restarting in 30s ===" >> "$LOG"
  pkill -9 -f "train\.py --profile final" 2>/dev/null
  sleep 30
done
