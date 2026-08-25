#!/bin/bash
# Supervisor: restart final training on GPU faults/crashes until "done".
# The Navi GPU hits an amdgpu gfxhub page fault roughly once per ~8k steps
# under sustained SDPA load (kernel-confirmed). Checkpoints every 1000 steps
# make restarts cheap (<=1000 steps lost); auto-resume is built into the
# trainer. Exits when train.py reports completion.
cd /lab/moonshine-italian
export TOKENIZERS_PARALLELISM=false
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== supervisor: attempt $attempt $(date -Is) ===" >> /tmp/opencode/final_train3.log
  if uv run --no-sync python train.py --profile final --hardware rocm12g \
      >> /tmp/opencode/final_train3.log 2>&1; then
    if grep -aq "done at step" <(tail -50 /tmp/opencode/final_train3.log); then
      echo "=== supervisor: TRAINING COMPLETE $(date -Is) ===" >> /tmp/opencode/final_train3.log
      exit 0
    fi
  fi
  echo "=== supervisor: crash detected, restarting in 30s ===" >> /tmp/opencode/final_train3.log
  pkill -9 -f "train\.py --profile final" 2>/dev/null
  sleep 30
done
