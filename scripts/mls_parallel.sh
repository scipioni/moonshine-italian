#!/bin/bash
# Parallel MLS preparation: 16 shard workers per split, then merge.
set -e
cd ~/moonshine-italian
export TOKENIZERS_PARALLELISM=false
N=16
for split in train validation test; do
  echo "=== shards: $split (x$N) ==="
  pids=""
  for k in $(seq 0 $((N - 1))); do
    uv run --no-sync python -m moonshine_it.prepare --dataset mls \
      --shard "$k" --num-shards "$N" &
    pids="$pids $!"
  done
  fail=0
  for p in $pids; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ]
  echo "=== merge: $split ==="
  uv run --no-sync python -m moonshine_it.prepare --dataset mls --merge --num-shards "$N"
done
echo "=== MLS DONE ==="
echo "=== common_voice: unavailable (repos removed from HF) — MLS-only mix ==="
echo "=== ALL DONE ==="
