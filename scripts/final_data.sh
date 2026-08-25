#!/bin/bash
set -e
cd ~/moonshine-italian
export TOKENIZERS_PARALLELISM=false
echo "=== download fleurs ==="
uv run --no-sync python -m moonshine_it.download data fleurs
echo "=== download mls ==="
uv run --no-sync python -m moonshine_it.download data mls
echo "=== download common_voice ==="
uv run --no-sync python -m moonshine_it.download data common_voice
echo "=== prepare fleurs ==="
uv run --no-sync python -m moonshine_it.prepare --dataset fleurs
echo "=== prepare mls ==="
uv run --no-sync python -m moonshine_it.prepare --dataset mls
echo "=== prepare common_voice ==="
uv run --no-sync python -m moonshine_it.prepare --dataset common_voice
echo "=== DONE ==="
