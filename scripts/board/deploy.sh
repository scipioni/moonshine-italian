#!/bin/bash
# Deploy the validated .ort release bundle to the Arduino UNO Q (Debian side).
#
# Usage:
#   BOARD_HOST=root@192.168.x.x ./scripts/board/deploy.sh            (env)
#   ./scripts/board/deploy.sh root@192.168.x.x                       (arg)
#   ./scripts/board/deploy.sh --verify-only /path/on/host            (on-board)
#
# Transfer is checksum-gated: the on-board verify step recomputes sha256 for
# every file in manifest.json and ABORTS the deployment on any mismatch.

set -euo pipefail

DEPLOY_PATH="${BOARD_DEPLOY_PATH:-/opt/moonshine-it}"
RELEASE_DIR="${RELEASE_DIR:-artifacts/release/checkpoint-best}"

# --- on-board verification mode (run on the board itself) ---
if [ "${1:-}" = "--verify-only" ]; then
    dir="${2:?usage: deploy.sh --verify-only <release-dir-on-board>}"
    echo "board: verifying checksums in ${dir}"
    python3 - "$dir" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

d = Path(sys.argv[1])
manifest = json.loads((d / "manifest.json").read_text())
failed = False
for name, rec in manifest["files"].items():
    f = d / name
    if not f.exists():
        print(f"ABORT: {name} missing on board"); failed = True; continue
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h != rec["sha256"]:
        print(f"ABORT: checksum mismatch for {name}\n"
              f"  manifest: {rec['sha256']}\n  actual:   {h}")
        failed = True
    else:
        print(f"ok: {name}")
sys.exit(1 if failed else 0)
PYEOF
    exit $?
fi

# --- host-side transfer ---
HOST="${1:-${BOARD_HOST:-}}"
if [ -z "$HOST" ]; then
    echo "ERROR: no board host. Set BOARD_HOST or pass user@host as argument." >&2
    exit 1
fi

if [ ! -f "${RELEASE_DIR}/manifest.json" ]; then
    echo "ERROR: ${RELEASE_DIR}/manifest.json missing — run: task validate" >&2
    exit 1
fi

echo "deploy: ${RELEASE_DIR} -> ${HOST}:${DEPLOY_PATH}"
ssh "$HOST" "mkdir -p ${DEPLOY_PATH}"
# manifest first so verify can run over an existing bundle atomically
scp -q "${RELEASE_DIR}/manifest.json" "${HOST}:${DEPLOY_PATH}/"
scp -q "${RELEASE_DIR}"/*.ort "${RELEASE_DIR}"/tokenizer*.json \
       "${RELEASE_DIR}"/special_tokens_map.json "${RELEASE_DIR}"/preprocessor_config.json \
       "${RELEASE_DIR}"/processor_config.json "${RELEASE_DIR}"/generation_config.json \
       "${RELEASE_DIR}"/config.json "${HOST}:${DEPLOY_PATH}/" 2>/dev/null || \
scp -q "${RELEASE_DIR}"/* "${HOST}:${DEPLOY_PATH}/"

echo "deploy: verifying checksums on board"
ssh "$HOST" "RELEASE_DIR='${DEPLOY_PATH}' bash -s" <<'REMOTE'
python3 - "$RELEASE_DIR" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

d = Path(sys.argv[1])
manifest = json.loads((d / "manifest.json").read_text())
failed = False
for name, rec in manifest["files"].items():
    f = d / name
    if not f.exists():
        print(f"ABORT: {name} missing on board"); failed = True; continue
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h != rec["sha256"]:
        print(f"ABORT: checksum mismatch for {name}\n"
              f"  manifest: {rec['sha256']}\n  actual:   {h}")
        failed = True
    else:
        print(f"ok: {name}")
sys.exit(1 if failed else 0)
PYEOF
REMOTE
status=$?
if [ $status -ne 0 ]; then
    echo "deploy: FAILED — checksum verification aborted; bundle NOT deployed" >&2
    exit $status
fi
echo "deploy: OK — ${DEPLOY_PATH} verified on ${HOST}"
