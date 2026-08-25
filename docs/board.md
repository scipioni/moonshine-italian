# Board Deployment (Arduino UNO Q 4 GB)

Target: Qualcomm Dragonwing QRB2210, Debian (arm64) on the Linux side; the
STM32U585 side is out of scope for this repo. The board runs the Moonshine
Voice streaming runtime over `.ort` models.

## Budget (config.yaml `board.budget`)

| metric                        | budget       |
|-------------------------------|--------------|
| re-decode latency             | ≤ 1500 ms    |
| pause → final-text latency    | ≤ 3000 ms    |
| peak RSS (model loaded)       | ≤ 2048 MB    |

## 1. Transfer (checksum-gated)

```bash
BOARD_HOST=root@<board-ip> task board-deploy PROFILE=rocm12g
# or directly:
BOARD_HOST=root@<board-ip> ./scripts/board/deploy.sh
```

The script scps the validated bundle to `/opt/moonshine-it` (override with
`BOARD_DEPLOY_PATH`), then recomputes sha256 for every file in
`manifest.json` **on the board**; any mismatch aborts the deployment naming
the offending file. On-board re-verification can be run standalone:

```bash
python3 scripts/board/deploy.sh --verify-only /opt/moonshine-it
```

## 2. Runtime bring-up (recorded)

**Path used: Python wheels.** `onnxruntime` 1.29.0 ships aarch64 manylinux
wheels; the board runs the `.ort` graphs in a venv (`~/ort-venv`,
CPUExecutionProvider) with `numpy` + `tokenizers` — the standalone runtime
module `src/moonshine_it/ort_runtime.py` is copied next to the board
scripts and needs no repo checkout and no libsndfile (stdlib `wave` reader
for PCM_16 clips). No C++ build was necessary.

The `.ort` graphs load in ~3.3 s; a prepared clip
(`artifacts/test/it_test_clip.wav`) streams line events while audio is
still arriving, and the board's final transcript matches the host eval
harness's streaming-mode transcript for the same clip (deterministic
same-code-path decode):

```bash
# on the board
~/ort-venv/bin/python /opt/moonshine-it/runtime_smoke.py \
    --release-dir /opt/moonshine-it --audio /opt/moonshine-it/it_test_clip.wav
```

## 3. Measurement

```bash
# on the board
python3 scripts/board/measure.py --release-dir /opt/moonshine-it \
    --audio it_test_clip.wav
```

Measures re-decode latency (speculative on/off), pause→text latency, peak
RSS; writes `budget_report.json` with per-metric pass/fail against the
configured budget.

## 4. Fallback

If small misses the re-decode budget, the report names the documented
fallback: export a `streaming-tiny-it` fine-tune through the same pipeline
(task `8.4` in the change plan). The board budget verdict lands in
`docs/results.md`.

## Measured (smoke checkpoint, 2026-08-23)

All budget metrics **pass** — see `docs/results.md` § Board measurements
and `results/board/budget_report.json`. The tiny fallback is therefore not
exercised.
