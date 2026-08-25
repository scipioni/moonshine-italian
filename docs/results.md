# Results

Numbers below are copied from the JSON artifacts in `results/` and
`artifacts/release/` — they are the record, not a summary of it.

> **Stage: smoke** (60 training steps, FLEURS-it smoke slice; process
> validation, not model quality). Final-training numbers will be appended
> when the Strix Halo run completes.

## Baseline (un-tuned English checkpoint, FLEURS-it)

| mode      | n  | WER %  | CER %  |
|-----------|----|--------|--------|
| full      | 10 | 153.29 | 84.51  |
| streaming | 10 | 106.59 | 96.22  |

Source: `results/eval/baseline_full.json`, `baseline_streaming.json`.

## Smoke checkpoint (60 steps)

| mode      | n  | WER %  | CER % |
|-----------|----|--------|-------|
| full      | 40 | 122.18 | 62.64 |
| streaming | 40 | 102.89 | 83.50 |

Smoke gate: 122.18 ≤ 168.62 (baseline × 1.10) — **passed**
(`results/gates/smoke.json`).
Source: `results/eval/smoke_full.json`, `smoke_streaming.json`.

## INT8 `.ort` bundle (post-quantization, streaming mode)

| metric                          | value    |
|---------------------------------|----------|
| streaming WER (n=40)            | 103.99 % |
| post-quant delta vs pre-quant   | +1.10 points (102.89 → 103.99, ~1.1 % relative) |
| re-decode latency (host, ROCm ORT) | 92.4 ms mean |
| RTF (host)                      | 0.127    |

Post-quant gate: 103.99 ≤ 128.61 (pre-quant × 1.25) — **passed**
(`results/gates/post_quant.json`).
Source: `results/eval/smoke_int8_streaming.json`.

## ONNX export parity (fp32, before quantization)

| graph              | max abs diff | tolerance |
|--------------------|--------------|-----------|
| encoder            | 3.29e-04     | 2e-02     |
| adapter            | 4.93e-04     | 2e-02     |
| cross_kv           | 1.53e-05     | 2e-02     |
| decoder_kv         | 2.29e-05     | 2e-02     |
| decoder_kv (cached)| 2.67e-05     | 2e-02     |

Reimplementation vs HF logits: argmax matches on the reference prefix.
Source: `results/export/checkpoint-best/parity.json`.

## Quantization size report

| graph       | fp32 ONNX MB | int8 `.ort` MB |
|-------------|--------------|----------------|
| encoder     | 205.17       | 205.33 (weight-only int8 expanded at load) |
| adapter     | 11.43        | 2.87           |
| cross_kv    | 20.99        | 5.36           |
| decoder_kv  | 323.87       | 82.07          |
| **total**   | **561.46**   | **295.63** (1.9×) |

Bundle fits the 2048 MB RSS board budget. Note: the encoder `.ort` stores
weight-only quantization expanded back to fp32 size; splitting
frontend/encoder like upstream would halve it.
Source: `artifacts/release/checkpoint-best/size_report.json`.

## Board measurements (UNO Q 4 GB, smoke checkpoint)

Runtime path: Python wheels — `onnxruntime` 1.29.0 (aarch64),
CPUExecutionProvider, venv at `~/ort-venv`; bundle deployed to
`/opt/moonshine-it` with on-board checksum verification
(`scripts/board/deploy.sh`). Graphs load in ~3.3 s; board final transcript
matches the host streaming-eval transcript for the same clip.

| metric                              | measured | budget | verdict |
|-------------------------------------|----------|--------|---------|
| re-decode latency (speculative ON)  | 479.2 ms | ≤ 1500 ms | pass |
| re-decode latency (speculative OFF) | 449.4 ms | —      | —       |
| pause → final-text latency          | 345.6 ms | ≤ 3000 ms | pass |
| peak RSS (model loaded)             | ~513 MB  | ≤ 2048 MB | pass |
| RTF                                 | ~0.15    | < 1    | —       |

Measured on the 5.8 s test clip, 100 ms hops. Latency grows with
accumulated audio (the encoder re-encodes the full prefix per hop —
same behavior as the upstream runtime). Source:
`results/board/budget_report.json`; numbers will be re-measured for the
final checkpoint. The streaming-tiny fallback is **not triggered**
(verified to name itself when the budget is exceeded).

## Final model (Strix Halo)

### Data preparation (task 10.1, on Strix Halo `max`)

| split      | chunks  | hours | mean   |
|------------|---------|-------|--------|
| train      | 110,004 | 216.67| 7.09 s |
| validation |   2,266 |   4.54| 7.21 s |
| test       |   2,389 |   4.75| 7.15 s |
| **total**  | 114,659 | **225.95** | — |

All durations within [1, 10] s; zero empty transcripts. This answers the
design's open question (MLS-it usable hours after segmentation+filtering):
**~226 h of 1–10 s chunks** (95% yield; the initial 86%-loss to
"unsplittable" was a text-splitter bug on single-sentence audiobook prose,
fixed with proportional word-boundary fallback).

Common Voice mix-in: **unavailable** — all `mozilla-foundation/common_voice_*`
dataset repos were removed from the HF hub (404; 17_0 survives as an empty
README shell). Training proceeds MLS-only per design D4 (CV was optional).

### Final training (task 10.2)

Deviations from plan, measured and forced by hardware reality:

- **Strix Halo benched.** Its iGPU shows bimodal per-batch performance —
  identical batch shapes alternate between ~1 s and ~130 s GPU time,
  attention-backend-independent (mem-efficient / AOTriton flash / math all
  affected) — making a 40k-step run a ~30-day proposition. Additionally its
  torch-2.12 mem-efficient SDPA backward produces inf LayerNorm gradients
  on this data (4/6 batches → NaN losses).
- **Final training runs on the 12 GB PC** (`rocm12g` profile, batch 8,
  math SDPA): uniform ~1.5 s/step (~16.5 h total), zero non-finite
  gradients over 10 measured batches, loss 4.9 → 1.8 by step 860.

Training-loop hardening (both machines): gradient clipping (max-norm 1.0),
non-finite loss/grad skip guard, and forced math SDPA.

Status: **running** (40k steps, curriculum 8k/16k/16k, eval + checkpoints
every 1000 steps).
