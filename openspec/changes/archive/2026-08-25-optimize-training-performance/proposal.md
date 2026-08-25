## Why

The Italian fine-tune of `moonshine-streaming-small` (123M params) trains
pathologically slowly on ROCm: ~0.5–3 s/step at batch 8 on the RX 9060 XT
(`rocm12g`), and the Strix Halo iGPU (`strix`) shows a bimodal ~1 s / ~130 s
per-batch behavior that made the planned 40k-step final run effectively a
~30-day proposition. The data pipeline is *not* the bottleneck (measured:
~26 ms/batch of WAV decode + Whisper mel + tokenization at batch 8), so the
cost is in GPU compute. The current final run (40k steps) is executing on
the 12 GB PC at ~1.5 s/step (~16.5 h total) precisely because Strix is
unusable — we need Strix (or a faster path) back in play to hit the
multi-day training budget comfortably.

## What Changes

- **Profile the slow step before optimizing further.** Add a profiling
  spike that runs `torch.profiler`/`rocprof` over one slow training step on
  the Strix box and names the single dominant kernel (~130 s). This turns
  the current hypothesis tree (rocBLAS/hipBLASLt kernel flapping vs
  shared-memory contention vs attention-backend) into one measured answer.
- **Remove the every-step `torch.cuda.synchronize()`** in `train_loop.py`
  (currently an uncommitted working-tree change). A full device drain each
  step destroys GPU pipelining and is worst on a slow iGPU; replace it with
  a targeted fix that still prevents the gfx1200 zero_grad race (e.g.
  `zero_grad(set_to_none=True)` ordering / sync only at save and eval
  boundaries).
- **Fix the documented bf16-weights dtype bug on the streaming encoder** so
  weights can load in bf16 (not fp32) and matmuls run bf16 (RDNA-native),
  halving memory traffic and unlocking fast GEMMs. This is the biggest
  single lever on a memory/FP32-bound iGPU.
- **Add cheap data-loading overlaps:** `persistent_workers=True` +
  `pin_memory=True`.
- **Stop `shutil.copytree`-ing the 1.6 GB best checkpoint every save**;
  use a symlink/hardlink so save cost is not a per-save 1.6 GB copy.
- **Record a performance gate** (measured wall-time/step per hardware
  profile) so a regression or a new-machine profile can be checked against
  a number rather than re-derived.

## Capabilities

### New Capabilities
- `training-performance`: profiling and optimization of the ROCm training
  loop — a profiling spike gate, bf16 weight handling for the streaming
  encoder, per-step sync hygiene, data-loading overlap, cheap checkpoint
  promotion, and a per-hardware wall-time/step gate.

### Modified Capabilities
<!-- None: this is a new capability. Existing capabilities (training-pipeline,
     task-orchestration) are unchanged at the spec level. -->

## Impact

- `src/moonshine_it/train_loop.py` — sync, dtype, data-loader, checkpoint
  promotion changes; the profiling spike.
- `src/moonshine_it/model_io.py` — bf16 weight loading for the streaming
  encoder.
- `config.yaml` — optional per-hardware `steps_per_second` gate / profiling
  toggle under `training`.
- `Taskfile.yml` — a `profile-steps` (or equivalent) spike target.
- `docs/results.md` — record measured per-step timing after the change.
- No change to model weights, eval methodology, export, or the `.ort`
  runtime contract.
