## Context

See `proposal.md` — Why. Current state: `train_loop.py` runs the shared
smoke/final loop; weights load fp32 (`model_io.py`), compute is bf16 via
autocast; an uncommitted `torch.cuda.synchronize()` sits in the per-step
optimizer path; best-checkpoint promotion does a 1.6 GB `shutil.copytree`
per save; the data loader uses `num_workers` but no `persistent_workers` /
`pin_memory`. Measured data-pipeline cost is ~26 ms/batch at batch 8, so
GPU compute dominates. Strix (`gfx1151` iGPU) shows bimodal ~1 s/~130 s
per-batch that the current code comments attribute to attention-backend
issues, but the `math` fallback did not remove the bimodality — the cause
is unverified.

## Goals / Non-Goals

**Goals:**
- Name the dominant kernel of the slow step on Strix via a profiling spike
  (this is the gate for any optimization).
- Restore pipelining by removing the per-step device drain.
- Enable bf16 encoder weights so matmuls run bf16.
- Overlap data loading and make best-checkpoint promotion O(1).
- Add a per-hardware wall-time/step gate.

**Non-Goals:**
- Changing model architecture, eval methodology, export, or the `.ort`
  runtime contract.
- Switching the base model or the training data.
- Fixing the underlying rocBLAS/hipBLASLt kernel-selection behavior in
  ROCm itself (out of our control); we only route around it.
- Re-running the 40k-step final training as part of this change (that is
  a later execution step once the loop is fast).

## Decisions

### D1: Profile before optimizing — `torch.profiler` step spike (gated)
Run `torch.profiler` (with `record_shapes=True`, `with_stack=False`) over
one representative slow step on the target hardware, plus `rocprof` if the
torch-side attribution is ambiguous, and dump a table of
kernel / self-CPU-time / self-CUDA-time / fraction-of-step. Write the
dominant kernel + fraction to a JSON artifact (e.g. `results/profile/<hw>/`).
This is the first task and gates the rest — an optimization is only
accepted if its design cites this artifact. *Alternative rejected:* tuning
blindly (the attention-backend swaps already did that and missed).

### D2: Replace per-step `torch.cuda.synchronize()` with a scoped fix
The sync was added to avoid a gfx1200 race where `zero_grad`'s async free
races optimizer-step kernels still reading `.grad`. Instead of a full
device drain every step, use `optimizer.zero_grad(set_to_none=True)` and
keep the device drain only at save/eval boundaries (where we already
serialize). If the race reappears, restrict the sync to the first step
after a checkpoint load rather than every step. Verify by running the
smoke profile on the ROCm box and comparing wall-time/step before/after.
*Alternative considered:* keep the sync — rejected because it serializes
the whole step, worst on a slow iGPU.

### D3: bf16 encoder weights
`model_io.load_model_and_processor` currently forces fp32 because the
streaming encoder has a dtype-mismatch failure when weights are bf16.
Investigate and fix that mismatch (likely a conv/frontend op that needs an
explicit `.to(dtype)` or a cast at the boundary) so weights load bf16 when
the profile requests it. This halves weight memory traffic and lets GEMMs
run bf16 (RDNA-native), the largest lever on a memory/FP32-bound iGPU.
Correctness is guarded by the spec: finite loss/grads and eval-WER parity
with the fp32 path. *Alternative considered:* keep fp32 weights + autocast
— rejected because autocast alone still reads fp32 weights and does not
remove the fp32 GEMM cost.

### D4: Data-loader overlap
Set `persistent_workers=True` and `pin_memory=True` on the DataLoader(s)
in `train_loop.py`. Data is already fast (~26 ms), so this is a cheap
overlap win rather than a fix for a measured bottleneck. *Alternative
considered:* pre-extract mel features on disk — rejected as unnecessary
given the measured cost and added storage/complexity.

### D5: O(1) best-checkpoint promotion
Replace the `shutil.copytree` in `save_checkpoint` with a symlink to the
best checkpoint (or a relative symlink + marker JSON as today). Preserve
the `best_metric.json` semantics. *Alternative considered:* hardlink —
rejected because model shards are written fresh per checkpoint and a
symlink is simpler and equally O(1).

### D6: Per-hardware wall-time/step gate
Add an optional `steps_per_second_min` (or `wall_time_per_step_max`) per
hardware profile in `config.yaml`, recorded in run metadata and compared
after a run (mirroring `evaluation.gates`). A miss reports
measured-vs-allowed and fails the target. This gives future hardware
profiles a number to hit instead of re-deriving a budget.

## Risks / Trade-offs

- [Strix bimodality is a rocBLAS kernel-selection flake we can't patch] →
  Mitigation: the profile names it; we then route around it (bf16 GEMMs,
  pinned kernels, or accept Strix and run final on `rocm12g` as today).
- [bf16 encoder weights introduce a new dtype bug or NaN path] →
  Mitigation: spec requires finite loss/grads and eval-WER parity with
  fp32; the smoke profile exercises it before any final run.
- [Removing the sync reintroduces the gfx1200 race] → Mitigation: scoped
  sync at save/eval + `set_to_none`; verify on the ROCm box; if needed,
  sync only post-load.
- [Symlink best-checkpoint breaks a load that follows the symlink] →
  Mitigation: `save_pretrained`/`from_pretrained` follow symlinks fine; the
  eval/export path reads via the marker which points at the symlink target.

## Migration Plan

This is a training-loop optimization, not a schema/artifact change. The
running 40k-step final training is left untouched; the optimized loop is
validated on the smoke profile first, then used for any subsequent final
run. No data or artifact format changes; existing checkpoints remain
loadable. Rollback = revert to the committed loop (the sync change is
currently uncommitted, so restoring is a clean `git checkout` of
`train_loop.py`).

## Open Questions

- Which exact kernel dominates the Strix slow step (rocBLAS GEMM vs
  something else)? — answered by D1's profile; deferrable until the spike
  runs.
- Whether the bf16-encoder dtype mismatch is a one-line cast or a deeper
  frontend issue — sized during D3; does not change the approach.
