# Tasks: optimize-training-performance

## 1. Profiling spike (gate for all optimizations)

- [x] 1.1 Add a `profile-steps` target (Taskfile + module) that runs one
      representative training step on a given hardware profile under
      `torch.profiler` (record_shapes, no stack) and writes a
      kernel/self-time/fraction-of-step table to
      `results/profile/<hw>/profile.json`; verify the artifact names the
      dominant kernel and its fraction of step time
- [x] 1.2 Run the spike on the Strix box (`max`) and record the dominant
      kernel of the ~130 s slow step; verify the profile JSON exists and
      cite its finding in `docs/results.md` before any optimization is
      accepted (per spec: optimization must be profile-driven)

## 2. Remove per-step device drain

- [x] 2.1 Replace the per-step `torch.cuda.synchronize()` in
      `train_loop.py` with `zero_grad(set_to_none=True)` and scoped syncs
      only at save/eval boundaries; verify via a smoke run on the ROCm box
      that wall-time/step drops and no gfx1200 gradient race recurs
- [x] 2.2 If the race reappears, restrict sync to the first step after a
      checkpoint load (post-resume) rather than every step; verify the
      smoke run stays race-free and pipelined

## 3. bf16 encoder weights

- [x] 3.1 Reproduce and fix the streaming-encoder bf16 dtype-mismatch in
      `model_io.py` so weights load bf16 when the profile requests it;
      verify a forward+backward pass produces finite loss and gradients
- [x] 3.2 Run the smoke profile with bf16 encoder weights and verify eval
      WER parity with the fp32 path (no new NaN/inf steps); record the
      wall-time/step improvement

## 4. Data-loading overlap

- [x] 4.1 Set `persistent_workers=True` and `pin_memory=True` on the
      DataLoader(s) in `train_loop.py`; verify the smoke run still trains
      correctly and CPU-prep does not add serialized wall time to the step

## 5. Cheap best-checkpoint promotion

- [x] 5.1 Replace the `shutil.copytree` best-checkpoint promotion with a
      symlink (preserving `best_metric.json` semantics); verify the best
      checkpoint remains loadable and promotion is O(1)
- [x] 5.2 Verify the export/eval path (which reads the best checkpoint)
      follows the symlink correctly on a smoke export

## 6. Per-hardware performance gate

- [x] 6.1 Add a per-hardware `steps_per_second_min` (or
      `wall_time_per_step_max`) to `config.yaml` and record measured
      wall-time/step in run metadata; verify a below-gate profile reports
      measured-vs-allowed and fails the target (mirroring eval gates)
- [x] 6.2 Set the gate values from the measured numbers and verify the
      smoke profile passes the gate while a deliberately-slow profile fails

## 7. Documentation

- [x] 7.1 Update `docs/results.md` with measured per-step timing per
      hardware profile and the profiling-spike finding; verify numbers
      match the JSON artifacts (per repo convention)
- [x] 7.2 Update `docs/training.md` to describe the profiling spike and
      any new `profile-steps`/gate usage; verify commands match the
      Taskfile targets
