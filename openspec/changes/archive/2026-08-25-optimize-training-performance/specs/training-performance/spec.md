## Purpose

Optimizes the ROCm training loop for the Italian Moonshine streaming
fine-tune so per-step wall time is bounded and hardware profiles are
usable, with a profiling spike that names the dominant cost before any
optimization is accepted.

## ADDED Requirements

### Requirement: Profiling spike gate for training performance
The training pipeline SHALL provide a profiling target that runs the
training step on a specified hardware profile and records which single
operation dominates wall time (kernel name and fraction of step time).
The optimization change SHALL NOT be considered complete until this
profile has been recorded and its result is referenced in the design.

#### Scenario: Profile a slow step on Strix
- **WHEN** the profiling target runs with hardware profile `strix`
- **THEN** it produces a profile artifact naming the dominant kernel and
  its fraction of the ~slow step time, and the run exits non-zero if the
  profile cannot be produced

#### Scenario: Optimization must be profile-driven
- **WHEN** a per-step optimization is proposed
- **THEN** the design MUST cite the profiling artifact that identifies the
  cost being removed, rather than asserting a cause

### Requirement: No full device drain on every optimization step
The training loop SHALL NOT issue a full device synchronization on every
optimizer step. Any synchronization required to avoid a hardware-specific
gradient-buffer race SHALL be scoped to the minimum needed (e.g. at save
and eval boundaries, or by using `zero_grad(set_to_none=True)` ordering)
and MUST not serialize the whole step.

#### Scenario: Training steps pipeline without per-step sync
- **WHEN** the training loop runs on a ROCm device
- **THEN** it does not call a full device-wide synchronize inside the
  per-step optimizer path, and the measured wall-time/step is not
  inflated by an unconditional device drain

### Requirement: bf16 weight loading for the streaming encoder
The model loader SHALL support loading the streaming encoder in bf16 so
that matmuls run in bf16 (RDNA-native) instead of fp32, without the
dtype-mismatch failure currently documented for bf16 encoder weights.
When bf16 weight loading is enabled, training MUST still produce finite
losses and gradients equivalent to the fp32 path.

#### Scenario: Load encoder weights in bf16
- **WHEN** the training profile requests bf16 precision
- **THEN** the encoder weights load in bf16 with no dtype-mismatch error,
  and a forward+backward pass produces finite loss and gradients

#### Scenario: bf16 path preserves correctness
- **WHEN** training runs with bf16 encoder weights
- **THEN** the loss curve and eval WER remain consistent with the fp32
  baseline (no new NaN/inf steps introduced by the dtype change)

### Requirement: Data-loading overlap
The training data loader SHALL keep worker processes alive across epochs
and pin batch tensors to page-locked memory so that CPU-side data
preparation overlaps with GPU compute rather than stalling the step.

#### Scenario: Workers persist and tensors are pinned
- **WHEN** the training loader is constructed for a CUDA/ROCm device
- **THEN** it uses persistent workers and pinned-memory batches, and the
  CPU-side preparation (WAV decode + mel + tokenize) does not add
  serialized wall time to the GPU step

### Requirement: Cheap best-checkpoint promotion
Promoting the best checkpoint SHALL NOT copy the full 1.6 GB checkpoint
on every save. The best-checkpoint pointer SHALL be updated via a
symlink/hardlink or equivalent O(1) operation, preserving the previously
recorded best metric semantics.

#### Scenario: Best checkpoint promotes without full copy
- **WHEN** a checkpoint with a better eval WER is saved
- **THEN** the best-checkpoint marker is updated without copying the full
  checkpoint directory, and the best checkpoint remains loadable

### Requirement: Per-hardware training-performance gate
The training pipeline SHALL record a measured wall-time/step (or
steps/second) for each hardware profile and compare it against a
configured gate. A profile that falls below the gate SHALL report the
measured-vs-allowed values and fail the target, mirroring the existing
evaluation-gate behavior.

#### Scenario: Record and gate per-step timing
- **WHEN** a training run completes on a hardware profile
- **THEN** it records measured wall-time/step in run metadata and compares
  it against the configured gate for that profile, failing with
  measured-vs-allowed values if below threshold
