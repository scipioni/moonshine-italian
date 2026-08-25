# Training

One code path (`train.py` → `src/moonshine_it/train_loop.py`) serves both
profiles; only dataset size and step counts differ.

## Latches (enforced, not advisory)

- **Spike latch**: any training run refuses to start without a passing spike
  verdict (`results/spike/verdict.json`). A failed verdict requires an
  explicit fallback base in `config.yaml` (`base_model.selected_base`).
- **Smoke latch**: `task final-train` additionally requires a recorded smoke
  success (`results/smoke/record.json`, written by `task smoke`).

## Smoke training (12 GB PC)

```bash
task train PROFILE=rocm12g                    # TRAIN_PROFILE defaults to smoke
```

60 steps on the FLEURS smoke slice; saves reloadable checkpoints +
TensorBoard logs under `results/train-smoke/` (see `results/logs/`).
Loss curve and eval outcomes land in `results/eval/smoke_*.json`.

## Dry run / resume

```bash
uv run python train.py --profile smoke --hardware rocm12g --dry-run-steps 20
uv run python train.py --profile smoke --hardware rocm12g            # auto-resume
uv run python train.py --profile smoke --hardware rocm12g --no-resume
```

An interrupted run resumes at the recorded step without replaying optimizer
state.

## Performance profiling spike

`task profile-steps PROFILE=<hw>` runs one training step under
`torch.profiler` and writes a kernel/self-time/fraction-of-step table to
`results/profile/<hw>/profile.json`, naming the single dominant kernel of
the step. This is the gate for training-performance optimizations: an
optimization is only accepted if its design cites the dominant kernel named
here (see `openspec/changes/optimize-training-performance`).

```bash
task profile-steps PROFILE=strix     # name the dominant kernel of the slow step
```

## Per-hardware performance gate

Each hardware profile in `config.yaml` carries a `steps_per_second_min`.
After a training run the loop records measured `steps_per_second` /
`wall_time_per_step_s` in `run_metadata.json` and, if it falls below the
gate, fails the target with measured-vs-allowed values (mirroring the eval
gates). Set the value from a measured run; a regression or a newly-added
profile gets a number to hit instead of re-deriving a budget.

## Final training (Strix Halo)

```bash
task final-train PROFILE=strix
```

Multi-day run: schedule-free AdamW, curriculum stages (short → medium → full
length audio, staged per `config.yaml` `training.profiles.final.curriculum`),
chunked-augmentation regularizer, eval + checkpoints every 1000 steps to
`results/train-final/`. The best checkpoint (eval WER) is promoted as
`checkpoint-best`.

## Method notes

Adapted from the community `finetune-moonshine-asr` methods (HF
transformers-style loading, schedule-free AdamW, curriculum, jiwer hooks),
extended for the streaming checkpoint: full-utterance training, streaming is
a *runtime* behavior exercised at eval/export time; chunked augmentation
(random truncation with truncated transcripts) regularizes toward the
chunked streaming eval condition.
