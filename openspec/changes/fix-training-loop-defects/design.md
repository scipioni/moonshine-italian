## Context

See `proposal.md` — Why for the measured evidence. Design-relevant mechanics:

`AdamWScheduleFree` keeps a per-parameter `z` buffer and exposes two views of
the weights in `p.data`. `train()` and `eval()` are inverse in-place transforms
driven by the unchanged `z`, and `param_groups[0]['train_mode']` records which
view `p.data` currently holds:

```
     train_mode: True                        train_mode: False
     p.data = y  (raw iterate)  ──eval()──▶  p.data = x  (averaged iterate)
                                ◀─train()──
```

`eval()` only transforms parameters for which `'z' in state` — i.e. parameters
that have taken at least one `step()`. Verified in the installed source.

`x` is a weighted average over the trajectory with weight
`scheduled_lr ** weight_lr_power`. The run's recorded state at step 2,000:
`weight_lr_power: 2.0`, `lr_max: 5e-5`, `scheduled_lr: 5e-5`, `warmup_steps:
500`, `k: 2000`. Constant `scheduled_lr` past warm-up means near-uniform
weights, so `x` at step 2,000 is roughly the mean of all 2,000 iterates — still
substantially the English initialization.

Two facts constrain any fix:

- `checkpoint-2000` was written before commit `3390635` and stores `train_mode:
  True`. Its `model.safetensors` therefore holds `y`, and its recorded
  `eval_wer: 102.7` describes `x`. Resuming from it is nevertheless *correct*:
  schedule-free's `train()` early-returns when `train_mode` is already `True`,
  so no double transform occurs. This checkpoint is usable; only its metric is
  mislabelled.
- Every checkpoint written after `3390635` stores `x` with `train_mode: False`.
  These are recoverable — `train()` re-derives `y` from the saved `z` — but they
  are not interchangeable with pre-fix checkpoints, and nothing on disk
  currently distinguishes them beyond that flag.

## Goals / Non-Goals

**Goals:**
- One named iterate used consistently for in-loop eval, checkpointing,
  best-checkpoint ranking, export and release.
- Existing checkpoints remain loadable and correctly interpreted under the new
  convention without re-training.
- The three loop defects (dead augmentation, ineffective curriculum stages,
  accumulation-dependent step budget) fail loudly rather than silently.
- Re-establish a trustworthy convergence baseline so the
  `vocabulary-transplantation` change has a real precondition to test against.

**Non-Goals:**
- Retro-fitting valid metrics onto the discarded 2,000→56,000 trajectory. That
  data is gone; it is not worth reconstructing.
- Changing the optimizer. Schedule-free AdamW stays.
- Re-running the release pipeline. Correcting `results/` and
  `artifacts/release/` is downstream of a converged run, not part of this change.
- Deciding the vocabulary question. See `vocabulary-transplantation`.

## Decisions

### 1. Standardize on the raw iterate `y`, and treat `x` as opt-in
- **Decision:** Use `y` for in-loop eval, checkpoint save, ranking and export.
  Save with `train_mode: True`. Record the iterate name in
  `trainer_state.json` and in the eval results JSON.
- **Rationale:** `y` is the only iterate that has ever been measured as better
  than the initialization (2.630 vs 3.146 loss; 83.46% vs 149.87% WER). `x` is
  currently worse than the untouched English base model and emits English. A
  metric that ranks below the initialization cannot order checkpoints, which is
  the whole job of `best_metric.json`.
- **Alternatives considered:**
  - *Keep `x`, add LR decay so the average stops being contaminated.* This is
    the schedule-free-native answer and is where the pipeline should end up for
    a long, converged run — averaging is a real variance reduction once the
    trajectory is inside a basin. Rejected as the immediate fix because it
    requires a fresh LR schedule and a re-tune before any measurement is
    trustworthy, and the change's purpose is to restore a trustworthy
    measurement first.
  - *Keep `x` but start averaging at step N.* Not supported by the optimizer
    without patching it; adds a hyperparameter with no evidence behind it.
  - *Report both.* Doubles in-loop eval cost (already 64 utterances of greedy
    decode per eval) and leaves the ranking question unanswered.
- **Follow-up:** once a run converges on `y`, evaluate `x` once at the end. If
  it wins there, revisit with a decaying LR. Record that measurement rather
  than assuming either way.
- **Sanity latch requires a sustained streak, not a single eval.** The initial
  implementation compared every in-loop eval against a fixed run-start
  baseline with zero tolerance, and it fired on the very first eval of a
  2-step test run: a model that has barely moved from its initialization is
  statistically indistinguishable from the baseline on a small sample, so a
  single regression is close to a coin flip, not evidence of the metric fault
  this latch exists to catch. Revised to require
  `REGRESSION_STREAK_THRESHOLD` (3) consecutive regressions before failing,
  resetting to 0 on any eval that does not regress. A sustained streak is the
  actual failure signature (the `x` iterate stayed worse than baseline for
  every eval across the discarded 2,000->56,000 trajectory); a lone noisy
  eval is not.

### 2. `train_mode` is the on-disk iterate discriminator; do not migrate checkpoints
- **Decision:** Read `optimizer.pt`'s `train_mode` to determine which iterate a
  checkpoint's `model.safetensors` holds, and normalize to `y` on load. Leave
  existing checkpoint files untouched.
- **Rationale:** the flag is already persisted and already correct in both eras;
  it just was never read as the discriminator. Rewriting existing checkpoints
  would destroy the only record of which era they came from, and the transform
  is cheap and lossless in either direction given `z`.
- **Consequence:** `checkpoint-2000` needs no repair. Post-`3390635` checkpoints
  are loadable but their `best_metric.json` ranking is void and should be
  recomputed, not trusted.

### 3. Re-baseline the metric-derived config values one at a time
- **Decision:** Revert `learning_rate` to 5.0e-5 and `grad_accum_steps` to 1 as
  the starting point, then change at most one at a time, each justified by a
  metric that satisfies the evaluation spec.
- **Rationale:** both values were set from `eval_wer` on `x`. Reverting is not a
  claim that they were wrong — `grad_accum_steps: 4` demonstrably tightened
  gnorm from 30–270 to 13–27, which is a real measurement independent of the
  metric fault. It is a claim that they are *unjustified*, and the cheapest way
  to a trustworthy baseline is the configuration whose only valid measurement
  (step 0→2,000 on `y`) was working.
- **Note:** gnorm is not a strong signal here regardless. Pre-clip norms of
  13–30 against `clip_grad_norm_(max_norm=1.0)` mean every update is scaled down
  by 13–30×, and Adam is approximately invariant to a uniform gradient rescale,
  so the clipped-norm band mostly reports how far the raw gradient sits above
  the clip rather than anything about update quality.

### 4. Fix the chunk-planning contract at the callee, not the caller
- **Decision:** `plan_chunks` SHALL distinguish "no split needed"
  (`[(0, total)]`) from "split impossible" (an explicit refusal). The training
  caller either supplies real VAD spans or asks for a duration-proportional
  split, and treats a refusal as a skipped augmentation for that sample.
- **Rationale:** the bug is that both outcomes currently reach the caller as a
  falsy `len(chunks) > 1` check, so an impossible split is indistinguishable
  from an unnecessary one. `prepare.py` passes real spans and works; only the
  training caller passes the degenerate single span. Fixing the contract makes
  the failure visible to both callers instead of fixing one call site.
- **Alternative considered:** synthesize a mid-point boundary inside
  `plan_chunks` when no candidate exists. Rejected: it would silently cut mid-word
  in the preparation path too, where cutting at silence is the point.

### 5. Curriculum: trim the ineffective stage, don't raise the preparation ceiling
- **Decision:** Merge the former stage 3 (`max_audio_s: 30.0`, identical to
  stage 2 against this corpus) into stage 2, keeping the total step budget
  unchanged (8,000 + 32,000 = 40,000). Do not raise `preparation.max_duration_s`
  or re-prepare.
- **Resolved by measurement:** measured the cap-bound fraction of each
  prepared train set -- rows at or above 9.9s, i.e. rows a higher cap could
  actually lengthen: `common_voice` 0.0%, `fleurs` 1.3%, `mls` 3.5%.
  `common_voice` and `fleurs` are natively short recordings, not truncated by
  the cap -- raising it would only ever affect `mls`'s continuous audiobook
  narration, and even there for a small minority of rows. No requirement
  anywhere in the board budget (`line_latency_ms_max`, `redecode_ms_max`)
  calls for training on utterances longer than the eval domain ever produces.
- **Rationale:** raising the ceiling costs re-preparing 285,401 rows (mls
  alone is 216.7h) for a benefit measured at ≤3.5% of rows in the one dataset
  it would touch. Deleting the ineffective stage costs nothing and preserves
  the curriculum's real transition (stage 1's 5.0s bound admits 144,498 of
  285,401 rows; stage 2's 10.0s bound admits all of them) -- verified by
  running `validate_curriculum` against the actual prepared corpus.
- **Alternative considered:** raise `preparation.max_duration_s` and re-prepare.
  Rejected on the measurement above; revisit only if a future board-latency
  measurement shows the deployed use case needs longer training utterances
  than 10s -- nothing currently indicates it does.

### 6. Express the step budget in samples, derive steps
- **Decision:** Configure epochs (or samples) for the final profile and derive
  `max_steps` from `batch_size × grad_accum_steps` at run start, recording all
  four numbers in `run_metadata.json`.
- **Rationale:** `max_steps: 72000` was hand-derived from batch 8 and silently
  became 8.1 epochs when accumulation landed. Deriving it removes the class of
  error rather than correcting this instance. `eval_steps` and `save_steps`
  should be derived the same way so eval cadence is data-proportional.

### 7. Annotate stale results rather than deleting them
- **Decision:** Leave `results/eval/*.json` and `results/gates/*.json` in place;
  add the iterate name and a note where a recorded number is known not to
  describe the current convention.
- **Rationale:** the repo's convention is that `results/` is the record. Deleting
  a misleading number destroys the evidence that the fault existed;
  `smoke_int8_streaming.json`'s −22.86 WER "improvement" after INT8
  quantization is diagnostically valuable precisely as a symptom.

## Risks / Trade-offs

- **[Risk] `y` is the noisier iterate.** Without averaging, step-to-step WER on
  `y` will be less stable than `x` was, and best-checkpoint selection may chase
  noise.
  - **Mitigation:** `in_loop_samples: 64` already gives finer granularity than
    the original 8; keep ranking to `save_steps` boundaries and accept noise in
    exchange for a metric that tracks the model.
- **[Risk] Reverting `learning_rate` and `grad_accum_steps` re-introduces the
  instability they were chosen to address.** The gnorm tightening under
  accumulation was real.
  - **Mitigation:** change one variable at a time against a valid metric; the
    `max_grad_norm_skip` and non-finite-loss guards remain in place.
- **[Risk] Post-`3390635` checkpoints are silently misread by any tool that
  does not consult `train_mode`.** Export, parity and release all load
  `model.safetensors` directly.
  - **Mitigation:** normalize on load in `model_io.py`, and fail if a checkpoint
    directory has `model.safetensors` without an `optimizer.pt` to disambiguate.
- **[Risk] The convergence question remains open.** This change fixes the
  instrument; it does not prove the model converges. `y` improving 149.87% →
  83.46% in 2,000 steps is encouraging but is 2,000 steps.
  - **Mitigation:** that is the point of the re-baselined run, and it is
    `vocabulary-transplantation`'s precondition.
- **[Trade-off] Deferring the LR-decay question.** Decision 1 keeps a constant
  LR with schedule-free, which is the optimizer's designed usage but leaves its
  averaging benefit unused.

## Migration Plan

1. Land the iterate convention and the loud failures (specs above). No
   re-training required.
2. Recompute `best_metric.json` by re-evaluating existing checkpoints on `y`
   over a fixed slice, so `checkpoint-best` points at a real winner.
3. Re-baseline config per Decision 3 and start a fresh final run from
   `checkpoint-2000` (which already holds `y`).
4. Read the convergence result. Only then revisit
   `vocabulary-transplantation`'s task 0.1.

Rollback: the iterate convention is a load/save-time transform; reverting the
commit restores the previous behaviour without invalidating checkpoints, since
`train_mode` records the truth in both eras.

## Open Questions

- Does `x` overtake `y` once a run is genuinely converged? Worth one
  measurement at the end of the re-baselined run; it does not affect this
  change's specs or tasks.
