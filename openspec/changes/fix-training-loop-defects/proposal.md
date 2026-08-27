## Why

The training loop's progress metric does not describe the model being trained,
and three other defects mean the loop does not do what `config.yaml` and
`CLAUDE.md` say it does. Measured on `checkpoint-2000`, identical 48-clip MLS
validation slice, identical loss batches:

| weights | teacher-forced loss | WER | CER | output language |
|---|---|---|---|---|
| base `streaming-small` (untouched English) | 3.146 | 149.87% | 95.84% | English |
| **x** iterate — what `eval_wer` measures | 3.450 | 123.11% | 69.71% | **English** |
| **y** iterate — what the checkpoint held | **2.630** | **83.46%** | **32.57%** | Italian |

`AdamWScheduleFree` exposes two views of the weights in `p.data`: `y` (raw) and
`x` (a weighted average over the trajectory, weighted by
`scheduled_lr ** weight_lr_power`). With `weight_lr_power: 2.0` and a
**constant** learning rate, every iterate past LR warm-up carries equal weight,
so at step 2,000 `x` is a near-uniform mean that still contains the English
initialization. It scores *worse than the model training started from*, and it
emits English (`però in pro del mondo…` → `but in the midst of the world…`)
where `y` emits Italian (`pero im pro del mundo…`).

`quick_eval_wer` measures `x`. Every tuning decision recorded in
`config.yaml` — `learning_rate` 5e-5 → 1e-5, `grad_accum_steps` 1 → 4, dozens of
`shuffle_seed` values, two clean restarts, and discarding the 2,000→56,000
trajectory as noise — was made from that metric. The conclusion it produced
("~70,000 steps of genuine non-improvement") is not supported: on `y`, 2,000
steps moved WER 149.87% → 83.46% and CER 95.84% → 32.57%.

Commit `3390635` diagnosed the underlying mismatch correctly (checkpoints held
`y`, `eval_wer` described `x`) and resolved it in the wrong direction: it made
`save_checkpoint` call `optimizer.eval()`, so checkpoints now save `x`. Every
checkpoint, export, `.ort` bundle and release artifact written after that commit
ships the iterate that is worse than the English base model.
`results/eval/final_full.json` (83.47%) was measured on `y` from the pre-fix
`checkpoint-2000` and is not reproducible by anything the current code saves.

## What Changes

- **BREAKING (artifact semantics):** Standardize the pipeline on a single
  named schedule-free iterate for both the in-loop metric and the saved
  checkpoint, so `best_metric.json` ranks the same weights that ship.
  Reverses the direction of commit `3390635`.
- **Re-baseline the tuning decisions taken from the broken metric.**
  `learning_rate: 1.0e-5` and `grad_accum_steps: 4` were both prescribed by
  `eval_wer` on `x` and have no support from a valid measurement.
- **Restore chunked augmentation, or remove it.** It has never executed:
  `ASRDataset.__getitem__` calls `plan_chunks` with a single span covering the
  whole utterance, whose only boundary candidates are `0` and `total` — both
  filtered out — so `plan_chunks` returns `[]` and the `len(chunks) > 1` guard
  is false for every input. Verified `[]` at 3 s, 6 s, 8 s and 10 s.
- **Fix or retire the curriculum.** `preparation.max_duration_s: 10.0` caps
  every prepared row at 10.0 s, so stage 2 (`max_audio_s: 10.0`) and stage 3
  (`max_audio_s: 30.0`) filter nothing and are the same stage. Only stage 1
  (`5.0`) is real, and it drops 85,583 of 110,004 MLS rows.
- **Make the step budget independent of `grad_accum_steps`.** `max_steps:
  72000` was derived as 2 epochs at batch 8; with accumulation `step` counts
  optimizer updates, so it is now 8.1 epochs (~46 h at the measured 0.7
  steps/s), and `eval_steps: 1000` covers 4× more data than when it was set.
- **Make the in-loop metric comparable to the gate it predicts.** In-loop eval
  runs on `mls/validation.jsonl`; the `final` gate runs on the FLEURS Italian
  test split. The two are different distributions (18.0 vs 13.1 median
  chars/s) and are not comparable step-to-step against a gate threshold.

## Capabilities

### New Capabilities

*(None)*

### Modified Capabilities

- `training-pipeline`: Add requirements that the saved checkpoint and the
  reported metric describe the same schedule-free iterate; that curriculum
  stages must be effective against the prepared corpus or absent; that the
  step budget is expressed independently of gradient accumulation; and that
  configured augmentation actually executes.
- `evaluation`: Add a requirement that the in-loop training metric names the
  iterate and the split it measures, and is comparable to the gate it is used
  to predict.

## Impact

- `src/moonshine_it/train_loop.py`: `save_checkpoint` iterate selection,
  `quick_eval_wer` iterate + split, `ASRDataset.__getitem__` augmentation call,
  curriculum `Subset` construction, step-budget accounting.
- `src/moonshine_it/prepare.py`: `plan_chunks` contract when handed a single
  span (either accept a synthetic mid-point or reject the degenerate call
  loudly rather than returning `[]`).
- `config.yaml`: `learning_rate`, `grad_accum_steps`, `max_steps`,
  `training.profiles.final.curriculum`, `preparation.max_duration_s`, and the
  recorded rationale comments that were written from the broken metric.
- `results/train-final/`: `best_metric.json` and `checkpoint-best` rank `x`
  snapshots and are meaningless as a ranking; `checkpoint-2000` holds `y` with
  `train_mode: True` and must keep being interpreted that way.
- `results/eval/*.json`, `results/gates/*.json`: `final_full.json` /
  `final_streaming.json` (83.47%) describe `y`; `smoke_int8_streaming.json`
  records a −22.86 WER *improvement* after INT8 quantization, which is the
  signature of a degenerate baseline and should be re-measured.
- `docs/results.md`: recorded numbers must state which iterate they describe.
