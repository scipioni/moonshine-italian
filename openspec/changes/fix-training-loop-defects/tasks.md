## 1. Iterate convention

- [x] 1.1 Add an iterate helper to `src/moonshine_it/train_loop.py` that reports
      which view `p.data` holds (from `optimizer.param_groups[0]['train_mode']`)
      and normalizes to `y`; verify with a unit test that round-trips
      `eval()`/`train()` on a tiny module and asserts the weights return
      bit-identical.
- [x] 1.2 Change `save_checkpoint` to save the `y` iterate with `train_mode:
      True` (reversing commit `3390635`'s direction, design Decision 1), and
      write `"iterate": "y"` into `trainer_state.json`; verify a smoke run's
      checkpoint records `train_mode: True` and the iterate name.
- [x] 1.3 Change `quick_eval_wer` to measure the same iterate that will be
      saved, and return the iterate name alongside the value; verify a smoke run
      logs the iterate name with every in-loop WER.
- [x] 1.4 Add the reproducibility check from the spec: after an in-loop eval +
      save at the same step, re-evaluating the saved checkpoint offline
      reproduces the recorded metric within tolerance. Verify as a test over a
      2-step smoke run with a 4-utterance slice.
- [x] 1.5 Make `model_io.load_model_and_processor` consult `optimizer.pt`'s
      `train_mode` when loading a checkpoint directory and normalize to `y`,
      failing loudly when `model.safetensors` is present without an
      `optimizer.pt` to disambiguate (design Decision 2); verify by loading both
      `checkpoint-2000` (pre-fix, `train_mode: True`) and a freshly written
      post-fix checkpoint and asserting both yield the `y` weights.
- [x] 1.6 Add the sanity latch from the spec: fail training if the configured
      iterate is sustainedly (3 consecutive evals — design Decision 1 addendum,
      found necessary when a 2-step test tripped a zero-tolerance single-eval
      version on ordinary early noise) worse on held-out Italian than the base
      checkpoint it was initialized from, reporting both measured values.
      Verified with tests for: a single noisy regression (tolerated), a
      sustained streak (raises), and streak reset on improvement.

## 2. Re-rank existing checkpoints

- [x] 2.1 Recompute `results/train-final/best_metric.json` by evaluating each
      existing `checkpoint-*` on the `y` iterate over one fixed slice, and
      re-point `checkpoint-best`; verify the recorded values are reproducible by
      re-running the same evaluation twice.
- [x] 2.2 Annotate the stale records in `results/eval/*.json` and
      `results/gates/*.json` with the iterate they describe, without deleting
      them (design Decision 7) — specifically that `final_full.json` /
      `final_streaming.json` (83.47%) describe `y` from the pre-fix
      `checkpoint-2000`, and that `smoke_int8_streaming.json`'s −22.86 WER
      post-quantization "improvement" is a degenerate-baseline artifact to be
      re-measured. Verify `task validate` still parses every annotated file.

## 3. Chunk-planning contract and augmentation

- [x] 3.1 Change `prepare.plan_chunks` to distinguish "no split needed" from
      "split impossible" (design Decision 4); verify with unit tests covering a
      single-span request, a real multi-span request, and a request with no
      admissible cut point.
- [x] 3.2 Fix `ASRDataset.__getitem__` so chunked augmentation actually applies
      at the configured probability — currently it never fires for any duration
      (verified `[]` at 3 s, 6 s, 8 s, 10 s); verify with a test that augments a
      known 8 s utterance and asserts both the audio and the text were split.
- [x] 3.3 Add the startup measurement required by the spec: record the fraction
      of samples actually augmented over a startup sample and fail if it is zero
      while the configured probability is not; verify by running with
      `probability: 0.3` and asserting the recorded fraction is non-zero, and by
      running with the pre-fix code path and asserting the failure fires.
- [x] 3.4 Confirm `prepare.py`'s own use of `plan_chunks` is unchanged by 3.1 —
      verify by re-preparing the FLEURS slice and diffing the manifest against
      the current one for byte equality.

## 4. Curriculum validation

- [x] 4.1 Validate each curriculum stage against the prepared manifests at
      training start and fail naming any stage whose bound excludes no rows the
      previous stage included; verify the current `config.yaml` final curriculum
      fails on stages 2 (`max_audio_s: 10.0`) and 3 (`30.0`) given the measured
      corpus maximum of 10.0 s.
- [x] 4.2 Record per-stage bound and resulting row count in
      `run_metadata.json`; verify a smoke run's metadata lists them.
- [x] 4.3 Resolved design Decision 5 by measurement: trimmed the curriculum
      (merged the ineffective former stage 3 into stage 2, total steps
      unchanged) rather than raising `preparation.max_duration_s` — measured
      cap-bound row fraction is 0.0% (common_voice), 1.3% (fleurs), 3.5%
      (mls), and no board-budget requirement calls for >10s training
      utterances. Verified: `validate_curriculum` passes against the real
      285,401-row corpus (stage 1 admits 144,498 rows, stage 2 all of them).

## 5. Step budget

- [x] 5.1 Express the final profile's training length in epochs (or samples) and
      derive `max_steps` from `batch_size × grad_accum_steps` at run start
      (design Decision 6); verify that setting `grad_accum_steps` to 1 and to 4
      yields the same recorded sample count and epoch count.
- [x] 5.2 Derive `eval_steps` and `save_steps` from the same budget so eval
      cadence is data-proportional; verify a dry run reports the derived values.
- [x] 5.3 Record effective batch size, total samples and epoch count in
      `run_metadata.json`; verify against a hand computation for the
      `mls + common_voice + fleurs` mix (285,401 rows).

## 6. Re-baseline and re-record

- [x] 6.1 Revert `learning_rate` to 5.0e-5 and `grad_accum_steps` to 1 (design
      Decision 3) and rewrite the `config.yaml` rationale comments that were
      written from the broken metric — including the `learning_rate` comment's
      claim about "curriculum stage 3, full-length up to 30s audio", which does
      not exist. Verify by reading back that no comment cites an `x`-iterate
      `eval_wer` as evidence.
- [ ] 6.2 Run a final training run from `checkpoint-2000` under the corrected
      convention and record the `y`-iterate WER curve; verify the run record
      satisfies the evaluation spec (split, sample count, iterate name present).
- [ ] 6.3 Update `docs/results.md` with iterate-labelled numbers copied verbatim
      from the JSON artifacts; verify every number in the doc appears in a file
      under `results/`.
- [x] 6.4 Update `CLAUDE.md`'s pipeline description where it states behaviour
      this change corrects (chunked-audio augmentation as active, curriculum
      staging on `final`); verify by re-reading against the corrected code.

## 7. Downstream check

- [ ] 7.1 Re-run `task eval PROFILE=rocm12g` and `task export PROFILE=rocm12g`
      against the re-ranked `checkpoint-best` and confirm `parity.py` still
      passes; verify the gate records name the iterate.
- [ ] 7.2 Re-measure the `post_quant` gate on a non-degenerate baseline and
      confirm the INT8 delta is a degradation rather than an improvement; verify
      against `results/gates/post_quant.json`.
- [ ] 7.3 Report the convergence outcome to `vocabulary-transplantation` task
      0.1 as its precondition evidence, replacing the void 102.7% `x`-iterate
      baseline. Verify that change's task 0.1 cites a measurement satisfying the
      evaluation spec.
