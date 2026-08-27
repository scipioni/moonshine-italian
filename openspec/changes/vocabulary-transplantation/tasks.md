## 0. Preconditions

- [ ] 0.1 Confirm final training converges on the current vocabulary (eval WER
      improves against `checkpoint-2000`'s 102.7% baseline) before starting this
      change — see design "Sequencing". Record the pre-transplant WER as the
      comparison baseline for task 5.2.
- [ ] 0.2 Record the pre-transplant tokenizer fertility on held-out Italian
      validation text (measured: 1.738 tok/word, 3.37 chars/tok) as the
      comparison baseline for task 5.1.

## 1. Custom Tokenizer Generation & Configuration

- [ ] 1.1 Concatenate training transcripts from MLS, Common Voice and FLEURS
      into a unified Italian text corpus.
- [ ] 1.2 Implement a HuggingFace Tokenizers training routine to train a
      12,288-token Italian byte-level BPE tokenizer, saving `tokenizer.json`
      plus processor configuration.
- [ ] 1.3 Add custom tokenizer settings to `config.yaml` under `training`
      (`custom_tokenizer` enable flag, `vocab_size: 12288`,
      `tokenizer_warmup_steps: 5000`, `fertility_target`). Use a distinct key
      from the existing `training.warmup_steps`, which is the schedule-free
      AdamW LR warm-up and must not be overloaded.

## 2. Model Resizing & Weight Initialization

- [ ] 2.1 Update `src/moonshine_it/model_io.py` to detect a custom Italian
      tokenizer and call `model.resize_token_embeddings(vocab_size)` at model-load
      time — **before** `train_loop.py` constructs the optimizer (design
      Decision 3). The existing optimizer/model parameter-identity guard in
      `train_loop.py` must remain in place as the backstop.
- [ ] 2.2 **Explicitly re-initialize** both `model.decoder.embed_tokens.weight`
      and `proj_out.weight` after the resize (normal distribution, matching the
      model's initializer range). Note: the projection is named `proj_out` on
      this architecture, **not** `lm_head`; and because 32,768 → 12,288 is a
      *shrink*, HuggingFace truncates rather than adding rows, so nothing is
      re-initialized by default — new Italian token IDs would otherwise inherit
      unrelated retained English embeddings.
- [ ] 2.3 Add an assertion/test that after resize+init, neither matrix is
      bitwise-equal to the first `vocab_size` rows of the pre-trained matrices,
      so a future regression to plain truncation fails loudly.
- [ ] 2.4 Add a test that `config.vocab_size`, `embed_tokens`, and `proj_out`
      all agree on the new vocabulary size after loading.

## 3. Warm-up Training Phase (Stage-Based Freezing)

- [ ] 3.1 Write a helper in `src/moonshine_it/train_loop.py` to freeze/unfreeze
      the audio encoder weights (`model.encoder` and its projection).
- [ ] 3.2 Integrate Stage A (warm-up): keep the encoder frozen for the first
      `tokenizer_warmup_steps` so only the text embedding/projection are optimized.
- [ ] 3.3 Integrate Stage B: unfreeze the encoder once the step count exceeds the
      warm-up threshold.
- [ ] 3.4 Handle the schedule-free AdamW interaction (design Decision 5): frozen
      parameters never acquire a `z` buffer and are skipped by the `train()`/`eval()`
      iterate swap. Verify that at the Stage A→B boundary, and in any Stage A
      checkpoint, all parameters agree on which iterate ("x" vs "y") they hold —
      the repo has already had one x/y-iterate mismatch bug between eval and
      checkpointing.
- [ ] 3.5 Verify resume works across the stage boundary (interrupt during Stage A
      and during Stage B; confirm the correct freeze state is restored).

## 4. Evaluation, Export & Release Pipeline Updates

- [ ] 4.1 Update `src/moonshine_it/evaluate.py` and `evaluate_cli.py` to load the
      custom tokenizer and resized model.
- [ ] 4.2 Adjust `src/moonshine_it/export.py` so vocabulary-dimension-dependent
      graphs (notably `decoder_kv`) resolve the new size dynamically, and confirm
      `parity.py` still passes against the resized model.
- [ ] 4.3 Update `src/moonshine_it/release.py` so the custom `tokenizer.json` and
      processor config are packaged into the release bundle and covered by the
      checksum manifest.
- [ ] 4.4 Confirm `src/moonshine_it/ort_runtime.py` loads the custom tokenizer
      on-board (it reads `tokenizer.json` directly and must stay dependency-free).
- [ ] 4.5 Re-check `evaluation.streaming.max_tokens_per_second` against the new
      fertility. Currently non-binding (Italian needs ~3.8–4.2 tok/s against a
      budget of 13.0); fewer tokens/word makes it more generous, so this is a
      confirmation, not an expected change.

## 5. Outcome Gate & Re-baselining

- [ ] 5.1 Record measured post-transplant fertility on held-out Italian text and
      fail the change if it does not beat the task 0.2 baseline by the configured
      `fertility_target` (expected ≈1.295 tok/word, −25.5%).
- [ ] 5.2 Run the full evaluation and fail the change if streaming WER regresses
      against the task 0.1 pre-transplant baseline.
- [ ] 5.3 Re-record the spike baseline with the new vocabulary — the `smoke` and
      `post_quant` gates are *relative* to a baseline measured with the old
      32,768-entry vocabulary and are not comparable across the transplant.
- [ ] 5.4 Update `src/moonshine_it/spike.py`'s tokenizer spike to measure fertility
      against a reference baseline and fail on a threshold. It currently sets `ok`
      purely from `exact_roundtrip` on a hand-picked sample, so it reports
      "tokenizer: OK" for any vocabulary that can merely represent Italian —
      which is why the English vocabulary's unsuitability was never flagged.
- [ ] 5.5 Update `docs/results.md` with the measured fertility and WER numbers,
      copied verbatim from the JSON artifacts per the repo's results convention.
