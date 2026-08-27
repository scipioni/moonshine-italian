## Context

`moonshine-streaming-small` was pre-trained primarily on English with a BPE
vocabulary of 32,768 embedding rows (32,000 tokenizer entries). Its merges
encode English statistics, so Italian fragments into more, less meaningful
pieces than English does. This is an efficiency problem, not a representability
problem — measured over all 17,415 prepared Italian validation transcripts, the
existing tokenizer round-trips exactly with zero `<unk>`, and Italian accented
characters are single in-vocab tokens.

Measured fertility on held-out Italian validation text (candidate tokenizers
trained on the actual prepared corpus: 285,401 lines / 21.3M chars of
`mls` + `common_voice` + `fleurs`):

| tokenizer | tok/word | chars/tok | vs current |
|---|---|---|---|
| current English BPE (32,768) | 1.738 | 3.37 | — |
| Italian byte-BPE 8,192 | 1.374 | 4.27 | −21.0% |
| **Italian byte-BPE 12,288** | **1.295** | **4.53** | **−25.5%** |
| Italian byte-BPE 16,384 | 1.247 | 4.70 | −28.3% |

Swapping in a custom Italian tokenizer requires re-aligning the model's text
decoder embedding and output projection without disrupting the pre-trained
audio encoder.

**What this does and does not address.** A word-level error analysis of
`checkpoint-best` (40 FLEURS-it utterances, WER 83.47% / CER 25.46%) found
H=199 S=489 D=38 I=79, and only **31% of substitutions within 2 character
edits** of the reference. The majority are unrelated words, often English or
Spanish (`sovrintendente → the`, `richiesto → estudindo`, `è → que`,
`di → ser`). That is an under-converged model, not correct acoustics defeated
by spelling. This change lowers the achievable WER ceiling and shortens
decoding; it is not a remedy for a training run whose WER does not respond.

## Goals / Non-Goals

**Goals:**
- Train a 12,288-token custom Italian byte-level BPE tokenizer from the prepared
  Italian corpus (`mls` + `common_voice` + `fleurs`).
- Resize the text embedding and output projection of `moonshine-streaming-small`
  to the new vocabulary size, with explicit re-initialization of both matrices.
- Implement a two-stage training loop:
  1. **Acoustic-Text Warm-up:** freeze the audio encoder, train only the new
     text embedding/projection to align Italian subwords to pre-trained
     acoustic features.
  2. **Full Fine-tuning:** unfreeze and fine-tune the whole model.
- Carry the custom tokenizer through evaluation, ONNX export, release packaging,
  and the on-board runtime.
- Gate the change on measured fertility and a WER non-regression.

**Non-Goals:**
- Train a multilingual or monolingual model from scratch.
- Alter the audio encoder's structure or downsampling behavior.
- Fix training convergence. This change assumes convergence is already
  demonstrated; see Sequencing.

## Decisions

### 1. Tokenizer Choice: Byte-Level BPE (12,288 Vocabulary Size)
- **Decision:** Train a byte-level BPE tokenizer with vocabulary size 12,288.
- **Rationale:** Measured (table above) to capture 25.5% of the ~28% fertility
  gain available at 16,384, while keeping the embedding light. Vocabulary-tied
  parameters fall 33.6M → 12.6M; total model 140.1M → 119.2M, which also helps
  the UNO Q RSS budget.
- **Alternatives Considered:** 8,192 (leaves ~4.5 points of fertility on the
  table); 16,384 (only 2.8 points better than 12,288 for 33% more embedding
  parameters).
- **Unverified claim:** earlier drafts justified 12,288 as "matching Moonshine's
  official Spanish monolingual model layout." That has not been confirmed and
  is not load-bearing — the choice stands on the measurements above.

### 2. Resizing: `resize_token_embeddings` is necessary but NOT sufficient
- **Decision:** Call `model.resize_token_embeddings(12288)` at load time, then
  **explicitly re-initialize both the input embedding and the output projection.**
- **Verified mechanics:** on this model `resize_token_embeddings(12288)`
  correctly resizes `model.decoder.embed_tokens` *and* the output projection,
  and updates `config.vocab_size` (32768 → 12288). `tie_word_embeddings` is
  `False`, so the two matrices are independent.
- **Layer naming:** the output projection on this architecture is **`proj_out`**,
  not `lm_head`. The resize works because the model implements
  `get_output_embeddings()`; any code addressing `lm_head` directly will fail.
- **The trap (this is why explicit re-init is required):** 32,768 → 12,288 is a
  *shrink*. HuggingFace **truncates**, keeping the first 12,288 pre-trained rows
  verbatim (verified: rows 0 and 12,287 are bit-identical before/after). No rows
  are "newly added", so an init routine that targets newly-added parameters is a
  no-op. Left as-is, new Italian token ID *n* silently inherits English token
  *n*'s embedding — misleading structure that is arguably worse than a clean
  random init.
- **Consequence:** the transplant deliberately discards 33.6M pre-trained
  parameters (24% of the model). This is the real cost of the change and the
  reason the warm-up stage exists.

### 3. Resize must precede optimizer construction
- **Decision:** Perform the resize in `model_io.py` at model-load time, before
  `train_loop.py` builds the optimizer.
- **Rationale:** `resize_token_embeddings` replaces the embedding/projection
  **parameter tensors**. An optimizer constructed over the pre-resize parameters
  would hold references to discarded tensors and its `step()` would silently
  no-op for them — the exact failure class already fixed once in this repo
  (see the optimizer/model parameter-identity guard in `train_loop.py`, which
  raises rather than training silently). That guard now backstops this ordering.

### 4. Frozen-Encoder Warm-up Training (5,000 steps)
- **Decision:** For the first 5,000 steps (configurable), freeze `model.encoder`
  (and its projection) via `p.requires_grad = False`.
- **Rationale:** freshly re-initialized text embeddings emit large, uninformative
  gradients. Freezing the encoder forces the decoder to align to stable
  pre-trained acoustic features before the encoder is exposed to any update.

### 5. Schedule-free AdamW interaction with freeze/unfreeze
- **Decision:** Unfreezing at the Stage A→B boundary must be treated as an
  optimizer-state event, not just a `requires_grad` flip.
- **Rationale:** `AdamWScheduleFree` maintains a per-parameter `z` buffer created
  on that parameter's first `step()`, and its `train()`/`eval()` methods
  transform `p.data` only for parameters that already have `z`. Frozen encoder
  parameters therefore have **no** `z` during Stage A and are skipped by the
  eval/train iterate swap; they begin acquiring `z` only after unfreezing. This
  must not be allowed to desynchronize which iterate ("x" vs "y") different
  parameter groups hold — the repo has already been bitten once by an
  x/y-iterate mismatch between eval and checkpointing. Stage A checkpoints and
  the Stage A→B transition need explicit verification that all parameters agree
  on iterate state.

## Risks / Trade-offs

- **[Risk] 24% of pre-trained parameters are discarded.** Unavoidable given a
  vocabulary swap; this is the change's core cost.
  - **Mitigation:** frozen-encoder warm-up re-aligns the new embeddings against
    intact acoustic features before any encoder update.
- **[Risk] Shrink-truncation silently produces meaningless embeddings.**
  - **Mitigation:** Decision 2 mandates explicit re-initialization of both
    matrices, with a task-level assertion that the resized matrices are not
    bitwise-equal to the retained pre-trained rows.
- **[Risk] Result is unattributable if convergence is still broken.**
  - **Mitigation:** Sequencing (below) plus the outcome gate.
- **[Risk] Relative gates are calibrated against the old vocabulary.** The
  `smoke` and `post_quant` gates compare against baselines measured with the
  32,768 vocabulary.
  - **Mitigation:** re-record the spike baseline after the transplant and before
    interpreting any gate result.
- **[Risk] Increased total step budget from the warm-up stage.**
  - **Mitigation:** accepted; warm-up steps are configurable.

## Sequencing

This change is **blocked on demonstrated training convergence.** At time of
writing, final training has not improved on `checkpoint-2000` (eval_wer 102.7%)
across ~70,000 further steps, two clean restarts, dozens of shuffle seeds, and
three separate fixes (resume-optimizer binding, checkpoint/eval-iterate
mismatch, gradient accumulation). Transplanting the vocabulary into that run
would destroy 33.6M pre-trained parameters while leaving the outcome
uninterpretable. Land convergence first; then apply this change and read the
outcome gate.
