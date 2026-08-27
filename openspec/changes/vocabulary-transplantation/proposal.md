## Why

The Moonshine Italian fine-tune inherits the base model's English BPE
vocabulary (32,768 embedding rows / 32,000 tokenizer entries). That vocabulary
is lossless for Italian — measured over all 17,415 prepared validation
transcripts: zero `<unk>`, exact round-trip, and accented characters
(`è ù à perché così più città`) are single in-vocab tokens — but it is
**inefficient** for Italian, because its merges were learned from English
statistics. Morphologically complex Italian words fragment badly
(`sovrintendente` → `so|vr|int|end|ente`, `ginocchia` → `g|in|oc|chia`) while
their English counterparts are single tokens.

Measured on held-out Italian validation text:

| tokenizer | tok/word | chars/tok |
|---|---|---|
| current English BPE (32,768) | 1.738 | 3.37 |
| Italian byte-BPE 8,192 | 1.374 | 4.27 |
| **Italian byte-BPE 12,288** | **1.295** | **4.53** |
| Italian byte-BPE 16,384 | 1.247 | 4.70 |

Training a native Italian vocabulary and transplanting it into the pre-trained
model therefore buys **~25% shorter target sequences** and a **smaller model**
(vocabulary-tied parameters drop 33.6M → 12.6M; total 140.1M → 119.2M, which
also helps the UNO Q budget).

**Scope of the expected benefit.** This is a sequence-length and model-size
win, not a fix for a model that has not learned Italian. A word-level error
analysis of `checkpoint-best` found only 31% of substitutions are near-misses
(≤2 character edits); the remaining 69% are unrelated words, frequently
English or Spanish (`sovrintendente → the`, `è → que`, `di → ser`). That is
the signature of an under-converged model, not of correct phonetics defeated
by spelling. Vocabulary transplantation should be expected to lower the WER
*ceiling* and speed decoding — it should not be expected, on its own, to move
a run whose WER is not responding to training.

**Sequencing.** Because this change discards 33.6M pre-trained parameters
(24% of the model) and forces the decoder to relearn its entire vocabulary
mapping, it should land only once final training demonstrably converges.
Applying it to a non-converging run makes the result unattributable in either
direction.

## What Changes

- **Custom Tokenizer Training:** Introduce a pipeline utility (`train_tokenizer.py`
  or within `prepare.py`) to build a 12,288-token byte-level BPE Italian
  tokenizer from the combined Italian transcription corpus.
- **Model Vocabulary Resizing:** Add support for detecting a custom tokenizer,
  resizing the model's text embedding and output projection to the new
  vocabulary size, and **explicitly re-initializing both matrices** (see
  design: resizing *down* truncates rather than adding new rows, so the
  default behavior silently maps Italian token IDs onto unrelated retained
  English embeddings).
- **Warm-up Training Phase:** Introduce a two-stage training strategy:
  1. **Stage A (Warm-up):** Freeze the audio encoder and fine-tune *only* the
     re-initialized text embedding and projection layers to align the
     acoustic representations to the new Italian subword tokens.
  2. **Stage B (Fine-tuning):** Unfreeze the entire model and proceed with the
     standard curriculum-staged fine-tuning.
- **Outcome Gate:** Record the measured tokenizer fertility and a post-transplant
  WER comparison, and fail the change if the transplant does not meet its
  stated fertility target or regresses WER against the pre-transplant baseline.
- **Evaluation, Export & Release Adjustments:** Update evaluation, ONNX export,
  release packaging, and the on-board runtime to carry the custom tokenizer and
  the new vocabulary dimension instead of assuming the base English BPE.

## Capabilities

### New Capabilities

*(None)*

### Modified Capabilities

- `training-pipeline`: Introduce vocabulary transplantation requirements —
  Italian BPE tokenizer generation, model embedding/projection resizing with
  explicit re-initialization, the frozen-encoder warm-up stage, and an
  outcome gate on fertility and WER.

## Impact

- `src/moonshine_it/prepare.py`: Italian corpus concatenation and BPE tokenizer training.
- `src/moonshine_it/model_io.py`: custom-tokenizer detection, resize, explicit re-init.
- `src/moonshine_it/train_loop.py`: encoder freeze/unfreeze staging; resize must
  occur before optimizer construction (see design).
- `config.yaml`: custom tokenizer settings (enable flag, vocab size, warm-up steps,
  fertility target).
- `src/moonshine_it/evaluate.py` / `evaluate_cli.py`: load the custom tokenizer.
- `src/moonshine_it/export.py` / `parity.py`: vocabulary-dimension-dependent graphs.
- `src/moonshine_it/release.py` / `ort_runtime.py`: package and load the custom
  `tokenizer.json` on-board.
- `src/moonshine_it/spike.py`: tokenizer spike must measure fertility, not only
  round-trip losslessness.
- Gate/baseline re-calibration: the `smoke` and `post_quant` gates are relative
  to a baseline measured with the old vocabulary.
