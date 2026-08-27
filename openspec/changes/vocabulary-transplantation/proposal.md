## Why

The current Moonshine Italian fine-tuning pipeline uses the default English Llama tokenizer. Because the vocabulary lacks native Italian subwords, the model is forced to spell out Italian phonetics character-by-character, leading to high exact-match Word Error Rates (WER) despite clear legibility. By building a custom Italian tokenizer and transplanting its vocabulary into the pre-trained Moonshine model (resizing embedding layers and performing a warm-up training phase), we can achieve native Italian decoding, correct spellings, and drastically reduce WER.

## What Changes

- **Custom Tokenizer Training:** Introduce a pipeline utility (`train_tokenizer.py` or within `prepare.py`) to build an optimized 12,288-token byte-level BPE Italian tokenizer from our combined Italian transcription corpus.
- **Model Vocabulary Resizing:** Add support in `train_loop.py` to automatically detect a custom tokenizer, resize the model's text embedding and projection layers (`embed_tokens` and `lm_head`), and initialize the new tokens safely.
- **Warm-up Training Phase:** Introduce a two-stage training strategy:
  1. **Stage A (Warm-up):** Freeze the audio encoder and fine-tune *only* the re-initialized text embedding and projection layers to quickly align the acoustic representations to the new Italian subword tokens.
  2. **Stage B (Fine-tuning):** Unfreeze the entire model and proceed with the standard curriculum-staged fine-tuning.
- **Evaluation Adjustments:** Update the evaluation and export code to load and package the custom tokenizer alongside the model instead of assuming the default English BPE.

## Capabilities

### New Capabilities

*(None)*

### Modified Capabilities

- `training-pipeline`: Introduce vocabulary transplantation requirements, including Italian BPE tokenizer generation, model embedding resizing, and the warm-up training stage (freezing the audio encoder during initial steps).

## Impact

- `src/moonshine_it/train_loop.py`: Loader, model-resizing, freezing/unfreezing logic, and optimization loop updates.
- `src/moonshine_it/prepare.py`: Integration of Italian corpus concatenation and BPE tokenizer training.
- `config.yaml`: Configuration options for the custom tokenizer (vocabulary size, path, warm-up steps).
- `src/moonshine_it/evaluate.py` / `src/moonshine_it/export.py`: Loading and exporting the custom tokenizer and model structure.
