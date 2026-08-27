## 1. Custom Tokenizer Generation & Configuration

- [ ] 1.1 Concatenate training transcripts from MLS and Common Voice to build a unified Italian text corpus.
- [ ] 1.2 Implement a Hugging Face Tokenizers training script inside `src/moonshine_it/prepare.py` to train a 12,288-token Italian byte-level BPE tokenizer.
- [ ] 1.3 Add custom tokenizer configuration settings to `config.yaml` under `training` (e.g. `custom_tokenizer: true`, `vocab_size: 12288`, `warmup_steps: 5000`).

## 2. Model Resizing & Weight Initialization

- [ ] 2.1 Update `src/moonshine_it/model_io.py` to detect a custom Italian tokenizer and invoke `model.resize_token_embeddings()` upon model instantiation.
- [ ] 2.2 Implement safe normal-distribution weight initialization for newly added embedding and logit-projection parameters.

## 3. Warm-up Training Phase (Stage-Based Freezing)

- [ ] 3.1 Write a helper in `src/moonshine_it/train_loop.py` to freeze/unfreeze the audio encoder weights (`model.model.encoder`).
- [ ] 3.2 Integrate Stage A (warm-up) into the active step loop, keeping the encoder frozen for the first N `warmup_steps` so that only text embeddings are optimized.
- [ ] 3.3 Integrate Stage B (full fine-tuning) by unfreezing the encoder weights dynamically as soon as the step count exceeds the warm-up threshold.

## 4. Evaluation & Export Pipeline Updates

- [ ] 4.1 Update `src/moonshine_it/evaluate.py` and `src/moonshine_it/evaluate_cli.py` to support loading custom tokenizers and resized models during validation.
- [ ] 4.2 Adjust the ONNX export script `src/moonshine_it/export.py` to dynamically resolve vocabulary-dimension layers for encoder, decoder, and key-value cache exports.
