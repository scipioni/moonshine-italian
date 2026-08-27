## Context

The Moonshine streaming-small model was pre-trained primarily on English, utilizing an English Llama-compatible tokenizer with 32,768 vocabulary entries. In our Italian fine-tuning pipeline, this tokenizer causes phonetic character-by-character approximations (e.g., `tanto donnie` instead of `tanto d'ogni`) because it lacks native Italian subwords. This keeps the exact-match Word Error Rate (WER) high, even though acoustic transcription is highly legible. Swapping in a custom-built Italian tokenizer is possible, but requires adapting and aligning the model's text decoder embedding and projection layers without disrupting the pre-trained audio encoder's representation.

## Goals / Non-Goals

**Goals:**
- Train an optimized 12,288-token custom Italian byte-level BPE tokenizer using our prepared Italian training transcription corpus (`mls` + `common_voice` + `fleurs`).
- Resize the text embedding and projection layers (`embed_tokens` and `lm_head`) of `moonshine-streaming-small` to match the new Italian tokenizer.
- Implement a two-stage training loop:
  1. **Acoustic-Text Warm-up:** Freeze the audio encoder and train *only* the new text embeddings to align Italian subwords to pre-trained acoustic features.
  2. **Full Fine-tuning:** Unfreeze the entire model and fine-tune both encoder and decoder.
- Ensure the custom tokenizer integrates seamlessly with evaluation and ONNX export.

**Non-Goals:**
- Train a multilingual or monolingual model from scratch (which requires massive compute and datasets).
- Alter the audio encoder's downsampling or structural downsampling behavior.

## Decisions

### 1. Tokenizer Choice: Byte-Level BPE (12,288 Vocabulary Size)
- **Decision:** Train a byte-level BPE tokenizer with a vocabulary size of 12,288, matching Moonshine's official Spanish monolingual model layout.
- **Rationale:** 12,288 is small enough to keep the embedding layer parameters light (faster decoding on the UNO Q), but large enough to cover all common Italian subwords, roots, prefixes, and suffixes.
- **Alternatives Considered:** 32,000 vocab size (rejected due to excessive memory usage and slower training steps on the edge device).

### 2. Model Vocabulary Resizing via `resize_token_embeddings`
- **Decision:** Automatically invoke `model.resize_token_embeddings(12288)` upon loading if a custom tokenizer is detected.
- **Rationale:** This is PyTorch and Hugging Face's standard, safe method to adjust the dimensions of `model.model.decoder.embed_tokens` and `model.lm_head` and safely initialize the new parameter weights.

### 3. Frozen-Encoder Warm-up Training (5,000 steps)
- **Decision:** In the initial 5,000 steps (configurable), freeze all parameters of the audio encoder by setting `p.requires_grad = False` on `model.model.encoder` and its projection layers.
- **Rationale:** If we update the entire model immediately, the raw/untrained Italian embeddings will propagate wild gradients backward, potentially destroying the encoder's carefully pre-trained acoustic representations. Freezing the encoder forces the decoder to quickly learn how to map the encoder's stable acoustic features to the new Italian subwords first.

## Risks / Trade-offs

- **[Risk]:** Mismatch between pre-trained acoustic features and randomized embedding weights on warm-up startup.
  - **Mitigation:** The frozen-encoder warm-up phase forces the embeddings to quickly align to pre-trained acoustic outputs before the encoder is ever exposed to parameter updates.
- **[Risk]:** Increased initial training step requirements.
  - **Mitigation:** Adding a warm-up phase extends the total final step budget slightly, but ensures overall training is stable and prevents NaN losses.
