## ADDED Requirements

### Requirement: Custom Italian BPE Tokenizer training
The dataset preparation stage SHALL optionally train a custom byte-level BPE Italian tokenizer of a configured vocabulary size (e.g., 12,288 tokens) from the combined transcripts of the prepared datasets. The resulting `tokenizer.json` and associated processor configuration SHALL be saved to the training results output directory to serve as the new vocabulary foundation for training and inference.

#### Scenario: Custom tokenizer generation
- **WHEN** dataset preparation runs with custom tokenizer training enabled in the configuration
- **THEN** it concatenates all prepared transcripts, trains a byte-level BPE tokenizer of the specified vocabulary size, and saves `tokenizer.json` and processor configurations to the output directory

## MODIFIED Requirements

### Requirement: Streaming checkpoint fine-tuning
The pipeline SHALL load `moonshine-ai/moonshine-streaming-small` safetensors. If a custom tokenizer is configured, the pipeline SHALL automatically resize the model's text embedding (`embed_tokens`) and projection (`lm_head`) layers to match the new vocabulary size, freeze the audio encoder weights for a configured number of initial warm-up steps to quickly align the new vocabulary embeddings, and then unfreeze the full model for curriculum-staged fine-tuning. The training loop SHALL save resumable checkpoints plus TensorBoard logs under a profile-specific output directory. The smoke profile SHALL complete the identical code path as the final profile, differing only in dataset slice size and step counts, so that a successful smoke run validates the full training mechanics.

#### Scenario: Smoke training run
- **WHEN** training runs with the smoke profile on the FLEURS Italian slice
- **THEN** it completes without error, saves at least one checkpoint that can be reloaded, and records loss curves viewable in TensorBoard

#### Scenario: Checkpoint resume
- **WHEN** training is interrupted and restarted from an existing checkpoint
- **THEN** it resumes from the recorded step without replaying already-completed optimizer steps

#### Scenario: Vocabulary transplantation and embedding warm-up
- **WHEN** final training starts with a custom Italian tokenizer and warm-up steps configured
- **THEN** the model's text embedding and projection layers are resized to the custom tokenizer's vocabulary size, the audio encoder is frozen during warm-up steps, and then fully unfrozen for subsequent steps
