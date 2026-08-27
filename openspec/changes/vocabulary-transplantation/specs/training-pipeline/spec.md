## ADDED Requirements

### Requirement: Custom Italian BPE Tokenizer training
The dataset preparation stage SHALL optionally train a custom byte-level BPE
Italian tokenizer of a configured vocabulary size (default 12,288) from the
combined transcripts of the prepared datasets. The resulting `tokenizer.json`
and associated processor configuration SHALL be saved to the training results
output directory to serve as the new vocabulary foundation for training and
inference.

#### Scenario: Custom tokenizer generation
- **WHEN** dataset preparation runs with custom tokenizer training enabled in
  the configuration
- **THEN** it concatenates all prepared transcripts, trains a byte-level BPE
  tokenizer of the specified vocabulary size, and saves `tokenizer.json` and
  processor configurations to the output directory

### Requirement: Explicit re-initialization of resized vocabulary layers
When the model is resized to a custom vocabulary, the pipeline SHALL explicitly
re-initialize both the text embedding and the output projection. Resizing a
32,768-row vocabulary down to a smaller custom vocabulary **truncates** rather
than adding rows, so no parameters are "newly added" and default
newly-added-parameter initialization is a no-op. Without explicit
re-initialization, custom token ID *n* silently inherits the pre-trained
embedding of unrelated base-vocabulary token *n*.

#### Scenario: Resized layers are re-initialized, not truncated
- **WHEN** the model is loaded with a custom tokenizer configured
- **THEN** the text embedding and output projection are resized to the custom
  vocabulary size, `config.vocab_size` agrees with both, and neither matrix is
  bitwise-equal to the first *N* rows of the pre-trained matrices

### Requirement: Vocabulary resize precedes optimizer construction
The vocabulary resize SHALL occur before the optimizer is constructed. Resizing
replaces the embedding and projection parameter tensors; an optimizer built over
the pre-resize parameters would hold references to discarded tensors and its
updates for those parameters would be silently discarded.

#### Scenario: Optimizer owns the post-resize parameters
- **WHEN** training starts with a custom tokenizer configured
- **THEN** the optimizer's parameter set is identical to the model's post-resize
  parameter set, and training fails loudly rather than silently no-opping if it
  is not

### Requirement: Vocabulary transplantation outcome gate
The change SHALL record measured tokenizer fertility and a post-transplant WER
comparison, and SHALL fail if either regresses. Fertility SHALL be measured on
held-out Italian text and compared against the pre-transplant baseline; WER
SHALL be compared against the pre-transplant streaming WER on the same slice.
Mechanical success (layers resized, encoder frozen) SHALL NOT by itself
constitute completion of this change.

#### Scenario: Fertility target met
- **WHEN** the custom tokenizer is trained and evaluated on held-out Italian text
- **THEN** it records tokens/word and chars/token, and fails with
  measured-vs-required values if it does not beat the recorded pre-transplant
  baseline by the configured target

#### Scenario: WER does not regress
- **WHEN** evaluation runs on the transplanted model
- **THEN** streaming WER is compared against the recorded pre-transplant
  baseline, and the change fails with measured-vs-allowed values on regression

### Requirement: Tokenizer spike measures suitability, not only representability
The tokenizer spike SHALL measure vocabulary fertility on real prepared Italian
transcripts against a reference baseline and fail on a configured threshold.
Exact round-trip and absence of unknown tokens SHALL NOT alone be sufficient for
the spike to pass, since an English vocabulary satisfies both while remaining
substantially less efficient for Italian.

#### Scenario: Spike rejects a representable but inefficient vocabulary
- **WHEN** the tokenizer spike runs against a vocabulary that round-trips Italian
  losslessly but whose measured fertility on real prepared transcripts is worse
  than the configured threshold
- **THEN** the spike records the measured fertility and reports failure rather
  than passing on round-trip alone

## MODIFIED Requirements

### Requirement: Streaming checkpoint fine-tuning
The pipeline SHALL load `moonshine-ai/moonshine-streaming-small` safetensors. If
a custom tokenizer is configured, the pipeline SHALL resize the model's text
embedding and output projection to match the new vocabulary size, explicitly
re-initialize both, freeze the audio encoder weights for a configured number of
initial warm-up steps to align the new vocabulary embeddings, and then unfreeze
the full model for curriculum-staged fine-tuning. The training loop SHALL save
resumable checkpoints plus TensorBoard logs under a profile-specific output
directory. The smoke profile SHALL complete the identical code path as the final
profile, differing only in dataset slice size and step counts, so that a
successful smoke run validates the full training mechanics.

#### Scenario: Smoke training run
- **WHEN** training runs with the smoke profile on the FLEURS Italian slice
- **THEN** it completes without error, saves at least one checkpoint that can be
  reloaded, and records loss curves viewable in TensorBoard

#### Scenario: Checkpoint resume
- **WHEN** training is interrupted and restarted from an existing checkpoint
- **THEN** it resumes from the recorded step without replaying already-completed
  optimizer steps

#### Scenario: Vocabulary transplantation and embedding warm-up
- **WHEN** final training starts with a custom Italian tokenizer and warm-up
  steps configured
- **THEN** the model's text embedding and projection layers are resized to the
  custom tokenizer's vocabulary size and re-initialized, the audio encoder is
  frozen during warm-up steps, and then fully unfrozen for subsequent steps

#### Scenario: Resume across the warm-up stage boundary
- **WHEN** training is interrupted during the frozen-encoder warm-up stage, or
  after the transition to full fine-tuning, and then resumed
- **THEN** the correct freeze state is restored for the resumed step, and all
  parameters agree on which schedule-free iterate they hold
