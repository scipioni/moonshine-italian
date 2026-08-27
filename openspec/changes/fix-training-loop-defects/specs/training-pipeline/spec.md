## ADDED Requirements

### Requirement: Reported metric and saved checkpoint describe the same weights
The training loop SHALL ensure that the weights scored by an in-loop evaluation
and the weights written to the checkpoint at that step are the same weights.
Where the optimizer maintains more than one view of the parameters (the
schedule-free averaged iterate and the raw iterate), the pipeline SHALL name
which view it uses, use that same view for evaluation, checkpointing,
best-checkpoint ranking, export and release, and record the name in the
checkpoint and in the results JSON.

A checkpoint SHALL NOT be promoted to `checkpoint-best` on the basis of a metric
measured on a different view of the parameters than the one it stores.

#### Scenario: Metric and artifact agree
- **WHEN** an in-loop evaluation runs and a checkpoint is saved at the same step
- **THEN** re-evaluating the saved checkpoint offline reproduces the recorded
  metric within evaluation tolerance, and the recorded metric names the iterate
  it describes

#### Scenario: Averaged iterate is rejected while it is sustainedly worse than the starting point
- **WHEN** the configured iterate scores worse on held-out Italian than the
  un-fine-tuned base checkpoint it was initialized from, for several
  consecutive in-loop evaluations in a row
- **THEN** training fails loudly with both measured values rather than
  continuing to report progress against it, because a progress metric that
  stays below the initialization cannot order checkpoints

#### Scenario: A single noisy regression does not abort training
- **WHEN** exactly one in-loop evaluation scores worse than the starting
  baseline, immediately followed by an evaluation that does not
- **THEN** training continues -- a model that has barely moved from its
  initialization is statistically indistinguishable from the baseline on a
  small sample, and a zero-tolerance single-eval check would abort a healthy
  run on ordinary early noise

#### Scenario: Resume preserves iterate identity
- **WHEN** training resumes from a checkpoint
- **THEN** the loaded weights and the loaded optimizer state agree on which
  iterate the parameters currently hold, and training refuses to start if they
  do not

### Requirement: Configured augmentation must execute
Any augmentation declared enabled in configuration SHALL either take effect on
the configured fraction of samples or fail loudly at startup. A configured
augmentation that silently applies to no samples SHALL be treated as a
configuration error, not as a no-op default.

#### Scenario: Enabled augmentation reaches the model
- **WHEN** training starts with chunked augmentation enabled at a non-zero
  probability
- **THEN** the loop records the measured fraction of samples actually augmented
  over a startup sample, and fails if that fraction is zero while the
  configured probability is not

#### Scenario: Degenerate chunk request is rejected
- **WHEN** chunk planning is asked to split a span that offers no admissible
  cut point
- **THEN** it reports the refusal to its caller distinguishably from "no split
  was needed", so a caller cannot mistake an impossible split for a skipped one

### Requirement: Curriculum stages must be effective against the prepared corpus
Each curriculum stage SHALL be validated against the prepared manifests at
training start. A stage whose constraint excludes no rows that the previous
stage included SHALL be reported as ineffective, and the pipeline SHALL fail
rather than run a curriculum whose stages are indistinguishable, so that a
recorded stage boundary always corresponds to a real change in the training
distribution.

#### Scenario: Stage bound exceeds the corpus maximum
- **WHEN** a curriculum stage's maximum audio duration is greater than or equal
  to the longest duration present in the prepared training manifests, and the
  preceding stage's bound is also greater than or equal to it
- **THEN** training fails naming the ineffective stage, its bound, and the
  corpus maximum

#### Scenario: Effective curriculum reports its staging
- **WHEN** training starts with a validated curriculum
- **THEN** it records, per stage, the bound and the resulting row count, so the
  training distribution at any step is recoverable from the run record

### Requirement: Step budget is expressed independently of gradient accumulation
The configured training length SHALL denote a fixed amount of training data
regardless of the gradient-accumulation factor. Changing gradient accumulation
SHALL NOT silently change how many epochs a run performs, and the run record
SHALL state the resulting sample count, epoch count and effective batch size.

#### Scenario: Accumulation change preserves the data budget
- **WHEN** the gradient-accumulation factor is changed and training is run with
  an otherwise identical configuration
- **THEN** the number of training samples consumed over the run is unchanged,
  and the run record reports the same epoch count

#### Scenario: Run record states the derived budget
- **WHEN** a training run starts
- **THEN** it records the effective batch size, the total samples it will
  consume, and the resulting epoch count over the configured dataset mix
