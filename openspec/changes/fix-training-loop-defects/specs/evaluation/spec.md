## ADDED Requirements

### Requirement: In-loop training metric is self-describing and gate-comparable
The metric reported during training to steer a run SHALL record, alongside its
value, the evaluation split it was measured on, the sample count, and the name
of the parameter view (optimizer iterate) it describes. It SHALL be measured on
the same split as the phase gate it is used to predict, or the results record
SHALL state explicitly that it is not comparable to that gate.

A tuning decision SHALL NOT be recorded as evidence in configuration unless the
metric it rests on satisfies this requirement, so that a recorded rationale can
always be traced to a reproducible measurement.

#### Scenario: In-loop metric names its provenance
- **WHEN** the training loop reports an in-loop evaluation result
- **THEN** the recorded value carries the split, sample count and iterate name,
  and is written to the run record rather than only to the training log

#### Scenario: In-loop metric is comparable to its gate
- **WHEN** the in-loop metric is measured on a different split than the phase
  gate it predicts
- **THEN** the results record states the divergence and the gate threshold is
  not applied to the in-loop value

#### Scenario: Recorded measurement is reproducible
- **WHEN** a fertility, WER or CER number is written into documentation or
  configuration rationale
- **THEN** a JSON artifact under `results/` records the same number together
  with enough parameters to re-run the measurement, per the repository's
  results-are-the-record convention
