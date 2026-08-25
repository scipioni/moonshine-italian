# task-orchestration

## Purpose

Defines the Taskfile interface that sequences every pipeline phase, gates phases on their inputs, supports hardware profiles, and provides a single-command smoke run of the entire process.

## ADDED Requirements

### Requirement: Phase targets with input gating
The Taskfile SHALL expose targets for downloading models and datasets, preparing data, running the gradient/tokenizer spike, training, evaluating, exporting to ONNX, converting to `.ort`, validating artifacts, and running board deployment — each of which (except download) SHALL verify its upstream inputs exist and fail fast with a pointer to the target that produces them.

#### Scenario: Training without prepared data
- **WHEN** the train target runs and the prepared-dataset manifest is missing
- **THEN** it fails immediately with a message naming the prepare target, without loading the model

#### Scenario: Chained full run
- **WHEN** the full pipeline target runs from a clean data directory
- **THEN** phases execute in dependency order and each phase's outputs appear where the next phase expects them

### Requirement: Hardware profiles
The Taskfile SHALL support hardware profiles (`rocm12g`, `strix`, `cuda`) selectable by variable, each setting appropriate batch sizes, precision, thread counts, and accelerator provider selection read from `config.yaml`. An unset or unrecognized profile SHALL fail with the list of valid profiles rather than silently using defaults.

#### Scenario: Profile selects batch size
- **WHEN** training runs with profile `rocm12g` versus `strix`
- **THEN** the effective batch size and precision for each match the values configured for those profiles in `config.yaml`

#### Scenario: Invalid profile rejected
- **WHEN** a target runs with profile `foo`
- **THEN** it fails listing `rocm12g`, `strix`, and `cuda` as valid choices

### Requirement: Single-command smoke run
A single target SHALL execute the complete process end-to-end on a small dataset slice: download the base model and smoke dataset, run the spike, train briefly, evaluate against the smoke gate, export, quantize, serialize to `.ort`, and validate — producing a pass/fail result and a summary of every phase's outcome.

#### Scenario: Smoke passes on a fresh clone
- **WHEN** the smoke target runs on a machine with prerequisites installed and no cached data
- **THEN** it completes all phases, prints a per-phase summary, and exits zero

#### Scenario: Smoke failure isolates the phase
- **WHEN** any smoke phase fails
- **THEN** the run stops at that phase, reports which phase failed and why, and exits non-zero

### Requirement: Process validation before final training
The smoke target SHALL be the documented and enforced precondition for launching the final training run: the final-train target SHALL require a recorded successful smoke result, so the multi-day Strix Halo run is never started against an unvalidated process.

#### Scenario: Final training blocked without smoke record
- **WHEN** the final-train target runs and no successful smoke record exists
- **THEN** it refuses to start and instructs the user to run the smoke target first
