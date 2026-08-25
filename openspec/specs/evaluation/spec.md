# evaluation Specification

## Purpose

Defines how the Italian model's accuracy is measured: full-utterance WER/CER, chunked streaming simulation consistent with on-device inference, pass/fail gates per pipeline phase, and recorded results.

## Requirements

### Requirement: Full-utterance evaluation
The evaluation harness SHALL compute Word Error Rate and Character Error Rate with jiwer on the held-out FLEURS Italian test split for any checkpoint, and SHALL write a JSON results file containing the model identifier, dataset split, sample count, WER, and CER.

#### Scenario: Baseline before fine-tuning
- **WHEN** the un-fine-tuned English streaming-small checkpoint is evaluated on FLEURS Italian
- **THEN** a results file is produced that serves as the recorded baseline for gate comparisons

### Requirement: Streaming simulation evaluation
The harness SHALL additionally evaluate checkpoints in a chunked streaming mode that feeds audio incrementally (configurable hop of 32–100 ms) through the same chunked decode path used at runtime, with speculative decoding enabled, and SHALL report streaming WER/CER alongside a measured wall-clock latency per re-decode and an overall real-time factor.

#### Scenario: Streaming degradation is quantified
- **WHEN** a fine-tuned checkpoint is evaluated in both full-utterance and streaming modes
- **THEN** both result sets are written and the report shows the WER delta between the two modes

#### Scenario: Hallucination guard honored
- **WHEN** a streaming decode emits an abnormally high token rate for the audio duration
- **THEN** the harness terminates that utterance's decode the same way the on-device runtime would, and reflects the truncation in the score

### Requirement: Phase gates
Each pipeline phase that produces a model artifact SHALL have a configurable evaluation gate (a maximum allowed WER on the smoke or final eval split). A gate failure SHALL fail the phase with a clear comparison of measured versus allowed values, so a weak model cannot silently flow into export or deployment.

#### Scenario: Smoke gate
- **WHEN** the smoke-trained checkpoint's full-utterance WER on its eval slice exceeds the configured smoke gate
- **THEN** the smoke phase fails and downstream targets refuse to run

#### Scenario: Export gate comparison
- **WHEN** evaluation runs before export with a configured final gate
- **THEN** export proceeds only if measured WER is at or below the gate, otherwise export is blocked
