# board-deployment Specification

## Purpose

Covers transferring validated `.ort` artifacts to the Arduino UNO Q 4 GB Linux side (Qualcomm QRB2210, Debian), running streaming inference there, and measuring whether latency and RAM meet the voice-agent budget.

## Requirements

### Requirement: Artifact transfer and layout
The deployment step SHALL copy the validated `.ort` release bundle (models, tokenizer assets, checksum manifest) to a documented location on the UNO Q's Linux filesystem and SHALL verify checksums on the board after transfer, refusing to proceed on mismatch.

#### Scenario: Transfer integrity
- **WHEN** artifacts are copied to the board and the on-board checksum step runs
- **THEN** all checksums match the manifest and deployment continues, or the step fails naming the mismatched file

### Requirement: Streaming inference on board
The board SHALL run streaming transcription over the deployed artifacts using the Moonshine Voice runtime (`.ort` models only), emitting line-level events as speech arrives, driven by 16 kHz mono PCM input from file playback or microphone capture.

#### Scenario: Live transcription smoke
- **WHEN** a prepared Italian test clip is streamed through the on-board runtime
- **THEN** line events are emitted while audio is still arriving and the final line matches the eval harness's streaming-mode transcript for the same clip

### Requirement: Latency and RAM measurement
The deployment step SHALL measure and record, on the board: re-decode latency distribution with speculative decoding enabled and disabled, end-to-end line latency from speech pause to final text, and peak RSS of the streaming runtime with the small model loaded. Results SHALL be compared against the configured voice-agent budget for the UNO Q.

#### Scenario: Budget comparison report
- **WHEN** the measurement step completes
- **THEN** a report shows measured latency and RAM against the configured budget and marks each metric pass or fail

#### Scenario: Fallback trigger
- **WHEN** the small model's measured re-decode cadence exceeds the configured budget
- **THEN** the report names the documented fallback (exporting a streaming-tiny Italian variant through the same pipeline) rather than leaving the failure open-ended
