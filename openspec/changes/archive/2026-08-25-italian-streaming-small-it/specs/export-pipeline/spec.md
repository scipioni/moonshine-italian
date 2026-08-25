# export-pipeline

## Purpose

Covers conversion of fine-tuned checkpoints into the runtime format the Moonshine Voice engine accepts: ONNX graphs with KV-cache inputs/outputs for streaming, INT8 quantization, `.ort` FlatBuffer serialization, and artifact validation.

## ADDED Requirements

### Requirement: ONNX export with KV-cache support
The export step SHALL produce separate encoder and decoder ONNX graphs from a fine-tuned checkpoint, where the decoder exposes past/present key-value cache inputs and outputs so the runtime can decode incrementally, and where graph outputs on a reference input match the PyTorch checkpoint within a configured numerical tolerance.

#### Scenario: Graph parity
- **WHEN** the exported ONNX graphs and the source PyTorch checkpoint are run on the same reference audio
- **THEN** the decoded transcripts are identical and logit/encoding differences stay within the configured tolerance

#### Scenario: Decoder cache interfaces exist
- **WHEN** the exported decoder graph is inspected
- **THEN** it declares past and present key-value tensors for every cached layer, matching the upstream streaming graph interface

### Requirement: INT8 quantization and .ort serialization
The export step SHALL quantize the graphs to INT8, convert them to ONNX Runtime `.ort` FlatBuffers using the upstream conversion approach, and record the weight-size reduction. The runtime SHALL reject the intermediate `.onnx` artifacts and accept only the `.ort` outputs, mirroring the engine's own format enforcement.

#### Scenario: Size reduction recorded
- **WHEN** quantization and serialization complete
- **THEN** the artifact report lists pre- and post-quantization sizes and shows the expected order-of-magnitude reduction for the small architecture

#### Scenario: Only .ort is consumed downstream
- **WHEN** a downstream load or deployment step is given the `.onnx` intermediates instead of `.ort` files
- **THEN** it fails with an error naming the `.ort` artifacts to use

### Requirement: Artifact validation before release
Every export SHALL produce a validation record containing file names, sizes, and checksums of the `.ort` artifacts, plus a smoke load proving each artifact opens in an ONNX Runtime session on the training machine. Artifacts failing validation SHALL NOT be promoted to the release/board-deploy directories.

#### Scenario: Checksummed release bundle
- **WHEN** export succeeds
- **THEN** the release directory contains the `.ort` files, the tokenizer assets, and a manifest whose checksums match the actual files

#### Scenario: Corrupt artifact blocked
- **WHEN** an `.ort` file is truncated or corrupted before promotion
- **THEN** the smoke load or checksum comparison fails and the artifact is not promoted
