# Tasks: italian-streaming-small-it

## 1. Environment & Configuration

- [x] 1.1 Create `.env.example` (HF_TOKEN placeholder), git-ignored `.env` entry, and `.gitignore` for `.venv/ data/ results/ artifacts/ .env`; verify `git status` stays clean with a populated `.env`
- [x] 1.2 Author `config.yaml` with smoke/final training profiles, hardware profiles (`rocm12g`, `strix`, `cuda`), dataset selections, eval gates (smoke + placeholder final), and board budget; verify a config-loader round-trips every key used by later tasks
- [x] 1.3 Set up `uv`-managed `.venv` with `--system-site-packages` and lock pure-Python deps (transformers, datasets, jiwer, schedulefree, tensorboard, etc.) in `pyproject.toml`/`uv.lock`; verify `uv run python -c "import torch"` resolves the system ROCm/CUDA build
- [x] 1.4 Implement `task env-check` (accelerator kind, VRAM, GPU torch + ORT import, profile validation); verify it reports ROCm + VRAM on the 12 GB PC and fails with pacman guidance when GPU packages are absent

## 2. Download & Data Preparation

- [x] 2.1 Implement `task download-model` (moonshine-ai/moonshine-streaming-small safetensors + tokenizer assets, checksummed); verify re-run is a no-op with matching checksums
- [x] 2.2 Implement `task download-data`: FLEURS-it and MLS-it (public), Common Voice it behind `HF_TOKEN`; verify missing-token case fails with a pointer to `.env.example` before download starts
- [x] 2.3 Implement audio preparation (16 kHz mono, VAD-aware segmentation 1–10 s, per-chunk aligned transcripts) and Italian text normalization (accents preserved, casing/number rules); verify manifest durations all within bounds and normalization unit tests pass on accented/apostrophe/number fixtures
- [x] 2.4 Implement smoke slicing (configurable FLEURS-it subset with fixed seed); verify two preparations produce identical manifests and checksums

## 3. Spike (Gate for Everything)

- [x] 3.1 Spike 1 — forward+backward pass through the streaming-small checkpoint on a prepared batch; verify finite loss and non-trivial gradient norms, recorded to `results/spike/`
- [x] 3.2 Spike 2 — tokenizer Italian round-trip (à è é ì ò ù, apostrophes) plus vocab coverage stats; verify lossless round-trip after normalization or record failure + sized embedding-extension plan
- [x] 3.3 Spike 3 — baseline dual-mode eval (full-utterance + chunked streaming) of the un-tuned English checkpoint on FLEURS-it; verify baseline JSON lands in `results/eval/` and calibrates harness runtime
- [x] 3.4 Implement the fallback-latch: final training refuses to run without a spike record; failure paths require explicit base selection in `config.yaml`; verify refusal by deleting the spike record and invoking the gate

## 4. Training

- [x] 4.1 Implement the profile-aware training loop (schedule-free AdamW, curriculum stages, chunked-augmentation regularizer, checkpointing, TensorBoard) sharing one code path for smoke/final; verify a 20-step dry-run on the smoke slice saves reloadable checkpoints and logs
- [x] 4.2 Implement checkpoint resume; verify an interrupted run resumes at the recorded step without replaying optimizer state
- [x] 4.3 Wire hardware profiles to batch size / precision / workers; verify `rocm12g` vs `strix` produce the configured values in run metadata
- [x] 4.4 Run smoke training to completion on the 12 GB PC; verify loss curve decreases and best checkpoint passes the smoke eval gate

## 5. Evaluation

- [x] 5.1 Implement full-utterance WER/CER eval (jiwer) writing model/dataset/sample-count/WER/CER JSON; verify output on the baseline and smoke checkpoints
- [x] 5.2 Implement chunked streaming simulation (32–100 ms hops, speculative decoding, `max_tokens_per_second` guard) reporting streaming WER/CER + per-re-decode latency + RTF; verify streaming-vs-full delta is reported and truncation behavior matches the runtime heuristic
- [x] 5.3 Implement phase gates (measured vs configured, fail blocks downstream); verify export target refuses to run when a gate is exceeded

## 6. Export & Release

- [x] 6.1 Implement ONNX export (encoder + KV-cached decoder graphs); verify parity vs PyTorch on reference audio within configured tolerance and presence of past/present KV tensors
- [x] 6.2 Vendor upstream `convert-models-to-ort.py` (pinned) and implement INT8 quantization + `.ort` serialization; verify size-reduction report and that `.onnx` intermediates are rejected downstream
- [x] 6.3 Implement validation + release promotion (checksums, sizes, ORT smoke-load, manifest); verify a corrupted artifact fails promotion and a clean bundle lands in `artifacts/release/`
- [x] 6.4 Evaluate the INT8 `.ort` bundle in streaming mode; verify post-quantization WER delta is recorded and gate-checked before promotion

## 7. Orchestration

- [x] 7.1 Author `Taskfile.yml` with all phase targets, input gating (fail-fast pointing at producing target), and profile selection; verify train-without-prepare and invalid-profile failures behave per spec
- [x] 7.2 Implement `task smoke` chained run (download → spike → train → eval → export → ort → validate) with per-phase summary; verify fresh-clone pass on the 12 GB PC and that a forced phase failure stops the chain non-zero
- [x] 7.3 Enforce smoke record as precondition for `final-train`; verify final-train refuses without a recorded smoke success

## 8. Board Deployment (UNO Q 4 GB)

- [x] 8.1 Document and script artifact transfer to the board (scp + on-board checksum verify); verify mismatched checksum aborts deployment
- [x] 8.2 Bring up the Moonshine Voice runtime on the board's Debian side (wheel or C++ build — record which); verify `.ort` models load and a line event streams from a test clip
- [x] 8.3 Implement on-board measurement (re-decode latency speculative on/off, pause→text latency, peak RSS) with budget report; verify pass/fail per metric against `config.yaml` budget
- [x] 8.4 If small misses budget: run the tiny-variant fallback export path; verify report names the fallback and, if exercised, a `streaming-tiny-it` bundle passes the budget

## 9. Documentation

- [x] 9.1 Write `docs/` guides: environment (pacman + uv per platform), data preparation, training (smoke + final), evaluation methodology, export/quantization, board deployment; verify each guide's commands match the Taskfile targets
- [x] 9.2 Write concise `README.md` (what/why, quickstart `task smoke`, task map, board notes, doc links); verify a newcomer can go from clone to smoke run using only the README
- [ ] 9.3 Record final results (WER/CER full + streaming, quantization delta, board measurements) in `docs/results.md`; verify numbers match the JSON artifacts in `results/`

## 10. Final Training & Release (Strix Halo)

- [x] 10.1 Run full data preparation (MLS-it complete + CV-it mix) on Strix Halo; verify manifest stats and record usable-hours answer to the design's open question
- [ ] 10.2 Launch final training with the `strix` profile; verify curriculum completes, eval gates pass, and best checkpoint is promoted
- [ ] 10.3 Produce and validate the final `.ort` release bundle; verify streaming-mode gate passes post-quantization and manifest checksums are complete
- [ ] 10.4 Deploy final bundle to the UNO Q, run the measurement suite, and record budget verdict; verify `docs/results.md` updated with final numbers
