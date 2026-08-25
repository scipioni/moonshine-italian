# Design: italian-streaming-small-it

## Context

Moonshine Voice (`moonshine-ai/moonshine`, v0.1.3) ships English-only streaming STT models. Verified facts that shape this design:

- Streaming checkpoints exist as safetensors on HF: `moonshine-ai/moonshine-streaming-small` (123M params, ~7.84% WER on English OpenASR). No Italian model exists in the catalog (any architecture). Language models follow the `moonshine-{arch}-{lang}` naming convention.
- The runtime is `.ort`-only (ONNX Runtime FlatBuffers) since v0.1.1; conversion tooling is `scripts/convert-models-to-ort.py` in the upstream repo. Streaming relies on KV-cached decoder graphs + speculative decoding (`use_speculative_decoding=true`, default).
- No public recipe exists for fine-tuning a *streaming* Moonshine checkpoint. The community toolkit `pierre-cheneau/finetune-moonshine-asr` (French, 21.8% WER) fine-tunes non-streaming `moonshine-tiny` via HF transformers + schedule-free AdamW + curriculum learning — its *methods* transfer, its streaming flags (as cited in `docs/gemini-deepresearch.md`) do not exist.
- Upstream v0.1.3 added `moonshine-voice[lora]` with `fit_adapter` (decoder-only LoRA, ATCOSIM example) — proven for domain adaptation, unproven for full language swap.
- Target board: Arduino UNO Q 4 GB — Qualcomm Dragonwing QRB2210 running Debian (arm64 Linux) + STM32U585 on Zephyr. Linux side runs the Moonshine Voice runtime. Upstream measured Small Streaming at 527 ms/re-decode on Raspberry Pi 5; QRB2210 CPU cores are weaker, so board latency is a first-class risk.
- Training hardware: AMD ROCm 12 GB PC (smoke/testing) and Strix Halo ROCm (final). Arch Linux native packages exist: `python-pytorch-rocm`/`-cuda` 2.13, `python-onnxruntime-rocm`/`-cuda` 1.28 (each provides/conflicts `python-onnxruntime`/`python-pytorch`).
- `docs/gemini-deepresearch.md` is directionally correct but contains confabulations (wrong HF repo names, nonexistent streaming flags in the community toolkit, wrong param counts for non-streaming models). This design follows verified facts only.

## Goals / Non-Goals

**Goals:**
- A reproducible, task-driven pipeline: download → prepare → spike → train → eval → export → `.ort` → board validation, runnable end-to-end on a small slice (smoke) before the multi-day final run.
- Italian fine-tune of `moonshine-streaming-small` with quantified streaming-mode accuracy (not just utterance-level WER).
- Artifacts deployable to the UNO Q 4 GB with measured latency/RAM against an explicit budget.
- Everything configured from `config.yaml`; secrets only in `.env`.

**Non-Goals:**
- Upstream PR to `moonshine-ai/moonshine` (catalog entry, enum registration, bindings) — explicitly deferred to a later change.
- STM32U585-side work (wake word, Moonshine Micro components) — later change.
- Training tiny/medium Italian variants (tiny exists only as a *fallback export* path if the board budget fails; medium is out of scope).
- Multilingual or code-switching support; Italian only.

## Decisions

### D1: Base checkpoint = `moonshine-ai/moonshine-streaming-small`, full fine-tune
User-selected. Alternatives: tiny-streaming (44M — faster iterate, lower ceiling), medium-streaming (300M — exceeds comfortable 12 GB smoke-training memory and board latency budget). Full fine-tune over LoRA because a language swap touches the acoustic encoder and the full output distribution, not just decoding style; LoRA (`fit_adapter`) is kept as fallback rung 2. VRAM: full FT of 123M params in bf16 with schedule-free AdamW ≈ 123M×(2+4+4+4) bytes ≈ 1.8 GB + activations — comfortable on 12 GB, unconstrained on Strix Halo unified memory.

### D2: Training loop adapted from `finetune-moonshine-asr` methods, extended for streaming checkpoints
Reuse its proven choices — HF transformers-style loading, schedule-free AdamW (`schedulefree` package), curriculum staging (short→long audio), jiwer eval hooks — implemented in this repo (the upstream repo is a guide, not a dependency; vendoring it would import its non-streaming assumptions). Streaming checkpoints are trained on full utterances (same as upstream trains them); KV-cache and speculative decoding are *runtime* behaviors exercised at eval/export time. Chunked-augmentation (random truncation to partial utterances with correspondingly truncated transcripts) is added as a regularizer so the chunked streaming eval condition is seen in training. Alternative rejected: training strictly on chunked streams — no evidence upstream does this, and it complicates transcript alignment.

### D3: Spike gate before any long run (highest technical risk)
Spike 1: forward+backward pass through `moonshine-streaming-small` in the training stack; success criterion = finite loss and non-trivial gradients. Spike 2: tokenizer inspection — encode/decode Italian text with accents and apostrophes; success = lossless round-trip after normalization (the family ships Japanese/Korean models, so vocab is likely byte-level or multilingual; if not representable, embedding-surgery cost must be sized before proceeding). Spike 3 (cheap): baseline streaming eval of the *English* checkpoint on FLEURS-it to calibrate the streaming harness and record the "before" number. Fallback ladder on spike failure, selected explicitly in `config.yaml`: (1) non-streaming `moonshine-base` bring-up (proven path, validates data/eval/export machinery, loses streaming), (2) `moonshine-voice[lora]` `fit_adapter` (partial), (3) report findings and re-scope. The spike gate is a spec requirement (see `training-pipeline` spec).

### D4: Datasets — FLEURS-it smoke+eval, MLS-it primary, CV-it optional mix
FLEURS-it (~12 h, no auth) powers the smoke profile and is the held-out eval for all gates (upstream's own non-English models are FLEURS-evaluated — comparable numbers). MLS Italian is the primary training corpus (read speech, large). Common Voice Italian is an optional mix-in for microphone/accent diversity, gated on `HF_TOKEN` in `.env`. VoxPopuli-it excluded from v1 (noisy domain, adds prep complexity; revisit if WER plateaus). Preparation: 16 kHz mono resample, VAD-aware segmentation to 1–10 s chunks, Italian text normalization (preserve accents, lowercase target convention matched to tokenizer, number expansion).

### D5: Eval = dual-mode with gates
Mode A: full-utterance WER/CER (jiwer) — comparable to upstream catalog numbers. Mode B: chunked streaming simulation (100 ms hop default, 32–100 ms configurable) through the same chunked decode path used at runtime, speculative decoding on, `max_tokens_per_second` hallucination guard active — this is the number that predicts board behavior. Gates in `config.yaml`: smoke gate generous (validates process, not quality); final gate set after baseline + first real run (initial placeholder: ≤ 15% streaming WER on FLEURS-it, to be ratified with data). Gate failure blocks the next phase (spec: `evaluation`).

### D6: Export chain mirrors upstream exactly
PyTorch → ONNX (encoder + KV-cached decoder as separate graphs) → INT8 dynamic quantization → `.ort` via upstream's `convert-models-to-ort.py` approach (vendored script or submodule pinned to a tag; prefer vendoring for stability). Numerical parity check ONNX-vs-PyTorch before quantization (tolerance in `config.yaml`); post-quantization WER delta measured in eval mode B and reported (upstream ships quantized models with "a little higher" WER — Tiny notably; Small expected milder). Validation record: sizes, checksums, smoke-load in an ORT session. This chain is the contract for the board.

### D7: Taskfile with hardware profiles and a `smoke` aggregate target
Targets: `env-check`, `download-model`, `download-data`, `prepare`, `spike`, `train` (profile-aware), `eval`, `export`, `ort`, `validate`, `release`, `board-deploy`, `smoke`, `final-train`. Hardware profiles `rocm12g|strix|cuda` map to `config.yaml` sections (batch size, precision, ORT provider, jobs). `smoke` = chained full-process run on the FLEURS slice with a per-phase summary and pass/fail record; `final-train` requires the recorded smoke success (spec: `task-orchestration`). Taskfile chosen over Make/just per user requirement.

### D8: Environment = uv venv + Arch native GPU packages
`.venv` created by `uv` with `--system-site-packages` so pacman's `python-pytorch-rocm` (or `-cuda`) and `python-onnxruntime-rocm` (or `-cuda`) are visible; uv manages all pure-Python deps (transformers, datasets, jiwer, schedulefree, tensorboard, …). The two ROCm/CUDA package families conflict with each other and with the default `python-pytorch`/`python-onnxruntime` — `env-check` detects which is installed and errors with install instructions otherwise. Docs record exact pacman commands per platform. `.env` (git-ignored) holds `HF_TOKEN`; `.env.example` documents keys.

### D9: Board deployment = Linux-side runtime + measurement, tiny fallback pre-planned
Deploy the `.ort` release bundle + tokenizer assets to the UNO Q Debian side over SSH/scp (documented path, checksums verified on-board). Run streaming inference via the Moonshine Voice runtime (`moonshine-voice` Python wheel if arm64 wheels exist, else the C++ core build — decided during implementation, documented either way). Measure: re-decode latency distribution (speculative on/off), pause→final-text latency, peak RSS; compare against a budget in `config.yaml` (initial: re-decode cadence ≤ ~1.5 s, RSS ≤ 2 GB — ratified on first measurement). If small misses the budget, the same pipeline exports a `streaming-tiny-it` fine-tune as the documented fallback (task exists, gated).

### D10: Repo layout
```
config.yaml            # all profiles (smoke/final), hardware profiles, gates, budgets
.env.example           # HF_TOKEN etc.
Taskfile.yml
train.py               # profile-aware entry (spike/smoke/final share one code path)
src/moonshine_it/      # data prep, normalization, training loop, eval harness, export
scripts/               # convert-models-to-ort (vendored), board deploy, fetch helpers
docs/                  # environment, data, training, evaluation, export, board guides
README.md              # concise: quickstart, task map, pointers into docs/
data/ results/ artifacts/   # git-ignored products
```

## Risks / Trade-offs

- [Streaming checkpoint not trainable in community stack] → Spike 1 gates everything; fallback ladder D3 keeps the repo productive (non-streaming bring-up still validates data/eval/export) even in the worst case.
- [Tokenizer lacks Italian subwords] → Spike 2 measures round-trip loss; if partial, vocabulary extension + embedding init (new rows near zero) is a scoped, sized task before final training; if byte-fallback covers accents, no action.
- [QRB2210 re-decode slower than usable] → speculative decoding masks cadence; budget gate D9 makes failure explicit; tiny-variant export is the pre-planned escape hatch (cost: ~1 extra fine-tune run on Strix).
- [INT8 WER penalty on small-streaming unknown] → post-quant eval in mode B is mandatory before release; FP16 `.ort` variant is a cheap parallel artifact if INT8 misses gate.
- [ROCm 12 GB OOM during smoke full-FT] → profile knobs (batch, grad-accum, curriculum caps audio length); activations are the dominant term at small scale; bf16 + short-stage curriculum keeps smoke comfortable.
- [Upstream API drift (moonshine-voice is v0.x)] → pin the runtime version for board work; vendored convert script insulates export from upstream churn.
- [Long final run wastes compute if process wrong] → smoke gate is enforced (spec), and smoke shares the exact code path (D2); only dataset size and step counts differ.

## Migration Plan

Greenfield repo; no migration. Rollback strategy = artifacts are content-addressed (checksums) per run; a failed run leaves prior `results/` untouched; `final-train` never overwrites smoke outputs (separate output roots per profile).

## Open Questions

- Exact MLS-it usable hours after segmentation and filtering (affects final-run step budget; answer during data prep on the 12 GB box).
- arm64 wheel availability for `moonshine-voice` on Debian (decides Python-vs-C++ board runtime; answered at board-deploy time, spec-neutral).
- Final WER gate value (ratify after baseline + first fine-tune curve; placeholder ≤ 15% streaming WER).
