# Export & Quantization

Chain: PyTorch checkpoint → ONNX graphs → INT8 → `.ort` FlatBuffers.
Mirrors the upstream moonshine recipe exactly (vendored converter +
onnx-shrink-ray).

## Graphs

| graph         | role                                                       |
|---------------|------------------------------------------------------------|
| `encoder.onnx`| input_values → enc_hidden (frontend merged in)             |
| `adapter.onnx`| enc_hidden → adapted (decoder dims)                        |
| `cross_kv.onnx`| adapted → per-layer cross-attention K/V                    |
| `decoder_kv.onnx`| input_ids + cross K/V + past self K/V → logits + present K/V (KV-cached streaming decoder) |

Per-layer K/V travel as stacked tensors `[n_layers, 1, H, ·, D]`.

## Commands

```bash
task export PROFILE=rocm12g        # gate-checked ONNX export + parity verification
task ort PROFILE=rocm12g           # INT8 quantization + .ort serialization
task ort-eval PROFILE=rocm12g      # .ort streaming eval + post-quant gate
task validate PROFILE=rocm12g      # checksums, ORT smoke-load, promotion manifest
task release PROFILE=rocm12g       # export → ort → ort-eval → validate
```

## Parity

`task export` ends with `src/moonshine_it/parity.py`: every graph is run
against the PyTorch checkpoint on reference audio; max-abs logit/encoding
differences must stay within `export.onnx.tolerance` (2e-2), the decoder
reimplementation is cross-checked against the HF forward (argmax must
match), and the cached path (non-empty past K/V) is verified too. Record:
`results/export/<name>/parity.json`.

## Quantization details

`onnx-shrink-ray` with `--per-channel` (load-bearing for accuracy: the
frontend's weight-norm'd conv directions span 17× magnitudes; a single scale
per tensor costs ~8 WER points). Method per graph:

- `encoder`: `integer_weights` (weight-only) — activation quantization is
  unusable there (measured ~15 logits max-abs end-to-end degradation)
- `adapter`, `cross_kv`, `decoder_kv`: `integer_activations`

`.ort` serialization via the upstream `onnxruntime.tools.convert_onnx_models_to_ort`
(pinned vendored copy: `scripts/convert-models-to-ort.py`). The runtime
consumes `.ort` only — every downstream loader rejects `.onnx` intermediates
with an error naming the `.ort` artifact
(`moonshine_it.release.require_ort_file`).

## Release bundle

`artifacts/release/<checkpoint>/` contains the four `.ort` graphs, tokenizer
assets, `size_report.json` (pre/post-quantization sizes) and, after
`task validate`, `manifest.json` (sha256 per file) + `validation.json` (ORT
smoke-load per graph). A corrupted or truncated artifact fails validation
(non-zero exit) and no manifest is written, so nothing unvalidated reaches
board-deploy.
