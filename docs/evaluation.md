# Evaluation Methodology

Dual-mode evaluation on the held-out FLEURS Italian test split (jiwer).

## Mode A — full utterance

Whole-clip decode, WER/CER comparable to upstream catalog numbers.

## Mode B — chunked streaming simulation

Mirrors the on-device runtime: audio appended in hops (`hop_ms`, 32–100 ms,
default 100), each re-decode teacher-forces the previous hypothesis
(speculative verification — keep tokens up to the first mismatch), then
continues greedy generation under a token budget
(`max_tokens_per_second`, hallucination guard: excessive token rate
truncates and freezes the line exactly like the runtime heuristic). Reports
streaming WER/CER, per-re-decode latency (mean/p95), and real-time factor.

## Commands

```bash
task eval PROFILE=rocm12g                                    # both modes + smoke gate
uv run python -m moonshine_it.evaluate_cli --model results/train-smoke/checkpoint-best \
    --mode streaming --name smoke
```

Results JSON: model, dataset, split, sample count, WER, CER (+ latency/RTF
for streaming) under `results/eval/`.

## Gates

Configured in `config.yaml` (`evaluation.gates`), enforced via
`src/moonshine_it/gates.py`:

| gate        | mode      | rule                                          | blocks           |
|-------------|-----------|-----------------------------------------------|------------------|
| `smoke`     | full      | WER ≤ baseline × 1.10 (process validation)    | train → export   |
| `final`     | streaming | WER ≤ 15% absolute (ratified after real runs)  | export           |
| `post_quant`| streaming | .ort WER ≤ pre-quant WER × 1.25               | release promotion|

A failed gate exits non-zero with measured-vs-allowed values and blocks all
downstream targets (`task export` refuses to run without a passing gate
record in `results/gates/`).

## Post-quantization delta

`task ort-eval` runs mode B against the INT8 `.ort` bundle (ORT runtime,
same graphs the board loads), records the WER delta versus the pre-quant
checkpoint, and applies the `post_quant` gate before the release bundle can
be promoted.
