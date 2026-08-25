"""Streaming evaluation of the INT8 `.ort` release bundle.

Mirrors evaluate.transcribe_streaming but runs the four `.ort` graphs
(encoder, adapter, cross_kv, decoder_kv) through ONNX Runtime instead of the
PyTorch model: audio appended per hop, speculative prefix verification,
KV-cached greedy continuation, hallucination guard, per-re-decode latency.

Writes results/eval/<name>_streaming_ort.json and (with --gate) checks the
post-quantization degradation gate before the bundle may be promoted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from moonshine_it.config import REPO_ROOT, load_config
from moonshine_it.evaluate import EvalResult, load_audio, load_manifest
from moonshine_it.gates import check_wer_gate
from moonshine_it.model_io import results_dir
from moonshine_it.normalize_it import normalize_text
from moonshine_it.ort_runtime import OrtPipeline
from moonshine_it.release import require_ort_file


def evaluate_ort(
    pipe: OrtPipeline,
    manifest_path: Path,
    audio_root: Path,
    *,
    streaming_cfg: dict,
    limit: int | None = None,
    model_name: str = "int8",
    dataset: str = "fleurs-it",
    split: str = "test",
) -> EvalResult:
    import jiwer

    rows = load_manifest(manifest_path)
    if limit:
        rows = rows[:limit]
    refs, hyps, stats_all = [], [], []
    for row in rows:
        audio = load_audio(audio_root / row["audio"])
        ref = normalize_text(row["text"], expand_nums=False)
        hyp, stats = pipe.transcribe_streaming(
            audio,
            hop_ms=streaming_cfg["hop_ms"],
            max_tokens_per_s=streaming_cfg["max_tokens_per_second"],
            speculative=streaming_cfg.get("speculative_decoding", True),
        )
        hyp = normalize_text(hyp, expand_nums=False)
        refs.append(ref)
        hyps.append(hyp)
        stats_all.append(stats)

    wer = float(jiwer.wer(refs, hyps)) * 100
    cer = float(jiwer.cer(refs, hyps)) * 100
    extra = {
        "redecode_latency_ms_mean": float(np.mean(
            [s["redecode_latency_ms_mean"] for s in stats_all])),
        "redecode_latency_ms_p95": float(np.mean(
            [s["redecode_latency_ms_p95"] for s in stats_all])),
        "rtf_mean": float(np.mean([s["rtf"] for s in stats_all])),
        "provider": pipe.provider,
    }
    return EvalResult(
        model=model_name, dataset=dataset, split=split, mode="streaming",
        n=len(rows), wer=round(wer, 2), cer=round(cer, 2), extra=extra,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="checkpoint name (default: base)")
    parser.add_argument("--release-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gate", action="store_true",
                        help="check the post_quant gate (blocks promotion on failure)")
    parser.add_argument("--delta-vs", default=None,
                        help="PyTorch streaming result JSON to compute the "
                             "post-quant WER delta against")
    parser.add_argument("--name", default=None)
    parser.add_argument("--provider", default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    name = args.name or ((Path(args.model).name if args.model else "base") + "_int8")
    release_dir = Path(args.release_dir) if args.release_dir else \
        REPO_ROOT / cfg["release"]["dir"] / (Path(args.model).name if args.model else "base")
    manifest = (Path(args.manifest) if args.manifest
                else REPO_ROOT / cfg["smoke"]["slice_manifest"] / "test.jsonl")
    audio_root = manifest.parent / "audio"

    pipe = OrtPipeline(release_dir, provider=args.provider)
    print(f"ort-eval: provider={pipe.provider} graphs={list(pipe.sess)}")

    res = evaluate_ort(
        pipe, manifest, audio_root,
        streaming_cfg=cfg["evaluation"]["streaming"],
        limit=args.limit, model_name=name,
        dataset="fleurs-it", split=manifest.stem,
    )

    if args.delta_vs:
        ref_json = Path(args.delta_vs)
        if ref_json.exists():
            ref = json.loads(ref_json.read_text())
            res.extra["pre_quant_streaming_wer"] = ref["wer"]
            res.extra["post_quant_wer_delta"] = round(res.wer - ref["wer"], 2)
            res.extra["pre_quant_n"] = ref.get("n")
            print(f"ort-eval: post-quant delta {res.extra['post_quant_wer_delta']:+.2f} "
                  f"WER points vs {ref_json.name}")

    out_dir = results_dir(cfg, "eval")
    out = out_dir / f"{name}_streaming.json"
    out.write_text(res.to_json())
    print(f"ort-eval[{name}/streaming]: WER {res.wer:.2f}% CER {res.cer:.2f}% "
          f"(n={res.n}) redecode {res.extra['redecode_latency_ms_mean']}ms "
          f"rtf {res.extra['rtf_mean']}")

    if args.gate:
        baseline_wer = res.extra.get("pre_quant_streaming_wer")
        check_wer_gate("post_quant", res.wer, baseline_wer=baseline_wer, cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
