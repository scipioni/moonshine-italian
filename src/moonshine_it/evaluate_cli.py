"""Evaluation CLI: full-utterance and streaming modes, with gates.

Usage:
  uv run python -m moonshine_it.evaluate --model <path|base> --mode full
  uv run python -m moonshine_it.evaluate --model <path|base> --mode streaming
  uv run python -m moonshine_it.evaluate --model <path|base> --mode both --gate smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, load_config
from moonshine_it.download import model_dir
from moonshine_it.evaluate import evaluate_manifest
from moonshine_it.gates import check_wer_gate
from moonshine_it.model_io import load_model_and_processor, results_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="checkpoint dir, or 'base' for the untuned snapshot")
    parser.add_argument("--mode", choices=["full", "streaming", "both"],
                        default="both")
    parser.add_argument("--manifest", default=None,
                        help="manifest jsonl (default: smoke test slice)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gate", choices=["smoke", "final"], default=None,
                        help="apply the configured WER gate after eval")
    parser.add_argument("--name", default=None,
                        help="result file stem (default: derived from model path)")
    args = parser.parse_args(argv)

    cfg = load_config()
    model_path = model_dir(cfg) if args.model == "base" else Path(args.model)
    manifest = (Path(args.manifest) if args.manifest
                else REPO_ROOT / cfg["smoke"]["slice_manifest"] / "test.jsonl")
    audio_root = manifest.parent / "audio"
    name = args.name or (args.model if args.model == "base"
                         else Path(args.model).name)

    model, proc = load_model_and_processor(cfg, model_path=model_path,
                                           device="cuda")
    streaming_cfg = cfg["evaluation"]["streaming"]
    out_dir = results_dir(cfg, "eval")
    modes = ["full", "streaming"] if args.mode == "both" else [args.mode]

    results = {}
    for mode in modes:
        res = evaluate_manifest(
            model, proc, manifest, audio_root,
            mode=mode, streaming_cfg=streaming_cfg, limit=args.limit,
            model_name=name, dataset="fleurs-it", split=manifest.stem,
        )
        out = out_dir / f"{name}_{mode}.json"
        out.write_text(res.to_json())
        results[mode] = res
        print(f"eval[{name}/{mode}]: WER {res.wer:.2f}% CER {res.cer:.2f}% "
              f"(n={res.n})" + (f" redecode {res.extra.get('redecode_latency_ms_mean')}ms"
                                if res.extra else ""))

    if args.gate:
        gate_cfg = cfg["evaluation"]["gates"][args.gate]
        gate_mode = gate_cfg.get("mode", "full")
        measured = results[gate_mode].wer if gate_mode in results else None
        if measured is None:  # gate mode not evaluated this run
            res = evaluate_manifest(
                model, proc, manifest, audio_root,
                mode=gate_mode, streaming_cfg=streaming_cfg, limit=args.limit,
                model_name=name, dataset="fleurs-it", split=manifest.stem)
            out_dir.joinpath(f"{name}_{gate_mode}.json").write_text(res.to_json())
            measured = res.wer
        baseline_path = out_dir / f"baseline_{gate_mode}.json"
        baseline_wer = (json.loads(baseline_path.read_text())["wer"]
                        if baseline_path.exists() else None)
        check_wer_gate(args.gate, measured, baseline_wer=baseline_wer, cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
