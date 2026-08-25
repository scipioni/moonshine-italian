#!/usr/bin/env python3
"""On-board measurement suite for the UNO Q (QRB2210, Debian).

Run ON THE BOARD against the deployed .ort bundle:

    python3 scripts/board/measure.py --release-dir /opt/moonshine-it \
        --audio test_clip.wav

Measures, and compares against the board budget in config.yaml (or the
inline defaults mirrored from it):
  - re-decode latency distribution, speculative decoding ON and OFF
  - pause->final-text latency (decode of the trailing chunk after a pause)
  - peak RSS of the streaming runtime with the small model loaded

Writes a budget_report.json next to the release dir with per-metric
pass/fail. If the re-decode cadence budget fails, the report names the
documented fallback (streaming-tiny Italian export through the same
pipeline) per the board-deployment spec.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    if file_sr != sr:
        raise SystemExit(f"clip must be {sr} Hz mono (got {file_sr})")
    return data


def measure(deploy_dir: Path, audio_path: Path, hop_ms: int,
            max_tokens_per_s: float, budget: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))          # board: ort_runtime.py next to this script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo
    try:
        from moonshine_it.ort_runtime import OrtPipeline, read_wav_mono16k
    except ImportError:  # board standalone: module lives next to this script
        from ort_runtime import OrtPipeline, read_wav_mono16k

    pipe = OrtPipeline(deploy_dir, provider="CPUExecutionProvider")
    audio = read_wav_mono16k(audio_path)

    def run(speculative: bool) -> dict:
        t0 = time.perf_counter()
        _, stats = pipe.transcribe_streaming(
            audio, hop_ms=hop_ms, max_tokens_per_s=max_tokens_per_s,
            speculative=speculative)
        wall = time.perf_counter() - t0
        stats["wall_s"] = round(wall, 3)
        return stats

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    spec_on = run(True)
    spec_off = run(False)
    peak_rss_mb = max(peak_rss_mb,
                      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

    # pause->text latency: single re-decode of the trailing hop after a pause,
    # measured as one full encode+decode of the last chunk with warm cache
    hop = int(16000 * hop_ms / 1000)
    t0 = time.perf_counter()
    cross_k, cross_v = pipe.encode(audio[-hop:])
    prefix = [pipe.bos]
    token_budget = int((len(audio) / 16000) * max_tokens_per_s)
    pipe.greedy_continue(prefix, cross_k, cross_v, min(token_budget, 32))
    pause_to_text_ms = round(1000 * (time.perf_counter() - t0), 1)

    report = {
        "metrics": {
            "redecode_latency_ms_mean": {
                "speculative_on": spec_on["redecode_latency_ms_mean"],
                "speculative_off": spec_off["redecode_latency_ms_mean"],
                "budget_ms_max": budget["redecode_ms_max"],
                "pass": spec_on["redecode_latency_ms_mean"] <= budget["redecode_ms_max"],
            },
            "pause_to_text_ms": {
                "measured": pause_to_text_ms,
                "budget_ms_max": budget["line_latency_ms_max"],
                "pass": pause_to_text_ms <= budget["line_latency_ms_max"],
            },
            "peak_rss_mb": {
                "measured": round(peak_rss_mb, 1),
                "budget_mb_max": budget["rss_mb_max"],
                "pass": peak_rss_mb <= budget["rss_mb_max"],
            },
            "rtf": {"speculative_on": spec_on["rtf"]},
        },
        "all_pass": None,
        "fallback": None,
    }
    report["all_pass"] = all(m["pass"] for m in report["metrics"].values()
                             if isinstance(m, dict) and "pass" in m)
    if not report["metrics"]["redecode_latency_ms_mean"]["pass"]:
        report["fallback"] = (
            "streaming-small missed the re-decode budget: export a "
            "streaming-tiny Italian variant through the same pipeline "
            "(documented fallback, board-deployment spec)"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default="/opt/moonshine-it")
    parser.add_argument("--audio", required=True, help="16 kHz mono test clip")
    parser.add_argument("--hop-ms", type=int, default=100)
    parser.add_argument("--max-tokens-per-second", type=float, default=13.0)
    parser.add_argument("--budget", default=None,
                        help="budget JSON {redecode_ms_max, line_latency_ms_max, rss_mb_max}"
                             " (default: config.yaml board budget values)")
    args = parser.parse_args()

    if args.budget:
        budget = json.loads(args.budget)
    else:
        budget = {"redecode_ms_max": 1500, "line_latency_ms_max": 3000,
                  "rss_mb_max": 2048}  # mirrors config.yaml board.budget

    report = measure(Path(args.release_dir), Path(args.audio),
                     args.hop_ms, args.max_tokens_per_second, budget)
    out = Path(args.release_dir) / "budget_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
