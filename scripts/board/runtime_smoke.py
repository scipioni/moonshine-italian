#!/usr/bin/env python3
"""On-board streaming smoke: emit line events while audio arrives.

Runs the deployed `.ort` bundle through ONNX Runtime on the board's Debian
side, feeding a 16 kHz mono clip in hops and printing a line event per
re-decode (the streaming contract the voice agent consumes). Verifies the
graphs load and stream from a test clip.

    python3 runtime_smoke.py --release-dir /opt/moonshine-it --audio clip.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default="/opt/moonshine-it")
    parser.add_argument("--audio", required=True, help="16 kHz mono wav")
    parser.add_argument("--hop-ms", type=int, default=100)
    parser.add_argument("--max-tokens-per-second", type=float, default=13.0)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))          # board: ort_runtime.py next to this script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo
    try:
        from moonshine_it.ort_runtime import OrtPipeline, read_wav_mono16k
    except ImportError:  # board standalone: module lives next to this script
        from ort_runtime import OrtPipeline, read_wav_mono16k

    audio = read_wav_mono16k(Path(args.audio))

    t_load = time.perf_counter()
    pipe = OrtPipeline(Path(args.release_dir), provider="CPUExecutionProvider")
    print(f"runtime: 4 graphs loaded in {time.perf_counter() - t_load:.1f}s "
          f"(provider={pipe.provider})")

    sr = 16000
    hop = int(sr * args.hop_ms / 1000)
    prefix = [pipe.bos]
    last_text = ""
    for end in range(hop, len(audio) + 1, hop):
        chunk = audio[:end]
        t0 = time.perf_counter()
        cross_k, cross_v = pipe.encode(chunk)
        if len(prefix) > 1:
            prefix = pipe.verify_prefix(prefix, cross_k, cross_v)
        budget = int((len(chunk) / sr) * args.max_tokens_per_second)
        prefix = pipe.greedy_continue(prefix, cross_k, cross_v,
                                      min(max(4, budget - len(prefix) + 1), 32))
        dt = (time.perf_counter() - t0) * 1000
        text = pipe.decode_tokens(prefix)
        if text != last_text:  # line event: only on change, while audio arrives
            print(f"line [{dt:7.1f} ms] {text}", flush=True)
            last_text = text
    print(f"final: {last_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
