"""INT8 quantization (onnx-shrink-ray) + ORT-format conversion.

Mirrors the upstream moonshine recipe: integer_activations with per-channel
scales (accuracy-critical for weight-norm'd conv directions), then
convert_onnx_models_to_ort. Writes <graph>.ort into the release dir plus a
size report (config: export.ort.size_report).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, load_config

GRAPHS = ("encoder", "adapter", "cross_kv", "decoder_kv")

# The frontend (weight-norm'd CausalConv1d embedder, merged into our encoder
# graph) needs weight-only quantization; activation quantization there costs
# ~15 logits max-abs end to end. The rest use full integer activations.
GRAPH_METHODS = {
    "encoder": "integer_weights",
    "adapter": "integer_activations",
    "cross_kv": "integer_activations",
    "decoder_kv": "integer_activations",
}


def _mb(path: Path) -> float:
    size = path.stat().st_size
    data = path.with_name(path.name + ".data")
    if data.exists():
        size += data.stat().st_size
    return round(size / 1e6, 2)


def _run(cmd: list[str]) -> None:
    print("quantize:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def quantize_release(export_dir: Path, release_dir: Path,
                     graphs: tuple[str, ...] = GRAPHS) -> dict:
    release_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"graphs": {}}

    for name in graphs:
        src = export_dir / f"{name}.onnx"
        if not src.exists():
            raise FileNotFoundError(f"missing exported graph: {src}")
        fp32_mb = _mb(src)

        _run([sys.executable, "-m", "onnx_shrink_ray.shrink",
              "--ir-version", "10", "--method", GRAPH_METHODS[name],
              "--per-channel", str(src)])

        produced = sorted(export_dir.glob(f"{name}_quantized*.onnx"))
        if not produced:
            raise RuntimeError(f"onnx-shrink-ray produced no output for {name}")
        quantized = produced[0]
        int8_mb = _mb(quantized)

        _run([sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
              str(quantized)])
        ort_file = quantized.with_suffix(".ort")
        if not ort_file.exists():
            raise RuntimeError(f"ort conversion produced no output for {name}")
        dest = release_dir / f"{name}.ort"
        shutil.move(str(ort_file), dest)
        for leftover in export_dir.glob(f"{name}_quantized*"):
            if leftover.is_dir():
                shutil.rmtree(leftover)
            else:
                leftover.unlink()

        report["graphs"][name] = {
            "fp32_onnx_mb": fp32_mb,
            "int8_onnx_mb": int8_mb,
            "ort_mb": _mb(dest),
            "ort_path": str(dest),
        }

    report["totals"] = {
        "fp32_onnx_mb": round(sum(g["fp32_onnx_mb"] for g in report["graphs"].values()), 2),
        "ort_mb": round(sum(g["ort_mb"] for g in report["graphs"].values()), 2),
    }
    report["totals"]["compression"] = round(
        report["totals"]["fp32_onnx_mb"] / report["totals"]["ort_mb"], 2)

    out = release_dir / "size_report.json"
    out.write_text(json.dumps(report, indent=2))
    print("quantize:", json.dumps(report["totals"]))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="checkpoint name (default: base)")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--release-dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_config()
    name = Path(args.model).name if args.model else "base"
    export_dir = Path(args.export_dir) if args.export_dir else \
        REPO_ROOT / cfg["paths"]["results"] / "export" / name
    release_dir = Path(args.release_dir) if args.release_dir else \
        REPO_ROOT / cfg["release"]["dir"] / name
    quantize_release(export_dir, release_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
