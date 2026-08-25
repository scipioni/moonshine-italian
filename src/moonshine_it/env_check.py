"""Environment check: accelerator, native GPU packages, ORT providers, profile.

Spec: training-pipeline / Environment reproducibility. Reports accelerator
kind (CUDA/ROCm) and VRAM on success; fails with pacman guidance when a
GPU-enabled PyTorch is not importable.

Usage: uv run python -m moonshine_it.env_check [--profile rocm12g] [--json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from moonshine_it.config import (
    REPO_ROOT,
    VALID_HW_PROFILES,
    load_config,
    resolve_profile,
)

PACMAN_ROCM = "sudo pacman -S python-pytorch-rocm rocm-hip-sdk"
PACMAN_CUDA = "sudo pacman -S python-pytorch-cuda cuda cudnn"
PACMAN_ORT_ROCM = "sudo pacman -S python-onnxruntime-rocm"
PACMAN_ORT_CUDA = "sudo pacman -S python-onnxruntime-cuda"


def check_torch() -> dict:
    """Import torch, classify the build, detect accelerator and VRAM."""
    info: dict = {"torch": None}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on host
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
        return info

    info["torch"] = torch.__version__
    info["torch_native"] = str(torch.__file__).startswith("/usr/lib/python")
    info["torch_rocm_build"] = bool(torch.version.hip)
    info["torch_cuda_build"] = bool(torch.version.cuda)

    if not torch.cuda.is_available():
        info["accelerator"] = None
        info["vram_mb"] = None
        return info

    props = torch.cuda.get_device_properties(0)
    info["accelerator"] = "rocm" if torch.version.hip else "cuda"
    info["vram_mb"] = round(props.total_memory / 1024 / 1024)
    info["gpu_name"] = props.name
    return info


def check_ort() -> dict:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        return {"onnxruntime": None, "ort_error": f"{type(exc).__name__}: {exc}"}
    providers = list(ort.get_available_providers())
    native = not str(ort.__file__).startswith(str(Path(sys.prefix)))
    gpu_provider = any(
        p in providers for p in ("ROCMExecutionProvider", "CUDAExecutionProvider")
    )
    return {
        "onnxruntime": ort.__version__,
        "providers": providers,
        "ort_native": native,
        "ort_gpu": gpu_provider,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=VALID_HW_PROFILES, default=None)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    cfg = load_config()
    report: dict = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": check_torch(),
        "ort": check_ort(),
    }

    problems: list[str] = []

    t = report["torch"]
    if t["torch"] is None:
        problems.append(
            "PyTorch is not importable in this venv. Install a GPU build via pacman:\n"
            f"  ROCm: {PACMAN_ROCM}\n  CUDA: {PACMAN_CUDA}\n"
            "and recreate the venv: uv venv --clear --system-site-packages "
            "--python /usr/bin/python3.14 .venv && uv sync"
        )
    else:
        if not t["torch_native"]:
            problems.append(
                "PyTorch resolves from the venv (PyPI wheel), not the native "
                "pacman build. Remove it (uv pip uninstall torch torchaudio) so the "
                "system ROCm/CUDA build is used."
            )
        if not t.get("accelerator"):
            problems.append(
                "No GPU visible to PyTorch. For AMD GPUs install:\n"
                f"  {PACMAN_ROCM}\nFor NVIDIA:\n  {PACMAN_CUDA}"
            )

    o = report["ort"]
    if o.get("onnxruntime") is None:
        problems.append(
            "onnxruntime is not importable. Install via pacman:\n"
            f"  ROCm: {PACMAN_ORT_ROCM}\n  CUDA: {PACMAN_ORT_CUDA}"
        )
    elif not o.get("ort_gpu"):
        # CPU wheel (fine for export/quantization) but note native GPU option.
        report["ort_note"] = (
            "onnxruntime is a CPU build (export/quantization work; GPU inference "
            f"would need the native package). ROCm: {PACMAN_ORT_ROCM} CUDA: {PACMAN_ORT_CUDA}"
        )

    if args.profile:
        try:
            rp = resolve_profile(cfg, args.profile, "smoke")
            report["profile"] = rp.run_metadata()
            if t.get("accelerator") and rp.accelerator_kind != t["accelerator"]:
                problems.append(
                    f"profile '{args.profile}' expects {rp.accelerator_kind} "
                    f"but detected {t['accelerator']}"
                )
        except Exception as exc:
            problems.append(str(exc))

    report["ok"] = not problems
    report["problems"] = problems

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env-check.json").write_text(json.dumps(report, indent=2))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        t = report["torch"]
        print(f"python       {report['python']} ({report['machine']})")
        print(f"torch        {t.get('torch')} native={t.get('torch_native')}")
        print(f"accelerator  {t.get('accelerator')} ({t.get('gpu_name', '?')})")
        print(f"vram         {t.get('vram_mb')} MB")
        print(f"onnxruntime  {report['ort'].get('onnxruntime')} "
              f"gpu={report['ort'].get('ort_gpu')}")
        if args.profile:
            print(f"profile      {args.profile}: ok")
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        print("env-check:", "OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
