"""Shared model loading (PyTorch, from the local snapshot)."""

from __future__ import annotations

from pathlib import Path

import torch

from moonshine_it.config import REPO_ROOT, load_config
from moonshine_it.download import model_dir


def load_model_and_processor(
    cfg: dict | None = None,
    *,
    model_path: str | Path | None = None,
    device: str = "cuda",
    dtype: str = "fp32",
):
    """Load MoonshineStreaming model+processor from the local snapshot.

    dtype: "fp32" (parity/baseline safe) or "bf16" (training/fast eval).
    Note: weights are always fp32 here; "bf16" currently still loads fp32
    weights because the streaming encoder path has dtype-mismatch issues
    with bf16 weights on ROCm. Use torch.autocast for bf16 compute.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    cfg = cfg or load_config()
    path = Path(model_path) if model_path else model_dir(cfg)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(path), local_files_only=True, dtype=torch.float32
    )
    model.to(device)
    if device.startswith("cuda"):
        model.eval()
    proc = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    return model, proc


def results_dir(cfg: dict, *parts: str) -> Path:
    d = REPO_ROOT / cfg["paths"]["results"]
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d
