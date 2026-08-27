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
    With "bf16" the weights load in bf16 so matmuls run bf16 (RDNA-native)
    instead of fp32. Callers MUST pass bf16 inputs (the streaming encoder's
    embedder linear requires input and weight dtypes to match), e.g. via
    torch.autocast or an explicit cast of `input_values` to bf16.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    cfg = cfg or load_config()
    path = Path(model_path) if model_path else model_dir(cfg)
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(path), local_files_only=True, dtype=torch_dtype
    )
    model.to(device)
    if model_path is not None:
        # An explicit model_path is a training checkpoint, not the plain base
        # snapshot (which has no optimizer.pt and no iterate to disambiguate).
        # Standardize on "y" regardless of which era wrote it (design Decision
        # 2, fix-training-loop-defects).
        _normalize_checkpoint_iterate(model, path)
    if device.startswith("cuda"):
        model.eval()
    proc = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    return model, proc


def _normalize_checkpoint_iterate(model, path: Path) -> None:
    """Ensure a loaded checkpoint's weights are the "y" (raw) schedule-free
    iterate, converting in place if the checkpoint stored "x" (the averaged
    iterate -- written only by a checkpoint saved while commit 3390635's
    since-reversed direction was in effect).

    Fails loudly if the checkpoint directory has model.safetensors but no
    optimizer.pt: without it there is no way to tell which iterate the
    weights hold.

    Reconstructs the "x" -> "y" conversion directly from the saved optimizer
    state_dict's 'z' buffers (the same formula as AdamWScheduleFree.train(),
    see design Context) rather than re-instantiating the optimizer class, so
    this works regardless of the checkpoint's device/dtype at load time.
    """
    opt_path = path / "optimizer.pt"
    if not opt_path.exists():
        raise RuntimeError(
            f"{path} looks like a training checkpoint (model.safetensors "
            f"present) but has no optimizer.pt -- cannot determine which "
            f"schedule-free iterate ('x' or 'y') its weights hold")
    state = torch.load(opt_path, map_location="cpu", weights_only=False)
    group = state["param_groups"][0]
    if group["train_mode"]:
        return  # already "y"
    beta1 = group["betas"][0]
    opt_state = state["state"]
    with torch.no_grad():
        for idx, p in zip(group["params"], model.parameters()):
            st = opt_state.get(idx)
            if st is None or "z" not in st:
                continue
            z = st["z"].to(device=p.device, dtype=p.dtype)
            p.lerp_(end=z, weight=1 - beta1)  # inverse of eval(): x -> y


def results_dir(cfg: dict, *parts: str) -> Path:
    d = REPO_ROOT / cfg["paths"]["results"]
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d
