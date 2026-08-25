"""Profile one training step under torch.profiler to name the dominant cost.

Writes a kernel/self-time/fraction-of-step table to
`results/profile/<hardware>/profile.json`. This is the profiling spike gate
for training-performance optimizations: an optimization is only accepted if
its design cites the dominant kernel named here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from moonshine_it.config import REPO_ROOT, load_config, resolve_profile
from moonshine_it.train_loop import ASRDataset, Collator


def _step(model, proc, batch, optimizer, autocast_dtype):
    import torch

    batch = {k: v.to("cuda") for k, v in batch.items()}
    if autocast_dtype is not None:
        batch["input_values"] = batch["input_values"].to(autocast_dtype)
    ctx = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
           if autocast_dtype else torch.autocast(device_type="cuda", enabled=False))
    with ctx:
        out = model(input_values=batch["input_values"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"])
        loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def main(argv: list[str] | None = None) -> int:
    import argparse

    from torch.utils.data import DataLoader
    from schedulefree import AdamWScheduleFree as ScheduleFreeAdamW

    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="rocm12g")
    parser.add_argument("--steps", type=int, default=3,
                        help="steps to profile (last step is the measured one)")
    parser.add_argument("--source", default="smoke",
                        choices=["smoke", "mls"],
                        help="data source: smoke slice (fast) or prepared MLS")
    args = parser.parse_args(argv)

    cfg = load_config()
    rp = resolve_profile(cfg, args.hardware, "smoke")  # smoke profile, any hardware
    import torch

    # same SDPA hardening as the training loop (see train_loop.py)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    from moonshine_it.model_io import load_model_and_processor

    model, proc = load_model_and_processor(cfg, device="cuda", dtype=rp.precision)
    model.train()

    if args.source == "smoke":
        manifest = REPO_ROOT / cfg["smoke"]["slice_manifest"] / "train.jsonl"
        audio_root = REPO_ROOT / cfg["smoke"]["slice_manifest"] / "audio"
    else:
        data_root = REPO_ROOT / cfg["paths"]["data"] / "prepared" / "mls"
        manifest = data_root / "train.jsonl"
        audio_root = data_root
    if not manifest.exists():
        raise SystemExit(f"training manifest missing: {manifest} — run prepare first")

    dataset = ASRDataset(manifest, audio_root, cfg, augment=True,
                         seed=cfg["smoke"]["seed"])
    loader = DataLoader(
        dataset,
        batch_size=rp.batch_size,
        shuffle=True,
        num_workers=rp.num_workers,
        collate_fn=Collator(proc),
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(cfg["smoke"]["seed"]),
    )

    tcfg = cfg["training"]
    optimizer = ScheduleFreeAdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=tuple(tcfg["betas"]),
        weight_decay=tcfg["weight_decay"],
        warmup_steps=tcfg["warmup_steps"],
    )
    optimizer.train()
    autocast_dtype = {"bf16": torch.bfloat16, "fp32": None}[rp.precision]

    it = iter(loader)
    # warm up + time the un-profiled step to get a baseline wall-time
    t0 = time.time()
    for _ in range(max(1, args.steps - 1)):
        _step(model, proc, next(it), optimizer, autocast_dtype)
    torch.cuda.synchronize()
    wall_measured = time.time() - t0

    # profile the final step
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    )
    prof.start()
    step_t0 = time.time()
    _step(model, proc, next(it), optimizer, autocast_dtype)
    torch.cuda.synchronize()
    step_wall = time.time() - step_t0
    prof.stop()

    # aggregate by kernel name
    agg: dict[str, dict] = {}
    for evt in prof.key_averages():
        name = evt.key
        agg.setdefault(name, {"count": 0, "cpu_ms": 0.0, "cuda_ms": 0.0})
        agg[name]["count"] += evt.count
        agg[name]["cpu_ms"] += evt.self_cpu_time_total / 1000.0
        agg[name]["cuda_ms"] += evt.self_device_time_total / 1000.0

    total_cuda = sum(v["cuda_ms"] for v in agg.values()) or 1.0
    table = sorted(
        ({**v, "name": k, "fraction_of_step": v["cuda_ms"] / total_cuda}
         for k, v in agg.items()),
        key=lambda r: r["cuda_ms"],
        reverse=True,
    )
    dominant = table[0] if table else None

    out = {
        "hardware": args.hardware,
        "source": args.source,
        "batch_size": rp.batch_size,
        "precision": rp.precision,
        "steps_profiled": args.steps,
        "step_wall_s": step_wall,
        "unprofiled_steps_wall_s": wall_measured,
        "dominant_kernel": dominant["name"] if dominant else None,
        "dominant_fraction_of_step": dominant["fraction_of_step"] if dominant else None,
        "top_kernels": table[:20],
    }
    out_dir = REPO_ROOT / cfg["paths"]["results"] / "profile" / args.hardware
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "profile.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"profile: dominant kernel {out['dominant_kernel']} "
          f"({out['dominant_fraction_of_step']:.2%} of step) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())