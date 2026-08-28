"""LR probe: does the corpus-scale plateau move at a higher learning rate?

Each arm starts from the SAME checkpoint with the SAME data order and runs a
short budget, so the only difference is the learning rate. One arm per
process: an amdgpu page fault then costs one arm, not the whole probe.

Produced the evidence behind the final profile's learning_rate. Measured
2026-08-28 from checkpoint-2997, 500 steps per arm, y iterate on
mls/validation (n=64) -- see results/lr-probe/results.jsonl:

    lr      loss first100 -> last100    WER after   d WER
    5e-5    2.5516 -> 2.4820 (0.070)    81.81%      -0.69
    1e-4    2.6576 -> 2.4301 (0.228)    79.75%      -2.75
    2e-4    3.0869 -> 2.4699 (0.617)    88.33%      +5.84  (33 steps skipped)
    5e-4    4.0585 -> 2.2129 (1.846)    73.34%      -9.15

At the then-configured 5e-5, 500 steps bought 0.69 WER points; at 5e-4 they
bought 9.15. That is why 12,000 steps of the final run moved WER 82.61% ->
84.67%: the run was not broken, it was crawling.

Sweep one arm per process so a GPU fault costs one arm:

    for lr in 5e-5 1e-4 2e-4 5e-4; do
      uv run --no-sync python scripts/lr_probe.py --lr $lr --steps 500
    done

The __main__ guard is required -- DataLoader workers use the forkserver start
method, which re-imports this module in each worker.
"""
import argparse
import json
import pathlib
import time


def main():
    import torch
    from schedulefree import AdamWScheduleFree

    from moonshine_it.config import load_config, resolve_profile
    from moonshine_it.model_io import load_model_and_processor
    from moonshine_it.train_loop import (ASRDataset, Collator, MultiASRDataset,
                                         quick_eval_wer)

    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--ckpt", default="results/train-final/checkpoint-2997")
    ap.add_argument("--out", default="results/lr-probe/results.jsonl")
    args = ap.parse_args()

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    cfg = load_config()
    rp = resolve_profile(cfg, "rocm12g", "final")
    tcfg = cfg["training"]

    model, proc = load_model_and_processor(cfg, device="cuda", dtype="bf16",
                                           model_path=args.ckpt)

    # Same mix and same curriculum stage the live run was in (stage 1, <=10s).
    parts = []
    for name in rp.datasets:
        root = pathlib.Path("data/prepared") / name
        parts.append(ASRDataset(root / "train.jsonl", root, cfg,
                                max_audio_s=10.0, augment=True,
                                seed=tcfg["shuffle_seed"]))
    dataset = MultiASRDataset(parts)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=rp.batch_size, shuffle=True,
        num_workers=rp.num_workers, collate_fn=Collator(proc), drop_last=True,
        persistent_workers=True, pin_memory=True,
        generator=torch.Generator().manual_seed(tcfg["shuffle_seed"]))

    # Fresh optimizer per arm => its warmup counter starts at 0. The live run
    # restores it (verified k=13750, scheduled_lr=5e-5), so leaving the
    # configured 500-step warmup here would hold every arm in warmup for its
    # entire budget and blunt the only variable under test.
    opt = AdamWScheduleFree(model.parameters(), lr=args.lr,
                            betas=tuple(tcfg["betas"]),
                            weight_decay=tcfg["weight_decay"],
                            warmup_steps=10)

    val_root = pathlib.Path("data/prepared/mls")
    val_manifest = val_root / "validation.jsonl"

    model.eval()
    wer_before = quick_eval_wer(model, proc, val_manifest, val_root, cfg,
                                split_name="mls/validation", iterate="y")
    model.train()
    opt.train()

    losses, step, skipped, t0 = [], 0, 0, time.time()
    for batch in loader:
        if step >= args.steps:
            break
        batch = {k: v.to("cuda") for k, v in batch.items()}
        for k, v in batch.items():
            if v.dtype == torch.float32:
                batch[k] = v.to(torch.bfloat16)
        loss = model(**batch).loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn) or float(gn) > tcfg["max_grad_norm_skip"]:
            opt.zero_grad(set_to_none=True)
            skipped += 1
            continue
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            print(f"  lr={args.lr:g} step {step}/{args.steps} "
                  f"loss {sum(losses[-100:]) / 100:.4f}", flush=True)

    model.eval()
    wer_after = quick_eval_wer(model, proc, val_manifest, val_root, cfg,
                               split_name="mls/validation", iterate="y")

    def window(a, b):
        chunk = losses[a:b]
        return sum(chunk) / max(1, len(chunk))

    rec = {"lr": args.lr, "steps": step, "skipped": skipped,
           "loss_first100": round(window(0, 100), 4),
           "loss_last100": round(window(max(0, step - 100), step), 4),
           "loss_delta": round(window(0, 100) - window(max(0, step - 100), step), 4),
           "wer_before": round(wer_before["eval_wer"], 2),
           "wer_after": round(wer_after["eval_wer"], 2),
           "wer_delta": round(wer_after["eval_wer"] - wer_before["eval_wer"], 2),
           "eval_n": wer_after["eval_n"], "eval_split": "mls/validation",
           "iterate": "y", "ckpt": args.ckpt, "warmup_steps": 10,
           "wall_s": round(time.time() - t0, 1)}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
