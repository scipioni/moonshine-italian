"""Profile-aware training loop: schedule-free AdamW, curriculum stages,
chunked augmentation, checkpointing, TensorBoard.

One code path for smoke and final profiles (only config differs).
Resume: checkpoints carry trainer_state.json + optimizer.pt.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from moonshine_it.config import REPO_ROOT, load_config, resolve_profile
from moonshine_it.evaluate import load_audio
from moonshine_it.model_io import load_model_and_processor
from moonshine_it.prepare import plan_chunks, split_text_for_chunks


class ASRDataset:
    """Audio+text pairs from a prepared manifest, with chunked augmentation.

    Augmentation re-splits an utterance into sentence-aligned sub-chunks
    (same machinery as preparation) and keeps one at random — exposing the
    model to partial utterances as seen in chunked streaming.
    """

    def __init__(self, manifest: Path, audio_root: Path, cfg: dict,
                 *, max_audio_s: float | None = None, augment: bool = False,
                 seed: int = 0):
        from moonshine_it.evaluate import load_manifest

        self.rows = load_manifest(manifest)
        # Filter out corrupted rows with astronomically long transcripts (causes SDPA CUDA OOM)
        self.rows = [r for r in self.rows if len(r.get("text", "")) <= 500]
        if max_audio_s is not None:
            self.rows = [r for r in self.rows if r["duration_s"] <= max_audio_s]
        self.audio_root = audio_root
        self.cfg = cfg
        self.augment = augment
        self.seed = seed
        self.min_len = int(cfg["preparation"]["min_duration_s"] * 16000)
        self.max_len = int(cfg["preparation"]["max_duration_s"] * 16000)
        aug = cfg["training"]["chunked_augmentation"]
        self.augment_p = aug.get("probability", 0.0) if augment else 0.0
        self.min_fraction = aug.get("min_fraction", 0.4)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        audio = load_audio(self.audio_root / row["audio"])
        text = row["text"]
        rng = random.Random(f"{self.seed}:{i}")
        if self.augment_p and rng.random() < self.augment_p and len(audio) > self.min_len * 2:
            # pretend this utterance is 2x max: split into 2 aligned chunks
            half = max(self.min_len, len(audio) // 2)
            chunks = plan_chunks([(0, len(audio))], len(audio), self.min_len,
                                 max(half, self.min_len + 1))
            if len(chunks) > 1:
                texts = split_text_for_chunks(text, chunks)
                if texts:
                    ci = rng.randrange(len(chunks))
                    cs, ce = chunks[ci]
                    frac = (ce - cs) / len(audio)
                    if frac >= self.min_fraction:
                        audio, text = audio[cs:ce], texts[ci]
        return {"audio": audio, "text": text}


class MultiASRDataset:
    """Concatenates several ASRDataset instances (each its own manifest +
    audio_root) into one flat, shuffleable dataset.

    Keeps a flat `.rows` list mirroring ASRDataset's, so curriculum staging
    (which filters `dataset.rows` by duration_s to build a Subset) works
    unchanged whether training on one dataset or several.
    """

    def __init__(self, datasets: list[ASRDataset]):
        if not datasets:
            raise ValueError("MultiASRDataset needs at least one dataset")
        self.datasets = datasets
        self.rows = [r for d in datasets for r in d.rows]
        self._offsets = []
        total = 0
        for d in datasets:
            self._offsets.append(total)
            total += len(d)
        self._len = total

    def __len__(self):
        return self._len

    def __getitem__(self, i: int):
        import bisect

        di = bisect.bisect_right(self._offsets, i) - 1
        return self.datasets[di][i - self._offsets[di]]


class Collator:
    def __init__(self, proc):
        self.proc = proc

    def __call__(self, batch):
        import torch

        inputs = self.proc(
            audio=[b["audio"] for b in batch],
            text=[b["text"] for b in batch],
            padding=True,
            return_tensors="pt",
            sampling_rate=16000,
        )
        labels = inputs["labels"]
        # Two label conventions the processor gets wrong for this model:
        # 1. It prepends BOS, but the loss aligns logits to labels without
        #    re-shifting (decoder_start_token supplies the input-side BOS) —
        #    training with BOS-in-labels teaches "BOS after BOS" and greedy
        #    decode collapses to empty output.
        # 2. It appends no EOS — the model then never learns to stop and
        #    babbles to the generation cap (measured: eval WER ~330% from
        #    pure insertions while transcription itself is correct).
        tok = self.proc.tokenizer
        bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
        if labels.numel() and bool((labels[:, 0] == bos).all()):
            labels = labels[:, 1:]
        # grow by one column so even full-length rows get an EOS slot
        labels = torch.nn.functional.pad(labels, (0, 1), value=pad)
        pad_mask = labels == pad
        first_pad = pad_mask.float().argmax(dim=1)
        labels[torch.arange(labels.shape[0]), first_pad] = eos
        labels = labels.masked_fill(labels == pad, -100)
        labels = labels.masked_fill(labels == bos, -100)
        inputs["labels"] = labels
        return {k: v for k, v in inputs.items()}


def stage_for_step(curriculum: list[dict], step: int) -> dict | None:
    """Curriculum stages list cumulative step budgets."""
    cumulative = 0
    for stage in curriculum:
        cumulative += stage["steps"]
        if step < cumulative:
            return stage
    return curriculum[-1] if curriculum else None


def save_checkpoint(model, proc, optimizer, out_dir: Path, step: int,
                    metrics: dict) -> Path:
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    import torch

    # Schedule-free AdamW keeps two live views of the weights in p.data: "y"
    # (raw, used while .train()) and "x" (a slow-moving average, only
    # materialized under .eval()) -- .eval()/.train() are inverse in-place
    # transforms of p.data driven by the unchanged optimizer state['z'], and
    # param_groups['train_mode'] records which one p currently holds.
    # quick_eval_wer above is measured on "x" (called right after
    # optimizer.eval()); without this, the checkpoint saved a few lines later
    # captured "y" instead -- a DIFFERENT set of weights than what was just
    # scored, than what best_metric.json compares against, and than what
    # ships to export/inference. Toggling to eval mode for the save (and
    # saving optimizer.pt in the same window, so its train_mode flag agrees
    # with which iterate model.safetensors holds -- resume's optimizer.train()
    # call only re-derives "y" correctly if it does) makes the saved
    # checkpoint the same weights the reported eval_wer describes.
    was_training = optimizer.param_groups[0]["train_mode"]
    if was_training:
        optimizer.eval()
    model.save_pretrained(ckpt)
    proc.save_pretrained(ckpt)
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    if was_training:
        optimizer.train()
    (ckpt / "trainer_state.json").write_text(json.dumps(
        {"global_step": step, "metrics": metrics}, indent=2))
    best = out_dir / "checkpoint-best"
    marker = out_dir / "best_metric.json"
    prev_best = json.loads(marker.read_text()) if marker.exists() else None
    score = metrics.get("eval_wer")
    if score is not None and (prev_best is None or score < prev_best["eval_wer"]):
        # O(1) promotion: point the best-checkpoint at the winning step via a
        # symlink instead of copying the (potentially ~1.6 GB) checkpoint.
        if best.is_symlink():
            best.unlink()
        elif best.exists():
            import shutil

            shutil.rmtree(best)  # pre-existing real dir from before this change
        best.symlink_to(ckpt, target_is_directory=True)
        marker.write_text(json.dumps({"global_step": step, "eval_wer": score}, indent=2))
    return ckpt


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None
    ckpts = sorted(
        (p for p in out_dir.glob("checkpoint-[0-9]*") if p.is_dir()
         and (p / "trainer_state.json").exists()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return ckpts[-1] if ckpts else None


def quick_eval_wer(model, proc, manifest: Path, audio_root: Path, cfg: dict,
                   limit: int | None = None) -> float:
    import jiwer

    from moonshine_it.evaluate import transcribe_full

    if limit is None:
        limit = cfg["evaluation"].get("in_loop_samples", 64)
    rows = [json.loads(l) for l in manifest.read_text().splitlines()][:limit]
    max_tps = cfg["evaluation"]["streaming"]["max_tokens_per_second"]
    refs, hyps = [], []
    for row in rows:
        audio = load_audio(audio_root / row["audio"])
        hyp = transcribe_full(model, proc, audio, max_tokens_per_s=max_tps)
        from moonshine_it.normalize_it import normalize_text

        refs.append(normalize_text(row["text"], expand_nums=False))
        hyps.append(normalize_text(hyp, expand_nums=False))
    return float(jiwer.wer(refs, hyps)) * 100


def train(
    hardware_profile: str,
    training_profile: str,
    *,
    dry_run_steps: int | None = None,
    resume: bool = True,
) -> Path:
    import torch
    from schedulefree import AdamWScheduleFree as ScheduleFreeAdamW
    from torch.utils.data import DataLoader, Subset
    from torch.utils.tensorboard import SummaryWriter

    # torch-2.12 ROCm: the SDPA mem-efficient backward kernel intermittently
    # produces inf gradients (decoder LayerNorm weights; measured 4/6 batches
    # on MLS-it data -> NaN losses). The AOTriton flash path is numerically
    # fine but ~300x slower here; the math fallback is both correct (~1/8
    # inf-free-skip rate) and fast under bf16 autocast.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    cfg = load_config()
    rp = resolve_profile(cfg, hardware_profile, training_profile)
    out_dir = rp.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(
        json.dumps({**rp.run_metadata(),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "dry_run": dry_run_steps is not None}, indent=2))

    model, proc = load_model_and_processor(cfg, device=rp.device, dtype=rp.precision)
    model.train()

    is_smoke = training_profile == "smoke"
    smoke_root = REPO_ROOT / cfg["smoke"]["slice_manifest"]
    if is_smoke:
        manifest = smoke_root / "train.jsonl"
        audio_root = smoke_root / "audio"
        val_manifest = smoke_root / "test.jsonl"
        val_audio = smoke_root / "audio"
        if not manifest.exists():
            raise SystemExit(f"training manifest missing: {manifest} — run prepare first")
        dataset = ASRDataset(manifest, audio_root, cfg, augment=True,
                             seed=cfg["smoke"]["seed"])
    else:
        # In-loop eval intentionally stays on mls/validation.jsonl regardless
        # of rp.datasets, so eval/wer is comparable across runs even as the
        # training mix changes.
        val_root = REPO_ROOT / cfg["paths"]["data"] / "prepared" / "mls"
        val_manifest = val_root / "validation.jsonl"
        val_audio = val_root

        parts = []
        for name in rp.datasets:
            data_root = REPO_ROOT / cfg["paths"]["data"] / "prepared" / name
            data_manifest = data_root / "train.jsonl"
            if not data_manifest.exists():
                raise SystemExit(
                    f"training manifest missing: {data_manifest} — run: "
                    f"task prepare DATASET={name}")
            parts.append(ASRDataset(data_manifest, data_root, cfg, augment=True,
                                    seed=cfg["smoke"]["seed"]))
            print(f"train[{training_profile}]: +{name} "
                  f"({len(parts[-1])} rows, {data_manifest})")
        dataset = parts[0] if len(parts) == 1 else MultiASRDataset(parts)
    # Deliberately separate from cfg["smoke"]["seed"] (which still seeds
    # per-item augmentation via ASRDataset above) -- see training.shuffle_seed
    # in config.yaml for why. MOONSHINE_SHUFFLE_SEED lets a retry wrapper vary
    # this per attempt after a hard crash, since resuming with an unchanged
    # seed deterministically re-walks into the same fault (confirmed: it
    # recurred even after one reseed, at a different but nearby step).
    shuffle_seed = int(os.environ.get(
        "MOONSHINE_SHUFFLE_SEED",
        cfg["training"].get("shuffle_seed", cfg["smoke"]["seed"])))
    loader = DataLoader(
        dataset,
        batch_size=rp.batch_size,
        shuffle=True,
        num_workers=rp.num_workers,
        collate_fn=Collator(proc),
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        generator=torch.Generator().manual_seed(shuffle_seed),
    )

    tcfg = cfg["training"]

    start_step = 0
    latest = find_latest_checkpoint(out_dir)
    opt_state_path = None
    if resume and latest is not None:
        state = json.loads((latest / "trainer_state.json").read_text())
        start_step = state["global_step"]
        from transformers import AutoModelForSpeechSeq2Seq

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(latest), local_files_only=True,
            dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[rp.precision],
        ).to(rp.device)
        model.train()
        opt_state_path = latest / "optimizer.pt"

    # The optimizer MUST be built after the resume reload. Building it first
    # binds it to the pre-reload parameter tensors, which from_pretrained then
    # discards -- optimizer.step() silently becomes a no-op (it iterates params
    # whose .grad is None) while loss.backward() writes grads to the new module.
    # Symptom when this regresses: flat loss, byte-identical checkpoints, and a
    # frozen eval metric, with no error anywhere.
    optimizer = ScheduleFreeAdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        betas=tuple(tcfg["betas"]),
        weight_decay=tcfg["weight_decay"],
        warmup_steps=tcfg["warmup_steps"],
    )
    if opt_state_path is not None:
        optimizer.load_state_dict(torch.load(opt_state_path,
                                             map_location=rp.device,
                                             weights_only=False))
        print(f"resume: from step {start_step} ({latest.name})")

    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if owned != {id(p) for p in model.parameters()}:
        raise RuntimeError(
            "optimizer is not bound to the model's parameters — optimizer.step() "
            "would be a no-op and training would silently do nothing")

    max_steps = dry_run_steps if dry_run_steps is not None else rp.max_steps
    writer = SummaryWriter(log_dir=str(out_dir / "runs"))
    autocast_dtype = {"bf16": torch.bfloat16, "fp32": None}[rp.precision]
    import contextlib

    print(f"train[{training_profile}]: steps {start_step}..{max_steps}, "
          f"batch={rp.batch_size}, device={rp.device} ({rp.accelerator_kind}), "
          f"samples={len(dataset)}, shuffle_seed={shuffle_seed}")

    step = start_step
    epoch = 0
    t_start = time.time()
    while step < max_steps:
        # curriculum: rebuild dataset view for the current stage
        stage = stage_for_step(rp.curriculum, step) if rp.curriculum else None
        max_audio_s = stage["max_audio_s"] if stage else None
        if max_audio_s:
            view = Subset(dataset, [i for i, r in enumerate(dataset.rows)
                                    if r["duration_s"] <= max_audio_s])
            stage_loader = DataLoader(view, batch_size=rp.batch_size, shuffle=True,
                                      num_workers=rp.num_workers,
                                      collate_fn=Collator(proc), drop_last=True,
                                      persistent_workers=True, pin_memory=True,
                                      generator=torch.Generator().manual_seed(
                                          shuffle_seed + step))
        else:
            stage_loader = loader

        optimizer.train()
        skipped = 0
        # Gradient accumulation: batch=8 with no accumulation gave a very
        # noisy per-step gradient (frequent large gnorm spikes, e.g. 50-200+
        # against a typical 10-60, needing aggressive clip_grad_norm_ every
        # step) -- a plausible contributor to a run where loss barely moved
        # and eval WER got noisier/worse over 72k steps rather than
        # improving. accum_steps averages the gradient over more samples
        # before each update, at the cost of accum_steps x wall time per
        # optimizer step. 1 (default) reproduces the original behavior
        # exactly.
        accum_steps = max(1, int(tcfg.get("grad_accum_steps", 1)))
        micro = 0
        accum_loss = 0.0
        for batch in stage_loader:
            if step >= max_steps:
                break
            batch = {k: v.to(rp.device) for k, v in batch.items()}
            # bf16 weights require bf16 inputs (embedder linear dtype match)
            if autocast_dtype is not None:
                batch["input_values"] = batch["input_values"].to(autocast_dtype)
            ctx = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
                   if autocast_dtype else contextlib.nullcontext())
            with ctx:
                out = model(input_values=batch["input_values"],
                            attention_mask=batch.get("attention_mask"),
                            labels=batch["labels"])
                loss = out.loss
            # Untuned-on-Italian checkpoints produce huge LayerNorm gradients
            # (confidently-wrong predictions over a 32k vocab; measured
            # ~4e7 pre-clip) which overflow to inf across the deep backward
            # chain. Skip any micro-batch whose loss is already non-finite so
            # bad weights can't poison the run -- this drops the WHOLE
            # accumulation window (zero_grad clears any finite micro-batches
            # already summed into it too), simplest safe behavior.
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                accum_loss = 0.0
                skipped += 1
                if skipped <= 10 or skipped % 100 == 0:
                    print(f"  step {step}: non-finite loss, skipping "
                          f"({skipped} skipped so far)", flush=True)
                continue
            (loss / accum_steps).backward()
            micro += 1
            accum_loss += loss.item()
            if micro < accum_steps:
                continue
            mean_loss = accum_loss / accum_steps
            micro = 0
            accum_loss = 0.0
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue
            # clip_grad_norm_ returns the PRE-clip norm; the stored .grad is
            # already clipped to max_norm=1.0 above regardless. A large-but-
            # finite pre-clip norm still means this step's gradient direction
            # is coming from a batch the model is confidently wrong about --
            # skip the update entirely rather than take even a clipped step
            # from it. See max_grad_norm_skip in config.yaml for why.
            max_grad_norm_skip = tcfg.get("max_grad_norm_skip")
            if max_grad_norm_skip is not None and float(grad_norm) > max_grad_norm_skip:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                if skipped <= 10 or skipped % 100 == 0:
                    print(f"  step {step}: gnorm {float(grad_norm):.2f} > "
                          f"{max_grad_norm_skip}, skipping ({skipped} skipped so far)",
                          flush=True)
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            writer.add_scalar("train/loss", mean_loss, step)
            writer.add_scalar("train/grad_norm",
                              float(grad_norm) if torch.is_tensor(grad_norm)
                              else grad_norm, step)
            if step % 10 == 0:
                print(f"  step {step}/{max_steps} loss {mean_loss:.4f} "
                      f"gnorm {float(grad_norm):.2f} "
                      f"({(time.time()-t_start):.0f}s)", flush=True)

            if step % rp.eval_steps == 0:
                torch.cuda.synchronize()  # scoped: settle the step before eval
                optimizer.eval()
                model.eval()
                wer = quick_eval_wer(model, proc, val_manifest, val_audio, cfg)
                model.train()
                optimizer.train()
                writer.add_scalar("eval/wer", wer, step)
                print(f"  step {step} eval WER {wer:.1f}%", flush=True)
                metrics = {"eval_wer": wer}
            if step % rp.save_steps == 0:
                torch.cuda.synchronize()  # scoped: settle the step before saving
                metrics = locals().get("metrics", {}) or {}
                save_checkpoint(model, proc, optimizer, out_dir, step, metrics)
        epoch += 1

    metrics = locals().get("metrics", {}) or {}
    save_checkpoint(model, proc, optimizer, out_dir, step, metrics)
    writer.close()

    # per-hardware performance gate: record measured wall-time/step and fail
    # (mirroring eval gates) if it falls below the configured steps-per-second.
    total_s = time.time() - t_start
    done_steps = step - start_step
    steps_per_second = done_steps / total_s if total_s > 0 else 0.0
    meta = json.loads((out_dir / "run_metadata.json").read_text())
    meta["wall_time_s"] = round(total_s, 1)
    meta["steps_per_second"] = round(steps_per_second, 4)
    meta["wall_time_per_step_s"] = round(total_s / done_steps, 4) if done_steps else None
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    if rp.steps_per_second_min is not None and done_steps > 0:
        if steps_per_second < rp.steps_per_second_min:
            raise SystemExit(
                f"training-performance gate failed on {hardware_profile}: "
                f"measured {steps_per_second:.3f} steps/s "
                f"(allowed ≥ {rp.steps_per_second_min}); wall {total_s:.0f}s / "
                f"{done_steps} steps. Optimize the loop (see "
                f"openspec/changes/optimize-training-performance)."
            )
        print(f"  performance gate: {steps_per_second:.3f} steps/s "
              f"(allowed ≥ {rp.steps_per_second_min}) — pass")

    print(f"train[{training_profile}]: done at step {step} -> {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="rocm12g")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "final"])
    parser.add_argument("--dry-run-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    from moonshine_it.gates import require_smoke_ok, require_spike_ok

    require_spike_ok()               # fallback latch: no training without spikes
    if args.profile == "final":       # final-train latch: process validated by smoke
        require_smoke_ok()
    out = train(args.hardware, args.profile,
                dry_run_steps=args.dry_run_steps, resume=not args.no_resume)
    print(f"output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
