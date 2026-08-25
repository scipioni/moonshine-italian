"""Spikes: gradient path, tokenizer, baseline eval — recorded gates.

Records land in results/spike/:
  grad.json, tokenizer.json, baseline.json, verdict.json

verdict.json drives the fallback latch (see gates.py): a failed verdict must
be answered by an explicit selected_base change in config.yaml before
training runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, load_config, resolve_profile
from moonshine_it.model_io import load_model_and_processor, results_dir

ITALIAN_SAMPLE = (
    "perché l'acqua dell'amico è più bella così, à è ì ò ù é "
    "un po' di più vent'anni città perché però così"
)


def spike_grad(cfg, profile: str, limit_audio_s: float = 5.0) -> dict:
    import numpy as np
    import torch

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from moonshine_it.download import model_dir

    rp = resolve_profile(cfg, profile, "smoke")
    # fp32 master weights + autocast compute: pure-bf16 weights hit dtype
    # mismatches inside the streaming encoder path; autocast is the supported
    # mixed-precision route.
    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16}.get(rp.precision)
    path = model_dir(cfg)
    record: dict = {"spike": "grad", "profile": profile, "precision": rp.precision}
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(path), local_files_only=True
        ).to(rp.device)
        proc = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(int(16000 * limit_audio_s)) * 0.05).astype("float32")
        inputs = proc(audio=audio, text=ITALIAN_SAMPLE, return_tensors="pt",
                      sampling_rate=16000)
        inputs = {k: v.to(rp.device) for k, v in inputs.items()}
        import contextlib

        ctx = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
               if autocast_dtype else contextlib.nullcontext())
        with ctx:
            out = model(input_values=inputs["input_values"],
                        attention_mask=inputs.get("attention_mask"),
                        labels=inputs["labels"])
            loss = out.loss
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_norm = float(torch.linalg.vector_norm(
            torch.stack([g.detach().float().norm() for g in grads])))
        record.update({
            "ok": bool(torch.isfinite(loss).item()) and grad_norm > 0,
            "loss": float(loss.item()),
            "params_with_grad": len(grads),
            "grad_norm": grad_norm,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)
            if rp.device.startswith("cuda") else None,
            "model_class": type(model).__name__,
        })
    except Exception as exc:
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def spike_tokenizer(cfg) -> dict:
    from transformers import AutoTokenizer

    from moonshine_it.download import model_dir

    record: dict = {"spike": "tokenizer"}
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir(cfg)), local_files_only=True)
        ref = ITALIAN_SAMPLE.lower()
        ids = tok(ref, add_special_tokens=True)["input_ids"]
        decoded = tok.decode(ids, skip_special_tokens=True)
        exact = decoded.strip() == ref.strip()
        toks = tok.tokenize(ref)
        vocab = tok.vocab_size
        record.update({
            "ok": bool(exact),
            "exact_roundtrip": bool(exact),
            "decoded": decoded,
            "vocab_size": vocab,
            "n_tokens": len(toks),
            "chars_per_token": round(len(ref) / max(1, len(toks)), 2),
            "unknown_tokens": [t for t in toks if t in ("<unk>",)],
        })
    except Exception as exc:
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def spike_baseline(cfg, profile: str, limit: int) -> dict:
    from moonshine_it.evaluate import evaluate_manifest
    from moonshine_it.model_io import load_model_and_processor

    rp = resolve_profile(cfg, profile, "smoke")
    streaming_cfg = cfg["evaluation"]["streaming"]
    smoke_root = REPO_ROOT / cfg["smoke"]["slice_manifest"]
    audio_root = smoke_root / "audio"
    manifest = smoke_root / "test.jsonl"
    if not manifest.exists():
        return {"spike": "baseline", "ok": False,
                "error": f"smoke slice missing: {manifest} (run prepare + slice-smoke)"}

    model, proc = load_model_and_processor(cfg, device=rp.device, dtype="bf16")
    record: dict = {"spike": "baseline", "model": cfg["base_model"]["id"], "modes": {}}
    try:
        for mode in ("full", "streaming"):
            res = evaluate_manifest(
                model, proc, manifest, audio_root,
                mode=mode, streaming_cfg=streaming_cfg, limit=limit,
                model_name=cfg["base_model"]["id"], dataset="fleurs-it", split="smoke-test",
            )
            (results_dir(cfg, "eval") / f"baseline_{mode}.json").write_text(res.to_json())
            record["modes"][mode] = {"wer": res.wer, "cer": res.cer, **res.extra}
        record["ok"] = True
    except Exception as exc:
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        del model
    return record


def aggregate_verdict(records: list[dict]) -> dict:
    ok = all(r.get("ok") for r in records if r)
    return {"ok": ok, "spikes": {r["spike"]: r.get("ok") for r in records if r},
            "fallback_required": not ok}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["grad", "tokenizer", "baseline", "verdict", "all"])
    parser.add_argument("--profile", default=None,
                        help="hardware profile (default: auto-detect)")
    parser.add_argument("--limit", type=int, default=10,
                        help="samples for baseline eval")
    args = parser.parse_args(argv)

    cfg = load_config()
    profile = args.profile or "rocm12g"
    out = results_dir(cfg, "spike")

    if args.what in ("grad", "all"):
        rec = spike_grad(cfg, profile)
        (out / "grad.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"spike grad: ok={rec['ok']} loss={rec.get('loss')} "
              f"grad_norm={rec.get('grad_norm') and round(rec['grad_norm'], 1)} "
              f"vram={rec.get('peak_vram_mb')}MB")

    if args.what in ("tokenizer", "all"):
        rec = spike_tokenizer(cfg)
        (out / "tokenizer.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"spike tokenizer: ok={rec['ok']} exact={rec.get('exact_roundtrip')} "
              f"vocab={rec.get('vocab_size')}")

    if args.what in ("baseline", "all"):
        rec = spike_baseline(cfg, profile, args.limit)
        (out / "baseline.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        modes = rec.get("modes", {})
        print(f"spike baseline: ok={rec['ok']} " +
              " ".join(f"{m}={v.get('wer')}%WER" for m, v in modes.items()) +
              (f" err={rec.get('error', '')}" if not rec.get("ok") else ""))

    if args.what == "verdict" or args.what == "all":
        records = []
        for name in ("grad", "tokenizer", "baseline"):
            p = out / f"{name}.json"
            records.append(json.loads(p.read_text()) if p.exists() else {"spike": name, "ok": False, "missing": True})
        verdict = aggregate_verdict(records)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        print("spike verdict:", json.dumps(verdict["spikes"]),
              "fallback_required =", verdict["fallback_required"])
        return 0 if verdict["ok"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
