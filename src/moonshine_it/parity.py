"""Parity verification: ONNX graphs vs PyTorch, per graph.

Also cross-checks the exported decoder reimplementation against the HF
model's own forward (logits argmax must match on a teacher-forced prefix).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moonshine_it.config import REPO_ROOT, load_config
from moonshine_it.download import model_dir
from moonshine_it.evaluate import load_audio
from moonshine_it.export import (CrossKVWrapper, _AsinhExportSafe,
                                 AdapterWrapper, EncoderWrapper, _make_decoder_class)
from moonshine_it.model_io import load_model_and_processor


def _ort_session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel(0)
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def check_parity(model_path: Path | None, export_dir: Path, ref_audio_path: Path,
                 tolerance: float) -> dict:
    import torch

    cfg = load_config()
    if model_path is None:
        model_path = model_dir(cfg)
    model, proc = load_model_and_processor(cfg, model_path=model_path, device="cpu")
    model.eval()
    model.model.encoder.embedder.comp = _AsinhExportSafe(model.model.encoder.embedder.comp)

    audio = load_audio(ref_audio_path)
    pad = (-len(audio)) % 80  # frame_len: processor pads to multiple of 80
    if pad:
        audio = np.pad(audio, (0, pad))
    report: dict = {"tolerance": tolerance, "graphs": {}}

    # ---- encoder ----
    # Must pass a real attention_mask: with None the encoder skips its per-layer
    # sliding windows, and since the exported graph used to do the same, parity
    # compared two identically-degenerate graphs and passed. See EncoderWrapper.
    audio_t = torch.from_numpy(audio)[None]
    enc_mask = torch.ones_like(audio_t, dtype=torch.long)
    with torch.no_grad():
        enc_pt = model.model.encoder(input_values=audio_t,
                                     attention_mask=enc_mask).last_hidden_state.numpy()
    sess = _ort_session(export_dir / "encoder.onnx")
    enc_onx = sess.run(None, {"input_values": audio[None].astype(np.float32)})[0]
    diff = float(np.abs(enc_pt - enc_onx).max())
    report["graphs"]["encoder"] = {"max_abs_diff": diff, "ok": diff <= tolerance,
                                   "shape": list(enc_pt.shape)}

    # ---- adapter ----
    with torch.no_grad():
        pt = AdapterWrapper(model)(torch.from_numpy(enc_pt)).numpy()
    sess = _ort_session(export_dir / "adapter.onnx")
    onx = sess.run(None, {"enc_hidden": enc_onx.astype(np.float32)})[0]
    diff = float(np.abs(pt - onx).max())
    report["graphs"]["adapter"] = {"max_abs_diff": diff, "ok": diff <= tolerance}

    # ---- cross_kv ----
    with torch.no_grad():
        pt_outs = CrossKVWrapper(model)(torch.from_numpy(onx))
        pt_outs = [t.numpy() for t in pt_outs]
    sess = _ort_session(export_dir / "cross_kv.onnx")
    onx_outs = sess.run(None, {"adapted": onx.astype(np.float32)})
    diffs = [float(np.abs(a - b).max()) for a, b in zip(pt_outs, onx_outs)]
    report["graphs"]["cross_kv"] = {"max_abs_diff": max(diffs),
                                    "ok": max(diffs) <= tolerance,
                                    "n_outputs": len(onx_outs)}

    # ---- decoder_kv (and reimplementation check vs HF) ----
    n_layers = len(model.model.decoder.layers)
    dec_cfg = model.model.decoder.layers[0].encoder_attn.config
    heads = dec_cfg.num_key_value_heads
    head_dim = model.model.decoder.layers[0].encoder_attn.head_dim
    text = "questo è un test di parità per il modello italiano"
    ids = proc.tokenizer(text, return_tensors="pt")["input_ids"]
    bos = torch.tensor([[proc.tokenizer.bos_token_id]])
    prefix = torch.cat([bos, ids], dim=1)

    cross_flat = []
    for k_or_v in range(2):
        for i in range(n_layers):
            cross_flat.append(onx_outs[i * 2 + k_or_v])
    cross_k = np.stack(cross_flat[:n_layers])            # [L, 1, H, F, D]
    cross_v = np.stack(cross_flat[n_layers:])            # [L, 1, H, F, D]
    past_k = np.zeros((n_layers, 1, heads, 0, head_dim), dtype=np.float32)
    past_v = np.zeros((n_layers, 1, heads, 0, head_dim), dtype=np.float32)

    cls = _make_decoder_class(n_layers)
    wrapper = cls(model, heads, head_dim, n_layers).eval()
    with torch.no_grad():
        pt_out = wrapper(prefix, torch.from_numpy(cross_k),
                         torch.from_numpy(cross_v), torch.from_numpy(past_k),
                         torch.from_numpy(past_v))
    pt_logits = pt_out[0].numpy()

    # reimplementation vs HF forward (teacher-forced)
    with torch.no_grad():
        hf = model(input_values=audio_t,
                   attention_mask=enc_mask, decoder_input_ids=prefix).logits.numpy()
    hf_diff = float(np.abs(hf - pt_logits).max())
    report["reimpl_vs_hf_logits"] = {
        "max_abs_diff": hf_diff,
        "argmax_match": bool((hf.argmax(-1) == pt_logits.argmax(-1)).all()),
    }

    sess = _ort_session(export_dir / "decoder_kv.onnx")
    feed = {"input_ids": prefix.numpy(), "cross_k": cross_k,
            "cross_v": cross_v, "past_k": past_k, "past_v": past_v}
    onx = sess.run(None, feed)
    diff = float(np.abs(pt_logits - onx[0]).max())
    present_diff = float(np.abs(pt_out[1].numpy() - onx[1]).max())
    out_names = [o.name for o in sess.get_outputs()]
    kv_ok = any(n.startswith("present_k") for n in out_names) and \
        any(n.startswith("present_v") for n in out_names)
    report["graphs"]["decoder_kv"] = {
        "max_abs_diff": diff, "ok": diff <= tolerance,
        "present_kv_diff": present_diff,
        "present_kv_outputs": kv_ok,
    }

    # multi-token step with nonempty past (cache path)
    past_len = 3
    past_k3 = np.zeros((n_layers, 1, heads, past_len, head_dim), dtype=np.float32)
    past_v3 = np.zeros((n_layers, 1, heads, past_len, head_dim), dtype=np.float32)
    with torch.no_grad():
        pt_out2 = wrapper(prefix, torch.from_numpy(cross_k),
                          torch.from_numpy(cross_v), torch.from_numpy(past_k3),
                          torch.from_numpy(past_v3))
    feed2 = dict(feed, past_k=past_k3, past_v=past_v3)
    onx2 = sess.run(None, feed2)
    diff2 = float(np.abs(pt_out2[0].numpy() - onx2[0]).max())
    report["graphs"]["decoder_kv_cached"] = {
        "max_abs_diff": diff2, "ok": diff2 <= tolerance,
    }
    report["ok"] = all(g["ok"] for g in report["graphs"].values()) and \
        report["reimpl_vs_hf_logits"]["argmax_match"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--ref-audio", default=None)
    args = parser.parse_args(argv)
    cfg = load_config()
    name = Path(args.model).name if args.model else "base"
    export_dir = Path(args.export_dir) if args.export_dir else \
        REPO_ROOT / cfg["paths"]["results"] / "export" / name
    ref = Path(args.ref_audio) if args.ref_audio else \
        REPO_ROOT / cfg["smoke"]["slice_manifest"] / "audio" / \
        sorted((REPO_ROOT / cfg["smoke"]["slice_manifest"] / "audio").glob("*.wav"))[0].name
    report = check_parity(Path(args.model) if args.model else None, export_dir, ref,
                          cfg["export"]["onnx"]["tolerance"])
    out = export_dir / "parity.json"
    out.write_text(json.dumps(report, indent=2))
    for g, r in report["graphs"].items():
        print(f"parity[{g}]: max_diff={r['max_abs_diff']:.2e} ok={r['ok']}")
    print(f"reimpl vs HF logits: max_diff={report['reimpl_vs_hf_logits']['max_abs_diff']:.2e} "
          f"argmax_match={report['reimpl_vs_hf_logits']['argmax_match']}")
    print("parity:", "OK" if report["ok"] else "FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
