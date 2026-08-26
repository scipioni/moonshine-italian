"""ONNX export for a Moonshine streaming checkpoint (4-graph decomposition).

Graphs (mirror the upstream runtime roles; frontend merged into encoder):
  encoder.onnx    input_values [1,N]            -> enc_hidden [1,T,enc_dim]
  adapter.onnx    enc_hidden [1,T,enc_dim]      -> adapted    [1,T,dec_dim]
  cross_kv.onnx   adapted                       -> cross K,V per layer [1,H,T,D]
  decoder_kv.onnx input_ids [1,S] + cross K/V + past self K/V (P)
                  -> logits [1,S,V] + present self K/V

Parity is checked per graph against the PyTorch model (tolerance from
config.yaml export.onnx.tolerance). Batch 1, no padding: matches the
single-clip PyTorch eval path. The encoder runs *masked* — it must be given an
attention_mask or it skips its per-layer sliding windows entirely; see
EncoderWrapper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from moonshine_it.config import REPO_ROOT, load_config
from moonshine_it.model_io import load_model_and_processor

N_LAYERS = None  # set at export time


class _AsinhExportSafe(torch.nn.Module):
    """aten::asinh has no legacy-exporter mapping; use its closed form.

    asinh(y) = log(y + sqrt(y^2 + 1)) — numerically identical for our range.
    """

    def __init__(self, orig):
        super().__init__()
        self.log_k = orig.log_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.exp(self.log_k) * x
        return torch.log(y + torch.sqrt(y * y + 1.0))


def _rope_cos_sin(inv_freq, hidden_states, position_ids):
    """Replicate MoonshineStreamingRotaryEmbedding.forward (export-safe).

    Uses broadcast multiply instead of expand+matmul to avoid Expand shape issues.
    """
    if inv_freq is not hidden_states.device or inv_freq.dtype != hidden_states.dtype:
        inv_freq = inv_freq.to(device=hidden_states.device, dtype=torch.float)
    inv_freq = inv_freq[None, :, None]
    pos = position_ids[:, None, :].float()
    freqs = (inv_freq * pos).transpose(1, 2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _apply_rope(q, k, cos, sin):
    """Interleaved RoPE (matches moonshine_streaming.apply_rotary_pos_emb)."""

    def rotate_half(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_out = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_out = (k_rot * cos) + (rotate_half(k_rot) * sin)
    if q_pass.shape[-1] == 0:  # full rotary: pass-through slice is empty
        return q_out, k_out
    q_full = q.clone()
    q_full[..., :rotary_dim] = q_out
    k_full = k.clone()
    k_full[..., :rotary_dim] = k_out
    return q_full, k_full


class EncoderWrapper(torch.nn.Module):
    """Encoder graph. Keeps the single-input contract the .ort runtime expects.

    The all-ones padding mask is built here rather than taken as a second graph
    input: the encoder applies its per-layer sliding windows only when
    attention_mask is not None (see MoonshineStreamingEncoder.forward), so
    passing None makes all 10 layers attend globally and silently produces a
    different encoder output (measured 7.57 max-abs drift, decoder then never
    emits EOS). Batch 1, no padding, so an all-ones mask is exact.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_values):
        mask = torch.ones_like(input_values, dtype=torch.long)
        return self.encoder(input_values=input_values,
                            attention_mask=mask).last_hidden_state


class AdapterWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        dec = model.model.decoder
        self.pos_emb = dec.pos_emb
        self.proj = dec.proj

    def forward(self, enc_hidden):
        t = enc_hidden.shape[1]
        positions = torch.arange(t, device=enc_hidden.device, dtype=torch.long)
        return self.proj(enc_hidden + self.pos_emb(positions))


class CrossKVWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.model.decoder.layers

    def forward(self, adapted):
        outs = []
        for layer in self.layers:
            attn = layer.encoder_attn
            b, t, _ = adapted.shape
            heads = attn.config.num_key_value_heads
            dim = attn.head_dim
            k = attn.k_proj(adapted).view(b, t, heads, dim).transpose(1, 2)
            v = attn.v_proj(adapted).view(b, t, heads, dim).transpose(1, 2)
            outs.extend([k, v])
        return tuple(outs)


def _decoder_step(hidden, dec, proj_out, cos, sin, mask, cross_k, cross_v,
                  past_k, past_v, heads, head_dim):
    """Shared per-layer decoder computation (export-safe, list-driven)."""
    present_k, present_v = [], []
    for i in range(len(dec.layers)):
        layer = dec.layers[i]

        residual = hidden
        h = layer.input_layernorm(hidden)
        t = layer.self_attn.q_proj(h)
        q = t.reshape(t.shape[0], t.shape[1], -1, head_dim).transpose(1, 2)
        t = layer.self_attn.k_proj(h)
        k = t.reshape(t.shape[0], t.shape[1], -1, head_dim).transpose(1, 2)
        t = layer.self_attn.v_proj(h)
        v = t.reshape(t.shape[0], t.shape[1], -1, head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        k = torch.cat([past_k[i], k], dim=2)
        v = torch.cat([past_v[i], v], dim=2)
        present_k.append(k)
        present_v.append(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) * layer.self_attn.scaling
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        hidden = residual + layer.self_attn.o_proj(out)

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        t = layer.encoder_attn.q_proj(h)
        q = t.reshape(t.shape[0], t.shape[1], -1, head_dim).transpose(1, 2)
        scores = torch.matmul(q, cross_k[i].transpose(-1, -2)) * layer.encoder_attn.scaling
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, cross_v[i]).transpose(1, 2)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        hidden = residual + layer.encoder_attn.o_proj(out)

        residual = hidden
        h = layer.final_layernorm(hidden)
        hidden = residual + layer.mlp(h)

    hidden = dec.norm(hidden)
    return proj_out(hidden), present_k, present_v


class _SlicedDecoder(torch.nn.Module):
    """Decoder wrapper that exposes only the first n_layers."""
    def __init__(self, dec, n_layers):
        super().__init__()
        self.embed_tokens = dec.embed_tokens
        self.register_buffer('rotary_emb_inv_freq', dec.rotary_emb.inv_freq)
        self.norm = dec.norm
        self.layers = torch.nn.ModuleList(list(dec.layers)[:n_layers])


_DECODER_CLASS_SRC = '''
class _DecoderStepExplicit(torch.nn.Module):
    """Decoder step with stacked per-layer K/V inputs.

    cross_k/cross_v: [n_layers, 1, H, frames, D]; past_k/past_v:
    [n_layers, 1, H, past, D]. Stacking avoids the dynamo ONNX exporter's
    arg-crosswiring with dozens of same-shape tensor parameters.
    """

    def __init__(self, model, heads, head_dim, n_layers):
        super().__init__()
        self.decoder = _SlicedDecoder(model.model.decoder, n_layers)
        self.proj_out = model.proj_out
        self.heads = heads
        self.head_dim = head_dim

    def forward(self, input_ids, cross_k, cross_v, past_k, past_v):
        dec = self.decoder
        s = input_ids.shape[1]
        past_len = past_k.shape[3]
        hidden = dec.embed_tokens(input_ids)
        position_ids = torch.arange(0, s, device=hidden.device).unsqueeze(0) + past_len
        cos, sin = _rope_cos_sin(dec.rotary_emb_inv_freq, hidden, position_ids)
        mask = None
        if s > 1:
            total = past_len + s
            q_pos = torch.arange(0, s, device=hidden.device) + past_len
            k_pos = torch.arange(0, total, device=hidden.device)
            neg = torch.full((), torch.finfo(torch.float32).min)
            mask = torch.where(k_pos[None, :] <= q_pos[:, None],
                               torch.zeros(()), neg).unsqueeze(0).unsqueeze(0)
        logits, present_k, present_v = _decoder_step(
            hidden, dec, self.proj_out, cos, sin, mask,
            cross_k.unbind(0), cross_v.unbind(0),
            past_k.unbind(0), past_v.unbind(0),
            self.heads, self.head_dim)
        return (logits, torch.stack(present_k), torch.stack(present_v))
'''


def _make_decoder_class(n_layers: int):
    src = _DECODER_CLASS_SRC
    ns = {"torch": torch, "_rope_cos_sin": _rope_cos_sin,
          "_apply_rope": _apply_rope, "_decoder_step": _decoder_step,
          "_SlicedDecoder": _SlicedDecoder,
          "__name__": "gen"}
    exec(src, ns)
    return ns["_DecoderStepExplicit"]


def _dynamic(names_axes, extra=None):
    d = dict(extra or {})
    for name, axes in names_axes.items():
        d[name] = axes
    return d


def _dims(spec):
    """Convert string dim labels to torch.export.Dim (shared per name)."""
    from torch.export import Dim
    cache: dict = {}

    def conv(x):
        if isinstance(x, str):
            if x not in cache:
                cache[x] = Dim(x)
            return cache[x]
        if isinstance(x, dict):
            return {_axis(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(conv(i) for i in x)
        return x

    def _axis(k):
        return int(k) if isinstance(k, str) and k.isdigit() else k

    return conv(spec)


def _torch_export(module, args, path, opset, input_names, output_names,
                  dynamic_axes, dynamic_shapes, prefer_dynamo=False):
    """Export with order control: legacy-first by default, dynamo-first on request.

    dynamic_shapes: tuple aligned with args (dynamo); dynamic_axes: legacy.
    prefer_dynamo: needed for the decoder graph — the legacy exporter bakes
    wrong head-count constants (3 instead of 8) into its reshape shapes.
    """
    dynamic_shapes = _dims(dynamic_shapes)
    common = dict(
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
    )
    order = ("dynamo", "legacy") if prefer_dynamo else ("legacy", "dynamo")
    last: Exception | None = None
    for mode in order:
        try:
            if mode == "legacy":
                torch.onnx.export(module, args, str(path), dynamo=False,
                                  dynamic_axes=dynamic_axes,
                                  do_constant_folding=True, **common)
            else:
                # dynamo emits opset-18 ops (variadic Split, LayerNormalization);
                # the version converter mangles them when downgrading
                dynamo_common = dict(common, opset_version=max(opset, 18))
                torch.onnx.export(module, args, str(path), dynamo=True,
                                  dynamic_shapes=dynamic_shapes, **dynamo_common)
            return
        except Exception as exc:
            last = exc
    raise last


def export_all(model_path: Path | None, out_dir: Path, opset: int) -> dict:
    cfg = load_config()
    if model_path is None:
        from moonshine_it.download import model_dir
        model_path = model_dir(cfg)
    model, _ = load_model_and_processor(cfg, model_path=model_path, device="cpu")
    model.eval()
    model.model.encoder.embedder.comp = _AsinhExportSafe(model.model.encoder.embedder.comp)
    out_dir.mkdir(parents=True, exist_ok=True)

    dec_cfg = model.config
    n_layers = len(model.model.decoder.layers)
    enc = model.model.encoder
    enc_dim = enc.config.hidden_size
    enc_heads = enc.config.num_key_value_heads if hasattr(enc.config, "num_key_value_heads") else enc.config.num_attention_heads
    enc_head_dim = getattr(enc.config, "head_dim", enc_dim // enc.config.num_attention_heads)
    dec_dim = model.model.decoder.config.hidden_size
    dec_heads = model.model.decoder.layers[0].encoder_attn.config.num_key_value_heads
    dec_head_dim = model.model.decoder.layers[0].encoder_attn.head_dim
    vocab = model.config.vocab_size

    written: dict[str, Path] = {}

    # --- encoder ---
    # prefer_dynamo: the encoder now builds its own attention_mask (see
    # EncoderWrapper), and the legacy TorchScript exporter constant-folds that
    # mask chain into fixed-size constants, freezing audio_len at the trace
    # length. torch.export keeps it symbolic.
    wrapper = EncoderWrapper(model).eval()
    path = out_dir / "encoder.onnx"
    _torch_export(
        wrapper, (torch.zeros(1, 16000, dtype=torch.float32),), path, opset,
        ["input_values"], ["enc_hidden"],
        {"input_values": {1: "audio_len"}, "enc_hidden": {1: "frames"}},
        ({1: "audio_len"},),  # dim 1 is audio length; dim 0 is batch
        prefer_dynamo=True,
    )
    written["encoder"] = path

    # --- adapter ---
    wrapper = AdapterWrapper(model).eval()
    path = out_dir / "adapter.onnx"
    _torch_export(
        wrapper, (torch.zeros(1, 32, enc_dim, dtype=torch.float32),), path, opset,
        ["enc_hidden"], ["adapted"],
        {"enc_hidden": {1: "frames"}, "adapted": {1: "frames"}},
        ({1: "frames"},),
    )
    written["adapter"] = path

    # --- cross_kv ---
    wrapper = CrossKVWrapper(model).eval()
    out_names = []
    for i in range(n_layers):
        out_names += [f"cross_k_{i}", f"cross_v_{i}"]
    dyn = {"adapted": {1: "frames"}}
    for nm in out_names:
        dyn[nm] = {2: "frames"}
    path = out_dir / "cross_kv.onnx"
    _torch_export(
        wrapper, (torch.zeros(1, 32, dec_dim, dtype=torch.float32),), path, opset,
        ["adapted"], out_names, dyn, ({1: "frames"},),
    )
    written["cross_kv"] = path

    # --- decoder_kv ---
    cls = _make_decoder_class(n_layers)
    wrapper = cls(model, dec_heads, dec_head_dim, n_layers).eval()
    in_names = ["input_ids", "cross_k", "cross_v", "past_k", "past_v"]
    in_args = (
        torch.ones(1, 4, dtype=torch.long),
        torch.zeros(n_layers, 1, dec_heads, 32, dec_head_dim),
        torch.zeros(n_layers, 1, dec_heads, 32, dec_head_dim),
        torch.zeros(n_layers, 1, dec_heads, 4, dec_head_dim),
        torch.zeros(n_layers, 1, dec_heads, 4, dec_head_dim),
    )
    dyn = {
        "input_ids": {1: "seq"}, "logits": {1: "seq"},
        "cross_k": {3: "frames"}, "cross_v": {3: "frames"},
        "past_k": {3: "past"}, "past_v": {3: "past"},
        "present_k": {3: "past_seq"}, "present_v": {3: "past_seq"},
    }
    out_names = ["logits", "present_k", "present_v"]
    shapes = ({"1": "seq"}, {"3": "frames"}, {"3": "frames"},
              {"3": "past"}, {"3": "past"})
    path = out_dir / "decoder_kv.onnx"
    _torch_export(
        wrapper, in_args, path, opset,
        in_names, out_names, dyn, shapes, prefer_dynamo=True,
    )
    written["decoder_kv"] = path

    sizes = {}
    for k, v in written.items():
        total = v.stat().st_size
        data = v.with_name(v.name + ".data")
        if data.exists():
            total += data.stat().st_size
        sizes[k] = round(total / 1e6, 1)
    (out_dir / "export_report.json").write_text(json.dumps(sizes, indent=2))
    print("export:", json.dumps(sizes))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="checkpoint dir (default: base)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--gate", choices=["smoke", "final"], default=None,
                        help="require this eval gate to have passed before exporting")
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.gate:
        from moonshine_it.gates import require_gate_passed
        require_gate_passed(args.gate, cfg)
    name = Path(args.model).name if args.model else "base"
    out = Path(args.out) if args.out else REPO_ROOT / cfg["paths"]["results"] / "export" / name
    export_all(Path(args.model) if args.model else None, out,
               cfg["export"]["onnx"]["opset"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
