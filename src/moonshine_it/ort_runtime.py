"""Self-contained `.ort` streaming runtime (no repo dependencies).

Runs the four release graphs (encoder, adapter, cross_kv, decoder_kv)
through ONNX Runtime with KV-cached incremental decoding. Imports only
numpy + onnxruntime and (transformers OR tokenizers) — small enough to be
copied to the target board and used standalone by scripts/board/*.py.

The same module powers the host-side .ort evaluation (ort_eval.py).
"""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np

GRAPHS = ("encoder", "adapter", "cross_kv", "decoder_kv")
FRAME_LEN = 80  # encoder embedder frame; audio length must be a multiple


class OrtPipeline:
    def __init__(self, release_dir: Path, provider: str | None = None):
        import onnxruntime as ort

        available = ort.get_available_providers()
        candidates = [p for p in (provider, "AzureExecutionProvider",
                                  "CPUExecutionProvider") if p in available]
        self.sess = {}
        for name in GRAPHS:
            path = release_dir / f"{name}.ort"
            if path.suffix == ".onnx":
                raise SystemExit(
                    f"ONNX intermediate rejected: {path.name}\n"
                    f"The runtime consumes .ort only. Use: {path.with_suffix('.ort').name} "
                    "(produced by: task ort / moonshine_it.quantize)"
                )
            if not path.exists():
                raise SystemExit(f"missing .ort artifact: {path}")
            self.sess[name] = ort.InferenceSession(str(path), providers=candidates)
        self.provider = self.sess["encoder"].get_providers()[0]

        tok_json = release_dir / "tokenizer.json"
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(tok_json))
            self._tok_kind = "tokenizers"
            self.bos = tok.token_to_id("<s>")
            self.eos = tok.token_to_id("</s>")
            self._decode = lambda ids: tok.decode(ids, skip_special_tokens=True)
        except ImportError:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(release_dir))
            self._tok_kind = "transformers"
            self.bos = tok.bos_token_id
            self.eos = tok.eos_token_id
            self._decode = lambda ids: tok.decode(ids, skip_special_tokens=True)
        assert self.bos is not None and self.eos is not None, \
            "special tokens <s>/</s> not found in tokenizer"

        # geometry from the graphs themselves
        dec_in = {i.name: i for i in self.sess["decoder_kv"].get_inputs()}
        ck = dec_in["cross_k"]
        self.n_layers = ck.shape[0] if isinstance(ck.shape[0], int) else 10
        self.heads = ck.shape[2] if isinstance(ck.shape[2], int) else 8
        self.head_dim = ck.shape[4] if isinstance(ck.shape[4], int) else 64

    def decode_tokens(self, ids: list[int]) -> str:
        return self._decode(ids)

    # ---- audio ----
    def encode(self, audio: np.ndarray):
        pad = (-len(audio)) % FRAME_LEN
        x = np.pad(audio, (0, pad)).astype(np.float32)[None]
        e = self.sess["encoder"].run(None, {"input_values": x})[0]
        a = self.sess["adapter"].run(None, {"enc_hidden": e})[0]
        outs = self.sess["cross_kv"].run(None, {"adapted": a})
        cross_k = np.stack(outs[0::2])
        cross_v = np.stack(outs[1::2])
        return cross_k, cross_v

    def _past(self, length: int):
        shape = (self.n_layers, 1, self.heads, length, self.head_dim)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def _dec(self, ids: list[int], cross_k, cross_v, pk, pv):
        feed = {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "cross_k": cross_k, "cross_v": cross_v, "past_k": pk, "past_v": pv,
        }
        out = self.sess["decoder_kv"].run(None, feed)
        return out[0], out[1], out[2]

    # ---- decoding primitives ----
    def verify_prefix(self, prefix: list[int], cross_k, cross_v) -> list[int]:
        """Teacher-force prefix; keep tokens up to the first mismatch."""
        if len(prefix) <= 1:
            return list(prefix)
        logits, _, _ = self._dec(prefix, cross_k, cross_v, *self._past(0))
        pred = logits[0].argmax(axis=-1)
        kept = [prefix[0]]
        for i in range(1, len(prefix)):
            if pred[i - 1] == prefix[i]:
                kept.append(prefix[i])
            else:
                break
        return kept

    def greedy_continue(self, prefix: list[int], cross_k, cross_v,
                        max_new: int) -> list[int]:
        """KV-cached greedy continuation from prefix."""
        ids = list(prefix)
        logits, pk, pv = self._dec(ids, cross_k, cross_v, *self._past(0))
        nxt = int(logits[0, -1].argmax())
        emitted = 0
        while emitted < max_new and nxt != self.eos:
            ids.append(nxt)
            emitted += 1
            logits, pk, pv = self._dec([nxt], cross_k, cross_v, pk, pv)
            nxt = int(logits[0, -1].argmax())
        if nxt == self.eos:
            ids.append(nxt)
        return ids

    def transcribe_streaming(self, audio: np.ndarray, *, hop_ms: int,
                             max_tokens_per_s: float, speculative: bool = True,
                             sr: int = 16000) -> tuple[str, dict]:
        """Chunked streaming decode. Returns (text, timing stats)."""
        hop = int(sr * hop_ms / 1000)
        prefix = [self.bos]
        latencies: list[float] = []
        compute_s = 0.0

        for end in range(hop, len(audio) + 1, hop):
            chunk = audio[:end]
            dur = len(chunk) / sr
            token_budget = int(dur * max_tokens_per_s)

            t0 = time.perf_counter()
            cross_k, cross_v = self.encode(chunk)
            if speculative and len(prefix) > 1:
                prefix = self.verify_prefix(prefix, cross_k, cross_v)
            remaining = max(4, token_budget - (len(prefix) - 1))
            prefix = self.greedy_continue(prefix, cross_k, cross_v,
                                          min(remaining, 32))
            latencies.append(time.perf_counter() - t0)
            compute_s += latencies[-1]

            if len(prefix) - 1 > token_budget:
                prefix = [prefix[0]] + prefix[1: token_budget + 1]
                break

        text = self.decode_tokens(prefix)
        stats = {
            "hops": len(latencies),
            "redecode_latency_ms_mean": round(1000 * float(np.mean(latencies)), 1)
            if latencies else None,
            "redecode_latency_ms_p95": round(1000 * float(np.percentile(latencies, 95)), 1)
            if latencies else None,
            "compute_s": round(compute_s, 3),
            "rtf": round(compute_s / (len(audio) / sr), 3),
        }
        return text, stats


def read_wav_mono16k(path: Path) -> np.ndarray:
    """Stdlib PCM-wav reader (16 kHz mono, PCM_16/32) — no libsndfile needed."""
    with wave.open(str(path), "rb") as w:
        n, ch, width, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sr != 16000:
        raise SystemExit(f"clip must be 16 kHz (got {sr})")
    if ch > 1:
        raise SystemExit("clip must be mono")
    dtype = {2: np.int16, 4: np.int32}[width]
    return (np.frombuffer(raw, dtype=dtype).astype(np.float32)
            / float(np.iinfo(dtype).max))
