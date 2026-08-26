"""Evaluation harness: full-utterance and chunked-streaming modes.

Streaming simulation mirrors the on-device runtime: audio is appended in
hops; each re-decode teacher-forces the previous hypothesis, detects the
first mismatch (speculative verification), and continues generation from
that point. An excessive token rate (hallucination) truncates the line the
same way the runtime's max_tokens_per_second heuristic does.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from moonshine_it.normalize_it import normalize_text


@dataclass
class EvalResult:
    model: str
    dataset: str
    split: str
    mode: str                     # "full" | "streaming"
    n: int
    wer: float
    cer: float
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


def load_manifest(manifest: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]  # always return 1-D; 2-D input corrupts batching
    if file_sr != sr:
        from moonshine_it.prepare import to_target_sr

        data = to_target_sr(data, file_sr, sr)
    return data


# Per-hop generation floor/ceiling for the streaming decode. _MIN_NEW is the
# smallest number of tokens a hop may emit; the token budget is clamped to it so
# the hallucination guard stays satisfiable on short leading chunks.
_MIN_NEW = 4
_MAX_NEW_PER_HOP = 32


def transcribe_full(model, proc, audio: np.ndarray, max_tokens_per_s: float,
                    sr: int = 16000) -> str:
    import torch

    # NB: audio must be a LIST — a bare 1-D array is treated as a batch of
    # single-sample clips by the feature extractor.
    inputs = proc(audio=[audio], return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # match the model's weight dtype (bf16 training checkpoints need bf16 input)
    inputs["input_values"] = inputs["input_values"].to(
        next(model.parameters()).dtype)
    dur = len(audio) / sr
    max_new = max(8, int(dur * max_tokens_per_s))
    with torch.no_grad():
        ids = model.generate(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new,
            do_sample=False,
        )[0]
    return proc.tokenizer.decode(ids, skip_special_tokens=True)


def _verify_prefix(model, inputs, prefix) -> tuple[list[int], bool]:
    """Teacher-force prefix; return (tokens up to first mismatch, changed)."""
    import torch

    if len(prefix) <= 1:
        return list(prefix), True
    with torch.no_grad():
        out = model(
            input_values=inputs["input_values"].to(next(model.parameters()).dtype),
            attention_mask=inputs.get("attention_mask"),
            decoder_input_ids=torch.tensor([prefix], device=model.device),
            use_cache=False,
        )
    # logits[i] predicts token prefix[i+1]
    logits = out.logits[0].argmax(dim=-1).tolist()
    kept = [prefix[0]]
    changed = False
    for i in range(1, len(prefix)):
        if logits[i - 1] == prefix[i]:
            kept.append(prefix[i])
        else:
            changed = True
            break
    return kept, changed


def transcribe_streaming(
    model,
    proc,
    audio: np.ndarray,
    *,
    hop_ms: int,
    max_tokens_per_s: float,
    speculative: bool = True,
    sr: int = 16000,
) -> tuple[str, dict]:
    """Chunked streaming decode. Returns (text, timing stats)."""
    import torch

    hop = int(sr * hop_ms / 1000)
    eos_id = proc.tokenizer.eos_token_id
    prefix = [proc.tokenizer.bos_token_id]
    latencies: list[float] = []
    compute_s = 0.0

    # range() stops at the last whole hop, which drops up to hop_ms of trailing
    # audio; append the true end so the final partial chunk is still decoded.
    ends = list(range(hop, len(audio) + 1, hop))
    if not ends or ends[-1] < len(audio):
        ends.append(len(audio))

    for end in ends:
        chunk = audio[:end]
        inputs = proc(audio=[chunk], return_tensors="pt", sampling_rate=sr)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["input_values"] = inputs["input_values"].to(
            next(model.parameters()).dtype)
        dur = len(chunk) / sr
        # generate() below always emits at least _MIN_NEW tokens, so a budget
        # under that floor can never be satisfied. Without this clamp the guard
        # fires on hop 1 (at hop_ms=100: int(0.1 * 13.0) == 1 < 4) on every
        # utterance, ending the decode after ~100 ms of audio.
        token_budget = max(_MIN_NEW, int(dur * max_tokens_per_s))

        t0 = time.perf_counter()
        if speculative and len(prefix) > 1:
            prefix, _changed = _verify_prefix(model, inputs, prefix)
        remaining = max(_MIN_NEW, token_budget - len(prefix) + 1)
        with torch.no_grad():
            ids = model.generate(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask"),
                decoder_input_ids=torch.tensor([prefix], device=model.device),
                max_new_tokens=min(remaining, _MAX_NEW_PER_HOP),
                do_sample=False,
            )[0].tolist()
        latencies.append(time.perf_counter() - t0)
        compute_s += latencies[-1]

        # Strip a generated EOS before the prefix is fed back as
        # decoder_input_ids: an EOS emitted against partial audio only means
        # "done with what I've heard so far", and leaving it inside the prefix
        # makes every later hop decode past end-of-sequence.
        if eos_id is not None and eos_id in ids[1:]:
            ids = ids[:ids.index(eos_id, 1)]
        prefix = ids
        # Hallucination guard: token rate too high -> clamp this hop back to
        # budget and carry on. This used to break out of the loop, which froze
        # the transcript at the first over-budget hop and turned the rest of the
        # utterance into deletions (measured 88% deletions on the smoke slice).
        if len(prefix) - 1 > token_budget:
            prefix = [prefix[0]] + prefix[1: token_budget + 1]

    text = proc.tokenizer.decode(prefix, skip_special_tokens=True)
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


def evaluate_manifest(
    model,
    proc,
    manifest_path: Path,
    audio_root: Path,
    *,
    mode: str,
    streaming_cfg: dict,
    limit: int | None = None,
    model_name: str = "model",
    dataset: str = "dataset",
    split: str = "test",
) -> EvalResult:
    import jiwer

    rows = load_manifest(manifest_path)
    if limit:
        rows = rows[:limit]
    refs: list[str] = []
    hyps: list[str] = []
    stats_all: list[dict] = []

    for row in rows:
        audio = load_audio(audio_root / row["audio"])
        ref = normalize_text(row["text"], expand_nums=False)
        if mode == "full":
            hyp = transcribe_full(model, proc, audio,
                                  max_tokens_per_s=streaming_cfg["max_tokens_per_second"])
            stats_all.append({})
        elif mode == "streaming":
            hyp, stats = transcribe_streaming(
                model, proc, audio,
                hop_ms=streaming_cfg["hop_ms"],
                max_tokens_per_s=streaming_cfg["max_tokens_per_second"],
                speculative=streaming_cfg.get("speculative_decoding", True),
            )
            stats_all.append(stats)
        else:
            raise ValueError(f"unknown mode {mode}")
        hyp = normalize_text(hyp, expand_nums=False)
        refs.append(ref)
        hyps.append(hyp)

    wer = float(jiwer.wer(refs, hyps)) * 100
    cer = float(jiwer.cer(refs, hyps)) * 100
    extra: dict = {}
    if mode == "streaming" and stats_all and stats_all[0]:
        extra["redecode_latency_ms_mean"] = float(np.mean(
            [s["redecode_latency_ms_mean"] for s in stats_all if s]))
        extra["redecode_latency_ms_p95"] = float(np.mean(
            [s["redecode_latency_ms_p95"] for s in stats_all if s]))
        extra["rtf_mean"] = float(np.mean([s["rtf"] for s in stats_all if s]))
    return EvalResult(
        model=model_name, dataset=dataset, split=split, mode=mode,
        n=len(rows), wer=round(wer, 2), cer=round(cer, 2), extra=extra,
    )


if __name__ == "__main__":
    from moonshine_it.evaluate_cli import main

    raise SystemExit(main())
