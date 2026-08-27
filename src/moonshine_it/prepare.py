"""Audio preparation: 16 kHz mono chunks with aligned, normalized transcripts.

Pipeline per utterance:
  1. decode audio bytes -> float32 mono
  2. resample to target_sr (resample_poly)
  3. VAD (silero, energy fallback) -> speech spans
  4. trim to speech bounds; drop if remaining speech < min_duration
  5. if longer than max_duration: split at silence gaps into chunks within
     [min, max]; text is split at sentence boundaries proportionally to the
     chunks' speech time. Chunks that cannot be aligned or kept in bounds
     are dropped and counted.
  6. emit PCM16 wav + normalized transcript + JSONL manifest

Usage: uv run python -m moonshine_it.prepare [--dataset fleurs] [--smoke]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from moonshine_it.config import REPO_ROOT, load_config
from moonshine_it.normalize_it import normalize_text

VAD_MODEL = None


def _load_silero():
    global VAD_MODEL
    if VAD_MODEL is not None:
        return VAD_MODEL
    from silero_vad import load_silero_vad

    VAD_MODEL = load_silero_vad()
    return VAD_MODEL


def speech_spans(
    audio: np.ndarray, sr: int, vad_cfg: dict
) -> list[tuple[int, int]]:
    """Return [(start_sample, end_sample)] speech spans."""
    backend = vad_cfg.get("backend", "silero")
    threshold = vad_cfg.get("threshold", 0.5)
    min_speech = int(vad_cfg.get("min_speech_ms", 250) * sr / 1000)
    min_silence = int(vad_cfg.get("min_silence_ms", 100) * sr / 1000)

    if backend == "silero":
        try:
            model = _load_silero()
            from silero_vad import get_speech_timestamps

            ts = get_speech_timestamps(
                torch_tensor(audio), model,
                threshold=threshold,
                min_speech_duration_ms=vad_cfg.get("min_speech_ms", 250),
                min_silence_duration_ms=vad_cfg.get("min_silence_ms", 100),
                sampling_rate=sr,
            )
            return [(t["start"], t["end"]) for t in ts]
        except Exception as exc:  # pragma: no cover - depends on host
            print(f"prepare: silero VAD unavailable ({exc}); using energy VAD",
                  file=sys.stderr)
    return _energy_spans(audio, threshold, min_speech, min_silence)


def torch_tensor(audio: np.ndarray):
    import torch

    return torch.from_numpy(audio)


def _energy_spans(
    audio: np.ndarray, threshold: float, min_speech: int, min_silence: int
) -> list[tuple[int, int]]:
    frame = 480  # 30 ms at 16 kHz
    rms = np.sqrt(
        np.convolve(audio**2, np.ones(frame) / frame, mode="same")
    )
    peak = float(rms.max()) or 1.0
    loud = rms > threshold * 0.1 * peak  # relative gate
    spans: list[tuple[int, int]] = []
    start = None
    silence = 0
    for i, v in enumerate(loud):
        if v:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence * 1 >= min_silence:
                end = i - silence
                if end - start >= min_speech:
                    spans.append((start, end))
                start = None
    if start is not None and len(loud) - start >= min_speech:
        spans.append((start, len(loud)))
    return spans


def decode_audio(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return mono, sr


def to_target_sr(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio
    from math import gcd

    g = gcd(sr, target)
    return resample_poly(audio, target // g, sr // g).astype(np.float32)


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in ".!?;":
            parts.append(cur.strip())
            cur = ""
    if cur.strip():
        parts.append(cur.strip())
    return [p for p in parts if p]


class ChunkSplitImpossible(Exception):
    """Raised by plan_chunks when no admissible cut point exists.

    Distinguishes "split impossible" from "no split was needed" ([(0, total)])
    -- both used to reach callers as an empty list, which made an unsplittable
    utterance indistinguishable from one that didn't need splitting at all.
    """


def plan_chunks(
    spans: list[tuple[int, int]],
    total: int,
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]]:
    """Plan chunk boundaries [start, end) within [min_len, max_len], cutting
    at silence gaps when possible. Greedy left-to-right.

    Raises ChunkSplitImpossible if total > max_len but no admissible cut point
    exists; returns [(0, total)] (a single chunk) when no split is needed.
    """
    if total <= max_len:
        return [(0, total)]
    boundaries = sorted({0, total, *[s for s, _ in spans], *[e for _, e in spans]})
    boundaries = [b for b in boundaries if 0 < b < total]
    chunks: list[tuple[int, int]] = []
    start = 0
    while total - start > max_len:
        need_lo, need_hi = start + min_len, start + max_len
        candidates = [b for b in boundaries if need_lo <= b <= need_hi]
        if not candidates:
            raise ChunkSplitImpossible(
                f"no admissible cut point in [{need_lo}, {need_hi}] "
                f"for span [{start}, {total})")
        cut = max(candidates)  # longest possible chunk
        chunks.append((start, cut))
        start = cut
    if total - start >= min_len or not chunks:
        chunks.append((start, total))
    else:
        prev = chunks[-1]
        chunks[-1] = (prev[0], total)
    return chunks


def _split_proportional(units: list[str], weights: list[float]) -> list[str] | None:
    """Distribute units across chunks proportionally to chunk durations."""
    n = len(weights)
    if len(units) < n:
        return None
    out: list[str] = []
    idx = 0
    remaining = len(units)
    for i, w in enumerate(weights[:-1]):
        take = max(1, round(w * len(units)))
        take = min(take, remaining - (n - 1 - i))
        out.append(" ".join(units[idx: idx + take]))
        idx += take
        remaining -= take
    out.append(" ".join(units[idx:]))
    if any(not p.strip() for p in out):
        return None
    return out


def split_text_for_chunks(
    text: str, chunks: list[tuple[int, int]]
) -> list[str] | None:
    """Split normalized transcript across chunks proportionally.

    Prefers sentence boundaries; falls back to word boundaries when the
    utterance is a single long sentence (common in MLS audiobook prose) —
    word-level splits of partial utterances are exactly the chunked
    condition the streaming model sees at runtime.
    """
    if len(chunks) == 1:
        return [text]
    total = sum(e - s for s, e in chunks)
    weights = [(e - s) / total for s, e in chunks]
    sentences = split_sentences(text)
    if len(sentences) >= len(chunks):
        return _split_proportional(sentences, weights)
    words = text.split()
    return _split_proportional(words, weights)


def prepare_split(
    dataset_split,
    out_dir: Path,
    prep_cfg: dict,
    norm_cfg: dict,
    dataset_name: str,
    split_name: str,
    limit: int | None = None,
    *,
    shard: int | None = None,
    num_shards: int = 1,
) -> Path:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if shard is None else f".part{shard}"
    manifest = out_dir / f"{split_name}{suffix}.jsonl"
    stats = {"kept": 0, "dropped_oversize": 0, "dropped_nospeech": 0,
             "dropped_unsplittable": 0, "dropped_empty": 0}
    sr_target = prep_cfg["target_sr"]
    min_len = int(prep_cfg["min_duration_s"] * sr_target)
    max_len = int(prep_cfg["max_duration_s"] * sr_target)

    rows = []
    count = 0
    for idx, ex in enumerate(dataset_split):
        if shard is not None and idx % num_shards != shard:
            continue
        if limit is not None and count >= limit:
            break
        raw = ex["audio"]
        audio_bytes = raw["bytes"] if isinstance(raw, dict) else raw
        audio, sr = decode_audio(audio_bytes)
        audio = to_target_sr(audio, sr, sr_target)
        spans = speech_spans(audio, sr_target, prep_cfg["vad"])
        if not spans:
            stats["dropped_nospeech"] += 1
            continue
        first = max(0, spans[0][0] - int(0.1 * sr_target))
        last = min(len(audio), spans[-1][1] + int(0.1 * sr_target))
        audio = audio[first:last]
        spans = [(s - first, e - first) for s, e in spans if e > first]
        if len(audio) < min_len:
            stats["dropped_nospeech"] += 1
            continue

        text_field = ex.get("transcription") or ex.get("transcript") or ex.get("text") or ""
        text = normalize_text(
            text_field,
            lowercase=norm_cfg.get("lowercase", True),
            expand_nums=norm_cfg.get("expand_numbers", True),
        )
        if not text:
            stats["dropped_empty"] += 1
            continue

        try:
            chunks = plan_chunks(spans, len(audio), min_len, max_len)
        except ChunkSplitImpossible:
            stats["dropped_oversize"] += 1
            continue
        texts = split_text_for_chunks(text, chunks)
        if texts is None:
            stats["dropped_unsplittable"] += 1
            continue

        for ci, ((cs, ce), ctext) in enumerate(zip(chunks, texts)):
            dur = (ce - cs) / sr_target
            if not prep_cfg["min_duration_s"] <= dur <= prep_cfg["max_duration_s"]:
                stats["dropped_oversize"] += 1
                continue
            name = f"{dataset_name}_{split_name}_{idx:06d}_{ci:02d}.wav"
            path = out_dir / name
            sf.write(path, audio[cs:ce], sr_target, subtype="PCM_16")
            rows.append({
                "audio": name,
                "text": ctext,
                "duration_s": round(dur, 3),
                "source": f"{dataset_name}/{split_name}/{idx}",
            })
            stats["kept"] += 1
            count += 1

    with open(manifest, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if shard is None:
        (out_dir / f"{split_name}_stats.json").write_text(json.dumps(stats, indent=2))
    else:
        (out_dir / f"{split_name}.part{shard}.stats.json").write_text(
            json.dumps(stats, indent=2))
    print(f"prepare[{dataset_name}/{split_name}{suffix}]: {stats}")
    return manifest


def merge_parts(out_dir: Path, split_name: str, num_shards: int) -> bool:
    """Merge <split>.partK.jsonl (sorted by source idx) into <split>.jsonl."""
    parts = [out_dir / f"{split_name}.part{k}.jsonl" for k in range(num_shards)]
    if not all(p.exists() for p in parts):
        return False
    rows: list[tuple[int, dict]] = []
    stats: dict[str, int] = {}
    for k, p in enumerate(parts):
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                idx = int(row["audio"].rsplit("_", 2)[-2])
                rows.append((idx, row))
        sp = out_dir / f"{split_name}.part{k}.stats.json"
        if sp.exists():
            for key, val in json.loads(sp.read_text()).items():
                stats[key] = stats.get(key, 0) + val
    rows.sort(key=lambda r: r[0])
    with open(out_dir / f"{split_name}.jsonl", "w") as fh:
        for _, row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"{split_name}_stats.json").write_text(json.dumps(stats, indent=2))
    for k in range(num_shards):
        (out_dir / f"{split_name}.part{k}.jsonl").unlink(missing_ok=True)
        (out_dir / f"{split_name}.part{k}.stats.json").unlink(missing_ok=True)
    print(f"prepare[{split_name}]: merged {len(rows)} rows, {stats}")
    return True


class _CommonVoiceSplit:
    """Iterable over one Common Voice split, reading clip bytes from the
    local extraction directory (see download.download_common_voice_local).
    Yields the same {"audio": {"bytes": ...}, "transcription": ...} shape
    prepare_split() expects from an HF dataset split."""

    def __init__(self, tsv_path: Path, clips_dir: Path):
        self.tsv_path = tsv_path
        self.clips_dir = clips_dir

    def __iter__(self):
        import csv

        with open(self.tsv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                clip_path = self.clips_dir / row["path"]
                if not clip_path.exists():
                    continue  # upstream-invalidated/removed clip
                yield {
                    "audio": {"bytes": clip_path.read_bytes()},
                    "transcription": row["sentence"],
                }


def _iter_common_voice_local(cfg: dict, ds_cfg: dict) -> dict:
    from moonshine_it.download import cv_extract_dir

    extract_dir = cv_extract_dir(cfg)
    clips_dir = extract_dir / "clips"
    if not clips_dir.exists():
        raise SystemExit(
            "Common Voice clips not found — run: "
            "task download-data DATASET=common_voice"
        )
    return {
        split: _CommonVoiceSplit(extract_dir / tsv_name, clips_dir)
        for split, tsv_name in ds_cfg["local"]["splits"].items()
    }


def iter_dataset(name: str, cfg: dict):
    ds_cfg = cfg["datasets"][name]
    if ds_cfg.get("local"):
        return _iter_common_voice_local(cfg, ds_cfg)

    from datasets import Audio, load_dataset

    repo, config = ds_cfg["repo"], ds_cfg["config"]
    from moonshine_it.config import hf_token

    pq = ds_cfg.get("parquet")
    if pq:
        # Load the in-repo parquet shards (audio bytes embedded). The legacy
        # script/loader path fetches audio per-example over HTTP, which is
        # neither resumable nor robust to network blips.
        data_files = {
            split: f"hf://datasets/{pq['repo']}/{pattern}"
            for split, pattern in pq["splits"].items()
        }
        ds = load_dataset(
            "parquet", data_files=data_files,
            cache_dir=str(REPO_ROOT / cfg["paths"]["hf_cache"]),
            token=hf_token(),
        )
    else:
        ds = load_dataset(repo, config, cache_dir=str(REPO_ROOT / cfg["paths"]["hf_cache"]),
                          token=hf_token())
    for split in ds:
        ds[split] = ds[split].cast_column("audio", Audio(decode=False))
    return ds


SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fleurs", choices=["fleurs", "mls", "common_voice"])
    parser.add_argument("--limit", type=int, default=None, help="cap utterances (smoke)")
    parser.add_argument("--force", action="store_true",
                        help="re-prepare even if split manifests already exist")
    parser.add_argument("--shard", type=int, default=None,
                        help="process only rows with idx %% num-shards == shard")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="merge <split>.partK.jsonl shards into final manifests")
    args = parser.parse_args(argv)

    cfg = load_config()
    prep_cfg = cfg["preparation"]
    norm_cfg = prep_cfg["transcript_normalization"]
    out_root = REPO_ROOT / cfg["paths"]["data"] / "prepared" / args.dataset

    if args.merge:
        ok = True
        for split in ("train", "validation", "test"):
            if not merge_parts(out_root, split, args.num_shards):
                print(f"prepare[{args.dataset}/{split}]: parts incomplete, skipped")
        return 0 if ok else 1

    ds = iter_dataset(args.dataset, cfg)
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        manifest = out_root / f"{split}.jsonl"
        if args.shard is None and not args.force and manifest.exists():
            print(f"prepare[{args.dataset}/{split}]: manifest exists, skipping "
                  f"({manifest}; use --force to re-prepare)")
            continue
        prepare_split(ds[split], out_root, prep_cfg, norm_cfg,
                      args.dataset, split, limit=args.limit,
                      shard=args.shard, num_shards=args.num_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
