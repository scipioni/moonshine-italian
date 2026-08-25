"""Deterministic smoke slicing: fixed-seed subset of prepared manifests.

Writes train/test JSONL slices + a checksum file. Two runs with the same
config and prepared data must be byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, load_config


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slice_smoke(cfg) -> dict:
    smoke = cfg["smoke"]
    dataset = smoke["dataset"]
    seed = smoke["seed"]
    n_train = smoke["train_samples"]
    n_eval = smoke["eval_samples"]

    src_root = REPO_ROOT / cfg["paths"]["data"] / "prepared" / dataset
    out_root = REPO_ROOT / smoke["slice_manifest"]
    out_root.mkdir(parents=True, exist_ok=True)

    def load(split: str) -> list[dict]:
        path = src_root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.sort(key=lambda r: r["audio"])  # stable order regardless of write order
        return rows

    summary = {"dataset": dataset, "seed": seed, "files": {}}
    for split, n in (("train", n_train), ("test", n_eval)):
        rows = load(split)
        if len(rows) < n:
            raise SystemExit(
                f"smoke slice needs {n} {split} samples but prepared manifest "
                f"has {len(rows)} — run the full prepare first "
                f"(task prepare / python -m moonshine_it.prepare --dataset {dataset})"
            )
        rng = random.Random(seed)
        picked = sorted(rng.sample(rows, n), key=lambda r: r["audio"])
        out = out_root / f"{split}.jsonl"
        with open(out, "w") as fh:
            for row in picked:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["files"][split] = {
            "path": str(out.relative_to(REPO_ROOT)),
            "n": len(picked),
            "sha256": _digest(out),
        }

    # Copy audio files referenced by the slices so the smoke set is compact.
    audio_dir = out_root / "audio"
    audio_dir.mkdir(exist_ok=True)
    import shutil

    for split in ("train", "test"):
        for row in load_split(out_root, split):
            src = src_root / row["audio"]
            dst = audio_dir / row["audio"]
            if not dst.exists():
                shutil.copy2(src, dst)
    return summary


def load_split(root: Path, split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / f"{split}.jsonl").read_text().splitlines()
    ]


def main() -> int:
    cfg = load_config()
    summary = slice_smoke(cfg)
    out = REPO_ROOT / cfg["smoke"]["slice_manifest"] / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("slice-smoke:", json.dumps(summary["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
