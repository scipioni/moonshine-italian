"""Download base model and datasets.

download-model: snapshot + sha256 manifest, idempotent re-runs.
download-data:  FLEURS-it / MLS-it (public), Common Voice it (gated by HF_TOKEN).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, hf_token, load_config

MODEL_ALLOW = [
    "*.safetensors",
    "*.json",
    "*.txt",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_dir(cfg) -> Path:
    name = cfg["base_model"]["id"].split("/")[-1]
    return REPO_ROOT / cfg["paths"]["data"] / "models" / name


def write_model_manifest(directory: Path) -> Path:
    manifest = directory / "MANIFEST.sha256"
    lines = []
    for path in sorted(directory.rglob("*")):
        if (
            path.is_file()
            and path.name != "MANIFEST.sha256"
            and ".cache" not in path.parts
        ):
            lines.append(f"{_sha256(path)}  {path.relative_to(directory).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def verify_model_manifest(directory: Path) -> bool:
    manifest = directory / "MANIFEST.sha256"
    if not manifest.exists():
        return False
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        path = directory / rel
        if not path.exists() or _sha256(path) != digest:
            return False
    return True


def download_model(cfg, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    directory = model_dir(cfg)
    if not force and verify_model_manifest(directory):
        print(f"download-model: up to date ({directory})")
        return directory

    directory.mkdir(parents=True, exist_ok=True)
    print(f"download-model: fetching {cfg['base_model']['id']} -> {directory}")
    snapshot_download(
        repo_id=cfg["base_model"]["id"],
        local_dir=str(directory),
        allow_patterns=MODEL_ALLOW,
        token=hf_token(),
    )
    manifest = write_model_manifest(directory)
    print(f"download-model: manifest written ({manifest.name}, "
          f"{len(manifest.read_text().splitlines())} files)")
    return directory


DATASETS = {
    "fleurs": ("google/fleurs", "it_it", False),
    "mls": ("facebook/multilingual_librispeech", "italian", False),
    "common_voice": ("mozilla-foundation/common_voice_21_0", "it", True),
}


def download_data(cfg, name: str) -> Path:
    from datasets import load_dataset

    if name not in DATASETS:
        raise SystemExit(
            f"unknown dataset '{name}'. Valid: {', '.join(DATASETS)}"
        )
    ds_cfg = cfg["datasets"].get(name, {})
    if not ds_cfg.get("enabled", False) and name != "fleurs":
        raise SystemExit(
            f"dataset '{name}' is disabled in config.yaml (datasets.{name}.enabled)"
        )

    repo, config, gated = DATASETS[name]
    if gated:
        token = hf_token()
        if not token:
            raise SystemExit(
                "Common Voice Italian is gated and needs HF_TOKEN.\n"
                "Copy .env.example to .env and set HF_TOKEN "
                "(https://huggingface.co/settings/tokens), then accept the dataset "
                "terms at https://huggingface.co/datasets/" + repo
            )
        print(f"download-data[{name}]: using HF_TOKEN from .env")

    cache = REPO_ROOT / cfg["paths"]["hf_cache"]
    print(f"download-data[{name}]: loading {repo}:{config} (cached under {cache})")
    load_dataset(repo, config, cache_dir=str(cache), token=hf_token(), trust_remote_code=False)
    print(f"download-data[{name}]: ready")
    return cache


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config()
    if not argv or argv[0] == "model":
        download_model(cfg)
        return 0
    if argv[0] == "data":
        names = argv[1:] or ["fleurs"]
        for name in names:
            download_data(cfg, name)
        return 0
    raise SystemExit("usage: download.py [model | data <fleurs|mls|common_voice>...]")


if __name__ == "__main__":
    raise SystemExit(main())
