"""Download base model and datasets.

download-model: snapshot + sha256 manifest, idempotent re-runs.
download-data:  FLEURS-it / MLS-it (public HF); Common Voice it (local CC0
                archive -- see datasets.common_voice.local in config.yaml).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, env_var, hf_token, load_config

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
    "voxpopuli": ("facebook/voxpopuli", "it", False),
}


def cv_extract_dir(cfg) -> Path:
    return REPO_ROOT / cfg["paths"]["data"] / "raw" / "common_voice_it"


def _cv_archive_path(cfg) -> Path:
    local = cfg["datasets"]["common_voice"]["local"]
    env_key = local["archive_env"]
    raw = env_var(env_key)
    if not raw:
        raise SystemExit(
            f"Common Voice archive not configured. Set {env_key} in .env to the "
            "downloaded cv-corpus-*.tar.gz path (see .env.example)."
        )
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise SystemExit(f"Common Voice archive not found: {path}")
    return path


def download_common_voice_local(cfg, force: bool = False) -> Path:
    """Extract only the clips referenced by train/dev/test.tsv from a local
    Common Voice archive. mozilla-foundation/common_voice_* is 404 on HF, so
    this bypasses load_dataset() entirely.

    Single sequential pass over the (typically ~10GB compressed) tar.gz: the
    per-language metadata TSVs sort alphabetically before clips/ in these
    archives, so by the time the first clip is reached, the full
    train+dev+test clip-name set is already known and every later member can
    be filtered against it without a second pass.
    """
    import csv
    import io
    import tarfile

    ds_cfg = cfg["datasets"]["common_voice"]
    local = ds_cfg["local"]
    archive = _cv_archive_path(cfg)
    out_dir = cv_extract_dir(cfg)
    clips_dir = out_dir / "clips"
    marker = out_dir / ".extracted.json"
    if not force and marker.exists():
        print(f"download-data[common_voice]: up to date ({out_dir})")
        return out_dir

    prefix = local["inner_prefix"]
    tsv_members = {f"{prefix}/{name}": name for name in local["splits"].values()}
    clips_prefix = f"{prefix}/clips/"

    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(exist_ok=True)

    print(f"download-data[common_voice]: extracting from {archive.name} "
          f"({archive.stat().st_size / 1e9:.1f} GB) -> {out_dir}")
    tsv_bytes: dict[str, bytes] = {}
    needed: set[str] | None = None
    kept = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name in tsv_members:
                fh = tar.extractfile(member)
                tsv_bytes[tsv_members[member.name]] = fh.read() if fh else b""
                if len(tsv_bytes) == len(tsv_members):
                    needed = set()
                    for tsv_name in tsv_members.values():
                        reader = csv.DictReader(
                            io.StringIO(tsv_bytes[tsv_name].decode("utf-8")),
                            delimiter="\t")
                        needed.update(row["path"] for row in reader)
                    print(f"download-data[common_voice]: {len(needed)} clips "
                          f"needed across {len(tsv_members)} splits")
                continue
            if needed is None or not member.name.startswith(clips_prefix):
                continue
            clip_name = member.name[len(clips_prefix):]
            if clip_name not in needed:
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            (clips_dir / clip_name).write_bytes(fh.read())
            kept += 1
            if kept % 20000 == 0:
                print(f"download-data[common_voice]: {kept}/{len(needed)} clips extracted")

    if needed is None:
        raise SystemExit(
            "Common Voice archive: never found all of "
            f"{sorted(tsv_members)} -- wrong inner_prefix in config.yaml?"
        )
    for tsv_name in tsv_members.values():
        (out_dir / tsv_name).write_bytes(tsv_bytes[tsv_name])
    marker.write_text(json.dumps(
        {"needed": len(needed), "extracted": kept}, indent=2))
    print(f"download-data[common_voice]: done -- {kept}/{len(needed)} clips "
          f"present (missing ones were removed/invalidated upstream)")
    return out_dir


def download_data(cfg, name: str, force: bool = False) -> Path:
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
    if ds_cfg.get("local"):
        return download_common_voice_local(cfg, force=force)

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
        rest = argv[1:]
        force = "--force" in rest
        names = [n for n in rest if n != "--force"] or ["fleurs"]
        for name in names:
            download_data(cfg, name, force=force)
        return 0
    raise SystemExit(
        "usage: download.py [model | data <fleurs|mls|common_voice|voxpopuli>... [--force]]"
    )


if __name__ == "__main__":
    raise SystemExit(main())
