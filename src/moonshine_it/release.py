"""Release validation and promotion of the `.ort` bundle.

The runtime consumes `.ort` only — every downstream loader goes through
require_ort_file(), which rejects `.onnx` intermediates with an error naming
the `.ort` artifact to use (mirrors the engine's format enforcement).

Validation record (validation.json + manifest.json in the release dir):
  - per-artifact size + sha256 checksum
  - ORT smoke load of every graph on the training machine
  - tokenizer assets copied next to the models
A corrupted or missing artifact fails validation (non-zero exit) and no
manifest is (re)written, so nothing unvalidated can be promoted to
board-deploy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from moonshine_it.config import REPO_ROOT, load_config

GRAPHS = ("encoder", "adapter", "cross_kv", "decoder_kv")
TOKENIZER_ASSETS = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "preprocessor_config.json", "processor_config.json",
    "generation_config.json", "config.json",
)


def require_ort_file(path: Path) -> Path:
    """Accept only `.ort` artifacts; reject `.onnx` intermediates."""
    if path.suffix == ".onnx":
        ort = path.with_suffix(".ort")
        raise SystemExit(
            f"ONNX intermediate rejected: {path.name}\n"
            f"The runtime consumes .ort only. Use: {ort.name} "
            "(produced by: task ort / moonshine_it.quantize)"
        )
    if not path.exists():
        raise SystemExit(f"missing .ort artifact: {path}")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _smoke_load(path: Path) -> str:
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return f"ok ({len(sess.get_inputs())} inputs)"
    except Exception as exc:  # corrupted / truncated artifact
        raise SystemExit(f"ORT smoke load FAILED for {path.name}: {exc}") from exc


def validate_release(release_dir: Path, model_snapshot: Path) -> dict:
    release_dir.mkdir(parents=True, exist_ok=True)

    # stale manifests from a previous (possibly invalid) run must not survive
    for stale in ("manifest.json", "validation.json"):
        (release_dir / stale).unlink(missing_ok=True)

    manifest: dict = {"files": {}}
    validation: dict = {"artifacts": {}, "ok": True}

    for name in GRAPHS:
        ort_path = require_ort_file(release_dir / f"{name}.ort")
        digest = _sha256(ort_path)
        load_result = _smoke_load(ort_path)
        manifest["files"][ort_path.name] = {
            "size_bytes": ort_path.stat().st_size,
            "sha256": digest,
        }
        validation["artifacts"][name] = {
            "file": ort_path.name,
            "size_mb": round(ort_path.stat().st_size / 1e6, 2),
            "sha256": digest,
            "smoke_load": load_result,
        }
        print(f"validate[{name}]: smoke load {load_result}, sha256 {digest[:12]}…")

    copied: list[str] = []
    for asset in TOKENIZER_ASSETS:
        src = model_snapshot / asset
        if not src.exists():
            raise SystemExit(f"missing tokenizer asset in model snapshot: {src}")
        dst = release_dir / asset
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        manifest["files"][asset] = {
            "size_bytes": dst.stat().st_size,
            "sha256": _sha256(dst),
        }
        copied.append(asset)
    validation["tokenizer_assets"] = copied

    (release_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    out = release_dir / "validation.json"
    out.write_text(json.dumps(validation, indent=2))
    print(f"validate: OK — manifest + validation written to {release_dir}")
    return validation


def verify_manifest(release_dir: Path) -> None:
    """Re-verify checksums against manifest.json (used by board deploy)."""
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest in {release_dir} — run the validate target first"
        )
    manifest = json.loads(manifest_path.read_text())
    for name, rec in manifest["files"].items():
        f = release_dir / name
        if not f.exists():
            raise SystemExit(f"checksum verify FAILED: {name} is missing")
        actual = _sha256(f)
        if actual != rec["sha256"]:
            raise SystemExit(
                f"checksum verify FAILED: {name}\n"
                f"  manifest: {rec['sha256']}\n  actual:   {actual}"
            )
        print(f"checksum ok: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="checkpoint name (default: base)")
    parser.add_argument("--release-dir", default=None)
    parser.add_argument("--verify-only", action="store_true",
                        help="only re-verify checksums against the manifest")
    parser.add_argument("--gate", action="store_true",
                        help="require the post_quant gate to have passed "
                             "before writing the promotion manifest")
    args = parser.parse_args(argv)
    cfg = load_config()
    name = Path(args.model).name if args.model else "base"
    release_dir = Path(args.release_dir) if args.release_dir else \
        REPO_ROOT / cfg["release"]["dir"] / name
    if args.verify_only:
        verify_manifest(release_dir)
        return 0
    if args.gate:
        from moonshine_it.gates import require_gate_passed
        require_gate_passed("post_quant", cfg)
    from moonshine_it.download import model_dir
    validate_release(release_dir, model_dir(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
