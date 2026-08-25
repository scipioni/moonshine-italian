"""Download gating tests (no network)."""

import pytest

from moonshine_it.download import download_data, model_dir, write_model_manifest
from moonshine_it import download as dl


def test_unknown_dataset_rejected():
    with pytest.raises(SystemExit, match="unknown dataset"):
        download_data(load_cfg(), "voxpopuli")


def test_common_voice_requires_token(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "hf_token", lambda: None)
    cfg = load_cfg()
    cfg["datasets"]["common_voice"]["enabled"] = True
    with pytest.raises(SystemExit, match=r"\.env\.example"):
        download_data(cfg, "common_voice")


def test_model_manifest_excludes_cache(tmp_path):
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "x").write_text("cache")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    manifest = write_model_manifest(tmp_path)
    assert "model.safetensors" in manifest.read_text()
    assert ".cache" not in manifest.read_text()


def load_cfg():
    from moonshine_it.config import load_config

    return load_config()


def test_model_dir_layout():
    cfg = load_cfg()
    d = model_dir(cfg)
    assert d.name == "moonshine-streaming-small"
    assert "data/models" in str(d)
