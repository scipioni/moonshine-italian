"""MultiASRDataset: flat concatenation across several prepared manifests,
used to mix mls + common_voice + fleurs into one training set (final profile).
"""

import json

import pytest

from moonshine_it.train_loop import ASRDataset, MultiASRDataset


@pytest.fixture
def two_manifests(tmp_path):
    cfg = {
        "preparation": {"min_duration_s": 0.1, "max_duration_s": 30.0},
        "training": {"chunked_augmentation": {"probability": 0.0, "min_fraction": 0.4}},
    }
    roots = []
    for tag, rows in [
        ("a", [{"audio": "a0.wav", "text": "uno", "duration_s": 1.0},
              {"audio": "a1.wav", "text": "due", "duration_s": 2.0}]),
        ("b", [{"audio": "b0.wav", "text": "tre", "duration_s": 3.0}]),
    ]:
        root = tmp_path / tag
        root.mkdir()
        manifest = root / "train.jsonl"
        manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        roots.append((manifest, root))
    return cfg, roots


def test_multi_dataset_flattens_rows_and_length(two_manifests, monkeypatch):
    cfg, roots = two_manifests
    monkeypatch.setattr(
        "moonshine_it.train_loop.load_audio", lambda p: __import__("numpy").zeros(16000)
    )
    parts = [ASRDataset(m, r, cfg, augment=False) for m, r in roots]
    combined = MultiASRDataset(parts)

    assert len(combined) == 3
    assert [r["text"] for r in combined.rows] == ["uno", "due", "tre"]


def test_multi_dataset_getitem_routes_to_correct_source(two_manifests, monkeypatch):
    cfg, roots = two_manifests
    monkeypatch.setattr(
        "moonshine_it.train_loop.load_audio", lambda p: __import__("numpy").zeros(16000)
    )
    parts = [ASRDataset(m, r, cfg, augment=False) for m, r in roots]
    combined = MultiASRDataset(parts)

    # indices 0,1 come from dataset "a"; index 2 crosses into dataset "b"
    assert combined[0]["text"] == "uno"
    assert combined[1]["text"] == "due"
    assert combined[2]["text"] == "tre"


def test_single_dataset_needs_no_wrapper():
    with pytest.raises(ValueError):
        MultiASRDataset([])
