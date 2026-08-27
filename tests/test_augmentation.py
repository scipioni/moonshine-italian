"""Chunked augmentation must actually execute (fix-training-loop-defects
tasks 3.2/3.3). Before this fix ASRDataset.__getitem__ called plan_chunks
with a span offering no admissible cut point, so it silently never fired for
any duration.
"""

import json

import numpy as np
import pytest

from moonshine_it.train_loop import ASRDataset, measure_augmentation_fraction


@pytest.fixture
def eight_second_row(tmp_path, monkeypatch):
    cfg = {
        "preparation": {"min_duration_s": 1.0, "max_duration_s": 10.0},
        "training": {"chunked_augmentation": {"probability": 1.0, "min_fraction": 0.4}},
    }
    root = tmp_path
    manifest = root / "train.jsonl"
    text = "questa e una frase abbastanza lunga da poter essere divisa in due parti"
    manifest.write_text(json.dumps(
        {"audio": "a.wav", "text": text, "duration_s": 8.0}) + "\n")
    audio = np.zeros(8 * 16000, dtype=np.float32)
    monkeypatch.setattr("moonshine_it.train_loop.load_audio", lambda p: audio)
    return manifest, root, cfg, audio


def test_augmentation_fires_and_splits_audio_and_text(eight_second_row):
    manifest, root, cfg, audio = eight_second_row
    ds = ASRDataset(manifest, root, cfg, augment=True, seed=0)
    item = ds[0]
    assert len(item["audio"]) < len(audio)
    assert item["text"] != ds.rows[0]["text"]
    assert ds.augmented_count == 1
    assert ds.augment_eligible_count == 1


def test_augmentation_never_fires_when_disabled(eight_second_row):
    manifest, root, cfg, audio = eight_second_row
    cfg["training"]["chunked_augmentation"]["probability"] = 0.0
    ds = ASRDataset(manifest, root, cfg, augment=True, seed=0)
    item = ds[0]
    assert len(item["audio"]) == len(audio)
    assert ds.augmented_count == 0


def test_measure_augmentation_fraction_is_nonzero_when_enabled(eight_second_row):
    manifest, root, cfg, audio = eight_second_row
    # replicate the single row so the sampler has more than one item to draw
    rows = [json.loads(l) for l in manifest.read_text().splitlines()] * 50
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = ASRDataset(manifest, root, cfg, augment=True, seed=0)
    frac = measure_augmentation_fraction(ds, n=50)
    assert frac > 0.0
