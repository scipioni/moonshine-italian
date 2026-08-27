"""prepare_split's own use of plan_chunks is unchanged by the
ChunkSplitImpossible contract (fix-training-loop-defects task 3.4): an
unsplittable utterance must still be dropped as "oversize", now via an
explicit exception instead of a falsy empty list.
"""

import numpy as np

from moonshine_it import prepare


def _fake_example(duration_s: float):
    return {"audio": {"bytes": b"fake"}, "text": f"utterance of {duration_s} seconds"}


def test_unsplittable_oversize_utterance_is_dropped(tmp_path, monkeypatch):
    sr = 16000
    prep_cfg = {
        "target_sr": sr,
        "min_duration_s": 1.0,
        "max_duration_s": 3.0,   # forces a split for anything longer
        "vad": {},
    }
    norm_cfg = {"lowercase": True, "expand_numbers": False}

    # 5s of audio with a single VAD span covering nearly all of it -- no
    # internal boundary exists, so plan_chunks cannot find a cut point
    # within [min_len, max_len] and must refuse the split.
    total = 5 * sr
    audio = np.zeros(total, dtype=np.float32)
    monkeypatch.setattr(prepare, "decode_audio", lambda b: (audio, sr))
    monkeypatch.setattr(prepare, "to_target_sr", lambda a, s, t: a)
    monkeypatch.setattr(prepare, "speech_spans", lambda a, s, cfg: [(0, total)])

    manifest = prepare.prepare_split(
        [_fake_example(5.0)], tmp_path, prep_cfg, norm_cfg, "test", "train")
    stats = __import__("json").loads((tmp_path / "train_stats.json").read_text())
    assert stats["dropped_oversize"] == 1
    assert stats["kept"] == 0
    assert manifest.read_text() == ""


def test_splittable_oversize_utterance_is_kept_in_chunks(tmp_path, monkeypatch):
    sr = 16000
    prep_cfg = {
        "target_sr": sr,
        "min_duration_s": 1.0,
        "max_duration_s": 3.0,
        "vad": {},
    }
    norm_cfg = {"lowercase": True, "expand_numbers": False}

    # 5s of audio with two VAD spans separated by a gap at 2.5s -- gives
    # plan_chunks a real boundary to cut at.
    total = 5 * sr
    audio = np.zeros(total, dtype=np.float32)
    monkeypatch.setattr(prepare, "decode_audio", lambda b: (audio, sr))
    monkeypatch.setattr(prepare, "to_target_sr", lambda a, s, t: a)
    gap = int(2.5 * sr)
    monkeypatch.setattr(prepare, "speech_spans",
                        lambda a, s, cfg: [(0, gap), (gap, total)])

    manifest = prepare.prepare_split(
        [_fake_example(5.0)], tmp_path, prep_cfg, norm_cfg, "test", "train")
    stats = __import__("json").loads((tmp_path / "train_stats.json").read_text())
    assert stats["kept"] >= 1
    assert manifest.read_text() != ""
