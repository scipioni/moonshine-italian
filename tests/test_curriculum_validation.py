"""Curriculum stages must be effective against the prepared corpus
(fix-training-loop-defects task 4.1/4.2): a stage whose bound admits no
additional rows over the preceding stage is a config artifact, not a real
training-distribution change, and must fail loudly.
"""

import pytest

from moonshine_it.config import load_config
from moonshine_it.train_loop import validate_curriculum


def _rows(*durations):
    return [{"duration_s": d} for d in durations]


def test_effective_curriculum_reports_row_counts():
    rows = _rows(2.0, 4.0, 6.0, 8.0, 10.0)
    curriculum = [{"steps": 10, "max_audio_s": 5.0, "description": "short"},
                  {"steps": 10, "max_audio_s": 10.0, "description": "full"}]
    report = validate_curriculum(curriculum, rows)
    assert report == [
        {"steps": 10, "max_audio_s": 5.0, "row_count": 2},
        {"steps": 10, "max_audio_s": 10.0, "row_count": 5},
    ]


def test_ineffective_stage_fails_loudly():
    rows = _rows(2.0, 4.0, 6.0, 8.0, 10.0)  # corpus max is 10.0
    curriculum = [
        {"steps": 10, "max_audio_s": 5.0, "description": "short"},
        {"steps": 10, "max_audio_s": 10.0, "description": "medium"},
        {"steps": 10, "max_audio_s": 30.0, "description": "full length"},
    ]
    with pytest.raises(SystemExit, match="stage 2"):
        validate_curriculum(curriculum, rows)


def test_empty_curriculum_is_valid_noop():
    assert validate_curriculum([], _rows(1.0, 2.0)) == []


def test_real_config_final_curriculum_is_now_effective():
    """Pins the fix (task 4.3, design Decision 5): config.yaml's final
    curriculum used to have a third stage (max_audio_s: 30.0) identical to
    the second against the prepared corpus's 10.0s cap
    (preparation.max_duration_s) -- merged into stage 2, total steps
    unchanged. The two remaining stages must each admit more rows than the
    last against a corpus that actually spans the full 0-10s range.
    """
    cfg = load_config()
    curriculum = cfg["training"]["profiles"]["final"]["curriculum"]
    corpus_max = cfg["preparation"]["max_duration_s"]
    assert len(curriculum) == 2, "expected the merged 2-stage curriculum"
    rows = _rows(1.0, 3.0, 5.0, 7.0, corpus_max)  # spans the full range
    report = validate_curriculum(curriculum, rows)
    assert [s["row_count"] for s in report] == [3, 5]  # strictly increasing
    total_steps = sum(s["steps"] for s in curriculum)
    assert total_steps == 40_000, "trim must preserve the total step budget"
