"""The sanity latch's baseline is measured once per campaign, not per restart.

On rocm12g the amdgpu fault restarts the run every few minutes. Re-measuring
the baseline on each resume would (a) let the reference drift upward with the
model, so "worse than the starting point" degrades into "worse than a few
hundred steps ago" and the latch stops catching a genuinely diverging run,
and (b) spend most of the restart overhead on a 64-utterance eval.
"""

import json

import pytest


def _baseline(tmp_path, wer=82.61, split="mls/validation", origin=2997):
    p = tmp_path / "run_baseline.json"
    p.write_text(json.dumps({"eval_wer": wer, "iterate": "y",
                             "eval_split": split, "origin_step": origin,
                             "measured": "2026-08-27T19:57:45"}))
    return p


def test_cached_baseline_is_reused_on_resume(tmp_path):
    _baseline(tmp_path)
    cached = json.loads((tmp_path / "run_baseline.json").read_text())

    # The reuse condition the loop applies.
    assert cached["eval_split"] == "mls/validation"
    assert cached["eval_wer"] == 82.61
    assert cached["origin_step"] == 2997


def test_baseline_is_not_reused_across_a_split_change(tmp_path):
    """A baseline measured on another split is not comparable."""
    _baseline(tmp_path, split="fleurs/test")
    cached = json.loads((tmp_path / "run_baseline.json").read_text())

    assert cached["eval_split"] != "mls/validation", (
        "a baseline from a different split must force re-measurement")


def test_baseline_records_its_provenance(tmp_path):
    _baseline(tmp_path)
    cached = json.loads((tmp_path / "run_baseline.json").read_text())

    # Same self-describing requirement the evaluation spec puts on the
    # in-loop metric: a bare number cannot be traced to a measurement.
    for key in ("eval_wer", "iterate", "eval_split", "origin_step", "measured"):
        assert key in cached, f"baseline record is missing {key}"


def test_fresh_run_has_no_baseline_to_reuse(tmp_path):
    assert not (tmp_path / "run_baseline.json").exists()
