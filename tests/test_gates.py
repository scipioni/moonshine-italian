"""Gate + latch tests using an isolated results tree."""

import json

import pytest

from moonshine_it import gates


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    return tmp_path


def write_verdict(root, ok):
    d = root / "results" / "spike"
    d.mkdir(parents=True, exist_ok=True)
    (d / "verdict.json").write_text(json.dumps({"ok": ok, "spikes": {"grad": ok}}))


def test_spike_latch_missing_verdict(isolated):
    with pytest.raises(SystemExit, match="No spike verdict"):
        gates.require_spike_ok()


def test_spike_latch_failed_demands_fallback(isolated):
    write_verdict(isolated, ok=False)
    with pytest.raises(SystemExit, match="fallback"):
        gates.require_spike_ok()


def test_spike_latch_pass(isolated):
    write_verdict(isolated, ok=True)
    gates.require_spike_ok()  # no raise


def test_smoke_latch(isolated):
    with pytest.raises(SystemExit, match="smoke"):
        gates.require_smoke_ok()
    gates.record_phase("smoke", {"ok": True, "phases": []})
    gates.require_smoke_ok()  # no raise


def test_relative_gate_math(isolated):
    outcome = gates.check_wer_gate("smoke", measured_wer=100.0, baseline_wer=95.0)
    assert outcome["passed"] is True       # 100 <= 95*1.10 = 104.5
    with pytest.raises(SystemExit, match="GATE FAILED"):
        gates.check_wer_gate("smoke", measured_wer=110.0, baseline_wer=95.0)


def test_absolute_gate(isolated):
    from moonshine_it.config import load_config

    cfg = load_config()
    gates.check_wer_gate("final", measured_wer=14.0, cfg=cfg)  # <= 15
    with pytest.raises(SystemExit):
        gates.check_wer_gate("final", measured_wer=16.0, cfg=cfg)


def test_relative_gate_needs_baseline(isolated):
    with pytest.raises(SystemExit, match="baseline"):
        gates.check_wer_gate("smoke", measured_wer=50.0, baseline_wer=None)
