"""Crash-recovery checkpoint cadence and retention.

Measured 2026-08-27: with the save cadence tied to the eval cadence (999
steps), the rocm12g amdgpu page fault fired before the next save on four
consecutive supervisor attempts, so the run rolled back to the same
checkpoint every time and made no net progress. Saving more often is what
breaks the livelock; retention is what keeps that affordable.
"""

import pytest

from moonshine_it.config import load_config, resolve_profile
from moonshine_it.train_loop import derive_step_budget, prune_checkpoints


def _ckpt(out_dir, step):
    d = out_dir / f"checkpoint-{step}"
    d.mkdir()
    (d / "trainer_state.json").write_text("{}")
    return d


def test_save_cadence_is_capped_below_eval_cadence():
    cfg = load_config()
    rp = resolve_profile(cfg, "rocm12g", "final")
    budget = derive_step_budget(rp, 285395, 1)

    assert budget["save_steps"] <= rp.max_save_interval_steps
    # The eval cadence is deliberately left wide -- it is the expensive half.
    assert budget["eval_steps"] > budget["save_steps"]


def test_cap_never_widens_the_save_cadence():
    """A cap above the derived cadence must not push saves further apart."""
    cfg = load_config()
    rp = resolve_profile(cfg, "rocm12g", "final")
    object.__setattr__(rp, "max_save_interval_steps", 10**6)
    budget = derive_step_budget(rp, 285395, 1)

    assert budget["save_steps"] == budget["eval_steps"]


def test_prune_keeps_newest_and_protected(tmp_path):
    for step in (1000, 2000, 2997, 3250, 3500, 3750, 4000):
        _ckpt(tmp_path, step)

    removed = prune_checkpoints(tmp_path, keep_last=3,
                                protected={1000, 2000, 2997})

    kept = sorted(int(p.name.split("-")[1])
                  for p in tmp_path.glob("checkpoint-*"))
    assert kept == [1000, 2000, 2997, 3500, 3750, 4000]
    assert [p.name for p in removed] == ["checkpoint-3250"]


def test_prune_never_removes_the_best_checkpoint(tmp_path):
    for step in (1000, 2000, 3000, 4000, 5000):
        _ckpt(tmp_path, step)
    (tmp_path / "checkpoint-best").symlink_to(
        tmp_path / "checkpoint-2000", target_is_directory=True)

    prune_checkpoints(tmp_path, keep_last=1, protected=set())

    kept = sorted(int(p.name.split("-")[1])
                  for p in tmp_path.glob("checkpoint-[0-9]*"))
    assert 2000 in kept, "checkpoint-best's target was deleted"
    assert kept == [2000, 5000]


def test_prune_disabled_keeps_everything(tmp_path):
    for step in (1000, 2000, 3000):
        _ckpt(tmp_path, step)

    assert prune_checkpoints(tmp_path, keep_last=None) == []
    assert len(list(tmp_path.glob("checkpoint-*"))) == 3
