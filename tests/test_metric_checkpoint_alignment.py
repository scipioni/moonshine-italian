"""A checkpoint may only be ranked by a metric measured on its own weights.

Decoupling the save cadence from the eval cadence (to survive the amdgpu
fault) reintroduced this change's core defect in miniature: with eval at 999
and save at 250, the eval landed at step 4995 and the checkpoint at 5000, so
best_metric.json attributed a WER to weights five optimizer steps newer than
the ones measured. Two independent guards: the cadences are aligned, and a
mismatch refuses to rank.
"""

import json

from moonshine_it.config import load_config, resolve_profile
from moonshine_it.train_loop import derive_step_budget


def test_every_eval_step_falls_on_a_save_step():
    cfg = load_config()
    rp = resolve_profile(cfg, "rocm12g", "final")
    budget = derive_step_budget(rp, 285395, 1)

    assert budget["eval_steps"] % budget["save_steps"] == 0, (
        f"eval every {budget['eval_steps']} steps and save every "
        f"{budget['save_steps']} interleave; a checkpoint would be ranked by "
        "a metric measured on different weights")


def test_alignment_keeps_the_eval_cadence_close_to_configured():
    """Alignment must not distort the budget it is snapping."""
    cfg = load_config()
    rp = resolve_profile(cfg, "rocm12g", "final")
    budget = derive_step_budget(rp, 285395, 1)

    steps_per_epoch = 285395 // budget["effective_batch_size"]
    configured = rp.eval_every_epoch_fraction * steps_per_epoch
    assert abs(budget["eval_steps"] - configured) <= budget["save_steps"]


def test_stale_metric_is_not_ranked(tmp_path, monkeypatch):
    """The guard fires if cadence alignment ever regresses."""
    import moonshine_it.train_loop as tl

    marker = tmp_path / "best_metric.json"
    marker.write_text(json.dumps({"global_step": 100, "eval_wer": 90.0}))

    # A metric measured at 4995 must not promote the step-5000 checkpoint,
    # even though its value beats the incumbent.
    stale = {"eval_wer": 10.0, "iterate": "y", "eval_split": "mls/validation",
             "eval_n": 64, "global_step": 4995}
    tl.save_checkpoint(_Model(tmp_path), _Model(tmp_path), _Opt(),
                       tmp_path, 5000, stale)

    assert json.loads(marker.read_text())["eval_wer"] == 90.0
    assert not (tmp_path / "checkpoint-best").exists()


def test_matching_metric_is_ranked(tmp_path):
    import moonshine_it.train_loop as tl

    marker = tmp_path / "best_metric.json"
    marker.write_text(json.dumps({"global_step": 100, "eval_wer": 90.0}))

    fresh = {"eval_wer": 10.0, "iterate": "y", "eval_split": "mls/validation",
             "eval_n": 64, "global_step": 5000}
    tl.save_checkpoint(_Model(tmp_path), _Model(tmp_path), _Opt(),
                       tmp_path, 5000, fresh)

    best = json.loads(marker.read_text())
    assert best["global_step"] == 5000
    assert best["eval_wer"] == 10.0
    assert (tmp_path / "checkpoint-best").is_symlink()


class _Model:
    """Minimal stand-in: save_checkpoint only calls save_pretrained."""

    def __init__(self, root):
        self.root = root

    def save_pretrained(self, path):
        pass


class _Opt:
    param_groups = [{"train_mode": True}]

    def state_dict(self):
        return {}


def test_intermediate_save_records_no_borrowed_metric(tmp_path):
    """Saves outrun evals 4:1; the in-between checkpoints must not inherit
    the previous eval's WER as if it were their own."""
    import moonshine_it.train_loop as tl

    tl.save_checkpoint(_Model(tmp_path), _Model(tmp_path), _Opt(),
                       tmp_path, 9500, {})

    state = json.loads((tmp_path / "checkpoint-9500" /
                        "trainer_state.json").read_text())
    assert state["metrics"] == {}
    assert not (tmp_path / "best_metric.json").exists()
