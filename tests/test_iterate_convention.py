"""Schedule-free AdamW iterate convention (fix-training-loop-defects task 1.x):
save_checkpoint and quick_eval_wer must agree on which iterate ("x" vs "y")
they describe, and the pipeline must standardize on "y".
"""

import json

import pytest
import torch
from schedulefree import AdamWScheduleFree

from moonshine_it.train_loop import (
    REGRESSION_STREAK_THRESHOLD,
    check_iterate_not_regressed,
    ensure_iterate,
    iterate_name,
    save_checkpoint,
)


def _tiny_optimizer(n_steps: int = 2):
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 4)
    opt = AdamWScheduleFree(module.parameters(), lr=0.1, warmup_steps=0)
    opt.train()
    for _ in range(n_steps):
        # x is a weighted average over every step taken so far; after a
        # single step x == y trivially (the average of one point). Take more
        # than one step wherever the test needs x and y to actually diverge.
        x = torch.randn(2, 4)
        loss = module(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return module, opt


def test_iterate_name_reflects_train_mode():
    module, opt = _tiny_optimizer()
    assert iterate_name(opt) == "y"
    opt.eval()
    assert iterate_name(opt) == "x"
    opt.train()
    assert iterate_name(opt) == "y"


def test_eval_train_roundtrip_is_bit_identical():
    module, opt = _tiny_optimizer()
    before = [p.detach().clone() for p in module.parameters()]
    opt.eval()
    opt.train()
    after = [p.detach() for p in module.parameters()]
    for b, a in zip(before, after):
        assert torch.equal(b, a)


def test_ensure_iterate_is_idempotent_and_correct():
    module, opt = _tiny_optimizer()
    y_weights = [p.detach().clone() for p in module.parameters()]

    ensure_iterate(opt, "y")  # already y: no-op
    assert iterate_name(opt) == "y"
    assert all(torch.equal(a, b) for a, b in
              zip(y_weights, [p.detach() for p in module.parameters()]))

    ensure_iterate(opt, "x")
    assert iterate_name(opt) == "x"
    x_weights = [p.detach().clone() for p in module.parameters()]
    # x and y differ once a step has been taken (z has moved from init)
    assert any(not torch.equal(a, b) for a, b in zip(x_weights, y_weights))

    ensure_iterate(opt, "y")
    assert iterate_name(opt) == "y"
    assert all(torch.equal(a, b) for a, b in
              zip(y_weights, [p.detach() for p in module.parameters()]))


def test_ensure_iterate_rejects_unknown_name():
    module, opt = _tiny_optimizer()
    with pytest.raises(ValueError):
        ensure_iterate(opt, "z")


def test_check_iterate_not_regressed_tolerates_a_single_noisy_eval():
    # A lone regression (e.g. the model has barely moved from its
    # initialization) must not abort the run -- only a sustained streak does.
    streak = check_iterate_not_regressed(current_wer=101.0, baseline_wer=100.0,
                                         iterate="y", streak=0)
    assert streak == 1


def test_check_iterate_not_regressed_raises_after_sustained_streak():
    streak = 0
    for _ in range(REGRESSION_STREAK_THRESHOLD - 1):
        streak = check_iterate_not_regressed(current_wer=123.11, baseline_wer=100.0,
                                             iterate="x", streak=streak)
    with pytest.raises(SystemExit, match="consecutive evals"):
        check_iterate_not_regressed(current_wer=123.11, baseline_wer=100.0,
                                    iterate="x", streak=streak)


def test_check_iterate_not_regressed_resets_streak_on_improvement():
    streak = check_iterate_not_regressed(current_wer=123.11, baseline_wer=100.0,
                                         iterate="x", streak=0)
    assert streak == 1
    streak = check_iterate_not_regressed(current_wer=83.46, baseline_wer=149.87,
                                         iterate="y", streak=streak)
    assert streak == 0


def test_save_checkpoint_saves_y_and_records_iterate(tmp_path):
    module, opt = _tiny_optimizer()
    opt.eval()  # simulate having been left in "x" mode by a caller

    class _FakeModel:
        def save_pretrained(self, path):
            (path / "model.safetensors").write_text("stub")

    class _FakeProc:
        def save_pretrained(self, path):
            pass

    y_weights_expected = None
    ensure_iterate(opt, "y")
    y_weights_expected = [p.detach().clone() for p in module.parameters()]
    opt.eval()  # leave it in x again to prove save_checkpoint forces y itself

    ckpt = save_checkpoint(_FakeModel(), _FakeProc(), opt, tmp_path, 100,
                           {"eval_wer": 42.0, "iterate": "y"})
    assert iterate_name(opt) == "y"
    assert all(torch.equal(a, b) for a, b in
              zip(y_weights_expected, [p.detach() for p in module.parameters()]))

    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["iterate"] == "y"
    saved_opt_state = torch.load(ckpt / "optimizer.pt", weights_only=False)
    assert saved_opt_state["param_groups"][0]["train_mode"] is True
