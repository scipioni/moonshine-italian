"""Step budget derivation (fix-training-loop-defects tasks 5.1-5.3): the
configured training length denotes a fixed amount of *data*, independent of
grad_accum_steps -- the bug this fixes let max_steps: 72000 (derived for
accum=1) silently become 8.1 epochs once grad_accum_steps: 4 landed.
"""

from types import SimpleNamespace

import pytest

from moonshine_it.config import load_config, resolve_profile
from moonshine_it.train_loop import derive_step_budget


def _rp(batch_size=8, target_epochs=2.0, eval_every_epoch_fraction=0.028,
       max_steps=None, eval_steps=None, save_steps=None,
       max_save_interval_steps=None):
    # Mirrors ResolvedProfile's fields that derive_step_budget reads.
    # max_save_interval_steps defaults to None here so these cases keep
    # asserting the uncapped derivation; the cap has its own tests in
    # tests/test_checkpoint_retention.py.
    return SimpleNamespace(batch_size=batch_size, target_epochs=target_epochs,
                          eval_every_epoch_fraction=eval_every_epoch_fraction,
                          max_steps=max_steps, eval_steps=eval_steps,
                          save_steps=save_steps,
                          max_save_interval_steps=max_save_interval_steps)


def test_accumulation_change_preserves_sample_and_epoch_count():
    total_samples = 285_401
    b1 = derive_step_budget(_rp(), total_samples, accum_steps=1)
    b4 = derive_step_budget(_rp(), total_samples, accum_steps=4)

    assert b1["total_samples"] == b4["total_samples"] == total_samples
    # Integer rounding of max_steps means epoch_count only approximately
    # recovers target_epochs, but that rounding error is independent of
    # accum_steps -- both must land at the same (tiny) drift from 2.0.
    assert b1["epoch_count"] == pytest.approx(2.0, abs=1e-3)
    assert b1["epoch_count"] == pytest.approx(b4["epoch_count"], abs=1e-2)
    # max_steps differs (fewer, larger optimizer steps at higher accum) --
    # that's the whole point: accum_steps changes step *granularity*, not
    # the amount of data consumed. Only approximately 4x due to floor
    # division in steps_per_epoch at each granularity.
    assert b1["max_steps"] == pytest.approx(4 * b4["max_steps"], rel=1e-3)
    assert b1["effective_batch_size"] == 8
    assert b4["effective_batch_size"] == 32


def test_hand_computation_matches_the_mls_common_voice_fleurs_mix():
    total_samples = 110_004 + 173_174 + 2_223  # mls + common_voice + fleurs
    assert total_samples == 285_401
    budget = derive_step_budget(_rp(batch_size=8, target_epochs=2.0),
                                total_samples, accum_steps=1)
    steps_per_epoch = total_samples // 8
    assert steps_per_epoch == 35_675
    assert budget["max_steps"] == round(2.0 * steps_per_epoch) == 71_350
    assert budget["eval_steps"] == budget["save_steps"] == round(0.028 * steps_per_epoch)


def test_fixed_step_profile_ignores_accum_steps():
    rp = _rp(target_epochs=None, eval_every_epoch_fraction=None,
             max_steps=60, eval_steps=30, save_steps=30)
    b1 = derive_step_budget(rp, total_samples=200, accum_steps=1)
    b4 = derive_step_budget(rp, total_samples=200, accum_steps=4)
    assert b1["max_steps"] == b4["max_steps"] == 60
    assert b1["eval_steps"] == b4["eval_steps"] == 30


def test_dry_run_steps_overrides_derived_budget():
    budget = derive_step_budget(_rp(), total_samples=285_401, accum_steps=1,
                                dry_run_steps=5)
    assert budget["max_steps"] == 5


def test_real_config_final_profile_resolves_to_epoch_budget():
    cfg = load_config()
    rp = resolve_profile(cfg, "rocm12g", "final")
    assert rp.target_epochs == 2.0
    assert rp.max_steps is None
    budget = derive_step_budget(rp, total_samples=285_401, accum_steps=1)
    assert budget["max_steps"] == 71_350
