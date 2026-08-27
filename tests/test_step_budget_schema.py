"""Step budget schema (fix-training-loop-defects task 5.1): a training
profile must set exactly one of a fixed step count or an epoch-derived
budget, never both or neither, so a run's length is never ambiguous.
"""

import copy

import pytest

from moonshine_it.config import ConfigError, _validate, load_config


def _minimal_final_profile():
    cfg = copy.deepcopy(load_config())
    return cfg


def test_target_epochs_without_max_steps_is_valid():
    cfg = _minimal_final_profile()
    prof = cfg["training"]["profiles"]["final"]
    assert "target_epochs" in prof and "max_steps" not in prof
    _validate(cfg)  # must not raise


def test_both_fixed_and_epoch_budget_is_rejected():
    cfg = _minimal_final_profile()
    cfg["training"]["profiles"]["final"]["max_steps"] = 100
    cfg["training"]["profiles"]["final"]["eval_steps"] = 10
    cfg["training"]["profiles"]["final"]["save_steps"] = 10
    with pytest.raises(ConfigError, match="exactly one"):
        _validate(cfg)


def test_neither_fixed_nor_epoch_budget_is_rejected():
    cfg = _minimal_final_profile()
    del cfg["training"]["profiles"]["final"]["target_epochs"]
    with pytest.raises(ConfigError, match="exactly one"):
        _validate(cfg)


def test_fixed_steps_without_eval_and_save_steps_is_rejected():
    cfg = _minimal_final_profile()
    del cfg["training"]["profiles"]["final"]["target_epochs"]
    del cfg["training"]["profiles"]["final"]["eval_every_epoch_fraction"]
    cfg["training"]["profiles"]["final"]["max_steps"] = 100
    with pytest.raises(ConfigError, match="eval_steps"):
        _validate(cfg)
