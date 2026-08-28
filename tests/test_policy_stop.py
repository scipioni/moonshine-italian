"""A deliberate refusal to continue must be distinguishable from a crash.

Measured 2026-08-28: the regression latch fired correctly at step 14,000 of
the final run, but it raised a plain SystemExit, leaving exit status 1 --
indistinguishable from the amdgpu page fault. scripts/supervise_final_train.sh
restarted the run ~127 times over eight hours, re-running the same three
failing evals each time and making no progress.
"""

import subprocess
import sys

import pytest

from moonshine_it.train_loop import (POLICY_STOP_EXIT_CODE, PolicyStop,
                                     check_iterate_not_regressed)


def test_policy_stop_is_a_system_exit_carrying_its_message():
    exc = PolicyStop("the latch tripped")
    assert isinstance(exc, SystemExit)
    assert str(exc) == "the latch tripped"


def test_policy_stop_exits_with_the_distinct_code():
    assert PolicyStop("x").code == POLICY_STOP_EXIT_CODE
    assert POLICY_STOP_EXIT_CODE != 1, (
        "a policy stop that exits 1 cannot be told apart from a crash")


def test_latch_raises_a_policy_stop_not_a_bare_system_exit():
    with pytest.raises(PolicyStop) as exc:
        check_iterate_not_regressed(90.0, 82.61, iterate="y", streak=2)
    assert "worse than" in str(exc.value)
    assert exc.value.code == POLICY_STOP_EXIT_CODE


def test_policy_stop_propagates_to_the_process_exit_status():
    """What the supervisor actually branches on."""
    code = subprocess.run(
        [sys.executable, "-c",
         "from moonshine_it.train_loop import check_iterate_not_regressed as c;"
         "c(90.0, 82.61, iterate='y', streak=2)"],
        capture_output=True, text=True).returncode
    assert code == POLICY_STOP_EXIT_CODE


def test_supervisor_branches_on_the_policy_stop_code():
    script = open("scripts/supervise_final_train.sh").read()
    assert "rc -eq 3" in script, (
        "supervisor must stop on POLICY_STOP_EXIT_CODE rather than treating "
        "a latch verdict as a crash to retry")
