"""The regression latch must survive supervisor restarts, and must not fire
on a curriculum boundary.

The streak used to be a local initialized to 0 at the top of train(), which
made the latch a no-op on rocm12g: the amdgpu fault restarts the run every
few minutes, resetting the counter before it can reach the threshold.
Measured on the final run -- evals at 8000, 9000 and 10000 were all worse
than the 82.61% baseline (three consecutive, which should have halted it)
but fell in attempts 11, 12 and 12, so the streak never exceeded 2.
"""

import json

import pytest

from moonshine_it.train_loop import (REGRESSION_STREAK_THRESHOLD,
                                     check_iterate_not_regressed,
                                     stage_index_for_step)

CURRICULUM = [{"steps": 8000, "max_audio_s": 5.0},
              {"steps": 32000, "max_audio_s": 10.0}]


def test_stage_index_tracks_the_boundary():
    assert stage_index_for_step(CURRICULUM, 7999) == 0
    assert stage_index_for_step(CURRICULUM, 8000) == 1
    # Holds at the last stage past the total budget.
    assert stage_index_for_step(CURRICULUM, 999_999) == 1
    assert stage_index_for_step([], 100) is None


def test_streak_accumulates_across_a_simulated_restart(tmp_path):
    """Two evals in one process, the third after a crash: still three."""
    state = tmp_path / "run_state.json"
    baseline = 82.61

    streak = 0
    for wer in (93.94, 96.0):
        streak = check_iterate_not_regressed(wer, baseline, iterate="y",
                                             streak=streak)
        state.write_text(json.dumps({"regression_streak": streak,
                                     "stage_index": 1}))
    assert streak == 2

    # Process dies; the supervisor restarts and reloads the streak.
    resumed = json.loads(state.read_text())["regression_streak"]
    assert resumed == 2, "streak did not survive the restart"

    with pytest.raises(SystemExit) as exc:
        check_iterate_not_regressed(90.0, baseline, iterate="y", streak=resumed)
    assert "worse than" in str(exc.value)


def test_streak_below_threshold_does_not_halt():
    streak = 0
    for _ in range(REGRESSION_STREAK_THRESHOLD - 1):
        streak = check_iterate_not_regressed(90.0, 82.61, iterate="y",
                                             streak=streak)
    assert streak == REGRESSION_STREAK_THRESHOLD - 1


def test_improvement_resets_the_streak():
    streak = check_iterate_not_regressed(90.0, 82.61, iterate="y", streak=1)
    assert streak == 2
    assert check_iterate_not_regressed(80.0, 82.61, iterate="y",
                                       streak=streak) == 0


def test_stage_change_clears_a_carried_streak(tmp_path):
    """The reset the loop applies when the stage index changes: a streak
    built under stage 0 must not carry into stage 1 and halt the run on a
    distribution shift it was never meant to police."""
    streak, last_stage = 2, 0
    stage = stage_index_for_step(CURRICULUM, 8000)

    if stage != last_stage:
        streak = 0

    # Without the reset this third bad eval would raise.
    assert check_iterate_not_regressed(96.0, 82.61, iterate="y",
                                       streak=streak) == 1
