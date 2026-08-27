"""In-loop evaluations must reach the run record, not only the training log
(fix-training-loop-defects task 6.2, evaluation spec "In-loop training metric
is self-describing and gate-comparable").

A WER value with no split, sample count or iterate name attached is exactly
the failure this change exists to correct: `eval_wer` described the "x"
iterate while checkpoints held "y", and nothing in the recorded artifacts
said so.
"""

import json

from moonshine_it.train_loop import record_eval


def _metrics(wer=82.5, n=64):
    return {"eval_wer": wer, "iterate": "y",
            "eval_split": "mls/validation", "eval_n": n}


def test_record_eval_writes_provenance(tmp_path):
    record_eval(tmp_path, 2997, _metrics())

    rows = [json.loads(l) for l in
            (tmp_path / "eval_history.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["global_step"] == 2997
    assert row["eval_wer"] == 82.5
    # The three fields whose absence made the old metric untraceable.
    assert row["iterate"] == "y"
    assert row["eval_split"] == "mls/validation"
    assert row["eval_n"] == 64
    assert row["recorded"]


def test_record_eval_appends_a_curve(tmp_path):
    for step, wer in [(999, 90.0), (1998, 85.0), (2997, 82.5)]:
        record_eval(tmp_path, step, _metrics(wer=wer))

    rows = [json.loads(l) for l in
            (tmp_path / "eval_history.jsonl").read_text().splitlines()]
    assert [r["global_step"] for r in rows] == [999, 1998, 2997]
    assert [r["eval_wer"] for r in rows] == [90.0, 85.0, 82.5]


def test_record_eval_survives_a_supervisor_restart(tmp_path):
    """The supervisor restarts the run on the known amdgpu fault; the curve
    must accumulate across attempts rather than being truncated."""
    record_eval(tmp_path, 999, _metrics(wer=90.0))
    # second process, same out_dir
    record_eval(tmp_path, 1998, _metrics(wer=85.0))

    rows = (tmp_path / "eval_history.jsonl").read_text().strip().splitlines()
    assert len(rows) == 2
