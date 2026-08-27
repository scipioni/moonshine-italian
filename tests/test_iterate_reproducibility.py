"""Integration check for fix-training-loop-defects task 1.4: after an in-loop
eval + checkpoint save at the same step, re-evaluating the saved checkpoint
offline must reproduce the recorded metric (evaluation spec: "Metric and
artifact agree").

Runs a real 2-step smoke training run against the existing smoke slice
(data/manifests/smoke), redirecting only the output directory -- via an
absolute-path override, which `REPO_ROOT / abs_path` resolves to unchanged --
so nothing under results/ is touched. Requires a GPU and the local model
snapshot; skipped if either is unavailable.
"""

import copy
import json

import pytest
import torch

from moonshine_it.config import load_config as real_load_config
from moonshine_it.model_io import load_model_and_processor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU for a real training step")


def test_saved_checkpoint_reproduces_recorded_eval_wer(tmp_path, monkeypatch):
    from moonshine_it import train_loop

    cfg = copy.deepcopy(real_load_config())
    out_dir = tmp_path / "out"
    cfg["training"]["profiles"]["smoke"]["output_dir"] = str(out_dir)
    cfg["training"]["profiles"]["smoke"]["eval_steps"] = 2
    cfg["training"]["profiles"]["smoke"]["save_steps"] = 2
    cfg["evaluation"]["in_loop_samples"] = 4
    # The steps/s performance gate is calibrated for real runs; a 2-step test
    # is dominated by one-time model-load overhead and isn't what that gate
    # is checking (that's optimize-training-performance's concern, not this
    # task's).
    cfg["hardware_profiles"]["rocm12g"]["steps_per_second_min"] = None
    monkeypatch.setattr(train_loop, "load_config", lambda: cfg)

    train_loop.train("rocm12g", "smoke", dry_run_steps=2, resume=False)

    ckpt = out_dir / "checkpoint-2"
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["iterate"] == "y"
    recorded_wer = state["metrics"]["eval_wer"]
    assert state["metrics"]["eval_split"] == "smoke-slice/test"
    assert state["metrics"]["eval_n"] == 4

    model, proc = load_model_and_processor(cfg, model_path=ckpt, device="cuda",
                                           dtype="bf16")
    model.eval()
    smoke_root = train_loop.REPO_ROOT / cfg["smoke"]["slice_manifest"]
    replayed = train_loop.quick_eval_wer(
        model, proc, smoke_root / "test.jsonl", smoke_root / "audio", cfg,
        split_name="smoke-slice/test", iterate="y", limit=4)

    assert replayed["eval_wer"] == pytest.approx(recorded_wer, abs=1e-6)
