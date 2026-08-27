"""model_io normalizes a loaded checkpoint to the "y" schedule-free iterate
(fix-training-loop-defects task 1.5).
"""

import torch
from schedulefree import AdamWScheduleFree

from moonshine_it.model_io import _normalize_checkpoint_iterate


def _tiny_checkpoint(tmp_path, *, save_as_x: bool):
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 4)
    opt = AdamWScheduleFree(module.parameters(), lr=0.1, warmup_steps=0)
    opt.train()
    for _ in range(3):
        x = torch.randn(2, 4)
        loss = module(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    y_weights = [p.detach().clone() for p in module.parameters()]
    if save_as_x:
        opt.eval()
    torch.save(opt.state_dict(), tmp_path / "optimizer.pt")
    (tmp_path / "model.safetensors").write_text("stub")  # presence check only
    return module, y_weights


def test_already_y_checkpoint_is_a_noop(tmp_path):
    module, y_weights = _tiny_checkpoint(tmp_path, save_as_x=False)
    _normalize_checkpoint_iterate(module, tmp_path)
    after = [p.detach() for p in module.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(y_weights, after))


def test_x_checkpoint_is_converted_to_y(tmp_path):
    module, y_weights = _tiny_checkpoint(tmp_path, save_as_x=True)
    # module's live parameters currently hold "x" (opt.eval() was called
    # before saving) -- confirm they differ from y before normalizing.
    before = [p.detach().clone() for p in module.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, y_weights))

    _normalize_checkpoint_iterate(module, tmp_path)
    after = [p.detach() for p in module.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(y_weights, after))


def test_missing_optimizer_pt_fails_loudly(tmp_path):
    (tmp_path / "model.safetensors").write_text("stub")
    module = torch.nn.Linear(4, 4)
    try:
        _normalize_checkpoint_iterate(module, tmp_path)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "optimizer.pt" in str(e)
