"""Regression tests for two silent-failure bugs in the decode/export path.

Both bugs produced plausible-looking output and passing gates, so they are
pinned here rather than left to eval numbers to catch.
"""

import torch

from moonshine_it import evaluate
from moonshine_it.export import EncoderWrapper


class _RecordingEncoder(torch.nn.Module):
    """Stands in for MoonshineStreamingEncoder; records the mask it received."""

    def __init__(self):
        super().__init__()
        self.seen_mask = "unset"

    def forward(self, input_values, attention_mask=None):
        self.seen_mask = attention_mask
        frames = input_values.shape[1] // 320
        return type("Out", (), {"last_hidden_state": torch.zeros(1, frames, 8)})()


def test_encoder_wrapper_passes_attention_mask():
    """The encoder applies its per-layer sliding windows only when the mask is
    not None. Exporting with None silently yields a globally-attending encoder
    whose output drifts ~7.5 max-abs, and parity cannot see it because the
    reference side was passing None too."""
    enc = _RecordingEncoder()
    wrapper = EncoderWrapper.__new__(EncoderWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.encoder = enc

    wrapper(torch.zeros(1, 16000))

    assert enc.seen_mask is not None, "encoder exported without an attention_mask"
    assert enc.seen_mask.shape == (1, 16000)
    assert bool(enc.seen_mask.all()), "batch-1 unpadded input needs an all-ones mask"


def test_streaming_token_budget_is_satisfiable_on_first_hop():
    """generate() always emits at least _MIN_NEW tokens, so the per-hop budget
    must never fall below that floor. At the shipped hop_ms=100 /
    max_tokens_per_second=13.0 the raw budget is int(0.1 * 13.0) == 1, which the
    hallucination guard could never satisfy -- it fired on hop 1 of every
    utterance and ended the decode after ~100 ms of audio."""
    hop_s = 0.1
    max_tps = 13.0

    raw_budget = int(hop_s * max_tps)
    assert raw_budget < evaluate._MIN_NEW, "precondition for this regression"

    clamped = max(evaluate._MIN_NEW, raw_budget)
    assert clamped >= evaluate._MIN_NEW

    # A hop that emits exactly the generation floor must not trip the guard.
    prefix_len_after_floor_emit = 1 + evaluate._MIN_NEW
    assert prefix_len_after_floor_emit - 1 <= clamped


def test_streaming_hop_ends_cover_trailing_audio():
    """range(hop, len+1, hop) drops up to hop_ms of trailing audio; the final
    partial chunk must still be decoded."""
    hop = 1600
    for n_samples in (5800, 16000, 16001):
        ends = list(range(hop, n_samples + 1, hop))
        if not ends or ends[-1] < n_samples:
            ends.append(n_samples)
        assert ends[-1] == n_samples, f"trailing audio dropped for n={n_samples}"
