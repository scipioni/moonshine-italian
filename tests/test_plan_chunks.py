"""plan_chunks: distinguishes "no split needed" from "split impossible"
(fix-training-loop-defects task 3.1) -- both used to reach callers as an
empty list, which made the two cases indistinguishable.
"""

import pytest

from moonshine_it.prepare import ChunkSplitImpossible, plan_chunks


def test_no_split_needed_returns_single_chunk():
    assert plan_chunks([(0, 100)], 100, min_len=10, max_len=200) == [(0, 100)]


def test_real_multi_span_splits_at_boundaries():
    # total=100, spans give boundaries at 40 and 60; max_len=60 forces a split
    chunks = plan_chunks([(0, 40), (60, 100)], 100, min_len=10, max_len=60)
    assert len(chunks) == 2
    assert chunks[0][1] in (40, 60)
    assert chunks[-1][1] == 100


def test_impossible_split_raises_distinguishable_exception():
    # total=100, max_len=30, but the only boundary (50) is outside every
    # admissible [start+min_len, start+max_len] window once min_len=40
    with pytest.raises(ChunkSplitImpossible):
        plan_chunks([(0, 50), (50, 100)], 100, min_len=40, max_len=30)
