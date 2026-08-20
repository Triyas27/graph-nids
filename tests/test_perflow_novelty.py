"""Unit tests for compute_per_flow_novelty in scripts/14_perflow_novelty_eval.py.

This is the fix for the dilution problem: no bucketing anywhere, just "has
this source ever contacted this destination before, chronologically." The
risky parts: it must process rows in TIME order regardless of how they're
given (a shuffled DataFrame must give the same answer as a sorted one),
ties at the same timestamp need a defined tie-break, and each source's
history must be tracked independently of every other source's.
"""
import pandas as pd
import pytest


def _df(rows):
    """rows: list of (source, dest, timestamp_str)"""
    return pd.DataFrame({
        "src": [r[0] for r in rows],
        "dst": [r[1] for r in rows],
        "ts": pd.to_datetime([r[2] for r in rows]),
    })


class TestComputePerFlowNovelty:
    def test_first_ever_flow_from_a_source_is_novel(self, perflow_novelty_module):
        df = _df([("A", "X", "2017-07-06 10:00")])
        result = perflow_novelty_module.compute_per_flow_novelty(df, "src", "dst", "ts")
        assert list(result) == [True]

    def test_repeat_contact_to_same_destination_is_not_novel(self, perflow_novelty_module):
        df = _df([
            ("A", "X", "2017-07-06 10:00"),
            ("A", "X", "2017-07-06 10:05"),
        ])
        result = perflow_novelty_module.compute_per_flow_novelty(df, "src", "dst", "ts")
        assert list(result) == [True, False]

    def test_different_destination_is_novel_again(self, perflow_novelty_module):
        df = _df([
            ("A", "X", "2017-07-06 10:00"),
            ("A", "Y", "2017-07-06 10:05"),
        ])
        result = perflow_novelty_module.compute_per_flow_novelty(df, "src", "dst", "ts")
        assert list(result) == [True, True]

    def test_sources_are_tracked_independently(self, perflow_novelty_module):
        # B contacting X isn't affected by A having already contacted X
        df = _df([
            ("A", "X", "2017-07-06 10:00"),
            ("B", "X", "2017-07-06 10:05"),
        ])
        result = perflow_novelty_module.compute_per_flow_novelty(df, "src", "dst", "ts")
        assert list(result) == [True, True]

    def test_processes_in_chronological_order_regardless_of_row_order(self, perflow_novelty_module):
        # row order is REVERSED relative to time -- the later contact (row 0)
        # must still be correctly identified as the repeat, not the first
        df = _df([
            ("A", "X", "2017-07-06 10:05"),  # later in time, appears first in the frame
            ("A", "X", "2017-07-06 10:00"),  # earlier in time, appears second
        ])
        result = perflow_novelty_module.compute_per_flow_novelty(df, "src", "dst", "ts")
        # the row at 10:00 (index 1) is the true first contact -> True
        # the row at 10:05 (index 0) is the repeat -> False
        assert result.iloc[0] == False
        assert result.iloc[1] == True

    def test_this_is_the_actual_pivot_chain_shape(self, perflow_novelty_module):
        # mirrors the real scenario: one host's ordinary high-volume traffic
        # (many repeat contacts to a known peer) plus one genuine new contact
        # buried among it -- the per-flow feature must isolate the new one
        # regardless of how much repeat traffic surrounds it in time
        rows = [("host", "known_peer", f"2017-07-06 15:{m:02d}") for m in range(0, 40, 2)]
        rows.insert(10, ("host", "new_target", "2017-07-06 15:11"))
        df = _df(rows)
        result = perflow_novelty_module.compute_per_flow_novelty(df, "src", "dst", "ts")
        novel_positions = df.index[result].tolist()
        # first known_peer contact (position 0) and the new_target contact are novel; nothing else
        assert set(df.loc[novel_positions, "dst"]) == {"known_peer", "new_target"}
        assert result.sum() == 2
