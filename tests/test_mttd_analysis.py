"""Unit tests for scripts/11_mttd_analysis.py.

first_detection_time must use the EARLIEST flagged timestamp, not just
the first row in whatever order the DataFrame happens to be in (this
mirrors a real risk: an unsorted or re-grouped timeline should still give
the right answer). hosts_reached_by must treat a never-detected case (None
cutoff) as zero hosts reached before it, not raise or return something
nonsensical.
"""
import pandas as pd
import pytest


class TestFirstDetectionTime:
    def test_returns_earliest_flagged_timestamp_even_out_of_order(self, mttd_module):
        # deliberately NOT sorted -- the earliest flagged row is in the middle
        timeline = pd.DataFrame({
            "ParsedTimestamp": pd.to_datetime(["2017-07-06 15:00", "2017-07-06 14:30", "2017-07-06 15:30"]),
            "model_flagged_attack": [True, True, False],
        })
        result = mttd_module.first_detection_time(timeline)
        assert result == pd.Timestamp("2017-07-06 14:30")

    def test_none_when_nothing_ever_flagged(self, mttd_module):
        timeline = pd.DataFrame({
            "ParsedTimestamp": pd.to_datetime(["2017-07-06 15:00", "2017-07-06 15:30"]),
            "model_flagged_attack": [False, False],
        })
        assert mttd_module.first_detection_time(timeline) is None

    def test_mask_restricts_to_subset(self, mttd_module):
        # earliest flagged row overall is at 14:30, but it's excluded by the mask
        timeline = pd.DataFrame({
            "ParsedTimestamp": pd.to_datetime(["2017-07-06 14:30", "2017-07-06 15:30"]),
            "model_flagged_attack": [True, True],
        })
        mask = pd.Series([False, True])
        result = mttd_module.first_detection_time(timeline, mask)
        assert result == pd.Timestamp("2017-07-06 15:30")

    def test_none_when_mask_excludes_all_flagged_rows(self, mttd_module):
        timeline = pd.DataFrame({
            "ParsedTimestamp": pd.to_datetime(["2017-07-06 14:30"]),
            "model_flagged_attack": [True],
        })
        mask = pd.Series([False])
        assert mttd_module.first_detection_time(timeline, mask) is None


class TestHostsReachedBy:
    def test_counts_hosts_reached_at_or_before_cutoff(self, mttd_module):
        pivot_summary = pd.DataFrame({
            "first_contact": pd.to_datetime(["2017-07-06 15:07", "2017-07-06 15:10", "2017-07-06 15:14"]),
        })
        result = mttd_module.hosts_reached_by(pivot_summary, pd.Timestamp("2017-07-06 15:10"))
        assert result == 2

    def test_zero_when_cutoff_is_none(self, mttd_module):
        pivot_summary = pd.DataFrame({"first_contact": pd.to_datetime(["2017-07-06 15:07"])})
        assert mttd_module.hosts_reached_by(pivot_summary, None) == 0

    def test_all_hosts_when_cutoff_after_everything(self, mttd_module):
        pivot_summary = pd.DataFrame({
            "first_contact": pd.to_datetime(["2017-07-06 15:07", "2017-07-06 15:14"]),
        })
        result = mttd_module.hosts_reached_by(pivot_summary, pd.Timestamp("2017-07-06 16:00"))
        assert result == 2
