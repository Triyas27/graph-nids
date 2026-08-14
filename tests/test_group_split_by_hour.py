"""Unit tests for group_split_by_hour in scripts/08_classifier_with_graph_features.py.

This exists specifically to fix a leakage bug: graph features are scoped
to (Host, HourBucket), so a row-level random split let siblings from the
same block land on both sides of train/test. The fix splits by WHOLE HOUR
instead -- these tests pin the property that actually matters: no hour
ever appears on both sides, regardless of how many rows share that hour.
"""
import pandas as pd
import pytest


class TestGroupSplitByHour:
    def _hours(self, n_hours=50, rows_per_hour=20):
        # mimics the real shape: many rows sharing a small number of hours
        hour_values = [pd.Timestamp("2017-07-03") + pd.Timedelta(hours=h) for h in range(n_hours)]
        return pd.Series([h for h in hour_values for _ in range(rows_per_hour)])

    def test_no_hour_appears_in_both_train_and_test(self, classifier_graph_module):
        hours = self._hours()
        train_mask, test_mask = classifier_graph_module.group_split_by_hour(hours, test_size=0.2, random_state=42)
        train_hours = set(hours[train_mask])
        test_hours = set(hours[test_mask])
        assert train_hours.isdisjoint(test_hours)

    def test_every_row_assigned_to_exactly_one_side(self, classifier_graph_module):
        hours = self._hours()
        train_mask, test_mask = classifier_graph_module.group_split_by_hour(hours, test_size=0.2, random_state=42)
        assert (train_mask ^ test_mask).all()  # XOR: exactly one of the two is True everywhere

    def test_all_rows_sharing_an_hour_go_together(self, classifier_graph_module):
        # this is the actual property that fixes the leakage: every row
        # from the SAME hour ends up on the SAME side, however many there are
        hours = self._hours(n_hours=5, rows_per_hour=1000)
        train_mask, test_mask = classifier_graph_module.group_split_by_hour(hours, test_size=0.2, random_state=42)
        for hour in hours.unique():
            rows_for_hour = hours == hour
            sides = set(train_mask[rows_for_hour]) | set(test_mask[rows_for_hour])
            # every row for this hour must agree: all-train (train=True everywhere) or all-test
            assert train_mask[rows_for_hour].nunique() == 1
            assert test_mask[rows_for_hour].nunique() == 1

    def test_test_fraction_roughly_matches_requested_size(self, classifier_graph_module):
        hours = self._hours(n_hours=50, rows_per_hour=1)  # even weight per hour for a clean check
        train_mask, test_mask = classifier_graph_module.group_split_by_hour(hours, test_size=0.2, random_state=42)
        assert test_mask.sum() == pytest.approx(10, abs=1)

    def test_deterministic_with_same_random_state(self, classifier_graph_module):
        hours = self._hours()
        r1 = classifier_graph_module.group_split_by_hour(hours, test_size=0.2, random_state=42)
        r2 = classifier_graph_module.group_split_by_hour(hours, test_size=0.2, random_state=42)
        assert (r1[0] == r2[0]).all()
        assert (r1[1] == r2[1]).all()
