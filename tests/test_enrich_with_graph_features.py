"""Unit tests for scripts/07_enrich_with_graph_features.py's join logic.

The risky part isn't the merge itself (pandas handles that), it's getting
the column renaming right on both sides without collisions, and not
silently duplicating or dropping flow rows -- a left join with a
non-unique right-hand key would multiply rows without erroring.
"""
import pandas as pd


class TestEnrichFlowsWithGraphFeatures:
    def _host_hour_features(self):
        return pd.DataFrame({
            "HourBucket": [pd.Timestamp("2017-07-06 15:00"), pd.Timestamp("2017-07-06 15:00")],
            "Host": ["192.168.10.8", "192.168.10.50"],
            "fan_out": [497, 3],
            "fan_in": [442, 900],
            "pagerank": [0.01, 0.02],
            "clustering_coefficient": [0.0, 0.0],
            "degree_ratio_change": [1.02, 0.9],
            "novelty_score": [0.36, 0.1],
            "flow_volume": [68843, 5],
            "byte_volume": [1000.0, 200.0],
            "fan_out_zscore": [1.02, -0.5],
            "fan_in_zscore": [0.5, -0.3],
            "novelty_zscore": [-0.14, -1.0],
            "flow_volume_zscore": [20.68, -0.2],
            "byte_volume_zscore": [0.1, -0.1],
        })

    def test_row_count_preserved(self, enrich_module):
        flows = pd.DataFrame({
            "Source IP": ["192.168.10.8"],
            "Destination IP": ["192.168.10.50"],
            "HourBucket": [pd.Timestamp("2017-07-06 15:00")],
        })
        result = enrich_module.enrich_flows_with_graph_features(flows, self._host_hour_features())
        assert len(result) == 1

    def test_src_and_dst_columns_are_prefixed_and_distinct(self, enrich_module):
        flows = pd.DataFrame({
            "Source IP": ["192.168.10.8"],
            "Destination IP": ["192.168.10.50"],
            "HourBucket": [pd.Timestamp("2017-07-06 15:00")],
        })
        result = enrich_module.enrich_flows_with_graph_features(flows, self._host_hour_features())
        assert result["src_flow_volume_zscore"].iloc[0] == 20.68
        assert result["dst_flow_volume_zscore"].iloc[0] == -0.2
        assert result["src_fan_out"].iloc[0] == 497
        assert result["dst_fan_out"].iloc[0] == 3

    def test_no_matching_hour_gives_nan_not_a_dropped_row(self, enrich_module):
        flows = pd.DataFrame({
            "Source IP": ["192.168.10.8"],
            "Destination IP": ["192.168.10.50"],
            "HourBucket": [pd.Timestamp("2017-07-03 08:00")],  # not in fixture
        })
        result = enrich_module.enrich_flows_with_graph_features(flows, self._host_hour_features())
        assert len(result) == 1
        assert pd.isna(result["src_fan_out"].iloc[0])

    def test_multiple_flows_same_hour_all_get_the_same_host_features(self, enrich_module):
        flows = pd.DataFrame({
            "Source IP": ["192.168.10.8", "192.168.10.8", "192.168.10.8"],
            "Destination IP": ["192.168.10.50", "192.168.10.5", "192.168.10.9"],
            "HourBucket": [pd.Timestamp("2017-07-06 15:00")] * 3,
        })
        result = enrich_module.enrich_flows_with_graph_features(flows, self._host_hour_features())
        assert len(result) == 3
        assert (result["src_flow_volume_zscore"] == 20.68).all()
