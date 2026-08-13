"""Unit tests for scripts/05_build_hourly_graphs.py.

parse_timestamp is the risky one: CIC-IDS2017's raw Timestamp strings are
a 12-hour clock with no am/pm marker, disambiguated only by knowing every
capture is a single business day (~8am-5pm). Get this wrong and every
hourly graph in Week 2-3 silently buckets afternoon traffic into the wrong
hour. See the module docstring for the ground-truth check this rule was
validated against (Heartbleed's documented 15:12-15:32 window).
"""
import pandas as pd
import pytest


class TestParseTimestamp:
    def test_morning_hour_unchanged(self, build_hourly_graphs_module):
        result = build_hourly_graphs_module.parse_timestamp(pd.Series(["7/7/2017 8:59"]))
        assert result.iloc[0] == pd.Timestamp("2017-07-07 08:59:00")

    def test_afternoon_hour_gets_plus_twelve(self, build_hourly_graphs_module):
        # raw "3:30" is actually 3:30pm, not 3:30am -- this is the whole point
        result = build_hourly_graphs_module.parse_timestamp(pd.Series(["7/7/2017 3:30"]))
        assert result.iloc[0] == pd.Timestamp("2017-07-07 15:30:00")

    def test_noon_stays_twelve(self, build_hourly_graphs_module):
        result = build_hourly_graphs_module.parse_timestamp(pd.Series(["5/7/2017 12:15"]))
        assert result.iloc[0] == pd.Timestamp("2017-07-05 12:15:00")

    def test_handles_zero_padded_date_and_seconds(self, build_hourly_graphs_module):
        # Monday's file format differs from the rest: zero-padded date, includes seconds
        result = build_hourly_graphs_module.parse_timestamp(pd.Series(["03/07/2017 08:55:58"]))
        assert result.iloc[0] == pd.Timestamp("2017-07-03 08:55:58")

    def test_zero_padded_afternoon_also_gets_plus_twelve(self, build_hourly_graphs_module):
        # this is the exact case that looked like "2:40am in the middle of
        # a workday" before the am/pm bug was found -- it's really 14:40
        result = build_hourly_graphs_module.parse_timestamp(pd.Series(["03/07/2017 02:40:14"]))
        assert result.iloc[0] == pd.Timestamp("2017-07-03 14:40:14")

    def test_mixed_formats_in_one_series(self, build_hourly_graphs_module):
        # the real dataset mixes both formats after concatenating all 8 files
        result = build_hourly_graphs_module.parse_timestamp(pd.Series([
            "03/07/2017 08:55:58", "7/7/2017 3:30",
        ]))
        assert result.iloc[0] == pd.Timestamp("2017-07-03 08:55:58")
        assert result.iloc[1] == pd.Timestamp("2017-07-07 15:30:00")

    def test_heartbleed_window_matches_published_schedule(self, build_hourly_graphs_module):
        # ground-truth check: paper documents Heartbleed at 15:12-15:32 on
        # Wednesday 5/7/2017 -- raw rows show hour token "3"
        result = build_hourly_graphs_module.parse_timestamp(pd.Series(["5/7/2017 3:12", "5/7/2017 3:32"]))
        assert result.iloc[0] == pd.Timestamp("2017-07-05 15:12:00")
        assert result.iloc[1] == pd.Timestamp("2017-07-05 15:32:00")


class TestBuildHourlyGraphs:
    def test_builds_one_graph_per_hour_bucket(self, build_hourly_graphs_module):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "Destination IP": ["10.0.0.2", "10.0.0.2", "10.0.0.3"],
            "HourBucket": [
                pd.Timestamp("2017-07-03 08:00"),
                pd.Timestamp("2017-07-03 08:00"),
                pd.Timestamp("2017-07-03 09:00"),
            ],
            "TotalBytes": [100.0, 50.0, 200.0],
        })
        graphs = build_hourly_graphs_module.build_hourly_graphs(df)
        assert set(graphs.keys()) == {pd.Timestamp("2017-07-03 08:00"), pd.Timestamp("2017-07-03 09:00")}

    def test_aggregates_flow_count_and_bytes_within_hour(self, build_hourly_graphs_module):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.1"],
            "Destination IP": ["10.0.0.2", "10.0.0.2"],
            "HourBucket": [pd.Timestamp("2017-07-03 08:00")] * 2,
            "TotalBytes": [100.0, 50.0],
        })
        graphs = build_hourly_graphs_module.build_hourly_graphs(df)
        G = graphs[pd.Timestamp("2017-07-03 08:00")]
        assert G.number_of_edges() == 1
        edge = G.edges["10.0.0.1", "10.0.0.2"]
        assert edge["flow_count"] == 2
        assert edge["total_bytes"] == 150.0

    def test_fan_out_produces_multiple_edges_from_one_source(self, build_hourly_graphs_module):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.1", "10.0.0.1"],
            "Destination IP": ["10.0.0.2", "10.0.0.3", "10.0.0.4"],
            "HourBucket": [pd.Timestamp("2017-07-03 08:00")] * 3,
            "TotalBytes": [10.0, 10.0, 10.0],
        })
        graphs = build_hourly_graphs_module.build_hourly_graphs(df)
        G = graphs[pd.Timestamp("2017-07-03 08:00")]
        assert G.out_degree("10.0.0.1") == 3
