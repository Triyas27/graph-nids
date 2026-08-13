"""Unit tests for scripts/06_compute_graph_features.py.

Focuses on the two functions with real custom logic: degree_ratio_change
(must return None, not 0 or an error, for a host with no prior-hour
baseline) and novelty_score (must score a host's very first appearance
as fully novel, and must never let the current hour leak into its own
baseline).
"""
import networkx as nx
import pytest


class TestComputeDegreeRatioChange:
    def test_ratio_for_host_present_in_both_hours(self, graph_features_module):
        prev = {"10.0.0.1": 4}
        curr = {"10.0.0.1": 8}
        result = graph_features_module.compute_degree_ratio_change(prev, curr)
        assert result["10.0.0.1"] == 2.0

    def test_none_for_host_with_no_prior_baseline(self, graph_features_module):
        prev = {}
        curr = {"10.0.0.9": 5}
        result = graph_features_module.compute_degree_ratio_change(prev, curr)
        assert result["10.0.0.9"] is None

    def test_host_only_in_prev_hour_is_not_in_result(self, graph_features_module):
        # a host that vanished this hour has no "current" row to attach to
        prev = {"10.0.0.1": 4, "10.0.0.2": 6}
        curr = {"10.0.0.1": 4}
        result = graph_features_module.compute_degree_ratio_change(prev, curr)
        assert set(result.keys()) == {"10.0.0.1"}


class TestNoveltyScores:
    def test_first_appearance_is_fully_novel(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.2")
        G.add_edge("10.0.0.1", "10.0.0.3")
        scores = graph_features_module.compute_novelty_scores(G, prior_destinations={})
        assert scores["10.0.0.1"] == 1.0

    def test_zero_novelty_when_all_destinations_seen_before(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.2")
        prior = {"10.0.0.1": {"10.0.0.2", "10.0.0.3"}}
        scores = graph_features_module.compute_novelty_scores(G, prior)
        assert scores["10.0.0.1"] == 0.0

    def test_partial_novelty(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.2")  # seen before
        G.add_edge("10.0.0.1", "10.0.0.9")  # new
        prior = {"10.0.0.1": {"10.0.0.2"}}
        scores = graph_features_module.compute_novelty_scores(G, prior)
        assert scores["10.0.0.1"] == 0.5

    def test_host_with_no_outgoing_edges_is_excluded(self, graph_features_module):
        # a pure fan-in target (never initiates contact) has nothing to score
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.2")
        scores = graph_features_module.compute_novelty_scores(G, prior_destinations={})
        assert "10.0.0.2" not in scores

    def test_current_hour_does_not_leak_into_its_own_baseline(self, graph_features_module):
        # scoring must use prior_destinations exactly as passed in, never
        # mutate it to include this hour's own edges
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.9")
        prior = {"10.0.0.1": set()}
        graph_features_module.compute_novelty_scores(G, prior)
        assert prior["10.0.0.1"] == set()


class TestVolumeAggregation:
    def test_total_out_flow_count_sums_outgoing_edges_only(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.2", flow_count=5, total_bytes=100.0)
        G.add_edge("10.0.0.1", "10.0.0.3", flow_count=3, total_bytes=50.0)
        G.add_edge("10.0.0.9", "10.0.0.1", flow_count=1000, total_bytes=9999.0)  # inbound to .1, must not count
        totals = graph_features_module.total_out_flow_count(G)
        assert totals["10.0.0.1"] == 8

    def test_total_out_bytes_sums_outgoing_edges_only(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.2", flow_count=5, total_bytes=100.0)
        G.add_edge("10.0.0.1", "10.0.0.3", flow_count=3, total_bytes=50.0)
        G.add_edge("10.0.0.9", "10.0.0.1", flow_count=1000, total_bytes=9999.0)
        totals = graph_features_module.total_out_bytes(G)
        assert totals["10.0.0.1"] == 150.0

    def test_high_volume_to_same_peers_does_not_affect_fan_out(self, graph_features_module):
        # this is the actual 192.168.10.8 scenario: contact rate explodes,
        # neighbor set doesn't -- flow_volume must move, fan_out must not
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.2", flow_count=50000, total_bytes=1.0)
        assert graph_features_module.total_out_flow_count(G)["10.0.0.1"] == 50000
        assert dict(G.out_degree())["10.0.0.1"] == 1


class TestUpdatePriorDestinations:
    def test_folds_new_destinations_into_history(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.9")
        prior = {"10.0.0.1": {"10.0.0.2"}}
        updated = graph_features_module.update_prior_destinations(G, prior)
        assert updated["10.0.0.1"] == {"10.0.0.2", "10.0.0.9"}

    def test_does_not_mutate_input(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.1", "10.0.0.9")
        prior = {"10.0.0.1": {"10.0.0.2"}}
        graph_features_module.update_prior_destinations(G, prior)
        assert prior["10.0.0.1"] == {"10.0.0.2"}

    def test_creates_new_host_entry(self, graph_features_module):
        G = nx.DiGraph()
        G.add_edge("10.0.0.5", "10.0.0.6")
        updated = graph_features_module.update_prior_destinations(G, {})
        assert updated["10.0.0.5"] == {"10.0.0.6"}
