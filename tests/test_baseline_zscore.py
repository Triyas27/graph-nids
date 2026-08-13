"""Unit tests for RunningStats and compute_baseline_zscore in
scripts/06_compute_graph_features.py.

These exist because raw fan_out/novelty turned out to be a poor signal
for hosts that are naturally high-traffic every hour (found by hand-
checking 192.168.10.8, the Infiltration attack's compromised host) -- the
z-score features are meant to catch "unusual FOR THIS HOST", so it's worth
pinning down the exact behavior at the edges: no baseline yet, a perfectly
constant baseline, and normal variation.
"""
import pytest


class TestRunningStats:
    def test_mean_after_updates(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        for x in [2, 4, 6]:
            stats.update(x)
        assert stats.mean == pytest.approx(4.0)

    def test_std_is_none_with_fewer_than_two_observations(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        assert stats.std is None
        stats.update(5)
        assert stats.std is None

    def test_std_after_two_observations(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        stats.update(2)
        stats.update(4)
        # sample std of [2, 4] = sqrt(((2-3)^2 + (4-3)^2) / (2-1)) = sqrt(2)
        assert stats.std == pytest.approx(2 ** 0.5)

    def test_std_zero_for_constant_values(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        stats.update(5)
        stats.update(5)
        stats.update(5)
        assert stats.std == pytest.approx(0.0)


class TestComputeBaselineZscore:
    def test_none_with_no_prior_observations(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        result = graph_features_module.compute_baseline_zscore(10, stats)
        assert result is None

    def test_none_with_only_one_prior_observation(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        stats.update(5)
        result = graph_features_module.compute_baseline_zscore(10, stats)
        assert result is None

    def test_zero_when_value_matches_baseline_mean(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        stats.update(4)
        stats.update(4)
        result = graph_features_module.compute_baseline_zscore(4, stats)
        assert result == pytest.approx(0.0)

    def test_large_positive_zscore_for_spike_above_constant_baseline(self, graph_features_module):
        # this is exactly the 192.168.10.8 scenario in miniature: a host
        # whose baseline never varies gets a huge (not undefined/NaN)
        # z-score the moment it deviates, thanks to the epsilon regularizer
        stats = graph_features_module.RunningStats()
        stats.update(2)
        stats.update(2)
        result = graph_features_module.compute_baseline_zscore(900, stats)
        assert result > 1000

    def test_normal_variation_gives_moderate_zscore(self, graph_features_module):
        stats = graph_features_module.RunningStats()
        for x in [10, 12, 8, 11, 9]:
            stats.update(x)
        result = graph_features_module.compute_baseline_zscore(11, stats)
        assert -2 < result < 2

    def test_scale_appropriate_eps_keeps_spike_bounded_not_millions(self, graph_features_module):
        # regression test for a real bug: with the tiny default eps=1e-6, a
        # host whose first two observed fan_out values happened to be equal
        # (std == 0 exactly -- common with only 2 samples) produced z-scores
        # in the millions the moment it changed at all, e.g. 5000000.00 in
        # the real run's top-fan_out_zscore table. A count-scale eps (1.0)
        # must keep this in a usable range instead.
        stats = graph_features_module.RunningStats()
        stats.update(1)
        stats.update(1)
        result = graph_features_module.compute_baseline_zscore(9, stats, eps=1.0)
        assert result == pytest.approx(8.0)
        assert result < 1000
