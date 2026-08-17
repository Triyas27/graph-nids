"""Unit tests for scripts/10_extract_lateral_movement_chain.py.

find_patient_zero must fail loudly (not silently pick one) if an attack
label ever has more than one source -- the whole chain-extraction design
assumes a single patient zero. build_post_compromise_timeline's novelty
tagging is the flow-level replacement for Week 2's hourly novelty_score,
so it needs to get the "before vs. at-or-after compromise_time" boundary
and the running distinct-count exactly right.
"""
import pandas as pd
import pytest


class TestFindPatientZero:
    def test_returns_the_single_source(self, chain_module):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.8", "10.0.0.8", "10.0.0.1"],
            "Label": ["Infiltration", "Infiltration", "BENIGN"],
        })
        assert chain_module.find_patient_zero(df) == "10.0.0.8"

    def test_raises_if_attack_has_multiple_sources(self, chain_module):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.8", "10.0.0.9"],
            "Label": ["Infiltration", "Infiltration"],
        })
        with pytest.raises(ValueError):
            chain_module.find_patient_zero(df)

    def test_raises_if_label_not_present(self, chain_module):
        df = pd.DataFrame({"Source IP": ["10.0.0.1"], "Label": ["BENIGN"]})
        with pytest.raises(ValueError):
            chain_module.find_patient_zero(df)


class TestIsInternalIp:
    def test_internal_subnet(self, chain_module):
        assert chain_module.is_internal_ip("192.168.10.51") is True

    def test_external_ip(self, chain_module):
        assert chain_module.is_internal_ip("205.174.165.73") is False

    def test_external_ip_that_merely_contains_the_octets(self, chain_module):
        # must match as a prefix, not just "contains 192.168" anywhere
        assert chain_module.is_internal_ip("10.192.168.1") is False


class TestBuildPostCompromiseTimeline:
    def _df(self):
        return pd.DataFrame({
            "Source IP": ["10.0.0.8", "10.0.0.8", "10.0.0.8", "10.0.0.8", "10.0.0.8"],
            "Destination IP": ["10.0.0.2", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.3"],
            "ParsedTimestamp": pd.to_datetime([
                "2017-07-06 10:00",  # before compromise -- establishes .2 as a prior destination
                "2017-07-06 15:05",  # at/after compromise, .2 is NOT novel (seen before)
                "2017-07-06 15:06",  # .3 IS novel
                "2017-07-06 15:07",  # .4 IS novel
                "2017-07-06 15:08",  # .3 again, not novel this time (already seen post-compromise)
            ]),
        })

    def test_excludes_flows_before_compromise_time(self, chain_module):
        result = chain_module.build_post_compromise_timeline(
            self._df(), "10.0.0.8", pd.Timestamp("2017-07-06 15:00")
        )
        assert len(result) == 4
        assert result["ParsedTimestamp"].min() >= pd.Timestamp("2017-07-06 15:00")

    def test_prior_destination_is_not_novel(self, chain_module):
        result = chain_module.build_post_compromise_timeline(
            self._df(), "10.0.0.8", pd.Timestamp("2017-07-06 15:00")
        )
        first_row = result.iloc[0]
        assert first_row["Destination IP"] == "10.0.0.2"
        assert first_row["is_novel_destination"] == False

    def test_new_destination_is_novel_only_on_first_contact(self, chain_module):
        result = chain_module.build_post_compromise_timeline(
            self._df(), "10.0.0.8", pd.Timestamp("2017-07-06 15:00")
        )
        dest_3_rows = result[result["Destination IP"] == "10.0.0.3"]
        assert list(dest_3_rows["is_novel_destination"]) == [True, False]

    def test_cumulative_distinct_destinations_increases_correctly(self, chain_module):
        result = chain_module.build_post_compromise_timeline(
            self._df(), "10.0.0.8", pd.Timestamp("2017-07-06 15:00")
        )
        # order: .2 (seen before -> cum stays at prior+1=1 new total... but
        # cumulative counts ALL destinations ever seen including prior ones)
        assert list(result["cumulative_distinct_destinations"]) == [1, 2, 3, 3]

    def test_records_prior_destination_count(self, chain_module):
        result = chain_module.build_post_compromise_timeline(
            self._df(), "10.0.0.8", pd.Timestamp("2017-07-06 15:00")
        )
        assert (result["n_prior_destinations"] == 1).all()

    def test_only_includes_flows_initiated_by_patient_zero(self, chain_module):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.8", "10.0.0.9"],
            "Destination IP": ["10.0.0.2", "10.0.0.8"],  # second row: someone ELSE contacts patient zero
            "ParsedTimestamp": pd.to_datetime(["2017-07-06 15:05", "2017-07-06 15:06"]),
        })
        result = chain_module.build_post_compromise_timeline(df, "10.0.0.8", pd.Timestamp("2017-07-06 15:00"))
        assert len(result) == 1
        assert result.iloc[0]["Destination IP"] == "10.0.0.2"
