"""Unit tests for stratified_sample in scripts/09_shap_analysis.py.

The risky part: SHAP on the real 1.16M-row test set would be slow and
mostly redundant (it's 99.97% BENIGN), so a sample is needed -- but a
plain random sample of that set would lose rare classes entirely
(Infiltration is 36 rows total). stratified_sample must cap large groups
while keeping every row of groups smaller than the cap.
"""
import pandas as pd


class TestStratifiedSample:
    def test_caps_large_group_at_max_per_label(self, shap_analysis_module):
        df = pd.DataFrame({"Label": ["BENIGN"] * 1000, "x": range(1000)})
        result = shap_analysis_module.stratified_sample(df, "Label", max_per_label=50, random_state=42)
        assert len(result) == 50

    def test_keeps_all_rows_of_a_group_smaller_than_cap(self, shap_analysis_module):
        # this is the actual requirement: Infiltration (36 rows in the real
        # data) must not be truncated just because BENIGN is huge
        df = pd.DataFrame({
            "Label": ["BENIGN"] * 1000 + ["Infiltration"] * 36,
            "x": range(1036),
        })
        result = shap_analysis_module.stratified_sample(df, "Label", max_per_label=500, random_state=42)
        assert (result["Label"] == "Infiltration").sum() == 36

    def test_every_distinct_label_represented(self, shap_analysis_module):
        df = pd.DataFrame({
            "Label": ["BENIGN"] * 500 + ["DDoS"] * 300 + ["Bot"] * 10,
            "x": range(810),
        })
        result = shap_analysis_module.stratified_sample(df, "Label", max_per_label=100, random_state=42)
        assert set(result["Label"].unique()) == {"BENIGN", "DDoS", "Bot"}

    def test_deterministic_with_same_random_state(self, shap_analysis_module):
        df = pd.DataFrame({"Label": ["BENIGN"] * 200, "x": range(200)})
        r1 = shap_analysis_module.stratified_sample(df, "Label", max_per_label=20, random_state=7)
        r2 = shap_analysis_module.stratified_sample(df, "Label", max_per_label=20, random_state=7)
        assert list(r1["x"]) == list(r2["x"])
