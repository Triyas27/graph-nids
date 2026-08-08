"""Unit tests for the cleaning functions in scripts/01_prepare_data.py.

Each test builds a small synthetic DataFrame by hand (not the real
dataset) so these run in milliseconds and catch regressions in the
cleaning rules themselves, independent of the actual CIC-IDS2017 data.
"""
import numpy as np
import pandas as pd


class TestDropBlankRows:
    def test_drops_fully_blank_row(self, prepare_data):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.1", np.nan],
            "Flow Duration": [4.0, np.nan],
            "Label": ["BENIGN", np.nan],
        })
        result = prepare_data.drop_blank_rows(df)
        assert len(result) == 1
        assert result.iloc[0]["Source IP"] == "10.0.0.1"

    def test_keeps_row_with_label_but_other_cols_blank_is_still_dropped(self, prepare_data):
        # matches the real bug: the padding rows have Label blank too, but
        # the function only requires the *non-Label* columns to be blank
        df = pd.DataFrame({
            "Source IP": ["10.0.0.1", np.nan],
            "Flow Duration": [4.0, np.nan],
            "Label": ["BENIGN", "SomeLabel"],
        })
        result = prepare_data.drop_blank_rows(df)
        assert len(result) == 1

    def test_no_blank_rows_keeps_everything(self, prepare_data):
        df = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.2"],
            "Flow Duration": [4.0, 5.0],
            "Label": ["BENIGN", "DDoS"],
        })
        result = prepare_data.drop_blank_rows(df)
        assert len(result) == 2

    def test_partial_nan_row_is_not_dropped(self, prepare_data):
        # only SOME columns NaN -> real (if messy) data, must survive
        df = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.2"],
            "Flow Duration": [4.0, np.nan],
            "Label": ["BENIGN", "DDoS"],
        })
        result = prepare_data.drop_blank_rows(df)
        assert len(result) == 2


class TestFixLabelEncoding:
    def test_replaces_miscoded_en_dash(self, prepare_data):
        df = pd.DataFrame({"Label": ["Web Attack \x96 Brute Force"]})
        result = prepare_data.fix_label_encoding(df)
        assert result["Label"].iloc[0] == "Web Attack - Brute Force"

    def test_leaves_clean_labels_untouched(self, prepare_data):
        df = pd.DataFrame({"Label": ["BENIGN", "DDoS", "PortScan"]})
        result = prepare_data.fix_label_encoding(df)
        assert list(result["Label"]) == ["BENIGN", "DDoS", "PortScan"]

    def test_strips_stray_whitespace(self, prepare_data):
        df = pd.DataFrame({"Label": ["  BENIGN  "]})
        result = prepare_data.fix_label_encoding(df)
        assert result["Label"].iloc[0] == "BENIGN"

    def test_does_not_mutate_input(self, prepare_data):
        df = pd.DataFrame({"Label": ["Web Attack \x96 XSS"]})
        prepare_data.fix_label_encoding(df)
        assert df["Label"].iloc[0] == "Web Attack \x96 XSS"


class TestCoerceInfToNan:
    def test_replaces_positive_and_negative_inf(self, prepare_data):
        df = pd.DataFrame({
            "Flow Bytes/s": [1.0, np.inf, -np.inf, 2.5],
            "Label": ["BENIGN", "BENIGN", "BENIGN", "BENIGN"],
        })
        result = prepare_data.coerce_inf_to_nan(df)
        assert result["Flow Bytes/s"].iloc[0] == 1.0
        assert pd.isna(result["Flow Bytes/s"].iloc[1])
        assert pd.isna(result["Flow Bytes/s"].iloc[2])
        assert result["Flow Bytes/s"].iloc[3] == 2.5

    def test_ignores_non_numeric_columns(self, prepare_data):
        # a non-numeric column can't contain a float Inf, but make sure
        # the function doesn't choke on / touch object-dtype columns
        df = pd.DataFrame({
            "Flow Bytes/s": [1.0, np.inf],
            "Label": ["BENIGN", "DDoS"],
        })
        result = prepare_data.coerce_inf_to_nan(df)
        assert list(result["Label"]) == ["BENIGN", "DDoS"]

    def test_preserves_existing_nan(self, prepare_data):
        df = pd.DataFrame({"Flow Bytes/s": [np.nan, np.inf, 3.0]})
        result = prepare_data.coerce_inf_to_nan(df)
        assert result["Flow Bytes/s"].isna().sum() == 2
