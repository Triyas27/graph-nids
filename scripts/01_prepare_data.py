"""Load the 8 raw GeneratedLabelledFlows CSVs, apply the Week 0 fixes, and
write a single cleaned parquet file for downstream modeling.

Fixes applied (see reports/week0_sanity_check.md):
- Drop the fully-blank padding rows (Thursday-WorkingHours-Morning-WebAttacks
  has ~289k of them).
- Normalize whitespace in column names.
- Fix the mis-encoded en-dash (\x96) in Web Attack labels.
- Tag every row with its source day (Monday..Friday) for the day-based split.
- Coerce +/-Inf to NaN (division-by-zero artifacts in rate columns); left as
  NaN for the modeling step to impute using train-only statistics.

The three cleaning steps below are standalone functions (rather than inline
code) specifically so they can be unit-tested in isolation against small
synthetic DataFrames — see tests/test_prepare_data.py. They're pure
DataFrame -> DataFrame transforms with no dependency on the real dataset.
"""
import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "extracted", "TrafficLabelling ")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flows_clean.parquet")

FILE_TO_DAY = {
    "Monday-WorkingHours.pcap_ISCX.csv": "Monday",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "Tuesday",
    "Wednesday-workingHours.pcap_ISCX.csv": "Wednesday",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Thursday",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Thursday",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "Friday",
}


def drop_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where every column except Label is NaN (CICFlowMeter
    padding artifact — see the Thursday-Morning-WebAttacks file)."""
    blank_mask = df.drop(columns=["Label"]).isna().all(axis=1)
    return df.loc[~blank_mask].copy()


def fix_label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the mis-encoded Windows-1252 en-dash (\x96) in Web Attack
    labels with a plain hyphen, and strip stray whitespace."""
    df = df.copy()
    df["Label"] = df["Label"].str.replace("\x96", "-", regex=False).str.strip()
    return df


def coerce_inf_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-Inf with NaN in numeric columns (division-by-zero
    artifacts in rate columns like Flow Bytes/s)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].mask(np.isinf(df[numeric_cols]))
    return df


def main():
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        name = os.path.basename(path)
        day = FILE_TO_DAY[name]

        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        n_before = len(df)
        df = drop_blank_rows(df)
        n_blank = n_before - len(df)

        df = fix_label_encoding(df)
        df["Day"] = day
        df["SourceFile"] = name

        print(f"{name}: {len(df):,} real rows (dropped {n_blank:,} blank), day={day}")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = coerce_inf_to_nan(full)

    print()
    print(f"Combined: {len(full):,} rows, {full['Label'].nunique()} distinct labels")
    print(full["Day"].value_counts())
    print(full["Label"].value_counts())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    full.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
