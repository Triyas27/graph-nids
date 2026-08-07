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

frames = []
for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
    name = os.path.basename(path)
    day = FILE_TO_DAY[name]

    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    blank_mask = df.drop(columns=["Label"]).isna().all(axis=1)
    n_blank = int(blank_mask.sum())
    df = df.loc[~blank_mask].copy()

    df["Label"] = df["Label"].str.replace("\x96", "-", regex=False).str.strip()
    df["Day"] = day
    df["SourceFile"] = name

    print(f"{name}: {len(df):,} real rows (dropped {n_blank:,} blank), day={day}")
    frames.append(df)

full = pd.concat(frames, ignore_index=True)

# +/-Inf -> NaN in numeric columns (Flow Bytes/s, Flow Packets/s division by zero)
numeric_cols = full.select_dtypes(include="number").columns
full[numeric_cols] = full[numeric_cols].mask(np.isinf(full[numeric_cols]))

print()
print(f"Combined: {len(full):,} rows, {full['Label'].nunique()} distinct labels")
print(full["Day"].value_counts())
print(full["Label"].value_counts())

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
full.to_parquet(OUT_PATH, index=False)
print(f"\nWrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")
