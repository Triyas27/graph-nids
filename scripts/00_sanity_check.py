"""Week 0 sanity checks on the raw CIC-IDS2017 GeneratedLabelledFlows CSVs.

Reports, per day/file: row counts, label distribution, duplicate rows,
inf/NaN values, and column-name whitespace issues. Writes a summary to
reports/week0_sanity_check.md.
"""
import glob
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "extracted", "TrafficLabelling ")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week0_sanity_check.md")

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

lines = ["# Week 0 sanity check — CIC-IDS2017 GeneratedLabelledFlows\n"]

total_rows = 0
total_dupes = 0
total_blank = 0
all_labels = {}
issues = []

for path in files:
    name = os.path.basename(path)
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    n_rows = len(df)
    total_rows += n_rows

    # fully-blank rows: every column except Label is NaN (Label is often
    # also NaN, but don't require it in case a stray label sneaks in)
    blank_mask = df.drop(columns=["Label"]).isna().all(axis=1)
    n_blank = int(blank_mask.sum())
    total_blank += n_blank
    if n_blank:
        blank_idx = df.index[blank_mask]
        issues.append(
            f"- **{name}**: {n_blank:,} fully-blank rows "
            f"({n_blank / n_rows:.1%} of the file), rows {blank_idx.min()}-{blank_idx.max()} "
            f"(contiguous: {(blank_idx.max() - blank_idx.min() + 1) == n_blank})"
        )

    df_real = df.loc[~blank_mask]
    dupes = df_real.duplicated().sum()
    total_dupes += dupes

    label_counts = df_real["Label"].value_counts()
    for k, v in label_counts.items():
        all_labels[k] = all_labels.get(k, 0) + v
        if "\x96" in str(k) or "�" in str(k):
            issues.append(f"- **{name}**: mis-encoded label repr {k!r} ({v:,} rows) — Windows-1252 en-dash read via latin1/utf-8")

    # inf/NaN check on numeric columns, real rows only
    numeric_df = df_real.select_dtypes(include="number")
    n_inf = ((numeric_df == float("inf")) | (numeric_df == float("-inf"))).sum().sum()
    n_nan = numeric_df.isna().sum().sum()

    lines.append(f"## {name}\n")
    lines.append(f"- Rows (raw): {n_rows:,}\n")
    lines.append(f"- Fully-blank rows: {n_blank:,} ({n_blank / n_rows:.2%})\n")
    lines.append(f"- Rows (real, after dropping blanks): {len(df_real):,}\n")
    lines.append(f"- Duplicate rows (of real rows): {dupes:,} ({dupes / max(len(df_real), 1):.2%})\n")
    lines.append(f"- Inf values (numeric cols, real rows): {n_inf:,}\n")
    lines.append(f"- NaN values (numeric cols, real rows): {n_nan:,}\n")
    lines.append("- Label distribution (real rows):\n")
    for label, count in label_counts.items():
        lines.append(f"  - {label!r}: {count:,}\n")
    lines.append("\n")

lines.append("## Overall\n")
lines.append(f"- Total raw rows across all files: {total_rows:,}\n")
lines.append(f"- Total fully-blank rows: {total_blank:,}\n")
lines.append(f"- Total real rows (raw minus blank): {total_rows - total_blank:,}\n")
lines.append(f"- Total duplicate rows (of real rows): {total_dupes:,} ({total_dupes / (total_rows - total_blank):.2%})\n")
lines.append("- Combined label distribution (real rows):\n")
for label, count in sorted(all_labels.items(), key=lambda x: -x[1]):
    lines.append(f"  - {label!r}: {count:,}\n")

lines.append("\n## Known issues found\n")
if issues:
    for issue in issues:
        lines.append(issue + "\n")
else:
    lines.append("- None detected.\n")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"Wrote {OUT_PATH}")
print(f"Total rows: {total_rows:,}, total duplicates: {total_dupes:,} ({total_dupes / total_rows:.2%})")
