"""Week 2, Day 6-7: SHAP analysis on the day-split flow+graph model from
Day 5 -- which features actually drove the F1 gain (0.442 -> 0.625), and
for which specific attack types?

Loads the model persisted by scripts/08_classifier_with_graph_features.py
(the exact fitted model, not a retrained stand-in) and explains it with
shap.TreeExplainer on a stratified sample of the Thu-Fri test set (SHAP on
the full 1.16M rows is unnecessary and slow; a few hundred rows per attack
type is enough to rank features and compare per-class reliance).

Two things asked of the data:
  1. Global ranking: mean |SHAP value| per feature, compared against
     Day 5's gain-based importance as a sanity cross-check (gain can be
     inflated by a feature used in few but decisive splits; SHAP averages
     actual contribution across every row).
  2. Per-attack-type breakdown: for each true label in the test set, what
     fraction of its graph-features vs flow-features' SHAP contribution
     goes into the ATTACK direction, and does that line up with which
     attack types Week 1's zero-day test and Week 2's by-hand host checks
     already flagged as needing connectivity/volume signal (PortScan,
     Infiltration) versus which were already flow-detectable (DoS-family)?
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
import xgboost as xgb

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flows_enriched.parquet")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "day_split_graph_model.json")
MODEL_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "day_split_graph_model_features.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week2_shap_analysis.md")
FIG_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week2_shap_summary.png")

TEST_DAYS = {"Thursday", "Friday"}
GRAPH_FEATURE_PREFIXES = ("src_", "dst_")
MAX_ROWS_PER_LABEL = 500
RANDOM_STATE = 42


def stratified_sample(df: pd.DataFrame, label_col: str, max_per_label: int, random_state: int) -> pd.DataFrame:
    """Up to `max_per_label` rows per distinct value of `label_col` (all
    rows if a group is smaller), for a SHAP sample that still represents
    rare classes like Infiltration (36 rows total) rather than losing them
    to a plain random sample of a 1M+-row, 99.97%-BENIGN test set."""
    parts = [
        group.sample(n=min(len(group), max_per_label), random_state=random_state)
        for _, group in df.groupby(label_col)
    ]
    return pd.concat(parts)


def main():
    with open(MODEL_FEATURES_PATH) as f:
        feature_cols = json.load(f)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    needed_cols = feature_cols + ["Label", "Day"]
    all_cols = pq.read_schema(DATA_PATH).names
    assert set(needed_cols) <= set(all_cols), "saved feature list doesn't match current enriched parquet"

    df = pd.read_parquet(DATA_PATH, columns=needed_cols)
    df = df[df["Day"].isin(TEST_DAYS)]

    sample = stratified_sample(df, "Label", MAX_ROWS_PER_LABEL, RANDOM_STATE)
    print(f"SHAP sample: {len(sample):,} rows across {sample['Label'].nunique()} labels")
    print(sample["Label"].value_counts())

    X_sample = sample[feature_cols].astype("float32")
    y_true_binary = (sample["Label"] != "BENIGN").astype(int)
    preds = model.predict(X_sample)

    explainer = shap.TreeExplainer(model.get_booster())
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    graph_cols = [c for c in feature_cols if c.startswith(GRAPH_FEATURE_PREFIXES)]

    lines = ["# Week 2, Day 6-7 — SHAP analysis of the day-split flow+graph model\n\n"]
    lines.append(
        f"SHAP TreeExplainer on a stratified sample ({len(sample):,} rows, up to "
        f"{MAX_ROWS_PER_LABEL}/label) of the Thursday-Friday test set, explaining the "
        f"exact model persisted from Week 2 Day 5 (day-based split, F1 0.625).\n\n"
    )

    lines.append("## Global feature ranking (mean |SHAP value|)\n\n")
    lines.append("| rank | feature | mean \\|SHAP\\| | type |\n|---|---|---|---|\n")
    for i, (name, val) in enumerate(mean_abs_shap.head(20).items(), 1):
        kind = "graph" if name.startswith(GRAPH_FEATURE_PREFIXES) else "flow"
        lines.append(f"| {i} | {name} | {val:.4f} | {kind} |\n")
    n_graph_in_top20 = sum(1 for name in mean_abs_shap.head(20).index if name.startswith(GRAPH_FEATURE_PREFIXES))
    lines.append(f"\n{n_graph_in_top20}/20 of the top SHAP-ranked features are graph features.\n\n")

    # ---- per-attack-type breakdown ----
    lines.append("## Per-attack-type: recall and graph-feature reliance\n\n")
    lines.append(
        "For each true label, recall in this sample (fraction flagged ATTACK), and "
        "what share of the total |SHAP| contribution (summed across all features, "
        "this label's rows only) comes from graph features vs flow features.\n\n"
    )
    lines.append("| label | n | recall | mean total \\|SHAP\\| | graph share | flow share |\n|---|---|---|---|---|---|\n")

    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=sample.index)
    abs_shap_df = shap_df.abs()
    graph_share_per_row = abs_shap_df[graph_cols].sum(axis=1) / abs_shap_df.sum(axis=1)

    results = []
    for label in sorted(sample["Label"].unique()):
        mask = (sample["Label"] == label).values
        n = int(mask.sum())
        recall = float(preds[mask].mean()) if label != "BENIGN" else float((preds[mask] == 0).mean())
        mean_total_shap = float(abs_shap_df.loc[mask].sum(axis=1).mean())
        graph_share = float(graph_share_per_row.loc[mask].mean())
        results.append((label, n, recall, mean_total_shap, graph_share))
        lines.append(f"| {label} | {n} | {recall:.3f} | {mean_total_shap:.3f} | {graph_share:.1%} | {1 - graph_share:.1%} |\n")

    lines.append("\n")

    # ---- figure: top-20 global bar chart ----
    fig, ax = plt.subplots(figsize=(8, 6))
    top20 = mean_abs_shap.head(20).iloc[::-1]
    colors = ["#d97706" if name.startswith(GRAPH_FEATURE_PREFIXES) else "#2563eb" for name in top20.index]
    ax.barh(top20.index, top20.values, color=colors)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top 20 features, day-split flow+graph model\n(orange = graph feature, blue = flow feature)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Wrote {FIG_PATH}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {OUT_PATH}")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
