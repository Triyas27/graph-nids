"""Week 2, Day 5: retrain the classifier with graph features and compare
against a flow-only baseline, FIXED to avoid a leakage bug found in the
first attempt.

The bug: graph features are computed per (Host, HourBucket), a coarser
unit than a flow -- e.g. all ~68,843 flows from 192.168.10.8 at Thursday
15:00 share the exact same src_flow_volume_zscore value. A row-level
random split can put siblings from the same host-hour block on both sides
of train/test, letting the model partly memorize block identity instead of
learning a generalizable pattern (this produced a suspicious 0.99999
binary F1). The fix: split by WHOLE HOUR instead of by row. Since every
graph feature (source or destination side) is scoped to a single
HourBucket, if an hour is wholly in train or wholly in test, no block can
ever straddle the split -- this closes the leakage regardless of which
host or which side of the flow it's on.

Because this changes the evaluation protocol, Week 1's old row-level
random-split number is no longer a valid comparison point -- the flow-only
baseline is rerun here under the same hour-grouped split for a fair
apples-to-apples comparison. The day-based split (train Mon-Wed, test
Thu-Fri) doesn't have this leakage problem (the two day-groups never share
an hour), so it's rerun as-is, plus feature importances are pulled from
that model to check whether graph features are responsible for its
collapse (0.442 -> 0.005 in the first attempt) via shortcut learning.

Second problem found, after fixing the above: F1 still collapsed under
BOTH split protocols even with leakage closed, via recall collapsing while
precision stayed ~1.0 -- a model relying on one narrow memorized rule
rather than a general one. Feature importance traced this to
src_clustering_coefficient (gain 70,461, 24x the next-highest feature):
it's 0.0 for the large majority of host-hours (needs 2+ mutually-connected
neighbors, which most low-fan-out hosts never have), so any nonzero value
is rare and easy to memorize instead of generalize from.
src_/dst_clustering_coefficient are excluded from the feature set below.
"""
import gc
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flows_enriched.parquet")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week2_classifier_with_graph_features.md")
MODEL_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "day_split_graph_model.json")
MODEL_FEATURES_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "day_split_graph_model_features.json")

DROP_COLS = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp",
    "Day", "SourceFile", "HourBucket", "Label",
]
GRAPH_FEATURE_PREFIXES = ("src_", "dst_")
TRAIN_DAYS = {"Monday", "Tuesday", "Wednesday"}
TEST_DAYS = {"Thursday", "Friday"}


def group_split_by_hour(hours: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Assign every row to train or test based on its WHOLE HourBucket, so
    no host-hour block (which every graph feature is scoped to) can
    straddle both sides. Returns (train_mask, test_mask) aligned to
    `hours`'s index."""
    unique_hours = pd.Series(hours.unique())
    shuffled = unique_hours.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_test = max(1, round(len(shuffled) * test_size))
    test_hours = set(shuffled.iloc[:n_test])
    train_mask = ~hours.isin(test_hours)
    test_mask = hours.isin(test_hours)
    return train_mask, test_mask


def fit_eval_binary(X_train, y_train, X_test, y_test, section_title, lines, compare_f1=None, compare_label=""):
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["BENIGN", "ATTACK"], digits=3)
    f1 = f1_score(y_test, preds)
    lines.append(f"### {section_title}\n")
    lines.append(f"Train rows: {len(X_train):,}, test rows: {len(X_test):,}\n")
    lines.append(f"Binary F1 (ATTACK class): {f1:.4f}")
    if compare_f1 is not None:
        lines.append(f"  ({compare_label}: {compare_f1:.4f}, delta: {f1 - compare_f1:+.4f})")
    lines.append("\n\n")
    lines.append("```\n" + report + "\n```\n\n")
    print(section_title, "F1:", f1)
    return model, f1


def load_metadata():
    """Load only Label/Day/HourBucket (tiny) to build split masks, without
    ever touching the 79 flow-stat or 26 graph-feature columns."""
    df = pd.read_parquet(DATA_PATH, columns=["Label", "Day", "HourBucket"])
    y_binary = (df["Label"] != "BENIGN").astype(int)
    hours = df["HourBucket"].copy()
    day = df["Day"].copy()
    return y_binary, hours, day


def load_feature_matrix(cols):
    """Read ONLY these columns from parquet and immediately cast to
    float32 -- called separately per feature-set (flow-only, then
    flow+graph) so the raw float64 table for one set is never resident at
    the same time as the other set's fit. Holding the full float64 table
    plus both float32 matrices at once (the first attempt's approach)
    exhausted available RAM on this machine."""
    df = pd.read_parquet(DATA_PATH, columns=cols)
    X = df.astype("float32")
    del df
    gc.collect()
    return X


def main():
    all_cols = pq.read_schema(DATA_PATH).names
    non_feature_cols = set(DROP_COLS) | {"Label", "Day", "HourBucket"}
    flow_feature_cols = [
        c for c in all_cols
        if c not in non_feature_cols and not c.startswith(GRAPH_FEATURE_PREFIXES)
    ]
    # src/dst_clustering_coefficient excluded: diagnosed as a severe
    # overfitting shortcut (gain 70,461 -- 24x the next-highest feature),
    # because it's exactly 0.0 for ~75%+ of host-hours (needs 2+ mutually-
    # connected neighbors, which most low-fan-out hosts never have), making
    # any nonzero value rare and easy to memorize rather than generalize.
    # See reports/week2_classifier_with_graph_features.md for the first
    # (all-graph-features) attempt this was diagnosed from.
    EXCLUDED_GRAPH_COLS = {"src_clustering_coefficient", "dst_clustering_coefficient"}
    graph_feature_cols = [
        c for c in all_cols
        if c.startswith(GRAPH_FEATURE_PREFIXES) and c not in EXCLUDED_GRAPH_COLS
    ]
    all_feature_cols = flow_feature_cols + graph_feature_cols
    print(f"{len(flow_feature_cols)} flow features, {len(graph_feature_cols)} graph features "
          f"(excluded: {sorted(EXCLUDED_GRAPH_COLS)})")

    y_binary, hours, day = load_metadata()
    train_mask, test_mask = group_split_by_hour(hours, test_size=0.2, random_state=42)
    print(f"Hour-grouped split: {hours[train_mask].nunique()} train hours, {hours[test_mask].nunique()} test hours")
    train_mask_d = day.isin(TRAIN_DAYS)
    test_mask_d = day.isin(TEST_DAYS)

    lines = ["# Week 2, Day 5 — classifier with graph features\n\n"]
    lines.append(
        "This is the third attempt, after two real problems were found and fixed:\n\n"
        "1. **Row-level random split leaked block identity.** Graph features are scoped "
        "to (Host, HourBucket), so many rows share an identical value; a naive random "
        "row split could put siblings from the same block on both sides of train/test, "
        "producing a suspicious 0.99999 F1 that didn't reflect real generalization. "
        "Fixed by splitting on WHOLE HOURS instead (**hour-grouped split**, below) -- "
        "no block can ever straddle train/test.\n"
        "2. **Even with leakage fixed, graph features still collapsed both splits** "
        "(hour-grouped 0.913 -> 0.228, day-based 0.442 -> 0.005), both times via "
        "recall collapsing while precision stayed ~1.0 -- the model finding one narrow, "
        "memorized rule instead of a general one. Feature importance pinned this on "
        "`src_clustering_coefficient` (gain 70,461 -- 24x the next-highest feature): "
        "it's exactly 0.0 for the large majority of host-hours (needs 2+ mutually-"
        "connected neighbors, which most low-fan-out hosts never have), so any nonzero "
        "value is rare and easy to memorize rather than generalize from. **Excluded "
        "`src_/dst_clustering_coefficient`** for this run.\n\n"
        "Two split protocols: **hour-grouped** (whole hours assigned to train/test) and "
        "**day-based** (train Mon-Wed, test Thu-Fri, unchanged from Week 1). Both are "
        "run flow-only and flow+graph for a direct before/after comparison.\n\n"
    )

    # ---- flow-only column set ----
    X_flow = load_feature_matrix(flow_feature_cols)
    _, f1_hour_flow = fit_eval_binary(
        X_flow[train_mask], y_binary[train_mask], X_flow[test_mask], y_binary[test_mask],
        "Hour-grouped split — flow-only", lines,
    )
    _, f1_day_flow = fit_eval_binary(
        X_flow[train_mask_d], y_binary[train_mask_d], X_flow[test_mask_d], y_binary[test_mask_d],
        "Day-based split — flow-only", lines,
    )
    del X_flow
    gc.collect()

    # ---- flow + graph column set ----
    X_all = load_feature_matrix(all_feature_cols)

    _, f1_hour_all = fit_eval_binary(
        X_all[train_mask], y_binary[train_mask], X_all[test_mask], y_binary[test_mask],
        "Hour-grouped split — flow + graph features", lines,
        compare_f1=f1_hour_flow, compare_label="flow-only, same split",
    )
    model_day_all, f1_day_all = fit_eval_binary(
        X_all[train_mask_d], y_binary[train_mask_d], X_all[test_mask_d], y_binary[test_mask_d],
        "Day-based split — flow + graph features", lines,
        compare_f1=f1_day_flow, compare_label="flow-only, same split",
    )

    # Persist this model (the one showing the real F1 gain) for Week 2
    # Day 6-7's SHAP analysis, so it's explaining the exact fitted model
    # rather than a freshly retrained (should-be-identical-but-not-verified) one.
    model_day_all.save_model(MODEL_OUT_PATH)
    with open(MODEL_FEATURES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_feature_cols, f)
    print(f"Saved model to {MODEL_OUT_PATH}")

    # ---- Diagnosis: feature importance on the day-split flow+graph model ----
    importances = model_day_all.get_booster().get_score(importance_type="gain")
    top20 = sorted(importances.items(), key=lambda x: -x[1])[:20]
    n_graph_in_top20 = sum(1 for name, _ in top20 if name.startswith(GRAPH_FEATURE_PREFIXES))
    lines.append("### Diagnosis: feature importance, day-split flow+graph model\n\n")
    lines.append(
        f"{n_graph_in_top20}/20 of the top-gain features are graph features "
        f"({', '.join(graph_feature_cols[:3])}, ...).\n\n"
    )
    lines.append("| feature | gain |\n|---|---|\n")
    for name, gain in top20:
        lines.append(f"| {name} | {gain:.1f} |\n")
    lines.append("\n")
    print(f"Top-20 day-split features, graph features in top 20: {n_graph_in_top20}/20")
    print(top20)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
