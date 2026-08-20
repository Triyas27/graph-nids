"""Week 3 improvement (part 2): enrich flows with the 5-minute graph
features from scripts/12, retrain the day-split classifier, and directly
check whether it now flags the 9 known lateral-movement pivot flows from
Week 3's chain extraction -- not just an aggregate recall number, the
exact flows we already know are real.

Reuses enrich_flows_with_graph_features (07), fit_eval_binary and
load_feature_matrix (08), and stratified_sample (09) unchanged.
"""
import gc
import json
import os

import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

SCRIPTS_DIR = os.path.dirname(__file__)


def _load_module(filename):
    import importlib.util
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.rstrip(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_enrich_mod = _load_module("07_enrich_with_graph_features.py")
_classifier_mod = _load_module("08_classifier_with_graph_features.py")
_shap_mod = _load_module("09_shap_analysis.py")
_finegrained_mod = _load_module("12_build_finegrained_graph_features.py")
_chain_mod = _load_module("10_extract_lateral_movement_chain.py")

parse_timestamp = _finegrained_mod.parse_timestamp
BUCKET_FREQ = _finegrained_mod.BUCKET_FREQ
enrich_flows_with_graph_features = _enrich_mod.enrich_flows_with_graph_features
fit_eval_binary = _classifier_mod.fit_eval_binary
DROP_COLS = _classifier_mod.DROP_COLS
GRAPH_FEATURE_PREFIXES = _classifier_mod.GRAPH_FEATURE_PREFIXES
TRAIN_DAYS = _classifier_mod.TRAIN_DAYS
TEST_DAYS = _classifier_mod.TEST_DAYS
stratified_sample = _shap_mod.stratified_sample
find_patient_zero = _chain_mod.find_patient_zero
build_post_compromise_timeline = _chain_mod.build_post_compromise_timeline
is_internal_ip = _chain_mod.is_internal_ip


def load_feature_matrix(cols, path):
    """Same memory-conscious pattern as 08_classifier_with_graph_features's
    version (read only these columns, cast to float32, drop the float64
    copy immediately), but parameterized on `path` since that one is
    hardcoded to the hourly-resolution enriched file, not this script's
    5-minute-resolution one."""
    df = pd.read_parquet(path, columns=cols)
    X = df.astype("float32")
    del df
    gc.collect()
    return X

FLOWS_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_clean.parquet")
FEATURES_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "finegrained_host_features.parquet")
ENRICHED_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_enriched_finegrained.parquet")
REPORT_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "reports", "week3_finegrained_classifier_eval.md")

WEEK2_DAY_SPLIT_F1 = {"flow_only": 0.4417, "hourly_graph": 0.6247}
WEEK2_STEALTHY_RECALL = {"Bot": 0.000, "Infiltration": 0.000, "PortScan": 0.048}


def main():
    # ---- identify the exact pivot flows FIRST, independent of graph features,
    # by rerunning the same tested chain-extraction logic from script 10 ----
    # Tagged by _row_id (original row position), not a (source, destination,
    # timestamp) composite key -- this dataset's timestamps are only
    # minute-precision, and a single host pair can exchange over a thousand
    # flows within the same recorded minute during a burst, so a composite
    # key is NOT unique enough to identify one specific flow (confirmed:
    # 192.168.10.8 -> 192.168.10.9 at exactly 15:07:00 alone matches 1,079
    # rows). _row_id survives the enrichment merge (a left-join adds columns,
    # never touches existing ones) and is saved as an explicit column since
    # to_parquet(index=False) below would otherwise drop the DataFrame index.
    flows = pd.read_parquet(FLOWS_PATH)
    flows["_row_id"] = flows.index
    flows["ParsedTimestamp"] = parse_timestamp(flows["Timestamp"])
    patient_zero = find_patient_zero(flows)
    compromise_time = flows.loc[flows["Label"] == "Infiltration", "ParsedTimestamp"].min()
    post_compromise = build_post_compromise_timeline(flows, patient_zero, compromise_time)
    pivot_row_ids = post_compromise.loc[
        post_compromise["is_novel_destination"] & post_compromise["Destination IP"].apply(is_internal_ip),
        "_row_id",
    ]
    print(f"Identified {len(pivot_row_ids)} pivot flows via chain extraction (expect 9)")

    # ---- enrich ----
    flows["HourBucket"] = parse_timestamp(flows["Timestamp"]).dt.floor(BUCKET_FREQ)
    host_features = pd.read_parquet(FEATURES_PATH)
    enriched = enrich_flows_with_graph_features(flows, host_features)
    assert len(enriched) == len(flows), "join duplicated or dropped rows"
    del flows, host_features
    gc.collect()
    enriched.to_parquet(ENRICHED_OUT_PATH, index=False)
    print(f"Wrote {ENRICHED_OUT_PATH} ({len(enriched):,} rows)")
    del enriched
    gc.collect()

    # ---- retrain day-split flow+graph model (same methodology as Week 2 Day 5) ----
    all_cols = pq.read_schema(ENRICHED_OUT_PATH).names
    non_feature_cols = set(DROP_COLS) | {"Label", "Day", "HourBucket", "ParsedTimestamp", "_row_id"}
    graph_feature_cols = [
        c for c in all_cols
        if c.startswith(GRAPH_FEATURE_PREFIXES) and "clustering_coefficient" not in c
    ]
    flow_feature_cols = [c for c in all_cols if c not in non_feature_cols and not c.startswith(GRAPH_FEATURE_PREFIXES)]
    all_feature_cols = flow_feature_cols + graph_feature_cols
    print(f"{len(flow_feature_cols)} flow features, {len(graph_feature_cols)} graph features (5-min resolution)")

    meta = pd.read_parquet(ENRICHED_OUT_PATH, columns=["Label", "Day", "Source IP", "Destination IP", "Timestamp", "_row_id"])
    y_binary = (meta["Label"] != "BENIGN").astype(int)
    train_mask = meta["Day"].isin(TRAIN_DAYS)
    test_mask = meta["Day"].isin(TEST_DAYS)

    X_all = load_feature_matrix(all_feature_cols, ENRICHED_OUT_PATH)
    lines = ["# Week 3 improvement -- day-split classifier retrained at 5-minute graph resolution\n\n"]
    model, f1 = fit_eval_binary(
        X_all[train_mask], y_binary[train_mask], X_all[test_mask], y_binary[test_mask],
        "Day-based split -- flow + 5-min graph features", lines,
        compare_f1=WEEK2_DAY_SPLIT_F1["hourly_graph"], compare_label="Week 2, hourly graph features",
    )
    preds_test = pd.Series(model.predict(X_all[test_mask]), index=X_all[test_mask].index)
    del X_all
    gc.collect()

    lines.append(
        f"Comparison points: Week 2 flow-only day-split F1 = {WEEK2_DAY_SPLIT_F1['flow_only']:.4f}, "
        f"Week 2 hourly-graph day-split F1 = {WEEK2_DAY_SPLIT_F1['hourly_graph']:.4f}, "
        f"this run (5-min graph) F1 = {f1:.4f}.\n\n"
    )

    # ---- per-class recall on a stratified sample, same method as Week 2 Day 6-7 ----
    test_meta = meta[test_mask]
    sample_idx = stratified_sample(test_meta, "Label", max_per_label=500, random_state=42).index
    X_sample = load_feature_matrix(all_feature_cols, ENRICHED_OUT_PATH).loc[sample_idx]
    sample_meta = meta.loc[sample_idx]
    sample_preds = model.predict(X_sample)
    lines.append("## Per-class recall (stratified sample, up to 500/label), vs. Week 2 hourly-resolution numbers\n\n")
    lines.append("| label | n | recall (5-min graph) | recall (Week 2 hourly, stealthy classes only) |\n|---|---|---|---|\n")
    for label in sorted(sample_meta["Label"].unique()):
        mask = (sample_meta["Label"] == label).values
        n = int(mask.sum())
        recall = float(sample_preds[mask].mean()) if label != "BENIGN" else float((sample_preds[mask] == 0).mean())
        week2_ref = WEEK2_STEALTHY_RECALL.get(label)
        week2_str = f"{week2_ref:.3f}" if week2_ref is not None else "--"
        lines.append(f"| {label} | {n} | {recall:.3f} | {week2_str} |\n")
        print(f"{label}: n={n}, recall={recall:.3f}" + (f" (Week 2 hourly: {week2_ref:.3f})" if week2_ref is not None else ""))
    lines.append("\n")
    del X_sample
    gc.collect()

    # ---- direct check: the exact 9 pivot flows, matched by _row_id (exact
    # original row identity, not a composite key -- see the note where
    # _row_id is built, above) ----
    pivot_row_mask = meta["_row_id"].isin(pivot_row_ids)
    X_pivots = load_feature_matrix(all_feature_cols, ENRICHED_OUT_PATH).loc[meta.index[pivot_row_mask]]
    pivot_preds = model.predict(X_pivots)
    pivot_proba = model.predict_proba(X_pivots)[:, 1]
    pivot_meta = meta[pivot_row_mask].copy()
    pivot_meta["flagged"] = pivot_preds.astype(bool)
    pivot_meta["probability"] = pivot_proba
    n_pivot_flagged = int(pivot_meta["flagged"].sum())
    assert len(pivot_meta) == len(pivot_row_ids), (
        f"expected exactly {len(pivot_row_ids)} pivot rows, matched {len(pivot_meta)}"
    )

    lines.append("## Direct check: the 9 known lateral-movement pivot flows\n\n")
    lines.append(
        f"Week 2 (hourly graph features) caught 1 of these 9. This run (5-minute graph features) "
        f"catches **{n_pivot_flagged} of {len(pivot_meta)}**.\n\n"
    )
    lines.append(pivot_meta[["Timestamp", "Destination IP", "probability", "flagged"]].sort_values("Timestamp").to_markdown(index=False))
    lines.append("\n")
    print(f"\nPivot flows caught: {n_pivot_flagged} / {len(pivot_meta)} (Week 2 hourly: 1 / 9)")

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nWrote {REPORT_OUT_PATH}")


if __name__ == "__main__":
    main()
