"""Week 3 improvement, take 2: a genuinely per-flow novelty feature,
not bucketed at any granularity.

The 5-minute graph features (script 13) helped a lot in aggregate and
specifically fixed PortScan/Web-Attack-SQLi, but did NOT catch the 9
known lateral-movement pivot flows -- because that host's traffic is
dense enough (8,000+ flows per 5-minute window during the burst) that
even 5-minute bucketing still dilutes 1 real event among thousands of
simultaneous benign ones sharing the same bucket-level feature value.

The fix: stop bucketing entirely for this specific signal. For every
single flow, ask directly "has this Source IP ever contacted this
Destination IP before, at any earlier point in the whole capture" --
exactly the logic that found the pivot chain by hand in script 10
(is_novel_destination), generalized from one host to every flow in the
dataset as a trainable feature. No aggregation step exists to dilute it.

Combines with the already-validated 5-minute graph features (script 13's
output) rather than replacing them, since those still help other attack
types (PortScan, Web Attack SQLi) that this new feature isn't aimed at.
"""
import gc
import os

import numpy as np
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


_classifier_mod = _load_module("08_classifier_with_graph_features.py")
_finegrained_mod = _load_module("12_build_finegrained_graph_features.py")
_chain_mod = _load_module("10_extract_lateral_movement_chain.py")

parse_timestamp = _finegrained_mod.parse_timestamp
fit_eval_binary = _classifier_mod.fit_eval_binary
DROP_COLS = _classifier_mod.DROP_COLS
GRAPH_FEATURE_PREFIXES = _classifier_mod.GRAPH_FEATURE_PREFIXES
TRAIN_DAYS = _classifier_mod.TRAIN_DAYS
TEST_DAYS = _classifier_mod.TEST_DAYS
find_patient_zero = _chain_mod.find_patient_zero
build_post_compromise_timeline = _chain_mod.build_post_compromise_timeline
is_internal_ip = _chain_mod.is_internal_ip

FLOWS_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_clean.parquet")
FINEGRAINED_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_enriched_finegrained.parquet")
ENRICHED_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_enriched_v3.parquet")
REPORT_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "reports", "week3_perflow_novelty_eval.md")

WEEK2_DAY_SPLIT_F1 = {"flow_only": 0.4417, "hourly_graph": 0.6247}
PRIOR_RUN = {"5min_graph": 0.8122}
PRIOR_STEALTHY_RECALL = {"Bot": 0.000, "Infiltration": 0.000, "PortScan": 0.048}


def compute_per_flow_novelty(df: pd.DataFrame, source_col: str, dest_col: str, timestamp_col: str) -> pd.Series:
    """True where `source_col`'s value has never sent a flow to
    `dest_col`'s value at any EARLIER timestamp -- processed in
    chronological order regardless of the input's row order (ties at the
    same timestamp broken by original row order, via a stable sort). The
    first-ever flow from a given source is always True (no prior
    history). No bucketing anywhere -- this is the whole point."""
    order = np.argsort(df[timestamp_col].values, kind="stable")
    src_vals = df[source_col].to_numpy()
    dst_vals = df[dest_col].to_numpy()
    result = np.zeros(len(df), dtype=bool)
    seen: dict = {}
    for pos in order:
        src, dst = src_vals[pos], dst_vals[pos]
        dests = seen.get(src)
        if dests is None:
            seen[src] = {dst}
            result[pos] = True
        elif dst not in dests:
            dests.add(dst)
            result[pos] = True
    return pd.Series(result, index=df.index)


def load_feature_matrix(cols, path):
    df = pd.read_parquet(path, columns=cols)
    X = df.astype("float32")
    del df
    gc.collect()
    return X


def main():
    # ---- per-flow novelty, no bucketing ----
    flows = pd.read_parquet(FLOWS_PATH, columns=["Source IP", "Destination IP", "Timestamp", "Label"])
    flows["_row_id"] = flows.index
    flows["ParsedTimestamp"] = parse_timestamp(flows["Timestamp"])

    flows["novel_dst_for_src"] = compute_per_flow_novelty(flows, "Source IP", "Destination IP", "ParsedTimestamp")
    flows["novel_src_for_dst"] = compute_per_flow_novelty(flows, "Destination IP", "Source IP", "ParsedTimestamp")
    print(f"novel_dst_for_src True rate: {flows['novel_dst_for_src'].mean():.3%}")
    print(f"novel_src_for_dst True rate: {flows['novel_src_for_dst'].mean():.3%}")

    # identify the 9 known pivot flows the same way script 10/13 did, for
    # the direct check later
    patient_zero = find_patient_zero(flows)
    compromise_time = flows.loc[flows["Label"] == "Infiltration", "ParsedTimestamp"].min()
    post_compromise = build_post_compromise_timeline(flows, patient_zero, compromise_time)
    pivot_row_ids = post_compromise.loc[
        post_compromise["is_novel_destination"] & post_compromise["Destination IP"].apply(is_internal_ip),
        "_row_id",
    ]
    print(f"Identified {len(pivot_row_ids)} pivot flows (expect 9)")

    novelty_cols = flows[["_row_id", "novel_dst_for_src", "novel_src_for_dst"]].copy()
    del flows
    gc.collect()

    # ---- merge onto the already-validated 5-minute graph features ----
    finegrained = pd.read_parquet(FINEGRAINED_PATH)
    enriched = finegrained.merge(novelty_cols, on="_row_id", how="left")
    assert len(enriched) == len(finegrained), "merge duplicated or dropped rows"
    assert enriched["novel_dst_for_src"].isna().sum() == 0, "unmatched rows after merge"
    del finegrained, novelty_cols
    gc.collect()
    enriched.to_parquet(ENRICHED_OUT_PATH, index=False)
    print(f"Wrote {ENRICHED_OUT_PATH} ({len(enriched):,} rows)")
    del enriched
    gc.collect()

    # ---- retrain day-split classifier: flow + 5-min graph + per-flow novelty ----
    all_cols = pq.read_schema(ENRICHED_OUT_PATH).names
    non_feature_cols = set(DROP_COLS) | {"Label", "Day", "HourBucket", "ParsedTimestamp", "_row_id"}
    graph_feature_cols = [c for c in all_cols if c.startswith(GRAPH_FEATURE_PREFIXES) and "clustering_coefficient" not in c]
    novelty_feature_cols = ["novel_dst_for_src", "novel_src_for_dst"]
    flow_feature_cols = [
        c for c in all_cols
        if c not in non_feature_cols and not c.startswith(GRAPH_FEATURE_PREFIXES) and c not in novelty_feature_cols
    ]
    all_feature_cols = flow_feature_cols + graph_feature_cols + novelty_feature_cols
    print(f"{len(flow_feature_cols)} flow + {len(graph_feature_cols)} 5-min graph + "
          f"{len(novelty_feature_cols)} per-flow novelty features")

    meta = pd.read_parquet(ENRICHED_OUT_PATH, columns=["Label", "Day", "_row_id", "Timestamp", "Source IP", "Destination IP"])
    y_binary = (meta["Label"] != "BENIGN").astype(int)
    train_mask = meta["Day"].isin(TRAIN_DAYS)
    test_mask = meta["Day"].isin(TEST_DAYS)

    X_all = load_feature_matrix(all_feature_cols, ENRICHED_OUT_PATH)
    lines = ["# Week 3 improvement, take 2 -- per-flow novelty feature (no bucketing)\n\n"]
    model, f1 = fit_eval_binary(
        X_all[train_mask], y_binary[train_mask], X_all[test_mask], y_binary[test_mask],
        "Day-based split -- flow + 5-min graph + per-flow novelty", lines,
        compare_f1=PRIOR_RUN["5min_graph"], compare_label="prior run, 5-min graph only",
    )
    del X_all
    gc.collect()

    lines.append(
        f"Comparison points: flow-only = {WEEK2_DAY_SPLIT_F1['flow_only']:.4f}, "
        f"hourly graph = {WEEK2_DAY_SPLIT_F1['hourly_graph']:.4f}, "
        f"5-min graph = {PRIOR_RUN['5min_graph']:.4f}, this run = {f1:.4f}.\n\n"
    )

    # ---- per-class recall ----
    _shap_mod = _load_module("09_shap_analysis.py")
    stratified_sample = _shap_mod.stratified_sample
    test_meta = meta[test_mask]
    sample_idx = stratified_sample(test_meta, "Label", max_per_label=500, random_state=42).index
    X_sample = load_feature_matrix(all_feature_cols, ENRICHED_OUT_PATH).loc[sample_idx]
    sample_meta = meta.loc[sample_idx]
    sample_preds = model.predict(X_sample)
    lines.append("## Per-class recall (stratified sample, up to 500/label)\n\n")
    lines.append("| label | n | recall | recall (prior: 5-min graph only) |\n|---|---|---|---|\n")
    for label in sorted(sample_meta["Label"].unique()):
        mask = (sample_meta["Label"] == label).values
        n = int(mask.sum())
        recall = float(sample_preds[mask].mean()) if label != "BENIGN" else float((sample_preds[mask] == 0).mean())
        prior = PRIOR_STEALTHY_RECALL.get(label)
        prior_str = f"{prior:.3f}" if prior is not None else "--"
        lines.append(f"| {label} | {n} | {recall:.3f} | {prior_str} |\n")
        print(f"{label}: n={n}, recall={recall:.3f}" + (f" (prior: {prior:.3f})" if prior is not None else ""))
    lines.append("\n")
    del X_sample
    gc.collect()

    # ---- direct check: the 9 known pivot flows ----
    pivot_row_mask = meta["_row_id"].isin(pivot_row_ids)
    X_pivots = load_feature_matrix(all_feature_cols, ENRICHED_OUT_PATH).loc[meta.index[pivot_row_mask]]
    pivot_preds = model.predict(X_pivots)
    pivot_proba = model.predict_proba(X_pivots)[:, 1]
    pivot_meta = meta[pivot_row_mask].copy()
    pivot_meta["flagged"] = pivot_preds.astype(bool)
    pivot_meta["probability"] = pivot_proba
    n_pivot_flagged = int(pivot_meta["flagged"].sum())
    assert len(pivot_meta) == len(pivot_row_ids), f"expected {len(pivot_row_ids)} pivot rows, matched {len(pivot_meta)}"

    lines.append("## Direct check: the 9 known lateral-movement pivot flows\n\n")
    lines.append(
        f"Hourly graph features caught 1/9. 5-min graph features alone caught 0/9. "
        f"This run (+ per-flow novelty) catches **{n_pivot_flagged} / 9**.\n\n"
    )
    lines.append(pivot_meta[["Timestamp", "Destination IP", "probability", "flagged"]].sort_values("Timestamp").to_markdown(index=False))
    lines.append("\n")
    print(f"\nPivot flows caught: {n_pivot_flagged} / 9 (prior runs: 1/9 hourly, 0/9 5-min-only)")

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nWrote {REPORT_OUT_PATH}")


if __name__ == "__main__":
    main()
