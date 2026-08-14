"""Week 2, Day 5 (part 1): attach each flow's Source IP and Destination IP
graph features (from Day 3-4) to the flow itself, producing a single
enriched table for the classifier.

Each flow gets 13 graph features from its Source IP's host-hour row
(prefixed src_) and the same 13 from its Destination IP's host-hour row
(prefixed dst_) -- both sides, since a flow's "attacker-like" signal can
show up on either end (e.g. src_flow_volume_zscore for a host suddenly
hammering known peers, or dst_fan_in_zscore for a host suddenly receiving
attention from many new sources).

Honest limitation, not fixed here: the graph features for hour H are
FULL-HOUR aggregates (e.g. fan_out counts every destination that host
contacted anywhere in that hour), so attaching them to an individual flow
inside that same hour gives the model a small amount of same-hour
hindsight it wouldn't have in a real streaming deployment (a flow at
14:05 gets a fan_out feature that also reflects what the host does at
14:55). This matches the plan's literal design (hourly graphs, features
attached per host per hour) and doesn't touch attack labels at all, but
it's not a strictly causal per-flow feature -- worth stating plainly
rather than glossing over.
"""
import importlib.util
import os

import pandas as pd

SCRIPTS_DIR = os.path.dirname(__file__)


def _load_module(filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.rstrip(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parse_timestamp = _load_module("05_build_hourly_graphs.py").parse_timestamp

FLOWS_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_clean.parquet")
FEATURES_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "host_hour_features.parquet")
OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_enriched.parquet")

GRAPH_FEATURE_COLS = [
    "fan_out", "fan_in", "pagerank", "clustering_coefficient", "degree_ratio_change",
    "novelty_score", "flow_volume", "byte_volume", "fan_out_zscore", "fan_in_zscore",
    "novelty_zscore", "flow_volume_zscore", "byte_volume_zscore",
]


def enrich_flows_with_graph_features(flows: pd.DataFrame, host_hour_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join each flow to its Source IP's and Destination IP's
    host-hour graph features for the flow's own HourBucket. `flows` must
    already have a HourBucket column built the same way (parse_timestamp +
    floor to hour) as was used to build host_hour_features, or the join
    keys won't line up and every graph feature will come back NaN."""
    src = host_hour_features.rename(
        columns={"Host": "Source IP", **{c: f"src_{c}" for c in GRAPH_FEATURE_COLS}}
    )
    dst = host_hour_features.rename(
        columns={"Host": "Destination IP", **{c: f"dst_{c}" for c in GRAPH_FEATURE_COLS}}
    )
    merged = flows.merge(src, on=["HourBucket", "Source IP"], how="left")
    merged = merged.merge(dst, on=["HourBucket", "Destination IP"], how="left")
    return merged


def main():
    flows = pd.read_parquet(FLOWS_PATH)
    flows["HourBucket"] = parse_timestamp(flows["Timestamp"]).dt.floor("h")

    host_hour_features = pd.read_parquet(FEATURES_PATH)

    enriched = enrich_flows_with_graph_features(flows, host_hour_features)

    n_rows_before = len(flows)
    assert len(enriched) == n_rows_before, "join duplicated or dropped rows"
    for prefix in ("src_", "dst_"):
        missing = enriched[f"{prefix}fan_out"].isna().sum()
        print(f"{prefix}fan_out missing after join: {missing:,} / {len(enriched):,}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    enriched.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(enriched):,} rows, {enriched.shape[1]} columns)")


if __name__ == "__main__":
    main()
