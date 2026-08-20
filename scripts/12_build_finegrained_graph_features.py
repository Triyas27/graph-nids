"""Week 3 improvement: rebuild the host graph and its features at 5-minute
resolution instead of hourly, to close the loop Week 2/3 identified but
never fixed in the actual classifier.

Why: Week 2's SHAP analysis + Week 3's flow-level chain extraction both
concluded the same thing -- hourly aggregate features dilute the signal
for individually malicious flows, because every flow in a host's hour
shares one feature value (36 real Infiltration flows drowned inside
~68,800 simultaneous benign ones). We PROVED the signal survives at
finer resolution (Week 3 found the 9-host pivot chain cleanly at flow
level), but never rebuilt the actual features the classifier trains on
at that resolution. This does that.

Reuses build_hourly_graphs and compute_all_features from scripts 05/06
unchanged -- both are already bucket-size-agnostic (they just group by
whatever's in the "HourBucket" column), so the only change needed is
flooring timestamps to 5 minutes instead of 1 hour before calling them.
The column is still named "HourBucket" for compatibility with that reused
code, even though it now holds 5-minute buckets -- noted here rather than
silently confusing, not renamed everywhere to avoid touching already-
tested code for a cosmetic reason.

Output: data/processed/finegrained_host_features.parquet, same schema as
Week 2's host_hour_features.parquet, just at finer time resolution.
"""
import importlib.util
import os
import pickle

import pandas as pd

SCRIPTS_DIR = os.path.dirname(__file__)
BUCKET_FREQ = "5min"


def _load_module(filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.rstrip(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_graphs_mod = _load_module("05_build_hourly_graphs.py")
_features_mod = _load_module("06_compute_graph_features.py")
parse_timestamp = _graphs_mod.parse_timestamp
build_hourly_graphs = _graphs_mod.build_hourly_graphs
compute_all_features = _features_mod.compute_all_features

FLOWS_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_clean.parquet")
GRAPHS_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "finegrained_graphs.pkl")
FEATURES_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "finegrained_host_features.parquet")


def main():
    df = pd.read_parquet(FLOWS_PATH, columns=[
        "Source IP", "Destination IP", "Timestamp",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    ])
    df["ParsedTimestamp"] = parse_timestamp(df["Timestamp"])
    df["HourBucket"] = df["ParsedTimestamp"].dt.floor(BUCKET_FREQ)
    df["TotalBytes"] = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]

    graphs = build_hourly_graphs(df)
    print(f"Built {len(graphs)} graphs at {BUCKET_FREQ} resolution (vs. 50 at hourly resolution)")

    os.makedirs(os.path.dirname(GRAPHS_OUT_PATH), exist_ok=True)
    with open(GRAPHS_OUT_PATH, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Wrote {GRAPHS_OUT_PATH}")

    features = compute_all_features(graphs)
    features.to_parquet(FEATURES_OUT_PATH, index=False)
    print(f"Wrote {FEATURES_OUT_PATH} ({len(features):,} host-bucket rows, "
          f"vs. 93,924 at hourly resolution)")


if __name__ == "__main__":
    main()
