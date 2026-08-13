"""Week 2, Day 1-2: build a directed host-connectivity graph per hour.

Nodes are hosts (IPs), edges are Source IP -> Destination IP, weighted by
flow count and total bytes exchanged within that hour. Graphs are saved to
data/processed/hourly_graphs.pkl (gitignored, regenerable) for Week 2
Day 3-4's graph feature computation to consume.

Timestamp parsing note (see reports/week2_hourly_graphs.md): every raw
Timestamp string in this dataset is a 12-hour clock with NO am/pm marker
(e.g. "7/7/2017 3:30", "03/07/2017 08:55:58"). Verified against the paper's
documented Heartbleed window (15:12-15:32 Wednesday): the raw rows show
hour token "3" (3:12-3:32), confirming captures run a single business day
(~8am-5pm) where hour tokens 8-12 are unambiguously AM/noon and 1-7 are
unambiguously PM.
"""
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flows_clean.parquet")
GRAPH_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "hourly_graphs.pkl")
REPORT_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week2_hourly_graphs.md")


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse CIC-IDS2017's ambiguous 12-hour-no-am/pm Timestamp strings
    into real datetimes. Hour tokens 8-12 -> AM/noon (unchanged); hour
    tokens 1-7 -> PM (+12). See module docstring for the validation this
    rule was checked against."""
    parts = series.str.split(" ", expand=True)
    date_part, time_part = parts[0], parts[1]

    date_split = date_part.str.split("/", expand=True).astype(int)
    day, month, year = date_split[0], date_split[1], date_split[2]

    time_split = time_part.str.split(":", expand=True)
    hour = time_split[0].astype(int)
    minute = time_split[1].astype(int)
    if time_split.shape[1] >= 3:
        second = time_split[2].fillna(0).astype(int)
    else:
        second = pd.Series(0, index=series.index)

    real_hour = np.where(hour == 12, 12, np.where(hour <= 7, hour + 12, hour))

    return pd.to_datetime(pd.DataFrame({
        "year": year, "month": month, "day": day,
        "hour": real_hour, "minute": minute, "second": second,
    }))


def build_hourly_graphs(df: pd.DataFrame) -> dict:
    """Group flows by hour bucket and build one weighted DiGraph per hour.
    Expects columns: Source IP, Destination IP, HourBucket, TotalBytes."""
    graphs = {}
    for bucket, group in df.groupby("HourBucket"):
        edge_agg = group.groupby(["Source IP", "Destination IP"]).agg(
            flow_count=("HourBucket", "size"),
            total_bytes=("TotalBytes", "sum"),
        )
        G = nx.DiGraph()
        for (src, dst), row in edge_agg.iterrows():
            G.add_edge(src, dst, flow_count=int(row["flow_count"]), total_bytes=float(row["total_bytes"]))
        graphs[bucket] = G
    return graphs


def main():
    df = pd.read_parquet(DATA_PATH, columns=[
        "Source IP", "Destination IP", "Timestamp", "Day", "SourceFile",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    ])
    df["ParsedTimestamp"] = parse_timestamp(df["Timestamp"])
    df["HourBucket"] = df["ParsedTimestamp"].dt.floor("h")
    df["TotalBytes"] = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]

    graphs = build_hourly_graphs(df)

    lines = ["# Week 2, Day 1-2 — hourly host-connectivity graphs\n\n"]
    lines.append(
        "One directed graph per hour: nodes are hosts (IPs), edges are "
        "Source IP -> Destination IP weighted by `flow_count` and "
        "`total_bytes` within that hour.\n\n"
    )
    lines.append(f"Total hourly buckets: {len(graphs)}\n\n")
    lines.append("| hour | day | nodes | edges | total flows |\n|---|---|---|---|---|\n")

    day_lookup = df.drop_duplicates("HourBucket").set_index("HourBucket")["Day"]
    for bucket in sorted(graphs):
        G = graphs[bucket]
        total_flows = sum(d["flow_count"] for _, _, d in G.edges(data=True))
        lines.append(f"| {bucket} | {day_lookup.loc[bucket]} | {G.number_of_nodes():,} | {G.number_of_edges():,} | {total_flows:,} |\n")

    os.makedirs(os.path.dirname(GRAPH_OUT_PATH), exist_ok=True)
    with open(GRAPH_OUT_PATH, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Wrote {GRAPH_OUT_PATH} ({len(graphs)} hourly graphs)")

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {REPORT_OUT_PATH}")


if __name__ == "__main__":
    main()
