"""Week 2, Day 3-4: per-host, per-hour graph features.

For every host present in each of the 50 hourly graphs from Day 1-2,
compute:
  - fan_out / fan_in: distinct destinations contacted / sources contacting
    this host in that hour (out-degree / in-degree).
  - pagerank: network importance within that hour's graph, weighted by
    flow_count.
  - clustering_coefficient: structural clustering (Fagiolo's directed
    generalization, via networkx), unweighted.
  - degree_ratio_change: this hour's total degree (in+out) divided by the
    SAME host's total degree in the previous hourly bucket. "Previous" is
    positional in the sorted list of 50 buckets, so it can cross a day
    boundary (Monday 17:00 -> Tuesday 08:00); this is a deliberate
    simplification, not a bug. None if the host wasn't present the
    previous hour (no baseline, not zero change).
  - novelty_score: fraction of this hour's destinations that the host has
    never contacted in any earlier hour (cumulative baseline, growing
    chronologically across the whole week). None for hosts with no
    outgoing edges this hour. A host's first-ever appearance scores 1.0
    for all its destinations, by definition (no prior baseline yet).

Baseline-relative (z-score) features, added after validating that raw
fan_out/novelty fail to flag a real host we checked by hand (192.168.10.8,
the Infiltration attack's compromised host): it's naturally a very
high-fan-out, high-novelty host every single hour, all week, including
before the attack -- so its absolute numbers during the attack don't look
unusual compared to OTHER hosts, only compared to ITS OWN history. Same
idea as an AML system flagging "this account is transacting more than its
own normal," not a flat threshold across all accounts:
  - fan_out_zscore / fan_in_zscore / novelty_zscore: (this hour's value -
    that host's own running mean over all STRICTLY EARLIER hours) /
    (that host's own running std + a small epsilon, to keep a tight-but-
    not-perfectly-constant baseline from producing an undefined/exploding
    ratio). None until a host has at least 2 prior hourly observations to
    build a baseline from (matches degree_ratio_change's "None with no
    baseline" convention).

Even with the z-score fix, 192.168.10.8 still didn't stand out during the
actual attack window -- because the real anomaly there isn't topological
(new neighbors) at all, it's volume (the SAME known neighbors, contacted
at a massively higher rate: 68,843 outgoing flows in one hour vs. a
1,000-8,000/hour norm, confirmed by dropping to raw 5-minute flow data).
None of the features above would ever catch that, since fan_out/novelty
only count distinct destinations, never how many times each was
contacted. Hence:
  - flow_volume / byte_volume: total outgoing flow count / total outgoing
    bytes for a host in that hour (summed across all its out-edges).
  - flow_volume_zscore / byte_volume_zscore: same running-baseline z-score
    treatment as above, applied to volume instead of topology.

Output: data/processed/host_hour_features.parquet, one row per
(HourBucket, Host).
"""
import os
import pickle

import networkx as nx
import pandas as pd


class RunningStats:
    """Welford's online mean/variance -- lets us maintain a running,
    causal (prior-hours-only) per-host baseline without recomputing from
    the full history each hour."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return None
        return (self._m2 / (self.n - 1)) ** 0.5


def compute_baseline_zscore(value: float, stats: "RunningStats", min_observations: int = 2, eps: float = 1e-6):
    """How many (regularized) standard deviations `value` is from this
    host's own running mean, using only observations strictly before now.
    None if there isn't yet enough history (matches degree_ratio_change's
    "no baseline" convention -- absence of a value, not a zero).

    `eps` must be chosen on the same natural scale as the feature: with a
    2-observation baseline it's common for both observations to be
    identical (std == 0 exactly), and a too-small eps (e.g. the default
    1e-6, appropriate for fractions) turns any later count-scale deviation
    into a meaningless multi-million z-score instead of a usable signal.
    Callers of count-like features (fan_out, fan_in) should pass eps=1.0
    (don't trust std below a single-destination granularity); callers of
    0-1 fraction features (novelty_score) should pass something like 0.05.
    """
    if stats.n < min_observations:
        return None
    return (value - stats.mean) / (stats.std + eps)

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "hourly_graphs.pkl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "host_hour_features.parquet")
REPORT_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week2_graph_features.md")


def total_degrees(G: nx.DiGraph) -> dict:
    return {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}


def total_out_flow_count(G: nx.DiGraph) -> dict:
    """Total outgoing flow VOLUME per host (sum of the flow_count edge
    weight across all of a host's out-edges) -- distinct from fan_out,
    which only counts distinct destinations, not how many times each was
    contacted. A host hammering the same few known peers far harder than
    usual changes this without moving fan_out at all."""
    totals = {n: 0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        totals[u] += data["flow_count"]
    return totals


def total_out_bytes(G: nx.DiGraph) -> dict:
    """Total outgoing bytes per host (sum of the total_bytes edge weight
    across all of a host's out-edges)."""
    totals = {n: 0.0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        totals[u] += data["total_bytes"]
    return totals


def compute_degree_ratio_change(prev_totals: dict, curr_totals: dict) -> dict:
    """curr_total_degree / prev_total_degree per host. None (no baseline)
    for hosts absent from prev_totals."""
    result = {}
    for host, curr_deg in curr_totals.items():
        prev_deg = prev_totals.get(host)
        result[host] = (curr_deg / prev_deg) if prev_deg else None
    return result


def compute_novelty_scores(G: nx.DiGraph, prior_destinations: dict) -> dict:
    """Fraction of each host's this-hour destinations not seen in any
    earlier hour. Uses prior_destinations as-is (read-only) -- does not
    fold in this hour's destinations; call update_prior_destinations for
    that, separately, so scoring always compares against a strictly-prior
    baseline."""
    scores = {}
    for host in G.nodes():
        curr_dests = set(G.successors(host))
        if not curr_dests:
            continue
        prior = prior_destinations.get(host, set())
        novel = curr_dests - prior
        scores[host] = len(novel) / len(curr_dests)
    return scores


def update_prior_destinations(G: nx.DiGraph, prior_destinations: dict) -> dict:
    """Return a NEW dict with this hour's destinations folded into each
    host's cumulative history. Does not mutate the input."""
    updated = {host: set(dests) for host, dests in prior_destinations.items()}
    for host in G.nodes():
        curr_dests = set(G.successors(host))
        if curr_dests:
            updated.setdefault(host, set()).update(curr_dests)
    return updated


def compute_all_features(graphs: dict) -> pd.DataFrame:
    rows = []
    prior_destinations = {}
    prev_totals = {}
    fan_out_stats, fan_in_stats, novelty_stats = {}, {}, {}
    flow_volume_stats, byte_volume_stats = {}, {}

    for hour in sorted(graphs):
        G = graphs[hour]
        fan_out = dict(G.out_degree())
        fan_in = dict(G.in_degree())
        totals = total_degrees(G)
        pagerank = nx.pagerank(G, weight="flow_count")
        clustering = nx.clustering(G)
        deg_ratio = compute_degree_ratio_change(prev_totals, totals)
        novelty = compute_novelty_scores(G, prior_destinations)
        flow_volume = total_out_flow_count(G)
        byte_volume = total_out_bytes(G)

        for host in G.nodes():
            fo, fi = fan_out.get(host, 0), fan_in.get(host, 0)
            nov = novelty.get(host)
            fv, bv = flow_volume.get(host, 0), byte_volume.get(host, 0.0)
            rows.append({
                "HourBucket": hour,
                "Host": host,
                "fan_out": fo,
                "fan_in": fi,
                "pagerank": pagerank.get(host),
                "clustering_coefficient": clustering.get(host),
                "degree_ratio_change": deg_ratio.get(host),
                "novelty_score": nov,
                "flow_volume": fv,
                "byte_volume": bv,
                "fan_out_zscore": compute_baseline_zscore(fo, fan_out_stats.get(host, RunningStats()), eps=1.0),
                "fan_in_zscore": compute_baseline_zscore(fi, fan_in_stats.get(host, RunningStats()), eps=1.0),
                "novelty_zscore": (
                    compute_baseline_zscore(nov, novelty_stats.get(host, RunningStats()), eps=0.05)
                    if nov is not None else None
                ),
                "flow_volume_zscore": compute_baseline_zscore(fv, flow_volume_stats.get(host, RunningStats()), eps=1.0),
                "byte_volume_zscore": compute_baseline_zscore(bv, byte_volume_stats.get(host, RunningStats()), eps=1.0),
            })

        # update baselines AFTER scoring this hour, so scoring only ever
        # sees strictly-prior history (same causal discipline as novelty_score)
        for host in G.nodes():
            fan_out_stats.setdefault(host, RunningStats()).update(fan_out.get(host, 0))
            fan_in_stats.setdefault(host, RunningStats()).update(fan_in.get(host, 0))
            flow_volume_stats.setdefault(host, RunningStats()).update(flow_volume.get(host, 0))
            byte_volume_stats.setdefault(host, RunningStats()).update(byte_volume.get(host, 0.0))
        for host, nov in novelty.items():
            novelty_stats.setdefault(host, RunningStats()).update(nov)

        prior_destinations = update_prior_destinations(G, prior_destinations)
        prev_totals = totals

    return pd.DataFrame(rows)


def main():
    with open(GRAPH_PATH, "rb") as f:
        graphs = pickle.load(f)

    features = compute_all_features(graphs)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    features.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(features):,} host-hour rows)")

    lines = ["# Week 2, Day 3-4 — per-host, per-hour graph features\n\n"]
    lines.append(f"Total host-hour rows: {len(features):,} across {features['HourBucket'].nunique()} hourly buckets, "
                  f"{features['Host'].nunique():,} distinct hosts.\n\n")
    lines.append("## Feature summary statistics\n\n")
    lines.append(features[["fan_out", "fan_in", "pagerank", "clustering_coefficient",
                            "degree_ratio_change", "novelty_score", "flow_volume", "byte_volume",
                            "fan_out_zscore", "fan_in_zscore", "novelty_zscore",
                            "flow_volume_zscore", "byte_volume_zscore"]].describe().to_markdown(floatfmt=".4f"))
    lines.append("\n\n## Top 10 fan-out host-hours (absolute)\n\n")
    lines.append(features.nlargest(10, "fan_out")[["HourBucket", "Host", "fan_out", "fan_in"]].to_markdown(index=False))
    lines.append("\n\n## Top 10 fan-out z-score host-hours (relative to each host's own baseline)\n\n")
    lines.append(
        features.nlargest(10, "fan_out_zscore")[["HourBucket", "Host", "fan_out", "fan_out_zscore"]]
        .to_markdown(index=False, floatfmt=".2f")
    )
    lines.append("\n\n## Top 10 flow-volume z-score host-hours (relative to each host's own baseline)\n\n")
    lines.append(
        features.nlargest(10, "flow_volume_zscore")[["HourBucket", "Host", "flow_volume", "flow_volume_zscore"]]
        .to_markdown(index=False, floatfmt=".2f")
    )
    lines.append("\n\n## Validation: the two hosts checked by hand against known attacks\n\n")
    lines.append(
        "`172.16.0.1` is the PortScan attacker (single-target multi-port scan against "
        "192.168.10.50 -- fan-out never rises, so no absolute OR baseline-relative signal "
        "is expected here; the graph is structurally blind to port-level scanning). "
        "`192.168.10.8` is the Infiltration attack's compromised host, active Thursday "
        "6/7 14:19-15:45. Its fan_out/novelty z-scores don't flag the attack window at all "
        "(checked and confirmed) -- the real anomaly there is volume, not topology: raw "
        "5-minute flow data shows 66,623 flows in the 15:05-15:45 window alone (vs. a "
        "1,000-8,000/hour norm), almost all repeat connections to the SAME already-known "
        "internal peers over SMB/NetBIOS/RPC ports (445/139/135). **Confirmed below**: "
        "`flow_volume_zscore` hits 20.68 at 2017-07-06 15:00 -- by far the largest value "
        "anywhere in this host's 50-hour timeline (everything else is -1.5 to +5.4) -- "
        "cleanly catching what fan_out/novelty structurally cannot, since they only count "
        "distinct destinations, never contact rate. (172.16.0.1's own volume spike during "
        "its PortScan hour is real but weaker, 2.65 -- it's CIC-IDS2017's shared attacker "
        "machine, reused across multiple days' attacks, so its own baseline is already "
        "noisy/elevated throughout the week.)\n\n"
    )
    for host in ["172.16.0.1", "192.168.10.8"]:
        sub = features[features["Host"] == host].sort_values("HourBucket")
        lines.append(f"### {host}\n\n")
        lines.append(
            sub[["HourBucket", "fan_out", "fan_out_zscore", "novelty_score", "novelty_zscore",
                 "flow_volume", "flow_volume_zscore"]]
            .to_markdown(index=False, floatfmt=".2f")
        )
        lines.append("\n\n")

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {REPORT_OUT_PATH}")


if __name__ == "__main__":
    main()
