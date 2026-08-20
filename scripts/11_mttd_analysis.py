"""Week 3, Day 5: Mean Time to Detection (MTTD) for the infiltration chain.

Not F1, not AUC -- the metric a SOC actually watches: given the attack
timeline reconstructed in Day 1-2, how long from compromise until the
model raises ANY alert, how long until it flags something that's actually
part of the lateral-movement chain, and how many of the 9 internal pivot
targets are already reached by the time that happens.

Uses data/processed/lateral_movement_chain.parquet (scripts/10) directly
-- no retraining, no new model, just timing analysis over the already-
scored timeline.
"""
import os

import pandas as pd

CHAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "lateral_movement_chain.parquet")
REPORT_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week3_mttd_analysis.md")


def first_detection_time(timeline: pd.DataFrame, mask: pd.Series = None):
    """Earliest ParsedTimestamp among rows where model_flagged_attack is
    True, optionally restricted to rows where `mask` is also True. None if
    no such row exists (uses .min(), not row order, so it doesn't assume
    the input is pre-sorted)."""
    subset = timeline if mask is None else timeline[mask]
    flagged = subset[subset["model_flagged_attack"]]
    if flagged.empty:
        return None
    return flagged["ParsedTimestamp"].min()


def hosts_reached_by(pivot_summary: pd.DataFrame, cutoff_time) -> int:
    """Count of pivot targets whose first_contact is at or before
    cutoff_time. None cutoff_time (never detected) means 0 -- can't have
    been "reached before" a detection that never happens."""
    if cutoff_time is None:
        return 0
    return int((pivot_summary["first_contact"] <= cutoff_time).sum())


def main():
    timeline = pd.read_parquet(CHAIN_PATH)
    compromise_time = timeline["ParsedTimestamp"].min()

    pivot_mask = timeline["is_novel_destination"] & timeline["is_internal_destination"]
    pivot_summary = (
        timeline[pivot_mask]
        .groupby("Destination IP")
        .agg(first_contact=("ParsedTimestamp", "min"))
        .sort_values("first_contact")
    )
    n_pivots = len(pivot_summary)

    mttd_any = first_detection_time(timeline)
    mttd_pivot = first_detection_time(timeline, pivot_mask)

    hosts_reached_before_any = hosts_reached_by(pivot_summary, mttd_any)
    hosts_reached_before_pivot_detection = hosts_reached_by(pivot_summary, mttd_pivot)

    print(f"Compromise start: {compromise_time}")
    print(f"First detection (ANY flagged flow): {mttd_any} "
          f"({'N/A -- never detected' if mttd_any is None else mttd_any - compromise_time})")
    print(f"  hosts already pivoted-to by then: {hosts_reached_before_any} / {n_pivots}")
    print(f"First detection OF A PIVOT flow specifically: {mttd_pivot} "
          f"({'N/A -- no pivot ever detected' if mttd_pivot is None else mttd_pivot - compromise_time})")
    print(f"  hosts already pivoted-to by then: {hosts_reached_before_pivot_detection} / {n_pivots}")

    lines = ["# Week 3, Day 5 -- Mean Time to Detection (MTTD)\n\n"]
    lines.append(
        "Not F1, not AUC -- the metric a SOC actually watches: how long from compromise until "
        "the model raises an alert, and how much damage is already done by then.\n\n"
    )
    lines.append(f"**Compromise start**: {compromise_time}\n\n")

    lines.append("## Detection timing\n\n")
    lines.append("| | time | latency from compromise | pivot targets already reached |\n|---|---|---|---|\n")
    any_latency = "N/A" if mttd_any is None else str(mttd_any - compromise_time)
    pivot_latency = "N/A" if mttd_pivot is None else str(mttd_pivot - compromise_time)
    lines.append(f"| First alert (any flow) | {mttd_any} | {any_latency} | {hosts_reached_before_any} / {n_pivots} |\n")
    lines.append(f"| First alert ON the pivot chain | {mttd_pivot} | {pivot_latency} | {hosts_reached_before_pivot_detection} / {n_pivots} |\n\n")

    if mttd_any is not None and mttd_pivot is not None and mttd_any < mttd_pivot:
        lines.append(
            f"**The model does alert earlier than it \"should\" get credit for** -- its first alert "
            f"on this host's post-compromise traffic comes at {mttd_any} ({mttd_any - compromise_time} "
            f"after compromise), {mttd_pivot - mttd_any} before it flags anything actually on the pivot "
            f"chain. That earlier alert isn't on a real lateral-movement flow (likely a false positive in "
            f"the large burst of incidental external traffic this host generates), but in a real SOC it "
            f"would still be the thing that puts an analyst's eyes on this host in the first place -- "
            f"worth noting as a practical mitigating factor even though it doesn't count as a correct "
            f"detection in the F1/recall numbers.\n\n"
        )

    lines.append(
        f"**The core result**: by the time the model flags anything actually tied to the lateral-movement "
        f"chain, **{hosts_reached_before_pivot_detection} of the {n_pivots} internal pivot targets have "
        f"already been reached** -- detection happens at (or after) the compromise is essentially complete, "
        f"not while it's in progress. This is the practical cost of the dilution problem diagnosed in Week 2: "
        f"a detector built on hourly aggregates can't intervene mid-chain because it isn't resolving individual "
        f"flows at all.\n\n"
    )

    lines.append("## Pivot chain reference\n\n")
    lines.append(pivot_summary.reset_index().to_markdown(index=False))
    lines.append("\n")

    os.makedirs(os.path.dirname(REPORT_OUT_PATH), exist_ok=True)
    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nWrote {REPORT_OUT_PATH}")


if __name__ == "__main__":
    main()
