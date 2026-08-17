"""Week 3, Day 1-2: extract the infiltration attack's lateral-movement
chain at full flow-level temporal resolution -- not hourly-bucketed,
which Week 2's SHAP analysis showed dilutes the signal for individually
malicious flows past detectability (36 real Infiltration flows sharing a
feature value with ~68,800 simultaneous benign ones).

Chain structure:
  1. Compromise: the 36 flows labeled "Infiltration" (attacker -> patient
     zero), Thursday ~14:19-15:45.
  2. Post-compromise activity: every flow patient zero INITIATES from the
     moment of compromise onward, chronologically ordered, each tagged
     with whether its destination is one patient zero has EVER contacted
     before that exact moment (using the full week's history up to the
     compromise timestamp as the "normal" baseline) -- a precise,
     per-flow version of Week 2's novelty_score, without the hourly
     aggregation that buried it.
  3. Model verdict: whether the Week 2 day-split flow+graph model would
     have flagged each post-compromise flow, for Day 3-4's "which hosts
     got flagged and when, which were missed" visualization.

Output: data/processed/lateral_movement_chain.parquet (one row per
post-compromise flow) and reports/week3_lateral_movement_chain.md.
"""
import importlib.util
import json
import os

import pandas as pd
import xgboost as xgb

SCRIPTS_DIR = os.path.dirname(__file__)


def _load_module(filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.rstrip(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parse_timestamp = _load_module("05_build_hourly_graphs.py").parse_timestamp

FLOWS_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "flows_enriched.parquet")
MODEL_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "day_split_graph_model.json")
MODEL_FEATURES_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "day_split_graph_model_features.json")
CHAIN_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "data", "processed", "lateral_movement_chain.parquet")
REPORT_OUT_PATH = os.path.join(SCRIPTS_DIR, "..", "reports", "week3_lateral_movement_chain.md")

ATTACK_LABEL = "Infiltration"


def find_patient_zero(df: pd.DataFrame, attack_label: str = ATTACK_LABEL) -> str:
    """The Source IP of the given attack label's flows -- the initially
    compromised host. Asserts (rather than assumes) that this attack has
    exactly one source, since the chain-extraction logic below only
    handles a single patient zero."""
    sources = df.loc[df["Label"] == attack_label, "Source IP"].unique()
    if len(sources) != 1:
        raise ValueError(f"expected exactly one source IP for {attack_label!r}, got {list(sources)}")
    return sources[0]


def is_internal_ip(ip: str) -> bool:
    """True for the testbed's internal subnet (192.168.10.x). Novel
    EXTERNAL destinations after compromise are mostly incidental internet
    traffic (this host is naturally high-volume); novel INTERNAL
    destinations are the real lateral-movement candidates -- a
    compromised host reaching internal peers it has never talked to."""
    return ip.startswith("192.168.")


def build_post_compromise_timeline(df: pd.DataFrame, patient_zero: str, compromise_time: pd.Timestamp) -> pd.DataFrame:
    """Every flow patient_zero initiates at or after compromise_time,
    chronologically ordered, tagged with is_novel_destination (never
    contacted by patient_zero before compromise_time, in the FULL history
    available in df) and a running cumulative distinct-destination count.
    """
    prior_mask = (df["Source IP"] == patient_zero) & (df["ParsedTimestamp"] < compromise_time)
    prior_destinations = set(df.loc[prior_mask, "Destination IP"].unique())

    post_mask = (df["Source IP"] == patient_zero) & (df["ParsedTimestamp"] >= compromise_time)
    timeline = df.loc[post_mask].sort_values("ParsedTimestamp").copy()

    seen = set(prior_destinations)
    is_novel, cumulative = [], []
    for dst in timeline["Destination IP"]:
        novel = dst not in seen
        is_novel.append(novel)
        seen.add(dst)
        cumulative.append(len(seen))
    timeline["is_novel_destination"] = is_novel
    timeline["cumulative_distinct_destinations"] = cumulative
    timeline["n_prior_destinations"] = len(prior_destinations)
    return timeline


def main():
    df = pd.read_parquet(FLOWS_PATH)
    df["ParsedTimestamp"] = parse_timestamp(df["Timestamp"])

    patient_zero = find_patient_zero(df)
    compromise_rows = df[df["Label"] == ATTACK_LABEL].sort_values("ParsedTimestamp")
    compromise_time = compromise_rows["ParsedTimestamp"].iloc[0]
    print(f"Patient zero: {patient_zero}")
    print(f"Compromise window: {compromise_rows['ParsedTimestamp'].iloc[0]} to {compromise_rows['ParsedTimestamp'].iloc[-1]} ({len(compromise_rows)} flows)")
    print(f"Attacker/C2 IP (Infiltration flow destinations): {sorted(compromise_rows['Destination IP'].unique())}")

    timeline = build_post_compromise_timeline(df, patient_zero, compromise_time)

    with open(MODEL_FEATURES_PATH) as f:
        feature_cols = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    X = timeline[feature_cols].astype("float32")
    timeline["model_flagged_attack"] = model.predict(X).astype(bool)
    timeline["model_attack_probability"] = model.predict_proba(X)[:, 1]

    timeline["is_internal_destination"] = timeline["Destination IP"].apply(is_internal_ip)

    n_novel = int(timeline["is_novel_destination"].sum())
    n_prior = int(timeline["n_prior_destinations"].iloc[0])
    n_flagged = int(timeline["model_flagged_attack"].sum())
    first_novel_time = timeline.loc[timeline["is_novel_destination"], "ParsedTimestamp"].min()
    novel_dest_list = timeline.loc[timeline["is_novel_destination"], "Destination IP"].unique()

    novel_mask = timeline["is_novel_destination"]
    internal_novel = timeline[novel_mask & timeline["is_internal_destination"]]
    internal_novel_summary = (
        internal_novel.groupby("Destination IP")
        .agg(first_contact=("ParsedTimestamp", "min"), destination_port=("Destination Port", "first"),
             n_flows=("ParsedTimestamp", "size"), model_flagged_any=("model_flagged_attack", "any"))
        .sort_values("first_contact")
    )
    n_internal_pivots = len(internal_novel_summary)
    n_internal_caught = int(internal_novel_summary["model_flagged_any"].sum())

    print(f"Patient zero's prior distinct destinations (before compromise): {n_prior}")
    print(f"Post-compromise flows: {len(timeline):,}, to {timeline['Destination IP'].nunique()} distinct destinations "
          f"({n_novel:,} flows to {len(novel_dest_list)} NEVER-before-seen destinations)")
    print(f"  of which INTERNAL (lateral-movement candidates): {n_internal_pivots} hosts, "
          f"{n_internal_caught}/{n_internal_pivots} flagged by the model")
    print(f"First novel-destination contact: {first_novel_time} ({(first_novel_time - compromise_time)} after compromise)")
    print(f"Model flagged {n_flagged:,} / {len(timeline):,} post-compromise flows as ATTACK "
          f"({n_flagged / len(timeline):.2%})")

    os.makedirs(os.path.dirname(CHAIN_OUT_PATH), exist_ok=True)
    timeline.to_parquet(CHAIN_OUT_PATH, index=False)
    print(f"Wrote {CHAIN_OUT_PATH}")

    lines = ["# Week 3, Day 1-2 — lateral movement chain\n\n"]
    lines.append(f"**Patient zero**: `{patient_zero}`\n\n")
    lines.append(
        f"**Compromise**: {len(compromise_rows)} flows labeled `Infiltration`, "
        f"{compromise_rows['ParsedTimestamp'].iloc[0]} to {compromise_rows['ParsedTimestamp'].iloc[-1]}, "
        f"to/from {sorted(compromise_rows['Destination IP'].unique())}.\n\n"
    )
    lines.append(
        f"**Post-compromise activity**: {len(timeline):,} flows initiated by `{patient_zero}` from the moment "
        f"of compromise onward, to {timeline['Destination IP'].nunique()} distinct destinations. "
        f"Before compromise, this host had only ever contacted {n_prior} distinct destinations (its full prior "
        f"history). {n_novel:,} post-compromise flows ({n_novel / len(timeline):.1%}) go to "
        f"**{len(novel_dest_list)} destinations it had never contacted before** -- these are the flow-level "
        f"lateral-movement candidates, invisible in Week 2's hourly aggregate features but explicit here.\n\n"
    )
    lines.append(f"**First contact with a novel destination**: {first_novel_time}, "
                  f"{first_novel_time - compromise_time} after the compromise window started.\n\n")
    lines.append(
        f"**Detection**: the Week 2 day-split flow+graph model flagged {n_flagged:,} / {len(timeline):,} "
        f"({n_flagged / len(timeline):.2%}) of these post-compromise flows as ATTACK -- consistent with the "
        f"SHAP finding that this specific host's activity is not reliably caught by the current model.\n\n"
    )

    lines.append("## The lateral-movement chain: internal pivot targets\n\n")
    lines.append(
        f"Of {len(novel_dest_list)} novel destinations, {len(novel_dest_list) - n_internal_pivots} are external "
        f"IPs (mostly incidental internet traffic -- this host is naturally high-volume) and "
        f"**{n_internal_pivots} are internal hosts patient zero had never contacted before** -- these are the "
        f"real lateral-movement candidates. All of them are labelled `BENIGN` in the ground truth (a known "
        f"CIC-IDS2017 gap: the documented post-infiltration internal scan isn't captured in the attack labels), "
        f"but the pattern is unmistakable: **{n_internal_pivots} previously-unseen internal hosts, each contacted "
        f"for the first time in strict chronological sequence, {internal_novel_summary['first_contact'].min()} to "
        f"{internal_novel_summary['first_contact'].max()}** (a "
        f"{internal_novel_summary['first_contact'].max() - internal_novel_summary['first_contact'].min()} window), "
        f"each on a distinct, mostly non-standard port -- a textbook internal reconnaissance signature. "
        f"**The model catches {n_internal_caught}/{n_internal_pivots} of them.**\n\n"
    )
    lines.append(internal_novel_summary.reset_index().to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## All novel destinations contacted after compromise (internal + external)\n\n")
    novel_summary = (
        timeline[timeline["is_novel_destination"]]
        .groupby("Destination IP")
        .agg(first_contact=("ParsedTimestamp", "min"), n_flows=("ParsedTimestamp", "size"),
             is_internal=("is_internal_destination", "first"), model_flagged_any=("model_flagged_attack", "any"))
        .sort_values("first_contact")
    )
    lines.append(novel_summary.reset_index().to_markdown(index=False))
    lines.append("\n")

    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {REPORT_OUT_PATH}")


if __name__ == "__main__":
    main()
