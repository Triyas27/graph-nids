"""Week 1, Day 1-3: flow-level XGBoost baseline.

Two evaluation regimes, binary and multi-class each:
  - Random split: stratified 80/20, ignoring which day a flow came from.
    This is the "easy", optimistic number most CIC-IDS2017 notebooks report.
  - Day-based split: train on Monday-Wednesday, test on Thursday-Friday.
    Mimics real deployment (train on past traffic, deploy on future).

XGBoost handles NaN natively (missing-value split direction is learned), so
no imputation is done here — the NaNs left by 01_prepare_data.py (Inf ->
NaN in the rate columns) are passed straight through.
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flows_clean.parquet")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week1_baseline_classifier.md")

DROP_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "Day", "SourceFile", "Label"]
TRAIN_DAYS = {"Monday", "Tuesday", "Wednesday"}
TEST_DAYS = {"Thursday", "Friday"}

df = pd.read_parquet(DATA_PATH)
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols]
y_binary = (df["Label"] != "BENIGN").astype(int)
y_multi_raw = df["Label"]

lines = ["# Week 1 — baseline flow classifier (XGBoost)\n"]


def fit_eval_binary(X_train, y_train, X_test, y_test, section_title):
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
    lines.append(f"Binary F1 (ATTACK class): {f1:.4f}\n\n")
    lines.append("```\n" + report + "\n```\n\n")
    print(section_title, "F1:", f1)
    return model


def fit_eval_multi(X_train, y_train_raw, X_test, y_test_raw, section_title):
    le = LabelEncoder()
    le.fit(y_train_raw)  # classes are defined by what TRAIN saw — realistic
    train_classes = set(le.classes_)
    test_classes = set(y_test_raw.unique())
    unseen_in_test = test_classes - train_classes

    y_train = le.transform(y_train_raw)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        eval_metric="mlogloss", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    # Any test label the model never saw in training cannot be predicted
    # correctly by construction; keep those rows in the report (as a
    # forced miss) rather than silently dropping them.
    known_mask = y_test_raw.isin(train_classes)
    preds_known = le.inverse_transform(model.predict(X_test[known_mask]))

    lines.append(f"### {section_title}\n")
    lines.append(f"Train rows: {len(X_train):,}, test rows: {len(X_test):,}\n")
    lines.append(f"Classes seen in training: {sorted(train_classes)}\n\n")
    if unseen_in_test:
        n_unseen_rows = int((~known_mask).sum())
        lines.append(
            f"**Classes present in test but never seen in training "
            f"(structurally undetectable, {n_unseen_rows:,} rows): "
            f"{sorted(unseen_in_test)}**\n\n"
        )
    report = classification_report(
        y_test_raw[known_mask], preds_known, digits=3, zero_division=0
    )
    lines.append("Per-class report (rows with a class unseen in training excluded — "
                  "see note above; those are 0% recall by construction):\n\n")
    lines.append("```\n" + report + "\n```\n\n")
    print(section_title, "done. Unseen-in-train test classes:", sorted(unseen_in_test))
    return model


# ---- Experiment A: random stratified 80/20 split ----
Xtr, Xte, ytr_bin, yte_bin = train_test_split(X, y_binary, test_size=0.2, stratify=y_binary, random_state=42)
fit_eval_binary(Xtr, ytr_bin, Xte, yte_bin, "Random split — binary (BENIGN vs ATTACK)")

Xtr, Xte, ytr_multi, yte_multi = train_test_split(X, y_multi_raw, test_size=0.2, stratify=y_multi_raw, random_state=42)
fit_eval_multi(Xtr, ytr_multi, Xte, yte_multi, "Random split — multi-class")

# ---- Experiment B: day-based split (train Mon-Wed, test Thu-Fri) ----
train_mask = df["Day"].isin(TRAIN_DAYS)
test_mask = df["Day"].isin(TEST_DAYS)
Xtr, Xte = X[train_mask], X[test_mask]
ytr_bin, yte_bin = y_binary[train_mask], y_binary[test_mask]
fit_eval_binary(Xtr, ytr_bin, Xte, yte_bin, "Day-based split (train Mon-Wed, test Thu-Fri) — binary")

ytr_multi, yte_multi = y_multi_raw[train_mask], y_multi_raw[test_mask]
fit_eval_multi(Xtr, ytr_multi, Xte, yte_multi, "Day-based split (train Mon-Wed, test Thu-Fri) — multi-class")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)
print(f"\nWrote {OUT_PATH}")
