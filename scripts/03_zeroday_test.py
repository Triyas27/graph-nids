"""Week 1, Day 4-5: controlled zero-day test.

For each attack class in turn, remove every row of that class from training
entirely (all other attack classes + BENIGN remain), train a binary
BENIGN-vs-ATTACK XGBoost classifier, then measure what fraction of the
held-out class's rows get flagged as ATTACK at test time ("zero-day catch
rate"). A fixed 10% BENIGN holdout, never used in training for any fold, is
used to measure the false-positive rate consistently across folds.

This generalizes the plan's "train on 5 classes, test on the 6th" idea to
every attack class, since the day-based split in Week 1 Day 3 already showed
(by accident) that most attack families get near-zero recall when unseen —
this isolates that effect one class at a time instead of 7 at once.
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flows_clean.parquet")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week1_zeroday_test.md")

DROP_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "Day", "SourceFile", "Label"]

df = pd.read_parquet(DATA_PATH)
feature_cols = [c for c in df.columns if c not in DROP_COLS]

benign_mask = df["Label"] == "BENIGN"
attack_classes = sorted(df.loc[~benign_mask, "Label"].unique())

benign_idx_train, benign_idx_holdout = train_test_split(
    df.index[benign_mask], test_size=0.10, random_state=42
)
X_benign_holdout = df.loc[benign_idx_holdout, feature_cols]

results = []
for cls in attack_classes:
    cls_mask = df["Label"] == cls
    other_attack_mask = (~benign_mask) & (~cls_mask)

    train_idx = np.concatenate([benign_idx_train.values, df.index[other_attack_mask].values])
    X_train = df.loc[train_idx, feature_cols]
    y_train = (df.loc[train_idx, "Label"] != "BENIGN").astype(int)

    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    X_held = df.loc[cls_mask, feature_cols]
    catch_rate = float(model.predict(X_held).mean())  # recall on the fully novel class
    fpr = float(model.predict(X_benign_holdout).mean())  # false-positive rate, fixed holdout

    n_support = int(cls_mask.sum())
    results.append({
        "class": cls, "support": n_support,
        "zero_day_catch_rate": catch_rate, "benign_fpr": fpr,
    })
    print(f"{cls} (n={n_support}): catch_rate={catch_rate:.3f}, benign_fpr={fpr:.4f}")

results_df = pd.DataFrame(results).sort_values("zero_day_catch_rate")

lines = ["# Week 1, Day 4-5 — controlled zero-day test\n\n"]
lines.append(
    "For each attack class, every row of that class is removed from training "
    "(all other attack classes + BENIGN remain). A fresh binary BENIGN-vs-ATTACK "
    "model is trained, then evaluated on the fully held-out class. "
    "`zero_day_catch_rate` = fraction of that class's flows flagged as ATTACK "
    "despite the model never having seen that attack type. `benign_fpr` is the "
    "false-positive rate on a fixed 10% BENIGN holdout never used in any fold's "
    "training, so it's comparable across rows.\n\n"
)
lines.append(results_df.to_markdown(index=False, floatfmt=".3f"))
lines.append("\n")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)
print(f"\nWrote {OUT_PATH}")
