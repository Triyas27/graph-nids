"""Week 1, Day 6-7: unsupervised anomaly baseline (Isolation Forest) vs the
supervised XGBoost classifier, in both its "known attack" and "zero-day"
regimes.

Isolation Forest is trained on BENIGN traffic only — no attack labels are
used anywhere in its training. This is the "no labels, no prior knowledge"
end of the spectrum, contrasted with:
  - "known" supervised catch rate: XGBoost binary classifier, attack class
    WAS present in training (standard random split).
  - "zero-day" supervised catch rate: XGBoost binary classifier, attack
    class was NOT present in training (from Day 4-5, reports/week1_zeroday_test.md).

sklearn's IsolationForest can't handle NaN, so numeric features are median-
imputed using statistics fit on the BENIGN training split only (no leakage
from attacks or from the holdout).
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "flows_clean.parquet")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "reports", "week1_isolation_forest.md")

DROP_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "Day", "SourceFile", "Label"]

# From reports/week1_zeroday_test.md (Week 1, Day 4-5) — supervised binary
# XGBoost catch rate when the class was entirely absent from training.
ZERO_DAY_CATCH_RATE = {
    "Bot": 0.000, "Heartbleed": 0.000, "Infiltration": 0.000,
    "PortScan": 0.003, "SSH-Patator": 0.004, "FTP-Patator": 0.370,
    "DoS Slowhttptest": 0.374, "Web Attack - Sql Injection": 0.476,
    "DoS Hulk": 0.631, "DDoS": 0.631, "DoS GoldenEye": 0.670,
    "Web Attack - Brute Force": 0.836, "Web Attack - XSS": 0.965,
    "DoS slowloris": 0.996,
}

df = pd.read_parquet(DATA_PATH)
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols]
y_label = df["Label"]
benign_mask = y_label == "BENIGN"
attack_classes = sorted(y_label[~benign_mask].unique())

# ---- "known" supervised baseline: standard random split, all classes present ----
y_bin = (~benign_mask).astype(int)
Xtr, Xte, ytr, yte, label_tr, label_te = train_test_split(
    X, y_bin, y_label, test_size=0.2, stratify=y_label, random_state=42
)
clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    eval_metric="logloss", n_jobs=-1, random_state=42,
)
clf.fit(Xtr, ytr)
preds_known = pd.Series(clf.predict(Xte), index=Xte.index)

known_catch_rate = {}
for cls in attack_classes:
    mask = label_te == cls
    known_catch_rate[cls] = float(preds_known[mask].mean()) if mask.sum() else float("nan")
known_benign_fpr = float(preds_known[label_te == "BENIGN"].mean())
print(f"Known-attack supervised model: benign FPR = {known_benign_fpr:.4f}")

# ---- Isolation Forest: trained on BENIGN only, no attack labels used ----
benign_idx_train, benign_idx_holdout = train_test_split(
    df.index[benign_mask], test_size=0.2, random_state=42
)
imputer = SimpleImputer(strategy="median")
X_benign_train = imputer.fit_transform(df.loc[benign_idx_train, feature_cols])

iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
iso.fit(X_benign_train)

X_benign_holdout = imputer.transform(df.loc[benign_idx_holdout, feature_cols])
iso_benign_fpr = float((iso.predict(X_benign_holdout) == -1).mean())
print(f"Isolation Forest: benign FPR = {iso_benign_fpr:.4f}")

iso_catch_rate = {}
for cls in attack_classes:
    cls_mask = y_label == cls
    X_cls = imputer.transform(df.loc[cls_mask, feature_cols])
    iso_catch_rate[cls] = float((iso.predict(X_cls) == -1).mean())
    print(f"{cls}: IF catch_rate={iso_catch_rate[cls]:.3f}")

# ---- combine into one comparison table ----
rows = []
for cls in attack_classes:
    rows.append({
        "class": cls,
        "support": int((y_label == cls).sum()),
        "known_supervised_catch_rate": known_catch_rate[cls],
        "zero_day_supervised_catch_rate": ZERO_DAY_CATCH_RATE.get(cls, float("nan")),
        "isolation_forest_catch_rate": iso_catch_rate[cls],
    })
results_df = pd.DataFrame(rows).sort_values("zero_day_supervised_catch_rate")

lines = ["# Week 1, Day 6-7 — Isolation Forest anomaly baseline\n\n"]
lines.append(
    "Isolation Forest is trained only on BENIGN traffic (80% of BENIGN rows; "
    "20% held out for the false-positive rate below) — it never sees an attack "
    "label. Compared against the supervised XGBoost binary classifier in two "
    "regimes: `known` (the attack class was present during training, standard "
    "random split) and `zero_day` (the attack class was entirely absent from "
    "training — see Week 1 Day 4-5, reports/week1_zeroday_test.md).\n\n"
)
lines.append(f"- Supervised model (known-attack regime) false-positive rate on BENIGN test: {known_benign_fpr:.4f}\n")
lines.append(f"- Isolation Forest false-positive rate on BENIGN holdout: {iso_benign_fpr:.4f}\n\n")
lines.append(results_df.to_markdown(index=False, floatfmt=".3f"))
lines.append("\n")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.writelines(lines)
print(f"\nWrote {OUT_PATH}")
