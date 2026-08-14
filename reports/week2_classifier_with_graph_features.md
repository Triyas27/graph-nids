# Week 2, Day 5 — classifier with graph features

This is the third attempt, after two real problems were found and fixed:

1. **Row-level random split leaked block identity.** Graph features are scoped to (Host, HourBucket), so many rows share an identical value; a naive random row split could put siblings from the same block on both sides of train/test, producing a suspicious 0.99999 F1 that didn't reflect real generalization. Fixed by splitting on WHOLE HOURS instead (**hour-grouped split**, below) -- no block can ever straddle train/test.
2. **Even with leakage fixed, graph features still collapsed both splits** (hour-grouped 0.913 -> 0.228, day-based 0.442 -> 0.005), both times via recall collapsing while precision stayed ~1.0 -- the model finding one narrow, memorized rule instead of a general one. Feature importance pinned this on `src_clustering_coefficient` (gain 70,461 -- 24x the next-highest feature): it's exactly 0.0 for the large majority of host-hours (needs 2+ mutually-connected neighbors, which most low-fan-out hosts never have), so any nonzero value is rare and easy to memorize rather than generalize from. **Excluded `src_/dst_clustering_coefficient`** for this run.

Two split protocols: **hour-grouped** (whole hours assigned to train/test) and **day-based** (train Mon-Wed, test Thu-Fri, unchanged from Week 1). Both are run flow-only and flow+graph for a direct before/after comparison.

### Hour-grouped split — flow-only
Train rows: 2,452,462, test rows: 378,281
Binary F1 (ATTACK class): 0.9126

```
              precision    recall  f1-score   support

      BENIGN      0.989     1.000     0.995    354074
      ATTACK      0.996     0.842     0.913     24207

    accuracy                          0.990    378281
   macro avg      0.993     0.921     0.954    378281
weighted avg      0.990     0.990     0.989    378281

```

### Day-based split — flow-only
Train rows: 1,668,530, test rows: 1,162,213
Binary F1 (ATTACK class): 0.4417

```
              precision    recall  f1-score   support

      BENIGN      0.807     1.000     0.893    871074
      ATTACK      0.998     0.284     0.442    291139

    accuracy                          0.820   1162213
   macro avg      0.902     0.642     0.667   1162213
weighted avg      0.855     0.820     0.780   1162213

```

### Hour-grouped split — flow + graph features
Train rows: 2,452,462, test rows: 378,281
Binary F1 (ATTACK class): 0.8844  (flow-only, same split: 0.9126, delta: -0.0281)

```
              precision    recall  f1-score   support

      BENIGN      0.986     1.000     0.993    354074
      ATTACK      1.000     0.793     0.884     24207

    accuracy                          0.987    378281
   macro avg      0.993     0.896     0.939    378281
weighted avg      0.987     0.987     0.986    378281

```

### Day-based split — flow + graph features
Train rows: 1,668,530, test rows: 1,162,213
Binary F1 (ATTACK class): 0.6247  (flow-only, same split: 0.4417, delta: +0.1829)

```
              precision    recall  f1-score   support

      BENIGN      0.847     0.992     0.914    871074
      ATTACK      0.949     0.465     0.625    291139

    accuracy                          0.860   1162213
   macro avg      0.898     0.729     0.769   1162213
weighted avg      0.873     0.860     0.841   1162213

```

### Diagnosis: feature importance, day-split flow+graph model

12/20 of the top-gain features are graph features (src_fan_out, src_fan_in, src_pagerank, ...).

| feature | gain |
|---|---|
| src_flow_volume | 45737.7 |
| Destination Port | 2024.1 |
| dst_novelty_score | 1203.9 |
| src_fan_out | 897.2 |
| dst_novelty_zscore | 250.3 |
| src_degree_ratio_change | 26.1 |
| src_fan_in | 7.5 |
| dst_byte_volume | 7.0 |
| Bwd Packet Length Mean | 6.5 |
| dst_pagerank | 5.0 |
| min_seg_size_forward | 3.9 |
| dst_fan_out_zscore | 3.1 |
| Fwd IAT Max | 3.0 |
| Bwd Packet Length Max | 2.9 |
| Flow IAT Min | 2.7 |
| src_flow_volume_zscore | 2.6 |
| dst_flow_volume_zscore | 2.5 |
| dst_byte_volume_zscore | 2.2 |
| Flow IAT Mean | 1.9 |
| Min Packet Length | 1.9 |

