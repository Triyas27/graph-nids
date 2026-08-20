# Week 3 improvement, take 2 -- per-flow novelty feature (no bucketing)

### Day-based split -- flow + 5-min graph + per-flow novelty
Train rows: 1,668,530, test rows: 1,162,213
Binary F1 (ATTACK class): 0.8122  (prior run, 5-min graph only: 0.8122, delta: -0.0000)

```
              precision    recall  f1-score   support

      BENIGN      0.904     1.000     0.950    871074
      ATTACK      0.999     0.684     0.812    291139

    accuracy                          0.921   1162213
   macro avg      0.952     0.842     0.881   1162213
weighted avg      0.928     0.921     0.915   1162213

```

Comparison points: flow-only = 0.4417, hourly graph = 0.6247, 5-min graph = 0.8122, this run = 0.8122.

## Per-class recall (stratified sample, up to 500/label)

| label | n | recall | recall (prior: 5-min graph only) |
|---|---|---|---|
| BENIGN | 500 | 1.000 | -- |
| Bot | 500 | 0.000 | 0.000 |
| DDoS | 500 | 0.936 | -- |
| Infiltration | 36 | 0.000 | 0.000 |
| PortScan | 500 | 0.500 | 0.048 |
| Web Attack - Brute Force | 500 | 1.000 | -- |
| Web Attack - Sql Injection | 21 | 1.000 | -- |
| Web Attack - XSS | 500 | 1.000 | -- |

## Direct check: the 9 known lateral-movement pivot flows

Hourly graph features caught 1/9. 5-min graph features alone caught 0/9. This run (+ per-flow novelty) catches **0 / 9**.

| Timestamp     | Destination IP   |   probability | flagged   |
|:--------------|:-----------------|--------------:|:----------|
| 6/7/2017 2:33 | 192.168.10.5     |   4.76322e-07 | False     |
| 6/7/2017 3:07 | 192.168.10.9     |   1.78859e-06 | False     |
| 6/7/2017 3:08 | 192.168.10.12    |   4.86595e-06 | False     |
| 6/7/2017 3:09 | 192.168.10.14    |   1.97666e-06 | False     |
| 6/7/2017 3:10 | 192.168.10.15    |   0.000224434 | False     |
| 6/7/2017 3:10 | 192.168.10.16    |   0.000121371 | False     |
| 6/7/2017 3:12 | 192.168.10.19    |   0.000136324 | False     |
| 6/7/2017 3:13 | 192.168.10.25    |   5.763e-05   | False     |
| 6/7/2017 3:14 | 192.168.10.51    |   0.000969044 | False     |
