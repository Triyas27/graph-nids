# Week 2, Day 6-7 — SHAP analysis of the day-split flow+graph model

SHAP TreeExplainer on a stratified sample (3,057 rows, up to 500/label) of the Thursday-Friday test set, explaining the exact model persisted from Week 2 Day 5 (day-based split, F1 0.625).

## Global feature ranking (mean |SHAP value|)

| rank | feature | mean \|SHAP\| | type |
|---|---|---|---|
| 1 | src_flow_volume | 6.0673 | graph |
| 2 | Destination Port | 3.7128 | flow |
| 3 | src_fan_out | 2.4840 | graph |
| 4 | src_fan_in | 1.4563 | graph |
| 5 | dst_novelty_zscore | 1.3921 | graph |
| 6 | dst_novelty_score | 1.3374 | graph |
| 7 | src_fan_in_zscore | 0.3954 | graph |
| 8 | Min Packet Length | 0.2805 | flow |
| 9 | dst_flow_volume_zscore | 0.1791 | graph |
| 10 | Flow IAT Min | 0.1779 | flow |
| 11 | dst_byte_volume_zscore | 0.1615 | graph |
| 12 | dst_pagerank | 0.1431 | graph |
| 13 | Fwd Packet Length Std | 0.1328 | flow |
| 14 | dst_fan_out | 0.1302 | graph |
| 15 | src_pagerank | 0.1263 | graph |
| 16 | src_byte_volume | 0.0992 | graph |
| 17 | src_degree_ratio_change | 0.0986 | graph |
| 18 | dst_byte_volume | 0.0841 | graph |
| 19 | Fwd IAT Mean | 0.0704 | flow |
| 20 | Fwd Header Length | 0.0552 | flow |

14/20 of the top SHAP-ranked features are graph features.

## Per-attack-type: recall and graph-feature reliance

For each true label, recall in this sample (fraction flagged ATTACK), and what share of the total |SHAP| contribution (summed across all features, this label's rows only) comes from graph features vs flow features.

| label | n | recall | mean total \|SHAP\| | graph share | flow share |
|---|---|---|---|---|---|
| BENIGN | 500 | 0.992 | 15.249 | 81.4% | 18.6% |
| Bot | 500 | 0.000 | 13.893 | 68.7% | 31.3% |
| DDoS | 500 | 1.000 | 17.491 | 91.3% | 8.7% |
| Infiltration | 36 | 0.000 | 18.983 | 65.2% | 34.8% |
| PortScan | 500 | 0.048 | 23.249 | 46.9% | 53.1% |
| Web Attack - Brute Force | 500 | 1.000 | 25.898 | 88.0% | 12.0% |
| Web Attack - Sql Injection | 21 | 0.238 | 16.491 | 84.5% | 15.5% |
| Web Attack - XSS | 500 | 0.932 | 17.389 | 81.9% | 18.1% |

