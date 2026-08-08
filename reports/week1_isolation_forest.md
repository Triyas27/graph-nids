# Week 1, Day 6-7 — Isolation Forest anomaly baseline

Isolation Forest is trained only on BENIGN traffic (80% of BENIGN rows; 20% held out for the false-positive rate below) — it never sees an attack label. Compared against the supervised XGBoost binary classifier in two regimes: `known` (the attack class was present during training, standard random split) and `zero_day` (the attack class was entirely absent from training — see Week 1 Day 4-5, reports/week1_zeroday_test.md).

- Supervised model (known-attack regime) false-positive rate on BENIGN test: 0.0006
- Isolation Forest false-positive rate on BENIGN holdout: 0.0778

| class                      |   support |   known_supervised_catch_rate |   zero_day_supervised_catch_rate |   isolation_forest_catch_rate |
|:---------------------------|----------:|------------------------------:|---------------------------------:|------------------------------:|
| Bot                        |      1966 |                         0.720 |                            0.000 |                         0.022 |
| Heartbleed                 |        11 |                         1.000 |                            0.000 |                         1.000 |
| Infiltration               |        36 |                         0.857 |                            0.000 |                         0.861 |
| PortScan                   |    158930 |                         1.000 |                            0.003 |                         0.002 |
| SSH-Patator                |      5897 |                         0.999 |                            0.004 |                         0.000 |
| FTP-Patator                |      7938 |                         0.999 |                            0.370 |                         0.000 |
| DoS Slowhttptest           |      5499 |                         0.997 |                            0.374 |                         0.802 |
| Web Attack - Sql Injection |        21 |                         1.000 |                            0.476 |                         0.000 |
| DoS Hulk                   |    231073 |                         1.000 |                            0.631 |                         0.672 |
| DDoS                       |    128027 |                         1.000 |                            0.631 |                         0.595 |
| DoS GoldenEye              |     10293 |                         0.999 |                            0.670 |                         0.629 |
| Web Attack - Brute Force   |      1507 |                         1.000 |                            0.836 |                         0.048 |
| Web Attack - XSS           |       652 |                         0.992 |                            0.965 |                         0.025 |
| DoS slowloris              |      5796 |                         0.998 |                            0.996 |                         0.541 |
