# Week 1, Day 4-5 — controlled zero-day test

For each attack class, every row of that class is removed from training (all other attack classes + BENIGN remain). A fresh binary BENIGN-vs-ATTACK model is trained, then evaluated on the fully held-out class. `zero_day_catch_rate` = fraction of that class's flows flagged as ATTACK despite the model never having seen that attack type. `benign_fpr` is the false-positive rate on a fixed 10% BENIGN holdout never used in any fold's training, so it's comparable across rows.

| class                      |   support |   zero_day_catch_rate |   benign_fpr |
|:---------------------------|----------:|----------------------:|-------------:|
| Bot                        |      1966 |                 0.000 |        0.001 |
| Heartbleed                 |        11 |                 0.000 |        0.001 |
| Infiltration               |        36 |                 0.000 |        0.001 |
| PortScan                   |    158930 |                 0.003 |        0.000 |
| SSH-Patator                |      5897 |                 0.004 |        0.001 |
| FTP-Patator                |      7938 |                 0.370 |        0.001 |
| DoS Slowhttptest           |      5499 |                 0.374 |        0.001 |
| Web Attack - Sql Injection |        21 |                 0.476 |        0.001 |
| DoS Hulk                   |    231073 |                 0.631 |        0.000 |
| DDoS                       |    128027 |                 0.631 |        0.001 |
| DoS GoldenEye              |     10293 |                 0.670 |        0.001 |
| Web Attack - Brute Force   |      1507 |                 0.836 |        0.001 |
| Web Attack - XSS           |       652 |                 0.965 |        0.001 |
| DoS slowloris              |      5796 |                 0.996 |        0.001 |
