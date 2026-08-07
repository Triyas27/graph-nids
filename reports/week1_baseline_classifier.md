# Week 1 — baseline flow classifier (XGBoost)
### Random split — binary (BENIGN vs ATTACK)
Train rows: 2,264,594, test rows: 566,149
Binary F1 (ATTACK class): 0.9981

```
              precision    recall  f1-score   support

      BENIGN      1.000     0.999     1.000    454620
      ATTACK      0.997     0.999     0.998    111529

    accuracy                          0.999    566149
   macro avg      0.999     0.999     0.999    566149
weighted avg      0.999     0.999     0.999    566149

```

### Random split — multi-class
Train rows: 2,264,594, test rows: 566,149
Classes seen in training: ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS']

Per-class report (rows with a class unseen in training excluded — see note above; those are 0% recall by construction):

```
                            precision    recall  f1-score   support

                    BENIGN      1.000     0.999     1.000    454620
                       Bot      0.855     0.781     0.816       393
                      DDoS      1.000     1.000     1.000     25606
             DoS GoldenEye      0.998     0.998     0.998      2059
                  DoS Hulk      0.999     1.000     0.999     46215
          DoS Slowhttptest      0.991     0.992     0.991      1100
             DoS slowloris      0.993     0.996     0.994      1159
               FTP-Patator      1.000     0.999     1.000      1588
                Heartbleed      1.000     1.000     1.000         2
              Infiltration      1.000     0.857     0.923         7
                  PortScan      0.994     1.000     0.997     31786
               SSH-Patator      0.999     1.000     1.000      1179
  Web Attack - Brute Force      0.743     0.854     0.794       301
Web Attack - Sql Injection      0.667     0.500     0.571         4
          Web Attack - XSS      0.506     0.346     0.411       130

                  accuracy                          0.999    566149
                 macro avg      0.916     0.888     0.900    566149
              weighted avg      0.999     0.999     0.999    566149

```

### Day-based split (train Mon-Wed, test Thu-Fri) — binary
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

### Day-based split (train Mon-Wed, test Thu-Fri) — multi-class
Train rows: 1,668,530, test rows: 1,162,213
Classes seen in training: ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'SSH-Patator']

**Classes present in test but never seen in training (structurally undetectable, 291,139 rows): ['Bot', 'DDoS', 'Infiltration', 'PortScan', 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS']**

Note on the table below: the 0/0/0 rows for DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed, SSH-Patator are **not** model failures — those attack families only occur on Monday-Wednesday, so they have 0 support in the Thursday-Friday test set by construction. The multi-class day-split table is therefore uninformative beyond "BENIGN is still recognized"; the real result of this split is the unseen-class list above and the binary numbers, which show every attack family CIC-IDS2017 puts on Thu-Fri going almost entirely undetected.

Per-class report (rows with a class unseen in training excluded — see note above; those are 0% recall by construction):

```
                  precision    recall  f1-score   support

          BENIGN      1.000     0.999     1.000    871074
   DoS GoldenEye      0.000     0.000     0.000         0
        DoS Hulk      0.000     0.000     0.000         0
DoS Slowhttptest      0.000     0.000     0.000         0
   DoS slowloris      0.000     0.000     0.000         0
     FTP-Patator      0.000     0.000     0.000         0
      Heartbleed      0.000     0.000     0.000         0
     SSH-Patator      0.000     0.000     0.000         0

        accuracy                          0.999    871074
       macro avg      0.125     0.125     0.125    871074
    weighted avg      1.000     0.999     1.000    871074

```

