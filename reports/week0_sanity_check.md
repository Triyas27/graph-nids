# Week 0 sanity check — CIC-IDS2017 GeneratedLabelledFlows
## Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Rows (raw): 225,745
- Fully-blank rows: 0 (0.00%)
- Rows (real, after dropping blanks): 225,745
- Duplicate rows (of real rows): 2 (0.00%)
- Inf values (numeric cols, real rows): 64
- NaN values (numeric cols, real rows): 4
- Label distribution (real rows):
  - 'DDoS': 128,027
  - 'BENIGN': 97,718

## Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Rows (raw): 286,467
- Fully-blank rows: 0 (0.00%)
- Rows (real, after dropping blanks): 286,467
- Duplicate rows (of real rows): 1 (0.00%)
- Inf values (numeric cols, real rows): 727
- NaN values (numeric cols, real rows): 15
- Label distribution (real rows):
  - 'PortScan': 158,930
  - 'BENIGN': 127,537

## Friday-WorkingHours-Morning.pcap_ISCX.csv
- Rows (raw): 191,033
- Fully-blank rows: 0 (0.00%)
- Rows (real, after dropping blanks): 191,033
- Duplicate rows (of real rows): 2 (0.00%)
- Inf values (numeric cols, real rows): 216
- NaN values (numeric cols, real rows): 28
- Label distribution (real rows):
  - 'BENIGN': 189,067
  - 'Bot': 1,966

## Monday-WorkingHours.pcap_ISCX.csv
- Rows (raw): 529,918
- Fully-blank rows: 0 (0.00%)
- Rows (real, after dropping blanks): 529,918
- Duplicate rows (of real rows): 34 (0.01%)
- Inf values (numeric cols, real rows): 810
- NaN values (numeric cols, real rows): 64
- Label distribution (real rows):
  - 'BENIGN': 529,918

## Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Rows (raw): 288,602
- Fully-blank rows: 0 (0.00%)
- Rows (real, after dropping blanks): 288,602
- Duplicate rows (of real rows): 142 (0.05%)
- Inf values (numeric cols, real rows): 396
- NaN values (numeric cols, real rows): 18
- Label distribution (real rows):
  - 'BENIGN': 288,566
  - 'Infiltration': 36

## Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Rows (raw): 458,968
- Fully-blank rows: 288,602 (62.88%)
- Rows (real, after dropping blanks): 170,366
- Duplicate rows (of real rows): 1 (0.00%)
- Inf values (numeric cols, real rows): 250
- NaN values (numeric cols, real rows): 20
- Label distribution (real rows):
  - 'BENIGN': 168,186
  - 'Web Attack \x96 Brute Force': 1,507
  - 'Web Attack \x96 XSS': 652
  - 'Web Attack \x96 Sql Injection': 21

## Tuesday-WorkingHours.pcap_ISCX.csv
- Rows (raw): 445,909
- Fully-blank rows: 0 (0.00%)
- Rows (real, after dropping blanks): 445,909
- Duplicate rows (of real rows): 4 (0.00%)
- Inf values (numeric cols, real rows): 327
- NaN values (numeric cols, real rows): 201
- Label distribution (real rows):
  - 'BENIGN': 432,074
  - 'FTP-Patator': 7,938
  - 'SSH-Patator': 5,897

## Wednesday-workingHours.pcap_ISCX.csv
- Rows (raw): 692,703
- Fully-blank rows: 0 (0.00%)
- Rows (real, after dropping blanks): 692,703
- Duplicate rows (of real rows): 17 (0.00%)
- Inf values (numeric cols, real rows): 1,586
- NaN values (numeric cols, real rows): 1,008
- Label distribution (real rows):
  - 'BENIGN': 440,031
  - 'DoS Hulk': 231,073
  - 'DoS GoldenEye': 10,293
  - 'DoS slowloris': 5,796
  - 'DoS Slowhttptest': 5,499
  - 'Heartbleed': 11

## Overall
- Total raw rows across all files: 3,119,345
- Total fully-blank rows: 288,602
- Total real rows (raw minus blank): 2,830,743
- Total duplicate rows (of real rows): 203 (0.01%)
- Combined label distribution (real rows):
  - 'BENIGN': 2,273,097
  - 'DoS Hulk': 231,073
  - 'PortScan': 158,930
  - 'DDoS': 128,027
  - 'DoS GoldenEye': 10,293
  - 'FTP-Patator': 7,938
  - 'SSH-Patator': 5,897
  - 'DoS slowloris': 5,796
  - 'DoS Slowhttptest': 5,499
  - 'Bot': 1,966
  - 'Web Attack \x96 Brute Force': 1,507
  - 'Web Attack \x96 XSS': 652
  - 'Infiltration': 36
  - 'Web Attack \x96 Sql Injection': 21
  - 'Heartbleed': 11

## Known issues found
- **Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv**: 288,602 fully-blank rows (62.9% of the file), rows 170366-458967 (contiguous: True)
- **Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv**: mis-encoded label repr 'Web Attack \x96 Brute Force' (1,507 rows) — Windows-1252 en-dash read via latin1/utf-8
- **Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv**: mis-encoded label repr 'Web Attack \x96 XSS' (652 rows) — Windows-1252 en-dash read via latin1/utf-8
- **Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv**: mis-encoded label repr 'Web Attack \x96 Sql Injection' (21 rows) — Windows-1252 en-dash read via latin1/utf-8
