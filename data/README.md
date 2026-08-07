# Dataset

This project uses **CIC-IDS2017** from the Canadian Institute for Cybersecurity (UNB).
The raw and processed CSV/parquet files are not committed to this repo (too large,
and redistributing the raw dataset isn't necessary) — download them yourself and
regenerate the processed file locally.

- Official dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
- File used: `GeneratedLabelledFlows.zip` from the CSVs folder at
  http://cicresearch.ca/CICDataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip
  (not `MachineLearningCSV.zip` — that version strips the `Source IP` /
  `Destination IP` / `Timestamp` columns needed for the graph/lateral-movement
  work in Week 2-3; `GeneratedLabelledFlows.zip` keeps them.)
- Reference paper: Sharafaldin, Lashkari, Ghorbani, "Toward Generating a New
  Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.

## Setup

1. Download `GeneratedLabelledFlows.zip` (and its `.md5`) from the link above.
2. Verify the checksum:
   ```
   md5sum GeneratedLabelledFlows.zip
   # expect: 5ca3f8f69e3514950681615824149973
   ```
3. Place the zip in `data/raw/` and extract it there, so you end up with:
   ```
   data/raw/extracted/TrafficLabelling /*.csv   (note: trailing space in the folder name, from the original zip)
   ```
4. Regenerate the cleaned dataset:
   ```
   python scripts/01_prepare_data.py
   ```
   This writes `data/processed/flows_clean.parquet` — combines all 8 daily CSVs,
   drops the ~289k fully-blank padding rows found in the Thursday-Morning-WebAttacks
   file, fixes the mis-encoded Web Attack labels, and tags every row with its
   source day for the day-based train/test split used in Week 1+.

See `reports/week0_sanity_check.md` for the full data-quality findings.
